import numpy as np
import multiprocessing
import flwr as fl
from sklearn.model_selection import train_test_split
from dataset import download_and_extract_data, wav_data_loader
from model import create_keras_model
from client import FlowerClient
import time
import tensorflow as tf
import pandas as pd
import collections
import librosa
import random

# ----- DATA AUGMENTATION FUNCTIONS -------

SR = 16000

## Noise addition

def add_noise(wav_data, noise_factor):

    # Generate noise signal with the same shape as input waveform
    noise = np.random.normal(0, 1, len(wav_data))

    # Scale noise signal with the permissible noise factor value
    noise *= noise_factor

    # Add noise signal to input waveform
    augmented_wav_data = wav_data + noise

    # Normalize the augmented waveform data
    augmented_wav_data = librosa.util.normalize(augmented_wav_data)

    return augmented_wav_data

def time_shift(audio, p):
    """
    Shift audio to the left or right by a random amount.
    """
    # Calculate the length of the audio array
    length = audio.shape[0]

    # Calculate the maximum number of samples to shift
    max_shift = int(length * p)

    # Generate a random shift value
    shift = random.randint(-max_shift, max_shift)

    # Create an empty array with the same shape as the audio array
    shifted_audio = np.zeros_like(audio)

    # Shift the audio by the specified number of samples
    if shift > 0:
      # Shift to the right
        shifted_audio[shift:] = audio[:length-shift]
    else:
        # Shift to the left
        shifted_audio[:length+shift] = audio[-shift:]
    
    if np.sum(shifted_audio) == 0:
        #revert the process if all information was erased
        shifted_audio = audio     

    return shifted_audio

def time_stretching(audio,factor):
    
    wav_time_stch = librosa.effects.time_stretch(audio,rate=factor)
    
    return wav_time_stch[:SR*5]

def augment_wavs(wav_dataset, y):

    y = list(y)
    
    wav_dataset_augmented = []

    for wav in wav_dataset:
        # Create a copy of the original wav to prevent unwanted side effects
        temp_wav = wav.copy()
        temp_wav = add_noise(temp_wav, 0.025) # We want to use values between 0.005 and 0.04
        temp_wav = time_shift(temp_wav, 0.3)  # We want to use a max shift of 30%
        temp_wav = time_stretching(temp_wav, 0.85)

        wav_dataset_augmented.append(temp_wav)

    # Add original wavs to augmented list
    wav_dataset_augmented.extend(wav_dataset)
    
    y = y + y #each spec is being appended at the bottom of the list

    return np.array(wav_dataset_augmented), np.array(y)

# ----------------

def balance_dataset(data_subset, label_subset, public_data, public_labels, beta):
    # Convert the labels to Series for easier manipulation
    s_private_labels = pd.Series(label_subset)
    s_public_labels = pd.Series(public_labels)

    # Find the maximum class sample size in the private dataset
    max_size = s_private_labels.value_counts().max()

    # Create empty lists for the balanced dataset
    balanced_data = []
    balanced_labels = []

    # Loop over each label in the public dataset
    for label in s_public_labels.unique():
        # Get indices of the current label from the private and public datasets
        private_indices = s_private_labels[s_private_labels == label].index.tolist()
        public_indices = s_public_labels[s_public_labels == label].index.tolist()

        # Calculate the number of public samples needed to satisfy the equation
        num_public_samples = max_size + int(beta * max_size) - len(private_indices)

        # If there are not enough public samples, take all of them, otherwise take the required number
        selected_public_indices = random.sample(public_indices, min(num_public_samples, len(public_indices)))

        # Collect data samples
        balanced_data.extend([data_subset[i] for i in private_indices] + [public_data[i] for i in selected_public_indices])
        balanced_labels.extend([label] * (len(private_indices) + len(selected_public_indices)))

    return np.array(balanced_data), np.array(balanced_labels)

def start_client(server_address, data_subset, label_subset):

    # Split the subset into training and testing datasets
    x_train, x_test, y_train, y_test = train_test_split(data_subset, label_subset, test_size=0.2, stratify=label_subset,)

    x_train, y_train = augment_wavs(x_train, y_train)
    
    # Create a Keras model for each client
    model = create_keras_model(NUM_CLASSES)

    # Create Flower client with the subset of data
    client = FlowerClient(model, x_train, y_train, x_test, y_test)

    # Start Flower client
    fl.client.start_numpy_client(server_address=server_address, client=client)

if __name__ == "__main__":
    # Load and preprocess data
    download_and_extract_data()

    dataset, NUM_CLASSES = wav_data_loader()

    data = [wav[0] for wav in dataset]
    labels = [wav[1] for wav in dataset]

    # create a pandas DataFrame from the data and labels
    df = pd.DataFrame({
        'data': data,
        'labels': labels
    })

    public_data = []
    public_labels = []
    drop_indices = []

    # Iterate over unique classes
    for class_label in df['labels'].unique():
        # Select 10 samples per class and append to public_data and public_labels
        sample = df[df['labels'] == class_label].sample(10)
        public_data.extend(sample['data'])
        public_labels.extend(sample['labels'])
        drop_indices.extend(sample.index)

    # Convert to numpy arrays
    public_data = np.array(public_data)
    public_labels = np.array(public_labels)

    # Drop these samples from your DataFrame
    df = df.drop(drop_indices)

    # Redefine data and labels variables
    data = df['data'].tolist()
    labels = df['labels'].tolist()

    # Create lists to hold the subsets of data and labels
    data_splits = []
    label_splits = []

    # Define the number of samples for each client, unevenly
    client_sizes = [60, 150, 100, 130, 90, 120]  # Adjust these numbers as per your data

    # Ensure that sum of client_sizes is less than or equal to number of samples
    assert sum(client_sizes) <= len(data), "The total number of samples is less than the sum of client_sizes"

    # Shuffle data and labels together
    combined = list(zip(data, labels))
    random.shuffle(combined)
    data, labels = zip(*combined)

    # Start index
    start = 0

    # Distribute data and labels unevenly
    for size in client_sizes:
        data_splits.append(data[start:start + size])
        label_splits.append(labels[start:start + size])
        start += size

    # Convert data and labels to numpy arrays
    data_splits = [np.array(split) for split in data_splits]
    label_splits = [np.array(split) for split in label_splits]    

    # Start a client for each data subset
    for data_subset, label_subset in zip(data_splits, label_splits):
        
        # Balance the dataset
        data_subset, label_subset = balance_dataset(data_subset, label_subset, public_data, public_labels, beta=0.5)
    
        # print out the distribution of the current client
        distribution = collections.Counter(label_subset)
        print(f"Distribution for client: {distribution}")

        process = multiprocessing.Process(target=start_client, args=("localhost:8080", data_subset, label_subset))
        process.start()
        time.sleep(2)