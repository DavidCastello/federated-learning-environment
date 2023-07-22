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

    # group the DataFrame by labels
    grouped = df.groupby('labels')

    # create lists to hold the subsets of data and labels
    data_splits = [[] for _ in range(6)]
    label_splits = [[] for _ in range(6)]

    # loop over each group
    for _, group in grouped:
        # split the group into 6 subsets
        group_data_splits = np.array_split(group['data'], 6)
        group_label_splits = np.array_split(group['labels'], 6)
        
        # add each subset to the corresponding final subset
        for i in range(6):
            data_splits[i].extend(group_data_splits[i])
            label_splits[i].extend(group_label_splits[i])
    
    # convert data and labels to numpy arrays
    data_splits = [np.array(split) for split in data_splits]
    label_splits = [np.array(split) for split in label_splits]
    
    # print out the distribution of the first client
    distribution_client1 = collections.Counter(label_splits[0])
    print(distribution_client1)

    # Start a client for each data subset
    for data_subset, label_subset in zip(data_splits, label_splits):
        process = multiprocessing.Process(target=start_client, args=("localhost:8080", data_subset, label_subset))
        process.start()
        time.sleep(2)