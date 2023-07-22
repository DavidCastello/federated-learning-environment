import os
import librosa
import subprocess

REPO_URL = 'https://github.com/DavidCastello/NBAC.git'  # replace this with your GitHub repository URL
CLONE_DIR = './audio_repo'  # this is the local directory where the repository will be cloned
AUDIO_DIR = os.path.join(CLONE_DIR, 'audio')  # assuming the audio files are stored in an 'audio' folder in the repo

def download_and_extract_data():
    if not os.path.exists(CLONE_DIR):
        try:
            subprocess.run(['git', 'clone', REPO_URL, CLONE_DIR], check=True)
            print(f'Successfully cloned repository {REPO_URL} into {CLONE_DIR}')
        except subprocess.CalledProcessError:
            print(f'Failed to clone repository {REPO_URL} into {CLONE_DIR}')
    else:
        print(f'Directory {CLONE_DIR} already exists. Skipping cloning process.')

# Sample rate to load the wav files

SR = 16000

def map_subfolders_to_int(path):
    reversed_labels = {}
    counter = 0

    # Walk through the directory
    for root, dirs, files in os.walk(path):
        # For each subdirectory
        for dir in dirs:
            # Add the subdirectory to the dictionary with the current count as the key
            reversed_labels[counter] = dir
            counter += 1

    return reversed_labels

def reverse_dict(original_dict):
    reversed_dict = {value: key for key, value in original_dict.items()}
    return reversed_dict

def wav_data_loader(directory=AUDIO_DIR, sr=SR, normalization=False):

    # Specify your directory path here
    reversed_labels = map_subfolders_to_int(AUDIO_DIR)
    print('NBAC dataset is getting prepared...')

    NUM_CLASSES = len(reversed_labels)

    # reversed dictionary
    labels_dict = reverse_dict(reversed_labels)

    all_fragments = []
    for root, dirs, _ in os.walk(directory):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]
            for file in files:
                file_path = os.path.join(dir_path, file)
                # Load the audio file
                sample, sample_rate = librosa.load(file_path, sr=sr)
                if normalization:
                    # Normalize the waveform
                    sample = librosa.util.normalize(sample)
                # Append the sample and its label (subfolder name) as a tuple
                all_fragments.append((sample, labels_dict[dir]))
    
    return all_fragments, NUM_CLASSES