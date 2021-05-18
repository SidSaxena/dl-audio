import librosa
import os 
import json 
import math 

DATASET_PATH = 'D:/development/python/valerio-velardo/dl-audio/dataset'
JSON_PATH = 'data_10.json'
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    
    # dictionary to store mapping, labels, and MFCCs
    data = {
        'mapping': [],
        'labels': [],
        'mfcc': []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through the genre folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre lavel (i.e. sub-foldern ame) in the mapping
            semantic_label = dirpath.split('//')[-1]
            data['mapping'].append(semantic_label)
            print(f'\nProcessing: {semantic_label}')

            # process all audio files in genre folder
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # store only mfcc features with expected number of vectors

                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data['mfcc'].append(mfcc.tolist())
                        data['labels'].append(i-1)
                        print(f'{file_path}, segment:{d+1}')
    # save mfcc to json
    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4)

if __name__ == '__main__':
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)