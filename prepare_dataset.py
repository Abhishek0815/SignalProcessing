import librosa
import os
import json

DATASET_PATH = "dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 2560 # 1 sec. of audio


def preprocess_dataset(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=256):
    
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        if dirpath is not dataset_path:

            label = dirpath.split("\\")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))
            
            
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                signal, sample_rate = librosa.load(file_path,sr=16000)

                if len(signal) >= SAMPLES_TO_CONSIDER:


                    signal = signal[:SAMPLES_TO_CONSIDER-1]
                    MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                 hop_length=hop_length)
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["labels"].append(i-1)
                    data["files"].append(file_path)
                    print("{}: {}".format(file_path, i-1))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH)
