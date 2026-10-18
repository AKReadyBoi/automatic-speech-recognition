import soundfile
import librosa.feature
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import os

def logmel_from_audio(file_path):
    # Single source of truth for the feature extraction, shared with transcribe.py
    # so the exam-day audio and the training features use identical parameters.
    y, sr = soundfile.read(file=file_path)
    spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=80,
        hop_length=int(0.01 * sr),
        n_fft=int(0.025 * sr),
        fmin=60, fmax=7600
    )
    log_spectrogram = np.log(np.maximum(spectrogram, 1e-10))
    return log_spectrogram.T   # (time, n_mels)


def mel_feature(corpora=("train_100_corpus.xml", "dev_corpus.xml"), output="spect.npy"):
    all_features = {}

    if os.path.exists(output):
        try:
            all_features = np.load(output, allow_pickle=True).item()
            print(f"Loaded {len(all_features)} existing features from {output}")
        except Exception:
            pass

    for corpus_name in corpora:
        if not os.path.exists(corpus_name):
            continue
        print(f"Processing corpus: {corpus_name}...")
        tree = ET.parse(corpus_name)
        root = tree.getroot()
        count = 0
        for i in root.iter('recording'):
            file_path = i.get('audio')
            if not os.path.exists(file_path):
                continue
            if file_path in all_features:
                continue
            
            try:
                all_features[file_path] = logmel_from_audio(file_path)
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} audio files in {corpus_name}...")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
    np.save(output, all_features)
    print(f"Saved {len(all_features)} features to {output}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", nargs="+", default=["train_100_corpus.xml", "dev_corpus.xml"])
    parser.add_argument("--output", default="spect.npy")
    args = parser.parse_args()
    mel_feature(args.corpus, args.output)

