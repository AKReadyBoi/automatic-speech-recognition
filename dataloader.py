import xml.etree.ElementTree as ET
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from dataset import BPEDataset
from dictionary import Dictionary
from bpe import bpe_decomp, bpe_apply

# build padding with collate function
def collate_fn(batch, pad_value):
    features, labels = zip(*batch) 

    padded_features = pad_sequence(features, batch_first=True)
    feature_lengths  = torch.tensor([f.shape[0] for f in features])
    
    label_lengths = torch.tensor([len(l)     for l in labels])
    padded_labels   = pad_sequence(labels,   batch_first=True, padding_value=pad_value)

    return padded_features, feature_lengths, padded_labels, label_lengths

# returns torch dataloader
def make_dataloader(dataset, dictionary, batch_size=10, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, pad_value=dictionary.blank_idx),
    )

# outputs random batch as indices and strings
def dump_random_batch(loader, dictionary, idx_path, str_path):
    padded_features, feature_lengths, padded_labels, label_lengths = next(iter(loader))

    print("=== batch shapes ===")
    print(f"features : {padded_features.shape}   (batch, T_max, n_mels)")
    print(f"feat lens: {feature_lengths.tolist()}")
    print(f"labels   : {padded_labels.shape}   (batch, L_max)")
    print(f"lbl lens : {label_lengths.tolist()}")
    print(f"feat/4 >= label? {(feature_lengths // 4 >= label_lengths).all().item()}")

    with open(idx_path, "w", encoding="utf-8") as f:
        for seq, length in zip(padded_labels, label_lengths):
            f.write(" ".join(map(str, seq[:length].tolist())) + "\n")

    with open(str_path, "w", encoding="utf-8") as f:
        for seq, length in zip(padded_labels, label_lengths):
            tokens = dictionary.decode(seq[:length].tolist())
            f.write(" ".join(tokens) + "\n")


if __name__ == "__main__":
    corpus = "train_100_corpus.xml"
    rules_path = "ResultsA2/Rules.txt"
    dict_path  = "ResultsA2/dictionary.json"

    import os
    from bpe import save_rules, load_rules

    # load or compute BPE rules
    if os.path.exists(rules_path):
        print("loading BPE rules...")
        rules = load_rules(rules_path)
    else:
        print("computing BPE rules (slow, one-time)...")
        rules = bpe_decomp(corpus, 200)
        save_rules(rules, rules_path)

    # load or build dictionary
    dictionary = Dictionary()
    if os.path.exists(dict_path):
        print("loading dictionary...")
        dictionary.load(dict_path)
    else:
        print("building dictionary...")
        tree = ET.parse(corpus)
        sentences = [s.text for s in tree.getroot().findall(".//orth")]
        bpe_sentences = []
        for s in sentences:
            nested = bpe_apply(s, rules)
            flat = [t for word in nested for t in word]
            bpe_sentences.append(flat)
        dictionary = Dictionary.build_from_bpe_text(bpe_sentences)
        dictionary.save(dict_path)

    # feature extractor from precomputed spect.npy
    all_features = np.load("spect.npy", allow_pickle=True).item()
    def feature_extractor(audio_path):
        return torch.tensor(all_features[audio_path], dtype=torch.float32)

    # Dataset + DataLoader
    ds = BPEDataset(corpus, rules, dictionary, feature_extractor)
    loader = make_dataloader(ds, dictionary, batch_size=10)

    # dump batches
    dump_random_batch(
        loader,
        dictionary,
        "ResultsA2/batch_indices.txt",
        "ResultsA2/batch_strings.txt",
    )
