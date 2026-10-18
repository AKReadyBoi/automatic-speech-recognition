import torch
import numpy as np
import xml.etree.ElementTree as ET
import argparse
import os

from model import Conformer
from trainer import Trainer
from dataset import BPEDataset, SpecAugmentConfig
from dictionary import Dictionary
from dataloader import make_dataloader
from bpe import bpe_apply, bpe_decomp, load_rules, save_rules

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",               type=int,   default=5)
    parser.add_argument("--batch-size",           type=int,   default=10)
    parser.add_argument("--num-heads",            type=int,   default=4)   # must divide d_model
    parser.add_argument("--max-lr",               type=float, default=1e-3)
    parser.add_argument("--checkpoint-interval",  type=int,   default=0)   # batches between mid-epoch saves
    parser.add_argument("--early-stopping",       action="store_true")
    parser.add_argument("--load-checkpoint",      type=str,   default=None)
    parser.add_argument("--no-spec-augment",      action="store_true")
    parser.add_argument("--freq-masks",           type=int,   default=2)
    parser.add_argument("--freq-mask-width",      type=int,   default=15)
    parser.add_argument("--time-masks",           type=int,   default=2)
    parser.add_argument("--time-mask-width",      type=int,   default=25)
    parser.add_argument("--max-time-mask-fraction", type=float, default=0.15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load BPE rules
    rules_path = "ResultsA2/Rules.txt"
    rules = load_rules(rules_path)

    # Load dict json
    dict_path = "ResultsA2/dictionary.json"
    dictionary = Dictionary()
    dictionary.load(dict_path)

    # extract features from spect.npy(need to run mel_feature for it to work)
    all_features = np.load("spect.npy", allow_pickle=True)[()]
    def feature_extractor(audio_path):
        return torch.tensor(all_features[audio_path], dtype=torch.float32)

    # --- datasets ---
    spec_augment_config = SpecAugmentConfig(
        freq_masks=args.freq_masks,
        freq_mask_width=args.freq_mask_width,
        time_masks=args.time_masks,
        time_mask_width=args.time_mask_width,
        max_time_mask_fraction=args.max_time_mask_fraction,
    )
    train_dataset = BPEDataset(
        "train_100_corpus.xml",
        rules,
        dictionary,
        feature_extractor,
        spec_augment=not args.no_spec_augment,
        spec_augment_config=spec_augment_config,
    )
    
    if os.path.exists("dev_corpus.xml"):
        dev_dataset = BPEDataset("dev_corpus.xml", rules, dictionary, feature_extractor, spec_augment=False)
        print("Loaded official dev_corpus.xml for validation.")
    else:
        print("WARNING: dev_corpus.xml not found! Falling back to 10% random split of train_100_corpus.xml for validation.")
        dev_size     = int(0.1 * len(train_dataset))
        train_size   = len(train_dataset) - dev_size
        train_dataset, dev_dataset = torch.utils.data.random_split(train_dataset, [train_size, dev_size])

    # --- dataloaders ---
    train_loader = make_dataloader(train_dataset, dictionary, batch_size=args.batch_size, shuffle=True)
    dev_loader   = make_dataloader(dev_dataset,   dictionary, batch_size=args.batch_size, shuffle=False)

    # --- model ---
    model = Conformer(
        d_model=512,
        num_blocks=6,
        vocab_size=len(dictionary),
        dropout=0.1,
        conv_channels=256,
        n_mels=80,
        num_heads=args.num_heads
    )

    # --- trainer ---
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        device=device,
        max_lr=args.max_lr,
        epochs=args.epochs,
        checkpoint_interval=args.checkpoint_interval,
        early_stopping=args.early_stopping,
        dictionary=dictionary,
    )

    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)

    if args.epochs == 0:
        trainer.evaluate()
    else:
        trainer.train()

if __name__ == "__main__":
    main()
