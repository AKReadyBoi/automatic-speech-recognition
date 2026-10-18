import argparse
import json
import os
import time
import torch
import numpy as np

# Use Agg backend for matplotlib to prevent GUI errors on headless HPC nodes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import Conformer
from trainer import Trainer
from dataset import BPEDataset, SpecAugmentConfig
from dictionary import Dictionary
from dataloader import make_dataloader
from bpe import load_rules

def run_trial(name, config, train_dataset, dev_dataset, dictionary, device, args):
    print(f"\n==================================================")
    print(f"Running Trial: {name}")
    print(f"Config: {config}")
    print(f"==================================================")

    # Configure SpecAugment dynamically for this trial
    train_dataset.spec_augment = config.get("spec_augment", False)
    train_dataset.spec_augment_config = SpecAugmentConfig(
        freq_masks=config.get("freq_masks", 2),
        freq_mask_width=config.get("freq_mask_width", 15),
        time_masks=config.get("time_masks", 2),
        time_mask_width=config.get("time_mask_width", 25),
        max_time_mask_fraction=config.get("max_time_mask_fraction", 0.15)
    )

    # Make dataloaders
    train_loader = make_dataloader(train_dataset, dictionary, batch_size=args.batch_size, shuffle=True)
    dev_loader = make_dataloader(dev_dataset, dictionary, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = Conformer(
        d_model=config["d_model"],
        num_blocks=config["num_blocks"],
        vocab_size=len(dictionary),
        dropout=config.get("dropout", args.dropout),
        conv_channels=config["conv_channels"],
        num_heads=config["num_heads"],
        use_attention=config["use_attention"]
    )

    # Initialize trainer
    # Note: we only pass parameters supported by the original Trainer class
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        device=device,
        max_lr=config.get("max_lr", args.max_lr),
        epochs=args.epochs,
        checkpoint_interval=0,
        early_stopping=False,
        dictionary=dictionary
    )

    # Run training
    start_time = time.time()
    best_loss = float('inf')
    best_wer = float('inf')

    if len(train_loader) > 0:
        for epoch in range(1, args.epochs + 1):
            epoch_loss = trainer.train_epoch(epoch)
            dev_loss = trainer.evaluate()
            
            if dev_loss < best_loss:
                best_loss = dev_loss
            
            if trainer.dictionary is not None:
                dev_wer = trainer.evaluate_wer()
                print(f"Epoch {epoch} | Dev WER: {dev_wer:.2%}")
                if dev_wer < best_wer:
                    best_wer = dev_wer

    elapsed = time.time() - start_time
    print(f"Completed Trial {name} | Time: {elapsed:.1f}s | Best Dev WER: {best_wer:.2%} | Best Dev Loss: {best_loss:.4f}")
    
    return {
        "best_wer": best_wer,
        "best_loss": best_loss,
        "time_taken": elapsed
    }

def main():
    parser = argparse.ArgumentParser(description="HPC Hyperparameter Analysis (Architectural Parameters)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs per trial")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--train-subset", type=int, default=None, help="Use a subset of training data (for speed testing)")
    parser.add_argument("--dev-subset", type=int, default=None, help="Use a subset of dev data")
    parser.add_argument("--max-lr", type=float, default=1e-3, help="Max learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    args = parser.parse_args()

    os.makedirs("ResultsA2", exist_ok=True)
    progress_file = "ResultsA2/tuning_progress.json"

    # Load existing results if any
    results = {}
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                results = json.load(f)
            print(f"Loaded {len(results)} completed trials from progress file.")
        except Exception as e:
            print(f"Warning: could not load progress file: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load BPE rules and dictionary
    rules = load_rules("ResultsA2/Rules.txt")
    dictionary = Dictionary()
    dictionary.load("ResultsA2/dictionary.json")

    # Load features map
    all_features = np.load("spect.npy", allow_pickle=True)[()]
    def feature_extractor(audio_path):
        return torch.tensor(all_features[audio_path], dtype=torch.float32)

    # Instantiate datasets
    print("Loading datasets...")
    train_dataset = BPEDataset("train_100_corpus.xml", rules, dictionary, feature_extractor)
    dev_dataset = BPEDataset("dev_corpus.xml", rules, dictionary, feature_extractor)

    # Dynamically filter out items missing in spect.npy
    train_dataset.data = [item for item in train_dataset.data if item[0] in all_features]
    dev_dataset.data = [item for item in dev_dataset.data if item[0] in all_features]

    # Fallback if training files are missing (e.g. running locally instead of HPC cluster)
    if len(train_dataset) == 0:
        print("WARNING: No training audio files found in spect.npy (probably running locally instead of HPC). Falling back to dev dataset for training data.")
        train_dataset.data = list(dev_dataset.data)

    # Limit subset if requested
    if args.train_subset is not None:
        train_dataset.data = train_dataset.data[:args.train_subset]
        print(f"Limited training dataset to {len(train_dataset)} samples.")
    else:
        print(f"Using all {len(train_dataset)} training samples.")

    if args.dev_subset is not None:
        dev_dataset.data = dev_dataset.data[:args.dev_subset]
        print(f"Limited validation dataset to {len(dev_dataset)} samples.")
    else:
        print(f"Using all {len(dev_dataset)} validation samples.")

    # Base configuration
    baseline_config = {
        "conv_channels": 31,
        "num_blocks": 3,
        "d_model": 100,
        "num_heads": 4,
        "use_attention": True,
        "spec_augment": True,
        "dropout": 0.15,
        "max_lr": 1e-3,
        "freq_masks": 2,
        "freq_mask_width": 15,
        "time_masks": 2,
        "time_mask_width": 25,
        "max_time_mask_fraction": 0.15
    }

    # Define trials by parameter
    parameter_sweeps = {
        "conv_channels": [16, 31, 64, 128, 256, 512],
        "num_blocks": [2, 3, 4, 6, 8, 10],
        "d_model_heads": [
            {"d_model": 64, "num_heads": 2},
            {"d_model": 100, "num_heads": 4},
            {"d_model": 128, "num_heads": 4},
            {"d_model": 256, "num_heads": 8},
            {"d_model": 512, "num_heads": 8},
            {"d_model": 768, "num_heads": 8}
        ],
        "use_attention": [True, False],
        "spec_augment": [True, False],
        "dropout": [0.0, 0.05, 0.1, 0.2],
        "max_lr": [3e-4, 1e-3, 3e-3],
        "freq_mask_width": [5, 15, 25],
        "time_mask_width": [10, 25, 45]
    }

    # Helper to generate config for a sweep
    def get_config(param_name, val):
        cfg = baseline_config.copy()
        if param_name == "d_model_heads":
            cfg["d_model"] = val["d_model"]
            cfg["num_heads"] = val["num_heads"]
        else:
            cfg[param_name] = val
        return cfg

    # Run sweeps
    for param_name, values in parameter_sweeps.items():
        for val in values:
            trial_key = f"{param_name}_{val}"
            if trial_key in results:
                print(f"Skipping trial {trial_key} (already completed).")
                continue

            config = get_config(param_name, val)
            trial_results = run_trial(trial_key, config, train_dataset, dev_dataset, dictionary, device, args)
            
            results[trial_key] = {
                "config": config,
                "param_value": val,
                "best_wer": trial_results["best_wer"],
                "best_loss": trial_results["best_loss"],
                "time_taken": trial_results["time_taken"]
            }

            # Save progress after each trial
            with open(progress_file, "w") as f:
                json.dump(results, f, indent=4)

    # --- Plotting & Reporting ---
    print("\nGenerating plots and markdown report...")
    
    # 1. conv_channels plot
    cc_vals = parameter_sweeps["conv_channels"]
    cc_wers = [results[f"conv_channels_{v}"]["best_wer"] for v in cc_vals if f"conv_channels_{v}" in results]
    if len(cc_wers) == len(cc_vals):
        plt.figure()
        plt.plot(cc_vals, [w * 100 for w in cc_wers], marker='o', color='#1f77b4', linewidth=2)
        plt.xlabel("Convolution Channels")
        plt.ylabel("Word Error Rate (WER) %")
        plt.title("Impact of Convolution Channels on WER")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig("ResultsA2/conv_channels_wer.png", bbox_inches='tight')
        plt.close()

    # 2. num_blocks plot
    nb_vals = parameter_sweeps["num_blocks"]
    nb_wers = [results[f"num_blocks_{v}"]["best_wer"] for v in nb_vals if f"num_blocks_{v}" in results]
    if len(nb_wers) == len(nb_vals):
        plt.figure()
        plt.plot(nb_vals, [w * 100 for w in nb_wers], marker='o', color='#2ca02c', linewidth=2)
        plt.xlabel("Number of Conformer Blocks")
        plt.ylabel("Word Error Rate (WER) %")
        plt.title("Impact of Number of Blocks on WER")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig("ResultsA2/num_blocks_wer.png", bbox_inches='tight')
        plt.close()

    # 3. d_model / num_heads plot
    dmh_vals = parameter_sweeps["d_model_heads"]
    dmh_x = [v['d_model'] for v in dmh_vals]
    dmh_wers = [results[f"d_model_heads_{v}"]["best_wer"] for v in dmh_vals if f"d_model_heads_{v}" in results]
    if len(dmh_wers) == len(dmh_vals):
        plt.figure()
        plt.plot(dmh_x, [w * 100 for w in dmh_wers], marker='o', color='#d62728', linewidth=2)
        plt.xlabel("Model Dimension (d_model)")
        plt.ylabel("Word Error Rate (WER) %")
        plt.title("Impact of Model Dimension & Heads on WER")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig("ResultsA2/model_dim_wer.png", bbox_inches='tight')
        plt.close()

    # 4. use_attention plot
    ua_vals = parameter_sweeps["use_attention"]
    ua_labels = ["Conformer (Attention=True)", "Conv-only (Attention=False)"]
    ua_wers = [results[f"use_attention_{v}"]["best_wer"] for v in ua_vals if f"use_attention_{v}" in results]
    if len(ua_wers) == len(ua_vals):
        plt.figure()
        plt.bar(ua_labels, [w * 100 for w in ua_wers], color=['#9467bd', '#ff7f0e'], width=0.4)
        plt.ylabel("Word Error Rate (WER) %")
        plt.title("Attention Module comparison on WER")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.savefig("ResultsA2/use_attention_wer.png", bbox_inches='tight')
        plt.close()

    # 5. spec_augment plot
    sa_vals = parameter_sweeps["spec_augment"]
    sa_labels = ["SpecAugment=True", "SpecAugment=False"]
    sa_wers = [results[f"spec_augment_{v}"]["best_wer"] for v in sa_vals if f"spec_augment_{v}" in results]
    if len(sa_wers) == len(sa_vals):
        plt.figure()
        plt.bar(sa_labels, [w * 100 for w in sa_wers], color=['#2ca02c', '#d62728'], width=0.4)
        plt.ylabel("Word Error Rate (WER) %")
        plt.title("Impact of SpecAugment on WER")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.savefig("ResultsA2/spec_augment_wer.png", bbox_inches='tight')
        plt.close()

    # 6. dropout plot
    do_vals = parameter_sweeps["dropout"]
    do_wers = [results[f"dropout_{v}"]["best_wer"] for v in do_vals if f"dropout_{v}" in results]
    if len(do_wers) == len(do_vals):
        plt.figure()
        plt.plot(do_vals, [w * 100 for w in do_wers], marker='o', color='#ff7f0e', linewidth=2)
        plt.xlabel("Dropout Rate")
        plt.ylabel("Word Error Rate (WER) %")
        plt.title("Impact of Dropout on WER (with SpecAugment)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig("ResultsA2/dropout_wer.png", bbox_inches='tight')
        plt.close()

    # 7. max_lr plot
    lr_vals = parameter_sweeps["max_lr"]
    lr_wers = [results[f"max_lr_{v}"]["best_wer"] for v in lr_vals if f"max_lr_{v}" in results]
    if len(lr_wers) == len(lr_vals):
        plt.figure()
        plt.plot(lr_vals, [w * 100 for w in lr_wers], marker='o', color='#1f77b4', linewidth=2)
        plt.xlabel("Max Learning Rate")
        plt.ylabel("Word Error Rate (WER) %")
        plt.title("Impact of Max Learning Rate on WER (with SpecAugment)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig("ResultsA2/max_lr_wer.png", bbox_inches='tight')
        plt.close()

    # 8. freq_mask_width plot
    fmw_vals = parameter_sweeps["freq_mask_width"]
    fmw_wers = [results[f"freq_mask_width_{v}"]["best_wer"] for v in fmw_vals if f"freq_mask_width_{v}" in results]
    if len(fmw_wers) == len(fmw_vals):
        plt.figure()
        plt.plot(fmw_vals, [w * 100 for w in fmw_wers], marker='o', color='#9467bd', linewidth=2)
        plt.xlabel("Frequency Mask Width")
        plt.ylabel("Word Error Rate (WER) %")
        plt.title("Impact of Freq Mask Width on WER")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig("ResultsA2/freq_mask_width_wer.png", bbox_inches='tight')
        plt.close()

    # 9. time_mask_width plot
    tmw_vals = parameter_sweeps["time_mask_width"]
    tmw_wers = [results[f"time_mask_width_{v}"]["best_wer"] for v in tmw_vals if f"time_mask_width_{v}" in results]
    if len(tmw_wers) == len(tmw_vals):
        plt.figure()
        plt.plot(tmw_vals, [w * 100 for w in tmw_wers], marker='o', color='#8c564b', linewidth=2)
        plt.xlabel("Time Mask Width")
        plt.ylabel("Word Error Rate (WER) %")
        plt.title("Impact of Time Mask Width on WER")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig("ResultsA2/time_mask_width_wer.png", bbox_inches='tight')
        plt.close()

    # Write Markdown Report
    report_content = f"""# Hyperparameter Analysis Report

This report summarizes the impact of structural model parameters on the Word Error Rate (WER) of the Conformer speech recognition system, trained for {args.epochs} epochs.

## Summary of Results

### 1. Convolution Channels
| Channels | Best Dev WER | Best Dev Loss | Run Time (s) |
| :---: | :---: | :---: | :---: |
"""
    for v in cc_vals:
        key = f"conv_channels_{v}"
        if key in results:
            r = results[key]
            report_content += f"| {v} | {r['best_wer']:.2%} | {r['best_loss']:.4f} | {r['time_taken']:.1f} |\n"

    report_content += """
![Convolution Channels Impact](conv_channels_wer.png)

---

### 2. Number of Conformer Blocks
| Blocks | Best Dev WER | Best Dev Loss | Run Time (s) |
| :---: | :---: | :---: | :---: |
"""
    for v in nb_vals:
        key = f"num_blocks_{v}"
        if key in results:
            r = results[key]
            report_content += f"| {v} | {r['best_wer']:.2%} | {r['best_loss']:.4f} | {r['time_taken']:.1f} |\n"

    report_content += """
![Number of Blocks Impact](num_blocks_wer.png)

---

### 3. Model Dimension and Attention Heads
| d_model / heads | Best Dev WER | Best Dev Loss | Run Time (s) |
| :---: | :---: | :---: | :---: |
"""
    for v in dmh_vals:
        key = f"d_model_heads_{v}"
        if key in results:
            r = results[key]
            report_content += f"| {v['d_model']} / {v['num_heads']} | {r['best_wer']:.2%} | {r['best_loss']:.4f} | {r['time_taken']:.1f} |\n"

    report_content += """
![Model Dimension Impact](model_dim_wer.png)

---

### 4. Attention Module Comparison
| Architecture | Best Dev WER | Best Dev Loss | Run Time (s) |
| :---: | :---: | :---: | :---: |
"""
    for v in ua_vals:
        key = f"use_attention_{v}"
        if key in results:
            r = results[key]
            arch_label = "Conformer (MHA=True)" if v else "Conv-Only (MHA=False)"
            report_content += f"| {arch_label} | {r['best_wer']:.2%} | {r['best_loss']:.4f} | {r['time_taken']:.1f} |\n"

    report_content += """
![Attention Comparison](use_attention_wer.png)

---

### 5. SpecAugment Activation
| SpecAugment | Best Dev WER | Best Dev Loss | Run Time (s) |
| :---: | :---: | :---: | :---: |
"""
    for v in sa_vals:
        key = f"spec_augment_{v}"
        if key in results:
            r = results[key]
            sa_label = "Enabled (True)" if v else "Disabled (False)"
            report_content += f"| {sa_label} | {r['best_wer']:.2%} | {r['best_loss']:.4f} | {r['time_taken']:.1f} |\n"

    report_content += """
![SpecAugment Impact](spec_augment_wer.png)

---

### 6. Dropout Rate (with SpecAugment)
| Dropout | Best Dev WER | Best Dev Loss | Run Time (s) |
| :---: | :---: | :---: | :---: |
"""
    for v in do_vals:
        key = f"dropout_{v}"
        if key in results:
            r = results[key]
            report_content += f"| {v} | {r['best_wer']:.2%} | {r['best_loss']:.4f} | {r['time_taken']:.1f} |\n"

    report_content += """
![Dropout Impact](dropout_wer.png)

---

### 7. Max Learning Rate (with SpecAugment)
| Max LR | Best Dev WER | Best Dev Loss | Run Time (s) |
| :---: | :---: | :---: | :---: |
"""
    for v in lr_vals:
        key = f"max_lr_{v}"
        if key in results:
            r = results[key]
            report_content += f"| {v} | {r['best_wer']:.2%} | {r['best_loss']:.4f} | {r['time_taken']:.1f} |\n"

    report_content += """
![Learning Rate Impact](max_lr_wer.png)

---

### 8. SpecAugment Frequency Mask Width
| Width | Best Dev WER | Best Dev Loss | Run Time (s) |
| :---: | :---: | :---: | :---: |
"""
    for v in fmw_vals:
        key = f"freq_mask_width_{v}"
        if key in results:
            r = results[key]
            report_content += f"| {v} | {r['best_wer']:.2%} | {r['best_loss']:.4f} | {r['time_taken']:.1f} |\n"

    report_content += """
![Freq Mask Width Impact](freq_mask_width_wer.png)

---

### 9. SpecAugment Time Mask Width
| Width | Best Dev WER | Best Dev Loss | Run Time (s) |
| :---: | :---: | :---: | :---: |
"""
    for v in tmw_vals:
        key = f"time_mask_width_{v}"
        if key in results:
            r = results[key]
            report_content += f"| {v} | {r['best_wer']:.2%} | {r['best_loss']:.4f} | {r['time_taken']:.1f} |\n"

    report_content += """
![Time Mask Width Impact](time_mask_width_wer.png)
"""

    report_path = "ResultsA2/hyperparameter_analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Analysis complete. Results and report saved to {report_path}")

if __name__ == "__main__":
    main()
