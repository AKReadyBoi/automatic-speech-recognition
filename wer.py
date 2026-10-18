import torch
import numpy as np
import xml.etree.ElementTree as ET
import argparse

from model import Conformer
from dictionary import Dictionary
from levenshtein_distance import lev_dist

def greedy_decode(logits, dictionary):
    
    high_idx = torch.argmax(logits, dim=-1).tolist()  
    blank_idx = dictionary.blank_idx

    if not high_idx:
        return ""

    seq = []
    prev = None  # Speichert das vorherige Element

    for val in high_idx:
        if val != prev:  # Duplikat-Check
            if val != blank_idx:  # Blank-Check
                seq.append(val)
            prev = val  
        
    token_list = dictionary.decode(seq)
    decoded_text = "".join(token_list).replace("</w>", " ").strip()

    return decoded_text

def parse_corpus(xml_path):
    
    tree = ET.parse(xml_path)
    root = tree.getroot()

    path_text_pairs = dict()

    for i in root.findall('.//recording'):
        audio_path = i.attrib.get('audio')
        orth = i.find('.//orth')

        if orth is not None and orth.text:
            orth_text = orth.text.strip()
            path_text_pairs[audio_path] = orth_text

    return path_text_pairs


def evaluate_model(model, corpus_data, features, dictionary, device):
    dist_count = 0
    total_ref_words = 0
    total_items = len(corpus_data)
    processed = 0

    model.eval()
    for audio_path, ref_text in corpus_data.items():
        if audio_path not in features:
            print(f'No features found for {audio_path}, skipping.')
            continue
        
        audio_features = features[audio_path]
        audio_tensor = torch.from_numpy(audio_features).float().unsqueeze(0).to(device)  # (1, T, F)
        input_lengths = torch.tensor([audio_tensor.shape[1] // 4], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(audio_tensor, input_lengths)  # (1, T, V)
        
        hyp_words = greedy_decode(logits.squeeze(0)[:input_lengths.item()], dictionary).split()
        ref_words = ref_text.split()
        dist_count += lev_dist(hyp_words, ref_words)
        total_ref_words += len(ref_words)
        
        processed += 1
        if processed % 250 == 0:
            percent = (processed / total_items) * 100
            print(f"Decoded {processed}/{total_items} utterances ({percent:.1f}%)")
    
    wer = 0.0
    if total_ref_words > 0: 
        wer = dist_count / total_ref_words
    return wer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint",     type=str,   default="checkpoints/checkpoint_epoch5.pt")
    parser.add_argument("--xml_corpus",           type=str,   default="dev_corpus.xml")
    parser.add_argument("--feature_path",         type=str,   default="spect.npy")
    parser.add_argument("--num_blocks",           type=int,   default=6)
    parser.add_argument("--dropout",              type=float, default=0.15)
    parser.add_argument("--d_model",              type=int,   default=100)
    parser.add_argument("--conv_channels",        type=int,   default=31)
    parser.add_argument("--num_heads",            type=int,   default=4)   # must match training; not inferable from weight shapes
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dictionary = Dictionary()
    dictionary.load("ResultsA2/dictionary.json")

    features = np.load(args.feature_path, allow_pickle=True)[()]
    
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    
    # Automatically infer model configuration from checkpoint
    if "frontend.linear.weight" in state_dict:
        inferred_d_model = state_dict["frontend.linear.weight"].shape[0]
        inferred_n_mels = state_dict["frontend.linear.weight"].shape[1] // 16
    else:
        inferred_d_model = 100
        inferred_n_mels = 80
        
    if "output_linear.weight" in state_dict:
        inferred_vocab_size = state_dict["output_linear.weight"].shape[0]
    else:
        inferred_vocab_size = len(dictionary)
        
    inferred_num_blocks = 0
    while f"blocks.{inferred_num_blocks}.layernorm.weight" in state_dict:
        inferred_num_blocks += 1
        
    if "blocks.0.convolutionalBlock.pointwise_conv1.weight" in state_dict:
        inferred_conv_channels = state_dict["blocks.0.convolutionalBlock.pointwise_conv1.weight"].shape[0] // 2
    else:
        inferred_conv_channels = 31
    inferred_use_attention = any("attentionBlock" in key for key in state_dict)

    print(f"Inferred model config from checkpoint: d_model={inferred_d_model}, num_blocks={inferred_num_blocks}, conv_channels={inferred_conv_channels}, num_heads={args.num_heads}, n_mels={inferred_n_mels}, vocab_size={inferred_vocab_size}, use_attention={inferred_use_attention}")

    model = Conformer(
        d_model=inferred_d_model,
        num_blocks=inferred_num_blocks,
        vocab_size=inferred_vocab_size,
        dropout=args.dropout,
        conv_channels=inferred_conv_channels,
        n_mels=inferred_n_mels,
        num_heads=args.num_heads,
        use_attention=inferred_use_attention
    ).to(device)
    
    model.load_state_dict(state_dict)

    corpus_data = parse_corpus(args.xml_corpus)
    
    print("Evaluating model...")
    wer = evaluate_model(model, corpus_data, features, dictionary, device)
    print(f"WER: {wer:.4f} ({wer:.2%})")


if __name__ == "__main__":
    main()
