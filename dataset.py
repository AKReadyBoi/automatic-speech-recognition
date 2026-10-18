from dataclasses import dataclass
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset

from bpe import bpe_apply


@dataclass
class SpecAugmentConfig:
    freq_masks: int = 2
    freq_mask_width: int = 15
    time_masks: int = 2
    time_mask_width: int = 25
    max_time_mask_fraction: float = 0.15
    mask_value: float = 0.0


class BPEDataset(Dataset):
    def __init__(
        self,
        corpus_path,
        learned_rules,
        dictionary,
        feature_extractor,
        spec_augment=False,
        spec_augment_config=None,
    ):
        self.dictionary = dictionary
        self.feature_extractor = feature_extractor
        self.spec_augment = spec_augment
        self.spec_augment_config = spec_augment_config or SpecAugmentConfig()
        self.data = []  # list of (audio_path, label_indices)

        tree = ET.parse(corpus_path)
        for recording in tree.getroot().findall(".//recording"):
            audio_path = recording.attrib["audio"]
            orth = recording.find(".//orth")
            if orth is None or not orth.text:
                continue
            nested = bpe_apply(orth.text, learned_rules)
            flat = [token for word in nested for token in word]
            indices = dictionary.encode(flat)
            self.data.append((audio_path, indices))

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _sample_mask_width(max_width, limit):
        width_limit = min(max_width, limit)
        if width_limit <= 0:
            return 0
        return torch.randint(1, width_limit + 1, (1,)).item()

    def apply_spec_augment(self, features):
        """Apply SpecAugment frequency and time masking to (time, n_mels)."""
        augmented = features.clone()
        time_steps, num_mels = augmented.shape
        config = self.spec_augment_config

        for _ in range(config.freq_masks):
            width = self._sample_mask_width(config.freq_mask_width, num_mels)
            if width > 0:
                start = torch.randint(0, num_mels - width + 1, (1,)).item()
                augmented[:, start:start + width] = config.mask_value

        max_time_width = min(
            config.time_mask_width,
            int(time_steps * config.max_time_mask_fraction),
        )
        for _ in range(config.time_masks):
            width = self._sample_mask_width(max_time_width, time_steps)
            if width > 0:
                start = torch.randint(0, time_steps - width + 1, (1,)).item()
                augmented[start:start + width, :] = config.mask_value

        return augmented

    def __getitem__(self, idx):
        audio_path, label_indices = self.data[idx]
        features = self.feature_extractor(audio_path)  # (time, n_mels)

        if self.spec_augment:
            features = self.apply_spec_augment(features)

        labels = torch.tensor(label_indices, dtype=torch.long)
        return features, labels
