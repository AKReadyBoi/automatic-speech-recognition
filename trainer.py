import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


class Trainer:
    def __init__(self, model, train_loader, dev_loader, device,
                 max_lr=1e-3, epochs=5,
                 checkpoint_interval=0,  # save every N batches, 0 = only per epoch
                 early_stopping=False,
                 patience=3,
                 dictionary=None):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.device = device
        self.epochs = epochs
        self.checkpoint_interval = checkpoint_interval
        self.early_stopping = early_stopping
        self.patience = patience
        self.dictionary = dictionary

        self.optimizer = AdamW(model.parameters(), lr=max_lr)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            steps_per_epoch=len(train_loader) if len(train_loader) > 0 else 1,
            epochs=epochs if epochs > 0 else 1
        )

        self.best_dev_loss = float('inf')
        self.best_dev_wer = float('inf')
        self.patience_counter = 0

    def train_step(self, batch):
        padded_features, feature_lengths, padded_labels, label_lengths = batch

        padded_features = padded_features.to(self.device)
        padded_labels   = padded_labels.to(self.device)
        feature_lengths = feature_lengths.to(self.device)
        label_lengths   = label_lengths.to(self.device)

        # frontend pool stride = 4, so reduce lengths accordingly
        input_lengths = feature_lengths // 4

        self.optimizer.zero_grad()
        logits = self.model(padded_features, input_lengths)  # (batch, time/4, vocab)

        # CTC loss expects (time, batch, vocab) with log_softmax
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)         # (time/4, batch, vocab)

        loss = F.ctc_loss(
            log_probs,
            padded_labels,
            input_lengths,
            label_lengths,
            zero_infinity=True
        )

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def train_epoch(self, epoch, log_interval=50):
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(self.train_loader):
            loss = self.train_step(batch)
            total_loss += loss

            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | CTC Loss: {loss:.4f}")

            # mid-epoch checkpoints
            if self.checkpoint_interval > 0 and batch_idx > 0 and batch_idx % self.checkpoint_interval == 0:
                self.save_checkpoint(epoch, step=batch_idx)

        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch} | Avg Train CTC Loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.dev_loader:
                padded_features, feature_lengths, padded_labels, label_lengths = batch

                padded_features = padded_features.to(self.device)
                padded_labels   = padded_labels.to(self.device)
                feature_lengths = feature_lengths.to(self.device)
                label_lengths   = label_lengths.to(self.device)

                input_lengths = feature_lengths // 4
                logits    = self.model(padded_features, input_lengths)
                log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

                loss = F.ctc_loss(
                    log_probs,
                    padded_labels,
                    input_lengths,
                    label_lengths,
                    zero_infinity=True
                )
                total_loss += loss.item()

        avg_loss = total_loss / len(self.dev_loader)
        print(f"Dev CTC Loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self, epoch, step=None):
        os.makedirs("checkpoints", exist_ok=True)
        name = f"checkpoint_epoch{epoch}" + (f"_step{step}" if step is not None else "") + ".pt"
        path = os.path.join("checkpoints", name)
        torch.save({
            "epoch": epoch,
            "step": step,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")

    def print_model_summary(self):
        print("\n=== Model Summary ===")
        total = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total:,}\n")
        print(f"{'Layer':<50} {'Output Shape'}")
        print("-" * 70)

        hooks = []

        def make_hook(name):
            def hook(_module, _input, output):
                if isinstance(output, torch.Tensor):
                    print(f"{name:<50} {list(output.shape)}")
            return hook

        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules only
                hooks.append(module.register_forward_hook(make_hook(name)))

        dummy = torch.randn(1, 200, 80).to(self.device)
        with torch.no_grad():
            self.model(dummy)

        for hook in hooks:
            hook.remove()
        print("=" * 70 + "\n")

    def evaluate_wer(self):
        # Greedy decoding over the dev set -> WER. Used to pick the best checkpoint.
        if self.dev_loader is None or len(self.dev_loader) == 0:
            print("No validation data available for WER. Skipping evaluation.")
            return 0.0

        from wer import greedy_decode
        from levenshtein_distance import lev_dist

        self.model.eval()
        dist_count = 0
        total_ref_words = 0

        with torch.no_grad():
            for batch in self.dev_loader:
                padded_features, feature_lengths, padded_labels, label_lengths = batch
                padded_features = padded_features.to(self.device)
                input_lengths = feature_lengths // 4
                # pass lengths so self-attention masks padded frames
                logits = self.model(padded_features, input_lengths.to(self.device))

                for i in range(logits.shape[0]):
                    utt_logits = logits[i][:input_lengths[i]]
                    hyp_text = greedy_decode(utt_logits, self.dictionary)

                    ref_indices = padded_labels[i][:label_lengths[i]].tolist()
                    ref_tokens = self.dictionary.decode(ref_indices)
                    ref_text = "".join(ref_tokens).replace("</w>", " ").strip()

                    hyp_words = hyp_text.split()
                    ref_words = ref_text.split()

                    dist_count += lev_dist(hyp_words, ref_words)
                    total_ref_words += len(ref_words)

        return dist_count / total_ref_words if total_ref_words > 0 else 0.0

    def save_best_checkpoint(self, epoch, dev_wer):
        os.makedirs("checkpoints", exist_ok=True)
        path = os.path.join("checkpoints", "best_wer_checkpoint.pt")
        torch.save({
            "epoch": epoch,
            "dev_wer": dev_wer,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)
        print(f"Saved new best WER checkpoint: {path} (WER: {dev_wer:.2%})")

    def train(self):
        self.print_model_summary()

        for epoch in range(1, self.epochs + 1):
            self.train_epoch(epoch)
            dev_loss = self.evaluate()
            self.save_checkpoint(epoch)  # always save at end of epoch

            # greedy-decode WER on dev + keep the best checkpoint (Aufgabe 3 hint)
            if self.dictionary is not None:
                dev_wer = self.evaluate_wer()
                print(f"Epoch {epoch} | Dev WER: {dev_wer:.2%}")
                if dev_wer < self.best_dev_wer:
                    self.best_dev_wer = dev_wer
                    self.save_best_checkpoint(epoch, dev_wer)

            if self.early_stopping:
                if dev_loss < self.best_dev_loss:
                    self.best_dev_loss = dev_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    print(f"No improvement: {self.patience_counter}/{self.patience}")
                    if self.patience_counter >= self.patience:
                        print("Early stopping triggered.")
                        break