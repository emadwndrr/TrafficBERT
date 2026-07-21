from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertModel

RESULTS_DIR = "results"
DATASETS_DIR = "generated_datasets"

# "20class": benign + malware (20 classes)
# "10class": malware only (10 classes)
TASK = "20class"


BATCH_SIZE = 512


class TrafficDataset(Dataset):
    def __init__(self, features, masks, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.int64)
        self.labels = torch.tensor(labels, dtype=torch.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.masks[idx], self.labels[idx]


class TrafficBERT(nn.Module):
    def __init__(self, num_features, seq_len, num_classes):
        super().__init__()

        config = BertConfig(
            vocab_size=2,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=1024,
            max_position_embeddings=seq_len,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )

        self.input_projection = nn.Linear(num_features, config.hidden_size)
        self.encoder = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x, attention_mask):
        x = self.input_projection(x)
        outputs = self.encoder(inputs_embeds=x, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)


def evaluate(model, loader, device, idx_to_label):
    model.eval()
    num_classes = len(idx_to_label)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for features, masks, labels in loader:
            features, masks = features.to(device), masks.to(device)
            preds = model(features, masks).argmax(dim=-1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    accuracy = (all_preds == all_labels).mean()
    f1_overall = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    class_correct = np.array(
        [(all_preds[all_labels == c] == c).sum() for c in range(num_classes)]
    )
    class_total = np.array([(all_labels == c).sum() for c in range(num_classes)])
    class_accs = {
        c: (class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0)
        for c in range(num_classes)
    }

    print(f"\n  {'Label':<30} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print(f"  {'-' * 58}")
    for idx, label in sorted(idx_to_label.items(), key=lambda x: x[0]):
        print(
            f"  {label:<30} {int(class_correct[idx]):>8} {int(class_total[idx]):>8} {class_accs[idx]:>8.4f}"
        )

    benign_idx = [i for i, l in idx_to_label.items() if l.startswith("Benign/")]
    malware_idx = [i for i, l in idx_to_label.items() if l.startswith("Malware/")]

    m_correct = class_correct[malware_idx].sum()
    m_total = class_total[malware_idx].sum()
    m_acc = m_correct / m_total if m_total > 0 else 0.0
    m_f1 = f1_score(
        all_labels, all_preds, labels=malware_idx, average="macro", zero_division=0
    )

    print(f"\n  {'':30} {'Acc':>8} {'F1 (macro)':>12}")
    print(f"  {'-' * 52}")
    print(f"  {'Overall':<30} {accuracy:>8.4f} {f1_overall:>12.4f}")
    if benign_idx:
        b_correct = class_correct[benign_idx].sum()
        b_total = class_total[benign_idx].sum()
        b_acc = b_correct / b_total if b_total > 0 else 0.0
        b_f1 = f1_score(
            all_labels, all_preds, labels=benign_idx, average="macro", zero_division=0
        )
        print(f"  {'Benign':<30} {b_acc:>8.4f} {b_f1:>12.4f}")
    print(f"  {'Malware':<30} {m_acc:>8.4f} {m_f1:>12.4f}")


def plot_confusion_matrix(model, loader, device, idx_to_label, save_path):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for features, masks, labels in loader:
            features, masks = features.to(device), masks.to(device)
            preds = model(features, masks).argmax(dim=-1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    label_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(16, 14))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()
    print(f"Saved: {save_path}")


task_dir = Path(RESULTS_DIR) / TASK
checkpoint = torch.load(task_dir / "traffic_bert_best.pt", weights_only=False)

label_to_idx = checkpoint["label_to_idx"]
idx_to_label = {v: k for k, v in label_to_idx.items()}
N = checkpoint["N"]
num_classes = len(label_to_idx)

data = np.load(Path(DATASETS_DIR) / TASK / "test.npz")
test_dataset = TrafficDataset(data["features"], data["masks"], data["labels"])
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")
print(f"task   : {TASK}")
print(f"epoch  : {checkpoint['epoch']}  malware_acc={checkpoint['malware_acc']:.4f}\n")

model = TrafficBERT(num_features=5, seq_len=N, num_classes=num_classes).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

evaluate(model, test_loader, device, idx_to_label)
plot_confusion_matrix(
    model,
    test_loader,
    device,
    idx_to_label,
    save_path=str(task_dir / "confusion_matrix.png"),
)
