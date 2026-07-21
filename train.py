import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertModel


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


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for features, masks, labels in loader:
        features, masks, labels = (
            features.to(device),
            masks.to(device),
            labels.to(device),
        )

        optimizer.zero_grad()
        loss = criterion(model(features, masks), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, idx_to_label):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    num_classes = len(idx_to_label)
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)

    with torch.no_grad():
        for features, masks, labels in loader:
            features, masks, labels = (
                features.to(device),
                masks.to(device),
                labels.to(device),
            )

            logits = model(features, masks)
            preds = logits.argmax(dim=-1)
            total_loss += criterion(logits, labels).item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for c in range(num_classes):
                mask_c = labels == c
                class_correct[c] += (preds[mask_c] == c).sum().item()
                class_total[c] += mask_c.sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    class_accs = {
        idx: (
            class_correct[idx].item() / class_total[idx].item()
            if class_total[idx] > 0
            else 0.0
        )
        for idx in range(num_classes)
    }
    class_correct_dict = {idx: class_correct[idx].item() for idx in range(num_classes)}
    class_total_dict = {idx: class_total[idx].item() for idx in range(num_classes)}

    print(f"  loss={avg_loss:.4f}  acc={accuracy:.4f}")
    print(f"\n  {'Label':<30} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print(f"  {'-' * 58}")
    for idx, label in sorted(idx_to_label.items(), key=lambda x: x[0]):
        ct = class_total[idx].item()
        cc = class_correct[idx].item()
        print(f"  {label:<30} {int(cc):>8} {int(ct):>8} {class_accs[idx]:>8.4f}")

    benign_idx = [i for i, l in idx_to_label.items() if l.startswith("Benign/")]
    malware_idx = [i for i, l in idx_to_label.items() if l.startswith("Malware/")]

    b_correct = sum(class_correct[i].item() for i in benign_idx)
    b_total = sum(class_total[i].item() for i in benign_idx)
    m_correct = sum(class_correct[i].item() for i in malware_idx)
    m_total = sum(class_total[i].item() for i in malware_idx)

    m_acc = m_correct / m_total if m_total > 0 else 0.0

    if benign_idx:
        b_acc = b_correct / b_total if b_total > 0 else 0.0
        print(f"\n  Benign  acc={b_acc:.4f}  ({int(b_correct)}/{int(b_total)})")
    print(f"  Malware acc={m_acc:.4f}  ({int(m_correct)}/{int(m_total)})")

    return avg_loss, accuracy, class_accs, class_correct_dict, class_total_dict


def plot_training_overview(
    history, idx_to_label, save_path="/kaggle/working/training_overview.png"
):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    benign_idx = sorted([i for i, l in idx_to_label.items() if l.startswith("Benign/")])
    malware_idx = sorted(
        [i for i, l in idx_to_label.items() if l.startswith("Malware/")]
    )

    malware_acc_per_epoch = []
    benign_acc_per_epoch = [] if benign_idx else None
    for e in range(len(epochs)):
        m_correct = sum(history["class_correct"][e][i] for i in malware_idx)
        m_total = sum(history["class_total"][e][i] for i in malware_idx)
        malware_acc_per_epoch.append(m_correct / m_total if m_total > 0 else 0.0)
        if benign_idx:
            b_correct = sum(history["class_correct"][e][i] for i in benign_idx)
            b_total = sum(history["class_total"][e][i] for i in benign_idx)
            benign_acc_per_epoch.append(b_correct / b_total if b_total > 0 else 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax1 = axes[0]
    ax1.plot(epochs, history["train_loss"], label="train", color="steelblue")
    ax1.plot(epochs, history["val_loss"], label="val", color="coral")
    if history["best_epoch"]:
        ax1.axvline(
            history["best_epoch"],
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"best={history['best_epoch']}",
        )
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(epochs, history["lr"], color="purple")
    ax2.set_title("Learning rate")
    ax2.set_xlabel("Epoch")
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    if benign_idx:
        ax3.plot(epochs, benign_acc_per_epoch, label="Benign", color="steelblue")
    ax3.plot(epochs, malware_acc_per_epoch, label="Malware", color="coral")
    if history["best_epoch"]:
        ax3.axvline(
            history["best_epoch"],
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"best epoch {history['best_epoch']}",
        )
    ax3.set_title(
        "Overall benign vs malware accuracy"
        if benign_idx
        else "Overall malware accuracy"
    )
    ax3.set_xlabel("Epoch")
    ax3.set_ylim(0, 1)
    ax3.axhline(0.9, color="gray", linestyle="--", alpha=0.4, label="0.90 target")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_per_class_accuracy(
    history, idx_to_label, save_path="/kaggle/working/per_class_accuracy.png"
):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    class_acc_matrix = np.array(
        [
            [history["class_accs"][e][i] for i in range(num_classes)]
            for e in range(len(epochs))
        ]
    )

    benign_idx = sorted([i for i, l in idx_to_label.items() if l.startswith("Benign/")])
    malware_idx = sorted(
        [i for i, l in idx_to_label.items() if l.startswith("Malware/")]
    )
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    nrows = 2 if benign_idx else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 6 * nrows))
    if nrows == 1:
        axes = [axes]

    if benign_idx:
        ax1 = axes[0]
        for i, idx in enumerate(benign_idx):
            label = idx_to_label[idx].replace("Benign/", "")
            final_acc = class_acc_matrix[-1, idx]
            ax1.plot(
                epochs,
                class_acc_matrix[:, idx],
                label=f"{label} ({final_acc:.3f})",
                color=colors[i],
            )
        ax1.set_title("Per-class accuracy: Benign")
        ax1.set_xlabel("Epoch")
        ax1.set_ylim(0, 1)
        ax1.axhline(0.9, color="gray", linestyle="--", alpha=0.4)
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)

    ax2 = axes[-1]
    for i, idx in enumerate(malware_idx):
        label = idx_to_label[idx].replace("Malware/", "")
        final_acc = class_acc_matrix[-1, idx]
        ax2.plot(
            epochs,
            class_acc_matrix[:, idx],
            label=f"{label} ({final_acc:.3f})",
            color=colors[i],
        )
    ax2.set_title("Per-class accuracy: Malware")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(0, 1)
    ax2.axhline(0.9, color="gray", linestyle="--", alpha=0.4)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_confusion_matrix(
    model,
    loader,
    device,
    idx_to_label,
    save_path="/kaggle/working/confusion_matrix.png",
):
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


def load_dataset(data_dir):
    data_dir = Path(data_dir)

    train = np.load(data_dir / "train.npz")
    test = np.load(data_dir / "test.npz")

    with open(data_dir / "label_to_idx.json") as f:
        label_to_idx = json.load(f)

    idx_to_label = {v: k for k, v in label_to_idx.items()}

    train_dataset = TrafficDataset(train["features"], train["masks"], train["labels"])
    test_dataset = TrafficDataset(test["features"], test["masks"], test["labels"])

    return train_dataset, test_dataset, label_to_idx, idx_to_label


def compute_class_weights(labels, num_classes):
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    weights = 1.0 / counts.clamp(min=1)
    return weights / weights.sum()


RESULTS_DIR = "results"


# "20class": benign + malware (20 classes)
# "10class": malware only (10 classes)
TASK = "10class"

DATA_DIR = f"generated_datasets/{TASK}"
N = 20
BATCH_SIZE = 512
EPOCHS = 50
LR = 1e-4

task_dir = Path(RESULTS_DIR) / TASK
task_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

train_dataset, test_dataset, label_to_idx, idx_to_label = load_dataset(DATA_DIR)

num_classes = len(label_to_idx)
print(f"num_classes : {num_classes}")
print(f"train flows : {len(train_dataset):,}")
print(f"test  flows : {len(test_dataset):,}")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
)

model = TrafficBERT(num_features=5, seq_len=N, num_classes=num_classes).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

class_weights = compute_class_weights(train_dataset.labels.numpy(), num_classes).to(
    device
)
criterion = nn.CrossEntropyLoss(weight=class_weights)

history = {
    "train_loss": [],
    "val_loss": [],
    "val_acc": [],
    "lr": [],
    "class_accs": [],
    "class_correct": [],
    "class_total": [],
    "best_epoch": None,
}
best_malware_acc = 0.0
malware_idx = [i for i, l in idx_to_label.items() if l.startswith("Malware/")]

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    print(f"\nepoch {epoch + 1:02d}/{EPOCHS}  train_loss={train_loss:.4f}")

    val_loss, val_acc, class_accs, class_correct, class_total = evaluate(
        model, test_loader, criterion, device, idx_to_label
    )
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["lr"].append(scheduler.get_last_lr()[0])
    history["class_accs"].append(class_accs)
    history["class_correct"].append(class_correct)
    history["class_total"].append(class_total)

    m_correct = sum(class_correct[i] for i in malware_idx)
    m_total = sum(class_total[i] for i in malware_idx)
    m_acc = m_correct / m_total if m_total > 0 else 0.0

    if m_acc > best_malware_acc:
        best_malware_acc = m_acc
        history["best_epoch"] = epoch + 1
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "malware_acc": m_acc,
                "label_to_idx": label_to_idx,
                "N": N,
            },
            str(task_dir / "traffic_bert_best.pt"),
        )
        print(f"  ✓ best model saved (malware_acc={m_acc:.4f})")


checkpoint = torch.load(str(task_dir / "traffic_bert_best.pt"))
model.load_state_dict(checkpoint["model_state_dict"])
print(
    f"\nloaded best model from epoch {checkpoint['epoch']} with malware_acc={checkpoint['malware_acc']:.4f}"
)

plot_training_overview(
    history, idx_to_label, save_path=str(task_dir / "training_overview.png")
)
plot_per_class_accuracy(
    history, idx_to_label, save_path=str(task_dir / "per_class_accuracy.png")
)
plot_confusion_matrix(
    model,
    test_loader,
    device,
    idx_to_label,
    save_path=str(task_dir / "confusion_matrix.png"),
)
