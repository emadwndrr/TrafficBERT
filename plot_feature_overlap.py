from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pcap.flow import extract_flows

DATASET_DIR = "./pcap_datasets"
N = 20


# "20class": benign + malware (20 classes)
# "10class": malware only (10 classes)
TASK = "10class"


RESULTS_DIR = "results"

FILE_LABELS = {
    "Miuref.pcap": "Malware/Miuref",
    "Zeus.pcap": "Malware/Zeus",
    "Tinba.pcap": "Malware/Tinba",
    "Neris.pcap": "Malware/Neris",
    "Nsis-ay.pcap": "Malware/Nsis-ay",
    "Geodo.pcap": "Malware/Geodo",
    "Htbot.pcap": "Malware/Htbot",
    "Virut.pcap": "Malware/Virut",
    "Cridex.pcap": "Malware/Cridex",
    "Shifu.pcap": "Malware/Shifu",
    "Outlook.pcap": "Benign/Outlook",
    "Gmail.pcap": "Benign/Gmail",
    "Facetime.pcap": "Benign/Facetime",
    "MySQL.pcap": "Benign/MySQL",
    "FTP.pcap": "Benign/FTP",
    "BitTorrent.pcap": "Benign/BitTorrent",
    "Skype.pcap": "Benign/Skype",
    "WorldOfWarcraft.pcap": "Benign/WorldOfWarcraft",
    "Weibo-4.pcap": "Benign/Weibo",
    "Weibo-1.pcap": "Benign/Weibo",
    "Weibo-3.pcap": "Benign/Weibo",
    "Weibo-2.pcap": "Benign/Weibo",
    "SMB-2.pcap": "Benign/SMB",
    "SMB-1.pcap": "Benign/SMB",
}


def plot_aggregated_distributions(
    labeled_flows, N, save_path="feature_distributions.png"
):
    labels = sorted(labeled_flows.keys())
    benign_labels = [l for l in labels if l.startswith("Benign/")]
    malware_labels = [l for l in labels if l.startswith("Malware/")]
    ordered_labels = benign_labels + malware_labels

    mean_sizes, log_mean_iats, c2s_ratios = [], [], []

    for label in ordered_labels:
        s_list, iat_list, c2s_list = [], [], []
        for flow in labeled_flows[label].values():
            pkts = flow.packets[:N]
            if not pkts:
                continue
            s_list.append(np.mean([p.size for p in pkts]))

            # position 0 skipped because iat is zero
            real_iats = [p.inter_arrival_time for p in pkts[1:]]
            iat_list.append(np.mean(real_iats) if real_iats else 0.0)
            c2s_list.append(
                np.mean([1.0 if p.direction == "c2s" else 0.0 for p in pkts])
            )
        mean_sizes.append(np.array(s_list))
        log_mean_iats.append(np.log1p(np.array(iat_list)))
        c2s_ratios.append(np.array(c2s_list))

    short_labels = [
        l.replace("Benign/", "B/").replace("Malware/", "M/") for l in ordered_labels
    ]
    colors = [
        "#4C72B0" if l.startswith("Benign/") else "#DD8452" for l in ordered_labels
    ]

    fig, axes = plt.subplots(3, 1, figsize=(18, 14))

    for ax, data, title, ylabel in zip(
        axes,
        [mean_sizes, log_mean_iats, c2s_ratios],
        ["Mean packet size (bytes)", "Mean IAT — log1p(seconds)", "C2S packet ratio"],
        ["bytes", "log1p(s)", "ratio [0–1]"],
    ):
        parts = ax.violinplot(
            data,
            positions=range(len(ordered_labels)),
            showmedians=True,
            showextrema=False,
        )
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("black")
        ax.set_xticks(range(len(ordered_labels)))
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        ax.axvline(
            len(benign_labels) - 0.5, color="gray", linestyle="--", linewidth=0.8
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {save_path}")


def plot_per_position(labeled_flows, N, save_path="per_position_features.png"):
    labels = sorted(labeled_flows.keys())
    benign_labels = [l for l in labels if l.startswith("Benign/")]
    malware_labels = [l for l in labels if l.startswith("Malware/")]

    positions = np.arange(N)
    iat_positions = np.arange(1, N)

    colors_m = plt.cm.tab10(np.linspace(0, 1, len(malware_labels)))

    groups = [(malware_labels, colors_m, "Malware")]
    if benign_labels:
        colors_b = plt.cm.tab10(np.linspace(0, 1, len(benign_labels)))
        groups.insert(0, (benign_labels, colors_b, "Benign"))

    ncols = len(groups)
    fig, axes = plt.subplots(2, ncols, figsize=(8 * ncols, 10))
    if ncols == 1:
        axes = axes[:, np.newaxis]

    for col, (group_labels, colors, group_name) in enumerate(groups):
        ax_size, ax_iat = axes[0, col], axes[1, col]
        for label, color in zip(group_labels, colors):
            size_sums = np.zeros(N)
            iat_sums = np.zeros(N)
            counts = np.zeros(N)

            for flow in labeled_flows[label].values():
                for i, pkt in enumerate(flow.packets[:N]):
                    size_sums[i] += pkt.size
                    iat_sums[i] += pkt.inter_arrival_time
                    counts[i] += 1

            valid = counts > 0
            mean_sizes = np.where(valid, size_sums / np.maximum(counts, 1), np.nan)
            mean_iats = np.where(valid, iat_sums / np.maximum(counts, 1), np.nan)

            short = label.split("/")[1]
            ax_size.plot(positions, mean_sizes, label=short, color=color, linewidth=1.5)
            ax_iat.plot(
                iat_positions, mean_iats[1:], label=short, color=color, linewidth=1.5
            )

        ax_size.set_title(f"Per-position mean packet size — {group_name}", fontsize=11)
        ax_size.set_xlabel("Packet position")
        ax_size.set_ylabel("Bytes")
        ax_size.legend(fontsize=7)
        ax_size.grid(True, alpha=0.3)

        ax_iat.set_title(f"Per-position mean IAT — {group_name}", fontsize=11)
        ax_iat.set_xlabel("Packet position")
        ax_iat.set_ylabel("Seconds")
        ax_iat.legend(fontsize=7)
        ax_iat.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {save_path}")


def plot_tsne(labeled_flows, N, max_samples_per_class=500, save_path="tsne.png"):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    labels = sorted(labeled_flows.keys())
    rng = np.random.default_rng(42)

    X, y = [], []

    for label_idx, label in enumerate(labels):
        flows = list(labeled_flows[label].values())
        if len(flows) > max_samples_per_class:
            idxs = rng.choice(len(flows), max_samples_per_class, replace=False)
            flows = [flows[i] for i in idxs]

        for flow in flows:
            pkts = flow.packets[:N]
            if not pkts:
                continue
            sizes = [p.size for p in pkts]
            real_iats = [p.inter_arrival_time for p in pkts[1:]]
            dirs = [1.0 if p.direction == "c2s" else 0.0 for p in pkts]
            c2s_sizes = [p.size if p.direction == "c2s" else 0.0 for p in pkts]
            s2c_sizes = [p.size if p.direction == "s2c" else 0.0 for p in pkts]

            X.append(
                [
                    np.mean(sizes),
                    np.log1p(np.mean(real_iats)) if real_iats else 0.0,
                    np.mean(dirs),
                    np.mean(c2s_sizes),
                    np.mean(s2c_sizes),
                ]
            )
            y.append(label_idx)

    X = StandardScaler().fit_transform(np.array(X))
    y = np.array(y)

    print("Running t-SNE...")
    X_2d = TSNE(
        n_components=2, random_state=42, perplexity=30, max_iter=1000
    ).fit_transform(X)

    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(14, 10))
    for label_idx, label in enumerate(labels):
        mask = y == label_idx
        short = label.replace("Benign/", "B/").replace("Malware/", "M/")
        marker = "o" if label.startswith("Benign/") else "x"
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            label=short,
            color=colors[label_idx],
            marker=marker,
            s=10,
            alpha=0.6,
        )

    ax.set_title(f"t-SNE of per-flow aggregated features (N={N})", fontsize=13)
    ax.legend(fontsize=7, markerscale=2, ncol=2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {save_path}")


task_dir = Path(RESULTS_DIR) / TASK
task_dir.mkdir(parents=True, exist_ok=True)

raw_pcaps_dir = Path(DATASET_DIR)
pcap_files = list(raw_pcaps_dir.rglob("*.pcap"))

flows = extract_flows(pcap_files, FILE_LABELS)

if TASK == "10class":
    flows = {label: f for label, f in flows.items() if label.startswith("Malware/")}

plot_aggregated_distributions(
    flows, N, save_path=str(task_dir / "feature_distributions.png")
)
plot_per_position(flows, N, save_path=str(task_dir / "per_position_features.png"))
plot_tsne(flows, N, save_path=str(task_dir / "tsne.png"))
