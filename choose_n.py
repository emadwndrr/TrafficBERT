from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pcap.flow import extract_flows

DATASET_DIR = "./pcap_datasets"
RESULTS_DIR = "results"
N = 20

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


def plot_flow_length_cdf(labeled_flows, N, save_path="flow_length_cdf.png"):
    labels = sorted(labeled_flows.keys())
    benign_labels = [l for l in labels if l.startswith("Benign/")]
    malware_labels = [l for l in labels if l.startswith("Malware/")]

    groups = [(malware_labels, "Malware")]
    if benign_labels:
        groups.insert(0, (benign_labels, "Benign"))

    colors_b = plt.cm.tab10(np.linspace(0, 1, len(benign_labels))) if benign_labels else []
    colors_m = plt.cm.tab10(np.linspace(0, 1, len(malware_labels)))
    group_colors = ([colors_b, colors_m] if benign_labels else [colors_m])

    ncols = len(groups)
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 6), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, (group_labels, group_name), colors in zip(axes, groups, group_colors):
        all_lengths = []
        for label, color in zip(group_labels, colors):
            lengths = sorted(len(flow.packets) for flow in labeled_flows[label].values())
            all_lengths.extend(lengths)
            cdf = np.arange(1, len(lengths) + 1) / len(lengths)
            ax.plot(lengths, cdf, label=label.split("/")[1], color=color, linewidth=1.5)

        ax.axvline(N, color="red", linestyle="--", linewidth=1.5, label=f"N={N}")
        ax.set_xlim(left=0, right=np.percentile(all_lengths, 99))
        ax.set_title(f"Flow length CDF — {group_name}", fontsize=11)
        ax.set_xlabel("Flow length (packets)")
        ax.set_ylabel("Fraction of flows")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {save_path}")


def plot_flow_length_distribution(labeled_flows, N, save_path="flow_length_dist.png"):
    labels = sorted(labeled_flows.keys())
    benign_labels = [l for l in labels if l.startswith("Benign/")]
    malware_labels = [l for l in labels if l.startswith("Malware/")]
    ordered_labels = benign_labels + malware_labels

    lengths = [
        [len(flow.packets) for flow in labeled_flows[label].values()]
        for label in ordered_labels
    ]
    short_labels = [l.replace("Benign/", "B/").replace("Malware/", "M/") for l in ordered_labels]
    colors = ["#4C72B0" if l.startswith("Benign/") else "#DD8452" for l in ordered_labels]

    fig, ax = plt.subplots(figsize=(18, 6))

    parts = ax.violinplot(lengths, positions=range(len(ordered_labels)), showmedians=True, showextrema=False)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")

    ax.axhline(N, color="red", linestyle="--", linewidth=1.5, label=f"N={N}")
    if benign_labels:
        ax.axvline(len(benign_labels) - 0.5, color="gray", linestyle="--", linewidth=0.8)

    ax.set_xticks(range(len(ordered_labels)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_title("Flow length distribution per class", fontsize=12)
    ax.set_ylabel("Flow length (packets, log scale)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {save_path}")


results_dir = Path(RESULTS_DIR)
results_dir.mkdir(parents=True, exist_ok=True)

pcap_files = list(Path(DATASET_DIR).rglob("*.pcap"))
flows = extract_flows(pcap_files, FILE_LABELS)

plot_flow_length_cdf(flows, N, save_path=str(results_dir / "flow_length_cdf.png"))
plot_flow_length_distribution(flows, N, save_path=str(results_dir / "flow_length_dist.png"))
