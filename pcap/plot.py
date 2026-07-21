from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def plot_ip4_tcp_udp_per_file(dataset_raw_stats, save_dir="results"):
    names = list(dataset_raw_stats.keys())
    counts = [
        s["ip4_tcp_count"] + s["ip4_udp_count"] for s in dataset_raw_stats.values()
    ]

    benign = sorted(
        [(n, c) for n, c in zip(names, counts) if n.startswith("Benign/")],
        key=lambda x: x[1],
        reverse=True,
    )
    malware = sorted(
        [(n, c) for n, c in zip(names, counts) if n.startswith("Malware/")],
        key=lambda x: x[1],
        reverse=True,
    )

    paired = benign + malware
    names, counts = zip(*paired)
    colors = ["#4C72B0" if n.startswith("Benign/") else "#DD8452" for n in names]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.45)))

    bars = ax.barh(names, counts, color=colors)

    if benign and malware:
        ax.axhline(len(benign) - 0.5, color="gray", linestyle="--", linewidth=0.9)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{width:,}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
        )

    ax.invert_yaxis()
    ax.set_xlabel("Packet Count (IPv4 TCP + UDP)", fontsize=12)
    ax.set_ylabel("PCAP File", fontsize=12)
    ax.set_title("IPv4 TCP+UDP Packets per PCAP File", fontsize=13)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_xlim(right=max(counts) * 1.12)

    legend_handles = [
        mpatches.Patch(color="#4C72B0", label="Benign"),
        mpatches.Patch(color="#DD8452", label="Malware"),
    ]
    ax.legend(handles=legend_handles, fontsize=10)

    plt.tight_layout()
    save_path = Path(save_dir) / "pcap_ip4_tcp_udp_per_file.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved to {save_path}")


def plot_raw_stats(dataset_raw_stats, save_dir="results"):
    stats = dataset_raw_stats.values()

    totals = {
        "ip4_tcp": sum(s["ip4_tcp_count"] for s in stats),
        "ip6_tcp": sum(s["ip6_tcp_count"] for s in stats),
        "ip4_udp": sum(s["ip4_udp_count"] for s in stats),
        "ip6_udp": sum(s["ip6_udp_count"] for s in stats),
        "ip4_other": sum(s["ip4_other_count"] for s in stats),
        "ip6_other": sum(s["ip6_other_count"] for s in stats),
    }

    groups = ["TCP", "UDP", "Other"]
    ip4_counts = [totals["ip4_tcp"], totals["ip4_udp"], totals["ip4_other"]]
    ip6_counts = [totals["ip6_tcp"], totals["ip6_udp"], totals["ip6_other"]]

    x = np.arange(len(groups))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_ip4 = ax.bar(
        x - bar_width / 2, ip4_counts, bar_width, label="IPv4", color="#4C72B0"
    )
    bars_ip6 = ax.bar(
        x + bar_width / 2, ip6_counts, bar_width, label="IPv6", color="#DD8452"
    )

    for bar in (*bars_ip4, *bars_ip6):
        height = bar.get_height()
        ax.annotate(
            f"{height:,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Protocol", fontsize=12)
    ax.set_ylabel("Packet Count", fontsize=12)
    ax.set_title("Packet Counts by Protocol and IP Version", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.legend(title="IP Version", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = Path(save_dir) / "pcap_stats.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved to {save_path}")
