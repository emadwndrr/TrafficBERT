from pathlib import Path
from pprint import pprint

from pcap.plot import plot_ip4_tcp_udp_per_file, plot_raw_stats
from pcap.stats import compute_stats

DATASET_DIR = "./pcap_datasets"
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

results_dir = Path(RESULTS_DIR)
results_dir.mkdir(parents=True, exist_ok=True)

pcap_files = list(Path(DATASET_DIR).rglob("*.pcap"))

dataset_raw_stats = compute_stats(pcap_files, FILE_LABELS)
plot_raw_stats(dataset_raw_stats, save_dir=str(results_dir))

plot_ip4_tcp_udp_per_file(dataset_raw_stats, save_dir=str(results_dir))

pprint(dataset_raw_stats)
