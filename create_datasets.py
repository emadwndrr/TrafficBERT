from pathlib import Path

from pcap.features import create_dataset
from pcap.flow import extract_flows

DATASET_DIR = "./pcap_datasets"
DATASETS_DIR = "generated_datasets"
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

pcap_files = list(Path(DATASET_DIR).rglob("*.pcap"))
flows = extract_flows(pcap_files, FILE_LABELS)

for task, task_flows in [
    ("20class", flows),
    ("10class", {l: f for l, f in flows.items() if l.startswith("Malware/")}),
]:
    task_dir = Path(DATASETS_DIR) / task
    task_dir.mkdir(parents=True, exist_ok=True)
    create_dataset(labeled_flows=task_flows, N=N, output_dir=str(task_dir))
