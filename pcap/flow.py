import socket
from dataclasses import dataclass, field
from typing import List

import dpkt


@dataclass
class Packet:
    timestamp: float
    size: int
    direction: str
    inter_arrival_time: float = 0.0


@dataclass
class Flow:
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    proto: int
    packets: List[Packet] = field(default_factory=list)


def flow_key(src_ip, src_port, dst_ip, dst_port, proto):
    if (src_ip, src_port) <= (dst_ip, dst_port):
        return (src_ip, src_port, dst_ip, dst_port, proto)
    return (dst_ip, dst_port, src_ip, src_port, proto)


def extract_flows(pcap_files, file_labels):
    labeled_flows = {}
    for label in file_labels.values():
        labeled_flows[label] = {}

    for pcap_path in pcap_files:
        flows = labeled_flows[file_labels[pcap_path.name]]

        with open(pcap_path, "rb") as f:
            print(pcap_path)

            pcap = dpkt.pcap.Reader(f)

            assert pcap.datalink() == 1

            for i, (ts, buf) in enumerate(pcap):
                if len(buf) < 14:
                    print(
                        f"  --> packet {i} is only {len(buf)} bytes: {buf.hex()} (SKIPPING)"
                    )
                    continue

                eth_pkt = dpkt.ethernet.Ethernet(buf)

                if eth_pkt.type == dpkt.ethernet.ETH_TYPE_IP:
                    ip_pkt = eth_pkt.data

                    if (ip_pkt.p == dpkt.ip.IP_PROTO_TCP) or (
                        ip_pkt.p == dpkt.ip.IP_PROTO_UDP
                    ):
                        src_ip = socket.inet_ntoa(ip_pkt.src)
                        src_port = ip_pkt.data.sport
                        dst_ip = socket.inet_ntoa(ip_pkt.dst)
                        dst_port = ip_pkt.data.dport
                        proto = ip_pkt.p

                        key = flow_key(src_ip, src_port, dst_ip, dst_port, proto)

                        if key not in flows:
                            flows[key] = Flow(*key)

                        pkt_dir = (
                            "c2s"
                            if (src_ip == key[0] and src_port == key[1])
                            else "s2c"
                        )
                        flow = flows[key]
                        iat = ts - flow.packets[-1].timestamp if flow.packets else 0.0

                        flow.packets.append(
                            Packet(
                                timestamp=ts,
                                size=ip_pkt.len,
                                direction=pkt_dir,
                                inter_arrival_time=iat,
                            )
                        )
    return labeled_flows


