import dpkt


def compute_stats(pcap_files, file_labels):
    dataset_raw_stats = {}
    for pcap_path in pcap_files:
        label = file_labels[pcap_path.name]

        with open(pcap_path, "rb") as f:
            print(pcap_path)

            pcap = dpkt.pcap.Reader(f)

            assert pcap.datalink() == 1

            raw_stats = {
                "ip4_tcp_count": 0,
                "ip4_udp_count": 0,
                "ip4_other_count": 0,
                "ip6_tcp_count": 0,
                "ip6_udp_count": 0,
                "ip6_other_count": 0,
            }

            for i, (_, buf) in enumerate(pcap):
                if len(buf) < 14:
                    print(
                        f"  --> packet {i} is only {len(buf)} bytes: {buf.hex()} (SKIPPING)"
                    )
                    continue

                eth_pkt = dpkt.ethernet.Ethernet(buf)

                if eth_pkt.type == dpkt.ethernet.ETH_TYPE_IP:
                    ip_pkt = eth_pkt.data
                    if ip_pkt.p == dpkt.ip.IP_PROTO_TCP:
                        raw_stats["ip4_tcp_count"] += 1
                    elif ip_pkt.p == dpkt.ip.IP_PROTO_UDP:
                        raw_stats["ip4_udp_count"] += 1
                    else:
                        raw_stats["ip4_other_count"] += 1

                elif eth_pkt.type == dpkt.ethernet.ETH_TYPE_IP6:
                    ip_pkt = eth_pkt.data
                    if ip_pkt.p == dpkt.ip.IP_PROTO_TCP:
                        raw_stats["ip6_tcp_count"] += 1
                    elif ip_pkt.p == dpkt.ip.IP_PROTO_UDP:
                        raw_stats["ip6_udp_count"] += 1
                    else:
                        raw_stats["ip6_other_count"] += 1

            if label not in dataset_raw_stats:
                dataset_raw_stats[label] = raw_stats
            else:
                for key, val in raw_stats.items():
                    dataset_raw_stats[label][key] += val

    return dataset_raw_stats
