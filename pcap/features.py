import json
from pathlib import Path

import numpy as np


def flow_to_features(flow, N):
    features = np.zeros((N, 5), dtype=np.float32)
    mask = np.zeros(N, dtype=np.int64)

    for i, pkt in enumerate(flow.packets[:N]):
        features[i, 0] = pkt.size
        features[i, 1] = pkt.inter_arrival_time
        features[i, 2] = 0.0 if pkt.direction == "c2s" else 1.0
        features[i, 3] = pkt.size if pkt.direction == "c2s" else 0.0
        features[i, 4] = pkt.size if pkt.direction == "s2c" else 0.0
        mask[i] = 1

    return features, mask


def create_dataset(labeled_flows, N, output_dir, test_ratio=0.2, seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_names = sorted(labeled_flows.keys())
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}

    all_features, all_masks, all_labels = [], [], []

    for label, flows in labeled_flows.items():
        idx = label_to_idx[label]
        for flow in flows.values():
            features, mask = flow_to_features(flow, N)
            all_features.append(features)
            all_masks.append(mask)
            all_labels.append(idx)

    all_features = np.stack(all_features).astype(np.float32)
    all_masks = np.stack(all_masks).astype(np.int64)
    all_labels = np.array(all_labels, dtype=np.int64)

    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []

    for label_idx in range(len(label_names)):
        class_idx = np.where(all_labels == label_idx)[0]
        rng.shuffle(class_idx)
        n_test = max(1, int(len(class_idx) * test_ratio))
        test_idx.extend(class_idx[:n_test])
        train_idx.extend(class_idx[n_test:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    train_features = all_features[train_idx]
    test_features = all_features[test_idx]
    train_masks = all_masks[train_idx]
    test_masks = all_masks[test_idx]
    train_labels = all_labels[train_idx]
    test_labels = all_labels[test_idx]

    train_mask_bool = train_masks.astype(bool)

    train_sizes = train_features[:, :, 0][train_mask_bool]
    mean_size = train_sizes.mean()
    std_size = train_sizes.std() + 1e-8

    train_log_iats = np.log1p(train_features[:, :, 1][train_mask_bool])
    mean_log_iat = train_log_iats.mean()
    std_log_iat = train_log_iats.std() + 1e-8

    train_c2s = train_features[:, :, 3][train_mask_bool]
    mean_c2s = train_c2s.mean()
    std_c2s = train_c2s.std() + 1e-8

    train_s2c = train_features[:, :, 4][train_mask_bool]
    mean_s2c = train_s2c.mean()
    std_s2c = train_s2c.std() + 1e-8

    def normalize(features, masks):
        out = features.copy()

        out[:, :, 0] = (out[:, :, 0] - mean_size) / std_size
        out[:, :, 1] = (np.log1p(out[:, :, 1]) - mean_log_iat) / std_log_iat

        out[:, :, 3] = (out[:, :, 3] - mean_c2s) / std_c2s
        out[:, :, 4] = (out[:, :, 4] - mean_s2c) / std_s2c

        out[~masks.astype(bool)] = 0.0

        return out

    train_features = normalize(train_features, train_masks)
    test_features = normalize(test_features, test_masks)

    np.savez_compressed(
        output_dir / "train.npz",
        features=train_features,
        masks=train_masks,
        labels=train_labels,
    )
    np.savez_compressed(
        output_dir / "test.npz",
        features=test_features,
        masks=test_masks,
        labels=test_labels,
    )

    np.savez(
        output_dir / "norm_stats.npz",
        mean_size=mean_size,
        std_size=std_size,
        mean_log_iat=mean_log_iat,
        std_log_iat=std_log_iat,
        mean_c2s=mean_c2s,
        std_c2s=std_c2s,
        mean_s2c=mean_s2c,
        std_s2c=std_s2c,
    )
    with open(output_dir / "label_to_idx.json", "w") as f:
        json.dump(label_to_idx, f, indent=2)

    print(f"\nN = {N}")
    print(f"Total : {len(all_labels):>8,}")
    print(f"Train : {len(train_labels):>8,}")
    print(f"Test  : {len(test_labels):>8,}\n")

    print(f"{'Label':<30} {'Train':>8} {'Test':>8} {'Pad %':>8}")
    print("-" * 58)
    for label, idx in sorted(label_to_idx.items(), key=lambda x: x[1]):
        n_train = (train_labels == idx).sum()
        n_test = (test_labels == idx).sum()
        class_mask = train_masks[train_labels == idx]
        pad_pct = 100.0 * (1 - class_mask.mean())
        print(f"{label:<30} {n_train:>8,} {n_test:>8,} {pad_pct:>7.1f}%")

    print("\nNormalization stats (from training set):")
    print(f"  packet size : mean={mean_size:.2f}   std={std_size:.2f}")
    print(f"  log1p(IAT)  : mean={mean_log_iat:.4f}  std={std_log_iat:.4f}")

    return label_to_idx
