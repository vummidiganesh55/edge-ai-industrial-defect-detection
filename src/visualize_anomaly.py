from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from src.dataset import MVTecDataset
from src.preprocessing import get_test_transform
from src.feature_extractor import ResNet18FeatureExtractor
from src.patchcore_pipeline import build_memory_bank
from src.anomaly_map import (
    create_anomaly_map,
    normalize_anomaly_map,
)


DATASET_ROOT = "data/mvtec_anomaly_detection"
OUTPUT_DIR = Path("outputs/anomaly_maps")


def visualize_samples(category="bottle"):
    device = torch.device("cpu")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("PATCHCORE ANOMALY MAP GENERATION")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Build PatchCore memory bank
    # --------------------------------------------------
    patchcore = build_memory_bank(
        category=category,
        batch_size=8,
        sampling_ratio=0.1,
    )

    # --------------------------------------------------
    # 2. Load test dataset
    # --------------------------------------------------
    dataset = MVTecDataset(
        root_dir=DATASET_ROOT,
        category=category,
        split="test",
        transform=get_test_transform(),
    )

    # --------------------------------------------------
    # 3. Feature extractor
    # --------------------------------------------------
    feature_extractor = (
        ResNet18FeatureExtractor()
        .to(device)
    )

    feature_extractor.eval()

    # --------------------------------------------------
    # 4. Representative test samples
    # --------------------------------------------------
    selected_indices = [
        0,    # broken_large
        20,   # broken_small
        42,   # contamination
        55,   # contamination
        63,   # good
        69,   # good
    ]

    # --------------------------------------------------
    # 5. Generate anomaly maps
    # --------------------------------------------------
    with torch.no_grad():

        for index in selected_indices:

            sample = dataset[index]

            image_tensor = (
                sample["image"]
                .unsqueeze(0)
                .to(device)
            )

            # Feature extraction
            features = feature_extractor(
                image_tensor
            )

            # PatchCore prediction
            anomaly_score, patch_distances = (
                patchcore.predict(features)
            )

            # Create anomaly map
            anomaly_map = create_anomaly_map(
                features["layer2"],
                features["layer3"],
                patch_distances,
            )

            # Normalize anomaly map
            normalized_map = (
                normalize_anomaly_map(
                    anomaly_map
                )
            )

            # --------------------------------------------------
            # 6. Load original image
            # --------------------------------------------------
            original_path = sample["path"]

            try:
                original = Image.open(original_path).convert("RGB")
            except (OSError, ValueError):
                print(f"Could not read: {original_path}")
                continue

            original = original.resize((224, 224), Image.Resampling.BILINEAR)

            # --------------------------------------------------
            # 7. Create heatmap
            # --------------------------------------------------
            heatmap_values = np.asarray(normalized_map).clip(0, 1)
            heatmap_image = Image.fromarray(
                (heatmap_values * 255).astype(np.uint8),
                mode="L",
            )
            heatmap = ImageOps.colorize(
                heatmap_image,
                black="#00007f",
                mid="#00ffff",
                white="#7f0000",
            ).resize((224, 224), Image.Resampling.BILINEAR)

            # --------------------------------------------------
            # 8. Overlay heatmap
            # --------------------------------------------------
            overlay = Image.blend(original, heatmap, alpha=0.4)

            # --------------------------------------------------
            # 9. Save result
            # --------------------------------------------------
            defect_type = sample["defect_type"]

            output_name = (
                f"{index:03d}_"
                f"{defect_type}_"
                f"score_{anomaly_score:.2f}.png"
            )

            output_path = (
                OUTPUT_DIR / output_name
            )

            overlay.save(output_path)

            print(
                f"Saved: {output_path}"
            )

    print("\n" + "=" * 60)
    print("ANOMALY MAP GENERATION COMPLETE")
    print("=" * 60)

    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    visualize_samples(
        category="bottle"
    )
