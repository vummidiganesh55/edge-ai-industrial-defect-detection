from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw

from src.dataset import MVTecDataset
from src.preprocessing import get_test_transform
from src.feature_extractor import ResNet18FeatureExtractor
from src.patchcore_pipeline import build_memory_bank
from src.anomaly_map import (
    create_anomaly_map,
    normalize_anomaly_map,
    get_anomaly_region,
)


DATASET_ROOT = "data/mvtec_anomaly_detection"
OUTPUT_DIR = Path("outputs/localization")

IMAGE_SIZE = 224
ANOMALY_THRESHOLD = 0.5


def load_ground_truth_mask(
    image_path,
    category,
    defect_type,
):
    """
    Load the MVTec ground-truth mask.
    Good images have no defect mask.
    """

    if defect_type == "good":
        return np.zeros(
            (IMAGE_SIZE, IMAGE_SIZE),
            dtype=np.uint8,
        )

    image_path = Path(image_path)

    mask_path = (
        Path(DATASET_ROOT)
        / category
        / "ground_truth"
        / defect_type
        / f"{image_path.stem}_mask.png"
    )

    if not mask_path.exists():
        print(
            f"WARNING: Mask not found: {mask_path}"
        )

        return np.zeros(
            (IMAGE_SIZE, IMAGE_SIZE),
            dtype=np.uint8,
        )

    mask = Image.open(mask_path).convert("L")

    mask = mask.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.NEAREST,
    )

    mask = np.asarray(
        mask,
        dtype=np.uint8,
    )

    return (
        mask > 0
    ).astype(np.uint8)


def create_heatmap(anomaly_map):
    """
    Convert normalized anomaly map into a visual heatmap.
    """

    heatmap_values = np.asarray(
        anomaly_map
    ).clip(0, 1)

    heatmap_image = Image.fromarray(
        (heatmap_values * 255).astype(
            np.uint8
        ),
        mode="L",
    )

    heatmap = ImageOps.colorize(
        heatmap_image,
        black="#00007f",
        mid="#00ffff",
        white="#7f0000",
    )

    return heatmap


def create_mask_image(mask):
    """
    Convert binary mask to RGB image.
    """

    mask_image = Image.fromarray(
        (mask * 255).astype(np.uint8),
        mode="L",
    )

    return ImageOps.colorize(
        mask_image,
        black="black",
        white="white",
    )


def add_title(image, title):
    """
    Add a title above an image.
    """

    canvas = Image.new(
        "RGB",
        (image.width, image.height + 30),
        "white",
    )

    canvas.paste(
        image,
        (0, 30),
    )

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (10, 8),
        title,
        fill="black",
    )

    return canvas


def create_comparison(
    original,
    ground_truth,
    heatmap,
    predicted_mask,
):
    """
    Create a 4-panel localization comparison.
    """

    original_panel = add_title(
        original,
        "Original",
    )

    ground_truth_panel = add_title(
        ground_truth,
        "Ground Truth",
    )

    heatmap_panel = add_title(
        heatmap,
        "PatchCore Heatmap",
    )

    predicted_panel = add_title(
        predicted_mask,
        "Predicted Region",
    )

    panels = [
        original_panel,
        ground_truth_panel,
        heatmap_panel,
        predicted_panel,
    ]

    total_width = sum(
        panel.width
        for panel in panels
    )

    max_height = max(
        panel.height
        for panel in panels
    )

    comparison = Image.new(
        "RGB",
        (
            total_width,
            max_height,
        ),
        "white",
    )

    x_offset = 0

    for panel in panels:

        comparison.paste(
            panel,
            (x_offset, 0),
        )

        x_offset += panel.width

    return comparison


def visualize_localization(
    category="bottle",
):
    device = torch.device("cpu")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("PATCHCORE LOCALIZATION VISUALIZATION")
    print("=" * 60)

    print(f"Device   : {device}")
    print(f"Category : {category}")

    # --------------------------------------------------
    # Build PatchCore memory bank
    # --------------------------------------------------

    patchcore = build_memory_bank(
        category=category,
        batch_size=8,
        sampling_ratio=0.1,
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = MVTecDataset(
        root_dir=DATASET_ROOT,
        category=category,
        split="test",
        transform=get_test_transform(),
    )

    # --------------------------------------------------
    # Feature extractor
    # --------------------------------------------------

    feature_extractor = (
        ResNet18FeatureExtractor()
        .to(device)
    )

    feature_extractor.eval()

    # --------------------------------------------------
    # Representative samples
    # --------------------------------------------------

    selected_indices = [
        0,   # broken_large
        20,  # broken_small
        42,  # contamination
        43,  # contamination
        47,  # contamination
        63,  # good
        69,  # good
    ]

    print(
        f"\nGenerating {len(selected_indices)} "
        "localization visualizations...\n"
    )

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------

    with torch.no_grad():

        for index in selected_indices:

            sample = dataset[index]

            image_tensor = (
                sample["image"]
                .unsqueeze(0)
                .to(device)
            )

            features = feature_extractor(
                image_tensor
            )

            anomaly_score, patch_distances = (
                patchcore.predict(
                    features
                )
            )

            anomaly_map = create_anomaly_map(
                features["layer2"],
                features["layer3"],
                patch_distances,
            )

            normalized_map = (
                normalize_anomaly_map(
                    anomaly_map
                )
            )

            predicted_binary = (
                get_anomaly_region(
                    anomaly_map,
                    threshold=ANOMALY_THRESHOLD,
                )
            )

            # --------------------------------------------------
            # Original image
            # --------------------------------------------------

            original_path = sample["path"]

            original = Image.open(
                original_path
            ).convert("RGB")

            original = original.resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.BILINEAR,
            )

            # --------------------------------------------------
            # Ground truth
            # --------------------------------------------------

            ground_truth_mask = (
                load_ground_truth_mask(
                    image_path=original_path,
                    category=category,
                    defect_type=sample[
                        "defect_type"
                    ],
                )
            )

            ground_truth_image = (
                create_mask_image(
                    ground_truth_mask
                )
            )

            # --------------------------------------------------
            # Heatmap
            # --------------------------------------------------

            heatmap = create_heatmap(
                normalized_map
            )

            # --------------------------------------------------
            # Predicted mask
            # --------------------------------------------------

            predicted_image = (
                create_mask_image(
                    predicted_binary
                )
            )

            # --------------------------------------------------
            # Create comparison
            # --------------------------------------------------

            comparison = create_comparison(
                original=original,
                ground_truth=ground_truth_image,
                heatmap=heatmap,
                predicted_mask=predicted_image,
            )

            # --------------------------------------------------
            # Save
            # --------------------------------------------------

            defect_type = sample[
                "defect_type"
            ]

            output_name = (
                f"{index:03d}_"
                f"{defect_type}_"
                f"score_{anomaly_score:.2f}.png"
            )

            output_path = (
                OUTPUT_DIR / output_name
            )

            comparison.save(
                output_path
            )

            print(
                f"Saved: {output_path}"
            )

    print("\n" + "=" * 60)
    print(
        "LOCALIZATION VISUALIZATION COMPLETE"
    )
    print("=" * 60)

    print(
        f"Output directory: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    visualize_localization(
        category="bottle"
    )
