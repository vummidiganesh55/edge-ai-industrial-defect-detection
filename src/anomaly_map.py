import numpy as np
import torch
import torch.nn.functional as F


def create_anomaly_map(
    layer2,
    layer3,
    patch_distances,
):
  # --------------------------------------------------
    # 1. Get spatial dimensions
    # --------------------------------------------------
    batch_size, _, height, width = layer2.shape

    expected_patches = height * width

    if len(patch_distances) != expected_patches:
        raise ValueError(
            f"Expected {expected_patches} patch distances, "
            f"but received {len(patch_distances)}"
        )

    # --------------------------------------------------
    # 2. Convert distances to 28 x 28 map
    # --------------------------------------------------
    anomaly_map = np.asarray(
        patch_distances,
        dtype=np.float32,
    ).reshape(height, width)

    # --------------------------------------------------
    # 3. Convert to tensor
    # --------------------------------------------------
    anomaly_map = torch.from_numpy(
        anomaly_map
    ).unsqueeze(0).unsqueeze(0)

    # --------------------------------------------------
    # 4. Resize to input resolution
    # --------------------------------------------------
    anomaly_map = F.interpolate(
        anomaly_map,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )

    anomaly_map = anomaly_map.squeeze().numpy()

    return anomaly_map


def normalize_anomaly_map(anomaly_map):
    """
    Normalize anomaly map to [0, 1].
    """

    anomaly_map = np.asarray(
        anomaly_map,
        dtype=np.float32,
    )

    minimum = anomaly_map.min()
    maximum = anomaly_map.max()

    if maximum - minimum < 1e-8:
        return np.zeros_like(anomaly_map)

    normalized = (
        anomaly_map - minimum
    ) / (
        maximum - minimum
    )

    return normalized.astype(np.float32)


def get_anomaly_region(
    anomaly_map,
    threshold=0.5,
):
    """
    Create binary anomaly region.
    """

    normalized_map = normalize_anomaly_map(
        anomaly_map
    )

    binary_map = (
        normalized_map >= threshold
    ).astype(np.uint8)

    return binary_map
