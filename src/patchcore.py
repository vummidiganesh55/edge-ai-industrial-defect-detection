import numpy as np
import torch
import faiss


class PatchCore:
    """
    Simplified PatchCore implementation.

    Pipeline:
        Feature maps
        -> patch embeddings
        -> memory bank
        -> nearest-neighbor search
        -> anomaly score
    """

    def __init__(
        self,
        sampling_ratio=0.1,
    ):
        self.sampling_ratio = sampling_ratio

        self.memory_bank = None
        self.index = None

    def _feature_map_to_patches(self, feature_map):
        """
        Convert feature map:

        [B, C, H, W]

        into:

        [B*H*W, C]
        """

        batch_size, channels, height, width = feature_map.shape

        features = feature_map.permute(
            0, 2, 3, 1
        )

        features = features.reshape(
            batch_size * height * width,
            channels,
        )

        return features

    def _combine_features(
        self,
        layer2,
        layer3,
    ):
        """
        Convert layer2 and layer3 feature maps
        into patch embeddings.

        Layer2:
            [B, 128, 28, 28]

        Layer3:
            [B, 256, 14, 14]

        Layer3 is resized to Layer2 spatial resolution.
        """

        layer3 = torch.nn.functional.interpolate(
            layer3,
            size=layer2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        layer2_patches = self._feature_map_to_patches(
            layer2
        )

        layer3_patches = self._feature_map_to_patches(
            layer3
        )

        combined = torch.cat(
            [
                layer2_patches,
                layer3_patches,
            ],
            dim=1,
        )

        return combined

    def _coreset_sampling(self, features):
        """
        Randomized coreset approximation.

        Reduces memory-bank size while preserving
        representative normal features.
        """

        features = np.asarray(
            features,
            dtype=np.float32,
        )

        num_samples = len(features)

        target_samples = max(
            1,
            int(num_samples * self.sampling_ratio),
        )

        if target_samples >= num_samples:
            return features

        rng = np.random.default_rng(42)
        indices = rng.choice(
            num_samples,
            target_samples,
            replace=False,
        )

        return features[indices]

    def fit(self, feature_batches):
        """
        Build the PatchCore memory bank.

        feature_batches:
            list of dictionaries containing:
                layer2
                layer3
        """

        all_features = []

        for features in feature_batches:

            layer2 = features["layer2"]
            layer3 = features["layer3"]

            combined = self._combine_features(
                layer2,
                layer3,
            )

            combined = combined.cpu().numpy()

            all_features.append(combined)

        all_features = np.concatenate(
            all_features,
            axis=0,
        )

        print(
            f"Total patch features: "
            f"{len(all_features)}"
        )

        memory_bank = self._coreset_sampling(
            all_features
        )

        self.memory_bank = memory_bank

        print(
            f"Memory bank size: "
            f"{len(self.memory_bank)}"
        )

        self.index = faiss.IndexFlatL2(
            self.memory_bank.shape[1]
        )

        self.index.add(
            self.memory_bank
        )

    def predict(self, features):
        """
        Calculate anomaly score for one image.
        """

        layer2 = features["layer2"]
        layer3 = features["layer3"]

        combined = self._combine_features(
            layer2,
            layer3,
        )

        combined_np = combined.cpu().numpy()

        distances, _ = self.index.search(
            combined_np,
            1,
        )

        patch_distances = distances[:, 0]

        anomaly_score = float(
            np.max(patch_distances)
        )

        return anomaly_score, patch_distances