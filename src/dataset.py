from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


class MVTecDataset(Dataset):
    """
    MVTec AD dataset loader.

    Supports:
    - train/good
    - test/good
    - test/defect_type
    """

    VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        root_dir: str,
        category: str = "bottle",
        split: str = "train",
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.transform = transform

        self.category_dir = self.root_dir / category

        if not self.category_dir.exists():
            raise FileNotFoundError(
                f"Category directory not found: {self.category_dir}"
            )

        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        self.samples: List[Tuple[Path, int, str]] = []

        self._load_samples()

        if not self.samples:
            raise RuntimeError(
                f"No images found for category='{category}', split='{split}'"
            )

    def _load_samples(self):
        split_dir = self.category_dir / self.split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}"
            )

        if self.split == "train":
            good_dir = split_dir / "good"

            self._collect_images(
                good_dir,
                label=0,
                defect_type="good",
            )

        else:
            for defect_dir in sorted(split_dir.iterdir()):

                if not defect_dir.is_dir():
                    continue

                defect_type = defect_dir.name

                label = 0 if defect_type == "good" else 1

                self._collect_images(
                    defect_dir,
                    label=label,
                    defect_type=defect_type,
                )

    def _collect_images(
        self,
        directory: Path,
        label: int,
        defect_type: str,
    ):
        if not directory.exists():
            return

        for image_path in sorted(directory.iterdir()):

            if image_path.suffix.lower() not in self.VALID_EXTENSIONS:
                continue

            self.samples.append(
                (
                    image_path,
                    label,
                    defect_type,
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label, defect_type = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "path": str(image_path),
            "defect_type": defect_type,
        }