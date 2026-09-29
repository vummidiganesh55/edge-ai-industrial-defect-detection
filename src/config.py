from pathlib import Path

import yaml


CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "configs"
    / "production.yaml"
)


def load_production_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Production config not found: {CONFIG_PATH}"
        )

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Production configuration must be a dictionary."
        )

    return config


PRODUCTION_CONFIG = load_production_config()

CATEGORY = PRODUCTION_CONFIG["category"]
DEVICE = PRODUCTION_CONFIG["device"]
PRODUCTION_CONFIG = {
    "threshold": 17.364517
}

DEFAULT_THRESHOLD = float(
    PRODUCTION_CONFIG["threshold"]
)