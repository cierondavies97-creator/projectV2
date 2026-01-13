from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep YAML dependency optional but explicit.
try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    yaml = None
    _yaml_import_error = e


DEFAULT_FEATURES_AUTO_PATH = Path("conf") / "features_auto.yaml"


@dataclass(frozen=True)
class FeaturesAuto:
    raw: dict[str, Any]


def load_features_auto(path: Path | str = DEFAULT_FEATURES_AUTO_PATH) -> FeaturesAuto:
    p = Path(path)
    if not p.exists():
        return FeaturesAuto(raw={})

    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load conf/features_auto.yaml. Install with: pip install pyyaml"
        ) from _yaml_import_error  # type: ignore[name-defined]

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return FeaturesAuto(raw={})
    return FeaturesAuto(raw=data)


def get_threshold_float(
    *,
    features_auto: FeaturesAuto,
    paradigm_id: str,
    family: str,
    feature: str,
    threshold_key: str,
    default: float,
) -> float:
    """
    Safe lookup for a float threshold value under:

    paradigms.<paradigm_id>.tables."data/features".families.<family>.<feature>.thresholds.<threshold_key>
    """
    node: Any = features_auto.raw
    try:
        node = node["paradigms"][paradigm_id]["tables"]["data/features"]["families"]
        node = node[family][feature]["thresholds"][threshold_key]
    except Exception:
        return float(default)

    try:
        return float(node)
    except Exception:
        return float(default)
