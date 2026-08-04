"""Configuration loading and the no-hardcoded-values rule (claude.md section 4)."""

from __future__ import annotations

import pytest

from roboworld_core.config import Config, ConfigError, deep_merge, load_config, load_yaml
from roboworld_core.paths import resolve_path


def test_ros_parameter_envelope_is_unwrapped(tmp_path):
    """The same file must read identically as YAML and as a ROS param file."""
    path = tmp_path / "sample.yaml"
    path.write_text(
        "/**:\n  ros__parameters:\n    inspection:\n      threshold: 0.42\n",
        encoding="utf-8",
    )
    assert load_yaml(path) == {"inspection": {"threshold": 0.42}}


def test_missing_key_raises_instead_of_returning_none():
    """A silently missing threshold is worse than a loud startup failure."""
    cfg = Config({"a": {"b": 1}})
    assert cfg.get("a.b") == 1
    with pytest.raises(ConfigError, match="missing config key"):
        cfg.get("a.missing")


def test_missing_key_with_default_returns_default():
    assert Config({}).get("nope", 7) == 7


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"x": 1, "y": 2}}
    override = {"a": {"y": 3, "z": 4}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 3, "z": 4}}
    assert base == {"a": {"x": 1, "y": 2}}


def test_section_rejects_scalars():
    with pytest.raises(ConfigError, match="not a section"):
        Config({"a": 1}).section("a")


def test_merged_with_applies_overrides():
    cfg = Config({"inspection": {"threshold": 0.5, "backend": "statistical"}})
    merged = cfg.merged_with({"inspection": {"threshold": 0.9}})
    assert merged.get("inspection.threshold") == 0.9
    assert merged.get("inspection.backend") == "statistical"
    assert cfg.get("inspection.threshold") == 0.5, "original must be untouched"


# --- the shipped configuration ------------------------------------------
def test_shipped_config_loads(cfg):
    assert cfg.get("pipeline.version")
    assert cfg.get("parts")


@pytest.mark.parametrize(
    "key",
    [
        "inspection.threshold",
        "inspection.backend",
        "pose.backend",
        "pose.min_fitness",
        "pose.output_frame_id",
        "camera.width",
        "camera.optical_frame_id",
        "pipeline.result_topic",
        "pipeline.skip_pose_when_ng",
        "pipeline.tact_budget_ms.total_limit",
    ],
)
def test_operational_values_are_externalized(cfg, key):
    """Every value the line operator may need to change must live in YAML."""
    assert cfg.get(key) is not None


def test_thresholds_are_normalized(cfg):
    threshold = float(cfg.get("inspection.threshold"))
    assert 0.0 <= threshold <= 1.0


def test_every_part_declares_a_reachable_mesh(cfg):
    for part_id, entry in cfg.get("parts").items():
        mesh_path = resolve_path(entry["mesh"])
        assert mesh_path.exists(), f"{part_id}: mesh not found at {mesh_path}"
        assert entry.get("mesh_units", "m") in ("m", "mm", "cm")


def test_default_part_exists(cfg):
    assert cfg.get("default_part_id") in cfg.get("parts")


def test_output_qos_is_declared(cfg):
    """The output QoS is part of the ICD; a missing profile breaks subscribers."""
    qos = cfg.section("pipeline.output_qos")
    assert qos.get("reliability") in ("reliable", "best_effort")
    assert qos.get("durability") in ("volatile", "transient_local")
    assert int(qos.get("depth")) > 0


def test_config_dir_discovery_finds_the_shipped_files():
    assert load_config().get("pipeline.result_topic") == "/roboworld/part_result"
