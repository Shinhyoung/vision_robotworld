"""Shared fixtures.

Mock frames are rendered from the real CAD, which costs ~150 ms each, so the
station and a small frame set are session-scoped.
"""

from __future__ import annotations

import pytest

from roboworld_core.config import load_config
from roboworld_core.mock_data import MockStation, parts_from_config
from roboworld_core.types import CameraIntrinsics

PART_ID = "guide_block"


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def intrinsics(cfg):
    return CameraIntrinsics.from_config(cfg.section("camera"))


@pytest.fixture(scope="session")
def station(cfg, intrinsics):
    return MockStation(parts_from_config(cfg), intrinsics)


@pytest.fixture(scope="session")
def clean_station(cfg, intrinsics):
    """Noise-free station.

    Injecting a defect consumes RNG draws, so the sensor noise added afterwards
    differs between a clean and a damaged render of the same pose. Tests that
    diff the two images need the noise switched off, or they measure noise.
    """
    return MockStation(
        parts_from_config(cfg),
        intrinsics,
        depth_noise_m=0.0,
        color_noise=0.0,
        dropout_ratio=0.0,
    )


@pytest.fixture(scope="session")
def good_frame(station):
    return station.sample_frame(PART_ID, defect=None, seed=11, sequence=1)


@pytest.fixture(scope="session")
def defective_frame(station):
    return station.sample_frame(PART_ID, defect="chip", seed=12, sequence=2)
