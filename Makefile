# Developer entry points. `make check` is the full pre-PR gate and needs no ROS.

PYTHON ?= python3
CORE   := ros2_ws/src/roboworld_core
export PYTHONPATH := $(CORE)

.PHONY: help install test lint dryrun evaluate dataset symmetry visualize live check ros-build ros-test ros-check run clean

help:
	@echo "make install    - install python dependencies (no ROS, no GPU)"
	@echo "make test       - unit tests (154)"
	@echo "make lint       - ruff"
	@echo "make dryrun     - E2E dry-run, stub + real CPU backends (contract gate)"
	@echo "make evaluate   - detection and pose accuracy vs tolerances"
	@echo "make dataset    - generate the mock RGB-D dataset"
	@echo "make symmetry   - report part symmetry ambiguity"
	@echo "make visualize  - render a PNG panel: color | depth | segmentation | anomaly"
	@echo "make live       - live RGB-D window (WSLg). LIVE_ARGS='--inspect --pose'"
	@echo "make check      - lint + test + dryrun + evaluate  (pre-PR gate)"
	@echo "make ros-build  - colcon build            (requires ROS 2 Humble)"
	@echo "make ros-test   - colcon test             (requires ROS 2 Humble)"
	@echo "make run        - launch the full pipeline with mock frames"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest $(CORE)/test -q

lint:
	$(PYTHON) -m ruff check ros2_ws/src tools

dryrun:
	$(PYTHON) tools/e2e_dryrun.py --cycles 9 --json data/dryrun_stub.json
	$(PYTHON) tools/e2e_dryrun.py --cycles 9 --inspection statistical --pose icp \
		--json data/dryrun_real.json

evaluate:
	$(PYTHON) tools/evaluate.py --part all --frames 25 --train-frames 40 \
		--json data/evaluation.json

dataset:
	$(PYTHON) tools/generate_mock_dataset.py --part all --train 40 --test 20

symmetry:
	$(PYTHON) tools/check_symmetry.py

# PART / DEFECT override the target, e.g. `make visualize PART=end_stopper DEFECT=scratch`
PART   ?= guide_block
DEFECT ?= chip
visualize:
	$(PYTHON) tools/visualize.py --part $(PART) --defect $(DEFECT)

# Needs a display (WSLg on Windows 11) and opencv-python.
# No display? use LIVE_ARGS='--backend mjpeg' and open http://localhost:8080
LIVE_ARGS ?= --inspect
live:
	$(PYTHON) tools/live_view.py --part $(PART) $(LIVE_ARGS)

check: lint test dryrun evaluate
	@echo ""
	@echo "all gates passed"

ros-build:
	cd ros2_ws && colcon build --symlink-install

ros-test:
	cd ros2_ws && colcon test --return-code-on-test-failure && colcon test-result --verbose

ros-check: ros-build ros-test

run:
	@echo "source ros2_ws/install/setup.bash first, then:"
	@echo "  ros2 launch roboworld_bringup pipeline.launch.py"

clean:
	rm -rf ros2_ws/build ros2_ws/install ros2_ws/log
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
	rm -rf .pytest_cache .ruff_cache
