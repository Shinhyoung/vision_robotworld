"""CAD-model registration backend (ICP).

FoundationPose is the production estimator (claude.md section 1) but it needs
Isaac ROS, CUDA and a non-commercial licence check. This backend gives the team
a real, measurable 6D pose on CPU from day one so the ICD, the TF wiring and
the robot-department integration can be validated before any of that lands.

Pipeline: segment the part -> PCA + yaw-seed initialisation -> point-to-point
ICP against a CAD surface sample -> fitness/RMSE scoring.

Correspondences run **scene -> model**: the depth image sees one face of the
block, so every scene point has a model counterpart but most model points have
none. Searching in the other direction would drag the estimate towards the
unobserved back faces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import Pose, make_transform, orthonormalize, transform_points
from ..mesh_io import Mesh
from ..segmentation import Segmentation, segment_part
from ..types import Frame
from .base import PoseBackend, PoseSettings


@dataclass
class IcpParams:
    """Registration tuning (see the ``pose.icp`` config section)."""

    model_points: int = 2000
    scene_max_points: int = 600
    voxel_size_m: float = 0.003
    max_iterations: int = 40
    tolerance_m: float = 1e-5
    max_correspondence_start_m: float = 0.030
    max_correspondence_end_m: float = 0.006
    #: Model axes that may end up facing the camera. A 200 mm block on a belt
    #: rests on one of its four long faces; it does not stand on end, so +/-x
    #: (the long axis) is excluded.
    resting_faces: tuple[str, ...] = ("+y", "-y", "+z", "-z")
    #: Also try the long axis reversed (PCA gives the axis, not its direction).
    try_long_axis_flip: bool = True
    #: Percentile of the point cloud treated as "the surface facing the camera"
    #: when anchoring the initial depth. A plain max would latch onto noise.
    surface_percentile: float = 98.0
    #: Coarse pass: every yaw seed is scored with this many iterations on
    #: decimated clouds, then only the best seed gets the full refinement.
    #: Registering all seeds at full resolution costs ~8x for no extra accuracy.
    coarse_iterations: int = 12
    coarse_model_points: int = 900
    coarse_scene_points: int = 300


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Average points falling in the same voxel. Removes depth-image density bias."""
    pts = np.asarray(points, dtype=np.float64)
    if voxel_size <= 0.0 or len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(sums, inverse, pts)
    return sums / counts[:, None]


def _nearest_squared(query: np.ndarray, reference: np.ndarray,
                     reference_sq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force nearest neighbour via the ``|a-b|^2 = |a|^2 + |b|^2 - 2ab`` identity.

    The matrix product is BLAS-bound, which beats a pure-Python KD-tree at these
    cloud sizes (~1e3 query x ~3e3 reference) and adds no dependency.
    """
    cross = query @ reference.T
    distances_sq = (query * query).sum(axis=1)[:, None] + reference_sq[None, :] - 2.0 * cross
    indices = np.argmin(distances_sq, axis=1)
    best = distances_sq[np.arange(len(query)), indices]
    return indices, np.maximum(best, 0.0)


def _kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares rigid transform mapping ``source`` onto ``target``."""
    src_center = source.mean(axis=0)
    dst_center = target.mean(axis=0)
    covariance = (source - src_center).T @ (target - dst_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:  # reflection -> flip the smallest axis
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    return make_transform(rotation, dst_center - rotation @ src_center)


class IcpPoseBackend(PoseBackend):
    """Point-to-point ICP against a CAD surface sample."""

    name = "icp"

    def __init__(
        self,
        settings: PoseSettings,
        mesh: Mesh,
        params: IcpParams | None = None,
        segmentation_kwargs: dict | None = None,
        use_open3d: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__(settings)
        self.mesh = mesh
        self.params = params or IcpParams()
        self.segmentation_kwargs = dict(segmentation_kwargs or {})
        self.use_open3d = bool(use_open3d)
        self.seed = int(seed)

        self._model_points = mesh.sample_surface(self.params.model_points, seed=seed)
        self._model_sq = (self._model_points * self._model_points).sum(axis=1)
        coarse_count = min(self.params.coarse_model_points, len(self._model_points))
        self._coarse_model = self._model_points[
            np.random.default_rng(seed).choice(
                len(self._model_points), size=coarse_count, replace=False
            )
        ]
        self._coarse_model_sq = (self._coarse_model * self._coarse_model).sum(axis=1)
        self._model_extents = mesh.extents

    # -- initialisation --------------------------------------------------
    _AXIS_VECTORS = {
        "+x": np.array([1.0, 0.0, 0.0]), "-x": np.array([-1.0, 0.0, 0.0]),
        "+y": np.array([0.0, 1.0, 0.0]), "-y": np.array([0.0, -1.0, 0.0]),
        "+z": np.array([0.0, 0.0, 1.0]), "-z": np.array([0.0, 0.0, -1.0]),
    }

    def _initial_transforms(self, segmentation: Segmentation) -> list[np.ndarray]:
        """Candidate ``T_camera_model`` guesses, one per resting hypothesis.

        The conveyor plane fixes two rotational degrees of freedom. What is left
        is discrete, not continuous: *which face the block is lying on* and
        *which way round its long axis points*. Sweeping arbitrary yaw angles
        wastes seeds on poses no block on a belt can adopt, and -- worse -- it
        assumes every face sits half an extent below the surface, which is false
        for a stepped part like the End Stopper.

        Each hypothesis is placed by anchoring the model's own camera-facing
        surface onto the observed surface, so a step in the part costs nothing.
        """
        points = segmentation.points
        centroid = points.mean(axis=0)

        # "Up" = plane normal, pointing towards the camera.
        if segmentation.plane is not None:
            up = np.asarray(segmentation.plane.normal, dtype=np.float64)
        else:
            up = np.array([0.0, 0.0, -1.0])
        up = up / np.linalg.norm(up)
        if up[2] > 0.0:  # ensure it points back towards the camera at the origin
            up = -up

        # Principal in-plane direction of the visible surface = the block's
        # long axis, which the model carries along its own +x.
        centered = points - centroid
        in_plane = centered - np.outer(centered @ up, up)
        if len(in_plane) >= 3:
            _, _, vt = np.linalg.svd(in_plane, full_matrices=False)
            long_axis = vt[0]
        else:
            long_axis = np.array([1.0, 0.0, 0.0])
        long_axis = long_axis - (long_axis @ up) * up
        norm = np.linalg.norm(long_axis)
        long_axis = long_axis / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])

        # Depth of the observed surface along `up` (higher = closer to camera).
        observed_top = float(
            np.percentile(points @ up, self.params.surface_percentile)
        )
        centroid_in_plane = centroid - up * float(centroid @ up)

        long_directions = [long_axis]
        if self.params.try_long_axis_flip:
            long_directions.append(-long_axis)

        transforms: list[np.ndarray] = []
        for face in self.params.resting_faces:
            model_up_axis = self._AXIS_VECTORS[face]
            for direction in long_directions:
                # Build R (model -> camera) from two matched orthonormal triads:
                # the model's long axis maps to `direction`, and the model face
                # named by `face` maps to `up` (towards the camera).
                model_basis = np.stack(
                    [
                        self._AXIS_VECTORS["+x"],
                        model_up_axis,
                        np.cross(self._AXIS_VECTORS["+x"], model_up_axis),
                    ],
                    axis=1,
                )
                camera_basis = np.stack(
                    [direction, up, np.cross(direction, up)], axis=1
                )
                rotation = orthonormalize(camera_basis @ model_basis.T)

                # Anchor depth on the model's own camera-facing surface rather
                # than on half an extent -- correct for stepped parts too.
                model_up_coords = (self._model_points @ rotation.T) @ up
                model_top = float(
                    np.percentile(model_up_coords, self.params.surface_percentile)
                )
                center = centroid_in_plane + up * (observed_top - model_top)
                transforms.append(make_transform(rotation, center))
        return transforms

    # -- registration ----------------------------------------------------
    def _icp(
        self,
        scene: np.ndarray,
        initial: np.ndarray,
        model: np.ndarray | None = None,
        model_sq: np.ndarray | None = None,
        iterations: int | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """Run ICP from ``initial``; returns ``(T_camera_model, fitness, rmse)``."""
        params = self.params
        model = self._model_points if model is None else model
        model_sq = self._model_sq if model_sq is None else model_sq
        iterations = params.max_iterations if iterations is None else iterations

        transform = np.asarray(initial, dtype=np.float64).copy()
        previous_rmse = np.inf
        fitness, rmse = 0.0, np.inf

        for iteration in range(iterations):
            # Linear schedule from coarse to fine correspondence gating.
            ratio = iteration / max(1, iterations - 1)
            max_distance = (
                params.max_correspondence_start_m
                + ratio * (params.max_correspondence_end_m - params.max_correspondence_start_m)
            )

            # Express the scene in model frame and match against the fixed model.
            inverse = np.linalg.inv(transform)
            scene_in_model = transform_points(inverse, scene)
            indices, distances_sq = _nearest_squared(scene_in_model, model, model_sq)
            inliers = distances_sq <= max_distance ** 2
            if inliers.sum() < 10:
                break

            correction = _kabsch(scene_in_model[inliers], model[indices[inliers]])
            # correction maps scene(model frame) -> model, so it refines the
            # inverse transform; fold it back into T_camera_model.
            transform = transform @ np.linalg.inv(correction)

            rmse = float(np.sqrt(distances_sq[inliers].mean()))
            fitness = float(inliers.mean())
            if abs(previous_rmse - rmse) < params.tolerance_m:
                break
            previous_rmse = rmse

        # Score the transform we are actually returning, not the one from the
        # iteration before the last correction.
        fitness, rmse = self.score(scene, transform)
        return transform, fitness, rmse

    def score(self, scene: np.ndarray, transform: np.ndarray) -> tuple[float, float]:
        """Fitness (inlier ratio) and inlier RMSE of ``transform`` on ``scene``."""
        scene_in_model = transform_points(np.linalg.inv(transform), scene)
        _, distances_sq = _nearest_squared(scene_in_model, self._model_points, self._model_sq)
        inliers = distances_sq <= self.params.max_correspondence_end_m ** 2
        if not inliers.any():
            return 0.0, float("inf")
        return float(inliers.mean()), float(np.sqrt(distances_sq[inliers].mean()))

    def _icp_open3d(self, scene: np.ndarray, initial: np.ndarray):
        """Open3D registration when available (KD-tree based, faster).

        Source is the *scene* and target the model, matching the numpy path:
        Open3D draws correspondences source -> target, and the scene is the
        partial cloud that must find counterparts, not the other way round.
        """
        import open3d as o3d

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(scene)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(self._model_points)
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            self.params.max_correspondence_end_m,
            np.linalg.inv(np.asarray(initial, dtype=np.float64)),  # T_model_camera
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.params.max_iterations
            ),
        )
        transform = np.linalg.inv(np.asarray(result.transformation))  # back to T_camera_model
        return transform, float(result.fitness), float(result.inlier_rmse)

    # -- backend API -----------------------------------------------------
    def estimate(self, frame: Frame) -> tuple[Pose, float, float, str]:
        segmentation = segment_part(frame, **self.segmentation_kwargs)
        if not segmentation.ok:
            # Forward segmentation's own reason: "found an object but it is not
            # this part" and "found nothing at all" call for different actions
            # on the line, and the robot department reads this in
            # PartResult.message.
            return (
                Pose.identity(frame.intrinsics.frame_id),
                0.0,
                float("inf"),
                segmentation.reason
                or "segmentation found no part above the conveyor plane",
            )

        scene = voxel_downsample(segmentation.points, self.params.voxel_size_m)
        if len(scene) > self.params.scene_max_points:
            rng = np.random.default_rng(self.seed)
            scene = scene[
                rng.choice(len(scene), size=self.params.scene_max_points, replace=False)
            ]
        if len(scene) < 20:
            return (
                Pose.identity(frame.intrinsics.frame_id),
                0.0,
                float("inf"),
                f"only {len(scene)} scene points after downsampling",
            )

        backend_note = ""
        use_open3d = False
        if self.use_open3d:
            try:
                import open3d  # noqa: F401  (probe only)

                use_open3d = True
            except ImportError:
                backend_note = "open3d unavailable, used numpy ICP"

        seeds = self._initial_transforms(segmentation)

        # Stage 1 -- score every yaw seed cheaply on decimated clouds.
        rng = np.random.default_rng(self.seed)
        coarse_scene = scene
        if len(scene) > self.params.coarse_scene_points:
            coarse_scene = scene[
                rng.choice(len(scene), size=self.params.coarse_scene_points, replace=False)
            ]

        best_seed, best_score = seeds[0], (-1.0, -np.inf)
        for initial in seeds:
            transform, fitness, rmse = self._icp(
                coarse_scene,
                initial,
                model=self._coarse_model,
                model_sq=self._coarse_model_sq,
                iterations=self.params.coarse_iterations,
            )
            # Rank by fitness first, RMSE second: a seed that converged onto the
            # wrong face can show a tiny RMSE over very few inliers.
            score = (round(fitness, 3), -rmse)
            if score > best_score:
                best_score, best_seed = score, transform

        # Stage 2 -- full-resolution refinement of the winning seed only.
        if use_open3d:
            transform, fitness, rmse = self._icp_open3d(scene, best_seed)
        else:
            transform, fitness, rmse = self._icp(scene, best_seed)

        pose = Pose.from_matrix(transform, frame.intrinsics.frame_id)
        return pose, fitness, rmse, backend_note
