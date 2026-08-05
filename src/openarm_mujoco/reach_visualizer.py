"""MuJoCo passive viewer visualization for reach analysis results.

Displays the workspace boundary as colored sphere markers and animates
the arm through a subset of sampled configurations to demonstrate the
reach envelope.
"""

from pathlib import Path
import time

import numpy as np
import mujoco
import mujoco.viewer


class ReachVisualizer:
    """Visualizes reach analysis results using MuJoCo's passive viewer."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

    @staticmethod
    def _distance_to_rgba(dist: float, min_dist: float, max_dist: float) -> list[float]:
        """Map a distance to a blue → green → red colour ramp (RGBA)."""
        t = (dist - min_dist) / max(max_dist - min_dist, 1e-8)
        t = max(0.0, min(1.0, t))
        if t < 0.5:
            return [0.0, t * 2.0, 1.0 - t * 2.0, 0.7]
        else:
            return [(t - 0.5) * 2.0, 1.0 - (t - 0.5) * 2.0, 0.0, 0.7]

    @staticmethod
    def _add_marker(viewer, pos, size: float, rgba) -> None:
        """Add a sphere marker to the user scene."""
        scn = viewer.user_scn
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            [size, 0.0, 0.0],
            np.asarray(pos, dtype=np.float64),
            np.eye(3).flatten(),
            np.asarray(rgba, dtype=np.float32),
        )
        scn.ngeom += 1

    @staticmethod
    def _add_line(viewer, p1, p2, radius: float, rgba) -> None:
        """Add a capsule line between two points."""
        scn = viewer.user_scn
        if scn.ngeom >= scn.maxgeom:
            return
        p1, p2 = np.asarray(p1, dtype=np.float64), np.asarray(p2, dtype=np.float64)
        mujoco.mjv_makeConnector(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            radius,
            p1[0], p1[1], p1[2],
            p2[0], p2[1], p2[2],
        )
        scn.geoms[scn.ngeom].rgba[:] = np.asarray(rgba, dtype=np.float32)
        scn.ngeom += 1

    def visualize(
        self,
        results: dict,
        arm_side: str,
        reference_stl_path: Path | None = None,
    ) -> None:
        """Launch the MuJoCo viewer with workspace markers and arm animation.

        Parameters
        ----------
        results
            Output dict from ``WorkspaceAnalyzer.compute_reach_envelope()``.
        arm_side
            ``'left'`` or ``'right'``.
        reference_stl_path
            Optional path to reference STL (unused by viewer but kept for API
            consistency).
        """
        hull_vertices = results.get("hull_vertices", np.empty((0, 3)))
        base_pos = np.asarray(results.get("base_position", [0, 0, 0]))
        configs = results.get("configs", [])
        max_reach_pos = results.get("max_reach_pos")
        min_reach_pos = results.get("min_reach_pos")

        # Distance bounds for colour mapping
        if len(hull_vertices) > 0:
            hull_dists = np.linalg.norm(hull_vertices - base_pos, axis=1)
            min_dist, max_dist = float(hull_dists.min()), float(hull_dists.max())
        else:
            min_dist, max_dist = 0.0, 1.0

        # Select ~500 configs for animation (evenly spaced subset)
        n_configs = len(configs)
        step = max(1, n_configs // 500)
        anim_configs = [configs[i] for i in range(0, n_configs, step)]

        # Look up joint / actuator IDs for the target arm
        jnt_ids: list[int] = []
        act_ids: list[int] = []
        for i in range(1, 8):
            jnt_name = f"openarm_{arm_side}_joint{i}"
            act_name = f"{arm_side}_j{i}_act"
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            a_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if j_id >= 0:
                jnt_ids.append(j_id)
            if a_id >= 0:
                act_ids.append(a_id)

        print(
            f"\nLaunching MuJoCo viewer — showing {len(hull_vertices)} hull "
            f"markers and animating {len(anim_configs)} arm poses for the "
            f"{arm_side.upper()} arm.  Close the viewer window to exit.\n"
        )

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # ── Pre-populate markers (must be inside viewer.lock()) ────────
            with viewer.lock():
                # Hull boundary sphere markers
                for v in hull_vertices:
                    d = float(np.linalg.norm(v - base_pos))
                    rgba = self._distance_to_rgba(d, min_dist, max_dist)
                    self._add_marker(viewer, v, 0.008, rgba)

                # Max-reach marker (large red)
                if max_reach_pos is not None:
                    self._add_marker(viewer, max_reach_pos, 0.025, [1, 0, 0, 1])
                    self._add_line(
                        viewer, base_pos, max_reach_pos, 0.003, [1, 0, 0, 0.5]
                    )

                # Min-reach marker (large blue)
                if min_reach_pos is not None:
                    self._add_marker(viewer, min_reach_pos, 0.025, [0, 0, 1, 1])
                    self._add_line(
                        viewer, base_pos, min_reach_pos, 0.003, [0, 0, 1, 0.5]
                    )

            # ── Animation loop ────────────────────────────────────────────
            config_idx = 0
            last_switch = time.time()
            pose_hold = 0.3  # seconds per pose

            while viewer.is_running():
                step_start = time.perf_counter()

                # Advance to the next animation config periodically
                now = time.time()
                if config_idx < len(anim_configs) and now - last_switch >= pose_hold:
                    angles = anim_configs[config_idx]
                    # Set joint positions and actuator targets
                    for idx, j_id in enumerate(jnt_ids):
                        if idx < len(angles):
                            self.data.qpos[self.model.jnt_qposadr[j_id]] = angles[idx]
                    for idx, a_id in enumerate(act_ids):
                        if idx < len(angles):
                            self.data.ctrl[a_id] = angles[idx]
                    config_idx += 1
                    last_switch = now

                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                # Maintain ~500 Hz to match timestep
                elapsed = time.perf_counter() - step_start
                sleep_time = self.model.opt.timestep - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
