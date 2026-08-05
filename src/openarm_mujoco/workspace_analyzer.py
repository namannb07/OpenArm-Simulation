import struct
import numpy as np
import mujoco
from pathlib import Path
from typing import Optional, Callable, Union

INCH_M = 0.0254   # 1 inch in meters
FOOT_M = 0.3048   # 1 foot in meters

def meters_to_imperial(m: float) -> str:
    """Convert meters to human-readable imperial string."""
    total_inches = m / INCH_M
    feet = int(total_inches // 12)
    inches = total_inches - (feet * 12)
    
    if feet > 0:
        return f"{feet} ft {inches:.1f} in"
    else:
        return f"{inches:.1f} in"

class WorkspaceAnalyzer:
    """Computes the reachable workspace of one OpenArm arm via Monte Carlo sampling of joint configurations."""
    
    TCP_OFFSET = np.array([-0.00143, 0.0, -0.068])

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, arm_side: str):
        """
        Initialize the analyzer with a MuJoCo model, data, and the side of the arm ('left' or 'right').
        """
        self.model = model
        self.data = data
        self.arm_side = arm_side
        self.tcp_points: np.ndarray = np.empty((0, 3))
        self.joint_configs: list[list[float]] = []
        
        self._joint_ids = []
        self._joint_ranges = []
        self._ee_body_id = -1
        self._base_body_id = -1
        
        self._setup()

    def _setup(self):
        """Sets up joint and body IDs from the MuJoCo model."""
        self._joint_ids = []
        self._joint_ranges = []
        
        for i in range(1, 8):
            name = f"openarm_{self.arm_side}_joint{i}"
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id == -1:
                raise ValueError(f"Joint not found: {name}")
            self._joint_ids.append(jnt_id)
            self._joint_ranges.append(self.model.jnt_range[jnt_id].copy())
            
        ee_name = f"openarm_{self.arm_side}_ee_base_link"
        self._ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
        if self._ee_body_id == -1:
            raise ValueError(f"Body not found: {ee_name}")
            
        # NOTE: MuJoCo merges fixed-joint links into their parent body.
        # openarm_{side}_base_link is merged into its parent, so we use
        # the anchor point of joint1 as the fixed shoulder reference for
        # measuring reach distances.  data.xanchor[jnt_id] gives the
        # world-frame position of the joint pivot, which stays constant
        # regardless of joint angle.
        self._j1_id = self._joint_ids[0]  # joint1 = shoulder

    def _get_tcp_position(self) -> np.ndarray:
        """Returns the global position of the Tool Center Point."""
        ee_pos = self.data.xpos[self._ee_body_id].copy()
        ee_rot = self.data.xmat[self._ee_body_id].reshape(3, 3)
        return ee_pos + ee_rot @ self.TCP_OFFSET

    def _set_joint_config(self, angles: Union[np.ndarray, list[float]]):
        """Sets the specified angles for the arm's joints."""
        for jnt_id, angle in zip(self._joint_ids, angles):
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            self.data.qpos[qpos_adr] = angle

    def get_base_position(self) -> np.ndarray:
        """Returns the shoulder (J1) anchor position in world frame.
        
        This is the fixed pivot point of joint1, used as the reference
        origin for measuring arm reach distances.
        """
        return self.data.xanchor[self._j1_id].copy()

    def sample_workspace(self, n_samples: int = 50000, progress_callback: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
        """
        Samples the workspace using a hybrid approach:
        - 80% random Monte Carlo within joint limits.
        - 20% boundary-focused (one joint at min/max limit, others random).
        """
        orig_qpos = self.data.qpos.copy()
        orig_qvel = self.data.qvel.copy()
        
        n_random = int(n_samples * 0.8)
        n_boundary = n_samples - n_random
        
        points = []
        configs = []
        
        # 1. 80% random Monte Carlo
        for i in range(n_random):
            angles = []
            for rng in self._joint_ranges:
                angles.append(np.random.uniform(rng[0], rng[1]))
            
            self._set_joint_config(angles)
            mujoco.mj_forward(self.model, self.data)
            points.append(self._get_tcp_position())
            configs.append(list(angles))
            
            if progress_callback and (i + 1) % 2000 == 0:
                progress_callback(i + 1, n_samples)
                
        # 2. 20% boundary-focused
        for i in range(n_boundary):
            angles = []
            boundary_idx = np.random.randint(0, 7)
            is_min = np.random.choice([True, False])
            
            for j, rng in enumerate(self._joint_ranges):
                if j == boundary_idx:
                    angles.append(rng[0] if is_min else rng[1])
                else:
                    angles.append(np.random.uniform(rng[0], rng[1]))
                    
            self._set_joint_config(angles)
            mujoco.mj_forward(self.model, self.data)
            points.append(self._get_tcp_position())
            configs.append(list(angles))
            
            curr_sample = n_random + i + 1
            if progress_callback and curr_sample % 2000 == 0:
                progress_callback(curr_sample, n_samples)
                
        self.tcp_points = np.array(points)
        self.joint_configs = configs
        
        # Restore state
        self.data.qpos[:] = orig_qpos
        self.data.qvel[:] = orig_qvel
        mujoco.mj_forward(self.model, self.data)
        
        return self.tcp_points

    def compute_reach_envelope(self) -> dict:
        """
        Computes the convex hull and reach statistics from the sampled points.
        Returns a dictionary with points, hull info, reach limits, volume, and area.
        """
        from scipy.spatial import ConvexHull
        
        if len(self.tcp_points) == 0:
            raise ValueError("No points sampled yet.")
            
        base_pos = self.get_base_position()
        distances = np.linalg.norm(self.tcp_points - base_pos, axis=1)
        
        hull = ConvexHull(self.tcp_points)
        
        # Find the actual max and min reach point positions
        max_idx = int(np.argmax(distances))
        min_idx = int(np.argmin(distances))

        return {
            'points': self.tcp_points,
            'configs': self.joint_configs,
            'hull': hull,
            'hull_vertices': self.tcp_points[hull.vertices],
            'hull_simplices': hull.simplices,
            'max_reach_m': float(np.max(distances)),
            'min_reach_m': float(np.min(distances)),
            'mean_reach_m': float(np.mean(distances)),
            'max_reach_pos': self.tcp_points[max_idx],
            'min_reach_pos': self.tcp_points[min_idx],
            'volume_m3': float(hull.volume),
            'surface_area_m2': float(hull.area),
            'base_position': base_pos,
            'distances': distances,
            'n_hull_vertices': len(hull.vertices),
            'n_hull_faces': len(hull.simplices)
        }

class MeshExporter:
    """Provides utilities to export workspace geometry to various formats."""

    @staticmethod
    def _write_triangles_stl(triangles: list[tuple], filepath: Path, label: str = 'OpenArm Workspace'):
        """Writes a list of triangles to a binary STL file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            # 80-byte header
            header = label.encode('ascii')[:80].ljust(80, b'\x00')
            f.write(header)
            
            f.write(struct.pack('<I', len(triangles)))
            
            for v0, v1, v2 in triangles:
                v0 = np.array(v0)
                v1 = np.array(v1)
                v2 = np.array(v2)
                
                normal = np.cross(v1 - v0, v2 - v0)
                norm = np.linalg.norm(normal)
                if norm > 1e-6:
                    normal = normal / norm
                else:
                    normal = np.zeros(3)
                    
                f.write(struct.pack('<3f', *normal))
                f.write(struct.pack('<3f', *v0))
                f.write(struct.pack('<3f', *v1))
                f.write(struct.pack('<3f', *v2))
                f.write(struct.pack('<H', 0))

    @staticmethod
    def _box_triangles(center: np.ndarray, half_extents: np.ndarray) -> list[tuple]:
        """Generates triangles for an axis-aligned bounding box."""
        hx, hy, hz = half_extents
        corners = [
            center + np.array([-hx, -hy, -hz]), # 0
            center + np.array([ hx, -hy, -hz]), # 1
            center + np.array([ hx,  hy, -hz]), # 2
            center + np.array([-hx,  hy, -hz]), # 3
            center + np.array([-hx, -hy,  hz]), # 4
            center + np.array([ hx, -hy,  hz]), # 5
            center + np.array([ hx,  hy,  hz]), # 6
            center + np.array([-hx,  hy,  hz]), # 7
        ]
        
        faces = [
            (0,2,1), (0,3,2), # bottom
            (4,5,6), (4,6,7), # top
            (0,1,5), (0,5,4), # front
            (2,3,7), (2,7,6), # back
            (0,3,7), (0,7,4), # left
            (1,2,6), (1,6,5)  # right
        ]
        
        return [(corners[f[0]], corners[f[1]], corners[f[2]]) for f in faces]

    @staticmethod
    def export_stl(points: np.ndarray, simplices: np.ndarray, filepath: Path, label: str = 'OpenArm Workspace'):
        """Exports hull geometry to a binary STL file."""
        triangles = []
        for face in simplices:
            triangles.append((points[face[0]], points[face[1]], points[face[2]]))
        MeshExporter._write_triangles_stl(triangles, filepath, label)

    @staticmethod
    def export_ply(points: np.ndarray, simplices: np.ndarray, filepath: Path, base_pos: np.ndarray):
        """Exports hull geometry to a binary PLY file with distance-based vertex coloring."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        unique_indices = np.unique(simplices.flatten())
        remap = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}
        
        v_points = points[unique_indices]
        
        distances = np.linalg.norm(v_points - base_pos, axis=1)
        max_d = np.max(distances)
        min_d = np.min(distances)
        if max_d > min_d:
            t = (distances - min_d) / (max_d - min_d)
        else:
            t = np.zeros_like(distances)
            
        colors = np.zeros((len(v_points), 3), dtype=np.uint8)
        for i, ti in enumerate(t):
            if ti < 0.5:
                colors[i] = [0, int(510*ti), int(255*(1-2*ti))]
            else:
                colors[i] = [int(510*(ti-0.5)), int(255*(2-2*ti)), 0]
                
        with open(filepath, 'wb') as f:
            header = f"""ply
format binary_little_endian 1.0
comment OpenArm workspace reach envelope
element vertex {len(v_points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face {len(simplices)}
property list uchar int vertex_indices
end_header
"""
            f.write(header.encode('ascii'))
            
            for i in range(len(v_points)):
                f.write(struct.pack('<3f', *v_points[i]))
                f.write(struct.pack('<3B', *colors[i]))
                
            for face in simplices:
                new_face = [remap[idx] for idx in face]
                f.write(struct.pack('<B', 3))
                f.write(struct.pack('<3i', *new_face))

    @staticmethod
    def generate_reference_mesh(origin_pos: np.ndarray, max_reach: float, filepath: Path):
        """Generates reference geometry (grid, circles, rulers) and saves to STL."""
        triangles = []
        origin_pos = np.array(origin_pos)
        max_extent_ft = int(np.ceil(max_reach / FOOT_M)) + 1
        
        # A) Floor Grid
        for i in range(-max_extent_ft, max_extent_ft + 1):
            # X lines
            center_x = origin_pos + np.array([0, i * FOOT_M, 0.00025])
            hx_x = np.array([max_extent_ft * FOOT_M, 0.0005, 0.00025])
            triangles.extend(MeshExporter._box_triangles(center_x, hx_x))
            
            # Y lines
            center_y = origin_pos + np.array([i * FOOT_M, 0, 0.00025])
            hx_y = np.array([0.0005, max_extent_ft * FOOT_M, 0.00025])
            triangles.extend(MeshExporter._box_triangles(center_y, hx_y))
            
        # B) Concentric Circles
        for r_in in np.arange(6, (max_reach / INCH_M) + 6, 6):
            r_m = r_in * INCH_M
            is_foot = (r_in % 12 == 0)
            
            thickness = 0.004 if is_foot else 0.002
            height = 0.002 if is_foot else 0.001
            z_offset = height / 2.0
            
            n_segs = 72
            for i in range(n_segs):
                a1 = i * 2 * np.pi / n_segs
                a2 = (i + 1) * 2 * np.pi / n_segs
                
                r_inner = r_m - thickness / 2
                r_outer = r_m + thickness / 2
                
                # Bottom vertices
                v0_b = origin_pos + np.array([r_inner * np.cos(a1), r_inner * np.sin(a1), 0])
                v1_b = origin_pos + np.array([r_outer * np.cos(a1), r_outer * np.sin(a1), 0])
                v2_b = origin_pos + np.array([r_outer * np.cos(a2), r_outer * np.sin(a2), 0])
                v3_b = origin_pos + np.array([r_inner * np.cos(a2), r_inner * np.sin(a2), 0])
                
                # Top vertices
                v0_t = v0_b + np.array([0, 0, height])
                v1_t = v1_b + np.array([0, 0, height])
                v2_t = v2_b + np.array([0, 0, height])
                v3_t = v3_b + np.array([0, 0, height])
                
                # bottom
                triangles.extend([(v0_b, v2_b, v1_b), (v0_b, v3_b, v2_b)])
                # top
                triangles.extend([(v0_t, v1_t, v2_t), (v0_t, v2_t, v3_t)])
                # inner
                triangles.extend([(v0_b, v0_t, v3_t), (v0_b, v3_t, v3_b)])
                # outer
                triangles.extend([(v1_b, v2_b, v2_t), (v1_b, v2_t, v1_t)])
                # side1 (a1)
                triangles.extend([(v0_b, v1_b, v1_t), (v0_b, v1_t, v0_t)])
                # side2 (a2)
                triangles.extend([(v3_b, v3_t, v2_t), (v3_b, v2_t, v2_b)])

        # C) Ruler Posts
        ruler_length = max_reach
        for axis_idx, axis in enumerate([np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]):
            # Shaft
            center = origin_pos + axis * (ruler_length / 2)
            hx = np.array([0.0015, 0.0015, 0.0015])
            hx[axis_idx] = ruler_length / 2
            triangles.extend(MeshExporter._box_triangles(center, hx))
            
            # Ticks
            perp = np.array([0,1,0]) if axis_idx != 1 else np.array([1,0,0])
            for d_in in np.arange(1, (ruler_length / INCH_M) + 1):
                d_m = d_in * INCH_M
                tick_center = origin_pos + axis * d_m
                
                if d_in % 12 == 0:
                    tick_len, tick_width = 0.015, 0.003
                    # Add cube marker
                    marker_center = tick_center + perp * (tick_len / 2)
                    triangles.extend(MeshExporter._box_triangles(marker_center, np.array([0.004, 0.004, 0.004])))
                elif d_in % 6 == 0:
                    tick_len, tick_width = 0.010, 0.002
                else:
                    tick_len, tick_width = 0.005, 0.001
                    
                tick_center = tick_center + perp * (tick_len / 2)
                hx_tick = np.array([tick_width/2, tick_width/2, tick_width/2])
                tick_axis_idx = np.argmax(np.abs(perp))
                hx_tick[tick_axis_idx] = tick_len / 2
                
                triangles.extend(MeshExporter._box_triangles(tick_center, hx_tick))
                
        MeshExporter._write_triangles_stl(triangles, filepath, 'Reference Mesh')

    @staticmethod
    def export_points_csv(points: np.ndarray, distances: np.ndarray, filepath: Path):
        """Exports the raw sampled points and distances to a CSV file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write("x_m,y_m,z_m,distance_m,distance_mm,distance_in\n")
            for p, d in zip(points, distances):
                f.write(f"{p[0]},{p[1]},{p[2]},{d},{d*1000},{d/INCH_M}\n")

    @staticmethod
    def combine_stl_files(input_paths: list[Path], output_path: Path):
        """Combines multiple STL files into one."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        total_triangles = 0
        data_blocks = []
        
        for p in input_paths:
            with open(p, 'rb') as f:
                f.seek(80)
                count = struct.unpack('<I', f.read(4))[0]
                total_triangles += count
                data_blocks.append(f.read(count * 50))
                
        with open(output_path, 'wb') as f:
            f.write(b'Combined STL'.ljust(80, b'\x00'))
            f.write(struct.pack('<I', total_triangles))
            for block in data_blocks:
                f.write(block)

def print_reach_report(results: dict, arm_side: str, n_samples: int) -> str:
    """Prints a styled reachability report based on workspace analysis results."""
    v_m3 = results['volume_m3']
    v_ft3 = v_m3 / (FOOT_M ** 3)
    a_m2 = results['surface_area_m2']
    a_ft2 = a_m2 / (FOOT_M ** 2)
    
    max_m = results['max_reach_m']
    min_m = results['min_reach_m']
    mean_m = results['mean_reach_m']
    
    lines = [
        "╔══════════════════════════════════════════════════════════════════════════╗",
        f"║ OpenArm Workspace Reach Report : {arm_side.upper():<40}║",
        "╠══════════════════════════════════════════════════════════════════════════╣",
        f"║ Samples Evaluated : {n_samples:<52} ║",
        "╠══════════════════════════════════════════════════════════════════════════╣",
        "║ REACH STATISTICS                                                         ║",
        f"║ Max Reach         : {max_m*1000:7.1f} mm  ({meters_to_imperial(max_m):<37}) ║",
        f"║ Min Reach         : {min_m*1000:7.1f} mm  ({meters_to_imperial(min_m):<37}) ║",
        f"║ Mean Reach        : {mean_m*1000:7.1f} mm  ({meters_to_imperial(mean_m):<37}) ║",
        "╠══════════════════════════════════════════════════════════════════════════╣",
        "║ GEOMETRY                                                                 ║",
        f"║ Volume            : {v_m3:7.3f} m³  ({v_ft3:6.2f} ft³){' ':>29} ║",
        f"║ Surface Area      : {a_m2:7.3f} m²  ({a_ft2:6.2f} ft²){' ':>29} ║",
        f"║ Hull Vertices     : {results['n_hull_vertices']:<52} ║",
        f"║ Hull Faces        : {results['n_hull_faces']:<52} ║",
        "╚══════════════════════════════════════════════════════════════════════════╝"
    ]
    
    report = "\n".join(lines)
    print(report)
    return report
