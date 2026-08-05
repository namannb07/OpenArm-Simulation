"""OpenArm Quest v2 — Workspace Reach Analysis.

Sweeps all joint configurations to determine the full reachable workspace.
Outputs 3D meshes (STL/PLY), point cloud CSVs, and a console report.

Usage:
    uv run python scripts/reach_analysis.py
    uv run python scripts/reach_analysis.py --samples 100000 --arm left --headless
"""

import argparse
import time
from pathlib import Path
import numpy as np
import mujoco
from openarm_mujoco.sim_controller import _build_model
from openarm_mujoco.workspace_analyzer import (
    WorkspaceAnalyzer, MeshExporter, print_reach_report
)

def main():
    parser = argparse.ArgumentParser(description="Workspace Reach Analysis")
    parser.add_argument('--samples', type=int, default=50000,
                        help="Number of random joint configurations (default: 50000)")
    parser.add_argument('--arm', type=str, choices=['left', 'right', 'both'], default='both',
                        help="Which arm(s) to analyze (default: both)")
    parser.add_argument('--headless', action='store_true',
                        help="Skip MuJoCo viewer animation")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parent.parent
    URDF = ROOT / 'models' / 'openarm_mujoco.urdf'
    OUTPUT_DIR = ROOT / 'output' / 'reach_analysis'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading MuJoCo model from: {URDF}")
    model = _build_model(str(URDF))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    analyzed_arms = []
    if args.arm == 'both':
        analyzed_arms = ['left', 'right']
    else:
        analyzed_arms = [args.arm]

    all_results = {}

    for arm_side in analyzed_arms:
        print(f"\n{'='*60}\nAnalyzing {arm_side.upper()} arm workspace...\n{'='*60}")
        analyzer = WorkspaceAnalyzer(model, data, arm_side)
        
        def progress_cb(current, total):
            if current % max(1, total // 10) == 0 or current == total:
                print(f"  Sampling: {current:,}/{total:,} ({100*current/total:.0f}%)")
                
        start_time = time.time()
        analyzer.sample_workspace(n_samples=args.samples, progress_callback=progress_cb)
        results = analyzer.compute_reach_envelope()
        elapsed = time.time() - start_time
        print(f"\nAnalysis completed in {elapsed:.1f} seconds")
        
        report = print_reach_report(results, arm_side, args.samples)
        all_results[arm_side] = results
        
        print("Exporting data files...")
        MeshExporter.export_stl(
            results['points'], results['hull_simplices'],
            OUTPUT_DIR / f'{arm_side}_arm_workspace.stl',
            label=f'OpenArm {arm_side} arm workspace'
        )
        MeshExporter.export_ply(
            results['points'], results['hull_simplices'],
            OUTPUT_DIR / f'{arm_side}_arm_workspace.ply',
            results['base_position']
        )
        MeshExporter.export_points_csv(
            results['points'], results['distances'],
            OUTPUT_DIR / f'reach_points_{arm_side}.csv'
        )
        
        report_path = OUTPUT_DIR / f'{arm_side}_report.txt'
        report_path.write_text(report, encoding='utf-8')

    # Reference grid mesh
    print("\nGenerating reference mesh and combined scene...")
    max_reach = max(r.get('max_reach_m', 1.0) for r in all_results.values())
    ref_mesh_path = OUTPUT_DIR / 'reference_grid.stl'
    MeshExporter.generate_reference_mesh(
        np.array([0.0, 0.0, 0.0]),
        max_reach,
        ref_mesh_path
    )
    
    # Combined STL
    stl_files = [OUTPUT_DIR / f'{side}_arm_workspace.stl' for side in analyzed_arms]
    stl_files.append(ref_mesh_path)
    combined_path = OUTPUT_DIR / 'combined_scene.stl'
    MeshExporter.combine_stl_files(stl_files, combined_path)

    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name:40s} {size_kb:8.1f} KB")

    if not args.headless and analyzed_arms:
        from openarm_mujoco.reach_visualizer import ReachVisualizer
        visualizer = ReachVisualizer(model, data)
        first_arm = analyzed_arms[0]
        visualizer.visualize(
            all_results[first_arm],
            first_arm,
            ref_mesh_path,
        )

if __name__ == '__main__':
    main()
