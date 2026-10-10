import glob
import os
import subprocess
import sys
import time

from build_demo_runner import build_runner

DEMOS = [
    "demos/01_2D_parallel_beam.ipynb",
    "demos/02_2D_cone_beam.ipynb",
    "demos/03_3D_parallel_beam.ipynb",
    "demos/04_3D_cone_beam.ipynb",
    "demos/05_iterative_comparison.ipynb",
    "demos/06_krylov_solvers.ipynb",
    "demos/07_preprocessing.ipynb",
    "demos/08_offset_detector_fov.ipynb",
    "demos/09_laminography_conebeam.ipynb",
    "demos/10_emission_mlem_osem.ipynb",
    "demos/11_low_dose_regularization.ipynb",
    "demos/12_sparse_view_cs.ipynb",
    "demos/13_workspace_reuse.ipynb",
]

def main():
    root_dir = os.path.abspath(".")
    os.makedirs(os.path.join(root_dir, "docs/src/assets/tutorials"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "docs/src/demos"), exist_ok=True)

    failed = []
    for demo_path in DEMOS:
        stem = os.path.splitext(os.path.basename(demo_path))[0]
        print(f"\n==========================================")
        print(f"Processing: {stem}")
        print(f"==========================================")
        t0 = time.time()
        runner = build_runner(demo_path)
        
        cmd = ["julia", "--startup-file=no", "--project=.", runner]
        res = subprocess.run(cmd, cwd=root_dir)
        elapsed = time.time() - t0
        if res.returncode != 0:
            print(f"FAILED: {stem} in {elapsed:.1f} s")
            failed.append(stem)
        else:
            print(f"SUCCESS: {stem} in {elapsed:.1f} s")

    print("\n==========================================")
    print("Summary of Demo Conversions:")
    print(f"Total: {len(DEMOS)}, Failed: {len(failed)}")
    if failed:
        print("Failed demos:", failed)
        sys.exit(1)
    else:
        print("All demos successfully converted!")

if __name__ == "__main__":
    main()
