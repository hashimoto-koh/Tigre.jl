import json
import os

DEDICATED_FISTA_CODE = """# =============================================================================
# Estimation of FISTA Lipschitz Hyperparameter (`hyper`)
#
# FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) relies on proximal gradient descent.
# Its gradient step size is proportional to 1 / hyper, where hyper represents the Lipschitz
# constant L ≈ ||Aᵀ A||₂ (the spectral radius of the forward-backward projection operator).
#
# If `hyper` is left to the default fallback value, the gradient step size can be drastically
# mismatched (e.g. orders of magnitude too conservative), leading to poor reconstruction quality
# and near-zero convergence.
#
# We explicitly estimate `fista_hyper` beforehand using the SVD power method via `estimate_hyper(geo)`.
# Storing it in an explicit variable before calling `fista` makes the parameter calibration step
# transparent and easily verifiable, emphasizing the critical importance of proper operator norm scaling.
# =============================================================================
fista_hyper = estimate_hyper(geo)
println("Estimated FISTA Lipschitz hyperparameter: ", fista_hyper)
"""

def make_code_cell(source_str):
    lines = [l + "\n" for l in source_str.split("\n")[:-1]] + [source_str.split("\n")[-1]]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

def update_2d_cone_beam():
    path = "demos/demo_2D_cone_beam.ipynb"
    with open(path) as f:
        nb = json.load(f)
    for idx, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if "rec_fista_3d, hist = fista(" in src:
            # Update cell
            src = src.replace("niter = 20", "niter = 40")
            src = src.replace("# 20 iterations of FISTA", "# 40 iterations of FISTA")
            src = src.replace("geo,\n    niter;", "geo,\n    niter;\n    hyper = fista_hyper,")
            if "hyper = fista_hyper" not in src:
                src = src.replace("niter;\n", "niter;\n    hyper = fista_hyper,\n")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
            # Insert dedicated cell before idx
            nb["cells"].insert(idx, make_code_cell(DEDICATED_FISTA_CODE))
            break
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("Updated", path)

def update_2d_parallel_beam():
    path = "demos/demo_2D_parallel_beam.ipynb"
    with open(path) as f:
        nb = json.load(f)
    for idx, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if "rec_fista_3d, hist = fista(" in src:
            src = src.replace("niter = 20", "niter = 40")
            src = src.replace("# Run 20 iterations of FISTA", "# Run 40 iterations of FISTA")
            src = src.replace("geo,\n    niter;", "geo,\n    niter;\n    hyper = fista_hyper,")
            if "hyper = fista_hyper" not in src:
                src = src.replace("niter;\n", "niter;\n    hyper = fista_hyper,\n")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
            nb["cells"].insert(idx, make_code_cell(DEDICATED_FISTA_CODE))
            break
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("Updated", path)

def update_3d_cone_beam():
    path = "demos/demo_3D_cone_beam.ipynb"
    with open(path) as f:
        nb = json.load(f)
    for idx, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if "rec_fista, hist = fista(" in src:
            src = src.replace("niter = 10", "niter = 40")
            src = src.replace("geo,\n    niter;", "geo,\n    niter;\n    hyper = fista_hyper,")
            if "hyper = fista_hyper" not in src:
                src = src.replace("niter;\n", "niter;\n    hyper = fista_hyper,\n")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
            nb["cells"].insert(idx, make_code_cell(DEDICATED_FISTA_CODE))
            break
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("Updated", path)

def update_3d_parallel_beam():
    path = "demos/demo_3D_parallel_beam.ipynb"
    with open(path) as f:
        nb = json.load(f)
    for idx, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if "rec_fista, hist = fista(" in src:
            src = src.replace("niter = 10", "niter = 40")
            src = src.replace("geo,\n    niter;", "geo,\n    niter;\n    hyper = fista_hyper,")
            if "hyper = fista_hyper" not in src:
                src = src.replace("niter;\n", "niter;\n    hyper = fista_hyper,\n")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
            nb["cells"].insert(idx, make_code_cell(DEDICATED_FISTA_CODE))
            break
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("Updated", path)

def update_iterative_comparison():
    path = "demos/demo_iterative_comparison.ipynb"
    with open(path) as f:
        nb = json.load(f)
    for idx, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if 'println("1. Running FDK...")' in src:
            src = src.replace('println("5. Running FISTA with TV (20 iterations)...")', 'println("5. Running FISTA with TV (40 iterations)...")')
            src = src.replace('fista(projections, geo, 20;', 'fista(projections, geo, 40; hyper = fista_hyper,')
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
            nb["cells"].insert(idx, make_code_cell(DEDICATED_FISTA_CODE))
            break
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("Updated", path)

def update_laminography():
    path = "demos/demo_laminography_conebeam.ipynb"
    with open(path) as f:
        nb = json.load(f)
    for idx, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if "t_fista = @timed fista(" in src:
            src = src.replace('println("--> Reconstructing with FISTA (TV, hyper=:auto, 15 iter)...")', 'println("--> Reconstructing with FISTA (TV, hyper=fista_hyper, 40 iter)...")')
            src = src.replace('fista(proj_noisy, geo, 15; tviter = 10, hyper = :auto)', 'fista(proj_noisy, geo, 40; tviter = 10, hyper = fista_hyper)')
            src = src.replace('"FISTA (15 iter)"', '"FISTA (40 iter)"')
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
            nb["cells"].insert(idx, make_code_cell(DEDICATED_FISTA_CODE))
            break
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("Updated", path)

def update_low_dose():
    path = "demos/demo_low_dose_regularization.ipynb"
    with open(path) as f:
        nb = json.load(f)
    for idx, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if "rec_fista = fista(" in src:
            src = src.replace('println("--> 4/6 Computing FISTA (15 iter, hyper=:auto)...")', 'println("--> 4/6 Computing FISTA (40 iter, hyper=fista_hyper)...")')
            src = src.replace('fista(low_dose_projections, geo, 10; hyper = :auto, tviter = 8)', 'fista(low_dose_projections, geo, 40; hyper = fista_hyper, tviter = 8)')
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
            nb["cells"].insert(idx, make_code_cell(DEDICATED_FISTA_CODE))
            break
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("Updated", path)

def update_workspace_reuse():
    path = "demos/demo_workspace_reuse.ipynb"
    with open(path) as f:
        nb = json.load(f)
    for idx, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if "fista!(reconstructed_frame, sinograms[1], geo" in src:
            src = src.replace('hyper = :auto', 'hyper = fista_hyper')
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
            nb["cells"].insert(idx, make_code_cell(DEDICATED_FISTA_CODE))
            break
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("Updated", path)

if __name__ == "__main__":
    update_2d_cone_beam()
    update_2d_parallel_beam()
    update_3d_cone_beam()
    update_3d_parallel_beam()
    update_iterative_comparison()
    update_laminography()
    update_low_dose()
    update_workspace_reuse()
    print("All notebooks updated with fista_hyper and niter=40.")
