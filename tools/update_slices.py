import json
import os

def update_notebooks():
    # 1. demo_2D_cone_beam.ipynb
    with open("demos/demo_2D_cone_beam.ipynb") as f:
        nb = json.load(f)
    for c in nb["cells"]:
        src = "".join(c["source"])
        if "extract the central slice" in src:
            src = src.replace("extract the central slice", "extract an off-center axial slice ($Z = 3N/8$) containing fine high-contrast structures")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
        if "phantom_2d = vol_3d_full[:, :, N ÷ 2]" in src:
            src = src.replace("phantom_2d = vol_3d_full[:, :, N ÷ 2]", "phantom_2d = vol_3d_full[:, :, 3 * N ÷ 8]")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
    with open("demos/demo_2D_cone_beam.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    print("Updated demo_2D_cone_beam.ipynb")

    # 2. demo_2D_parallel_beam.ipynb
    with open("demos/demo_2D_parallel_beam.ipynb") as f:
        nb = json.load(f)
    for c in nb["cells"]:
        src = "".join(c["source"])
        if "extract the central axial slice ($Z = N/2$)" in src:
            src = src.replace("extract the central axial slice ($Z = N/2$)", "extract an off-center axial slice ($Z = 3N/8$)")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
        if "extract central axial slice" in src:
            src = src.replace("extract central axial slice", "extract off-center axial slice ($Z = 3N/8$)")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
        if "phantom_2d = vol_3d_full[:, :, N ÷ 2]" in src:
            src = src.replace("phantom_2d = vol_3d_full[:, :, N ÷ 2]", "phantom_2d = vol_3d_full[:, :, 3 * N ÷ 8]")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
    with open("demos/demo_2D_parallel_beam.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    print("Updated demo_2D_parallel_beam.ipynb")

    # 3. demo_3D_cone_beam.ipynb
    with open("demos/demo_3D_cone_beam.ipynb") as f:
        nb = json.load(f)
    for c in nb["cells"]:
        src = "".join(c["source"])
        if "mid_z = N ÷ 2" in src:
            src = src.replace("mid_z = N ÷ 2", "mid_z = 3 * N ÷ 8")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
        if "Centerline horizontal profile through axial slice" in src:
            src = src.replace("Centerline horizontal profile through axial slice", "Horizontal profile through axial slice ($Z = 3N/8$)")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
        if "title = \"Central Profile (Y=$mid_y, Z=$mid_z)\"" in src:
            src = src.replace("title = \"Central Profile (Y=$mid_y, Z=$mid_z)\"", "title = \"Profile (Y=$mid_y, Z=$mid_z)\"")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
    with open("demos/demo_3D_cone_beam.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    print("Updated demo_3D_cone_beam.ipynb")

    # 4. demo_3D_parallel_beam.ipynb
    with open("demos/demo_3D_parallel_beam.ipynb") as f:
        nb = json.load(f)
    for c in nb["cells"]:
        src = "".join(c["source"])
        if "mid_z = N ÷ 2" in src:
            src = src.replace("mid_z = N ÷ 2", "mid_z = 3 * N ÷ 8")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
        if "central axial slice (Z = N/2)" in src:
            src = src.replace("central axial slice (Z = N/2)", "axial slice (Z = 3N/8)")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
    with open("demos/demo_3D_parallel_beam.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    print("Updated demo_3D_parallel_beam.ipynb")

    # 5. demo_iterative_comparison.ipynb
    with open("demos/demo_iterative_comparison.ipynb") as f:
        nb = json.load(f)
    for c in nb["cells"]:
        src = "".join(c["source"])
        if "phantom_2d = vol_3d_full[:, :, N ÷ 2]" in src:
            src = src.replace("phantom_2d = vol_3d_full[:, :, N ÷ 2]", "phantom_2d = vol_3d_full[:, :, 3 * N ÷ 8]")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
    with open("demos/demo_iterative_comparison.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    print("Updated demo_iterative_comparison.ipynb")

    # 6. demo_krylov_solvers.ipynb
    with open("demos/demo_krylov_solvers.ipynb") as f:
        nb = json.load(f)
    for c in nb["cells"]:
        src = "".join(c["source"])
        if "mid_z = size(phantom, 3) ÷ 2" in src:
            src = src.replace("mid_z = size(phantom, 3) ÷ 2", "mid_z = 3 * size(phantom, 3) ÷ 8")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
        if "title = \"Horizontal Central Line Profile Comparison\"" in src:
            src = src.replace("title = \"Horizontal Central Line Profile Comparison\"", "title = \"Horizontal Line Profile Comparison (Z=$mid_z)\"")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
    with open("demos/demo_krylov_solvers.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    print("Updated demo_krylov_solvers.ipynb")

    # 7. demo_low_dose_regularization.ipynb
    with open("demos/demo_low_dose_regularization.ipynb") as f:
        nb = json.load(f)
    for c in nb["cells"]:
        src = "".join(c["source"])
        if "central slices and line profiles" in src:
            src = src.replace("central slices and line profiles", "axial slices ($Z = 3N/8$) and line profiles")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
        if "mid_z = size(phantom, 3) ÷ 2" in src:
            src = src.replace("mid_z = size(phantom, 3) ÷ 2", "mid_z = 3 * size(phantom, 3) ÷ 8")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
        if "Central horizontal line profile" in src:
            src = src.replace("Central horizontal line profile", "Horizontal line profile (Z = 3N/8)")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
    with open("demos/demo_low_dose_regularization.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    print("Updated demo_low_dose_regularization.ipynb")

    # 8. demo_preprocessing.ipynb
    with open("demos/demo_preprocessing.ipynb") as f:
        nb = json.load(f)
    for c in nb["cells"]:
        src = "".join(c["source"])
        if "mid_z = size(phantom, 3) ÷ 2" in src:
            src = src.replace("mid_z = size(phantom, 3) ÷ 2", "mid_z = 3 * size(phantom, 3) ÷ 8")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
        if "Display central slice and sinogram" in src:
            src = src.replace("Display central slice and sinogram", "Display axial slice (Z = 3N/8) and sinogram")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
        if "Visualize central slices" in src:
            src = src.replace("Visualize central slices", "Visualize axial slices (Z = 3N/8)")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
    with open("demos/demo_preprocessing.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    print("Updated demo_preprocessing.ipynb")

    # 9. demo_sparse_view_cs.ipynb
    with open("demos/demo_sparse_view_cs.ipynb") as f:
        nb = json.load(f)
    for c in nb["cells"]:
        src = "".join(c["source"])
        if "mid_z = size(phantom, 3) ÷ 2" in src:
            src = src.replace("mid_z = size(phantom, 3) ÷ 8", "mid_z = 3 * size(phantom, 3) ÷ 8")
            if "mid_z = size(phantom, 3) ÷ 2" in src:
                src = src.replace("mid_z = size(phantom, 3) ÷ 2", "mid_z = 3 * size(phantom, 3) ÷ 8")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
    with open("demos/demo_sparse_view_cs.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    print("Updated demo_sparse_view_cs.ipynb")

    # 10. demo_workspace_reuse.ipynb
    with open("demos/demo_workspace_reuse.ipynb") as f:
        nb = json.load(f)
    for c in nb["cells"]:
        src = "".join(c["source"])
        if "base_phantom = vol_3d_full[:, :, N ÷ 2]" in src:
            src = src.replace("base_phantom = vol_3d_full[:, :, N ÷ 2]", "base_phantom = vol_3d_full[:, :, 3 * N ÷ 8]")
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
    with open("demos/demo_workspace_reuse.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    print("Updated demo_workspace_reuse.ipynb")

if __name__ == "__main__":
    update_notebooks()
