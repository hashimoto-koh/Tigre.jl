# About TIGRE & Upstream

!!! note "Disclaimer of Affiliation"
    `Tigre.jl` is an unofficial Julia port of [TIGRE: Tomographic Iterative GPU-based Reconstruction Toolbox](https://github.com/CERN/TIGRE).
    This project is **not** affiliated with, endorsed by, or maintained by CERN, the University of Bath, or the upstream TIGRE contributors.
    TIGRE is originally developed by the University of Bath and CERN contributors and is distributed under the BSD 3-Clause License.

## The TIGRE Project

The [TIGRE](https://github.com/CERN/TIGRE) (Tomographic Iterative GPU-based Reconstruction Toolbox) project is an open-source software suite developed by Ander Biguri, the University of Bath, CERN (European Organization for Nuclear Research), and international contributors. It was created to provide the medical imaging and tomography research communities with fast, accurate, and GPU-accelerated algorithms for 3D computed tomography (CT) reconstruction.

TIGRE revolutionized open tomographic reconstruction by providing highly optimized CUDA ray-tracing and voxel-driven backprojection kernels capable of handling complex geometric trajectories, helical scans, and arbitrary detector configurations.

`Tigre.jl` is a Julia port and interface built directly upon that upstream core. This repository is developed with explicit respect for the original project and with sincere gratitude to Ander Biguri, Manjit Dosanjh, Steven Hancock, Manuchehr Soleimani, and all contributors who made the upstream implementation, research, and documentation publicly available under an open-source license.

---

## Relationship Between Tigre.jl and Upstream TIGRE

### 1. Direct C/CUDA Integration (Zero Python / MATLAB Overhead)

While upstream TIGRE offers MATLAB and Python interfaces, `Tigre.jl` provides a Julia-first interface that binds directly to the compiled C++/CUDA kernels.
- **No Python runtime required**: No `PyCall` or `PythonCall` dependency.
- **No MATLAB runtime required**.
- Julia communicates with TIGRE's native projection and backprojection engines via a thin C ABI shim (`deps/wrapper/`) through Julia's zero-overhead `ccall`.

### 2. Canonical Column-Major Memory Layout

In C and Python (row-major), 3D volumes are commonly ordered as `(z, y, x)` and projections as `(angles, v, u)`.
Julia is natively 1-indexed and column-major:
- Volumes in `Tigre.jl` are stored as `volume[x, y, z]` (with X innermost and fastest in memory).
- Projections are stored as `projection[u, v, angle]` (with detector horizontal axis U innermost).

Because this memory layout aligns precisely with the underlying hardware textures and linear memory expected by TIGRE's CUDA kernels, **no memory transpositions, permutations, or hidden copies** occur during projection (`Ax`) or backprojection (`Atb`).

### 3. Pure Julia High-Level Solvers

All analytical reconstruction algorithms (`fdk`, `fbp`), algebraic solvers (`sart`, `ossart`, `sirt`, etc.), POCS methods (`asd_pocs`, `awasd_pocs`, `piccs`), and Krylov solvers (`cgls`, `lsmr`, `ab_gmres`, etc.) are implemented purely in Julia:
- **SciPy Independence**: Where upstream Python relied on SciPy (such as for `hybrid_flsqr_tv`'s inner solver), `Tigre.jl` implements a pure-Julia weighted Conjugate Gradient solver.
- **Customizability**: Julia's multiple dispatch allows users to easily extend algorithms, inject custom regularizers, or inspect convergence history without touching C++ or CUDA code.

### 4. Zero-Allocation Workspace Architecture

Iterative algorithms traditionally allocate temporary volumes and sinogram residuals at each iteration. `Tigre.jl` introduces pre-allocation APIs (`prepare_art_plan`, `prepare_first_order_workspace`, `prepare_krylov_workspace`) that allow users to allocate intermediate GPU and host memory once, enabling zero-allocation reconstruction loops ideal for real-time and time-series CT.

---

## Upstream Build Policy

This repository does **not** vendor the upstream `CERN/TIGRE` source tree inside Git.

Instead, `Tigre.jl` uses a reproducible build process (`deps/build.jl`):
1. Upstream source code is fetched from GitHub at a strictly pinned commit hash (`deps/upstream.toml`).
2. Local integration shims (`deps/wrapper/`) and necessary patches (`deps/patches/`) are applied.
3. The shared library `libtigre_julia` is compiled on the host machine using NVIDIA `nvcc`.

This architecture keeps the repository lightweight, clearly separates the Julia port from the upstream source, and ensures reproducible builds.

---

## Academic Citation

If you use `Tigre.jl` or TIGRE algorithms in your research, please cite the foundational publications:

- **Original TIGRE Paper**:
  Ander Biguri, Manjit Dosanjh, Steven Hancock, and Manuchehr Soleimani.
  *"TIGRE: a MATLAB-GPU toolbox for reproducible computed tomography, fast and accurate"*,
  **Biomedical Physics & Engineering Express**, 2(5): 055010, 2016.
  [DOI: 10.1088/2057-1976/2/5/055010](https://doi.org/10.1088/2057-1976/2/5/055010)

- **TIGRE 2.0 Paper (Multi-GPU & Trajectories)**:
  Ander Biguri, Reuben Lindroos, Robert Bryll, Hossein Towsyfyan, et al.
  *"Arbitrary trajectory and multi-GPU accelerated 3D tomographic reconstruction: TIGRE 2.0"*,
  **Biomedical Physics & Engineering Express**, 6(4): 045018, 2020.
  [DOI: 10.1088/2057-1976/ab9dbd](https://doi.org/10.1088/2057-1976/ab9dbd)

### Upstream Resources
- **Upstream Repository**: [https://github.com/CERN/TIGRE](https://github.com/CERN/TIGRE)
- **Upstream Documentation**: [https://github.com/CERN/TIGRE#readme](https://github.com/CERN/TIGRE#readme)
