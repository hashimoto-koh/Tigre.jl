# Tigre.jl Documentation

**Tigre.jl** is a high-performance Julia interface to the [CERN/TIGRE](https://github.com/CERN/TIGRE) (Tomographic Iterative GPU-based Reconstruction Toolbox) CT reconstruction engine.

It combines CUDA-accelerated projection and backprojection primitives with a comprehensive suite of Julia-native analytical and iterative reconstruction algorithms.

---

## Key Highlights

- **Zero-Copy Native Array Layout**:
  - Volume arrays are canonical column-major 3D arrays: `volume[x, y, z]` (X innermost).
  - Projection arrays follow: `projection[u, v, angle]` (detector U innermost).
  - Angles matrix: `angles[component, angle]`.
  - Memory pointers are passed directly into CUDA kernels via a clean C ABI shim without costly dimension transpositions or copy operations.
- **Pure Julia Algorithms (Zero Python / MATLAB Dependencies)**:
  - All high-level algorithms are implemented directly in Julia with native performance and clear customization paths.
  - Upstream methods that traditionally required SciPy (such as Krylov inner solvers) are replaced with native Julia solvers.
- **Rich Algorithm Catalog (28+ Solvers)**:
  - **Analytical / Direct**: `fdk`, `fbp` (with Ram-Lak, Shepp-Logan, Cosine, Hamming, Hann windowed filters).
  - **Algebraic / Subsets**: `sart`, `sirt`, `ossart`, `sart_tv`, `ossart_tv`.
  - **POCS & TV Minimization**: `asd_pocs`, `os_asd_pocs`, `awasd_pocs`, `os_awasd_pocs`.
  - **Statistical**: `mlem`.
  - **First-Order Proximal**: `ista`, `fista`.
  - **PICCS & PCSD Family**: `piccs`, `os_piccs`, `pcsd`, `aw_pcsd`, `os_pcsd`, `os_aw_pcsd`.
  - **Krylov Subspace Methods**: `cgls`, `lsqr`, `lsmr`, `hybrid_lsqr`, `hybrid_flsqr_tv`, `ab_gmres`, `ba_gmres`, `irn_tv_cgls`.
- **Zero-Allocation Workspace Reuse**:
  - `prepare_art_plan`, `prepare_first_order_workspace`, and `prepare_krylov_workspace` allow pre-allocating intermediate GPU/CPU buffers and weights for ultra-low latency dynamic and time-series CT reconstruction.
- **Extended Field of View & Geometric Offsets**:
  - Native support for detector offsets, roll, pitch, yaw, center-of-rotation (COR) correction, Parker weighting, and Wang padding/preweighting.
- **Upstream-Compatible Visualization & Slice Movies**:
  - Built-in `plotImg` and `plotProj` utilities matching upstream TIGRE via optional Package Extension (`using CairoMakie` or `using GLMakie`).
  - Supports 3D orthogonal slicing (axial, coronal, sagittal), 2D views, and high-speed slice animation / movie export (`.gif`, `.mp4`).

---

## Upstream Credits & Gratitude

`Tigre.jl` builds upon the pioneering C++/CUDA core developed by Ander Biguri, the University of Bath, CERN, and the TIGRE team. We express our sincere gratitude to the original authors for developing and maintaining the open-source TIGRE toolbox.

For architectural background, details on the relationship between Tigre.jl and upstream TIGRE, and formal academic citations, please see [About TIGRE & Upstream](about_tigre.md).

- **TIGRE Repository**: [https://github.com/CERN/TIGRE](https://github.com/CERN/TIGRE)
- **Primary Publication**: Biguri et al., *"TIGRE: a MATLAB-GPU toolbox for reproducible computed tomography, fast and accurate"*, *Biomed. Phys. Eng. Express* 2(5): 055010 (2016).

---

## Quick Navigation

- [About TIGRE & Upstream](about_tigre.md) - Project history, relationship to upstream TIGRE, and citations.
- [Installation Guide](installation.md) - System requirements, CUDA setup, and build troubleshooting.
- [Quick Start](quickstart.md) - Minimal working examples of forward projection and reconstruction.
- [Geometry Specification](geometry.md) - Coordinate systems, Cone-beam / Parallel-beam geometry setup, and detector offsets.
- [Reconstruction Algorithms](algorithms.md) - Complete guide to all 28+ available solvers.
- [Workspace & Plan Reuse](workspace_reuse.md) - Accelerating time-series reconstruction via pre-allocated workspaces.
- [Visualization](visualization.md) - Guide to `plotImg`, `plotProj`, and slice movie generation with Makie.
- [API Reference](api.md) - Comprehensive function index and signatures.
