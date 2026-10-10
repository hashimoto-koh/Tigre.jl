# Tigre.jl

[![CI](https://github.com/hashimoto-koh/Tigre.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/hashimoto-koh/Tigre.jl/actions/workflows/ci.yml)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://hashimoto-koh.github.io/Tigre.jl/dev/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Tigre.jl is an unofficial Julia port of
[TIGRE: Tomographic Iterative GPU-based Reconstruction Toolbox](https://github.com/CERN/TIGRE).

This project is **not** affiliated with, endorsed by, or maintained by CERN,
the University of Bath, or the upstream TIGRE contributors.

TIGRE is originally developed by the University of Bath and CERN contributors
and is distributed under the BSD 3-Clause License.

This repository exists because the original `TIGRE` project is an outstanding, highly performant, and widely used toolkit in computed tomography research. `Tigre.jl` is developed with explicit respect for that foundational work, and with sincere gratitude to the original authors and maintainers for making TIGRE open source and advancing accessible tomographic reconstruction.

---

## What This Repository Is

`Tigre.jl` provides a Julia-first interface directly to TIGRE's native C/CUDA projection and backprojection engines:

- **Zero Python or MATLAB dependencies**: Julia calls the compiled native C/CUDA kernels directly via a lightweight C ABI shim (`ccall`). Neither `PyCall`/`PythonCall` nor the MATLAB runtime is required.
- **Pure Julia algorithm implementations**: Analytical reconstruction (`fdk`, `fbp`), algebraic solvers (`sart`, `ossart`, `sirt`, etc.), statistical solvers (`mlem`), proximal methods (`fista`), and Krylov solvers (`cgls`, `lsmr`, `ab_gmres`, etc.) are written directly in Julia, providing full type safety, flexibility, and zero runtime overhead.
- **Canonical Julia array layout**: Eliminates hidden array copies or transpositions on the GPU hot path by matching column-major storage (`volume[x, y, z]` and `projection[u, v, angle]`).
- **Zero-allocation workspace APIs**: Buffer and weight pre-allocation mechanisms tailored for high-throughput workflows and time-series CT reconstruction.

## Upstream Policy

This repository does **not** vendor the upstream `TIGRE` source tree in Git.

Instead, `deps/build.jl` fetches the upstream C/CUDA sources at build time from a pinned upstream revision (`deps/upstream.toml`), applies minimal integration shims and patches, and compiles the shared library `libtigre_julia` locally on your system.

This policy maintains a transparent, reproducible build story and a clear boundary between upstream TIGRE and the Julia package.

---

## Features

- **Canonical Julia Memory Layout**: Fully column-major `volume[x, y, z]` and `projection[u, v, angle]`, eliminating hidden memory permutations in the CUDA hot path.
- **Analytical & Direct Reconstruction**: Filtered Backprojection (`fbp` for parallel beam) and Feldkamp-Davis-Kress (`fdk` for cone beam) with Parker redundancy weighting and Wang offset-detector padding.
- **Iterative & Algebraic Solvers**: Comprehensive suite of iterative algorithms:
  - ART / SIRT / SART / OS-SART (with Total Variation regularization variants)
  - ASD-POCS / OS-ASD-POCS / AW-ASD-POCS
  - PICCS / OS-PICCS
  - PCSD / AW-PCSD / OS-PCSD / OS-AW-PCSD
  - MLEM
  - ISTA / FISTA with 3D Total Variation proximal operator
  - Krylov subspace solvers: CGLS, LSQR, LSMR, AB-GMRES, BA-GMRES, Hybrid LSQR, Hybrid FLSQR-TV, IRN-TV-CGLS
- **Zero-Allocation Workspace Reuse**: Pre-allocated execution plans (`prepare_art_plan`, `prepare_first_order_workspace`, `prepare_krylov_workspace`) to eliminate GC overhead in iterative loops.
- **Built-in Phantoms**: Analytical 3D Shepp-Logan phantom generator (`shepp_logan_3d`).
- **Upstream-Compatible Visualization**: `plotImg` and `plotProj` utilities via optional Makie extension (`using CairoMakie` or `GLMakie`), supporting 2D/3D orthogonal slicing and slice movie export (`.gif`, `.mp4`).
- **Interactive Demos**: Ready-to-run Jupyter notebooks under `demos/` covering 2D/3D parallel and cone-beam workflows.

---

## Canonical Memory Layout

Tigre.jl uses Julia's native 1-based indexing and contiguous column-major storage:

- **Volume**: `volume[x, y, z]`
- **Projection**: `projection[u, v, angle]`
- **Angles**: `angles[component, angle]`
  - `component = 1: alpha` (roll / rotation around Y)
  - `component = 2: theta` (pitch / rotation around X)
  - `component = 3: psi` (yaw / gantry angle around Z)

### Correspondence with Python TIGRE

| Data Structure | Python TIGRE (Row-major) | Tigre.jl (Column-major) |
| :--- | :--- | :--- |
| **Volume** | `(z, y, x)` | `(x, y, z)` |
| **Projection** | `(angle, v, u)` | `(u, v, angle)` |
| **Angles** | `(angle, 3)` | `(3, angle)` |

Because the memory buffers match the physical axis ordering expected by TIGRE's underlying CUDA texture engines, no transposition or memory copying is performed during projection or backprojection.

---

## Installation & Build

### Prerequisites

Building `Tigre.jl` requires an NVIDIA GPU and native compilation tools:

- **NVIDIA GPU**: CUDA-capable GPU (Compute Capability 3.5 or higher recommended).
- **NVIDIA Driver**: Driver version compatible with your installed CUDA Toolkit.
- **NVIDIA CUDA Toolkit**: `nvcc` compiler must be accessible in your `PATH` (or specified via `CUDA_NVCC` / `CUDA_HOME`).
- **C++ Host Compiler**: A compatible C++ host compiler (e.g. `g++` on Linux, `MSVC` on Windows, or `clang++`).
- **Git**: Required by `Pkg.build` to clone the pinned upstream TIGRE repository sources.

> [!IMPORTANT]
> Because `Tigre.jl` builds the native CUDA C++ kernels (`libtigre_julia`) directly on your machine via `deps/build.jl`, `nvcc` and the NVIDIA CUDA Toolkit must be installed before running `Pkg.build("Tigre")`.

### Standard Build

Clone the repository and build the native C/CUDA shared library:

```bash
julia --startup-file=no --project=. -e 'import Pkg; Pkg.develop(path=pwd()); Pkg.build("Tigre")'
```

### Build Mechanics

Running `Pkg.build("Tigre")` automatically:
1. Reads `deps/upstream.toml` for the upstream repository URL and pinned Git commit (`6b0951a8a88aa88a7db5b2a95181a33a95aa8b28`).
2. Clones/fetches upstream `CERN/TIGRE` into `deps/src/TIGRE` (if not already cached).
3. Applies Julia wrapper shims and minimal overlays from `deps/wrapper/`.
4. Compiles `deps/usr/lib/libtigre_julia.$(Libdl.dlext)` with `nvcc`.
5. Writes build metadata to `deps/usr/build-info.toml`.

### Build Environment Variables

| Variable | Description |
| :--- | :--- |
| `CUDA_NVCC` | Path to custom `nvcc` executable |
| `CUDA_HOME` / `CUDA_PATH` | Path to CUDA installation root directory |
| `CUDA_INCLUDE_DIR` | Path to CUDA headers |
| `CUDA_LIB_DIR` | Path to CUDA libraries (`libcudart`, etc.) |
| `TIGRE_CUDA_ARCH_FLAGS` | Specific CUDA compute architecture flags (e.g. `-arch=sm_80`) |
| `TIGRE_EXTRA_NVCC_FLAGS` | Additional compilation flags passed to `nvcc` |
| `TIGRE_BUILD_FORCE=1` | Force clean recompile of native library |
| `TIGRE_BUILD_FETCH_FORCE=1` | Force re-fetching upstream source code |
| `TIGRE_BUILD_VERIFY_ONLY=1` | Only inspect staleness without compiling |
| `TIGRE_BUILD_DRY_RUN=1` | Print `nvcc` compilation commands without running them |

### Build Status Inspection

Inside Julia, you can verify the status of the native library:

```julia
using Tigre

# Detailed build status struct
status = library_status()
println(status)

# Boolean check
native_library_available()

# Check for stale compilation
native_build_stale()
native_build_stale_reasons()
```

Or from the command line using the diagnostic utility:

```bash
julia --startup-file=no --project=. doctor.jl --strict
julia --startup-file=no --project=. doctor.jl --json --strict
```

The `--json` output conforms to schema `schema="tigre-doctor-v1"`, `schema_version=1`.

---

## Quick Start

```julia
using Tigre

# 1. Setup acquisition geometry (Cone-beam CT)
angles = Float32.(range(0, 2pi, length=180))
geo = Geometry(
    mode = :cone,
    angles = angles,
    nVoxel = (128, 128, 128),
    sVoxel = (128f0, 128f0, 128f0),
    nDetector = (192, 192),
    sDetector = (192f0, 192f0),
    DSO = 1000f0, # Source-to-Origin (mm)
    DSD = 1500f0, # Source-to-Detector (mm)
)

# 2. Generate a 3D Shepp-Logan phantom
phantom = shepp_logan_3d(128; phantom_type = :yu_ye_wang)

# 3. Forward projection (GPU line integrals)
proj = Ax(phantom, geo)

# 4. Direct reconstruction (FDK)
rec_fdk = fdk(proj, geo; filter = :ram_lak)

# 5. Iterative reconstruction (FISTA with Total Variation regularization)
rec_fista = fista(proj, geo, 20; hyper = 2.0f8, tviter = 20, tvlambda = 0.1f0)

# 6. Visualize reconstruction (requires `using CairoMakie` or `GLMakie`)
using CairoMakie
plotImg(rec_fdk; dim = :z, slice = 64, title = "FDK Reconstruction")
```

---

## Interactive Jupyter Notebook Demos

Interactive demonstration notebooks are available in `demos/`:

- [`demos/01_2D_parallel_beam.ipynb`](demos/01_2D_parallel_beam.ipynb): 2D Parallel-Beam CT demo (`Ax` $\to$ `fbp` $\to$ `fista`).
- [`demos/02_2D_cone_beam.ipynb`](demos/02_2D_cone_beam.ipynb): 2D Cone-Beam (fan-beam) CT demo (`Ax` $\to$ `fdk` $\to$ `fista`).
- [`demos/03_3D_parallel_beam.ipynb`](demos/03_3D_parallel_beam.ipynb): 3D Parallel-Beam CT demo (`Ax` $\to$ `fbp` $\to$ `fista`).
- [`demos/04_3D_cone_beam.ipynb`](demos/04_3D_cone_beam.ipynb): 3D Cone-Beam CT (CBCT) demo (`Ax` $\to$ `fdk` $\to$ `fista`).
- [`demos/05_iterative_comparison.ipynb`](demos/05_iterative_comparison.ipynb): Comparison of iterative algorithms (FDK, SART, CGLS, FISTA) on sparse-angle data.
- [`demos/06_krylov_solvers.ipynb`](demos/06_krylov_solvers.ipynb): Krylov subspace solvers: CGLS, LSQR, LSMR, and hybrid methods.
- [`demos/07_preprocessing.ipynb`](demos/07_preprocessing.ipynb): CT projection preprocessing and calibration pipeline.
- [`demos/08_offset_detector_fov.ipynb`](demos/08_offset_detector_fov.ipynb): Offset detector geometry and extended field-of-view (FOV) reconstruction.
- [`demos/09_laminography_conebeam.ipynb`](demos/09_laminography_conebeam.ipynb): Cone-Beam Computed Laminography (CBCL) for planar / PCB inspection.
- [`demos/10_emission_mlem_osem.ipynb`](demos/10_emission_mlem_osem.ipynb): Statistical emission tomography (PET / SPECT) with MLEM and OSEM.
- [`demos/11_low_dose_regularization.ipynb`](demos/11_low_dose_regularization.ipynb): Low-dose CT and advanced regularization (TV, ASD-POCS, AW-ASD-POCS).
- [`demos/12_sparse_view_cs.ipynb`](demos/12_sparse_view_cs.ipynb): Sparse-view CT and compressed sensing algorithms (ASD-POCS, PICCS).
- [`demos/13_workspace_reuse.ipynb`](demos/13_workspace_reuse.ipynb): Zero-allocation iterative reconstruction using pre-allocated workspaces.

---

## Documentation

Full documentation is available in `docs/`:

To build the HTML documentation locally:

```bash
julia --startup-file=no --project=docs docs/make.jl
```

The generated HTML site will be placed in `docs/build/`.

---

## Testing

Run the test suite:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

- Geometry and array layout validation tests run in pure Julia even when CUDA/`nvcc` is not present.
- When the compiled native library is present, numerical regression tests against precomputed fixtures verify:
  `Ax`, `Atb`, `fdk` (with Hann filter and Wang weighting), `sart`, `fista`, `lsmr`, `hybrid_flsqr_tv`, and `piccs`.
- To update test regression fixtures when algorithms change intentionally:
  ```bash
  julia --startup-file=no --project=. test/update_regression_reference.jl
  ```

---

## Public API Policy & Contracts

- Names exported by `using Tigre` constitute the official public API.
  This includes `Geometry`, `Ax` / `Ax!`, `Atb` / `Atb!`, `fdk` / `fdk!`, `fbp` / `fbp!`, `shepp_logan_3d`, iterative solvers (`sart`, `fista`, `cgls`, etc.), workspace pre-allocation APIs (`prepare_*`), and diagnostic utilities (`doctor_report`, `library_status`).
- `test/runtests.jl` enforces exported surface constancy; unintentional additions or removals are caught as regressions.
- Non-exported helper functions and functions prefixed with `_` are private and subject to change.

### Default Solver Contracts

- `sart` is fixed as `ossart(blocksize=1)`, and `sirt` as `ossart(blocksize=nangles)`.
- Default parameters for `ossart` follow upstream Python: `lambda=1`, `lambda_red=1`, `order_strategy=:ordered`, `nonneg=true`, `projection_type=:siddon`, `backprojection_type=:fdk`.
- `sart_tv` / `ossart_tv` default to `tviter=50`, `tvlambda=50`.
- `asd_pocs` / `awasd_pocs` / `piccs` default to `lambda=1`, `lambda_red=0.99`, `alpha=0.002`, `alpha_red=0.95`, `rmax=0.95`, `tviter=20`. Additionally, `piccs` defaults to `prior_ratio=0.2`, and `awasd_pocs` defaults to `delta=-0.005`.
- `mlem` starts from flat image value `1` by default, with `projection_type=:siddon` and `backprojection_type=:matched`.
- `ista` / `fista` default to `hyper=2e8`, `tviter=20`, `tvlambda=0.1`, `projection_type=:siddon`, and `backprojection_type=:matched`.
- Krylov solvers (`cgls`, `lsqr`, `lsmr`, `hybrid_lsqr`, `hybrid_flsqr_tv`, `ab_gmres`, `ba_gmres`, `irn_tv_cgls`) default to `projection_type=:siddon` and `backprojection_type=:matched` to preserve operator adjointness, with `stop_on_l2_increase=true` as a divergence guard.
- All iterative solvers accept `show_progress::Bool = false`. When set to `true`, a progress meter displays iteration count, elapsed time, and ETA.

---

## Known Intentional Divergences from Upstream Python

1. **`hybrid_flsqr_tv` Inner Solver**:
   Upstream TIGRE's Python implementation invokes SciPy's `lsqr` for its inner solve. In order to avoid any Python runtime dependency, `Tigre.jl` replaces this with a weighted TV small-scale Conjugate Gradient solver `(I + λ R'R) z = v`. The outer loop role, interface, and workspace structure remain identical.
2. **`iterativereconalg` Class Wrapper**:
   In Python TIGRE, `iterativereconalg` is a decorator wrapper around an `IterativeReconAlg` base class. In Julia, this is superseded by the idiomatic `ossart(projection, geo, niter; blocksize=20)` function and the `prepare_art_plan` workspace API.

---

## Benchmarks & Performance

Benchmark entry points are located in `benchmark/`:

```bash
julia --startup-file=no --project=. benchmark/vertical_slice.jl
julia --startup-file=no --project=. benchmark/reuse_paths.jl
julia --startup-file=no --project=. benchmark/representative_geometry_suite.jl
julia --startup-file=no --project=. benchmark/generate_benchmark_svgs.jl --output-dir /tmp
```

Pre-allocated workspace paths (`prepare_first_order_workspace`, `prepare_art_plan`, `prepare_krylov_workspace`) reduce allocations in iterative loops down to negligible levels (e.g. ~1 KB per iteration for FISTA/ISTA), significantly reducing Julia GC pauses in high-throughput workloads.

---

## Repository Hygiene

- **Tracked in Git**: `src/`, `test/`, `demos/`, `docs/`, `examples/`, `benchmark/`, `deps/build.jl`, `deps/upstream.toml`, `deps/wrapper/`, `Project.toml`, `README.md`, `doctor.jl`.
- **Ignored / Ephemeral**:
  - `deps/src/` (cloned upstream TIGRE repository)
  - `deps/usr/` (compiled shared libraries and `build-info.toml`)
  - `docs/build/` (generated documentation site)
  - `deps/build.log` (local build logs)

The upstream revision is strictly pinned in `deps/upstream.toml` to guarantee reproducible builds across environments.

---

## Related Julia Packages

The following Julia packages and organizations may be useful for users
interested in tomography, image reconstruction, phantoms, sinograms, and
computational imaging. This list is not exhaustive.

### Related packages from this porting effort

- [AstraToolbox.jl](https://github.com/hashimoto-koh/AstraToolbox.jl) — unofficial Julia port of [ASTRA Toolbox](https://github.com/astra-toolbox/astra-toolbox).
- [TomoPhantom.jl](https://github.com/hashimoto-koh/TomoPhantom.jl) — unofficial Julia port of [TomoPhantom](https://github.com/dkazanc/tomophantom).

### Other related packages

- [JuliaImageRecon](https://github.com/JuliaImageRecon) — Julia packages for image reconstruction and related tools.
- [Sinograms.jl](https://github.com/JuliaImageRecon/Sinograms.jl) — tools for sinograms, tomography, Radon transforms, and CT/CBCT geometry.
- [ImagePhantoms.jl](https://github.com/JuliaImageRecon/ImagePhantoms.jl) — software phantoms and analytical Radon/X-ray transforms.
- [MIRT.jl](https://github.com/JeffFessler/MIRT.jl) — Julia version of the Michigan Image Reconstruction Toolbox.
- [RadonKA.jl](https://github.com/roflmaostc/RadonKA.jl) — Radon transforms with multithreading and CUDA support.

Contributions to this list are welcome via issues or pull requests.

---

## Acknowledgment & Citation

`Tigre.jl` stands on the shoulders of the original TIGRE project. If you use this software in academic or scientific research, please cite the foundational TIGRE papers:

- Ander Biguri, Manjit Dosanjh, Steven Hancock, and Manuchehr Soleimani.
  *"TIGRE: a MATLAB-GPU toolbox for reproducible computed tomography, fast and accurate"*,
  **Biomedical Physics & Engineering Express**, 2(5): 055010, 2016.
  [DOI: 10.1088/2057-1976/2/5/055010](https://doi.org/10.1088/2057-1976/2/5/055010)

- Ander Biguri, Reuben Lindroos, Robert Bryll, Hossein Towsyfyan, et al.
  *"Arbitrary trajectory and multi-GPU accelerated 3D tomographic reconstruction: TIGRE 2.0"*,
  **Biomedical Physics & Engineering Express**, 6(4): 045018, 2020.
  [DOI: 10.1088/2057-1976/ab9dbd](https://doi.org/10.1088/2057-1976/ab9dbd)

For the original project, documentation, and upstream implementation, see:
- Upstream repository: <https://github.com/CERN/TIGRE>
- Upstream documentation: <https://github.com/CERN/TIGRE#readme>

---

## License & Third-Party Notices

This package and wrapper are distributed under the BSD 3-Clause License. See [`LICENSE.md`](LICENSE.md) for details.
Upstream copyright, license terms, and third-party notices are documented in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
