# Contributing to Tigre.jl

Thank you for your interest in contributing to `Tigre.jl`!

`Tigre.jl` is an open-source Julia port of [CERN/TIGRE](https://github.com/CERN/TIGRE) (Tomographic Iterative GPU-based Reconstruction Toolbox). Our goal is to provide a pure Julia algorithm layer with zero Python/MATLAB dependencies, backed directly by upstream's high-performance C/CUDA projection and backprojection engines.

---

## Code of Conduct & Ground Rules

- Be respectful, constructive, and collaborative in discussions, issues, and code reviews.
- Adhere to the core architectural principles outlined below.

---

## Reporting Issues

If you find a bug, encounter a compilation failure, or have a feature request, please search existing issues before opening a new one.

When filing a bug report, please include:
1. **System & Environment Information**:
   - Julia version (`julia --version`)
   - OS and distribution
   - GPU model and NVIDIA driver version
   - CUDA Toolkit version (`nvcc --version`)
   - Output of diagnostic doctor script:
     ```bash
     julia --startup-file=no --project=. doctor.jl
     ```
2. **Minimal Reproducible Example (MRE)**:
   - Self-contained Julia code snippet demonstrating the failure or unexpected behavior.
3. **Full Stack Trace**:
   - Uncut error message and Julia backtrace.

---

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hashimoto-koh/Tigre.jl.git
   cd Tigre.jl
   ```

2. **Build the native C/CUDA library**:
   Ensure `nvcc` is in your `PATH`, then build the package:
   ```bash
   julia --startup-file=no --project=. -e 'import Pkg; Pkg.develop(path=pwd()); Pkg.build("Tigre")'
   ```

3. **Verify build status**:
   ```bash
   julia --startup-file=no --project=. doctor.jl --strict
   ```

4. **Run the test suite**:
   ```bash
   julia --startup-file=no --project=. -e 'import Pkg; Pkg.test()'
   ```
   or directly:
   ```bash
   julia --startup-file=no --project=. test/runtests.jl
   ```

---

## Architectural Principles & Guidelines

When submitting pull requests, please follow these core guidelines:

### 1. Zero Python / MATLAB Dependencies
- `Tigre.jl` must never rely on `PyCall`, `PythonCall`, or MATLAB runtime wrappers.
- All high-level algorithms, geometry handling, and iterative solvers must be implemented in pure Julia.

### 2. Respect Upstream C/CUDA Kernels
- Do not re-implement upstream's core projection, backprojection, or TV kernels in Julia unless explicitly discussed.
- Upstream C/CUDA sources are fetched at build time based on the pinned commit in `deps/upstream.toml`.
- Any patches or wrappers under `deps/patches/` and `deps/wrapper/` must be minimal and clearly documented.

### 3. Idiomatic Julia API Design
- Provide both non-allocating in-place mutating functions (e.g., `fdk!`, `fista!`) and convenient non-mutating wrappers (e.g., `fdk`, `fista`).
- Support workspace pre-allocation mechanisms (`prepare_first_order_workspace`, `prepare_krylov_workspace`, `prepare_art_plan`) to ensure zero heap allocations in iterative loops.
- Use canonical column-major array layouts (`volume[x, y, z]`, `projection[u, v, angle]`) without internal dimension flipping.
- Preserve operator adjointness: inner products and L2 norm accumulations must maintain mathematical consistency (using `Float64` accumulators where appropriate to prevent precision loss).

### 4. Optional Extensions
- Heavy visualization dependencies (such as Makie) must reside in package extensions (e.g., `ext/TigreMakieExt.jl`), keeping the core package lightweight and headless-friendly.

---

## Pull Request Process

1. **Keep PRs Focused**:
   - Prefer small, focused pull requests addressing a single bug fix or feature over large monolithic changes.
2. **Add Tests**:
   - Every bug fix should include a regression test.
   - New solvers or geometry features must have unit tests and, where applicable, convergence checks or numerical fixtures.
3. **Document Public APIs**:
   - All exported functions and structs must have comprehensive Julia docstrings formatted with `# Arguments`, `# Returns`, and `# Examples`.
   - List default values clearly with `(default: ...)` and do not hide kwargs behind unspecified `kwargs...`.
   - Update user documentation under `docs/` and API index in `docs/src/api.md` if public surfaces change.
4. **Verify Documentation Build**:
   ```bash
   julia --startup-file=no --project=docs docs/make.jl
   ```
5. **Ensure Tests Pass**:
   - Run `julia --startup-file=no --project=. -e 'import Pkg; Pkg.test()'` locally before opening the PR.
