# Installation Guide

This page explains how to install, build, and verify `Tigre.jl`.

---

## System Requirements

- **Operating System**: Linux (x86_64 or aarch64).
- **Julia**: Version 1.11 or later.
- **NVIDIA GPU**: CUDA-capable GPU (Compute Capability $\ge 5.0$, Maxwell or newer).
- **CUDA Toolkit**: CUDA 11.0 or newer (providing `nvcc` and `libcudart.so`).
- **Build Tools**: `git`, standard C++ toolchain (`g++` / `clang++` compatible with nvcc).

---

## Installing the Package

Install `Tigre.jl` using Julia's package manager:

```julia
using Pkg
Pkg.add(url = "https://github.com/hashimoto-koh/Tigre.jl.git")
```

Or for local development:

```julia
using Pkg
Pkg.develop(path = "/path/to/Tigre.jl")
```

---

## Building the Native CUDA Backend

During installation (or via `Pkg.build("Tigre")`), the build script `deps/build.jl` automatically:
1. Clones the upstream CERN/TIGRE repository at a pinned, verified revision.
2. Applies patches (e.g. Siddon non-intersecting ray fast return, C++ exception wrappers).
3. Compiles the C++/CUDA core into `deps/usr/lib/libtigre_julia.so`.
4. Writes compilation metadata and timestamps to `deps/usr/build-info.toml`.

To trigger the native build manually from a terminal:

```bash
julia --startup-file=no --project=. -e 'import Pkg; Pkg.build("Tigre")'
```

Or run the build script directly:

```bash
julia --startup-file=no --project=. deps/build.jl
```

---

## Verifying Your Installation: `doctor.jl`

`Tigre.jl` includes an automated diagnostics utility:

```bash
julia --startup-file=no --project=. doctor.jl
```

Example output:
```text
Tigre.jl doctor
library:             available
stale:               false
stale reasons:       none
build info:          available
build timestamp:     2026-09-25T20:25:33Z
gpu probe:           ok
gpus:                [0: NVIDIA GeForce RTX ...]
status:              ready
hint:                native build is available and fresh
```

For automated scripts or CI, pass `--json` to retrieve the report in JSON format:

```bash
julia --startup-file=no --project=. doctor.jl --json
```

---

## Environment Variables

You can customize compilation and device detection using the following environment variables:

| Variable | Description | Default |
|:---|:---|:---|
| `CUDA_HOME` / `CUDA_PATH` | Path to CUDA installation root directory | `/usr/local/cuda` |
| `CUDA_NVCC` | Explicit path to the `nvcc` compiler binary | `$(CUDA_HOME)/bin/nvcc` or `which nvcc` |
| `CUDA_LIB_DIR` | Directory containing `libcudart.so` | `$(CUDA_HOME)/lib64` |
| `TIGRE_CUDA_ARCH_FLAGS` | Custom GPU architecture flags for nvcc | Detected automatically via `nvcc --list-gpu-arch` |
| `TIGRE_BUILD_FORCE` | Force re-compilation even if output is fresh | `0` |
| `TIGREJL_UPSTREAM_SOURCE` | Upstream Git repository URL | `https://github.com/CERN/TIGRE.git` |
| `TIGREJL_UPSTREAM_REF` | Upstream Git branch or commit hash | Pinned stable hash |

---

## Troubleshooting

- **`nvcc not found`**: Ensure CUDA bin directory (e.g. `/usr/local/cuda/bin`) is in your system `PATH`, or set `CUDA_HOME=/usr/local/cuda`.
- **`stale build reported`**: Run with `TIGRE_BUILD_FORCE=1 julia --project=. deps/build.jl` to refresh the build artifacts.
- **Multiple GPUs**: Specify the target GPU ID (0-indexed) via `gpu_ids = [0]` in operator or algorithm calls.
