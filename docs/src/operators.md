# Operators & Weights

`Tigre.jl` provides low-level projection, backprojection, filtering, and TV-denoising primitives that directly interface with high-performance CUDA kernels.

---

## Forward Projection (`Ax`, `Ax!`)

The forward projection operator computes line integrals (radon transform) through the volume:

```julia
# Allocating form
projection = Ax(volume, geo; projection_type = :siddon, gpu_ids = nothing)

# In-place form (mutates destination projection array)
Ax!(projection, volume, geo; projection_type = :siddon, gpu_ids = nothing)
```

### Projection Methods (`projection_type`)
- `:siddon` (default): Fast Siddon ray-tracing algorithm.
- `:interpolated`: Trilinear ray-sampling algorithm.

---

## Backprojection (`Atb`, `Atb!`)

The backprojection operator maps detector projection data back into 3D voxel space:

```julia
# Allocating form
volume = Atb(projection, geo; backprojection_type = :matched, gpu_ids = nothing)

# In-place form
Atb!(volume, projection, geo; backprojection_type = :matched, gpu_ids = nothing)
```

### Backprojection Types (`backprojection_type`)
- `:matched`: Matched transpose of the forward operator ($A^T$). Essential for exact adjointness in Krylov subspace and gradient-based iterative methods.
- `:fdk`: FDK-weighted backprojection (with cone-beam distance weighting), used inside analytical FDK/FBP algorithms.

---

## Total Variation (TV) Denoising (`tvdenoise`, `tvdenoise!`)

`Tigre.jl` provides GPU-accelerated 3D Total Variation regularization:

```julia
# Allocating form
denoised = tvdenoise(volume; lambda = 15.0f0, spacing = (1f0, 1f0, 1f0), maxiter = 100, gpu_ids = nothing)

# In-place form
tvdenoise!(dst, src; lambda = 15.0f0, spacing = (1f0, 1f0, 1f0), maxiter = 100, gpu_ids = nothing)
```

### Parameters
- `lambda`: TV regularization parameter.
- `spacing`: Physical voxel spacing `(dx, dy, dz)`.
- `maxiter`: Number of steepest descent iterations for the TV functional.

---

## Projection Filtering (`filter_projections!`)

FBP and FDK require ramp filtering in the frequency domain:

```julia
filter_projections!(projection, geo; filter = :hann, parker = false)
```

### Available Windowed Filters
- `:ram_lak`: Pure ramp filter (standard high-pass).
- `:shepp_logan`: Sinc-windowed ramp filter (attenuates high frequencies).
- `:cosine`: Cosine-windowed ramp filter.
- `:hamming`: Hamming-windowed ramp filter.
- `:hann`: Hann-windowed ramp filter (smooth, low noise).

Inspect frequency response:
```julia
resp = filter_response(:hann, 128)
```

---

## Weighting Utilities

### Parker Weighting (`parkerweight!`)
Short-scan redundancy weighting for arcs covering $\pi + 2\gamma_m$:

```julia
parkerweight!(projection, geo)
```

### Wang Weighting & Padding (`wang_padding`, `wang_preweight!`)
For laterally shifted (offset) flat-panel detectors:

```julia
padded_proj, padded_geo, theta = wang_padding(projection, geo)
wang_preweight!(padded_proj, padded_geo, theta)
```

---

## Multi-GPU Support

Pass `gpu_ids` as a vector of integer device IDs (0-indexed):

```julia
Ax!(proj, vol, geo; gpu_ids = [0, 1])
```
