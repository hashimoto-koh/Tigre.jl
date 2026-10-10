function _normalize_spacing(spacing)
    data = collect(spacing)
    length(data) == 3 || throw(ArgumentError("spacing must have length 3"))
    return (Float32(data[1]), Float32(data[2]), Float32(data[3]))
end

"""
    tvdenoise(volume::AbstractArray{Float32, 3}; lambda::Real = 15f0, spacing = (1f0, 1f0, 1f0), maxiter::Integer = 100, gpu_ids = nothing) -> Array{Float32, 3}

3D Total Variation (TV) proximal denoising via GPU Chambolle/Fast Gradient Projection (FGP) algorithm matching upstream TIGRE `im3DDenoise`.

Following upstream TIGRE convention, the input volume is min-max normalized to ``[0, 1]`` before invoking the CUDA proximal solver, and scaled back to its original intensity range afterward:
```math
\\min_{u \\in [0, 1]} \\frac{\\lambda}{2} \\|u - \\tilde{f}\\|_2^2 + \\mathrm{TV}(u)
```
where ``\\tilde{f} = (f - \\min f) / (\\max f - \\min f)`` and ``\\lambda`` is the fidelity weight parameter (`lambda`).
Smaller values of `lambda` impose stronger TV regularization (more smoothing), while larger values preserve closer fidelity to the original volume.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Input noisy 3D image `(nx, ny, nz)`.

# Keyword Arguments
- `lambda::Real = 15f0`: TV fidelity parameter on ``[0, 1]``-normalized intensity (default: `15f0`). Lower values produce stronger TV smoothing.
- `spacing = (1f0, 1f0, 1f0)`: Physical voxel pitch `(dx, dy, dz)` in mm (default: `(1f0, 1f0, 1f0)`).
- `maxiter::Integer = 100`: Number of dual gradient ascent iterations (default: `100`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).

# Returns
- Denoised 3D volume `Array{Float32, 3}` of the same dimensions as `volume`.

# Examples
```julia
denoised = tvdenoise(recon; lambda = 50f0, maxiter = 50)
```
"""
function tvdenoise(volume::AbstractArray{Float32, 3}; lambda::Real = 15f0, spacing = (1f0, 1f0, 1f0), maxiter::Integer = 100, gpu_ids = nothing)
    dst = Array{Float32}(undef, size(volume))
    return tvdenoise!(dst, volume; lambda = lambda, spacing = spacing, maxiter = maxiter, gpu_ids = gpu_ids)
end

"""
    tvdenoise!(dst::AbstractArray{Float32, 3}, src::AbstractArray{Float32, 3}; lambda::Real = 15f0, spacing = (1f0, 1f0, 1f0), maxiter::Integer = 100, gpu_ids = nothing) -> dst

In-place 3D Total Variation (TV) proximal denoising from `src` into `dst` with upstream TIGRE ``[0, 1]`` min-max normalization.

# Arguments
- `dst::AbstractArray{Float32, 3}`: Pre-allocated destination array of size `(nx, ny, nz)`. May be identical to `src` for in-place denoising.
- `src::AbstractArray{Float32, 3}`: Input noisy 3D image `(nx, ny, nz)`.

# Keyword Arguments
- `lambda::Real = 15f0`: TV fidelity parameter on ``[0, 1]``-normalized intensity (default: `15f0`). Lower values produce stronger TV smoothing.
- `spacing = (1f0, 1f0, 1f0)`: Physical voxel pitch `(dx, dy, dz)` in mm (default: `(1f0, 1f0, 1f0)`).
- `maxiter::Integer = 100`: Number of dual gradient ascent iterations (default: `100`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).

# Returns
- The mutated `dst` array containing the denoised volume.

# Examples
```julia
tvdenoise!(recon, recon; lambda = 50f0, maxiter = 50)
```
"""
function tvdenoise!(dst::AbstractArray{Float32, 3}, src::AbstractArray{Float32, 3}; lambda::Real = 15f0, spacing = (1f0, 1f0, 1f0), maxiter::Integer = 100, gpu_ids = nothing)
    src_canonical = _require_canonical_array(src, "src", size(src))
    dst_canonical = _require_canonical_array(dst, "dst", size(src))

    vmin = minimum(src_canonical)
    vmax = maximum(src_canonical)
    vrange = vmax - vmin

    if vrange <= eps(Float32)
        if dst_canonical !== src_canonical
            copyto!(dst_canonical, src_canonical)
        end
        return dst
    end

    inv_range = 1f0 / vrange
    @inbounds for i in eachindex(dst_canonical, src_canonical)
        dst_canonical[i] = (src_canonical[i] - vmin) * inv_range
    end

    _native_tvdenoise!(dst_canonical, dst_canonical, Float32(lambda), _normalize_spacing(spacing), maxiter, gpu_ids)

    @inbounds for i in eachindex(dst_canonical)
        dst_canonical[i] = dst_canonical[i] * vrange + vmin
    end

    return dst
end

"""
    minTV(volume::AbstractArray{Float32, 3}; alpha::Real = 15f0, maxiter::Integer = 100, gpu_ids = nothing) -> Array{Float32, 3}

Total Variation gradient-descent minimization used in ASD-POCS and related algorithms.

Performs `maxiter` steps of gradient descent on the TV objective:
```math
x^{(k+1)} = x^{(k)} - \\alpha \\frac{\\nabla \\mathrm{TV}(x^{(k)})}{\\|\\nabla \\mathrm{TV}(x^{(k)})\\|}
```

# Arguments
- `volume::AbstractArray{Float32, 3}`: Input 3D volume.

# Keyword Arguments
- `alpha::Real = 15f0`: Step size scaling parameter (default: `15f0`).
- `maxiter::Integer = 100`: Iteration count (default: `100`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).

# Returns
- Allocated 3D volume `Array{Float32, 3}` after TV gradient minimization.

# Examples
```julia
tv_reduced = minTV(recon; alpha=5f0, maxiter=20)
```
"""
function minTV(volume::AbstractArray{Float32, 3}; alpha::Real = 15f0, maxiter::Integer = 100, gpu_ids = nothing)
    dst = Array{Float32}(undef, size(volume))
    return minTV!(dst, volume; alpha = alpha, maxiter = maxiter, gpu_ids = gpu_ids)
end

"""
    minTV!(dst::AbstractArray{Float32, 3}, src::AbstractArray{Float32, 3}; alpha::Real = 15f0, maxiter::Integer = 100, gpu_ids = nothing) -> dst

In-place Total Variation gradient-descent minimization from `src` into `dst`.

# Arguments
- `dst::AbstractArray{Float32, 3}`: Pre-allocated destination array of size `(nx, ny, nz)`.
- `src::AbstractArray{Float32, 3}`: Input 3D volume.

# Keyword Arguments
- `alpha::Real = 15f0`: Step size scaling parameter (default: `15f0`).
- `maxiter::Integer = 100`: Iteration count (default: `100`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).

# Returns
- The mutated `dst` array after TV gradient minimization.

# Examples
```julia
minTV!(recon, recon; alpha=5f0, maxiter=20)
```
"""
function minTV!(dst::AbstractArray{Float32, 3}, src::AbstractArray{Float32, 3}; alpha::Real = 15f0, maxiter::Integer = 100, gpu_ids = nothing)
    src = _require_canonical_array(src, "src", size(src))
    dst = _require_canonical_array(dst, "dst", size(src))
    return _native_min_tv!(dst, src, Float32(alpha), maxiter, gpu_ids)
end

"""
    awminTV(volume::AbstractArray{Float32, 3}; alpha::Real = 15f0, maxiter::Integer = 100, delta::Real = -0.005f0, gpu_ids = nothing) -> Array{Float32, 3}

Adaptive-Weighted Total Variation (AwTV) gradient descent for edge-preserving denoising.

Weights the TV gradient using edge-detection weights derived from local gradients.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Input 3D volume.

# Keyword Arguments
- `alpha::Real = 15f0`: Step size scaling parameter (default: `15f0`).
- `maxiter::Integer = 100`: Iteration count (default: `100`).
- `delta::Real = -0.005f0`: Scale parameter for edge weights; if negative, automatically computed from image gradient percentiles (default: `-0.005f0`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).

# Returns
- Allocated 3D volume `Array{Float32, 3}` after edge-preserving AwTV minimization.

# Examples
```julia
recon_awtv = awminTV(recon; alpha=10f0, maxiter=20, delta=0.001f0)
```
"""
function awminTV(volume::AbstractArray{Float32, 3}; alpha::Real = 15f0, maxiter::Integer = 100, delta::Real = -0.005f0, gpu_ids = nothing)
    dst = Array{Float32}(undef, size(volume))
    return awminTV!(dst, volume; alpha = alpha, maxiter = maxiter, delta = delta, gpu_ids = gpu_ids)
end

"""
    awminTV!(dst::AbstractArray{Float32, 3}, src::AbstractArray{Float32, 3}; alpha::Real = 15f0, maxiter::Integer = 100, delta::Real = -0.005f0, gpu_ids = nothing) -> dst

In-place Adaptive-Weighted Total Variation (AwTV) minimization from `src` into `dst`.

# Arguments
- `dst::AbstractArray{Float32, 3}`: Pre-allocated destination array of size `(nx, ny, nz)`.
- `src::AbstractArray{Float32, 3}`: Input 3D volume.

# Keyword Arguments
- `alpha::Real = 15f0`: Step size scaling parameter (default: `15f0`).
- `maxiter::Integer = 100`: Iteration count (default: `100`).
- `delta::Real = -0.005f0`: Scale parameter for edge weights (default: `-0.005f0`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).

# Returns
- The mutated `dst` array after edge-preserving AwTV minimization.

# Examples
```julia
awminTV!(recon, recon; alpha=10f0, maxiter=20, delta=0.001f0)
```
"""
function awminTV!(dst::AbstractArray{Float32, 3}, src::AbstractArray{Float32, 3}; alpha::Real = 15f0, maxiter::Integer = 100, delta::Real = -0.005f0, gpu_ids = nothing)
    src = _require_canonical_array(src, "src", size(src))
    dst = _require_canonical_array(dst, "dst", size(src))
    return _native_awmin_tv!(dst, src, Float32(alpha), maxiter, Float32(delta), gpu_ids)
end

"""
    piccsTV(volume::AbstractArray{Float32, 3}, prior::AbstractArray{Float32, 3}; alpha::Real = 15f0, ratio::Real = 0.95f0, maxiter::Integer = 100, gpu_ids = nothing) -> Array{Float32, 3}

Prior Image Constrained Compressed Sensing (PICCS) Total Variation gradient descent.

Minimizes the composite TV objective:
```math
\\alpha \\mathrm{TV}(x - prior) + (1 - \\alpha) \\mathrm{TV}(x)
```
where `ratio` corresponds to the relative weight between prior-difference TV and standard TV.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Current reconstruction volume.
- `prior::AbstractArray{Float32, 3}`: High-quality prior image matching volume size.

# Keyword Arguments
- `alpha::Real = 15f0`: Gradient descent step size (default: `15f0`).
- `ratio::Real = 0.95f0`: Weight fraction for prior difference TV in `[0, 1]` (default: `0.95f0`).
- `maxiter::Integer = 100`: Iteration count (default: `100`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).

# Returns
- Allocated 3D volume `Array{Float32, 3}` after PICCS TV gradient minimization.

# Examples
```julia
recon_piccs = piccsTV(vol, prior; alpha=5f0, ratio=0.8f0, maxiter=20)
```
"""
function piccsTV(volume::AbstractArray{Float32, 3}, prior::AbstractArray{Float32, 3}; alpha::Real = 15f0, ratio::Real = 0.95f0, maxiter::Integer = 100, gpu_ids = nothing)
    dst = Array{Float32}(undef, size(volume))
    return piccsTV!(dst, volume, prior; alpha = alpha, ratio = ratio, maxiter = maxiter, gpu_ids = gpu_ids)
end

"""
    piccsTV!(dst::AbstractArray{Float32, 3}, src::AbstractArray{Float32, 3}, prior::AbstractArray{Float32, 3}; alpha::Real = 15f0, ratio::Real = 0.95f0, maxiter::Integer = 100, gpu_ids = nothing) -> dst

In-place PICCS TV gradient descent minimization from `src` and `prior` into `dst`.

# Arguments
- `dst::AbstractArray{Float32, 3}`: Pre-allocated destination array of size `(nx, ny, nz)`.
- `src::AbstractArray{Float32, 3}`: Current reconstruction volume.
- `prior::AbstractArray{Float32, 3}`: High-quality prior image matching volume size.

# Keyword Arguments
- `alpha::Real = 15f0`: Gradient descent step size (default: `15f0`).
- `ratio::Real = 0.95f0`: Weight fraction for prior difference TV in `[0, 1]` (default: `0.95f0`).
- `maxiter::Integer = 100`: Iteration count (default: `100`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).

# Returns
- The mutated `dst` array after PICCS TV gradient minimization.

# Examples
```julia
piccsTV!(vol, vol, prior; alpha=5f0, ratio=0.8f0, maxiter=20)
```
"""
function piccsTV!(dst::AbstractArray{Float32, 3}, src::AbstractArray{Float32, 3}, prior::AbstractArray{Float32, 3}; alpha::Real = 15f0, ratio::Real = 0.95f0, maxiter::Integer = 100, gpu_ids = nothing)
    src = _require_canonical_array(src, "src", size(src))
    prior = _require_canonical_array(prior, "prior", size(src))
    dst = _require_canonical_array(dst, "dst", size(src))
    return _native_piccs_tv!(dst, src, prior, Float32(alpha), Float32(ratio), maxiter, gpu_ids)
end
