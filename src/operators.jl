"""
    is_canonical_layout(A::StridedArray) -> Bool

Check if array `A` has standard contiguous column-major stride ordering (Julia standard memory layout).
Returns `false` if `A` has non-contiguous strides or permuted dimensions.
"""
function is_canonical_layout(A::StridedArray)
    expected = 1
    for dim in 1:ndims(A)
        stride(A, dim) == expected || return false
        expected *= size(A, dim)
    end
    return true
end

is_canonical_layout(::Any) = false

function _require_canonical_array(A, name::AbstractString, expected_size::NTuple{3, Int})
    A isa StridedArray{Float32, 3} || throw(ArgumentError("$(name) must be a `StridedArray{Float32,3}`"))
    size(A) == expected_size || throw(ArgumentError("$(name) shape must be $(expected_size)"))
    is_canonical_layout(A) || throw(ArgumentError("$(name) must be a dense array in canonical layout"))
    return A
end

"""
    validate_volume_array(volume::AbstractArray{Float32, 3}, geo::Geometry) -> volume

Validate that `volume` is a contiguous `StridedArray{Float32, 3}` matching `volume_size(geo)`.
Throws `ArgumentError` if type, shape, or strides do not match.
"""
validate_volume_array(volume::AbstractArray{Float32, 3}, geo::Geometry) =
    _require_canonical_array(volume, "volume", volume_size(geo))

"""
    validate_projection_array(projection::AbstractArray{Float32, 3}, geo::Geometry) -> projection

Validate that `projection` is a contiguous `StridedArray{Float32, 3}` matching `projection_size(geo)`.
Throws `ArgumentError` if type, shape, or strides do not match.
"""
validate_projection_array(projection::AbstractArray{Float32, 3}, geo::Geometry) =
    _require_canonical_array(projection, "projection", projection_size(geo))

"""
    Ax(volume::AbstractArray{Float32, 3}, geo::Geometry; projection_type = :siddon, gpu_ids = nothing) -> Array{Float32, 3}

Compute X-ray forward projection (Radon transform) `p = A x` using GPU CUDA kernels.

# Arguments
- `volume::AbstractArray{Float32, 3}`: 3D voxel array of size `(nx, ny, nz)`.
- `geo::Geometry`: CT acquisition geometry specification (`Geometry`).

# Keyword Arguments
- `projection_type::Symbol = :siddon`: Ray-tracing algorithm (`:siddon` for exact Siddon line-integral tracing, or `:ray_voxel` for interpolated ray-voxel model) (default: `:siddon`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).

# Returns
- 3D projection tensor `Array{Float32, 3}` of size `(nu, nv, nangles)`.

# Examples
```julia
proj = Ax(phantom, geo; projection_type=:siddon)
```

See also: [`Ax!`](@ref), [`Atb`](@ref).
"""
function Ax(volume::AbstractArray{Float32, 3}, geo::Geometry; projection_type = :siddon, gpu_ids = nothing)
    projection = Array{Float32}(undef, projection_size(geo))
    return Ax!(projection, volume, geo; projection_type = projection_type, gpu_ids = gpu_ids)
end

"""
    Ax!(projection::AbstractArray{Float32, 3}, volume::AbstractArray{Float32, 3}, geo::Geometry; projection_type = :siddon, gpu_ids = nothing) -> projection

In-place X-ray forward projection (Radon transform) into pre-allocated `projection` array without heap allocations.

# Arguments
- `projection::AbstractArray{Float32, 3}`: Pre-allocated destination array of size `(nu, nv, nangles)`.
- `volume::AbstractArray{Float32, 3}`: 3D voxel array of size `(nx, ny, nz)`.
- `geo::Geometry`: CT acquisition geometry specification (`Geometry`).

# Keyword Arguments
- `projection_type::Symbol = :siddon`: Ray-tracing algorithm (`:siddon` or `:ray_voxel`) (default: `:siddon`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).

# Returns
- The mutated `projection` array of size `(nu, nv, nangles)`.

# Examples
```julia
proj_buf = similar(proj)
Ax!(proj_buf, phantom, geo)
```

See also: [`Ax`](@ref), [`Atb!`](@ref).
"""
function Ax!(projection::AbstractArray{Float32, 3}, volume::AbstractArray{Float32, 3}, geo::Geometry; projection_type = :siddon, gpu_ids = nothing)
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))
    return _native_ax!(projection, volume, geo, projection_type, gpu_ids)
end

"""
    Atb(projection::AbstractArray{Float32, 3}, geo::Geometry; backprojection_type = :fdk, gpu_ids = nothing) -> Array{Float32, 3}

Compute backprojection `x = Aᵀ b` of detector projections into 3D volume using GPU CUDA kernels.

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D sinogram array of size `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification (`Geometry`).

# Keyword Arguments
- `backprojection_type::Symbol = :fdk`: Backprojection kernel type (`:matched` for exact adjoint ray-tracing, `:fdk` for FDK-weighted backprojection, or `:voxel_cone` for voxel-driven cone-beam backprojection) (default: `:fdk`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).

# Returns
- 3D voxel array `Array{Float32, 3}` of size `(nx, ny, nz)`.

# Examples
```julia
recon = Atb(proj, geo; backprojection_type=:matched)
```

See also: [`Atb!`](@ref), [`Ax`](@ref).
"""
function Atb(projection::AbstractArray{Float32, 3}, geo::Geometry; backprojection_type = :fdk, gpu_ids = nothing)
    volume = Array{Float32}(undef, volume_size(geo))
    return Atb!(volume, projection, geo; backprojection_type = backprojection_type, gpu_ids = gpu_ids)
end

"""
    Atb!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry; backprojection_type = :fdk, gpu_ids = nothing) -> volume

In-place backprojection `x = Aᵀ b` into pre-allocated `volume` array without heap allocations.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Pre-allocated destination array of size `(nx, ny, nz)`.
- `projection::AbstractArray{Float32, 3}`: 3D sinogram array of size `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification (`Geometry`).

# Keyword Arguments
- `backprojection_type::Symbol = :fdk`: Backprojection kernel type (`:matched`, `:fdk`, or `:voxel_cone`) (default: `:fdk`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).

# Returns
- The mutated `volume` array of size `(nx, ny, nz)`.

# Examples
```julia
vol_buf = similar(recon)
Atb!(vol_buf, proj, geo; backprojection_type=:matched)
```

See also: [`Atb`](@ref), [`Ax!`](@ref).
"""
function Atb!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry; backprojection_type = :fdk, gpu_ids = nothing)
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))
    return _native_atb!(volume, projection, geo, backprojection_type, gpu_ids)
end
