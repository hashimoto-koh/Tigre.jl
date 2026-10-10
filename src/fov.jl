# Field of View (FOV) masking and CBCT cropping utilities
# Reference:
# - TIGRE cropCBCT.m / crop_CBCT.py
# - TIGRE out_of_FOV_mask.m

"""
    compute_fov_radius(geo::Geometry) -> Tuple{Float32, Float32}

Compute the physical and voxel-space field of view (FOV) radius in the transaxial (x-y) plane.

# Arguments
- `geo::Geometry`: CT acquisition geometry specification.

# Returns
- A tuple `(radius_mm, radius_voxels)` where:
  - `radius_mm`: Physical FOV radius in millimeters.
  - `radius_voxels`: FOV radius in voxel count.

# Examples
```julia
r_mm, r_vox = compute_fov_radius(geo)
```
"""
function compute_fov_radius(geo::Geometry)
    mean_dso = sum(geo.DSO) / length(geo.DSO)
    mean_dsd = sum(geo.DSD) / length(geo.DSD)
    mean_off_u = sum(abs, geo.offDetectorU) / length(geo.offDetectorU)

    half_det_u = geo.sDetector[1] / 2f0
    eff_det_u = half_det_u + mean_off_u

    radius_mm = if geo.mode === :cone
        eff_det_u * (mean_dso / mean_dsd)
    else
        eff_det_u
    end

    radius_vox = radius_mm / geo.dVoxel[1]
    max_vox = min(geo.nVoxel[1] - 1, geo.nVoxel[2] - 1) / 2f0
    clamped_vox = min(radius_vox, max_vox)

    return Float32(radius_mm), Float32(clamped_vox)
end

"""
    out_of_fov_mask(geo::Geometry; radius = nothing, circle = true) -> BitArray{3}
    out_of_fov_mask(dims::Tuple{Integer, Integer}; radius = nothing, circle = true) -> BitMatrix

Generate a boolean mask where `true` corresponds to voxels within the reconstructible
Field of View (FOV) and `false` to voxels outside the FOV.

# Arguments
- `geo::Geometry`: CT acquisition geometry (returns 3D `BitArray{3}` of size `geo.nVoxel`).
- `dims::Tuple{Integer, Integer}`: 2D dimensions `(nx, ny)` (returns 2D `BitMatrix`).

# Keyword Arguments
- `radius = nothing`: FOV radius in voxel units, or `nothing` to compute automatically from geometry (default: `nothing`).
- `circle::Bool = true`: If `true`, generates circular/cylindrical mask; if `false`, rectangular mask (default: `true`).

# Returns
- `BitArray{3}` or `BitMatrix` where `true` indicates reconstructible voxels inside the FOV.

# Examples
```julia
mask = out_of_fov_mask(geo; circle=true)
```
"""
function out_of_fov_mask(
    geo::Geometry;
    radius::Union{Real, Nothing} = nothing,
    circle::Bool = true,
)
    nx, ny, nz = geo.nVoxel
    r_vox = if radius !== nothing
        r = Float32(radius)
        r > 0f0 || throw(ArgumentError("radius must be positive, got $radius"))
        r
    else
        _, r = compute_fov_radius(geo)
        r
    end

    off_x = sum(geo.offOriginX) / length(geo.offOriginX)
    off_y = sum(geo.offOriginY) / length(geo.offOriginY)

    cx = (nx + 1f0) / 2f0 + off_x / geo.dVoxel[1]
    cy = (ny + 1f0) / 2f0 + off_y / geo.dVoxel[2]

    # Precompute 2D slice mask
    mask2d = BitMatrix(undef, nx, ny)
    r2 = r_vox^2
    @inbounds for iy in 1:ny
        dy = Float32(iy) - cy
        dy2 = dy * dy
        for ix in 1:nx
            dx = Float32(ix) - cx
            if circle
                mask2d[ix, iy] = (dx * dx + dy2) <= r2
            else
                mask2d[ix, iy] = abs(dx) <= r_vox && abs(dy) <= r_vox
            end
        end
    end

    mask3d = BitArray{3}(undef, nx, ny, nz)
    @inbounds for iz in 1:nz
        @views mask3d[:, :, iz] .= mask2d
    end

    return mask3d
end

function out_of_fov_mask(
    dims::Tuple{Integer, Integer};
    radius::Union{Real, Nothing} = nothing,
    circle::Bool = true,
)
    nx, ny = dims
    cx = (nx + 1f0) / 2f0
    cy = (ny + 1f0) / 2f0
    r_vox = if radius !== nothing
        r = Float32(radius)
        r > 0f0 || throw(ArgumentError("radius must be positive, got $radius"))
        r
    else
        min(nx - 1f0, ny - 1f0) / 2f0
    end

    mask2d = BitMatrix(undef, nx, ny)
    r2 = r_vox^2
    @inbounds for iy in 1:ny
        dy = Float32(iy) - cy
        dy2 = dy * dy
        for ix in 1:nx
            dx = Float32(ix) - cx
            if circle
                mask2d[ix, iy] = (dx * dx + dy2) <= r2
            else
                mask2d[ix, iy] = abs(dx) <= r_vox && abs(dy) <= r_vox
            end
        end
    end
    return mask2d
end

"""
    crop_cbct!(volume::AbstractArray{Float32, 3}, geo::Geometry; fill_value = 0f0, radius = nothing, circle = true) -> volume
    crop_cbct!(volume::AbstractArray{Float32, 3}; fill_value = 0f0, radius = nothing, circle = true) -> volume

In-place masking of reconstructed CT `volume` outside the reconstructible Field of View (FOV) with `fill_value`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: 3D reconstructed volume `(nx, ny, nz)` modified in-place.
- `geo::Geometry`: Optional CT geometry specification. If omitted, FOV is determined by volume dimensions.

# Keyword Arguments
- `fill_value::Real = 0f0`: Value assigned to voxels outside the FOV (default: `0f0`).
- `radius = nothing`: Mask radius in voxel units, or `nothing` to determine automatically (default: `nothing`).
- `circle::Bool = true`: If `true`, applies circular transaxial cylinder; if `false`, square/cuboid mask (default: `true`).

# Returns
- The mutated `volume` array with voxels outside the FOV replaced with `fill_value`.

# Examples
```julia
crop_cbct!(recon, geo; fill_value=0f0)
```
"""
function crop_cbct!(
    volume::AbstractArray{Float32, 3},
    geo::Geometry;
    fill_value::Real = 0f0,
    radius::Union{Real, Nothing} = nothing,
    circle::Bool = true,
)
    validate_volume_array(volume, geo)
    nx, ny, nz = size(volume)
    r_vox = if radius !== nothing
        r = Float32(radius)
        r > 0f0 || throw(ArgumentError("radius must be positive, got $radius"))
        r
    else
        _, r = compute_fov_radius(geo)
        r
    end

    off_x = sum(geo.offOriginX) / length(geo.offOriginX)
    off_y = sum(geo.offOriginY) / length(geo.offOriginY)

    cx = (nx + 1f0) / 2f0 + off_x / geo.dVoxel[1]
    cy = (ny + 1f0) / 2f0 + off_y / geo.dVoxel[2]
    r2 = r_vox^2
    fill_f32 = Float32(fill_value)

    # In-place masking slice by slice with zero 3D heap allocation
    @inbounds for iy in 1:ny
        dy = Float32(iy) - cy
        dy2 = dy * dy
        for ix in 1:nx
            dx = Float32(ix) - cx
            is_in = circle ? (dx * dx + dy2 <= r2) : (abs(dx) <= r_vox && abs(dy) <= r_vox)
            if !is_in
                for iz in 1:nz
                    volume[ix, iy, iz] = fill_f32
                end
            end
        end
    end

    return volume
end

function crop_cbct!(
    volume::AbstractArray{Float32, 3};
    fill_value::Real = 0f0,
    radius::Union{Real, Nothing} = nothing,
    circle::Bool = true,
)
    is_canonical_layout(volume) || throw(ArgumentError("volume must have standard column-major contiguous layout"))
    nx, ny, nz = size(volume)
    cx = (nx + 1f0) / 2f0
    cy = (ny + 1f0) / 2f0
    r_vox = if radius !== nothing
        r = Float32(radius)
        r > 0f0 || throw(ArgumentError("radius must be positive, got $radius"))
        r
    else
        min(nx - 1f0, ny - 1f0) / 2f0
    end

    r2 = r_vox^2
    fill_f32 = Float32(fill_value)

    @inbounds for iy in 1:ny
        dy = Float32(iy) - cy
        dy2 = dy * dy
        for ix in 1:nx
            dx = Float32(ix) - cx
            is_in = circle ? (dx * dx + dy2 <= r2) : (abs(dx) <= r_vox && abs(dy) <= r_vox)
            if !is_in
                for iz in 1:nz
                    volume[ix, iy, iz] = fill_f32
                end
            end
        end
    end

    return volume
end

"""
    crop_cbct(volume::AbstractArray{Float32, 3}, [geo::Geometry]; fill_value = 0f0, radius = nothing, circle = true) -> Array{Float32, 3}

Non-destructive version of `crop_cbct!`. Returns a new volume array with FOV masking applied.

# Arguments
- `volume::AbstractArray{Float32, 3}`: 3D reconstructed volume `(nx, ny, nz)`.
- `geo::Geometry`: Optional CT geometry specification. If omitted, FOV is determined by volume dimensions.

# Keyword Arguments
- `fill_value::Real = 0f0`: Value assigned to voxels outside the FOV (default: `0f0`).
- `radius = nothing`: Mask radius in voxel units, or `nothing` to determine automatically (default: `nothing`).
- `circle::Bool = true`: If `true`, applies circular transaxial cylinder; if `false`, square/cuboid mask (default: `true`).

# Returns
- A newly allocated `Array{Float32, 3}` copy of `volume` with out-of-FOV voxels replaced with `fill_value`.

# Examples
```julia
cropped = crop_cbct(recon, geo; fill_value=0f0, circle=true)
```
"""
function crop_cbct(volume::AbstractArray{Float32, 3}, args...; kwargs...)
    out = copy(volume)
    return crop_cbct!(out, args...; kwargs...)
end
