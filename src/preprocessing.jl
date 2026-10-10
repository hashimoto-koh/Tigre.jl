# Projection preprocessing and resampling utilities for CT reconstruction

"""
    binning(projections::AbstractArray{<:Real, 3}, factor; mode = :mean) -> Array{Float32, 3}
    binning(geo::Geometry, factor) -> Geometry
    binning(projections::AbstractArray{<:Real, 3}, geo::Geometry, factor; mode = :mean) -> Tuple{Array{Float32, 3}, Geometry}

Downsample CT detector projections and/or geometry by spatial pixel binning along detector dimensions `(u, v)`.

# Arguments
- `projections::AbstractArray{<:Real, 3}`: 3D detector projection array `(nu, nv, nangles)`.
- `geo::Geometry`: CT geometry specification.
- `factor::Union{Integer, Tuple{Integer, Integer}}`: Integer or `(bin_u, bin_v)` downsampling factor.

# Keyword Arguments
- `mode::Symbol = :mean`: Downsampling reduction operation (`:mean` to preserve attenuation scale, or `:sum` to preserve photon counts) (default: `:mean`).

# Returns
- When given `projections`, returns the binned `Array{Float32, 3}` of size `(nu ÷ bin_u, nv ÷ bin_v, nangles)`.
- When given `geo`, returns a new `Geometry` with scaled detector pixel sizes and counts.
- When given both, returns a tuple `(binned_projections, binned_geo)`.

# Examples
```julia
# 2x2 binning of projections and geometry
b_proj, b_geo = binning(proj, geo, 2)

# Asymmetric binning
b_proj, b_geo = binning(proj, geo, (2, 4); mode=:mean)
```
"""
function binning(
    projections::AbstractArray{<:Real, 3},
    factor::Union{Integer, Tuple{Integer, Integer}};
    mode::Symbol = :mean,
)
    mode in (:mean, :sum) || throw(ArgumentError("mode must be :mean or :sum, got :$mode"))
    bu, bv = if factor isa Integer
        Int(factor), Int(factor)
    else
        Int(factor[1]), Int(factor[2])
    end
    (bu >= 1 && bv >= 1) || throw(ArgumentError("binning factors must be positive integers, got ($bu, $bv)"))

    if bu == 1 && bv == 1
        return Array{Float32, 3}(projections)
    end

    nu, nv, nangles = size(projections)
    out_nu = div(nu, bu)
    out_nv = div(nv, bv)
    (out_nu >= 1 && out_nv >= 1) || throw(ArgumentError("binned detector dimensions ($out_nu, $out_nv) must be >= 1"))

    out = Array{Float32, 3}(undef, out_nu, out_nv, nangles)
    norm_factor = mode === :mean ? 1.0 / Float64(bu * bv) : 1.0

    @inbounds for ia in 1:nangles
        for j in 1:out_nv
            v_start = (j - 1) * bv + 1
            v_end = j * bv
            for i in 1:out_nu
                u_start = (i - 1) * bu + 1
                u_end = i * bu
                acc = 0.0
                for v in v_start:v_end
                    for u in u_start:u_end
                        acc += Float64(projections[u, v, ia])
                    end
                end
                out[i, j, ia] = Float32(acc * norm_factor)
            end
        end
    end

    return out
end

function binning(
    geo::Geometry,
    factor::Union{Integer, Tuple{Integer, Integer}},
)
    bu, bv = if factor isa Integer
        Int(factor), Int(factor)
    else
        Int(factor[1]), Int(factor[2])
    end
    (bu >= 1 && bv >= 1) || throw(ArgumentError("binning factors must be positive integers, got ($bu, $bv)"))

    old_nu, old_nv = geo.nDetector
    new_nu = div(old_nu, bu)
    new_nv = div(old_nv, bv)
    (new_nu >= 1 && new_nv >= 1) || throw(ArgumentError("binned detector dimensions ($new_nu, $new_nv) must be >= 1"))

    new_du = geo.dDetector[1] * Float32(bu)
    new_dv = geo.dDetector[2] * Float32(bv)
    new_su = Float32(new_nu) * new_du
    new_sv = Float32(new_nv) * new_dv

    return Geometry(
        geo.mode,
        geo.nVoxel,
        geo.sVoxel,
        geo.dVoxel,
        (new_nu, new_nv),
        (new_su, new_sv),
        (new_du, new_dv),
        copy(geo.angles),
        copy(geo.DSD),
        copy(geo.DSO),
        copy(geo.offOriginX),
        copy(geo.offOriginY),
        copy(geo.offOriginZ),
        copy(geo.offDetectorU),
        copy(geo.offDetectorV),
        copy(geo.rotDetectorRoll),
        copy(geo.rotDetectorPitch),
        copy(geo.rotDetectorYaw),
        copy(geo.COR),
        geo.accuracy,
    )
end

function binning(
    projections::AbstractArray{<:Real, 3},
    geo::Geometry,
    factor::Union{Integer, Tuple{Integer, Integer}};
    mode::Symbol = :mean,
)
    binned_proj = binning(projections, factor; mode = mode)
    binned_geo = binning(geo, factor)
    return binned_proj, binned_geo
end

"""
    subsample_angles(projections::AbstractArray{<:Real, 3}, indices) -> Array{Float32, 3}
    subsample_angles(geo::Geometry, indices) -> Geometry
    subsample_angles(projections::AbstractArray{<:Real, 3}, geo::Geometry, step_or_indices) -> Tuple{Array{Float32, 3}, Geometry}

Subsample CT projection data and geometry along the angular dimension (sparse-view CT generation).

# Arguments
- `projections::AbstractArray{<:Real, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT geometry specification.
- `step_or_indices`: Step size integer `k` (subsampling `1:k:nangles`), or an iterable collection of 1-based angle indices.

# Returns
- When given `projections`, returns `Array{Float32, 3}` sliced along the 3rd dimension.
- When given `geo`, returns a new `Geometry` containing only the sampled projection angles and geometry vectors.
- When given both, returns a tuple `(sub_projections, sub_geo)`.

# Examples
```julia
# Take every 2nd projection angle
sparse_proj, sparse_geo = subsample_angles(proj, geo, 2)

# Specific indices
sub_proj, sub_geo = subsample_angles(proj, geo, 1:5:90)
```
"""
function subsample_angles(
    projections::AbstractArray{<:Real, 3},
    indices,
)
    inds = collect(indices)
    isempty(inds) && throw(ArgumentError("indices cannot be empty"))
    nangles = size(projections, 3)
    all(1 <= i <= nangles for i in inds) || throw(BoundsError(1:nangles, inds))
    return Array{Float32, 3}(projections[:, :, inds])
end

function subsample_angles(
    geo::Geometry,
    indices,
)
    inds = collect(indices)
    isempty(inds) && throw(ArgumentError("indices cannot be empty"))
    nangles = size(geo.angles, 2)
    all(1 <= i <= nangles for i in inds) || throw(BoundsError(1:nangles, inds))

    new_angles = geo.angles[:, inds]
    new_DSD = geo.DSD[inds]
    new_DSO = geo.DSO[inds]
    new_offOriginX = geo.offOriginX[inds]
    new_offOriginY = geo.offOriginY[inds]
    new_offOriginZ = geo.offOriginZ[inds]
    new_offDetectorU = geo.offDetectorU[inds]
    new_offDetectorV = geo.offDetectorV[inds]
    new_rotDetectorRoll = geo.rotDetectorRoll[inds]
    new_rotDetectorPitch = geo.rotDetectorPitch[inds]
    new_rotDetectorYaw = geo.rotDetectorYaw[inds]
    new_COR = geo.COR[inds]

    return Geometry(
        geo.mode,
        geo.nVoxel,
        geo.sVoxel,
        geo.dVoxel,
        geo.nDetector,
        geo.sDetector,
        geo.dDetector,
        new_angles,
        new_DSD,
        new_DSO,
        new_offOriginX,
        new_offOriginY,
        new_offOriginZ,
        new_offDetectorU,
        new_offDetectorV,
        new_rotDetectorRoll,
        new_rotDetectorPitch,
        new_rotDetectorYaw,
        new_COR,
        geo.accuracy,
    )
end

function subsample_angles(
    projections::AbstractArray{<:Real, 3},
    geo::Geometry,
    step_or_indices,
)
    nangles = projection_count(geo)
    inds = step_or_indices isa Integer ? (1:step_or_indices:nangles) : step_or_indices
    sub_proj = subsample_angles(projections, inds)
    sub_geo = subsample_angles(geo, inds)
    return sub_proj, sub_geo
end
