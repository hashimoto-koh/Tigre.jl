"""
    Geometry

CT acquisition geometry specification for 2D/3D cone-beam and parallel-beam CT.

# Fields
- `mode::Symbol`: Beam geometry mode (`:cone` or `:parallel`).
- `nVoxel::NTuple{3, Int}`: Number of voxels `(nx, ny, nz)`.
- `sVoxel::NTuple{3, Float32}`: Physical volume size `(sx, sy, sz)` in mm.
- `dVoxel::NTuple{3, Float32}`: Voxel pitch `(dx, dy, dz)` in mm.
- `nDetector::NTuple{2, Int}`: Number of detector channels `(nu, nv)`.
- `sDetector::NTuple{2, Float32}`: Physical detector size `(su, sv)` in mm.
- `dDetector::NTuple{2, Float32}`: Detector pixel pitch `(du, dv)` in mm.
- `angles::Matrix{Float32}`: Gantry projection angles of size `3 × nangles` (radians).
- `DSD::Vector{Float32}`: Source-to-Detector distance in mm per projection.
- `DSO::Vector{Float32}`: Source-to-Origin distance in mm per projection.
- `offOriginX::Vector{Float32}`, `offOriginY::Vector{Float32}`, `offOriginZ::Vector{Float32}`: Object offset in mm.
- `offDetectorU::Vector{Float32}`, `offDetectorV::Vector{Float32}`: Detector panel lateral/vertical offset in mm.
- `rotDetectorRoll::Vector{Float32}`, `rotDetectorPitch::Vector{Float32}`, `rotDetectorYaw::Vector{Float32}`: Detector Euler angles (radians).
- `COR::Vector{Float32}`: Centre-of-rotation offset in mm per projection.
- `accuracy::Float32`: Interpolation accuracy parameter for ray-tracing kernels.
"""
struct Geometry
    mode::Symbol
    nVoxel::NTuple{3, Int}
    sVoxel::NTuple{3, Float32}
    dVoxel::NTuple{3, Float32}
    nDetector::NTuple{2, Int}
    sDetector::NTuple{2, Float32}
    dDetector::NTuple{2, Float32}
    angles::Matrix{Float32}  # 3 x nangles
    DSD::Vector{Float32}
    DSO::Vector{Float32}
    offOriginX::Vector{Float32}
    offOriginY::Vector{Float32}
    offOriginZ::Vector{Float32}
    offDetectorU::Vector{Float32}
    offDetectorV::Vector{Float32}
    rotDetectorRoll::Vector{Float32}
    rotDetectorPitch::Vector{Float32}
    rotDetectorYaw::Vector{Float32}
    COR::Vector{Float32}
    accuracy::Float32
end

"""
    normalize_angles(angles) -> Matrix{Float32}

Normalize gantry angles into a canonical `3 × nangles` `Matrix{Float32}`.
1D angle vectors covering `[0, 2π]` are placed into the first row (primary rotation) with zero tilt/roll.
"""
function normalize_angles(angles)
    if angles isa AbstractVector
        out = Matrix{Float32}(undef, 3, length(angles))
        out[1, :] .= Float32.(angles)
        out[2, :] .= 0f0
        out[3, :] .= 0f0
        return out
    elseif angles isa AbstractMatrix
        if size(angles, 1) == 3
            return Matrix{Float32}(angles)
        elseif size(angles, 2) == 3
            return permutedims(Float32.(angles))
        end
    end
    throw(ArgumentError("angles must be a Vector{<:Real} or a 3xN / Nx3 matrix"))
end

function _normalize_tuple(::Type{Int}, value, n::Int, name::Symbol)
    data = collect(value)
    length(data) == n || throw(ArgumentError("$(name) length must be $(n)"))
    return ntuple(i -> Int(data[i]), n)
end

function _normalize_tuple(::Type{Float32}, value, n::Int, name::Symbol)
    data = collect(value)
    length(data) == n || throw(ArgumentError("$(name) length must be $(n)"))
    return ntuple(i -> Float32(data[i]), n)
end

function _resolve_lengths(counts, total, delta, name::Symbol)
    if total === nothing && delta === nothing
        throw(ArgumentError("$(name) requires total, delta, or both"))
    elseif total === nothing
        d = _normalize_tuple(Float32, delta, length(counts), Symbol("d", name))
        s = ntuple(i -> Float32(counts[i]) * d[i], length(counts))
        return s, d
    elseif delta === nothing
        s = _normalize_tuple(Float32, total, length(counts), Symbol("s", name))
        d = ntuple(i -> s[i] / Float32(counts[i]), length(counts))
        return s, d
    else
        s = _normalize_tuple(Float32, total, length(counts), Symbol("s", name))
        d = _normalize_tuple(Float32, delta, length(counts), Symbol("d", name))
        for i in eachindex(counts)
            expected = Float32(counts[i]) * d[i]
            isapprox(expected, s[i]; atol=1f-5, rtol=0f0) ||
                throw(ArgumentError("$(name) mismatch: count * delta != total"))
        end
        return s, d
    end
end

function _expand_per_projection(value, nangles::Int, name::Symbol)
    if value isa Real
        return fill(Float32(value), nangles)
    end
    data = Float32.(collect(value))
    if length(data) == 1
        return fill(data[1], nangles)
    elseif length(data) == nangles
        return data
    end
    throw(ArgumentError("$(name) must be a scalar or an array of length $(nangles)"))
end

function _expand_components(value, ncomp::Int, nangles::Int, name::Symbol)
    mat = if value isa Real
        fill(Float32(value), ncomp, nangles)
    elseif value isa AbstractMatrix
        if size(value) == (ncomp, nangles)
            Matrix{Float32}(value)
        elseif size(value) == (nangles, ncomp)
            permutedims(Float32.(value))
        elseif size(value) == (ncomp, 1)
            repeat(Matrix{Float32}(value), 1, nangles)
        elseif size(value) == (1, ncomp)
            repeat(permutedims(Float32.(value)), 1, nangles)
        else
            throw(ArgumentError("$(name) size must be $(ncomp)×$(nangles) or $(nangles)×$(ncomp)"))
        end
    else
        data = Float32.(collect(value))
        if length(data) == 1
            fill(data[1], ncomp, nangles)
        elseif length(data) == ncomp
            repeat(reshape(data, ncomp, 1), 1, nangles)
        else
            throw(ArgumentError("$(name) must be a scalar, a length-$(ncomp) vector, or a $(ncomp)×$(nangles) / $(nangles)×$(ncomp) matrix"))
        end
    end

    return ntuple(i -> copy(vec(@view(mat[i, :]))), ncomp)
end

"""
    Geometry(; mode = :cone, angles, nVoxel, sVoxel = nothing, dVoxel = nothing, nDetector, sDetector = nothing, dDetector = nothing, DSD, DSO, offOrigin = 0f0, offDetector = 0f0, rotDetector = 0f0, COR = 0f0, accuracy = 0.5f0) -> Geometry

Construct a CT acquisition geometry specification for 2D/3D cone-beam or parallel-beam CT.

# Keyword Arguments
- `mode::Symbol = :cone`: Acquisition geometry mode (`:cone` for divergent cone-beam, or `:parallel` for parallel-beam) (default: `:cone`).
- `angles`: Gantry projection angles in radians. 1D vector of length `N` or `3 × N` matrix.
- `nVoxel`: Number of voxels `(nx, ny, nz)`.
- `sVoxel = nothing`: Physical volume dimensions `(sx, sy, sz)` in mm. Either `sVoxel` or `dVoxel` must be specified (default: `nothing`).
- `dVoxel = nothing`: Voxel pitch `(dx, dy, dz)` in mm (default: `nothing`).
- `nDetector`: Number of detector channels/pixels `(nu, nv)`.
- `sDetector = nothing`: Physical detector panel dimensions `(su, sv)` in mm. Either `sDetector` or `dDetector` must be specified (default: `nothing`).
- `dDetector = nothing`: Detector pixel pitch `(du, dv)` in mm (default: `nothing`).
- `DSD`: Distance from X-ray source to detector panel in mm (scalar or length-`N` vector).
- `DSO`: Distance from X-ray source to object center of rotation in mm (scalar or length-`N` vector).
- `offOrigin = 0f0`: Object center offset `(x, y, z)` in mm relative to rotation axis (default: `0f0`).
- `offDetector = 0f0`: Detector lateral/vertical offset `(u, v)` in mm (default: `0f0`).
- `rotDetector = 0f0`: Detector orientation roll, pitch, yaw in radians (default: `0f0`).
- `COR = 0f0`: Centre-of-rotation displacement in mm (scalar or length-`N` vector) (default: `0f0`).
- `accuracy::Real = 0.5f0`: Ray-tracing step size factor (Siddon / interpolation accuracy) (default: `0.5f0`).

# Returns
- A configured `Geometry` instance.

# Examples
```julia
geo = Geometry(
    mode = :cone,
    angles = Float32.(range(0, 2π, length = 180)),
    nVoxel = (256, 256, 256),
    sVoxel = (256f0, 256f0, 256f0),
    nDetector = (384, 384),
    sDetector = (384f0, 384f0),
    DSO = 1000f0,
    DSD = 1500f0,
)
```
"""
function Geometry(;
    mode::Symbol = :cone,
    angles,
    nVoxel,
    sVoxel = nothing,
    dVoxel = nothing,
    nDetector,
    sDetector = nothing,
    dDetector = nothing,
    DSD,
    DSO,
    offOrigin = 0f0,
    offDetector = 0f0,
    rotDetector = 0f0,
    COR = 0f0,
    accuracy::Real = 0.5f0,
)
    mode in (:cone, :parallel) || throw(ArgumentError("mode must be :cone or :parallel"))
    norm_angles = normalize_angles(angles)
    nangles = size(norm_angles, 2)

    voxel_counts = _normalize_tuple(Int, nVoxel, 3, :nVoxel)
    det_counts = _normalize_tuple(Int, nDetector, 2, :nDetector)
    any(x -> x < 1, voxel_counts) && throw(ArgumentError("nVoxel elements must all be positive"))
    any(x -> x < 1, det_counts) && throw(ArgumentError("nDetector elements must all be positive"))

    svoxel, dvoxel = _resolve_lengths(voxel_counts, sVoxel, dVoxel, :Voxel)
    sdet, ddet = _resolve_lengths(det_counts, sDetector, dDetector, :Detector)

    off_origin = _expand_components(offOrigin, 3, nangles, :offOrigin)
    off_detector = _expand_components(offDetector, 2, nangles, :offDetector)
    rot_detector = _expand_components(rotDetector, 3, nangles, :rotDetector)

    return Geometry(
        mode,
        voxel_counts,
        svoxel,
        dvoxel,
        det_counts,
        sdet,
        ddet,
        norm_angles,
        _expand_per_projection(DSD, nangles, :DSD),
        _expand_per_projection(DSO, nangles, :DSO),
        off_origin[1],
        off_origin[2],
        off_origin[3],
        off_detector[1],
        off_detector[2],
        rot_detector[1],
        rot_detector[2],
        rot_detector[3],
        _expand_per_projection(COR, nangles, :COR),
        Float32(accuracy),
    )
end

"""
    projection_count(geo::Geometry) -> Int

Return the number of projection views (angles) in `geo`.

# Arguments
- `geo::Geometry`: CT geometry specification.

# Returns
- Number of gantry projection angles as an `Int`.
"""
projection_count(geo::Geometry) = size(geo.angles, 2)

"""
    projection_size(geo::Geometry) -> NTuple{3, Int}

Return the expected projection array dimensions `(nu, nv, nangles)`.

# Arguments
- `geo::Geometry`: CT geometry specification.

# Returns
- Tuple `(nu, nv, nangles)` specifying projection dimensions.
"""
projection_size(geo::Geometry) = (geo.nDetector[1], geo.nDetector[2], projection_count(geo))

"""
    volume_size(geo::Geometry) -> NTuple{3, Int}

Return the expected 3D volume dimensions `(nx, ny, nz)`.

# Arguments
- `geo::Geometry`: CT geometry specification.

# Returns
- Tuple `(nx, ny, nz)` specifying volume dimensions.
"""
volume_size(geo::Geometry) = geo.nVoxel

