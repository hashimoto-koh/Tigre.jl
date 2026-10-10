const _FFTW3F_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
const _FFTW_R2C_PLANS = Dict{Tuple{Int, Int}, Ptr{Cvoid}}()
const _FFTW_C2R_PLANS = Dict{Tuple{Int, Int}, Ptr{Cvoid}}()

function _fftw3f_handle()
    if _FFTW3F_HANDLE[] == C_NULL
        path = Libdl.find_library(["fftw3f", "libfftw3f"], ["/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu", "/lib", "/usr/lib"])
        path == "" && return C_NULL
        _FFTW3F_HANDLE[] = Libdl.dlopen(path)
    end
    return _FFTW3F_HANDLE[]
end

function _fftw_r2c_plan(len::Int, howmany::Int)
    key = (len, howmany)
    haskey(_FFTW_R2C_PLANS, key) && return _FFTW_R2C_PLANS[key]
    handle = _fftw3f_handle()
    handle == C_NULL && return C_NULL

    input = zeros(Float32, len, howmany)
    output = zeros(ComplexF32, len ÷ 2 + 1, howmany)
    n = Ref{Cint}(Cint(len))
    plan = ccall(
        Libdl.dlsym(handle, :fftwf_plan_many_dft_r2c),
        Ptr{Cvoid},
        (Cint, Ptr{Cint}, Cint, Ptr{Cfloat}, Ptr{Cint}, Cint, Cint, Ptr{ComplexF32}, Ptr{Cint}, Cint, Cint, Cuint),
        Cint(1),
        n,
        Cint(howmany),
        pointer(input),
        Ptr{Cint}(C_NULL),
        Cint(1),
        Cint(len),
        pointer(output),
        Ptr{Cint}(C_NULL),
        Cint(1),
        Cint(len ÷ 2 + 1),
        Cuint(64),
    )
    _FFTW_R2C_PLANS[key] = plan
    finalizer(_FFTW_R2C_PLANS) do _
        if plan != C_NULL
            ccall(Libdl.dlsym(handle, :fftwf_destroy_plan), Cvoid, (Ptr{Cvoid},), plan)
        end
    end
    return plan
end

function _fftw_c2r_plan(len::Int, howmany::Int)
    key = (len, howmany)
    haskey(_FFTW_C2R_PLANS, key) && return _FFTW_C2R_PLANS[key]
    handle = _fftw3f_handle()
    handle == C_NULL && return C_NULL

    input = zeros(ComplexF32, len ÷ 2 + 1, howmany)
    output = zeros(Float32, len, howmany)
    n = Ref{Cint}(Cint(len))
    plan = ccall(
        Libdl.dlsym(handle, :fftwf_plan_many_dft_c2r),
        Ptr{Cvoid},
        (Cint, Ptr{Cint}, Cint, Ptr{ComplexF32}, Ptr{Cint}, Cint, Cint, Ptr{Cfloat}, Ptr{Cint}, Cint, Cint, Cuint),
        Cint(1),
        n,
        Cint(howmany),
        pointer(input),
        Ptr{Cint}(C_NULL),
        Cint(1),
        Cint(len ÷ 2 + 1),
        pointer(output),
        Ptr{Cint}(C_NULL),
        Cint(1),
        Cint(len),
        Cuint(64),
    )
    _FFTW_C2R_PLANS[key] = plan
    finalizer(_FFTW_C2R_PLANS) do _
        if plan != C_NULL
            ccall(Libdl.dlsym(handle, :fftwf_destroy_plan), Cvoid, (Ptr{Cvoid},), plan)
        end
    end
    return plan
end

function _fftw_rfft!(output::Matrix{ComplexF32}, input::Matrix{Float32})
    handle = _fftw3f_handle()
    handle == C_NULL && return false
    plan = _fftw_r2c_plan(size(input, 1), size(input, 2))
    plan == C_NULL && return false
    ccall(Libdl.dlsym(handle, :fftwf_execute_dft_r2c), Cvoid, (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{ComplexF32}), plan, pointer(input), pointer(output))
    return true
end

function _fftw_irfft!(output::Matrix{Float32}, input::Matrix{ComplexF32})
    handle = _fftw3f_handle()
    handle == C_NULL && return false
    plan = _fftw_c2r_plan(size(output, 1), size(output, 2))
    plan == C_NULL && return false
    ccall(Libdl.dlsym(handle, :fftwf_execute_dft_c2r), Cvoid, (Ptr{Cvoid}, Ptr{ComplexF32}, Ptr{Cfloat}), plan, pointer(input), pointer(output))
    return true
end

function _detector_axis(count::Int, spacing::Float32)
    axis = Vector{Float32}(undef, count)
    center = Float32(count) / 2f0 - 0.5f0
    @inbounds for i in 1:count
        axis[i] = (Float32(i - 1) - center) * spacing
    end
    return axis
end

function _copy_geometry(
    geo::Geometry;
    mode = geo.mode,
    nVoxel = geo.nVoxel,
    sVoxel = geo.sVoxel,
    dVoxel = geo.dVoxel,
    nDetector = geo.nDetector,
    sDetector = geo.sDetector,
    dDetector = geo.dDetector,
    angles = geo.angles,
    DSD = geo.DSD,
    DSO = geo.DSO,
    offOriginX = geo.offOriginX,
    offOriginY = geo.offOriginY,
    offOriginZ = geo.offOriginZ,
    offDetectorU = geo.offDetectorU,
    offDetectorV = geo.offDetectorV,
    rotDetectorRoll = geo.rotDetectorRoll,
    rotDetectorPitch = geo.rotDetectorPitch,
    rotDetectorYaw = geo.rotDetectorYaw,
    COR = geo.COR,
    accuracy = geo.accuracy,
)
    return Geometry(
        mode,
        nVoxel,
        sVoxel,
        dVoxel,
        nDetector,
        sDetector,
        dDetector,
        angles,
        DSD,
        DSO,
        offOriginX,
        offOriginY,
        offOriginZ,
        offDetectorU,
        offDetectorV,
        rotDetectorRoll,
        rotDetectorPitch,
        rotDetectorYaw,
        COR,
        accuracy,
    )
end

function _constant_offset_u(geo::Geometry)
    offset = geo.offDetectorU[1]
    all(x -> isapprox(x, offset; atol = 1f-5, rtol = 0f0), geo.offDetectorU) ||
        throw(ArgumentError("Wang weighting currently supports only constant offDetectorU"))
    return offset
end

function _apply_wang_padding(projection::StridedArray{Float32, 3}, geo::Geometry)
    padwidths = trunc.(Int, 2f0 .* geo.offDetectorU ./ geo.dDetector[1])
    maxpad = maximum(abs, padwidths)
    new_nu = geo.nDetector[1] + maxpad
    padded = zeros(Float32, new_nu, geo.nDetector[2], projection_count(geo))
    new_offsets = Vector{Float32}(undef, projection_count(geo))
    theta = Vector{Float32}(undef, projection_count(geo))

    @inbounds for ia in 1:projection_count(geo)
        left_pad = max(padwidths[ia], 0)
        right_pad = maxpad - left_pad
        padded[(left_pad + 1):(left_pad + geo.nDetector[1]), :, ia] .= projection[:, :, ia]
        new_offsets[ia] = geo.offDetectorU[ia] + Float32(right_pad - left_pad) / 2f0 * geo.dDetector[1]
        theta[ia] = (geo.sDetector[1] / 2f0 - abs(geo.offDetectorU[ia])) * sign(geo.offDetectorU[ia])
    end

    padded_geo = _copy_geometry(
        geo;
        nDetector = (new_nu, geo.nDetector[2]),
        sDetector = (Float32(new_nu) * geo.dDetector[1], geo.sDetector[2]),
        offDetectorU = new_offsets,
    )
    return padded, padded_geo, theta
end

"""
    wang_padding(projection::StridedArray{Float32, 3}, geo::Geometry) -> Tuple{Array{Float32, 3}, Geometry, Vector{Float32}}

Pad projection data along the detector lateral (u) axis and update geometry for offset-detector (displaced detector) CT.

# Arguments
- `projection::StridedArray{Float32, 3}`: Projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT geometry specification with non-zero detector offset (`geo.offDetectorU`).

# Returns
- A tuple `(padded_projection, padded_geometry, theta_redundancy_angles)` where:
  - `padded_projection`: Array with detector column dimension symmetrically padded.
  - `padded_geometry`: Geometry updated with expanded detector width and centered offsets.
  - `theta_redundancy_angles`: Per-projection overlap angular ranges (radians) for subsequent `wang_preweight!`.

# Examples
```julia
padded_proj, padded_geo, theta = wang_padding(proj, geo)
wang_preweight!(padded_proj, padded_geo, theta)
```

# Reference
- Wang, G., "X-ray micro-CT with a displaced detector array", Medical Physics, 29(7), 2002.
"""
function wang_padding(projection::StridedArray{Float32, 3}, geo::Geometry)
    _require_canonical_array(projection, "projection", projection_size(geo))
    return _apply_wang_padding(projection, geo)
end

function _wang_preweight!(projection::StridedArray{Float32, 3}, geo::Geometry, theta::AbstractVector{Float32})
    nu = geo.nDetector[1]
    detector_u = _detector_axis(nu, geo.dDetector[1])
    weights = Vector{Float32}(undef, nu)

    @inbounds for ia in 1:projection_count(geo)
        offset = geo.offDetectorU[ia] + (geo.DSD[ia] / geo.DSO[ia]) * geo.COR[ia]
        abstheta = abs(theta[ia] * geo.DSO[ia] / geo.DSD[ia])
        if iszero(abstheta)
            fill!(weights, 1f0)
            for iv in 1:size(projection, 2), iu in 1:nu
                projection[iu, iv, ia] *= 2f0
            end
            continue
        end
        denom = atan(abstheta / geo.DSO[ia])
        for iu in 1:nu
            t = (detector_u[iu] + abs(offset)) * geo.DSO[ia] / geo.DSD[ia]
            w = 1f0
            if abs(t) <= abstheta
                w = 0.5f0 * (sin((Float32(pi) / 2f0) * atan(t / geo.DSO[ia]) / denom) + 1f0)
            elseif t < -abstheta
                w = 0f0
            end
            weights[iu] = w
        end
        theta[ia] < 0f0 && reverse!(weights)
        for iv in 1:size(projection, 2), iu in 1:nu
            projection[iu, iv, ia] *= weights[iu] * 2f0
        end
    end
    return projection
end

"""
    wang_preweight!(projection::StridedArray{Float32, 3}, geo::Geometry, theta::AbstractVector{<:Real}) -> projection

Apply smooth Wang redundancy weighting in-place across the overlap detector region for offset-detector CT.

# Arguments
- `projection::StridedArray{Float32, 3}`: Padded projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: Corresponding CT geometry.
- `theta::AbstractVector{<:Real}`: Overlap angular range per projection angle (obtained from [`wang_padding`](@ref)).

# Returns
- The mutated `projection` array with redundancy weights applied.

# Examples
```julia
padded_proj, padded_geo, theta = wang_padding(proj, geo)
wang_preweight!(padded_proj, padded_geo, theta)
```
"""
function wang_preweight!(projection::StridedArray{Float32, 3}, geo::Geometry, theta::AbstractVector{<:Real})
    _require_canonical_array(projection, "projection", projection_size(geo))
    length(theta) == projection_count(geo) || throw(ArgumentError("theta length must match projection count"))
    return _wang_preweight!(projection, geo, Float32.(theta))
end

function _nextpow2_geq(n::Int)
    value = 1
    while value < n
        value <<= 1
    end
    return value
end

function _s_function(x::Float32)
    x <= -0.5f0 && return 0f0
    x >= 0.5f0 && return 1f0
    return 0.5f0 * (1f0 + sin(Float32(pi) * x))
end

_b_function(alpha::Float32, delta::Float32, epsilon::Float32) = 2f0 * delta - 2f0 * alpha + epsilon
_b_subf(alpha::Float32, delta::Float32, epsilon::Float32, q::Float32) = q * _b_function(alpha, delta, epsilon)

"""
    parkerweight!(projection::StridedArray{Float32, 3}, geo::Geometry; q = 1f0) -> projection

Apply Parker short-scan redundancy weighting in-place to projection data for cone-beam short-scan CT acquisitions.

# Arguments
- `projection::StridedArray{Float32, 3}`: Projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT geometry specification (angular span must cover at least π + 2 × fan_angle).

# Keyword Arguments
- `q::Real = 1f0`: Parker transition smoothness parameter (default: `1f0`).

# Returns
- The mutated `projection` array with Parker redundancy weights applied.

# Examples
```julia
parkerweight!(proj, geo; q=1f0)
```

# Reference
- Parker, D. L., "Optimal short scan convolution reconstruction for fanbeam CT", Medical Physics, 9(2), 1982.
"""
function parkerweight!(projection::StridedArray{Float32, 3}, geo::Geometry; q::Real = 1f0)
    _require_canonical_array(projection, "projection", projection_size(geo))
    alpha = @view geo.angles[1, :]
    u = _detector_axis(geo.nDetector[1], geo.dDetector[1])
    detalpha = similar(u)
    mean_dsd = sum(geo.DSD) / Float32(length(geo.DSD))
    @inbounds for i in eachindex(u)
        detalpha[i] = -atan(u[i] / mean_dsd)
    end
    delta = abs(detalpha[1] - detalpha[end]) / 2f0
    total_angle = abs(alpha[end] - alpha[1])
    epsilon = max(total_angle - (Float32(pi) + 2f0 * delta), 0f0)
    qf = Float32(q)
    weights = similar(u)

    @inbounds for ia in eachindex(alpha)
        beta = abs(alpha[ia] - alpha[1])
        for iu in eachindex(u)
            a = detalpha[iu]
            denom_a = _b_subf(a, delta, epsilon, qf)
            denom_b = _b_subf(-a, delta, epsilon, qf)
            weights[iu] = 0.5f0 * (
                _s_function(beta / denom_a - 0.5f0) +
                _s_function((beta - 2f0 * delta + 2f0 * a - epsilon) / denom_a + 0.5f0) -
                _s_function((beta - Float32(pi) + 2f0 * a) / denom_b - 0.5f0) -
                _s_function((beta - Float32(pi) - 2f0 * delta - epsilon) / denom_b + 0.5f0)
            )
        end
        for iv in 1:size(projection, 2), iu in 1:size(projection, 1)
            projection[iu, iv, ia] *= weights[iu]
        end
    end
    return projection
end

function _ramp_flat(n::Int)
    nn = collect((-n ÷ 2):(n ÷ 2 - 1))
    h = zeros(Float32, length(nn))
    h[n ÷ 2 + 1] = 0.25f0
    @inbounds for i in eachindex(nn)
        isodd(nn[i]) || continue
        h[i] = -1f0 / (Float32(pi) * Float32(nn[i]))^2
    end
    return h
end

function _dft(signal::Vector{ComplexF32})
    n = length(signal)
    out = Vector{ComplexF32}(undef, n)
    factor = -2f0 * Float32(pi) / Float32(n)
    @inbounds for k in 0:(n - 1)
        acc = 0f0 + 0f0im
        for t in 0:(n - 1)
            angle = factor * Float32(k * t)
            acc += signal[t + 1] * ComplexF32(cos(angle), sin(angle))
        end
        out[k + 1] = acc
    end
    return out
end

function _filter_spectrum(filter_name, kernel::Vector{Float32}, order::Int, d::Float32)
    filter_sym = filter_name === nothing ? :ram_lak : Symbol(lowercase(String(filter_name)))
    f_kernel = abs.(_dft(ComplexF32.(kernel))) .* 2f0
    half = Float32.(f_kernel[1:(order ÷ 2 + 1)])
    w = 2f0 * Float32(pi) .* Float32.(0:length(half)-1) ./ Float32(order)

    if filter_sym === :ram_lak
        nothing
    elseif filter_sym === :shepp_logan
        @inbounds for i in 2:length(half)
            half[i] *= sin(w[i] / (2f0 * d)) / (w[i] / (2f0 * d))
        end
    elseif filter_sym === :cosine
        @inbounds for i in 2:length(half)
            half[i] *= cos(w[i] / (2f0 * d))
        end
    elseif filter_sym === :hamming
        @inbounds for i in 2:length(half)
            half[i] *= 0.54f0 + 0.46f0 * cos(w[i] / d)
        end
    elseif filter_sym === :hann
        @inbounds for i in 2:length(half)
            half[i] *= (1f0 + cos(w[i] / d)) / 2f0
        end
    else
        throw(ArgumentError("unsupported filter: $(filter_name)"))
    end

    @inbounds for i in eachindex(half)
        if w[i] > Float32(pi) * d
            half[i] = 0f0
        end
    end
    return half
end

function _filter_response(filter_name, kernel::Vector{Float32}, order::Int, d::Float32)
    half = _filter_spectrum(filter_name, kernel, order, d)
    return vcat(half, reverse(half[2:end-1]))
end

"""
    filter_response(filter_name, order::Integer; d::Real = 1f0) -> Vector{Float32}

Compute the frequency-domain filter kernel response of length `order` for filtered backprojection.

# Arguments
- `filter_name`: Filter window name (`:ram_lak`, `:shepp_logan`, `:cosine`, `:hamming`, or `:hann`).
- `order::Integer`: Filter length (must be an even integer >= 2).

# Keyword Arguments
- `d::Real = 1f0`: Normalized cutoff frequency factor in `(0, 1]` (default: `1f0`).

# Returns
- A symmetric `Vector{Float32}` of length `order` containing the filter frequency response.

# Examples
```julia
H = filter_response(:ram_lak, 512)
H_hann = filter_response(:hann, 512; d=0.8f0)
```
"""
function filter_response(filter_name, order::Integer; d::Real = 1f0)
    order_int = Int(order)
    order_int >= 2 || throw(ArgumentError("order must be at least 2"))
    iseven(order_int) || throw(ArgumentError("order must be even"))
    return _filter_response(filter_name, _ramp_flat(order_int), order_int, Float32(d))
end

function _distance_weight!(dst::StridedArray{Float32, 3}, src::StridedArray{Float32, 3}, geo::Geometry)
    u = _detector_axis(geo.nDetector[1], geo.dDetector[1])
    v = _detector_axis(geo.nDetector[2], geo.dDetector[2])
    @inbounds for ia in 1:projection_count(geo)
        dsd = geo.DSD[ia]
        for iv in eachindex(v), iu in eachindex(u)
            dst[iu, iv, ia] = src[iu, iv, ia] * (dsd / sqrt(dsd^2 + u[iu]^2 + v[iv]^2))
        end
    end
    return dst
end

function _filter_projections!(projection::StridedArray{Float32, 3}, geo::Geometry; filter = :ram_lak, parker = false)
    parker && parkerweight!(projection, geo)
    nu = geo.nDetector[1]
    nv = geo.nDetector[2]
    nangles = projection_count(geo)
    nrows = nv * nangles
    filt_len = max(64, _nextpow2_geq(2 * nu))
    filt = _filter_spectrum(filter, _ramp_flat(filt_len), filt_len, 1f0)
    nfreq = length(filt)
    padding = (filt_len - nu) ÷ 2
    scale_factor = (geo.DSD[1] / geo.DSO[1]) * (2f0 * Float32(pi) / Float32(projection_count(geo))) / (4f0 * geo.dDetector[1])

    work = zeros(Float32, filt_len, nrows)
    spectra = zeros(ComplexF32, nfreq, nrows)

    @inbounds for ia in 1:nangles, iv in 1:nv
        col = (ia - 1) * nv + iv
        for iu in 1:nu
            work[padding + iu, col] = projection[iu, iv, ia]
        end
    end

    if _fftw_rfft!(spectra, work)
        @inbounds for col in 1:nrows, k in 1:nfreq
            spectra[k, col] *= filt[k]
        end
        _fftw_irfft!(work, spectra)
        invlen = inv(Float32(filt_len))
        @inbounds for ia in 1:nangles, iv in 1:nv
            col = (ia - 1) * nv + iv
            for iu in 1:nu
                projection[iu, iv, ia] = work[padding + iu, col] * invlen * scale_factor
            end
        end
    else
        filt_time = _filter_response(filter, _ramp_flat(filt_len), filt_len, 1f0)
        padded_row = zeros(Float32, filt_len)
        filtered = zeros(Float32, filt_len)
        @inbounds for ia in 1:nangles, iv in 1:nv
            row = @view projection[:, iv, ia]
            fill!(padded_row, 0f0)
            fill!(filtered, 0f0)
            for iu in 1:nu
                padded_row[padding + iu] = row[iu]
            end
            for out_u in 1:filt_len
                acc = 0f0
                for k in 1:filt_len
                    src_u = mod1(out_u - k + 1, filt_len)
                    acc += filt_time[k] * padded_row[src_u]
                end
                filtered[out_u] = acc
            end
            for iu in 1:nu
                row[iu] = filtered[padding + iu] * scale_factor
            end
        end
    end
    return projection
end

"""
    filter_projections!(projection::StridedArray{Float32, 3}, geo::Geometry; filter = :ram_lak, parker = false) -> projection

Filter projection data along detector rows in-place using 1D FFT/convolution with the specified ramp/window filter.

# Arguments
- `projection::StridedArray{Float32, 3}`: Projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: Corresponding `Geometry`.

# Keyword Arguments
- `filter::Symbol = :ram_lak`: Filter window type (`:ram_lak`, `:shepp_logan`, `:cosine`, `:hamming`, or `:hann`) (default: `:ram_lak`).
- `parker::Bool = false`: Apply Parker short-scan redundancy weighting prior to filtering (default: `false`).

# Returns
- The mutated `projection` array after frequency-domain filtering.

# Examples
```julia
filter_projections!(proj, geo; filter=:ram_lak)
```
"""
function filter_projections!(projection::StridedArray{Float32, 3}, geo::Geometry; filter = :ram_lak, parker::Bool = false)
    _require_canonical_array(projection, "projection", projection_size(geo))
    return _filter_projections!(projection, geo; filter = filter, parker = parker)
end

function _fdk_requires_wang(geo::Geometry)
    return any(x -> !iszero(x), geo.offDetectorU)
end

"""
    fdk(projection::StridedArray{Float32, 3}, geo::Geometry; filter = :ram_lak, parker = false, gpu_ids = nothing) -> volume

Feldkamp-Davis-Kress (FDK) cone-beam filtered backprojection reconstruction.

Reconstructs a 3D attenuation volume from cone-beam projection data using distance-weighted filtering
followed by cone-beam backprojection. Supports offset-detector acquisitions (automatic Wang weighting)
and short scans (Parker redundancy weighting).

# Arguments
- `projection::StridedArray{Float32, 3}`: Input projection array of size `(nu, nv, nangles)`.
- `geo::Geometry`: Cone-beam acquisition `Geometry` (`geo.mode === :cone`).

# Keyword Arguments
- `filter::Symbol = :ram_lak`: Ramp filter window (`:ram_lak`, `:shepp_logan`, `:cosine`, `:hamming`, or `:hann`) (default: `:ram_lak`).
- `parker::Bool = false`: Apply Parker redundancy weighting for short scans (default: `false`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) to execute on, or `nothing` for default device (default: `nothing`).

# Returns
- Reconstructed 3D volume `Array{Float32, 3}` of size `(geo.nVoxel[1], geo.nVoxel[2], geo.nVoxel[3])`.

# Examples
```julia
recon = fdk(proj, geo; filter=:ram_lak)
```

# Reference
- Feldkamp, L. A., Davis, L. C., & Kress, J. W., "Practical cone-beam algorithm", JOSA A, 1(6), 1984.
"""
function fdk(projection::StridedArray{Float32, 3}, geo::Geometry; filter = :ram_lak, parker = false, gpu_ids = nothing)
    volume = Array{Float32}(undef, volume_size(geo))
    return fdk!(volume, projection, geo; filter = filter, parker = parker, gpu_ids = gpu_ids)
end

"""
    fdk!(volume::StridedArray{Float32, 3}, projection::StridedArray{Float32, 3}, geo::Geometry; filter = :ram_lak, parker = false, gpu_ids = nothing) -> volume

In-place Feldkamp-Davis-Kress (FDK) cone-beam filtered backprojection reconstruction into `volume`.

# Arguments
- `volume::StridedArray{Float32, 3}`: Pre-allocated destination array of size `(nx, ny, nz)`.
- `projection::StridedArray{Float32, 3}`: Input projection array of size `(nu, nv, nangles)`.
- `geo::Geometry`: Cone-beam acquisition `Geometry` (`geo.mode === :cone`).

# Keyword Arguments
- `filter::Symbol = :ram_lak`: Ramp filter window (`:ram_lak`, `:shepp_logan`, `:cosine`, `:hamming`, or `:hann`) (default: `:ram_lak`).
- `parker::Bool = false`: Apply Parker redundancy weighting for short scans (default: `false`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) to execute on, or `nothing` for default device (default: `nothing`).

# Returns
- The mutated `volume` array of size `(nx, ny, nz)`.

# Examples
```julia
vol_buf = similar(recon)
fdk!(vol_buf, proj, geo; filter=:ram_lak)
```
"""
function fdk!(volume::StridedArray{Float32, 3}, projection::StridedArray{Float32, 3}, geo::Geometry; filter = :ram_lak, parker = false, gpu_ids = nothing)
    geo.mode === :cone || throw(ArgumentError("fdk supports only cone-beam geometry"))
    _require_canonical_array(volume, "volume", volume_size(geo))
    _require_canonical_array(projection, "projection", projection_size(geo))

    working_projection = projection
    working_geo = geo
    if _fdk_requires_wang(geo)
        working_projection, working_geo, theta = _apply_wang_padding(projection, geo)
        _wang_preweight!(working_projection, working_geo, theta)
    end

    work = similar(working_projection)
    _distance_weight!(work, working_projection, working_geo)
    _filter_projections!(work, working_geo; filter = filter, parker = parker)
    return Atb!(volume, work, working_geo; backprojection_type = :fdk, gpu_ids = gpu_ids)
end

"""
    fbp(projection::StridedArray{Float32, 3}, geo::Geometry; filter = :ram_lak, gpu_ids = nothing) -> volume

Filtered backprojection (FBP) for parallel-beam geometry (`geo.mode === :parallel`).

Filters each projection slice along detector rows and backprojects across parallel rays.

# Arguments
- `projection::StridedArray{Float32, 3}`: Input projection array of size `(nu, nv, nangles)`.
- `geo::Geometry`: Parallel-beam acquisition `Geometry` (`geo.mode === :parallel`).

# Keyword Arguments
- `filter::Symbol = :ram_lak`: Ramp filter window (`:ram_lak`, `:shepp_logan`, `:cosine`, `:hamming`, or `:hann`) (default: `:ram_lak`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).

# Returns
- Reconstructed 3D volume `Array{Float32, 3}` of size `(geo.nVoxel[1], geo.nVoxel[2], geo.nVoxel[3])`.

# Examples
```julia
recon = fbp(proj_parallel, geo_parallel; filter=:shepp_logan)
```
"""
function fbp(projection::StridedArray{Float32, 3}, geo::Geometry; filter = :ram_lak, gpu_ids = nothing)
    volume = Array{Float32}(undef, volume_size(geo))
    return fbp!(volume, projection, geo; filter = filter, gpu_ids = gpu_ids)
end

"""
    fbp!(volume::StridedArray{Float32, 3}, projection::StridedArray{Float32, 3}, geo::Geometry; filter = :ram_lak, gpu_ids = nothing) -> volume

In-place filtered backprojection (FBP) for parallel-beam geometry into `volume`.

# Arguments
- `volume::StridedArray{Float32, 3}`: Pre-allocated destination array of size `(nx, ny, nz)`.
- `projection::StridedArray{Float32, 3}`: Input projection array of size `(nu, nv, nangles)`.
- `geo::Geometry`: Parallel-beam acquisition `Geometry` (`geo.mode === :parallel`).

# Keyword Arguments
- `filter::Symbol = :ram_lak`: Ramp filter window (`:ram_lak`, `:shepp_logan`, `:cosine`, `:hamming`, or `:hann`) (default: `:ram_lak`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).

# Returns
- The mutated `volume` array of size `(nx, ny, nz)`.

# Examples
```julia
vol_buf = similar(recon)
fbp!(vol_buf, proj_parallel, geo_parallel; filter=:shepp_logan)
```
"""
function fbp!(volume::StridedArray{Float32, 3}, projection::StridedArray{Float32, 3}, geo::Geometry; filter = :ram_lak, gpu_ids = nothing)
    geo.mode === :parallel || throw(ArgumentError("fbp supports only parallel-beam geometry"))
    _require_canonical_array(volume, "volume", volume_size(geo))
    _require_canonical_array(projection, "projection", projection_size(geo))

    work = copy(projection)
    _filter_projections!(work, geo; filter = filter, parker = false)
    Atb!(volume, work, geo; backprojection_type = :fdk, gpu_ids = gpu_ids)
    volume .*= geo.DSO[1] / geo.DSD[1]
    return volume
end
