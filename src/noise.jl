# CT transmission noise simulation based on Lambert-Beer law and photon/electronic noise
# Reference: TIGRE CTnoise.py / addCTnoise.m
# Im = I0 * exp(-proj / max(proj))
# Im = Poisson(Im) + Normal(mu, sigma)
# noisy_proj = -log(max(Im, 1e-6) / I0) * max(proj)

function _rand_poisson(rng::Random.AbstractRNG, lambda::Float64)
    lambda <= 0.0 && return 0.0
    if lambda < 30.0
        L = exp(-lambda)
        k = 0
        p = 1.0
        while p > L
            k += 1
            p *= rand(rng)
        end
        return Float64(k - 1)
    else
        val = lambda + sqrt(lambda) * randn(rng)
        return val <= 0.0 ? 0.0 : Float64(round(Int, val))
    end
end

function _parse_gaussian(gaussian, gaussian_std)
    if gaussian_std !== nothing
        std_val = Float64(gaussian_std)
        std_val >= 0.0 || throw(ArgumentError("gaussian std must be non-negative, got $std_val"))
        return 0.0, std_val
    elseif gaussian isa Real
        std_val = Float64(gaussian)
        std_val >= 0.0 || throw(ArgumentError("gaussian std must be non-negative, got $std_val"))
        return 0.0, std_val
    elseif gaussian isa Tuple || gaussian isa AbstractVector
        length(gaussian) >= 2 || throw(ArgumentError("gaussian parameter must contain (mean, std)"))
        mu = Float64(gaussian[1])
        std_val = Float64(gaussian[2])
        std_val >= 0.0 || throw(ArgumentError("gaussian std must be non-negative, got $std_val"))
        return mu, std_val
    else
        throw(ArgumentError("gaussian must be a Real or (mean, std) collection"))
    end
end

"""
    add_ct_noise(projections::AbstractArray{<:Real}; I0 = nothing, gaussian = (0f0, 0.5f0), gaussian_std = nothing, poisson = true, rng = Random.default_rng()) -> Array{Float32}
    add_ct_noise!(projections::AbstractArray{Float32}; I0 = nothing, gaussian = (0f0, 0.5f0), gaussian_std = nothing, poisson = true, rng = Random.default_rng()) -> projections
    add_ct_noise!(out::AbstractArray{Float32}, projections::AbstractArray{<:Real}; I0 = nothing, gaussian = (0f0, 0.5f0), gaussian_std = nothing, poisson = true, rng = Random.default_rng()) -> out

Add realistic CT transmission noise to projections using the Beer-Lambert attenuation model.

Simulates incident photon count `I0`, exponential attenuation through the subject, Poisson quantum
noise on transmitted photons, and additive Gaussian electronic readout noise.

# Arguments
- `projections::AbstractArray`: Input clean projection data array `(nu, nv, nangles)`.
- `out::AbstractArray{Float32}`: Destination array to receive noisy projections (for 2-argument `add_ct_noise!`).

# Keyword Arguments
- `I0 = nothing`: Incident photon count before attenuation; if `nothing`, defaults to `60000.0` (or `maximum(projections)/5.0` if larger) (default: `nothing`).
- `gaussian = (0f0, 0.5f0)`: Electronic readout Gaussian noise parameters `(mean, std)` or scalar `std` (default: `(0f0, 0.5f0)`).
- `gaussian_std = nothing`: Convenience keyword to specify Gaussian noise standard deviation with zero mean (default: `nothing`).
- `poisson::Bool = true`: Simulate quantum Poisson photon counting noise (default: `true`).
- `rng::Random.AbstractRNG = Random.default_rng()`: Random number generator (default: `Random.default_rng()`).

# Returns
- 3D array of noisy projection values `Array{Float32}`.

# Examples
```julia
# Clean projection to noisy projection
noisy_proj = add_ct_noise(proj; I0=1e5, gaussian=(0f0, 1f0))

# In-place noise addition
add_ct_noise!(proj; I0=50000)
```

# Reference
- P. Rodriguez and B. Wohlberg, "An Iteratively Reweighted Norm Algorithm for Total Variation Regularization", IEEE ACSSC 2006.
- TIGRE Toolbox `addCTnoise`.
"""
function add_ct_noise!(
    projections::AbstractArray{Float32};
    I0::Union{Real, Nothing} = nothing,
    gaussian::Union{Real, Tuple{Real, Real}, AbstractVector{<:Real}} = (0f0, 0.5f0),
    gaussian_std::Union{Real, Nothing} = nothing,
    poisson::Bool = true,
    rng::Random.AbstractRNG = Random.default_rng(),
)
    isempty(projections) && return projections
    max_proj = maximum(projections)
    max_proj > 0f0 || return projections

    photon_count = if I0 !== nothing
        val = Float64(I0)
        (isfinite(val) && val > 0.0) || (!isfinite(val) && val > 0.0) || throw(ArgumentError("I0 must be positive, got $I0"))
        val
    else
        60000.0
    end
    if I0 === nothing && Float64(max_proj) > photon_count
        photon_count = Float64(max_proj) / 5.0
    end

    g_mu, g_sigma = _parse_gaussian(gaussian, gaussian_std)

    inv_max_proj = 1.0 / Float64(max_proj)

    if isinf(photon_count)
        # Infinite photon count means no Poisson quantum noise, only electronic noise in projection domain
        @inbounds for i in eachindex(projections)
            noise = g_sigma > 0.0 || g_mu != 0.0 ? Float32(g_sigma * randn(rng) + g_mu) : 0f0
            projections[i] += noise
        end
        return projections
    end

    log_I0 = log(photon_count)

    @inbounds for i in eachindex(projections)
        p = Float64(projections[i])
        intensity = photon_count * exp(-p * inv_max_proj)

        if poisson
            intensity = _rand_poisson(rng, intensity)
        end

        if g_sigma > 0.0 || g_mu != 0.0
            intensity += g_sigma * randn(rng) + g_mu
        end

        if intensity <= 0.0
            intensity = 1e-6
        end

        projections[i] = Float32(-(log(intensity) - log_I0) * Float64(max_proj))
    end

    return projections
end

function add_ct_noise!(
    out::AbstractArray{Float32},
    projections::AbstractArray{<:Real};
    kwargs...,
)
    size(out) == size(projections) || throw(DimensionMismatch("size(out) $(size(out)) does not match size(projections) $(size(projections))"))
    copyto!(out, projections)
    return add_ct_noise!(out; kwargs...)
end

"""
    add_ct_noise(projections::AbstractArray{<:Real}; I0 = nothing, gaussian = (0f0, 0.5f0), gaussian_std = nothing, poisson = true, rng = Random.default_rng()) -> Array{Float32}

Non-destructive version of `add_ct_noise!`. Returns a new array with realistic CT transmission noise.

# Arguments
- `projections::AbstractArray{<:Real}`: Input clean projection data array `(nu, nv, nangles)`.

# Keyword Arguments
- `I0 = nothing`: Incident photon count before attenuation; if `nothing`, defaults to `60000.0` (or `maximum(projections)/5.0` if larger) (default: `nothing`).
- `gaussian = (0f0, 0.5f0)`: Electronic readout Gaussian noise parameters `(mean, std)` or scalar `std` (default: `(0f0, 0.5f0)`).
- `gaussian_std = nothing`: Convenience keyword to specify Gaussian noise standard deviation with zero mean (default: `nothing`).
- `poisson::Bool = true`: Simulate quantum Poisson photon counting noise (default: `true`).
- `rng::Random.AbstractRNG = Random.default_rng()`: Random number generator (default: `Random.default_rng()`).

# Returns
- A newly allocated `Array{Float32}` of noisy projection values matching input shape.

# Examples
```julia
noisy_proj = add_ct_noise(proj; I0=1e5, gaussian=(0f0, 1f0))
```
"""
function add_ct_noise(projections::AbstractArray{<:Real}; kwargs...)
    out = copy(Array{Float32}(projections))
    return add_ct_noise!(out; kwargs...)
end
