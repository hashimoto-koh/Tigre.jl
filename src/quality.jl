# Quality measurement metrics for CT reconstructions
# Reference: TIGRE Measure_Quality.py / Quality_measures/

"""
    rmse(reconstructed::AbstractArray{<:Real}, target::AbstractArray{<:Real}) -> Float64

Compute the Root Mean Squared Error (RMSE) between `reconstructed` and `target` ground truth:
```math
\\text{RMSE} = \\sqrt{\\frac{1}{N} \\sum_{i=1}^N (x_i - y_i)^2}
```

# Arguments
- `reconstructed::AbstractArray{<:Real}`: Reconstructed image array.
- `target::AbstractArray{<:Real}`: Reference ground truth array of the same size.

# Returns
- Scalar `Float64` RMSE value.

# Examples
```julia
err = rmse(recon, phantom)
```
"""
function rmse(reconstructed::AbstractArray{<:Real}, target::AbstractArray{<:Real})
    size(reconstructed) == size(target) || throw(DimensionMismatch("array sizes must match: $(size(reconstructed)) vs $(size(target))"))
    isempty(reconstructed) && return 0.0
    accum = 0.0
    @inbounds for i in eachindex(reconstructed, target)
        diff = Float64(reconstructed[i]) - Float64(target[i])
        accum += diff * diff
    end
    return sqrt(accum / Float64(length(reconstructed)))
end

"""
    nrmse(reconstructed::AbstractArray{<:Real}, target::AbstractArray{<:Real}) -> Float64

Compute the Normalized Root Mean Squared Error (nRMSE):
```math
\\text{nRMSE} = \\frac{\\text{RMSE}}{\\sqrt{\\frac{1}{N} \\sum_{i=1}^N y_i^2} + 10^{-5}}
```

# Arguments
- `reconstructed::AbstractArray{<:Real}`: Reconstructed image array.
- `target::AbstractArray{<:Real}`: Reference ground truth array of the same size.

# Returns
- Scalar `Float64` normalized RMSE value.

# Examples
```julia
err = nrmse(recon, phantom)
```
"""
function nrmse(reconstructed::AbstractArray{<:Real}, target::AbstractArray{<:Real})
    size(reconstructed) == size(target) || throw(DimensionMismatch("array sizes must match: $(size(reconstructed)) vs $(size(target))"))
    isempty(reconstructed) && return 0.0
    r = rmse(reconstructed, target)
    norm_target = 0.0
    @inbounds for i in eachindex(target)
        val = Float64(target[i])
        norm_target += val * val
    end
    denom = sqrt(norm_target / Float64(length(target))) + 1e-5
    return r / denom
end

"""
    psnr(reconstructed::AbstractArray{<:Real}, target::AbstractArray{<:Real}; peak = nothing) -> Float64

Compute the Peak Signal-to-Noise Ratio (PSNR) in decibels (dB):
```math
\\text{PSNR} = 20 \\log_{10}\\left(\\frac{\\text{peak}}{\\text{RMSE}}\\right)
```

# Arguments
- `reconstructed::AbstractArray{<:Real}`: Reconstructed image array.
- `target::AbstractArray{<:Real}`: Reference ground truth array of the same size.

# Keyword Arguments
- `peak = nothing`: Peak dynamic range value; if `nothing`, defaults to `maximum(target) - minimum(target)` (or `1.0` if constant) (default: `nothing`).

# Returns
- Scalar `Float64` PSNR value in dB.

# Examples
```julia
p = psnr(recon, phantom; peak=1.0)
```
"""
function psnr(reconstructed::AbstractArray{<:Real}, target::AbstractArray{<:Real}; peak::Union{Real, Nothing} = nothing)
    r = rmse(reconstructed, target)
    r == 0.0 && return Inf
    dr = if peak !== nothing
        Float64(peak)
    else
        val_range = Float64(maximum(target)) - Float64(minimum(target))
        val_range > 0.0 ? val_range : 1.0
    end
    dr > 0.0 || throw(ArgumentError("peak dynamic range must be positive"))
    return 20.0 * log10(dr / r)
end

"""
    ncc(reconstructed::AbstractArray{<:Real}, target::AbstractArray{<:Real}) -> Float64

Compute the Normalized Cross-Correlation (Pearson correlation coefficient) between `reconstructed` and `target`:
```math
\\text{NCC} = \\frac{\\sum (x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum (x_i - \\bar{x})^2 \\sum (y_i - \\bar{y})^2}}
```

# Arguments
- `reconstructed::AbstractArray{<:Real}`: Reconstructed image array.
- `target::AbstractArray{<:Real}`: Reference ground truth array of the same size.

# Returns
- Scalar `Float64` correlation coefficient in `[-1, 1]`.

# Examples
```julia
c = ncc(recon, phantom)
```
"""
function ncc(reconstructed::AbstractArray{<:Real}, target::AbstractArray{<:Real})
    size(reconstructed) == size(target) || throw(DimensionMismatch("array sizes must match: $(size(reconstructed)) vs $(size(target))"))
    N = length(reconstructed)
    N == 0 && return 1.0

    mean_rec = 0.0
    mean_tgt = 0.0
    @inbounds for i in eachindex(reconstructed, target)
        mean_rec += Float64(reconstructed[i])
        mean_tgt += Float64(target[i])
    end
    mean_rec /= Float64(N)
    mean_tgt /= Float64(N)

    num = 0.0
    denom_rec = 0.0
    denom_tgt = 0.0
    @inbounds for i in eachindex(reconstructed, target)
        d_rec = Float64(reconstructed[i]) - mean_rec
        d_tgt = Float64(target[i]) - mean_tgt
        num += d_rec * d_tgt
        denom_rec += d_rec * d_rec
        denom_tgt += d_tgt * d_tgt
    end

    denom = sqrt(denom_rec * denom_tgt)
    denom > 0.0 || return (mean_rec == mean_tgt ? 1.0 : 0.0)
    return num / denom
end

"""
    ssim(reconstructed::AbstractArray{<:Real}, target::AbstractArray{<:Real}; K1 = 0.01, K2 = 0.03, dynamic_range = nothing) -> Float64

Compute the Structural Similarity Index Measure (SSIM) between `reconstructed` and `target` (Wang et al. 2004):
```math
\\text{SSIM} = \\left(\\frac{2 \\bar{x} \\bar{y} + C_1}{\\bar{x}^2 + \\bar{y}^2 + C_1}\\right) \\left(\\frac{2 \\sigma_x \\sigma_y + C_2}{\\sigma_x^2 + \\sigma_y^2 + C_2}\\right) \\left(\\frac{\\sigma_{xy} + C_2/2}{\\sigma_x \\sigma_y + C_2/2}\\right)
```

# Arguments
- `reconstructed::AbstractArray{<:Real}`: Reconstructed image array.
- `target::AbstractArray{<:Real}`: Reference ground truth array of the same size.

# Keyword Arguments
- `K1::Real = 0.01`: Stability constant for luminance comparison (default: `0.01`).
- `K2::Real = 0.03`: Stability constant for contrast and structural comparison (default: `0.03`).
- `dynamic_range = nothing`: Dynamic range `L`; if `nothing`, defaults to `maximum(target) - minimum(target)` (default: `nothing`).

# Returns
- Scalar `Float64` SSIM score in `[-1, 1]`, with 1.0 indicating identical structure.

# Examples
```julia
score = ssim(recon, phantom)
```

# Reference
- Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P., "Image quality assessment: from error visibility to structural similarity", IEEE TIP, 13(4), 2004.
"""
function ssim(
    reconstructed::AbstractArray{<:Real},
    target::AbstractArray{<:Real};
    K1::Real = 0.01,
    K2::Real = 0.03,
    dynamic_range::Union{Real, Nothing} = nothing,
)
    size(reconstructed) == size(target) || throw(DimensionMismatch("array sizes must match: $(size(reconstructed)) vs $(size(target))"))
    N = length(reconstructed)
    N == 0 && return 1.0
    N > 1 || return (reconstructed[1] == target[1] ? 1.0 : 0.0)

    d = if dynamic_range !== nothing
        Float64(dynamic_range)
    else
        val_range = Float64(maximum(target)) - Float64(minimum(target))
        val_range > 0.0 ? val_range : 1.0
    end

    C1 = (Float64(K1) * d)^2
    C2 = (Float64(K2) * d)^2

    mean_x = 0.0
    mean_y = 0.0
    @inbounds for i in eachindex(reconstructed, target)
        mean_x += Float64(reconstructed[i])
        mean_y += Float64(target[i])
    end
    mean_x /= Float64(N)
    mean_y /= Float64(N)

    var_x = 0.0
    var_y = 0.0
    cov_xy = 0.0
    @inbounds for i in eachindex(reconstructed, target)
        dx = Float64(reconstructed[i]) - mean_x
        dy = Float64(target[i]) - mean_y
        var_x += dx * dx
        var_y += dy * dy
        cov_xy += dx * dy
    end
    inv_N1 = 1.0 / Float64(N - 1)
    var_x *= inv_N1
    var_y *= inv_N1
    cov_xy *= inv_N1

    std_x = sqrt(max(0.0, var_x))
    std_y = sqrt(max(0.0, var_y))

    # Luminance
    l = (2.0 * mean_x * mean_y + C1) / (mean_x^2 + mean_y^2 + C1)
    # Contrast
    c = (2.0 * std_x * std_y + C2) / (var_x + var_y + C2)
    # Structure
    s = (cov_xy + C2 / 2.0) / (std_x * std_y + C2 / 2.0)

    return l * c * s
end

"""
    uqi(reconstructed::AbstractArray{<:Real}, target::AbstractArray{<:Real}) -> Float64

Compute the Universal Quality Index (UQI) (Wang & Bovik 2002):
```math
\\text{UQI} = \\frac{2 \\sigma_{xy}}{\\sigma_x^2 + \\sigma_y^2} \\cdot \\frac{2 \\bar{x} \\bar{y}}{\\bar{x}^2 + \\bar{y}^2}
```

# Arguments
- `reconstructed::AbstractArray{<:Real}`: Reconstructed image array.
- `target::AbstractArray{<:Real}`: Reference ground truth array of the same size.

# Returns
- Scalar `Float64` UQI score in `[-1, 1]`.

# Examples
```julia
score = uqi(recon, phantom)
```

# Reference
- Wang, Z., & Bovik, A. C., "A universal image quality index", IEEE Signal Process. Lett., 9(3), 2002.
"""
function uqi(reconstructed::AbstractArray{<:Real}, target::AbstractArray{<:Real})
    size(reconstructed) == size(target) || throw(DimensionMismatch("array sizes must match: $(size(reconstructed)) vs $(size(target))"))
    N = length(reconstructed)
    N == 0 && return 1.0
    N > 1 || return (reconstructed[1] == target[1] ? 1.0 : 0.0)

    mean_x = 0.0
    mean_y = 0.0
    @inbounds for i in eachindex(reconstructed, target)
        mean_x += Float64(reconstructed[i])
        mean_y += Float64(target[i])
    end
    mean_x /= Float64(N)
    mean_y /= Float64(N)

    var_x = 0.0
    var_y = 0.0
    cov_xy = 0.0
    @inbounds for i in eachindex(reconstructed, target)
        dx = Float64(reconstructed[i]) - mean_x
        dy = Float64(target[i]) - mean_y
        var_x += dx * dx
        var_y += dy * dy
        cov_xy += dx * dy
    end
    inv_N1 = 1.0 / Float64(N - 1)
    var_x *= inv_N1
    var_y *= inv_N1
    cov_xy *= inv_N1

    denom_var = var_x + var_y
    denom_mean = mean_x^2 + mean_y^2

    (denom_var > 0.0 && denom_mean > 0.0) || return (reconstructed ≈ target ? 1.0 : 0.0)

    front = (2.0 * cov_xy) / denom_var
    back = (2.0 * mean_x * mean_y) / denom_mean
    return front * back
end

"""
    measure_quality(reconstructed::AbstractArray{<:Real}, target::AbstractArray{<:Real}; metrics = (:rmse, :nrmse, :psnr, :ssim, :uqi, :ncc), peak = nothing, K1 = 0.01, K2 = 0.03, dynamic_range = nothing) -> NamedTuple

Evaluate one or more quality metrics comparing `reconstructed` image against `target` ground truth.

# Arguments
- `reconstructed::AbstractArray{<:Real}`: Reconstructed image volume or slice.
- `target::AbstractArray{<:Real}`: Reference ground truth array of matching shape.

# Keyword Arguments
- `metrics = (:rmse, :nrmse, :psnr, :ssim, :uqi, :ncc)`: Tuple or vector of metric symbols to compute (default: `(:rmse, :nrmse, :psnr, :ssim, :uqi, :ncc)`).
- `peak = nothing`: Peak dynamic range value used for PSNR calculation (default: `nothing`).
- `K1::Real = 0.01`: Stability parameter for SSIM luminance comparison (default: `0.01`).
- `K2::Real = 0.03`: Stability parameter for SSIM contrast/structural comparison (default: `0.03`).
- `dynamic_range = nothing`: Dynamic range value used for SSIM calculation (default: `nothing`).

# Returns
- A `NamedTuple` mapping each requested metric symbol to its computed `Float64` value.

# Examples
```julia
q = measure_quality(recon, phantom; metrics=(:rmse, :psnr, :ssim))
println("RMSE: ", q.rmse, ", PSNR: ", q.psnr, " dB, SSIM: ", q.ssim)
```
"""
function measure_quality(
    reconstructed::AbstractArray{<:Real},
    target::AbstractArray{<:Real};
    metrics = (:rmse, :nrmse, :psnr, :ssim, :uqi, :ncc),
    kwargs...,
)
    res = Dict{Symbol, Float64}()
    metric_syms = [Symbol(lowercase(string(m))) for m in metrics]

    for m in metric_syms
        if m in (:rmse, :rms)
            res[:rmse] = rmse(reconstructed, target)
        elseif m in (:nrmse, :nrms)
            res[:nrmse] = nrmse(reconstructed, target)
        elseif m in (:psnr,)
            res[:psnr] = psnr(reconstructed, target; kwargs...)
        elseif m in (:ssim, :mssim)
            res[:ssim] = ssim(reconstructed, target; kwargs...)
        elseif m in (:uqi,)
            res[:uqi] = uqi(reconstructed, target)
        elseif m in (:ncc, :cc)
            res[:ncc] = ncc(reconstructed, target)
        else
            throw(ArgumentError("unrecognized quality metric: $m. Supported: rmse, nrmse, psnr, ssim, uqi, ncc"))
        end
    end

    return NamedTuple{Tuple(metric_syms)}([res[m] for m in metric_syms])
end
