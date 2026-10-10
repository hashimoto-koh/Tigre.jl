function _normalize_phantom_size(size_out)
    if size_out isa Integer
        n = Int(size_out)
        n > 0 || throw(ArgumentError("size_out must be a positive integer"))
        return (n, n, n)
    end
    data = collect(size_out)
    length(data) == 3 || throw(ArgumentError("size_out must be an integer or a tuple/vector of length 3"))
    dims = ntuple(i -> Int(data[i]), 3)
    any(<=(0), dims) && throw(ArgumentError("all elements of size_out must be positive"))
    return dims
end

function _normalize_phantom_type(phantom_type)
    s = lowercase(replace(String(phantom_type), "-" => "_"))
    if s == "kak_slaney"
        return :kak_slaney, 0
    elseif s == "yu_ye_wang"
        return :yu_ye_wang, 0
    elseif s == "toft_schabel"
        return :toft_schabel, 1
    end
    throw(ArgumentError("phantom_type must be :kak_slaney, :yu_ye_wang, or :toft_schabel"))
end

function _kak_slaney_ellipsoids()
    return Float32[
        0.6900 0.920 0.900 0.000 0.000 0.000 0 0 0 2.00
        0.6624 0.874 0.880 0.000 0.000 0.000 0 0 0 -0.98
        0.4100 0.160 0.210 -0.220 0.000 -0.250 108 0 0 -0.02
        0.3100 0.110 0.220 0.220 0.000 -0.250 72 0 0 -0.02
        0.2100 0.250 0.500 0.000 0.350 -0.250 0 0 0 0.02
        0.0460 0.046 0.046 0.000 0.100 -0.250 0 0 0 0.02
        0.0460 0.023 0.020 -0.080 -0.650 -0.250 0 0 0 0.01
        0.0460 0.023 0.020 0.060 -0.650 -0.250 90 0 0 0.01
        0.0560 0.040 0.100 0.060 -0.105 0.625 90 0 0 0.02
        0.0560 0.056 0.100 0.000 0.100 0.625 0 0 0 -0.02
    ]
end

function _yu_ye_wang_ellipsoids()
    ellipsoids = _kak_slaney_ellipsoids()
    ellipsoids[:, 10] .= Float32[1.00, -0.80, -0.2, -0.2, 0.2, 0.2, 0.1, 0.1, 0.2, -0.2]
    return ellipsoids
end

function _toft_schabel_ellipsoids()
    return Float32[
        0.6900 0.9200 0.810 0 0 0 0 0 0 1.0
        0.6624 0.8740 0.780 0 -0.0184 0 0 0 0 -0.8
        0.1100 0.3100 0.220 0.22 0 0 -18 0 10 -0.2
        0.1600 0.4100 0.280 -0.22 0 0 18 0 10 -0.2
        0.2100 0.2500 0.410 0 0.35 -0.15 0 0 0 0.1
        0.0460 0.0460 0.050 0 0.1 0.25 0 0 0 0.1
        0.0460 0.0460 0.050 0 -0.1 0.25 0 0 0 0.1
        0.0460 0.0230 0.050 -0.08 -0.605 0 0 0 0 0.1
        0.0230 0.0230 0.020 0 -0.606 0 0 0 0 0.1
        0.0230 0.0460 0.020 0.06 -0.605 0 0 0 0 0.1
    ]
end

function _phantom_ellipsoids(kind::Symbol)
    kind === :kak_slaney && return _kak_slaney_ellipsoids()
    kind === :yu_ye_wang && return _yu_ye_wang_ellipsoids()
    kind === :toft_schabel && return _toft_schabel_ellipsoids()
    error("unreachable phantom type")
end

"""
    shepp_logan_3d(size_out = 128; phantom_type = :yu_ye_wang, get_ellipsoids = false) -> Array{Float32, 3}

Generate a 3D Shepp-Logan numerical phantom array of size `size_out`.

# Arguments
- `size_out = 128`: Cube dimension `N::Integer` or 3-tuple `(nx, ny, nz)` (default: `128`).

# Keyword Arguments
- `phantom_type::Symbol = :yu_ye_wang`: Variant of Shepp-Logan phantom (default: `:yu_ye_wang`):
  - `:yu_ye_wang`: Modified Shepp-Logan phantom with enhanced contrast (Yu, Ye, Wang 2005).
  - `:kak_slaney`: Classic Shepp-Logan phantom definition (Kak & Slaney 1988).
  - `:toft_schabel`: High-contrast modified Shepp-Logan phantom (Toft 1996 / Schabel 2006).
- `get_ellipsoids::Bool = false`: If `true`, returns `(volume, ellipsoids)` tuple where `ellipsoids` is the parameter matrix (default: `false`).

# Returns
- 3D phantom `Array{Float32, 3}` of size `(nx, ny, nz)`, or `(volume, ellipsoids)` if `get_ellipsoids=true`.

# Examples
```julia
# 256^3 cubic phantom
phantom = shepp_logan_3d(256)

# Non-cubic volume with classic Kak-Slaney contrast
phantom_ks = shepp_logan_3d((128, 128, 64); phantom_type=:kak_slaney)
```
"""
function shepp_logan_3d(size_out = 128; phantom_type = :yu_ye_wang, get_ellipsoids::Bool = false)
    nvoxel = _normalize_phantom_size(size_out)
    kind, formula = _normalize_phantom_type(phantom_type)
    ellipsoids = _phantom_ellipsoids(kind)
    volume = zeros(Float32, nvoxel)

    xs = collect(range(-1f0, 1f0; length = nvoxel[1]))
    ys = collect(range(-1f0, 1f0; length = nvoxel[2]))
    zs = collect(range(-1f0, 1f0; length = nvoxel[3]))

    @inbounds for ie in axes(ellipsoids, 1)
        a2 = ellipsoids[ie, 1]^2
        b2 = ellipsoids[ie, 2]^2
        c2 = ellipsoids[ie, 3]^2
        x0 = ellipsoids[ie, 4]
        y0 = ellipsoids[ie, 5]
        z0 = ellipsoids[ie, 6]
        phi1 = ellipsoids[ie, 7] * Float32(pi) / 180f0
        phi2 = ellipsoids[ie, 8] * Float32(pi) / 180f0
        phi3 = ellipsoids[ie, 9] * Float32(pi) / 180f0
        amplitude = ellipsoids[ie, 10]

        c1, s1 = cos(phi1), sin(phi1)
        c2r, s2 = cos(phi2), sin(phi2)
        c3, s3 = cos(phi3), sin(phi3)

        a11 = c2r
        a12 = -s2 * c1
        a13 = s2 * s1
        a21 = c3 * s2
        a22 = -s3 * s1 + c3 * c2r * c1
        a23 = -s3 * c1 - c3 * c2r * s1
        a31 = s3 * s2
        a32 = c3 * s1 + s3 * c2r * c1
        a33 = c3 * c1 - s3 * c2r * s1

        for iz in eachindex(zs), iy in eachindex(ys), ix in eachindex(xs)
            x = xs[ix]
            y = ys[iy]
            z = zs[iz]
            if formula == 0
                dz = z - z0
                dy = y - y0
                dx = x - x0
                rz = a11 * dz + a12 * dy + a13 * dx
                ry = a21 * dz + a22 * dy + a23 * dx
                rx = a31 * dz + a32 * dy + a33 * dx
                inside = rx^2 / a2 + ry^2 / b2 + rz^2 / c2 <= 1f0
            else
                rz = a11 * z + a12 * y + a13 * x
                ry = a21 * z + a22 * y + a23 * x
                rx = a31 * z + a32 * y + a33 * x
                inside = (rx - x0)^2 / a2 + (ry - y0)^2 / b2 + (rz - z0)^2 / c2 <= 1f0
            end
            inside && (volume[ix, iy, iz] += amplitude)
        end
    end

    return get_ellipsoids ? (volume, ellipsoids) : volume
end
