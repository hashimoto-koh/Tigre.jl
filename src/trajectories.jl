# Acquisition trajectories and non-standard CT scanning geometries
# Reference: TIGRE common_geometry.py

"""
    rotation_from_vecs(v1::AbstractVector{<:Real}, v2::AbstractVector{<:Real}) -> Matrix{Float64}

Compute the 3×3 rotation matrix `R` that rotates 3D vector `v1` to align with `v2`
using Rodrigues' rotation formula.
"""
function rotation_from_vecs(v1::AbstractVector{<:Real}, v2::AbstractVector{<:Real})
    length(v1) == 3 && length(v2) == 3 || throw(ArgumentError("Vectors must be length 3"))
    u = Float64.(v1) / norm(v1)
    w = Float64.(v2) / norm(v2)
    c = dot(u, w)
    if c >= 1.0 - 1e-9
        return Matrix{Float64}(I, 3, 3)
    elseif c <= -1.0 + 1e-9
        axis = abs(u[1]) < 0.8 ? cross(u, [1.0, 0.0, 0.0]) : cross(u, [0.0, 1.0, 0.0])
        axis = axis / norm(axis)
        K = [0.0 -axis[3] axis[2]; axis[3] 0.0 -axis[1]; -axis[2] axis[1] 0.0]
        return Matrix{Float64}(I, 3, 3) + 2.0 * (K * K)
    else
        v = cross(u, w)
        K = [0.0 -v[3] v[2]; v[3] 0.0 -v[1]; -v[2] v[1] 0.0]
        return Matrix{Float64}(I, 3, 3) + K + (K * K) / (1.0 + c)
    end
end

"""
    rot_to_euler(R::AbstractMatrix{<:Real}, order::Symbol = :zyz) -> NTuple{3, Float64}

Extract Euler angles from a 3×3 rotation matrix `R`. Supported orders: `:zyz` (intrinsic)
and `:xyz` (extrinsic roll-pitch-yaw).
"""
function rot_to_euler(R::AbstractMatrix{<:Real}, order::Symbol = :zyz)
    size(R) == (3, 3) || throw(ArgumentError("R must be 3×3 matrix"))
    if order === :zyz
        beta = acos(clamp(Float64(R[3, 3]), -1.0, 1.0))
        sin_beta = sin(beta)
        if abs(sin_beta) > 1e-8
            alpha = atan(Float64(R[2, 3]), Float64(R[1, 3]))
            gamma = atan(Float64(R[3, 2]), -Float64(R[3, 1]))
        else
            alpha = atan(Float64(R[2, 1]), Float64(R[1, 1]))
            gamma = 0.0
        end
        return (alpha, beta, gamma)
    elseif order === :xyz
        theta = asin(clamp(Float64(R[1, 3]), -1.0, 1.0))
        cos_theta = cos(theta)
        if abs(cos_theta) > 1e-8
            psi = atan(-Float64(R[2, 3]), Float64(R[3, 3]))
            phi = atan(-Float64(R[1, 2]), Float64(R[1, 1]))
        else
            psi = atan(Float64(R[3, 2]), Float64(R[2, 2]))
            phi = 0.0
        end
        return (psi, theta, phi)
    else
        throw(ArgumentError("Unsupported Euler order :$order. Use :zyz or :xyz."))
    end
end

"""
    euler_from_vecs(v1, v2; order = :zyz)

Calculate Euler angles describing the rotation aligning vector `v1` to `v2`.
"""
function euler_from_vecs(v1::AbstractVector{<:Real}, v2::AbstractVector{<:Real}; order::Symbol = :zyz)
    R = rotation_from_vecs(v1, v2)
    angles = rot_to_euler(R, order)
    return [abs(a) < 2e-8 ? 0.0 : a for a in angles]
end

"""
    static_detector_geo(geo::Geometry, scan_angles; rot = 0f0)

Generate a `Geometry` for digital breast tomosynthesis (DBT) or static detector scanning
where the flat-panel detector remains fixed and the X-ray tube rotates along an arc.

# Arguments
- `geo`: Base cone-beam `Geometry`.
- `scan_angles`: Vector of scan angles in radians (off the normal axis between source and detector).
- `rot`: Additional rotation (in radians) of both source and detector around the origin (default: 0).
"""
function static_detector_geo(
    geo::Geometry,
    scan_angles::AbstractVector{<:Real};
    rot::Real = 0f0,
)
    geo.mode === :cone || throw(ArgumentError("static_detector_geo is only defined for :cone beam geometry"))
    angles_rad = Float32.(scan_angles)
    nangles = length(angles_rad)
    rot_rad = Float32(rot)

    mean_dso = sum(geo.DSO) / length(geo.DSO)
    mean_dsd = sum(geo.DSD) / length(geo.DSD)
    R = mean_dsd - mean_dso

    new_dsd = mean_dso .+ R .* cos.(angles_rad)
    new_dso = fill(mean_dso, nangles)

    new_off_u = zeros(Float32, nangles)
    new_off_v = R .* sin.(angles_rad)

    new_rot_roll = zeros(Float32, nangles)
    new_rot_pitch = zeros(Float32, nangles)
    new_rot_yaw = -angles_rad

    new_angles = Matrix{Float32}(undef, 3, nangles)
    new_angles[1, :] .= angles_rad .+ rot_rad
    new_angles[2, :] .= 0f0
    new_angles[3, :] .= 0f0

    return Geometry(
        geo.mode,
        geo.nVoxel,
        geo.sVoxel,
        geo.dVoxel,
        geo.nDetector,
        geo.sDetector,
        geo.dDetector,
        new_angles,
        new_dsd,
        new_dso,
        zeros(Float32, nangles),
        zeros(Float32, nangles),
        zeros(Float32, nangles),
        new_off_u,
        new_off_v,
        new_rot_roll,
        new_rot_pitch,
        new_rot_yaw,
        zeros(Float32, nangles),
        geo.accuracy,
    )
end

"""
    static_det_linear_source_geo(geo::Geometry, s_pos; s_rot = 0f0, rot = 0f0)

Generate a `Geometry` where the detector remains fixed and the X-ray tube travels
along a straight line trajectory.

# Arguments
- `geo`: Base cone-beam `Geometry`.
- `s_pos`: Vector of source displacement positions along the linear trajectory in mm.
- `s_rot`: Rotation angle of linear trajectory relative to detector normal (radians, default 0).
- `rot`: Global rotation of system around origin (radians, default 0).
"""
function static_det_linear_source_geo(
    geo::Geometry,
    s_pos::AbstractVector{<:Real};
    s_rot::Real = 0f0,
    rot::Real = 0f0,
)
    geo.mode === :cone || throw(ArgumentError("static_det_linear_source_geo requires :cone beam geometry"))
    pos = Float32.(s_pos)
    nangles = length(pos)
    s_rot_rad = Float32(s_rot)
    rot_rad = Float32(rot)

    mean_dso = sum(geo.DSO) / length(geo.DSO)
    mean_dsd = sum(geo.DSD) / length(geo.DSD)
    R = mean_dsd - mean_dso

    ang = atan.(pos .* cos(s_rot_rad), mean_dso .+ pos .* sin(s_rot_rad))

    new_dso = sqrt.((pos .* cos(s_rot_rad)).^2 .+ (mean_dso .+ pos .* sin(s_rot_rad)).^2)
    new_dsd = sqrt.((pos .* cos(s_rot_rad)).^2 .+ (mean_dsd .+ pos .* sin(s_rot_rad)).^2)

    new_off_u = zeros(Float32, nangles)
    new_off_v = R .* sin.(ang)

    new_rot_roll = zeros(Float32, nangles)
    new_rot_pitch = zeros(Float32, nangles)
    new_rot_yaw = -ang

    new_angles = Matrix{Float32}(undef, 3, nangles)
    new_angles[1, :] .= ang .+ rot_rad
    new_angles[2, :] .= 0f0
    new_angles[3, :] .= 0f0

    return Geometry(
        geo.mode,
        geo.nVoxel,
        geo.sVoxel,
        geo.dVoxel,
        geo.nDetector,
        geo.sDetector,
        geo.dDetector,
        new_angles,
        new_dsd,
        new_dso,
        zeros(Float32, nangles),
        zeros(Float32, nangles),
        zeros(Float32, nangles),
        new_off_u,
        new_off_v,
        new_rot_roll,
        new_rot_pitch,
        new_rot_yaw,
        zeros(Float32, nangles),
        geo.accuracy,
    )
end

"""
    arbitrary_source_detector_geo(geo::Geometry, s_pos::AbstractMatrix{<:Real}, [d_pos], [d_rot])

Calculate TIGRE `Geometry` parameters for arbitrary 3D movements of source and detector (e.g. robotic arm or C-arm CT).

# Arguments
- `geo`: Base `Geometry` supplying voxel and detector physical dimensions.
- `s_pos`: N×3 matrix of source coordinates `(x, y, z)` in mm.
- `d_pos`: Optional N×3 matrix of detector center coordinates `(x, y, z)` in mm (default: standard position on negative x-axis).
- `d_rot`: Optional N×3 matrix of detector rotation angles `(roll, pitch, yaw)` in radians (default: 0).
"""
function arbitrary_source_detector_geo(
    geo::Geometry,
    s_pos::AbstractMatrix{<:Real},
    d_pos::Union{AbstractMatrix{<:Real}, Nothing} = nothing,
    d_rot::Union{AbstractMatrix{<:Real}, Nothing} = nothing,
)
    size(s_pos, 2) == 3 || throw(ArgumentError("s_pos must be N×3 matrix"))
    n = size(s_pos, 1)

    mean_dso = sum(geo.DSO) / length(geo.DSO)
    mean_dsd = sum(geo.DSD) / length(geo.DSD)

    det_pos = if d_pos !== nothing
        size(d_pos) == (n, 3) || throw(DimensionMismatch("d_pos must be N×3 matching s_pos"))
        Float64.(d_pos)
    else
        hcat(fill(Float64(mean_dso - mean_dsd), n), zeros(n), zeros(n))
    end

    det_rot = if d_rot !== nothing
        size(d_rot) == (n, 3) || throw(DimensionMismatch("d_rot must be N×3 matching s_pos"))
        Float64.(d_rot)
    else
        zeros(Float64, n, 3)
    end

    src_pos = Float64.(s_pos)

    dsd_vec = zeros(Float32, n)
    dso_vec = zeros(Float32, n)
    angles_mat = Matrix{Float32}(undef, 3, n)
    off_u = zeros(Float32, n)
    off_v = zeros(Float32, n)
    rot_roll = zeros(Float32, n)
    rot_pitch = zeros(Float32, n)
    rot_yaw = zeros(Float32, n)

    for i in 1:n
        s_i = src_pos[i, :]
        d_i = det_pos[i, :]
        rs = norm(s_i)
        rd = norm(d_i)
        ns = s_i / rs
        nd = d_i / rd

        # Source Euler angles away from x-axis in ZYZ order
        s0 = [s_i[1], 0.0, 0.0]
        s_ang = euler_from_vecs(s0, s_i; order = :zyz)
        angles_mat[:, i] .= Float32.(s_ang)

        # Detector ray angle and offset
        d_cross = cross(-ns, nd)
        d_ang = asin.(clamp.(d_cross, -1.0, 1.0))
        pd = d_cross .* rd

        off_u[i] = Float32(pd[2])
        off_v[i] = Float32(-pd[3])

        rot_roll[i] = Float32(d_ang[1] + det_rot[i, 1])
        rot_pitch[i] = Float32(d_ang[2] + det_rot[i, 2])
        rot_yaw[i] = Float32(d_ang[3] + det_rot[i, 3])

        dso_vec[i] = Float32(rs)
        dsd_vec[i] = Float32(norm(s_i - d_i))
    end

    return Geometry(
        geo.mode,
        geo.nVoxel,
        geo.sVoxel,
        geo.dVoxel,
        geo.nDetector,
        geo.sDetector,
        geo.dDetector,
        angles_mat,
        dsd_vec,
        dso_vec,
        zeros(Float32, n),
        zeros(Float32, n),
        zeros(Float32, n),
        off_u,
        off_v,
        rot_roll,
        rot_pitch,
        rot_yaw,
        zeros(Float32, n),
        geo.accuracy,
    )
end
