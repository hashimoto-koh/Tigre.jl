using LinearAlgebra
using Tigre

include("native_preflight.jl")

function run_parallel_fbp3d_demo(io::IO = stdout; preflight::Bool = true)
    preflight && ensure_native_example_ready("parallel_fbp3d_demo")

    n = 32
    nangles = 45
    angles = Float32.(range(0, Float32(pi); length = nangles + 1)[1:end-1])

    geo = Geometry(
        mode = :parallel,
        angles = angles,
        nVoxel = (n, n, n),
        sVoxel = (64f0, 64f0, 64f0),
        nDetector = (n, n),
        sDetector = (64f0, 64f0),
        # In parallel-beam mode, distances are not used for beam divergence.
        # Set DSO == DSD to keep the scaling factor inside fbp neutral.
        DSO = 1f0,
        DSD = 1f0,
    )

    phantom = zeros(Float32, volume_size(geo)...)
    phantom[12:21, 12:21, 12:21] .= 1f0
    phantom[8:11, 22:25, 10:16] .= 0.5f0

    projection = Ax(phantom, geo)
    recon = fbp(projection, geo; filter = :hann)

    result = (
        volume_size = size(phantom),
        projection_size = size(projection),
        recon_size = size(recon),
        projection_sum = Float32(sum(projection)),
        recon_sum = Float32(sum(recon)),
        recon_norm = Float32(norm(vec(recon))),
        recon_min = Float32(minimum(recon)),
        recon_max = Float32(maximum(recon)),
        center_voxel = Float32(recon[n ÷ 2, n ÷ 2, n ÷ 2]),
    )

    println(io, "Parallel-beam 3D FBP demo")
    println(io, "  volume size: ", result.volume_size)
    println(io, "  projection size: ", result.projection_size, "  # [u, v, angle]")
    println(io, "  recon size: ", result.recon_size, "  # [x, y, z]")
    println(io, "  projection sum: ", result.projection_sum)
    println(io, "  recon sum/norm: ", result.recon_sum, " / ", result.recon_norm)
    println(io, "  recon min/max: ", result.recon_min, " / ", result.recon_max)
    println(io, "  center voxel: ", result.center_voxel)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_parallel_fbp3d_demo()
end
