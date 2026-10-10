using LinearAlgebra
using Tigre

include("native_preflight.jl")

function run_fdk_demo(io::IO = stdout; preflight::Bool = true)
    preflight && ensure_native_example_ready("fdk_demo")

    geo = Geometry(
        mode = :cone,
        angles = Float32[0, 0.3, 0.8, 1.2],
        nVoxel = (32, 32, 32),
        sVoxel = (160f0, 160f0, 160f0),
        nDetector = (48, 32),
        sDetector = (240f0, 160f0),
        DSO = 1000f0,
        DSD = 1200f0,
        offDetector = [10f0, 0f0],
    )

    phantom = zeros(Float32, volume_size(geo)...)
    phantom[15:18, 15:18, 15:18] .= 1f0
    phantom[8:10, 22:24, 10:12] .= 0.5f0

    projection = Ax(phantom, geo)
    recon = fdk(projection, geo; filter = :hann)

    result = (
        projection_size = size(projection),
        recon_size = size(recon),
        recon_sum = Float32(sum(recon)),
        recon_norm = Float32(norm(vec(recon))),
        recon_min = Float32(minimum(recon)),
        recon_max = Float32(maximum(recon)),
        center_voxel = Float32(recon[16, 16, 16]),
    )

    println(io, "FDK demo")
    println(io, "  projection size: ", result.projection_size)
    println(io, "  recon size: ", result.recon_size)
    println(io, "  recon sum/norm: ", result.recon_sum, " / ", result.recon_norm)
    println(io, "  recon min/max: ", result.recon_min, " / ", result.recon_max)
    println(io, "  center voxel: ", result.center_voxel)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_fdk_demo()
end
