using LinearAlgebra
using Tigre

include("native_preflight.jl")

function run_ax_atb_demo(io::IO = stdout; preflight::Bool = true)
    preflight && ensure_native_example_ready("ax_atb_demo")

    geo = Geometry(
        mode = :cone,
        angles = Float32[0, 0.4, 0.9, 1.3],
        nVoxel = (32, 32, 32),
        sVoxel = (160f0, 160f0, 160f0),
        nDetector = (48, 32),
        sDetector = (240f0, 160f0),
        DSO = 1000f0,
        DSD = 1200f0,
    )

    volume = zeros(Float32, volume_size(geo)...)
    volume[16, 16, 16] = 1f0
    volume[12, 20, 18] = 0.5f0
    volume[22, 10, 8] = 0.25f0

    projection = Ax(volume, geo)
    backprojection = Atb(projection, geo)

    result = (
        volume_size = size(volume),
        projection_size = size(projection),
        projection_sum = Float32(sum(projection)),
        projection_norm = Float32(norm(vec(projection))),
        backprojection_sum = Float32(sum(backprojection)),
        backprojection_norm = Float32(norm(vec(backprojection))),
        sample_voxels = Float32[
            backprojection[16, 16, 16],
            backprojection[12, 20, 18],
            backprojection[22, 10, 8],
        ],
    )

    println(io, "Ax/Atb demo")
    println(io, "  volume size: ", result.volume_size)
    println(io, "  projection size: ", result.projection_size)
    println(io, "  projection sum/norm: ", result.projection_sum, " / ", result.projection_norm)
    println(io, "  backprojection sum/norm: ", result.backprojection_sum, " / ", result.backprojection_norm)
    println(io, "  sample voxels: ", result.sample_voxels)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_ax_atb_demo()
end
