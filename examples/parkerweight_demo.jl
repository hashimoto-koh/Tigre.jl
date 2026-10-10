using LinearAlgebra
using Tigre

function run_parkerweight_demo(io::IO = stdout)
    geo = Geometry(
        mode = :cone,
        angles = Float32[0, 1.2, 2.4, 3.6],
        nVoxel = (4, 4, 4),
        sVoxel = (40f0, 40f0, 40f0),
        nDetector = (8, 6),
        sDetector = (80f0, 60f0),
        DSO = 1000f0,
        DSD = 1200f0,
    )

    projection = ones(Float32, projection_size(geo)...)
    parkerweight!(projection, geo)

    result = (
        projection_size = size(projection),
        weighted_sum = Float32(sum(projection)),
        weighted_norm = Float32(norm(vec(projection))),
        weighted_min = Float32(minimum(projection)),
        weighted_max = Float32(maximum(projection)),
        samples = Float32[
            projection[1, 1, 1],
            projection[4, 3, 2],
            projection[8, 6, 4],
        ],
    )

    println(io, "Parkerweight demo")
    println(io, "  projection size: ", result.projection_size)
    println(io, "  weighted sum/norm: ", result.weighted_sum, " / ", result.weighted_norm)
    println(io, "  weighted min/max: ", result.weighted_min, " / ", result.weighted_max)
    println(io, "  sample weights: ", result.samples)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_parkerweight_demo()
end
