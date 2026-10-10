using LinearAlgebra
using Tigre

function run_wang_padding_demo(io::IO = stdout)
    geo = Geometry(
        mode = :cone,
        angles = Float32[0, 0.3, 0.8, 1.2],
        nVoxel = (4, 4, 4),
        sVoxel = (40f0, 40f0, 40f0),
        nDetector = (8, 6),
        sDetector = (80f0, 60f0),
        DSO = 1000f0,
        DSD = 1200f0,
        offDetector = Float32[10 0 -10 10; 0 0 0 0],
    )

    projection = ones(Float32, projection_size(geo)...)
    padded, padded_geo, theta = wang_padding(projection, geo)
    wang_preweight!(padded, padded_geo, theta)

    result = (
        original_size = size(projection),
        padded_size = size(padded),
        theta = copy(theta),
        padded_offsets = copy(padded_geo.offDetectorU),
        weighted_sum = Float32(sum(padded)),
        weighted_norm = Float32(norm(vec(padded))),
        weighted_min = Float32(minimum(padded)),
        weighted_max = Float32(maximum(padded)),
        sample_weights = Float32[
            padded[1, 1, 1],
            padded[3, 2, 2],
            padded[end, end, end],
        ],
    )

    println(io, "Wang padding demo")
    println(io, "  original/padded size: ", result.original_size, " / ", result.padded_size)
    println(io, "  theta: ", result.theta)
    println(io, "  padded offsets: ", result.padded_offsets)
    println(io, "  weighted sum/norm: ", result.weighted_sum, " / ", result.weighted_norm)
    println(io, "  weighted min/max: ", result.weighted_min, " / ", result.weighted_max)
    println(io, "  sample weights: ", result.sample_weights)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_wang_padding_demo()
end
