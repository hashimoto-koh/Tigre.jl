using Tigre

function run_filtering_demo(io::IO = stdout)
    geo = Geometry(
        mode = :cone,
        angles = Float32[0, 0.7, 1.4, 2.1],
        nVoxel = (4, 4, 4),
        sVoxel = (40f0, 40f0, 40f0),
        nDetector = (8, 6),
        sDetector = (80f0, 60f0),
        DSO = 1000f0,
        DSD = 1200f0,
    )
    projection = fill(1f0, projection_size(geo)...)
    response = filter_response(:hann, 64)
    filtered = copy(projection)
    filter_projections!(filtered, geo; filter = :hann, parker = false)

    println(io, "Filtering demo")
    println(io, "  response length: ", length(response))
    println(io, "  projection size: ", size(filtered))
    println(io, "  filtered norm: ", sum(abs, filtered))

    return (
        response_length = length(response),
        response_first = response[1],
        response_max = maximum(response),
        projection_size = size(filtered),
        filtered_sum = Float32(sum(filtered)),
        filtered_norm = Float32(sum(abs, filtered)),
        sample = Float32[
            filtered[1, 1, 1],
            filtered[2, 2, 1],
            filtered[4, 3, 2],
            filtered[8, 6, 4],
        ],
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_filtering_demo()
end
