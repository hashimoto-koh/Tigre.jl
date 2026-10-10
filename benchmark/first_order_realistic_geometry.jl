using Tigre

include("reuse_bench_utils.jl")
include("native_preflight.jl")

function _int_env(name::AbstractString, default::Int)
    value = get(ENV, name, nothing)
    value === nothing && return default
    return parse(Int, value)
end

function _geometry_preset(name::AbstractString)
    if name == "moderate"
        return Geometry(
            mode = :cone,
            angles = collect(LinRange(0f0, 2f0 * Float32(pi), 96)),
            nVoxel = (128, 128, 96),
            sVoxel = (220f0, 220f0, 180f0),
            nDetector = (256, 192),
            sDetector = (420f0, 320f0),
            DSO = 740f0,
            DSD = 1200f0,
        )
    elseif name == "dense"
        return Geometry(
            mode = :cone,
            angles = collect(LinRange(0f0, 2f0 * Float32(pi), 144)),
            nVoxel = (160, 160, 128),
            sVoxel = (240f0, 240f0, 220f0),
            nDetector = (320, 256),
            sDetector = (430f0, 340f0),
            DSO = 740f0,
            DSD = 1200f0,
        )
    else
        error("unknown REALISTIC_GEOMETRY_PRESET: $name")
    end
end

const PRESET = get(ENV, "REALISTIC_GEOMETRY_PRESET", "moderate")
const REPEATS = _int_env("REALISTIC_BENCH_REPEATS", 2)
const SAMPLES = _int_env("REALISTIC_BENCH_SAMPLES", 5)
const ISTA_ITERS = _int_env("REALISTIC_ISTA_ITERS", 5)
const FISTA_ITERS = _int_env("REALISTIC_FISTA_ITERS", 5)
const OS_ASD_OUTER_ITERS = _int_env("REALISTIC_OS_ASD_ITERS", 3)
const BLOCKSIZE = _int_env("REALISTIC_BLOCKSIZE", 8)
const TV_ITERS = _int_env("REALISTIC_TV_ITERS", 1)

geo = _geometry_preset(PRESET)

ensure_native_benchmark_ready("first_order_realistic_geometry benchmark")

volume = rand(Float32, volume_size(geo)...)
projection = Array{Float32}(undef, projection_size(geo))
Ax!(projection, volume, geo)

first_order_workspace = prepare_first_order_workspace(geo)
art_plan = prepare_art_plan(
    projection,
    geo;
    blocksize = BLOCKSIZE,
    need_full_projection = true,
    extra_volume_buffers = 3,
)

results = ReuseBenchSeriesResult[]

push!(results, run_reuse_bench_series("ISTA realistic ($PRESET, niter=$ISTA_ITERS)", REPEATS, SAMPLES,
    function ()
        for _ in 1:REPEATS
            dst = zeros(Float32, volume_size(geo)...)
            ista!(dst, projection, geo, ISTA_ITERS; tviter = TV_ITERS)
        end
    end,
    function ()
        return (prepare_first_order_workspace(geo), zeros(Float32, volume_size(geo)...))
    end,
    function (state)
        workspace, dst = state
        for _ in 1:REPEATS
            fill!(dst, 0f0)
            ista!(dst, projection, geo, ISTA_ITERS; tviter = TV_ITERS, workspace = workspace)
        end
    end,
))

push!(results, run_reuse_bench_series("FISTA realistic ($PRESET, niter=$FISTA_ITERS)", REPEATS, SAMPLES,
    function ()
        for _ in 1:REPEATS
            dst = zeros(Float32, volume_size(geo)...)
            fista!(dst, projection, geo, FISTA_ITERS; tviter = TV_ITERS)
        end
    end,
    function ()
        return (prepare_first_order_workspace(geo), zeros(Float32, volume_size(geo)...))
    end,
    function (state)
        workspace, dst = state
        for _ in 1:REPEATS
            fill!(dst, 0f0)
            fista!(dst, projection, geo, FISTA_ITERS; tviter = TV_ITERS, workspace = workspace)
        end
    end,
))

push!(results, run_reuse_bench_series("OS-ASD-POCS realistic ($PRESET, niter=$OS_ASD_OUTER_ITERS, blocksize=$BLOCKSIZE)", REPEATS, SAMPLES,
    function ()
        for _ in 1:REPEATS
            dst = zeros(Float32, volume_size(geo)...)
            os_asd_pocs!(dst, projection, geo, OS_ASD_OUTER_ITERS; blocksize = BLOCKSIZE, tviter = TV_ITERS)
        end
    end,
    function ()
        return (prepare_art_plan(projection, geo; blocksize = BLOCKSIZE, need_full_projection = true, extra_volume_buffers = 3), zeros(Float32, volume_size(geo)...))
    end,
    function (state)
        plan, dst = state
        for _ in 1:REPEATS
            fill!(dst, 0f0)
            os_asd_pocs!(dst, projection, geo, OS_ASD_OUTER_ITERS; blocksize = BLOCKSIZE, tviter = TV_ITERS, plan = plan)
        end
    end,
))

println("preset = ", PRESET)
println("angles = ", size(geo.angles, 2))
println("volume size = ", volume_size(geo))
println("projection size = ", projection_size(geo))
println("repeats = ", REPEATS, ", samples = ", SAMPLES)
println("prepared workspace ready: ", first_order_workspace !== nothing, ", prepared plan ready: ", art_plan !== nothing)

for result in results
    print_reuse_bench_series(result)
end
print_reuse_series_summary(results)
maybe_write_reuse_series_summary(results; argv = ARGS)
