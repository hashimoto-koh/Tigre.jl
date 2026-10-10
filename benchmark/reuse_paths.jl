using Tigre

include("reuse_bench_utils.jl")
include("native_preflight.jl")

function _int_env(name::AbstractString, default::Int)
    value = get(ENV, name, nothing)
    value === nothing && return default
    return parse(Int, value)
end

geo = Geometry(
    mode = :cone,
    angles = collect(LinRange(0f0, 2f0 * Float32(pi), 16)),
    nVoxel = (64, 64, 64),
    sVoxel = (128f0, 128f0, 128f0),
    nDetector = (96, 64),
    sDetector = (192f0, 128f0),
    DSO = 1000f0,
    DSD = 1200f0,
)

ensure_native_benchmark_ready("reuse_paths benchmark")

volume = rand(Float32, volume_size(geo)...)
projection = Array{Float32}(undef, projection_size(geo))
prior_volume = copy(volume)
Ax!(projection, volume, geo)

const REPEATS = _int_env("REUSE_BENCH_REPEATS", 5)
const OUTER_ITERS = _int_env("REUSE_BENCH_OUTER_ITERS", 3)

results = ReuseBenchResult[]

push!(results, run_reuse_bench("OS-ASD-POCS (blocksize=4, niter=3)", REPEATS,
    function ()
        for _ in 1:REPEATS
            dst = zeros(Float32, volume_size(geo)...)
            os_asd_pocs!(dst, projection, geo, OUTER_ITERS; blocksize = 4, tviter = 1)
        end
    end,
    function ()
        return (prepare_art_plan(projection, geo; blocksize = 4, need_full_projection = true, extra_volume_buffers = 3), zeros(Float32, volume_size(geo)...))
    end,
    function (state)
        plan, dst = state
        for _ in 1:REPEATS
            fill!(dst, 0f0)
            os_asd_pocs!(dst, projection, geo, OUTER_ITERS; blocksize = 4, tviter = 1, plan = plan)
        end
    end,
))

push!(results, run_reuse_bench("ISTA (niter=5)", REPEATS,
    function ()
        for _ in 1:REPEATS
            dst = zeros(Float32, volume_size(geo)...)
            ista!(dst, projection, geo, 5; tviter = 1)
        end
    end,
    function ()
        return (prepare_first_order_workspace(geo), zeros(Float32, volume_size(geo)...))
    end,
    function (state)
        workspace, dst = state
        for _ in 1:REPEATS
            fill!(dst, 0f0)
            ista!(dst, projection, geo, 5; tviter = 1, workspace = workspace)
        end
    end,
))

push!(results, run_reuse_bench("LSMR (niter=5)", REPEATS,
    function ()
        for _ in 1:REPEATS
            dst = zeros(Float32, volume_size(geo)...)
            lsmr!(dst, projection, geo, 5)
        end
    end,
    function ()
        return (prepare_krylov_workspace(geo; maxiter = 5), zeros(Float32, volume_size(geo)...))
    end,
    function (state)
        workspace, dst = state
        for _ in 1:REPEATS
            fill!(dst, 0f0)
            lsmr!(dst, projection, geo, 5; workspace = workspace)
        end
    end,
))

push!(results, run_reuse_bench("OS-PICCS (blocksize=4, niter=3)", REPEATS,
    function ()
        for _ in 1:REPEATS
            dst = zeros(Float32, volume_size(geo)...)
            os_piccs!(dst, projection, geo, prior_volume, OUTER_ITERS; blocksize = 4, tviter = 1, prior_ratio = 0.5f0)
        end
    end,
    function ()
        return (prepare_art_plan(projection, geo; blocksize = 4, need_full_projection = true, extra_volume_buffers = 3), zeros(Float32, volume_size(geo)...))
    end,
    function (state)
        plan, dst = state
        for _ in 1:REPEATS
            fill!(dst, 0f0)
            os_piccs!(dst, projection, geo, prior_volume, OUTER_ITERS; blocksize = 4, tviter = 1, prior_ratio = 0.5f0, plan = plan)
        end
    end,
))

for result in results
    print_reuse_bench(result)
end
print_reuse_summary(results)
maybe_write_reuse_summary(results; argv = ARGS)
