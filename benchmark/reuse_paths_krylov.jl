using Tigre

include("reuse_bench_utils.jl")
include("native_preflight.jl")

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

ensure_native_benchmark_ready("reuse_paths_krylov benchmark")

volume = rand(Float32, volume_size(geo)...)
projection = Array{Float32}(undef, projection_size(geo))
Ax!(projection, volume, geo)

const REPEATS = 5

results = ReuseBenchResult[]

push!(results, run_reuse_bench("Hybrid LSQR (niter=5)", REPEATS,
    function ()
        for _ in 1:REPEATS
            dst = zeros(Float32, volume_size(geo)...)
            hybrid_lsqr!(dst, projection, geo, 5; lambda = 0.1f0)
        end
    end,
    function ()
        return (prepare_krylov_workspace(geo; maxiter = 5), zeros(Float32, volume_size(geo)...))
    end,
    function (state)
        workspace, dst = state
        for _ in 1:REPEATS
            fill!(dst, 0f0)
            hybrid_lsqr!(dst, projection, geo, 5; lambda = 0.1f0, workspace = workspace)
        end
    end,
))

push!(results, run_reuse_bench("Hybrid fLSQR-TV (niter=5)", REPEATS,
    function ()
        for _ in 1:REPEATS
            dst = zeros(Float32, volume_size(geo)...)
            hybrid_flsqr_tv!(dst, projection, geo, 5; lambda = 0.1f0)
        end
    end,
    function ()
        return (prepare_krylov_workspace(geo; maxiter = 5, extra_volume_buffers = 8), zeros(Float32, volume_size(geo)...))
    end,
    function (state)
        workspace, dst = state
        for _ in 1:REPEATS
            fill!(dst, 0f0)
            hybrid_flsqr_tv!(dst, projection, geo, 5; lambda = 0.1f0, workspace = workspace)
        end
    end,
))

push!(results, run_reuse_bench("AB-GMRES (niter=5)", REPEATS,
    function ()
        for _ in 1:REPEATS
            dst = zeros(Float32, volume_size(geo)...)
            ab_gmres!(dst, projection, geo, 5)
        end
    end,
    function ()
        return (prepare_krylov_workspace(geo; maxiter = 5, extra_volume_buffers = 7), zeros(Float32, volume_size(geo)...))
    end,
    function (state)
        workspace, dst = state
        for _ in 1:REPEATS
            fill!(dst, 0f0)
            ab_gmres!(dst, projection, geo, 5; workspace = workspace)
        end
    end,
))

push!(results, run_reuse_bench("BA-GMRES (niter=5)", REPEATS,
    function ()
        for _ in 1:REPEATS
            dst = zeros(Float32, volume_size(geo)...)
            ba_gmres!(dst, projection, geo, 5)
        end
    end,
    function ()
        return (prepare_krylov_workspace(geo; maxiter = 5), zeros(Float32, volume_size(geo)...))
    end,
    function (state)
        workspace, dst = state
        for _ in 1:REPEATS
            fill!(dst, 0f0)
            ba_gmres!(dst, projection, geo, 5; workspace = workspace)
        end
    end,
))

push!(results, run_reuse_bench("IRN-TV-CGLS (niter_outer=2, niter=3)", REPEATS,
    function ()
        for _ in 1:REPEATS
            dst = zeros(Float32, volume_size(geo)...)
            irn_tv_cgls!(dst, projection, geo, 3; niter_outer = 2, lambda = 0.1f0)
        end
    end,
    function ()
        return (prepare_krylov_workspace(geo; extra_volume_buffers = 7), zeros(Float32, volume_size(geo)...))
    end,
    function (state)
        workspace, dst = state
        for _ in 1:REPEATS
            fill!(dst, 0f0)
            irn_tv_cgls!(dst, projection, geo, 3; niter_outer = 2, lambda = 0.1f0, workspace = workspace)
        end
    end,
))

for result in results
    print_reuse_bench(result)
end
print_reuse_summary(results)
maybe_write_reuse_summary(results; argv = ARGS)
