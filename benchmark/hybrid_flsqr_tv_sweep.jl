using Tigre
using Statistics

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

ensure_native_benchmark_ready("hybrid_flsqr_tv_sweep benchmark")

phantom = rand(Float32, volume_size(geo)...)
projection = Ax(phantom, geo)

function sample_times(f; samples = 4)
    f()
    GC.gc()
    times = Vector{Float64}(undef, samples)
    for i in 1:samples
        GC.gc()
        times[i] = @elapsed f()
    end
    return times
end

println("Hybrid fLSQR-TV parameter sweep")
for lambda in (0.03f0, 0.1f0, 0.3f0)
    for steps in (2, 4, 8)
        workspace = prepare_krylov_workspace(geo; maxiter = 4, extra_volume_buffers = 8)
        volume = zeros(Float32, volume_size(geo)...)
        times = sample_times(() -> hybrid_flsqr_tv!(volume, projection, geo, 4; lambda = lambda, precond_steps = steps, workspace = workspace))
        history_volume = zeros(Float32, volume_size(geo)...)
        _, history = hybrid_flsqr_tv!(history_volume, projection, geo, 2; lambda = lambda, precond_steps = steps, return_history = true, compute_l2 = true)
        final_l2 = isempty(history.residual_l2) ? NaN32 : history.residual_l2[end]
        println("  lambda=", lambda,
            ", precond_steps=", steps,
            ", median=", median(times),
            " s, std=", std(times),
            " s, final_l2=", final_l2)
    end
end
