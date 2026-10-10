using Tigre

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

ensure_native_benchmark_ready("reuse_paths_hybrid_flsqr_tv benchmark")

const REPEATS = 5

volume = rand(Float32, volume_size(geo)...)
projection = Array{Float32}(undef, projection_size(geo))
Ax!(projection, volume, geo)

function run_build()
    for _ in 1:REPEATS
        dst = zeros(Float32, volume_size(geo)...)
        hybrid_flsqr_tv!(dst, projection, geo, 5; lambda = 0.1f0)
    end
    return nothing
end

function run_reuse()
    workspace = prepare_krylov_workspace(geo; maxiter = 5, extra_volume_buffers = 8)
    dst = zeros(Float32, volume_size(geo)...)
    for _ in 1:REPEATS
        fill!(dst, 0f0)
        hybrid_flsqr_tv!(dst, projection, geo, 5; lambda = 0.1f0, workspace = workspace)
    end
    return nothing
end

run_build()
run_reuse()

GC.gc()
build_alloc = @allocated run_build()
GC.gc()
build_time = @elapsed run_build()

GC.gc()
reuse_alloc = @allocated run_reuse()
GC.gc()
reuse_time = @elapsed run_reuse()

println("Hybrid fLSQR-TV (niter=5)")
println("  build+run total (warmup excluded): ", build_time, " s, allocated: ", build_alloc, " bytes")
println("  reuse-only total (warmup excluded): ", reuse_time, " s, allocated: ", reuse_alloc, " bytes")
println("  build+run avg/outer: ", build_time / REPEATS, " s")
println("  reuse-only avg/outer: ", reuse_time / REPEATS, " s")
