using Tigre
using Statistics

include("reuse_bench_utils.jl")
include("native_preflight.jl")

const SAMPLES = 9

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

ensure_native_benchmark_ready("decompose_first_order_pocs benchmark")

phantom = rand(Float32, volume_size(geo)...)
prior_volume = copy(phantom)
projection = Array{Float32}(undef, projection_size(geo))
Ax!(projection, phantom, geo)

workspace = prepare_first_order_workspace(geo)
tv_src = zeros(Float32, volume_size(geo)...)
tv_dst = zeros(Float32, volume_size(geo)...)
grad_src = zeros(Float32, volume_size(geo)...)
piccs_dst = zeros(Float32, volume_size(geo)...)
mintv_dst = zeros(Float32, volume_size(geo)...)

function measure(f; samples = SAMPLES)
    f()
    times = Vector{Float64}(undef, samples)
    allocs = Vector{Int}(undef, samples)
    for i in 1:samples
        GC.gc()
        allocs[i] = @allocated times[i] = @elapsed f()
    end
    return (
        time = (median = median(times), mean = mean(times), std = std(times)),
        alloc = (median = median(allocs), mean = mean(allocs)),
    )
end

function print_case(label, stats)
    println(label)
    println("  time median/mean/std: ", stats.time.median, " / ", stats.time.mean, " / ", stats.time.std, " s")
    println("  alloc median/mean: ", stats.alloc.median, " / ", stats.alloc.mean, " bytes")
end

function _decompose_summary_header()
    return (
        "label",
        "samples",
        "time_median_s",
        "time_mean_s",
        "time_std_s",
        "alloc_median_B",
        "alloc_mean_B",
    )
end

function _decompose_summary_row(label, stats, samples)
    return (
        label,
        samples,
        stats.time.median,
        stats.time.mean,
        stats.time.std,
        stats.alloc.median,
        stats.alloc.mean,
    )
end

function fista_gradient_only!(volume, projection, geo, workspace)
    fill!(volume, 0f0)
    Ax!(workspace.proj_tmp, volume, geo; projection_type = :interpolated)
    @inbounds for i in eachindex(workspace.proj_tmp, projection)
        workspace.proj_tmp[i] = projection[i] - workspace.proj_tmp[i]
    end
    Atb!(workspace.backproj_tmp, workspace.proj_tmp, geo; backprojection_type = :matched)
    @inbounds for i in eachindex(volume, workspace.backproj_tmp)
        volume[i] += 1f-8 * workspace.backproj_tmp[i]
    end
    return volume
end

fista_grad_stats = measure() do
    fista_gradient_only!(grad_src, projection, geo, workspace)
end

tvdenoise_stats = measure() do
    fill!(tv_src, 0f0)
    tvdenoise!(tv_dst, tv_src; lambda = 1f8 / 20f0, maxiter = 1)
end

mintv_stats = measure() do
    fill!(tv_src, 0f0)
    minTV!(mintv_dst, tv_src; alpha = 0.002f0, maxiter = 1)
end

piccs_stats = measure() do
    fill!(tv_src, 0f0)
    piccsTV!(piccs_dst, tv_src, prior_volume; alpha = 0.002f0, ratio = 0.5f0, maxiter = 1)
end

ista_prepared_stats = measure() do
    fill!(tv_src, 0f0)
    ista!(tv_src, projection, geo, 1; tviter = 1, workspace = workspace)
end

fista_prepared_stats = measure() do
    fill!(tv_src, 0f0)
    fista!(tv_src, projection, geo, 1; tviter = 1, workspace = workspace)
end

results = [
    ("FISTA gradient-only step", fista_grad_stats),
    ("tvdenoise! primitive", tvdenoise_stats),
    ("minTV! primitive", mintv_stats),
    ("piccsTV! primitive", piccs_stats),
    ("ISTA prepared single-shot", ista_prepared_stats),
    ("FISTA prepared single-shot", fista_prepared_stats),
]

for (label, stats) in results
    print_case(label, stats)
end

rows = [_decompose_summary_row(label, stats, SAMPLES) for (label, stats) in results]
println()
println("Summary")
println(join(string.(collect(_decompose_summary_header())), " | "))
for row in rows
    println(join(string.(collect(row)), " | "))
end
maybe_write_table(
    _decompose_summary_header(),
    rows;
    argv = ARGS,
    env_tsv = "DECOMPOSE_FIRST_ORDER_TSV",
    env_csv = "DECOMPOSE_FIRST_ORDER_CSV",
    label = "decompose summary",
)
