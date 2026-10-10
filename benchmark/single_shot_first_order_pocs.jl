using Tigre
using Statistics

include("reuse_bench_utils.jl")
include("native_preflight.jl")

const SAMPLES = 7

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

ensure_native_benchmark_ready("single_shot_first_order_pocs benchmark")

phantom = rand(Float32, volume_size(geo)...)
prior_volume = copy(phantom)
projection = Array{Float32}(undef, projection_size(geo))
Ax!(projection, phantom, geo)

first_order_workspace = prepare_first_order_workspace(geo)
tv_art_plan = prepare_art_plan(projection, geo; blocksize = 1, need_full_projection = false, extra_volume_buffers = 1)
art_plan = prepare_art_plan(projection, geo; blocksize = 4, need_full_projection = true, extra_volume_buffers = 3)

function measure_case(build_run, prepared_run; samples = SAMPLES)
    build_run()
    prepared_run()

    build_times = Vector{Float64}(undef, samples)
    build_allocs = Vector{Int}(undef, samples)
    prepared_times = Vector{Float64}(undef, samples)
    prepared_allocs = Vector{Int}(undef, samples)

    for i in 1:samples
        GC.gc()
        build_allocs[i] = @allocated build_times[i] = @elapsed build_run()
        GC.gc()
        prepared_allocs[i] = @allocated prepared_times[i] = @elapsed prepared_run()
    end

    return (
        build_time = (median = median(build_times), mean = mean(build_times), std = std(build_times)),
        build_alloc = (median = median(build_allocs), mean = mean(build_allocs)),
        prepared_time = (median = median(prepared_times), mean = mean(prepared_times), std = std(prepared_times)),
        prepared_alloc = (median = median(prepared_allocs), mean = mean(prepared_allocs)),
    )
end

function print_case(label, stats)
    println(label)
    println("  build time median/mean/std: ", stats.build_time.median, " / ", stats.build_time.mean, " / ", stats.build_time.std, " s")
    println("  build alloc median/mean: ", stats.build_alloc.median, " / ", stats.build_alloc.mean, " bytes")
    println("  prepared time median/mean/std: ", stats.prepared_time.median, " / ", stats.prepared_time.mean, " / ", stats.prepared_time.std, " s")
    println("  prepared alloc median/mean: ", stats.prepared_alloc.median, " / ", stats.prepared_alloc.mean, " bytes")
end

function _single_shot_summary_header()
    return (
        "label",
        "samples",
        "build_time_median_s",
        "build_time_mean_s",
        "build_time_std_s",
        "build_alloc_median_B",
        "build_alloc_mean_B",
        "prepared_time_median_s",
        "prepared_time_mean_s",
        "prepared_time_std_s",
        "prepared_alloc_median_B",
        "prepared_alloc_mean_B",
        "speedup_median",
    )
end

function _single_shot_summary_row(label, stats, samples)
    return (
        label,
        samples,
        stats.build_time.median,
        stats.build_time.mean,
        stats.build_time.std,
        stats.build_alloc.median,
        stats.build_alloc.mean,
        stats.prepared_time.median,
        stats.prepared_time.mean,
        stats.prepared_time.std,
        stats.prepared_alloc.median,
        stats.prepared_alloc.mean,
        stats.build_time.median / stats.prepared_time.median,
    )
end

ista_build_dst = zeros(Float32, volume_size(geo)...)
ista_prepared_dst = zeros(Float32, volume_size(geo)...)
fista_build_dst = zeros(Float32, volume_size(geo)...)
fista_prepared_dst = zeros(Float32, volume_size(geo)...)
sart_tv_build_dst = zeros(Float32, volume_size(geo)...)
sart_tv_prepared_dst = zeros(Float32, volume_size(geo)...)
os_asd_build_dst = zeros(Float32, volume_size(geo)...)
os_asd_prepared_dst = zeros(Float32, volume_size(geo)...)
os_piccs_build_dst = zeros(Float32, volume_size(geo)...)
os_piccs_prepared_dst = zeros(Float32, volume_size(geo)...)

ista_stats = measure_case(
    () -> (fill!(ista_build_dst, 0f0); ista!(ista_build_dst, projection, geo, 1; tviter = 1)),
    () -> (fill!(ista_prepared_dst, 0f0); ista!(ista_prepared_dst, projection, geo, 1; tviter = 1, workspace = first_order_workspace)),
)
fista_stats = measure_case(
    () -> (fill!(fista_build_dst, 0f0); fista!(fista_build_dst, projection, geo, 1; tviter = 1)),
    () -> (fill!(fista_prepared_dst, 0f0); fista!(fista_prepared_dst, projection, geo, 1; tviter = 1, workspace = first_order_workspace)),
)
sart_tv_stats = measure_case(
    () -> (fill!(sart_tv_build_dst, 0f0); sart_tv!(sart_tv_build_dst, projection, geo, 1; tviter = 1, tvlambda = 1f0)),
    () -> (fill!(sart_tv_prepared_dst, 0f0); sart_tv!(sart_tv_prepared_dst, projection, geo, 1; tviter = 1, tvlambda = 1f0, plan = tv_art_plan)),
)
os_asd_stats = measure_case(
    () -> (fill!(os_asd_build_dst, 0f0); os_asd_pocs!(os_asd_build_dst, projection, geo, 1; blocksize = 4, tviter = 1)),
    () -> (fill!(os_asd_prepared_dst, 0f0); os_asd_pocs!(os_asd_prepared_dst, projection, geo, 1; blocksize = 4, tviter = 1, plan = art_plan)),
)
os_piccs_stats = measure_case(
    () -> (fill!(os_piccs_build_dst, 0f0); os_piccs!(os_piccs_build_dst, projection, geo, prior_volume, 1; blocksize = 4, tviter = 1, prior_ratio = 0.5f0)),
    () -> (fill!(os_piccs_prepared_dst, 0f0); os_piccs!(os_piccs_prepared_dst, projection, geo, prior_volume, 1; blocksize = 4, tviter = 1, prior_ratio = 0.5f0, plan = art_plan)),
)

results = [
    ("ISTA single-shot", ista_stats),
    ("FISTA single-shot", fista_stats),
    ("SART-TV single-shot", sart_tv_stats),
    ("OS-ASD-POCS single-shot", os_asd_stats),
    ("OS-PICCS single-shot", os_piccs_stats),
]

for (label, stats) in results
    print_case(label, stats)
end

rows = [_single_shot_summary_row(label, stats, SAMPLES) for (label, stats) in results]
println()
println("Summary")
println(join(string.(collect(_single_shot_summary_header())), " | "))
for row in rows
    println(join(string.(collect(row)), " | "))
end
maybe_write_table(
    _single_shot_summary_header(),
    rows;
    argv = ARGS,
    env_tsv = "FIRST_ORDER_SINGLE_SHOT_TSV",
    env_csv = "FIRST_ORDER_SINGLE_SHOT_CSV",
    label = "single-shot summary",
)
