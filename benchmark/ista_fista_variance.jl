using Tigre
using Statistics

include("reuse_bench_utils.jl")
include("native_preflight.jl")

function _int_env(name::AbstractString, default::Int)
    value = get(ENV, name, nothing)
    value === nothing && return default
    return parse(Int, value)
end

function _measure_series(f; samples::Int, gc_before_each::Bool, disable_gc::Bool)
    times = Vector{Float64}(undef, samples)
    allocs = Vector{Int}(undef, samples)
    previous_gc_state = true
    if disable_gc
        previous_gc_state = GC.enable(false)
    end
    try
        for i in 1:samples
            gc_before_each && GC.gc()
            allocs[i] = @allocated times[i] = @elapsed f()
        end
    finally
        disable_gc && GC.enable(previous_gc_state)
        GC.gc()
    end
    return (
        median = median(times),
        mean = mean(times),
        std = std(times),
        minimum = minimum(times),
        maximum = maximum(times),
        alloc_median = median(allocs),
        alloc_mean = mean(allocs),
    )
end

function _measure_cold(f)
    GC.gc()
    alloc = @allocated time = @elapsed f()
    return (time = time, alloc = alloc)
end

function _print_series(label, stats)
    println("  ", label, " time median/mean/std: ", stats.median, " / ", stats.mean, " / ", stats.std, " s")
    println("  ", label, " time min/max: ", stats.minimum, " / ", stats.maximum, " s")
    println("  ", label, " alloc median/mean: ", stats.alloc_median, " / ", stats.alloc_mean, " bytes")
end

function _print_variance_case(label, cold, warm_gc_on, warm_gc_off)
    println(label)
    println("  cold-first time/alloc: ", cold.time, " s / ", cold.alloc, " bytes")
    _print_series("warm-gc-on", warm_gc_on)
    _print_series("warm-gc-off", warm_gc_off)
end

function _variance_summary_header()
    return (
        "label",
        "samples",
        "cold_time_s",
        "cold_alloc_B",
        "warm_gc_on_median_s",
        "warm_gc_on_mean_s",
        "warm_gc_on_std_s",
        "warm_gc_on_alloc_median_B",
        "warm_gc_off_median_s",
        "warm_gc_off_mean_s",
        "warm_gc_off_std_s",
        "warm_gc_off_alloc_median_B",
    )
end

function _variance_summary_row(label, samples, cold, warm_gc_on, warm_gc_off)
    return (
        label,
        samples,
        cold.time,
        cold.alloc,
        warm_gc_on.median,
        warm_gc_on.mean,
        warm_gc_on.std,
        warm_gc_on.alloc_median,
        warm_gc_off.median,
        warm_gc_off.mean,
        warm_gc_off.std,
        warm_gc_off.alloc_median,
    )
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

ensure_native_benchmark_ready("ista_fista_variance benchmark")

samples = _int_env("FIRST_ORDER_VARIANCE_SAMPLES", 9)

phantom = rand(Float32, volume_size(geo)...)
projection = Array{Float32}(undef, projection_size(geo))
Ax!(projection, phantom, geo)

workspace = prepare_first_order_workspace(geo)
ista_build_dst = zeros(Float32, volume_size(geo)...)
ista_prepared_dst = zeros(Float32, volume_size(geo)...)
fista_build_dst = zeros(Float32, volume_size(geo)...)
fista_prepared_dst = zeros(Float32, volume_size(geo)...)

ista_build!() = (fill!(ista_build_dst, 0f0); ista!(ista_build_dst, projection, geo, 1; tviter = 1))
ista_prepared!() = (fill!(ista_prepared_dst, 0f0); ista!(ista_prepared_dst, projection, geo, 1; tviter = 1, workspace = workspace))
fista_build!() = (fill!(fista_build_dst, 0f0); fista!(fista_build_dst, projection, geo, 1; tviter = 1))
fista_prepared!() = (fill!(fista_prepared_dst, 0f0); fista!(fista_prepared_dst, projection, geo, 1; tviter = 1, workspace = workspace))

ista_build_cold = _measure_cold(ista_build!)
ista_build!()
ista_build_warm_gc_on = _measure_series(ista_build!; samples = samples, gc_before_each = true, disable_gc = false)
ista_build_warm_gc_off = _measure_series(ista_build!; samples = samples, gc_before_each = false, disable_gc = true)

ista_prepared_cold = _measure_cold(ista_prepared!)
ista_prepared!()
ista_prepared_warm_gc_on = _measure_series(ista_prepared!; samples = samples, gc_before_each = true, disable_gc = false)
ista_prepared_warm_gc_off = _measure_series(ista_prepared!; samples = samples, gc_before_each = false, disable_gc = true)

fista_build_cold = _measure_cold(fista_build!)
fista_build!()
fista_build_warm_gc_on = _measure_series(fista_build!; samples = samples, gc_before_each = true, disable_gc = false)
fista_build_warm_gc_off = _measure_series(fista_build!; samples = samples, gc_before_each = false, disable_gc = true)

fista_prepared_cold = _measure_cold(fista_prepared!)
fista_prepared!()
fista_prepared_warm_gc_on = _measure_series(fista_prepared!; samples = samples, gc_before_each = true, disable_gc = false)
fista_prepared_warm_gc_off = _measure_series(fista_prepared!; samples = samples, gc_before_each = false, disable_gc = true)

println("samples = ", samples)
results = [
    ("ISTA build", ista_build_cold, ista_build_warm_gc_on, ista_build_warm_gc_off),
    ("ISTA prepared", ista_prepared_cold, ista_prepared_warm_gc_on, ista_prepared_warm_gc_off),
    ("FISTA build", fista_build_cold, fista_build_warm_gc_on, fista_build_warm_gc_off),
    ("FISTA prepared", fista_prepared_cold, fista_prepared_warm_gc_on, fista_prepared_warm_gc_off),
]

for (label, cold, warm_gc_on, warm_gc_off) in results
    _print_variance_case(label, cold, warm_gc_on, warm_gc_off)
end

rows = [_variance_summary_row(label, samples, cold, warm_gc_on, warm_gc_off) for (label, cold, warm_gc_on, warm_gc_off) in results]
println()
println("Summary")
println(join(string.(collect(_variance_summary_header())), " | "))
for row in rows
    println(join(string.(collect(row)), " | "))
end
maybe_write_table(
    _variance_summary_header(),
    rows;
    argv = ARGS,
    env_tsv = "FIRST_ORDER_VARIANCE_TSV",
    env_csv = "FIRST_ORDER_VARIANCE_CSV",
    label = "variance summary",
)
