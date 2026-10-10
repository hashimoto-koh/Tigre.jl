using Statistics

const BENCHMARK_SUMMARY_SCHEMA = "tigre-benchmark-summary-v1"
const BENCHMARK_SUMMARY_SCHEMA_VERSION = 1

struct ReuseBenchResult
    label::String
    repeats::Int
    build_time::Float64
    build_alloc::Int
    prepare_time::Float64
    prepare_alloc::Int
    reuse_time::Float64
    reuse_alloc::Int
end

struct ReuseBenchSeriesResult
    label::String
    repeats::Int
    samples::Int
    prepare_time::Float64
    prepare_alloc::Int
    build_time_median::Float64
    build_time_mean::Float64
    build_time_std::Float64
    build_time_min::Float64
    build_time_max::Float64
    build_alloc_median::Float64
    build_alloc_mean::Float64
    reuse_time_median::Float64
    reuse_time_mean::Float64
    reuse_time_std::Float64
    reuse_time_min::Float64
    reuse_time_max::Float64
    reuse_alloc_median::Float64
    reuse_alloc_mean::Float64
end

function _sample_stats(values::AbstractVector{<:Real})
    return (
        median = median(values),
        mean = mean(values),
        std = length(values) > 1 ? std(values) : 0.0,
        minimum = minimum(values),
        maximum = maximum(values),
    )
end

function _summary_header()
    return (
        "label",
        "build_avg_s",
        "reuse_avg_s",
        "prepare_once_s",
        "build_alloc_B",
        "reuse_alloc_B",
        "speedup",
    )
end

function _series_summary_header()
    return (
        "label",
        "samples",
        "build_avg_median_s",
        "build_avg_std_s",
        "reuse_avg_median_s",
        "reuse_avg_std_s",
        "prepare_once_s",
        "build_alloc_median_B",
        "reuse_alloc_median_B",
        "speedup_median",
    )
end

function _summary_row(result::ReuseBenchResult)
    build_avg = result.build_time / result.repeats
    reuse_avg = result.reuse_time / result.repeats
    speedup = build_avg / reuse_avg
    return (
        result.label,
        string(build_avg),
        string(reuse_avg),
        string(result.prepare_time),
        string(result.build_alloc),
        string(result.reuse_alloc),
        string(speedup),
    )
end

function _series_summary_row(result::ReuseBenchSeriesResult)
    return (
        result.label,
        string(result.samples),
        string(result.build_time_median / result.repeats),
        string(result.build_time_std / result.repeats),
        string(result.reuse_time_median / result.repeats),
        string(result.reuse_time_std / result.repeats),
        string(result.prepare_time),
        string(result.build_alloc_median),
        string(result.reuse_alloc_median),
        string(result.build_time_median / result.repeats / (result.reuse_time_median / result.repeats)),
    )
end

function _csv_escape(field::AbstractString)
    escaped = replace(field, '"' => "\"\"")
    return occursin(',', escaped) || occursin('"', escaped) || occursin('\n', escaped) ? "\"" * escaped * "\"" : escaped
end

function _summary_metadata_pairs(label::AbstractString)
    return (
        "schema" => BENCHMARK_SUMMARY_SCHEMA,
        "schema_version" => string(BENCHMARK_SUMMARY_SCHEMA_VERSION),
        "table" => String(label),
    )
end

function _write_metadata_comments(io::IO, label::AbstractString)
    for (key, value) in _summary_metadata_pairs(label)
        println(io, "# ", key, "=", value)
    end
    return nothing
end

function read_summary_metadata(path::AbstractString)
    metadata = Dict{String, String}()
    for raw_line in eachline(path)
        line = strip(raw_line)
        isempty(line) && continue
        startswith(line, "#") || break
        body = strip(line[2:end])
        isempty(body) && continue
        occursin('=', body) || continue
        key, value = split(body, "=", limit = 2)
        metadata[strip(key)] = strip(value)
    end
    return metadata
end

function summary_schema_ok(path::AbstractString)
    metadata = read_summary_metadata(path)
    schema = get(metadata, "schema", nothing)
    version = get(metadata, "schema_version", nothing)
    if schema === nothing && version === nothing
        return true
    end
    return schema == BENCHMARK_SUMMARY_SCHEMA && version == string(BENCHMARK_SUMMARY_SCHEMA_VERSION)
end

function run_reuse_bench(label::AbstractString, repeats::Integer, build_run, prepare_reuse, reuse_run)
    repeats > 0 || throw(ArgumentError("repeats must be >= 1"))

    build_run()
    reuse_state = prepare_reuse()
    reuse_run(reuse_state)

    GC.gc()
    build_alloc = @allocated build_run()
    GC.gc()
    build_time = @elapsed build_run()

    GC.gc()
    prepare_alloc = @allocated prepare_reuse()
    GC.gc()
    prepare_time = @elapsed prepare_reuse()

    reuse_state = prepare_reuse()
    GC.gc()
    reuse_alloc = @allocated reuse_run(reuse_state)
    GC.gc()
    reuse_time = @elapsed reuse_run(reuse_state)

    return ReuseBenchResult(
        String(label),
        Int(repeats),
        build_time,
        build_alloc,
        prepare_time,
        prepare_alloc,
        reuse_time,
        reuse_alloc,
    )
end

function run_reuse_bench_series(label::AbstractString, repeats::Integer, samples::Integer, build_run, prepare_reuse, reuse_run)
    repeats > 0 || throw(ArgumentError("repeats must be >= 1"))
    samples > 0 || throw(ArgumentError("samples must be >= 1"))

    build_run()
    reuse_state = prepare_reuse()
    reuse_run(reuse_state)

    GC.gc()
    prepare_alloc = @allocated prepare_reuse()
    GC.gc()
    prepare_time = @elapsed prepare_reuse()

    build_times = Vector{Float64}(undef, samples)
    build_allocs = Vector{Int}(undef, samples)
    reuse_times = Vector{Float64}(undef, samples)
    reuse_allocs = Vector{Int}(undef, samples)

    for i in 1:samples
        GC.gc()
        build_allocs[i] = @allocated build_times[i] = @elapsed build_run()

        reuse_state = prepare_reuse()
        GC.gc()
        reuse_allocs[i] = @allocated reuse_times[i] = @elapsed reuse_run(reuse_state)
    end

    build_stats = _sample_stats(build_times)
    build_alloc_stats = _sample_stats(build_allocs)
    reuse_stats = _sample_stats(reuse_times)
    reuse_alloc_stats = _sample_stats(reuse_allocs)

    return ReuseBenchSeriesResult(
        String(label),
        Int(repeats),
        Int(samples),
        prepare_time,
        prepare_alloc,
        build_stats.median,
        build_stats.mean,
        build_stats.std,
        build_stats.minimum,
        build_stats.maximum,
        build_alloc_stats.median,
        build_alloc_stats.mean,
        reuse_stats.median,
        reuse_stats.mean,
        reuse_stats.std,
        reuse_stats.minimum,
        reuse_stats.maximum,
        reuse_alloc_stats.median,
        reuse_alloc_stats.mean,
    )
end

function print_reuse_bench(result::ReuseBenchResult)
    println(result.label)
    println("  build+run total (warmup excluded): ", result.build_time, " s, allocated: ", result.build_alloc, " bytes")
    println("  prepare-once: ", result.prepare_time, " s, allocated: ", result.prepare_alloc, " bytes")
    println("  reuse-only total (warmup excluded): ", result.reuse_time, " s, allocated: ", result.reuse_alloc, " bytes")
    println("  build+run avg/outer: ", result.build_time / result.repeats, " s")
    println("  reuse-only avg/outer: ", result.reuse_time / result.repeats, " s")
end

function print_reuse_bench_series(result::ReuseBenchSeriesResult)
    println(result.label)
    println("  samples: ", result.samples)
    println("  prepare-once: ", result.prepare_time, " s, allocated: ", result.prepare_alloc, " bytes")
    println("  build avg/outer median/mean/std: ",
        result.build_time_median / result.repeats, " / ",
        result.build_time_mean / result.repeats, " / ",
        result.build_time_std / result.repeats, " s")
    println("  build avg/outer min/max: ",
        result.build_time_min / result.repeats, " / ",
        result.build_time_max / result.repeats, " s")
    println("  build alloc median/mean: ", result.build_alloc_median, " / ", result.build_alloc_mean, " bytes")
    println("  reuse avg/outer median/mean/std: ",
        result.reuse_time_median / result.repeats, " / ",
        result.reuse_time_mean / result.repeats, " / ",
        result.reuse_time_std / result.repeats, " s")
    println("  reuse avg/outer min/max: ",
        result.reuse_time_min / result.repeats, " / ",
        result.reuse_time_max / result.repeats, " s")
    println("  reuse alloc median/mean: ", result.reuse_alloc_median, " / ", result.reuse_alloc_mean, " bytes")
    println("  median speedup: ", result.build_time_median / result.repeats / (result.reuse_time_median / result.repeats), "x")
end

function print_reuse_summary(results::AbstractVector{ReuseBenchResult})
    isempty(results) && return nothing
    println()
    println("Summary")
    println(join(_summary_header(), " | "))
    for result in results
        println(join(_summary_row(result), " | "))
    end
    return nothing
end

function print_reuse_series_summary(results::AbstractVector{ReuseBenchSeriesResult})
    isempty(results) && return nothing
    println()
    println("Summary")
    println("label | samples | build_avg_median_s | build_avg_std_s | reuse_avg_median_s | reuse_avg_std_s | prepare_once_s | build_alloc_median_B | reuse_alloc_median_B | speedup_median")
    for result in results
        println(
            result.label, " | ",
            result.samples, " | ",
            result.build_time_median / result.repeats, " | ",
            result.build_time_std / result.repeats, " | ",
            result.reuse_time_median / result.repeats, " | ",
            result.reuse_time_std / result.repeats, " | ",
            result.prepare_time, " | ",
            result.build_alloc_median, " | ",
            result.reuse_alloc_median, " | ",
            result.build_time_median / result.repeats / (result.reuse_time_median / result.repeats),
        )
    end
    return nothing
end

function write_reuse_summary_tsv(path::AbstractString, results::AbstractVector{ReuseBenchResult})
    open(path, "w") do io
        _write_metadata_comments(io, "reuse-summary")
        println(io, join(_summary_header(), '\t'))
        for result in results
            println(io, join(_summary_row(result), '\t'))
        end
    end
    return path
end

function write_reuse_summary_csv(path::AbstractString, results::AbstractVector{ReuseBenchResult})
    open(path, "w") do io
        _write_metadata_comments(io, "reuse-summary")
        println(io, join(_csv_escape.(collect(_summary_header())), ','))
        for result in results
            println(io, join(_csv_escape.(collect(_summary_row(result))), ','))
        end
    end
    return path
end

function write_reuse_series_summary_tsv(path::AbstractString, results::AbstractVector{ReuseBenchSeriesResult})
    open(path, "w") do io
        _write_metadata_comments(io, "reuse-series-summary")
        println(io, join(_series_summary_header(), '\t'))
        for result in results
            println(io, join(_series_summary_row(result), '\t'))
        end
    end
    return path
end

function write_reuse_series_summary_csv(path::AbstractString, results::AbstractVector{ReuseBenchSeriesResult})
    open(path, "w") do io
        _write_metadata_comments(io, "reuse-series-summary")
        println(io, join(_csv_escape.(collect(_series_summary_header())), ','))
        for result in results
            println(io, join(_csv_escape.(collect(_series_summary_row(result))), ','))
        end
    end
    return path
end

function write_table_tsv(path::AbstractString, header, rows; label::AbstractString = "summary")
    open(path, "w") do io
        _write_metadata_comments(io, label)
        println(io, join(string.(collect(header)), '\t'))
        for row in rows
            println(io, join(string.(collect(row)), '\t'))
        end
    end
    return path
end

function write_table_csv(path::AbstractString, header, rows; label::AbstractString = "summary")
    open(path, "w") do io
        _write_metadata_comments(io, label)
        println(io, join(_csv_escape.(string.(collect(header))), ','))
        for row in rows
            println(io, join(_csv_escape.(string.(collect(row))), ','))
        end
    end
    return path
end

function _summary_output_paths(argv = String[]; env_tsv::AbstractString = "REUSE_BENCH_TSV", env_csv::AbstractString = "REUSE_BENCH_CSV")
    tsv_path = get(ENV, env_tsv, nothing)
    csv_path = get(ENV, env_csv, nothing)
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if startswith(arg, "--summary-tsv=")
            tsv_path = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--summary-csv=")
            csv_path = split(arg, "=", limit = 2)[2]
        elseif arg == "--summary-tsv" && i < length(argv)
            i += 1
            tsv_path = argv[i]
        elseif arg == "--summary-csv" && i < length(argv)
            i += 1
            csv_path = argv[i]
        end
        i += 1
    end
    return tsv_path, csv_path
end

function maybe_write_reuse_summary(results::AbstractVector{ReuseBenchResult}; argv = String[])
    isempty(results) && return nothing
    tsv_path, csv_path = _summary_output_paths(argv)
    if tsv_path !== nothing && !isempty(tsv_path)
        write_reuse_summary_tsv(tsv_path, results)
        println("TSV summary written to ", tsv_path)
    end
    if csv_path !== nothing && !isempty(csv_path)
        write_reuse_summary_csv(csv_path, results)
        println("CSV summary written to ", csv_path)
    end
    return nothing
end

function maybe_write_reuse_series_summary(results::AbstractVector{ReuseBenchSeriesResult}; argv = String[])
    isempty(results) && return nothing
    tsv_path, csv_path = _summary_output_paths(argv)
    if tsv_path !== nothing && !isempty(tsv_path)
        write_reuse_series_summary_tsv(tsv_path, results)
        println("TSV summary written to ", tsv_path)
    end
    if csv_path !== nothing && !isempty(csv_path)
        write_reuse_series_summary_csv(csv_path, results)
        println("CSV summary written to ", csv_path)
    end
    return nothing
end

function maybe_write_table(header, rows; argv = String[], env_tsv::AbstractString, env_csv::AbstractString, label::AbstractString = "summary")
    tsv_path, csv_path = _summary_output_paths(argv; env_tsv = env_tsv, env_csv = env_csv)
    if tsv_path !== nothing && !isempty(tsv_path)
        write_table_tsv(tsv_path, header, rows; label = label)
        println("TSV ", label, " written to ", tsv_path)
    end
    if csv_path !== nothing && !isempty(csv_path)
        write_table_csv(csv_path, header, rows; label = label)
        println("CSV ", label, " written to ", csv_path)
    end
    return nothing
end
