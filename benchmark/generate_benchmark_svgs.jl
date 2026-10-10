using Printf

include(joinpath(@__DIR__, "reuse_bench_utils.jl"))

function _default_path(name::String)
    return joinpath("/tmp", name)
end

function _parse_args(args::Vector{String})
    output_dir = "/tmp"
    single = _default_path("first_order_single.tsv")
    variance = _default_path("ista_fista_variance.tsv")
    realistic = _default_path("first_order_realistic.tsv")
    decompose = _default_path("decompose_first_order.tsv")
    representative = _default_path("representative_geometry.tsv")
    representative_dense = _default_path("representative_geometry_dense.tsv")
    representative_wang = _default_path("representative_geometry_wang.tsv")
    title_main = "First-order Benchmark Summary"
    title_decompose = "First-order Decompose Summary"
    title_representative = "Representative Geometry Summary"
    run_main = true
    run_decompose = true
    run_representative = true
    main_args = String[]
    decompose_args = String[]
    representative_args = String[]

    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--output-dir=")
            output_dir = split(arg, "=", limit = 2)[2]
        elseif arg == "--output-dir" && i < length(args)
            i += 1
            output_dir = args[i]
        elseif startswith(arg, "--single=")
            single = split(arg, "=", limit = 2)[2]
        elseif arg == "--single" && i < length(args)
            i += 1
            single = args[i]
        elseif startswith(arg, "--variance=")
            variance = split(arg, "=", limit = 2)[2]
        elseif arg == "--variance" && i < length(args)
            i += 1
            variance = args[i]
        elseif startswith(arg, "--realistic=")
            realistic = split(arg, "=", limit = 2)[2]
        elseif arg == "--realistic" && i < length(args)
            i += 1
            realistic = args[i]
        elseif startswith(arg, "--decompose=")
            decompose = split(arg, "=", limit = 2)[2]
        elseif arg == "--decompose" && i < length(args)
            i += 1
            decompose = args[i]
        elseif startswith(arg, "--representative=")
            representative = split(arg, "=", limit = 2)[2]
        elseif arg == "--representative" && i < length(args)
            i += 1
            representative = args[i]
        elseif startswith(arg, "--representative-dense=")
            representative_dense = split(arg, "=", limit = 2)[2]
        elseif arg == "--representative-dense" && i < length(args)
            i += 1
            representative_dense = args[i]
        elseif startswith(arg, "--representative-wang=")
            representative_wang = split(arg, "=", limit = 2)[2]
        elseif arg == "--representative-wang" && i < length(args)
            i += 1
            representative_wang = args[i]
        elseif startswith(arg, "--title-main=")
            title_main = split(arg, "=", limit = 2)[2]
        elseif arg == "--title-main" && i < length(args)
            i += 1
            title_main = args[i]
        elseif startswith(arg, "--title-decompose=")
            title_decompose = split(arg, "=", limit = 2)[2]
        elseif arg == "--title-decompose" && i < length(args)
            i += 1
            title_decompose = args[i]
        elseif startswith(arg, "--title-representative=")
            title_representative = split(arg, "=", limit = 2)[2]
        elseif arg == "--title-representative" && i < length(args)
            i += 1
            title_representative = args[i]
        elseif arg == "--skip-main"
            run_main = false
        elseif arg == "--skip-decompose"
            run_decompose = false
        elseif arg == "--skip-representative"
            run_representative = false
        elseif arg in ("--hide-single-shot", "--show-single-shot", "--hide-variance", "--show-variance", "--hide-realistic", "--show-realistic", "--hide-source", "--show-source")
            push!(main_args, arg)
        elseif startswith(arg, "--single-sort=") || startswith(arg, "--variance-sort=") || startswith(arg, "--realistic-sort=")
            push!(main_args, arg)
        elseif arg in ("--single-sort", "--variance-sort", "--realistic-sort") && i < length(args)
            push!(main_args, arg)
            i += 1
            push!(main_args, args[i])
        elseif startswith(arg, "--decompose-sort=")
            push!(decompose_args, replace(arg, "--decompose-sort=" => "--sort="))
        elseif arg == "--decompose-sort" && i < length(args)
            i += 1
            append!(decompose_args, ["--sort", args[i]])
        elseif startswith(arg, "--decompose-top=")
            push!(decompose_args, replace(arg, "--decompose-top=" => "--top="))
        elseif arg == "--decompose-top" && i < length(args)
            i += 1
            append!(decompose_args, ["--top", args[i]])
        elseif arg == "--decompose-hide-source"
            push!(decompose_args, "--hide-source")
        elseif arg == "--decompose-show-source"
            push!(decompose_args, "--show-source")
        elseif arg == "--decompose-hide-ratio"
            push!(decompose_args, "--hide-ratio")
        elseif arg == "--decompose-show-ratio"
            push!(decompose_args, "--show-ratio")
        elseif startswith(arg, "--representative-sort=")
            push!(representative_args, replace(arg, "--representative-sort=" => "--sort="))
        elseif arg == "--representative-sort" && i < length(args)
            i += 1
            append!(representative_args, ["--sort", args[i]])
        elseif startswith(arg, "--representative-top=")
            push!(representative_args, replace(arg, "--representative-top=" => "--top="))
        elseif arg == "--representative-top" && i < length(args)
            i += 1
            append!(representative_args, ["--top", args[i]])
        elseif arg == "--representative-hide-source"
            push!(representative_args, "--hide-source")
        elseif arg == "--representative-show-source"
            push!(representative_args, "--show-source")
        elseif arg == "--representative-hide-ratio"
            push!(representative_args, "--hide-ratio")
        elseif arg == "--representative-show-ratio"
            push!(representative_args, "--show-ratio")
        else
            error("unknown argument: $arg")
        end
        i += 1
    end

    return (;
        output_dir,
        single,
        variance,
        realistic,
        decompose,
        representative,
        representative_dense,
        representative_wang,
        title_main,
        title_decompose,
        title_representative,
        run_main,
        run_decompose,
        run_representative,
        main_args,
        decompose_args,
        representative_args,
    )
end

function _run_plot(script::String, args::Vector{String})
    project_dir = dirname(@__DIR__)
    julia_bin = joinpath(Sys.BINDIR, Base.julia_exename())
    cmd = Cmd(vcat([julia_bin, "--startup-file=no", "--project=$project_dir", script], args))
    run(cmd)
    return nothing
end

function _checked_existing_inputs(paths::Vector{String})
    inputs = String[]
    for path in paths
        isfile(path) || continue
        summary_schema_ok(path) || error("unsupported benchmark summary schema in $path")
        push!(inputs, path)
    end
    return inputs
end

function _print_input_metadata(io::IO, label::AbstractString, paths::Vector{String})
    isempty(paths) && return nothing
    println(io, label, " inputs:")
    for path in paths
        metadata = read_summary_metadata(path)
        schema = get(metadata, "schema", "none")
        version = get(metadata, "schema_version", "none")
        table = get(metadata, "table", "unknown")
        println(io, @sprintf("  %-14s schema=%s version=%s table=%s", basename(path) * ":", schema, version, table))
    end
    return nothing
end

function _generate_main(config)
    inputs = _checked_existing_inputs([config.single, config.variance, config.realistic])
    isempty(inputs) && return nothing
    output = joinpath(config.output_dir, "first_order_summary.auto.svg")
    args = String["--output", output, "--title", config.title_main]
    append!(args, config.main_args)
    append!(args, inputs)
    _run_plot(joinpath(@__DIR__, "plot_first_order_summary.jl"), args)
    return output
end

function _generate_decompose(config)
    isfile(config.decompose) || return nothing
    summary_schema_ok(config.decompose) || error("unsupported benchmark summary schema in $(config.decompose)")
    output = joinpath(config.output_dir, "decompose_first_order_summary.auto.svg")
    args = String["--output", output, "--title", config.title_decompose]
    append!(args, config.decompose_args)
    push!(args, config.decompose)
    _run_plot(joinpath(@__DIR__, "plot_decompose_first_order_summary.jl"), args)
    return output
end

function _generate_representative(config)
    inputs = _checked_existing_inputs([config.representative, config.representative_dense, config.representative_wang])
    isempty(inputs) && return nothing
    output = joinpath(config.output_dir, "representative_geometry_summary.auto.svg")
    args = String["--output", output, "--title", config.title_representative]
    append!(args, config.representative_args)
    append!(args, inputs)
    _run_plot(joinpath(@__DIR__, "plot_representative_geometry_summary.jl"), args)
    return output
end

function main(args = copy(ARGS); io::IO = stdout)
    config = _parse_args(args)
    mkpath(config.output_dir)

    main_inputs = _checked_existing_inputs([config.single, config.variance, config.realistic])
    decompose_inputs = isfile(config.decompose) ? [config.decompose] : String[]
    for path in decompose_inputs
        summary_schema_ok(path) || error("unsupported benchmark summary schema in $path")
    end
    representative_inputs = _checked_existing_inputs([config.representative, config.representative_dense, config.representative_wang])

    generated = Pair{String, String}[]
    if config.run_main
        output = _generate_main(config)
        output === nothing || push!(generated, "main" => output)
    end
    if config.run_decompose
        output = _generate_decompose(config)
        output === nothing || push!(generated, "decompose" => output)
    end
    if config.run_representative
        output = _generate_representative(config)
        output === nothing || push!(generated, "representative" => output)
    end

    isempty(generated) && error("no matching summary TSV/CSV files were found")

    _print_input_metadata(io, "main", main_inputs)
    _print_input_metadata(io, "decompose", decompose_inputs)
    _print_input_metadata(io, "representative", representative_inputs)
    println(io, "Generated SVG files:")
    for (label, path) in generated
        println(io, @sprintf("  %-14s %s", label * ":", path))
    end
    return generated
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
