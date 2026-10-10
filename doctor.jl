using Tigre

function _parse_args(args::Vector{String})
    strict = false
    json = false
    for arg in args
        if arg == "--strict"
            strict = true
        elseif arg == "--json"
            json = true
        else
            error("unknown argument: $arg")
        end
    end
    return (; strict, json)
end

function _print_kv(io::IO, key::AbstractString, value)
    println(io, rpad(key * ":", 20), value)
end

function _hint(status)
    if !status.available
        return "julia --startup-file=no --project=. -e 'import Pkg; Pkg.build(\"Tigre\")'"
    elseif status.stale
        return "TIGRE_BUILD_FORCE=1 julia --startup-file=no --project=. -e 'import Pkg; Pkg.build(\"Tigre\")'"
    else
        return "native build is available and fresh"
    end
end

function _gpu_probe()
    try
        gpus = available_gpus()
        return (
            status = "ok",
            gpus = [(id = gpu.id, name = gpu.name) for gpu in gpus],
            error = nothing,
        )
    catch err
        return (
            status = "error",
            gpus = NamedTuple{(:id, :name), Tuple{Int, String}}[],
            error = sprint(showerror, err),
        )
    end
end

function doctor_report()
    status = library_status()
    info = native_build_info()
    stale_reasons = native_build_stale_reasons()
    gpu_probe = _gpu_probe()
    hint = _hint(status)
    return (
        schema = "tigre-doctor-v1",
        schema_version = 1,
        library = status.path,
        available = status.available,
        build_info = status.build_info_path,
        build_info_exists = status.build_info_available,
        stale = status.stale,
        stale_reasons = stale_reasons,
        sources = native_build_sources(),
        build_metadata = info,
        gpu_probe = gpu_probe,
        hint = hint,
    )
end

function _json_escape(s::AbstractString)
    io = IOBuffer()
    for c in s
        if c == '"'
            print(io, "\\\"")
        elseif c == '\\'
            print(io, "\\\\")
        elseif c == '\n'
            print(io, "\\n")
        elseif c == '\r'
            print(io, "\\r")
        elseif c == '\t'
            print(io, "\\t")
        elseif Int(c) < 0x20
            print(io, "\\u", uppercase(string(Int(c), base = 16, pad = 4)))
        else
            print(io, c)
        end
    end
    return String(take!(io))
end

function _write_json(io::IO, value)
    if value === nothing
        print(io, "null")
    elseif value isa Bool
        print(io, value ? "true" : "false")
    elseif value isa Integer || value isa AbstractFloat
        print(io, value)
    elseif value isa AbstractString
        print(io, '"', _json_escape(value), '"')
    elseif value isa AbstractVector
        print(io, "[")
        for (i, item) in enumerate(value)
            i > 1 && print(io, ",")
            _write_json(io, item)
        end
        print(io, "]")
    elseif value isa NamedTuple
        _write_json(io, Dict(String(key) => getproperty(value, key) for key in keys(value)))
    elseif value isa AbstractDict
        print(io, "{")
        first = true
        for key in sort!(collect(keys(value)); by = x -> String(x))
            first || print(io, ",")
            first = false
            print(io, "\"", _json_escape(String(key)), "\":")
            _write_json(io, value[key])
        end
        print(io, "}")
    else
        print(io, '"', _json_escape(sprint(show, value)), '"')
    end
    return nothing
end

function print_doctor_text(io::IO, report)
    println(io, "Tigre.jl doctor")
    _print_kv(io, "library", report.library)
    _print_kv(io, "available", report.available)
    _print_kv(io, "build_info", report.build_info)
    _print_kv(io, "build_info_exists", report.build_info_exists)
    _print_kv(io, "stale", report.stale)

    if report.build_metadata !== nothing
        _print_kv(io, "timestamp_utc", get(report.build_metadata, "timestamp_utc", ""))
        _print_kv(io, "nvcc", get(report.build_metadata, "nvcc", ""))
        _print_kv(io, "cuda_home", get(report.build_metadata, "cuda_home", ""))
        _print_kv(io, "cuda_include_dir", get(report.build_metadata, "cuda_include_dir", ""))
        _print_kv(io, "cuda_lib_dir", get(report.build_metadata, "cuda_lib_dir", ""))
        _print_kv(io, "arch_flags", join(get(report.build_metadata, "arch_flags", String[]), " "))
    end

    println(io, "sources:")
    for path in report.sources
        println(io, "  - ", path)
    end

    println(io, "stale reasons:")
    if isempty(report.stale_reasons)
        println(io, "  (none)")
    else
        for reason in report.stale_reasons
            println(io, "  - ", reason)
        end
    end

    println(io, "gpu probe:")
    if report.gpu_probe.status == "error"
        println(io, "  failed: ", report.gpu_probe.error)
    else
        if isempty(report.gpu_probe.gpus)
            println(io, "  (no visible GPUs)")
        else
            for gpu in report.gpu_probe.gpus
                println(io, "  - id=", gpu.id, " name=", gpu.name)
            end
        end
    end

    println(io, "hint: ", report.hint)
    return nothing
end

function main(args = copy(ARGS); io::IO = stdout, exit_on_failure::Bool = true)
    opts = _parse_args(args)
    report = doctor_report()

    if opts.json
        _write_json(io, report)
        println(io)
    else
        print_doctor_text(io, report)
    end

    code = (opts.strict && (!report.available || report.stale)) ? 1 : 0
    if exit_on_failure
        code != 0 && exit(code)
        return nothing
    end
    return code
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
