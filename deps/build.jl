using Dates
using Libdl
using TOML

const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))
const DEPS_DIR = @__DIR__
const SOURCE_DIR = joinpath(DEPS_DIR, "src")
const SOURCE_ROOT = joinpath(SOURCE_DIR, "TIGRE")
const COMMON_DIR = joinpath(SOURCE_ROOT, "Common", "CUDA")
const FETCH_INFO = joinpath(SOURCE_DIR, "source-info.toml")
const UPSTREAM_CONFIG = joinpath(DEPS_DIR, "upstream.toml")
const WRAPPER_DIR = joinpath(DEPS_DIR, "wrapper")
const WRAPPER_INCLUDE_DIR = joinpath(WRAPPER_DIR, "include")
const WRAPPER_SRC_DIR = joinpath(WRAPPER_DIR, "src")
const WRAPPER_OVERLAY_DIR = joinpath(WRAPPER_DIR, "overlay")
const PATCH_DIR = joinpath(DEPS_DIR, "patches")
const OUT_DIR = joinpath(ROOT_DIR, "deps", "usr", "lib")
const OUT_LIB = joinpath(OUT_DIR, "libtigre_julia.$(Libdl.dlext)")
const BUILD_INFO = joinpath(ROOT_DIR, "deps", "usr", "build-info.toml")
const DEFAULT_TIGRE_SOURCE = "https://github.com/CERN/TIGRE.git"
const DEFAULT_TIGRE_REF = "6b0951a8a88aa88a7db5b2a95181a33a95aa8b28"
const UPSTREAM_SOURCE_NAMES = [
    "projection.cpp",
    "TIGRE_common.cpp",
    "GpuIds.cpp",
    "gpuUtils.cu",
    "Siddon_projection.cu",
    "Siddon_projection_parallel.cu",
    "ray_interpolated_projection.cu",
    "ray_interpolated_projection_parallel.cu",
    "voxel_backprojection.cu",
    "voxel_backprojection2.cu",
    "voxel_backprojection_parallel.cu",
    "GD_TV.cu",
    "GD_AwTV.cu",
    "tv_proximal.cu",
    "PICCS.cu",
]
const WRAPPER_SOURCE_NAMES = [
    "tigre_julia_runtime.cpp",
    "tigre_julia_shim.cpp",
]
const OVERLAY_SOURCE_NAMES = [
    "TIGRE_common.hpp",
    "TIGRE_common.cpp",
]
const PATCH_NAMES = [
    "0001-siddon-return-on-missed-ray.patch",
    "0002-parallel-siddon-projection.patch",
]

function _env_flag(name::AbstractString)
    value = lowercase(strip(get(ENV, name, "")))
    return value in ("1", "true", "yes", "on")
end

function _env_path(primary::AbstractString, fallbacks::AbstractString...)
    for key in (primary, fallbacks...)
        haskey(ENV, key) || continue
        value = strip(ENV[key])
        isempty(value) || return value
    end
    return nothing
end

function _env_string(names::AbstractVector{<:AbstractString}, default::AbstractString)
    for name in names
        value = strip(get(ENV, name, ""))
        isempty(value) || return value
    end
    return default
end

function _read_upstream_config()
    config = isfile(UPSTREAM_CONFIG) ? TOML.parsefile(UPSTREAM_CONFIG) : Dict{String, Any}()
    source = strip(get(config, "source", DEFAULT_TIGRE_SOURCE))
    ref = strip(get(config, "ref", DEFAULT_TIGRE_REF))
    source = _env_string(["TIGREJL_UPSTREAM_SOURCE", "TIGRE_UPSTREAM_SOURCE"], source)
    ref = _env_string(["TIGREJL_UPSTREAM_REF", "TIGRE_UPSTREAM_REF"], ref)
    isempty(source) && error("TIGRE upstream source is empty. Please configure deps/upstream.toml or set TIGREJL_UPSTREAM_SOURCE.")
    isempty(ref) && error("TIGRE upstream ref is empty. Please configure deps/upstream.toml or set TIGREJL_UPSTREAM_REF.")
    return (; source, ref)
end

function _git_candidates()
    candidates = String[]
    override = _env_path("TIGREJL_GIT", "GIT")
    override !== nothing && push!(candidates, override)
    path_git = Sys.which("git")
    path_git !== nothing && push!(candidates, path_git)
    return unique(map(normpath, filter(!isempty, candidates)))
end

function find_git()
    for candidate in _git_candidates()
        isfile(candidate) && return candidate
    end
    error("git not found. git is required to acquire TIGRE upstream source. Please set PATH or TIGREJL_GIT.")
end

function run_logged(cmd::Cmd; dir::Union{Nothing, String} = nothing)
    shown = dir === nothing ? cmd : Cmd(cmd; dir = dir)
    println(">> ", shown)
    run(shown)
end

function reset_workdir(path::String)
    rm(path; recursive = true, force = true)
    mkpath(dirname(path))
end

function read_fetch_info()
    isfile(FETCH_INFO) || return nothing
    return TOML.parsefile(FETCH_INFO)
end

function write_fetch_info(source::AbstractString, ref::AbstractString)
    mkpath(dirname(FETCH_INFO))
    open(FETCH_INFO, "w") do io
        TOML.print(io, Dict(
            "source" => source,
            "ref" => ref,
            "timestamp_utc" => Dates.format(now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
        ))
    end
    return FETCH_INFO
end

function source_checkout_matches(source::AbstractString, ref::AbstractString)
    info = read_fetch_info()
    info === nothing && return false
    get(info, "source", "") == source || return false
    get(info, "ref", "") == ref || return false
    return isdir(COMMON_DIR)
end

function reset_upstream_source!(git::AbstractString, ref::AbstractString)
    run_logged(`$git checkout -f $ref`; dir = SOURCE_ROOT)
    run_logged(`$git clean -fd`; dir = SOURCE_ROOT)
    return SOURCE_ROOT
end

function acquire_tigre_source!(git::AbstractString, source::AbstractString, ref::AbstractString)
    force_fetch = _env_flag("TIGRE_BUILD_FETCH_FORCE")
    if !force_fetch && source_checkout_matches(source, ref)
        println("Upstream TIGRE source already present at requested ref: ", ref)
        reset_upstream_source!(git, ref)
        return SOURCE_ROOT
    end

    reset_workdir(SOURCE_ROOT)
    run_logged(`$git clone $source $SOURCE_ROOT`)
    run_logged(`$git checkout $ref`; dir = SOURCE_ROOT)
    write_fetch_info(source, ref)
    return SOURCE_ROOT
end

upstream_sources() = [joinpath(COMMON_DIR, name) for name in UPSTREAM_SOURCE_NAMES]
wrapper_sources() = [joinpath(WRAPPER_SRC_DIR, name) for name in WRAPPER_SOURCE_NAMES]
overlay_sources() = [joinpath(WRAPPER_OVERLAY_DIR, name) for name in OVERLAY_SOURCE_NAMES]
patch_files() = [joinpath(PATCH_DIR, name) for name in PATCH_NAMES]
tracked_sources() = vcat(upstream_sources(), wrapper_sources(), overlay_sources(), patch_files())

function ensure_overlay_files!()
    for name in OVERLAY_SOURCE_NAMES
        src = joinpath(WRAPPER_OVERLAY_DIR, name)
        dest = joinpath(COMMON_DIR, name)
        isfile(src) || error("overlay file not found: $(src)")
        if !isfile(dest) || read(src, String) != read(dest, String)
            cp(src, dest; force = true)
        end
    end
    return nothing
end

function _replace_once_preserving_eol!(path::AbstractString, old_lf::AbstractString, new_lf::AbstractString, label::AbstractString)
    text = read(path, String)
    new_crlf = replace(new_lf, "\n" => "\r\n")
    old_crlf = replace(old_lf, "\n" => "\r\n")
    if occursin(new_lf, text) || occursin(new_crlf, text)
        println("Patch already applied: ", label)
        return false
    elseif occursin(old_crlf, text)
        write(path, replace(text, old_crlf => new_crlf; count = 1))
        println("Applied patch: ", label)
        return true
    elseif occursin(old_lf, text)
        write(path, replace(text, old_lf => new_lf; count = 1))
        println("Applied patch: ", label)
        return true
    else
        error("patch context not found: $(label) in $(path)")
    end
end

function _apply_git_patch_if_needed!(git::AbstractString, patch_path::AbstractString, workdir::AbstractString, label::AbstractString)
    rev_cmd = Cmd(`$git apply --reverse --check $patch_path`; dir = workdir)
    rev_check = run(pipeline(ignorestatus(rev_cmd); stdout = devnull, stderr = devnull))
    if rev_check.exitcode == 0
        println("Patch already applied: ", label)
        return false
    end
    fwd_cmd = Cmd(`$git apply --check $patch_path`; dir = workdir)
    fwd_check = run(pipeline(ignorestatus(fwd_cmd); stdout = devnull, stderr = devnull))
    if fwd_check.exitcode == 0
        apply_cmd = Cmd(`$git apply $patch_path`; dir = workdir)
        run(apply_cmd)
        println("Applied patch: ", label)
        return true
    end
    error("patch cannot be applied forward or reversed: $(label) for $(patch_path)")
end

function apply_upstream_patches!(git::AbstractString = find_git())
    for patch in patch_files()
        isfile(patch) || error("patch file not found: $(patch)")
    end
    _replace_once_preserving_eol!(
        joinpath(COMMON_DIR, "Siddon_projection.cu"),
        "    if (am>=aM)\n        detector[idx]=0;\n",
        "    if (am>=aM) {\n        detector[idx]=0;\n        return;\n    }\n",
        "Siddon_projection.cu missed-ray return",
    )
    _replace_once_preserving_eol!(
        joinpath(COMMON_DIR, "Siddon_projection_parallel.cu"),
        "    if (am>=aM)\n        detector[idx]=0.0f;\n",
        "    if (am>=aM) {\n        detector[idx]=0.0f;\n        return;\n    }\n",
        "Siddon_projection_parallel.cu missed-ray return",
    )
    patch_0002 = joinpath(PATCH_DIR, "0002-parallel-siddon-projection.patch")
    if isfile(patch_0002)
        _apply_git_patch_if_needed!(git, patch_0002, SOURCE_ROOT, "0002-parallel-siddon-projection")
    end
    return nothing
end

function _nvcc_candidates()
    candidates = String[]
    override = _env_path("CUDA_NVCC")
    override !== nothing && push!(candidates, override)

    for root in (_env_path("CUDA_HOME", "CUDA_PATH"), "/usr/local/cuda", "/opt/cuda")
        root === nothing && continue
        push!(candidates, joinpath(root, "bin", "nvcc"))
    end

    path_nvcc = Sys.which("nvcc")
    path_nvcc !== nothing && push!(candidates, path_nvcc)
    return unique(map(normpath, filter(!isempty, candidates)))
end

function find_nvcc()
    for candidate in _nvcc_candidates()
        isfile(candidate) && return candidate
    end
    error(
        "nvcc not found. Please install CUDA toolkit and configure PATH, CUDA_HOME/CUDA_PATH, or CUDA_NVCC. " *
        "In sandbox environments, elevated permissions may be required."
    )
end

function cuda_home(nvcc::AbstractString)
    override = _env_path("CUDA_HOME", "CUDA_PATH")
    override !== nothing && return normpath(override)
    return normpath(joinpath(dirname(nvcc), ".."))
end

function cuda_arch_flags(nvcc::AbstractString)
    override = _env_path("TIGRE_CUDA_ARCH_FLAGS")
    override !== nothing && return split(override)
    try
        archs = split(readchomp(`$nvcc --list-gpu-arch`))
        flags = String[]
        for arch in archs
            startswith(arch, "compute_") || continue
            sm = split(arch, "_"; limit = 2)[2]
            push!(flags, "-gencode=arch=$(arch),code=sm_$(sm)")
        end
        return unique(flags)
    catch
        return String[]
    end
end

function extra_nvcc_flags()
    override = _env_path("TIGRE_EXTRA_NVCC_FLAGS")
    override === nothing && return String[]
    return split(override)
end

function cuda_include_dir(cuda::AbstractString)
    override = _env_path("CUDA_INCLUDE_DIR")
    return normpath(override === nothing ? joinpath(cuda, "include") : override)
end

function cuda_lib_dir(cuda::AbstractString)
    override = _env_path("CUDA_LIB_DIR")
    if override !== nothing
        return normpath(override)
    end
    lib64 = joinpath(cuda, "lib64")
    return normpath(isdir(lib64) ? lib64 : joinpath(cuda, "lib"))
end

required_sources() = vcat(upstream_sources(), wrapper_sources())

function source_records(sources::Vector{String})
    return [Dict(
        "path" => path,
        "mtime" => mtime(path),
        "size" => stat(path).size,
    ) for path in sources]
end

function read_build_info()
    isfile(BUILD_INFO) || return nothing
    return TOML.parsefile(BUILD_INFO)
end

function ensure_inputs!(sources, include_dir::AbstractString, lib_dir::AbstractString)
    missing = String[]
    isdir(COMMON_DIR) || push!(missing, COMMON_DIR)
    isdir(WRAPPER_INCLUDE_DIR) || push!(missing, WRAPPER_INCLUDE_DIR)
    isdir(include_dir) || push!(missing, include_dir)
    isdir(lib_dir) || push!(missing, lib_dir)
    append!(missing, filter(path -> !isfile(path), overlay_sources()))
    append!(missing, filter(path -> !isfile(path), patch_files()))
    append!(missing, filter(path -> !isfile(path), sources))
    isempty(missing) || error("missing inputs for native build: " * join(missing, ", "))
    return nothing
end

function build_command(nvcc::AbstractString, include_dir::AbstractString, lib_dir::AbstractString, sources::Vector{String})
    cmd = String[
        nvcc,
        "--shared",
        "-O3",
        "--std=c++14",
        "--expt-relaxed-constexpr",
        "-Xcompiler=-fPIC",
        "-Xlinker=--allow-multiple-definition",
        "-DIS_FOR_JULIA_TIGRE",
        "-I$(COMMON_DIR)",
        "-I$(WRAPPER_INCLUDE_DIR)",
        "-I$(include_dir)",
    ]
    append!(cmd, cuda_arch_flags(nvcc))
    append!(cmd, extra_nvcc_flags())
    append!(cmd, sources)
    append!(cmd, ["-L$(lib_dir)", "-lcudart", "-o", OUT_LIB])
    return cmd
end

function write_build_info(cmd::Vector{String}, nvcc::AbstractString, cuda::AbstractString, include_dir::AbstractString, lib_dir::AbstractString, sources::Vector{String}, upstream_source::AbstractString, upstream_ref::AbstractString)
    info = Dict(
        "output" => OUT_LIB,
        "timestamp_utc" => Dates.format(now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
        "julia_version" => string(VERSION),
        "platform" => string(Sys.MACHINE),
        "upstream_source" => upstream_source,
        "upstream_ref" => upstream_ref,
        "source_root" => SOURCE_ROOT,
        "nvcc" => nvcc,
        "cuda_home" => cuda,
        "cuda_include_dir" => include_dir,
        "cuda_lib_dir" => lib_dir,
        "command" => cmd,
        "arch_flags" => cuda_arch_flags(nvcc),
        "extra_nvcc_flags" => extra_nvcc_flags(),
        "sources" => sources,
        "source_records" => source_records(sources),
        "output_mtime" => isfile(OUT_LIB) ? mtime(OUT_LIB) : 0.0,
        "env" => Dict(
            "CUDA_NVCC" => get(ENV, "CUDA_NVCC", ""),
            "CUDA_HOME" => get(ENV, "CUDA_HOME", ""),
            "CUDA_PATH" => get(ENV, "CUDA_PATH", ""),
            "CUDA_INCLUDE_DIR" => get(ENV, "CUDA_INCLUDE_DIR", ""),
            "CUDA_LIB_DIR" => get(ENV, "CUDA_LIB_DIR", ""),
            "TIGRE_CUDA_ARCH_FLAGS" => get(ENV, "TIGRE_CUDA_ARCH_FLAGS", ""),
            "TIGRE_EXTRA_NVCC_FLAGS" => get(ENV, "TIGRE_EXTRA_NVCC_FLAGS", ""),
        ),
    )
    mkpath(dirname(BUILD_INFO))
    open(BUILD_INFO, "w") do io
        TOML.print(io, info)
    end
    return BUILD_INFO
end

function stale_reasons(cmd::Vector{String}, sources::Vector{String}, upstream_source::AbstractString, upstream_ref::AbstractString)
    reasons = String[]
    isfile(OUT_LIB) || push!(reasons, "output library is missing")
    info = read_build_info()
    info === nothing && push!(reasons, "build-info.toml is missing")

    if info !== nothing
        # Normalize output path with realpath to handle symlink differences.
        recorded_output = get(info, "output", "")
        current_output = OUT_LIB
        recorded_real = isfile(recorded_output) ? realpath(recorded_output) : normpath(recorded_output)
        current_real = isfile(current_output) ? realpath(current_output) : normpath(current_output)
        recorded_real == current_real || push!(reasons, "recorded output path differs")
        get(info, "upstream_source", "") == upstream_source || push!(reasons, "recorded upstream source differs")
        get(info, "upstream_ref", "") == upstream_ref || push!(reasons, "recorded upstream ref differs")
        get(info, "command", String[]) == cmd || push!(reasons, "recorded build command differs")
        # Compare source list by basename sets to handle symlink differences.
        recorded_sources = get(info, "sources", String[])
        current_basenames = Set(basename(p) for p in sources)
        recorded_basenames = Set(basename(p) for p in recorded_sources)
        current_basenames == recorded_basenames || push!(reasons, "recorded source list differs")
    end

    if isfile(OUT_LIB)
        out_mtime = mtime(OUT_LIB)
        for path in sources
            mtime(path) <= out_mtime || push!(reasons, "source is newer than output: $(basename(path))")
        end
    end

    return unique(reasons)
end

function print_build_plan(cmd::Vector{String}, nvcc::AbstractString, cuda::AbstractString, include_dir::AbstractString, lib_dir::AbstractString, upstream_source::AbstractString, upstream_ref::AbstractString)
    println("Build target: ", OUT_LIB)
    println("  upstream_source: ", upstream_source)
    println("  upstream_ref: ", upstream_ref)
    println("  source_root: ", SOURCE_ROOT)
    println("  nvcc: ", nvcc)
    println("  cuda_home: ", cuda)
    println("  include_dir: ", include_dir)
    println("  lib_dir: ", lib_dir)
    println("  arch_flags: ", isempty(cuda_arch_flags(nvcc)) ? "(nvcc default)" : join(cuda_arch_flags(nvcc), " "))
    !isempty(extra_nvcc_flags()) && println("  extra_nvcc_flags: ", join(extra_nvcc_flags(), " "))
    println("  command: ", join(cmd, " "))
end

function build()
    upstream = _read_upstream_config()
    git = find_git()
    acquire_tigre_source!(git, upstream.source, upstream.ref)
    ensure_overlay_files!()
    apply_upstream_patches!()

    nvcc = find_nvcc()
    cuda = cuda_home(nvcc)
    include_dir = cuda_include_dir(cuda)
    lib_dir = cuda_lib_dir(cuda)
    compile_sources = required_sources()
    sources = tracked_sources()
    ensure_inputs!(compile_sources, include_dir, lib_dir)

    mkpath(OUT_DIR)
    cmd = build_command(nvcc, include_dir, lib_dir, compile_sources)
    reasons = stale_reasons(cmd, sources, upstream.source, upstream.ref)
    force = _env_flag("TIGRE_BUILD_FORCE")
    dry_run = _env_flag("TIGRE_BUILD_DRY_RUN")
    verify_only = _env_flag("TIGRE_BUILD_VERIFY_ONLY")

    print_build_plan(cmd, nvcc, cuda, include_dir, lib_dir, upstream.source, upstream.ref)
    if isempty(reasons)
        println("  status: up-to-date")
    else
        println("  status: stale")
        for reason in reasons
            println("    - ", reason)
        end
    end

    if verify_only
        println("Verify-only mode: skipping compilation.")
        return (; output = OUT_LIB, stale = !isempty(reasons), reasons, command = cmd)
    end
    if dry_run
        println("Dry-run mode: skipping compilation.")
        return (; output = OUT_LIB, stale = !isempty(reasons), reasons, command = cmd)
    end
    if !force && isempty(reasons)
        println("Output is up-to-date; skipping compile.")
        return OUT_LIB
    end

    try
        run(Cmd(cmd))
    catch err
        println(stderr, "native build command failed:")
        println(stderr, "  ", join(cmd, " "))
        println(stderr, "  If needed, please set CUDA_NVCC / CUDA_HOME / CUDA_LIB_DIR / TIGRE_CUDA_ARCH_FLAGS.")
        rethrow(err)
    end

    info_path = write_build_info(cmd, nvcc, cuda, include_dir, lib_dir, sources, upstream.source, upstream.ref)
    println("Built ", OUT_LIB)
    println("Build info written to ", info_path)
    return OUT_LIB
end

build()
