using TOML

const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))
const DEPS_DIR = joinpath(ROOT_DIR, "deps")
const SOURCE_ROOT = joinpath(DEPS_DIR, "src", "TIGRE")
const COMMON_DIR = joinpath(SOURCE_ROOT, "Common", "CUDA")
const WRAPPER_DIR = joinpath(DEPS_DIR, "wrapper")
const WRAPPER_SRC_DIR = joinpath(WRAPPER_DIR, "src")
const WRAPPER_OVERLAY_DIR = joinpath(WRAPPER_DIR, "overlay")
const UPSTREAM_CONFIG = joinpath(DEPS_DIR, "upstream.toml")
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
    "0001-piccs-free-dimgtv-no-reset.patch",
]
const _NATIVE_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

struct CGeometry
    nVoxelX::Cint
    nVoxelY::Cint
    nVoxelZ::Cint
    sVoxelX::Cfloat
    sVoxelY::Cfloat
    sVoxelZ::Cfloat
    dVoxelX::Cfloat
    dVoxelY::Cfloat
    dVoxelZ::Cfloat
    nDetecU::Cint
    nDetecV::Cint
    sDetecU::Cfloat
    sDetecV::Cfloat
    dDetecU::Cfloat
    dDetecV::Cfloat
    offOrigX::Ptr{Cfloat}
    offOrigY::Ptr{Cfloat}
    offOrigZ::Ptr{Cfloat}
    DSO::Ptr{Cfloat}
    offDetecU::Ptr{Cfloat}
    offDetecV::Ptr{Cfloat}
    DSD::Ptr{Cfloat}
    dRoll::Ptr{Cfloat}
    dPitch::Ptr{Cfloat}
    dYaw::Ptr{Cfloat}
    COR::Ptr{Cfloat}
    accuracy::Cfloat
    mode::Cint
end

"""
    native_library_path() -> String

Return the expected filesystem path to the compiled native shared library (`libtigre_julia`).
"""
native_library_path() = joinpath(ROOT_DIR, "deps", "usr", "lib", "libtigre_julia.$(Libdl.dlext)")

"""
    native_library_available() -> Bool

Check whether the compiled native library (`libtigre_julia`) exists on disk.
"""
native_library_available() = isfile(native_library_path())

"""
    native_build_info_path() -> String

Return the filesystem path to the build metadata file (`build-info.toml`).
"""
native_build_info_path() = joinpath(ROOT_DIR, "deps", "usr", "build-info.toml")

function _env_string(names::AbstractVector{<:AbstractString}, default::AbstractString)
    for name in names
        value = strip(get(ENV, name, ""))
        isempty(value) || return value
    end
    return default
end

function _configured_upstream()
    config = isfile(UPSTREAM_CONFIG) ? TOML.parsefile(UPSTREAM_CONFIG) : Dict{String, Any}()
    source = strip(get(config, "source", "https://github.com/CERN/TIGRE.git"))
    ref = strip(get(config, "ref", "main"))
    source = _env_string(["TIGREJL_UPSTREAM_SOURCE", "TIGRE_UPSTREAM_SOURCE"], source)
    ref = _env_string(["TIGREJL_UPSTREAM_REF", "TIGRE_UPSTREAM_REF"], ref)
    return (; source, ref)
end

const PATCH_DIR = joinpath(DEPS_DIR, "patches")

"""
    native_build_sources() -> Vector{String}

Return the full list of C++/CUDA source files, shims, overlays, and patches required to build the native library.
"""
function native_build_sources()
    return vcat(
        [joinpath(COMMON_DIR, name) for name in UPSTREAM_SOURCE_NAMES],
        [joinpath(WRAPPER_SRC_DIR, name) for name in WRAPPER_SOURCE_NAMES],
        [joinpath(WRAPPER_OVERLAY_DIR, name) for name in OVERLAY_SOURCE_NAMES],
        [joinpath(PATCH_DIR, name) for name in PATCH_NAMES],
    )
end

"""
    native_build_info() -> Union{Dict{String, Any}, Nothing}

Parse and return the native library build metadata dictionary from `build-info.toml`, or `nothing` if missing.
"""
function native_build_info()
    path = native_build_info_path()
    isfile(path) || return nothing
    return TOML.parsefile(path)
end

# Helper to resolve symlinks and normalize to a real path.
# Returns normpath if the file/directory does not exist.
function _realpath_if_exists(path::AbstractString)
    isfile(path) && return realpath(path)
    isdir(path) && return realpath(path)
    return normpath(path)
end

"""
    native_build_stale_reasons() -> Vector{String}

Evaluate whether the native library needs rebuilding and return a list of reasons (e.g. missing files, newer source timestamps).
"""
function native_build_stale_reasons()
    reasons = String[]
    libpath = native_library_path()
    isfile(libpath) || push!(reasons, "output library is missing")
    info = native_build_info()
    info === nothing && push!(reasons, "build-info.toml is missing")

    sources = native_build_sources()
    for path in sources
        isfile(path) || push!(reasons, "missing source: $(basename(path))")
    end

    if info !== nothing
        # Compare source lists using basename sets after resolving symlinks
        # to prevent false-positive stale detection (e.g. /home/jov/hst/git/... vs /home/jov/git/...).
        recorded_raw = get(info, "sources", String[])
        current_basenames = Set(basename(p) for p in sources)
        recorded_basenames = Set(basename(p) for p in recorded_raw)
        if current_basenames != recorded_basenames
            push!(reasons, "recorded source list differs")
        end
        upstream = _configured_upstream()
        get(info, "upstream_source", "") == upstream.source || push!(reasons, "recorded upstream source differs")
        get(info, "upstream_ref", "") == upstream.ref || push!(reasons, "recorded upstream ref differs")
    end

    if isfile(libpath)
        lib_mtime = mtime(libpath)
        for path in sources
            isfile(path) || continue
            mtime(path) <= lib_mtime || push!(reasons, "source is newer than output: $(basename(path))")
        end
    end

    return unique(reasons)
end

"""
    native_build_stale() -> Bool

Return `true` if the native library is missing or out-of-date compared to its sources.
"""
native_build_stale() = !isempty(native_build_stale_reasons())

"""
    library_status() -> NamedTuple

Return a summary NamedTuple reporting current status of the native library:
- `path`: Path to `libtigre_julia`.
- `available`: Whether library exists.
- `build_info_path`: Path to `build-info.toml`.
- `build_info_available`: Whether build metadata exists.
- `stale`: Whether rebuild is needed.
- `stale_reasons`: Detailed reasons if stale.
"""
library_status() = (;
    path = native_library_path(),
    available = native_library_available(),
    build_info_path = native_build_info_path(),
    build_info_available = isfile(native_build_info_path()),
    stale = native_build_stale(),
    stale_reasons = native_build_stale_reasons(),
)

"""
    build_native()

Trigger native library compilation via `deps/build.jl`.
"""
function build_native()
    return include(joinpath(ROOT_DIR, "deps", "build.jl"))
end

_mode_code(geo::Geometry) = geo.mode === :parallel ? Cint(1) : Cint(0)

function _projection_type_code(projection_type)
    projection_type = projection_type isa Symbol ? Symbol(lowercase(String(projection_type))) : Symbol(lowercase(string(projection_type)))
    projection_type in (:siddon, Symbol("ray-voxel")) && return Cint(0)
    projection_type === :interpolated && return Cint(1)
    throw(ArgumentError("projection_type must be :siddon or :interpolated"))
end

function _backprojection_type_code(backprojection_type)
    backprojection_type = backprojection_type isa Symbol ? Symbol(lowercase(String(backprojection_type))) : Symbol(lowercase(string(backprojection_type)))
    backprojection_type === :fdk && return Cint(0)
    backprojection_type === :matched && return Cint(1)
    throw(ArgumentError("backprojection_type must be :fdk or :matched"))
end

function _gpu_id_buffer(gpu_ids)
    gpu_ids === nothing && return Cint[]
    return Cint.(collect(gpu_ids))
end

function _pointer_or_null(v::Vector{T}) where {T}
    isempty(v) ? Ptr{T}(C_NULL) : pointer(v)
end

function _cgeometry(geo::Geometry)
    return CGeometry(
        Cint(geo.nVoxel[1]),
        Cint(geo.nVoxel[2]),
        Cint(geo.nVoxel[3]),
        Cfloat(geo.sVoxel[1]),
        Cfloat(geo.sVoxel[2]),
        Cfloat(geo.sVoxel[3]),
        Cfloat(geo.dVoxel[1]),
        Cfloat(geo.dVoxel[2]),
        Cfloat(geo.dVoxel[3]),
        Cint(geo.nDetector[1]),
        Cint(geo.nDetector[2]),
        Cfloat(geo.sDetector[1]),
        Cfloat(geo.sDetector[2]),
        Cfloat(geo.dDetector[1]),
        Cfloat(geo.dDetector[2]),
        pointer(geo.offOriginX),
        pointer(geo.offOriginY),
        pointer(geo.offOriginZ),
        pointer(geo.DSO),
        pointer(geo.offDetectorU),
        pointer(geo.offDetectorV),
        pointer(geo.DSD),
        pointer(geo.rotDetectorRoll),
        pointer(geo.rotDetectorPitch),
        pointer(geo.rotDetectorYaw),
        pointer(geo.COR),
        Cfloat(geo.accuracy),
        _mode_code(geo),
    )
end

function _native_library()
    native_library_available() && return native_library_path()
    error("native library is not built. Please run `import Pkg; Pkg.build(\"Tigre\")`.")
end

function _native_handle()
    if _NATIVE_HANDLE[] == C_NULL
        _NATIVE_HANDLE[] = Libdl.dlopen(_native_library())
    end
    return _NATIVE_HANDLE[]
end

function _native_error(handle, status::Integer, callname::AbstractString)
    status_text = unsafe_string(ccall(Libdl.dlsym(handle, :tigre_julia_error_string), Ptr{Cchar}, (Cint,), Cint(status)))
    tag = unsafe_string(ccall(Libdl.dlsym(handle, :tigre_julia_last_error_tag), Ptr{Cchar}, ()))
    message = unsafe_string(ccall(Libdl.dlsym(handle, :tigre_julia_last_error_message), Ptr{Cchar}, ()))
    isempty(tag) && isempty(message) && return error("$(callname) failed with status $(status): $(status_text)")
    return error("$(callname) failed with status $(status): $(status_text) [$(tag)] $(message)")
end

function _native_ax!(projection::StridedArray{Float32, 3}, volume::StridedArray{Float32, 3}, geo::Geometry, projection_type, gpu_ids)
    handle = _native_handle()
    gpu_buffer = _gpu_id_buffer(gpu_ids)
    status = GC.@preserve geo volume projection gpu_buffer begin
        cgeo = _cgeometry(geo)
        ccall(
            Libdl.dlsym(handle, :tigre_julia_ax),
            Cint,
            (Ptr{Cfloat}, Ref{CGeometry}, Ptr{Cfloat}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Cfloat}),
            pointer(volume),
            Ref(cgeo),
            pointer(geo.angles),
            Cint(projection_count(geo)),
            _projection_type_code(projection_type),
            _pointer_or_null(gpu_buffer),
            Cint(length(gpu_buffer)),
            pointer(projection),
        )
    end
    status == 0 || _native_error(handle, status, "Ax")
    return projection
end

function _native_atb!(volume::StridedArray{Float32, 3}, projection::StridedArray{Float32, 3}, geo::Geometry, backprojection_type, gpu_ids)
    handle = _native_handle()
    gpu_buffer = _gpu_id_buffer(gpu_ids)
    status = GC.@preserve geo volume projection gpu_buffer begin
        cgeo = _cgeometry(geo)
        ccall(
            Libdl.dlsym(handle, :tigre_julia_atb),
            Cint,
            (Ptr{Cfloat}, Ref{CGeometry}, Ptr{Cfloat}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Cfloat}),
            pointer(projection),
            Ref(cgeo),
            pointer(geo.angles),
            Cint(projection_count(geo)),
            _backprojection_type_code(backprojection_type),
            _pointer_or_null(gpu_buffer),
            Cint(length(gpu_buffer)),
            pointer(volume),
        )
    end
    status == 0 || _native_error(handle, status, "Atb")
    return volume
end

function _native_min_tv!(dst::StridedArray{Float32, 3}, src::StridedArray{Float32, 3}, alpha::Float32, maxiter::Integer, gpu_ids)
    handle = _native_handle()
    gpu_buffer = _gpu_id_buffer(gpu_ids)
    status = GC.@preserve dst src gpu_buffer begin
        ccall(
            Libdl.dlsym(handle, :tigre_julia_min_tv),
            Cint,
            (Ptr{Cfloat}, Cint, Cint, Cint, Cfloat, Cint, Ptr{Cint}, Cint, Ptr{Cfloat}),
            pointer(src),
            Cint(size(src, 1)),
            Cint(size(src, 2)),
            Cint(size(src, 3)),
            Cfloat(alpha),
            Cint(maxiter),
            _pointer_or_null(gpu_buffer),
            Cint(length(gpu_buffer)),
            pointer(dst),
        )
    end
    status == 0 || _native_error(handle, status, "minTV")
    return dst
end

function _native_awmin_tv!(dst::StridedArray{Float32, 3}, src::StridedArray{Float32, 3}, alpha::Float32, maxiter::Integer, delta::Float32, gpu_ids)
    handle = _native_handle()
    gpu_buffer = _gpu_id_buffer(gpu_ids)
    status = GC.@preserve dst src gpu_buffer begin
        ccall(
            Libdl.dlsym(handle, :tigre_julia_awmin_tv),
            Cint,
            (Ptr{Cfloat}, Cint, Cint, Cint, Cfloat, Cint, Cfloat, Ptr{Cint}, Cint, Ptr{Cfloat}),
            pointer(src),
            Cint(size(src, 1)),
            Cint(size(src, 2)),
            Cint(size(src, 3)),
            Cfloat(alpha),
            Cint(maxiter),
            Cfloat(delta),
            _pointer_or_null(gpu_buffer),
            Cint(length(gpu_buffer)),
            pointer(dst),
        )
    end
    status == 0 || _native_error(handle, status, "awminTV")
    return dst
end

function _native_tvdenoise!(dst::StridedArray{Float32, 3}, src::StridedArray{Float32, 3}, lambda::Float32, spacing::NTuple{3, Float32}, maxiter::Integer, gpu_ids)
    handle = _native_handle()
    gpu_buffer = _gpu_id_buffer(gpu_ids)
    spacing_buf = Float32[spacing[1], spacing[2], spacing[3]]
    status = GC.@preserve dst src gpu_buffer spacing_buf begin
        ccall(
            Libdl.dlsym(handle, :tigre_julia_tvdenoise),
            Cint,
            (Ptr{Cfloat}, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Cint, Ptr{Cint}, Cint, Ptr{Cfloat}),
            pointer(src),
            Cint(size(src, 1)),
            Cint(size(src, 2)),
            Cint(size(src, 3)),
            Cfloat(lambda),
            pointer(spacing_buf),
            Cint(maxiter),
            _pointer_or_null(gpu_buffer),
            Cint(length(gpu_buffer)),
            pointer(dst),
        )
    end
    status == 0 || _native_error(handle, status, "tvdenoise")
    return dst
end

function _native_piccs_tv!(dst::StridedArray{Float32, 3}, src::StridedArray{Float32, 3}, prior::StridedArray{Float32, 3}, alpha::Float32, ratio::Float32, maxiter::Integer, gpu_ids)
    handle = _native_handle()
    gpu_buffer = _gpu_id_buffer(gpu_ids)
    status = GC.@preserve dst src prior gpu_buffer begin
        ccall(
            Libdl.dlsym(handle, :tigre_julia_piccs_tv),
            Cint,
            (Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint, Cint, Cfloat, Cfloat, Cint, Ptr{Cint}, Cint, Ptr{Cfloat}),
            pointer(src),
            pointer(prior),
            Cint(size(src, 1)),
            Cint(size(src, 2)),
            Cint(size(src, 3)),
            Cfloat(alpha),
            Cfloat(ratio),
            Cint(maxiter),
            _pointer_or_null(gpu_buffer),
            Cint(length(gpu_buffer)),
            pointer(dst),
        )
    end
    status == 0 || _native_error(handle, status, "piccsTV")
    return dst
end

"""
    available_gpus() -> Vector{@NamedTuple{id::Int, name::String}}

Query and return the list of CUDA-capable GPU devices detected by the native TIGRE runtime.
Each element is a NamedTuple `(id = device_id, name = device_name)`.
"""
function available_gpus()
    handle = _native_handle()
    count = Int(ccall(Libdl.dlsym(handle, :tigre_julia_get_gpu_count), Cint, ()))
    devices = NamedTuple{(:id, :name), Tuple{Int, String}}[]
    buffer = Vector{UInt8}(undef, 128)
    for device_id in 0:(count - 1)
        fill!(buffer, 0x00)
        status = ccall(
            Libdl.dlsym(handle, :tigre_julia_get_gpu_name),
            Cint,
            (Cint, Ptr{UInt8}, Csize_t),
            Cint(device_id),
            pointer(buffer),
            Csize_t(length(buffer)),
        )
        status == 0 || _native_error(handle, status, "available_gpus")
        push!(devices, (id = device_id, name = unsafe_string(pointer(buffer))))
    end
    return devices
end
