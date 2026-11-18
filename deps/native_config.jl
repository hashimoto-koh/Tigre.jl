if !@isdefined(NATIVE_BUILD_SCHEME)
    const NATIVE_BUILD_SCHEME = "per-tu-localize-v1"
end

if !@isdefined(UPSTREAM_SOURCE_NAMES)
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
end

if !@isdefined(WRAPPER_SOURCE_NAMES)
    const WRAPPER_SOURCE_NAMES = [
        "tigre_julia_runtime.cpp",
        "tigre_julia_shim.cpp",
    ]
end

if !@isdefined(OVERLAY_SOURCE_NAMES)
    const OVERLAY_SOURCE_NAMES = [
        "TIGRE_common.hpp",
        "TIGRE_common.cpp",
    ]
end

if !@isdefined(PATCH_NAMES)
    const PATCH_NAMES = String[]
end
