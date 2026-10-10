function _bench_env_flag(name::AbstractString)
    value = lowercase(strip(get(ENV, name, "")))
    return value in ("1", "true", "yes", "on")
end

function ensure_native_benchmark_ready(label::AbstractString)
    if !native_library_available()
        println("skipped $(label) because native library is not built")
        println("  build: julia --startup-file=no --project=. -e 'include(\"deps/build.jl\")'")
        exit(0)
    end

    status = library_status()
    if status.stale
        println("warning: native build may be stale: ", label)
        for reason in status.stale_reasons
            println("  - ", reason)
        end
        println("  build info: ", status.build_info_path)
        if _bench_env_flag("TIGRE_BENCH_REQUIRE_FRESH")
            println("exiting because TIGRE_BENCH_REQUIRE_FRESH=1")
            exit(1)
        end
    end

    if _bench_env_flag("TIGRE_BENCH_PRINT_BUILD_INFO")
        info = native_build_info()
        if info === nothing
            println("build info: unavailable")
        else
            println("build info:")
            println("  path: ", status.build_info_path)
            println("  timestamp_utc: ", get(info, "timestamp_utc", ""))
            println("  nvcc: ", get(info, "nvcc", ""))
            println("  cuda_home: ", get(info, "cuda_home", ""))
        end
    end

    return status
end
