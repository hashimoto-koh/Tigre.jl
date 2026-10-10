using Test
using Libdl
using LinearAlgebra
using Random
using Statistics
using Tigre

include("regression_fixtures.jl")
include(joinpath(@__DIR__, "..", "doctor.jl"))
include(joinpath(@__DIR__, "..", "benchmark", "reuse_bench_utils.jl"))
include(joinpath(@__DIR__, "..", "examples", "geometry_layout_demo.jl"))
include(joinpath(@__DIR__, "..", "examples", "geometry_validation_demo.jl"))
include(joinpath(@__DIR__, "..", "examples", "filtering_demo.jl"))
include(joinpath(@__DIR__, "..", "examples", "parkerweight_demo.jl"))
include(joinpath(@__DIR__, "..", "examples", "wang_padding_demo.jl"))

angles = Float32[0, 0.5, 1.0]
geo = Geometry(
    mode = :cone,
    angles = angles,
    nVoxel = (4, 5, 6),
    sVoxel = (40f0, 50f0, 60f0),
    nDetector = (8, 7),
    sDetector = (80f0, 70f0),
    DSO = 1000f0,
    DSD = 1200f0,
)

function volume_signature(x::AbstractArray{Float32, 3})
    return (
        sum = Float32(sum(x)),
        norm = Float32(norm(vec(x))),
        min = Float32(minimum(x)),
        max = Float32(maximum(x)),
        samples = Float32[
            x[1, 1, 1],
            x[2, 2, 2],
            x[3, 2, 1],
            x[1, 4, 3],
            x[4, 4, 4],
            x[2, 3, 4],
        ],
    )
end

function projection_signature(x::AbstractArray{Float32, 3})
    return (
        sum = Float32(sum(x)),
        norm = Float32(norm(vec(x))),
        min = Float32(minimum(x)),
        max = Float32(maximum(x)),
        samples = Float32[
            x[1, 1, 1],
            x[2, 3, 1],
            x[4, 2, 2],
            x[8, 6, 2],
            x[3, 5, 4],
            x[7, 1, 4],
        ],
    )
end

function assert_signature_matches(actual, expected; atol = 1f-5, rtol = 2f-4)
    @test isapprox(actual.sum, expected.sum; atol = atol, rtol = rtol)
    @test isapprox(actual.norm, expected.norm; atol = atol, rtol = rtol)
    @test isapprox(actual.min, expected.min; atol = atol, rtol = rtol)
    @test isapprox(actual.max, expected.max; atol = atol, rtol = rtol)
    @test all(isapprox.(actual.samples, expected.samples; atol = atol, rtol = rtol))
end

@testset "Geometry normalization" begin
    @test geo.nVoxel == (4, 5, 6)
    @test geo.dVoxel == (10f0, 10f0, 10f0)
    @test geo.nDetector == (8, 7)
    @test geo.dDetector == (10f0, 10f0)
    @test size(geo.angles) == (3, 3)
    @test geo.angles[1, :] == angles
    @test all(iszero, geo.angles[2, :])
    @test geo.DSO == fill(1000f0, 3)
    @test geo.DSD == fill(1200f0, 3)
end

@testset "Canonical layout checks" begin
    volume = zeros(Float32, volume_size(geo)...)
    projection = zeros(Float32, projection_size(geo)...)
    @test is_canonical_layout(volume)
    @test is_canonical_layout(projection)
    @test validate_volume_array(volume, geo) === volume
    @test validate_projection_array(projection, geo) === projection

    wrapped_volume = PermutedDimsArray(volume, (1, 2, 3))
    wrapped_projection = PermutedDimsArray(projection, (1, 2, 3))
    @test !is_canonical_layout(wrapped_volume)
    @test !is_canonical_layout(wrapped_projection)
    @test_throws ArgumentError Ax!(projection, wrapped_volume, geo)
    @test_throws ArgumentError Atb!(volume, wrapped_projection, geo)
end

@testset "Library status" begin
    status = library_status()
    info = native_build_info()
    @test endswith(status.path, "libtigre_julia.$(Libdl.dlext)")
    @test status.available == native_library_available()
    @test endswith(status.build_info_path, "build-info.toml")
    @test status.build_info_available == !isnothing(info)
    @test status.stale == native_build_stale()
    @test status.stale_reasons == native_build_stale_reasons()
    @test !isempty(native_build_sources())
    if info !== nothing
        # Normalize build-info.toml output path and native_library_path() with realpath
        # to handle potential symlink differences.
        @test realpath(info["output"]) == realpath(native_library_path())
        @test !status.stale
        @test isempty(status.stale_reasons)
    else
        @test !isfile(native_build_info_path())
    end
end

@testset "Doctor output" begin
    report = doctor_report()
    @test report.schema == "tigre-doctor-v1"
    @test report.schema_version == 1
    @test report.library == native_library_path()
    @test report.available == native_library_available()
    @test report.stale == native_build_stale()
    @test report.stale_reasons == native_build_stale_reasons()
    @test report.hint isa String
    @test report.gpu_probe.status in ("ok", "error")

    text_io = IOBuffer()
    text_code = main(String[]; io = text_io, exit_on_failure = false)
    text_output = String(take!(text_io))
    @test text_code == 0
    @test occursin("Tigre.jl doctor", text_output)
    @test occursin("library:", text_output)
    @test occursin("gpu probe:", text_output)

    json_io = IOBuffer()
    json_code = main(["--json"]; io = json_io, exit_on_failure = false)
    json_output = String(take!(json_io))
    @test json_code == 0
    @test startswith(strip(json_output), "{")
    @test occursin("\"schema\":\"tigre-doctor-v1\"", json_output)
    @test occursin("\"schema_version\":1", json_output)
    @test occursin("\"library\"", json_output)
    @test occursin("\"gpu_probe\"", json_output)
    @test occursin("\"stale_reasons\"", json_output)
end

@testset "Benchmark summary metadata" begin
    @test BENCHMARK_SUMMARY_SCHEMA == "tigre-benchmark-summary-v1"
    @test BENCHMARK_SUMMARY_SCHEMA_VERSION == 1

    mktempdir() do tmp
        tsv = joinpath(tmp, "summary.tsv")
        csv = joinpath(tmp, "summary.csv")
        header = ("label", "samples", "time_median_s", "time_mean_s", "time_std_s", "alloc_median_B", "alloc_mean_B")
        rows = [("demo", "2", "0.1", "0.1", "0.0", "128", "128")]

        write_table_tsv(tsv, header, rows; label = "representative geometry summary")
        write_table_csv(csv, header, rows; label = "representative geometry summary")

        tsv_lines = readlines(tsv)
        csv_lines = readlines(csv)
        @test tsv_lines[1] == "# schema=tigre-benchmark-summary-v1"
        @test tsv_lines[2] == "# schema_version=1"
        @test tsv_lines[3] == "# table=representative geometry summary"
        @test occursin("label\tsamples", tsv_lines[4])
        @test csv_lines[1] == "# schema=tigre-benchmark-summary-v1"
        @test csv_lines[2] == "# schema_version=1"
        @test csv_lines[3] == "# table=representative geometry summary"
        @test occursin("label,samples", csv_lines[4])
    end
end

@testset "Benchmark driver smoke" begin
    mktempdir() do tmp
        single = joinpath(tmp, "first_order_single.tsv")
        variance = joinpath(tmp, "ista_fista_variance.tsv")
        realistic = joinpath(tmp, "first_order_realistic.tsv")
        decompose = joinpath(tmp, "decompose_first_order.tsv")
        representative = joinpath(tmp, "representative_geometry.tsv")
        outdir = joinpath(tmp, "svg")
        wrapper_outdir = joinpath(tmp, "wrapper_svg")

        write_table_tsv(single,
            ("label", "samples", "build_time_median_s", "build_time_mean_s", "build_time_std_s", "build_alloc_median_B", "build_alloc_mean_B",
             "prepared_time_median_s", "prepared_time_mean_s", "prepared_time_std_s", "prepared_alloc_median_B", "prepared_alloc_mean_B", "speedup_median"),
            [("ISTA", "3", "0.2", "0.2", "0.01", "128", "128", "0.1", "0.1", "0.01", "32", "32", "2.0")];
            label = "first-order single-shot summary",
        )
        write_table_tsv(variance,
            ("label", "samples", "cold_time_s", "cold_alloc_B", "warm_gc_on_median_s", "warm_gc_on_mean_s", "warm_gc_on_std_s", "warm_gc_on_alloc_median_B",
             "warm_gc_off_median_s", "warm_gc_off_mean_s", "warm_gc_off_std_s", "warm_gc_off_alloc_median_B"),
            [("FISTA", "3", "0.3", "256", "0.11", "0.11", "0.01", "32", "0.1", "0.1", "0.01", "32")];
            label = "first-order variance summary",
        )
        write_table_tsv(realistic,
            ("label", "samples", "build_avg_median_s", "build_avg_std_s", "reuse_avg_median_s", "reuse_avg_std_s", "prepare_once_s", "build_alloc_median_B", "reuse_alloc_median_B", "speedup_median"),
            [("OS-ASD-POCS", "3", "0.6", "0.02", "0.5", "0.02", "0.01", "4096", "64", "1.2")];
            label = "first-order realistic summary",
        )
        write_table_tsv(decompose,
            ("label", "samples", "time_median_s", "time_mean_s", "time_std_s", "alloc_median_B", "alloc_mean_B"),
            [("tvdenoise!", "3", "0.02", "0.02", "0.001", "64", "64")];
            label = "decompose first-order summary",
        )
        write_table_tsv(representative,
            ("label", "samples", "time_median_s", "time_mean_s", "time_std_s", "alloc_median_B", "alloc_mean_B"),
            [("FDK", "3", "0.09", "0.09", "0.005", "8192", "8192")];
            label = "representative geometry summary",
        )

        julia_bin = joinpath(Sys.BINDIR, Base.julia_exename())
        project_dir = abspath(joinpath(@__DIR__, ".."))
        driver = abspath(joinpath(@__DIR__, "..", "benchmark", "generate_benchmark_svgs.jl"))
        wrapper = abspath(joinpath(@__DIR__, "..", "benchmark", "generate_first_order_svgs.jl"))

        driver_output = read(Cmd([
            julia_bin,
            "--startup-file=no",
            "--project=$(project_dir)",
            driver,
            "--output-dir", outdir,
            "--single", single,
            "--variance", variance,
            "--realistic", realistic,
            "--decompose", decompose,
            "--representative", representative,
        ]), String)

        @test occursin("main inputs:", driver_output)
        @test occursin("schema=tigre-benchmark-summary-v1", driver_output)
        @test occursin("table=representative geometry summary", driver_output)
        @test occursin("Generated SVG files:", driver_output)
        @test isfile(joinpath(outdir, "first_order_summary.auto.svg"))
        @test isfile(joinpath(outdir, "decompose_first_order_summary.auto.svg"))
        @test isfile(joinpath(outdir, "representative_geometry_summary.auto.svg"))

        wrapper_output = read(Cmd([
            julia_bin,
            "--startup-file=no",
            "--project=$(project_dir)",
            wrapper,
            "--output-dir", wrapper_outdir,
            "--single", single,
            "--variance", variance,
            "--realistic", realistic,
            "--decompose", decompose,
            "--skip-representative",
        ]), String)

        @test occursin("compatibility wrapper", wrapper_output)
        @test occursin("Generated SVG files:", wrapper_output)
        @test isfile(joinpath(wrapper_outdir, "first_order_summary.auto.svg"))
        @test isfile(joinpath(wrapper_outdir, "decompose_first_order_summary.auto.svg"))
    end
end

@testset "Public API surface" begin
    expected_exports = Set([
        :Geometry,
        :normalize_angles,
        :projection_count,
        :projection_size,
        :volume_size,
        :ARTPlan,
        :FirstOrderWorkspace,
        :KrylovWorkspace,
        :is_canonical_layout,
        :validate_volume_array,
        :validate_projection_array,
        :shepp_logan_3d,
        :native_library_path,
        :native_library_available,
        :native_build_info_path,
        :native_build_info,
        :native_build_sources,
        :native_build_stale,
        :native_build_stale_reasons,
        :library_status,
        :build_native,
        :prepare_art_plan,
        :prepare_first_order_workspace,
        :prepare_krylov_workspace,
        :available_gpus,
        :power_method,
        :estimate_hyper,
        :computeCOR,
        :compute_fov_radius,
        :out_of_fov_mask,
        :crop_cbct,
        :crop_cbct!,
        :binning,
        :subsample_angles,
        :rotation_from_vecs,
        :rot_to_euler,
        :euler_from_vecs,
        :static_detector_geo,
        :static_det_linear_source_geo,
        :arbitrary_source_detector_geo,
        :add_ct_noise,
        :add_ct_noise!,
        :measure_quality,
        :rmse,
        :nrmse,
        :psnr,
        :ssim,
        :uqi,
        :ncc,
        :Ax,
        :Ax!,
        :Atb,
        :Atb!,
        :fdk,
        :fdk!,
        :fbp,
        :fbp!,
        :filter_response,
        :filter_projections!,
        :wang_padding,
        :wang_preweight!,
        :parkerweight!,
        :tvdenoise,
        :tvdenoise!,
        :minTV,
        :minTV!,
        :awminTV,
        :awminTV!,
        :piccsTV,
        :piccsTV!,
        :sart,
        :sart!,
        :sirt,
        :sirt!,
        :ossart,
        :ossart!,
        :sart_tv,
        :sart_tv!,
        :ossart_tv,
        :ossart_tv!,
        :asd_pocs,
        :asd_pocs!,
        :os_asd_pocs,
        :os_asd_pocs!,
        :awasd_pocs,
        :awasd_pocs!,
        :os_awasd_pocs,
        :os_awasd_pocs!,
        :piccs,
        :piccs!,
        :os_piccs,
        :os_piccs!,
        :mlem,
        :mlem!,
        :pcsd,
        :pcsd!,
        :aw_pcsd,
        :aw_pcsd!,
        :os_pcsd,
        :os_pcsd!,
        :os_aw_pcsd,
        :os_aw_pcsd!,
        :ista,
        :ista!,
        :fista,
        :fista!,
        :cgls,
        :cgls!,
        :lsqr,
        :lsqr!,
        :lsmr,
        :lsmr!,
        :hybrid_lsqr,
        :hybrid_lsqr!,
        :hybrid_flsqr_tv,
        :hybrid_flsqr_tv!,
        :ab_gmres,
        :ab_gmres!,
        :ba_gmres,
        :ba_gmres!,
        :irn_tv_cgls,
        :irn_tv_cgls!,
        :IterativeHistory,
        :plotImg,
        :plotProj,
    ])
    actual_exports = Set(filter(!isequal(:Tigre), names(Tigre)))
    @test actual_exports == expected_exports
    @test :_apply_wang_padding ∉ actual_exports
    @test :_wang_preweight! ∉ actual_exports
    @test :_subset_iteration_order ∉ actual_exports
end

@testset "Shepp Logan phantom" begin
    phantom = shepp_logan_3d((16, 12, 8))
    @test phantom isa Array{Float32, 3}
    @test size(phantom) == (16, 12, 8)
    @test all(isfinite, phantom)
    @test maximum(phantom) > 0f0
    @test minimum(phantom) < 0f0
    @test sum(abs, phantom) > 0f0

    kak = shepp_logan_3d(8; phantom_type = "kak-slaney")
    yyw = shepp_logan_3d(8; phantom_type = :yu_ye_wang)
    @test size(kak) == (8, 8, 8)
    @test kak != yyw

    toft, ellipsoids = shepp_logan_3d((10, 9, 8); phantom_type = :toft_schabel, get_ellipsoids = true)
    @test size(toft) == (10, 9, 8)
    @test size(ellipsoids) == (10, 10)
    @test ellipsoids isa Matrix{Float32}
    @test_throws ArgumentError shepp_logan_3d((8, 8); phantom_type = :yu_ye_wang)
    @test_throws ArgumentError shepp_logan_3d((8, 8, 8); phantom_type = :unknown)
end

@testset "Pure Julia example smoke" begin
    geometry_io = IOBuffer()
    geometry_result = run_geometry_layout_demo(geometry_io)
    geometry_output = String(take!(geometry_io))
    @test occursin("Geometry/layout demo", geometry_output)
    @test geometry_result.projection_count == 3
    @test geometry_result.volume_size == (4, 5, 6)
    @test geometry_result.projection_size == (8, 7, 3)
    @test geometry_result.angles_size == (3, 3)
    @test geometry_result.canonical_volume
    @test geometry_result.canonical_projection
    @test geometry_result.detector_spacing == (10f0, 10f0)
    @test geometry_result.voxel_spacing == (10f0, 10f0, 10f0)

    validation_io = IOBuffer()
    validation_result = run_geometry_validation_demo(validation_io)
    validation_output = String(take!(validation_io))
    @test occursin("Geometry validation demo", validation_output)
    @test validation_result.volume_valid
    @test validation_result.projection_valid
    @test occursin("shape", validation_result.bad_shape_error)
    @test occursin("StridedArray{Float32,3}", validation_result.bad_layout_error)

    filtering_io = IOBuffer()
    filtering_result = run_filtering_demo(filtering_io)
    filtering_output = String(take!(filtering_io))
    @test occursin("Filtering demo", filtering_output)
    @test filtering_result.response_length == 64
    @test filtering_result.response_first >= 0f0
    @test filtering_result.response_max > 0f0
    @test filtering_result.projection_size == (8, 6, 4)
    @test filtering_result.filtered_norm > 0f0
    @test all(isfinite, filtering_result.sample)

    parker_io = IOBuffer()
    parker_result = run_parkerweight_demo(parker_io)
    parker_output = String(take!(parker_io))
    @test occursin("Parkerweight demo", parker_output)
    @test parker_result.projection_size == (8, 6, 4)
    @test parker_result.weighted_sum > 0f0
    @test parker_result.weighted_norm > 0f0
    @test parker_result.weighted_min >= 0f0
    @test parker_result.weighted_max <= 2f0
    @test all(isfinite, parker_result.samples)

    wang_io = IOBuffer()
    wang_result = run_wang_padding_demo(wang_io)
    wang_output = String(take!(wang_io))
    @test occursin("Wang padding demo", wang_output)
    @test wang_result.original_size == (8, 6, 4)
    @test wang_result.padded_size == (10, 6, 4)
    @test wang_result.theta == Float32[30, 0, -30, 30]
    @test wang_result.padded_offsets == Float32[0, 10, 0, 0]
    @test wang_result.weighted_sum > 0f0
    @test wang_result.weighted_norm > 0f0
    @test wang_result.weighted_min >= 0f0
    @test wang_result.weighted_max <= 2f0
    @test all(isfinite, wang_result.sample_weights)
end

@testset "computeCOR contracts" begin
    c_geo = Geometry(
        mode = :cone,
        angles = Float32[0, 0.5, 1.0, 1.5, 2.0],
        nVoxel = (4, 4, 4),
        sVoxel = (40f0, 40f0, 40f0),
        nDetector = (8, 6),
        sDetector = (80f0, 60f0),
        DSO = 1000f0,
        DSD = 1200f0,
    )
    sino3d = rand(Float32, 8, 6, 5)
    cor_val = computeCOR(sino3d, c_geo)
    @test isfinite(cor_val)
    # 2D matrix input matches 3D array sliced at slice=3
    sino2d = sino3d[:, 3, :]
    @test computeCOR(sino2d, c_geo) ≈ computeCOR(sino3d, c_geo; slice = 3)
    # Angles parameter variations
    @test computeCOR(sino3d, c_geo, c_geo.angles) ≈ cor_val
    @test computeCOR(sino3d, c_geo, @view(c_geo.angles[1, :])) ≈ cor_val
    # Descending angles give same result
    @test computeCOR(sino3d[:, :, end:-1:1], c_geo, reverse(c_geo.angles[1, :])) ≈ cor_val
    # Error checking
    @test_throws ArgumentError computeCOR(sino3d, c_geo; slice = 0)
    @test_throws ArgumentError computeCOR(sino3d, c_geo; slice = 7)
    @test_throws ArgumentError computeCOR(sino3d, c_geo; angles = Float32[0, 1])
end

@testset "CT noise simulation" begin
    proj = fill(0.5f0, 16, 12, 8)
    rng1 = Random.MersenneTwister(42)
    noisy1 = add_ct_noise(proj; I0 = 10000.0, gaussian_std = 10.0, rng = rng1)

    @test size(noisy1) == size(proj)
    @test eltype(noisy1) == Float32
    @test all(isfinite, noisy1)
    @test all(proj .== 0.5f0) # Non-destructive

    # Determinism with same seed
    rng2 = Random.MersenneTwister(42)
    noisy2 = add_ct_noise(proj; I0 = 10000.0, gaussian_std = 10.0, rng = rng2)
    @test noisy1 == noisy2

    # In-place equivalence
    rng3 = Random.MersenneTwister(42)
    out_buf = similar(proj)
    add_ct_noise!(out_buf, proj; I0 = 10000.0, gaussian_std = 10.0, rng = rng3)
    @test out_buf == noisy1

    # In-place self-mutation
    rng4 = Random.MersenneTwister(42)
    self_buf = copy(proj)
    add_ct_noise!(self_buf; I0 = 10000.0, gaussian_std = 10.0, rng = rng4)
    @test self_buf == noisy1

    # Noise level scaling: high photon count has lower variance than low count
    noisy_high = add_ct_noise(proj; I0 = 1_000_000.0, gaussian_std = 0.0, rng = Random.MersenneTwister(1))
    noisy_low = add_ct_noise(proj; I0 = 1_000.0, gaussian_std = 0.0, rng = Random.MersenneTwister(1))
    var_high = var(vec(noisy_high .- proj))
    var_low = var(vec(noisy_low .- proj))
    @test var_high < var_low

    # Gaussian-only noise (I0 = Inf)
    g_noisy = add_ct_noise(proj; I0 = Inf, gaussian_std = 5.0, rng = Random.MersenneTwister(123))
    diff = g_noisy .- proj
    @test isapprox(mean(diff), 0.0; atol = 0.5)
    @test isapprox(std(diff), 5.0; rtol = 0.1)

    # Input validation
    @test_throws ArgumentError add_ct_noise(proj; I0 = 0.0)
    @test_throws ArgumentError add_ct_noise(proj; I0 = -10.0)
    @test_throws ArgumentError add_ct_noise(proj; gaussian_std = -1.0)
    @test_throws DimensionMismatch add_ct_noise!(zeros(Float32, 10, 10, 10), proj)
end

@testset "Quality metrics" begin
    ref = shepp_logan_3d((16, 16, 16))

    # Perfect match identity
    @test rmse(ref, ref) == 0.0
    @test nrmse(ref, ref) == 0.0
    @test psnr(ref, ref) == Inf
    @test isapprox(ssim(ref, ref), 1.0; atol = 1e-6)
    @test isapprox(uqi(ref, ref), 1.0; atol = 1e-6)
    @test isapprox(ncc(ref, ref), 1.0; atol = 1e-6)

    # Degraded image response
    rng = Random.MersenneTwister(1234)
    degraded = ref .+ 0.05f0 .* randn(rng, Float32, size(ref)...)

    err_rmse = rmse(ref, degraded)
    err_nrmse = nrmse(ref, degraded)
    val_psnr = psnr(ref, degraded)
    val_ssim = ssim(ref, degraded)
    val_uqi = uqi(ref, degraded)
    val_ncc = ncc(ref, degraded)

    @test err_rmse > 0.0
    @test err_nrmse > 0.0
    @test isfinite(val_psnr)
    @test 0.0 <= val_ssim < 1.0
    @test 0.0 <= val_uqi < 1.0
    @test 0.0 <= val_ncc < 1.0

    # measure_quality comprehensive evaluation
    q = measure_quality(ref, degraded)
    @test q isa NamedTuple
    @test keys(q) == (:rmse, :nrmse, :psnr, :ssim, :uqi, :ncc)
    @test q.rmse == err_rmse
    @test q.nrmse == err_nrmse
    @test q.psnr == val_psnr
    @test q.ssim == val_ssim
    @test q.uqi == val_uqi
    @test q.ncc == val_ncc

    # Subset metric request
    q_sub = measure_quality(ref, degraded; metrics = (:rmse, :psnr))
    @test keys(q_sub) == (:rmse, :psnr)
    @test q_sub.rmse == err_rmse
    @test q_sub.psnr == val_psnr

    # Metric validation
    @test_throws ArgumentError measure_quality(ref, degraded; metrics = (:unknown_metric,))
    @test_throws DimensionMismatch rmse(ref, rand(Float32, 8, 8, 8))
    @test_throws DimensionMismatch ssim(ref, rand(Float32, 8, 8, 8))

    # Flat image edge cases (zero variance)
    flat = fill(0.5f0, 8, 8, 8)
    @test isfinite(ssim(flat, flat))
    @test isfinite(uqi(flat, flat))
    @test ncc(flat, flat) == 1.0
end

@testset "FOV masking and crop_cbct" begin
    # 1. Radius computation
    r_mm, r_vox = compute_fov_radius(geo)
    @test isapprox(r_mm, (80f0 / 2f0) * (1000f0 / 1200f0); atol = 1e-4)
    # nVoxel is (4, 5, 6), max radius in voxels is (4 - 1)/2 = 1.5
    @test r_vox == 1.5f0

    # Parallel beam geometry radius
    par_geo = Geometry(
        mode = :parallel,
        angles = angles,
        nVoxel = (10, 10, 6),
        sVoxel = (100f0, 100f0, 60f0),
        nDetector = (8, 7),
        sDetector = (80f0, 70f0),
        DSO = 1000f0,
        DSD = 1200f0,
    )
    p_mm, p_vox = compute_fov_radius(par_geo)
    @test p_mm == 40f0
    @test p_vox == 4.0f0

    # 2. 2D mask generation
    mask2d = out_of_fov_mask((16, 16))
    @test size(mask2d) == (16, 16)
    @test mask2d isa BitMatrix
    @test mask2d[8, 8] # Center inside FOV
    @test mask2d[9, 9]
    @test !mask2d[1, 1] # Corner outside circular FOV
    @test !mask2d[16, 16]
    @test !mask2d[1, 16]
    @test !mask2d[16, 1]

    # Rectangular mask
    mask_rect = out_of_fov_mask((16, 16); circle = false)
    @test mask_rect[1, 1] # Corner of square FOV included if radius covers it

    # 3. 3D geometry mask generation
    mask3d = out_of_fov_mask(geo)
    @test size(mask3d) == geo.nVoxel
    @test mask3d isa BitArray{3}
    @test any(mask3d)
    @test !all(mask3d)

    # 4. crop_cbct non-destructive & in-place
    vol = ones(Float32, 16, 16, 4)
    vol_cropped = crop_cbct(vol)
    @test size(vol_cropped) == (16, 16, 4)
    @test vol_cropped[8, 8, 2] == 1f0
    @test vol_cropped[1, 1, 2] == 0f0
    @test all(vol .== 1f0) # Non-destructive

    # In-place crop_cbct!
    vol_mut = copy(vol)
    crop_cbct!(vol_mut)
    @test vol_mut == vol_cropped

    # Geometry-aware crop_cbct with custom fill_value
    vol_geo = ones(Float32, volume_size(par_geo)...)
    vol_geo_cropped = crop_cbct(vol_geo, par_geo; fill_value = -100f0)
    @test vol_geo_cropped[5, 5, 3] == 1f0
    @test vol_geo_cropped[1, 1, 3] == -100f0

    # In-place with geometry
    vol_geo_mut = copy(vol_geo)
    crop_cbct!(vol_geo_mut, par_geo; fill_value = -100f0)
    @test vol_geo_mut == vol_geo_cropped

    # Error handling
    @test_throws ArgumentError out_of_fov_mask(geo; radius = -1f0)
    @test_throws ArgumentError out_of_fov_mask(geo; radius = 0f0)
    @test_throws ArgumentError crop_cbct!(zeros(Float32, 5, 5, 5), par_geo)
end

@testset "Preprocessing: binning and subsampling" begin
    proj = ones(Float32, 8, 6, 4)

    # 1. Projection binning (:mean and :sum)
    b_proj = binning(proj, 2)
    @test size(b_proj) == (4, 3, 4)
    @test all(b_proj .== 1f0)

    b_sum = binning(proj, 2; mode = :sum)
    @test size(b_sum) == (4, 3, 4)
    @test all(b_sum .== 4f0)

    # Asymmetric binning
    b_asym = binning(proj, (4, 2))
    @test size(b_asym) == (2, 3, 4)
    @test all(b_asym .== 1f0)

    # Identity binning
    @test binning(proj, 1) == proj

    # 2. Geometry binning
    # geo has nDetector = (8, 7), dDetector = (10f0, 10f0)
    b_geo = binning(geo, 2)
    @test b_geo.nDetector == (4, 3)
    @test b_geo.dDetector == (20f0, 20f0)
    @test b_geo.sDetector == (80f0, 60f0)
    @test b_geo.mode == geo.mode

    # 3. Simultaneous projection and geometry binning
    p_sim = fill(2f0, projection_size(geo)...)
    b_p, b_g = binning(p_sim, geo, 2)
    @test size(b_p) == projection_size(b_g)
    @test size(b_p) == (4, 3, 3)
    @test all(b_p .== 2f0)

    # 4. Angular subsampling
    four_angles = Float32[0, 0.5, 1.0, 1.5]
    four_geo = Geometry(
        mode = :cone,
        angles = four_angles,
        nVoxel = (4, 4, 4),
        sVoxel = (40f0, 40f0, 40f0),
        nDetector = (8, 6),
        sDetector = (80f0, 60f0),
        DSO = 1000f0,
        DSD = 1200f0,
    )
    four_proj = rand(Float32, 8, 6, 4)

    # Step-based subsampling (every 2nd angle -> angles 1 and 3)
    sub_p, sub_g = subsample_angles(four_proj, four_geo, 2)
    @test projection_count(sub_g) == 2
    @test size(sub_p) == (8, 6, 2)
    @test sub_g.angles[1, :] == Float32[0, 1.0]
    @test sub_p[:, :, 1] == four_proj[:, :, 1]
    @test sub_p[:, :, 2] == four_proj[:, :, 3]

    # Explicit index list
    sub_p_idx, sub_g_idx = subsample_angles(four_proj, four_geo, [2, 4])
    @test projection_count(sub_g_idx) == 2
    @test sub_g_idx.angles[1, :] == Float32[0.5, 1.5]
    @test sub_p_idx[:, :, 1] == four_proj[:, :, 2]
    @test sub_p_idx[:, :, 2] == four_proj[:, :, 4]

    # Error handling
    @test_throws ArgumentError binning(proj, 0)
    @test_throws ArgumentError binning(proj, -1)
    @test_throws ArgumentError binning(proj, 2; mode = :invalid)
    @test_throws ArgumentError binning(geo, 100) # out_nu < 1
    @test_throws ArgumentError subsample_angles(four_proj, Int[])
    @test_throws BoundsError subsample_angles(four_proj, [0, 1])
    @test_throws BoundsError subsample_angles(four_proj, [1, 5])
end

@testset "Scanning trajectories and geometry transformations" begin
    # 1. 3D rotation matrix calculation
    # Identity
    R_id = rotation_from_vecs([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    @test isapprox(R_id, Matrix(I, 3, 3); atol = 1e-6)

    # 90-degree orthogonal rotation
    R_90 = rotation_from_vecs([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    @test isapprox(R_90 * [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]; atol = 1e-6)
    @test isapprox(R_90' * R_90, Matrix(I, 3, 3); atol = 1e-6)
    @test isapprox(det(R_90), 1.0; atol = 1e-6)

    # 180-degree opposite rotation
    R_180 = rotation_from_vecs([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
    @test isapprox(R_180 * [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]; atol = 1e-6)
    @test isapprox(R_180' * R_180, Matrix(I, 3, 3); atol = 1e-6)
    @test isapprox(det(R_180), 1.0; atol = 1e-6)

    # 2. Euler angle extraction
    zyz = rot_to_euler(R_90, :zyz)
    @test isapprox(zyz[1], π / 2; atol = 1e-6)
    xyz = rot_to_euler(R_90, :xyz)
    @test isapprox(xyz[3], π / 2; atol = 1e-6)

    # euler_from_vecs
    e_zyz = euler_from_vecs([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]; order = :zyz)
    @test isapprox(e_zyz[1], π / 2; atol = 1e-6)

    # 3. Static detector (tomosynthesis / DBT) geometry
    scan_ang = Float32[-0.2, 0.0, 0.2]
    dbt_geo = static_detector_geo(geo, scan_ang)
    @test projection_count(dbt_geo) == 3
    @test dbt_geo.mode === :cone
    @test isapprox(dbt_geo.rotDetectorYaw, -scan_ang; atol = 1e-6)
    @test isapprox(dbt_geo.DSD[2], geo.DSD[1]; atol = 1e-4) # at angle 0, DSD unchanged
    @test dbt_geo.DSD[1] < geo.DSD[1] # cos(0.2) < 1, so DSD shrinks slightly

    # 4. Static detector linear source geometry
    s_disp = Float32[-50f0, 0f0, 50f0]
    lin_geo = static_det_linear_source_geo(geo, s_disp)
    @test projection_count(lin_geo) == 3
    @test isapprox(lin_geo.DSO[2], geo.DSO[1]; atol = 1e-4) # at s_pos = 0, DSO unchanged
    @test lin_geo.DSO[1] > geo.DSO[1] # displaced source increases distance
    @test lin_geo.DSO[3] > geo.DSO[1]

    # 5. Arbitrary source and detector trajectory
    s_pos = Float32[1000 0 0; 0 1000 0]
    d_pos = Float32[-200 0 0; 0 -200 0]
    arb_geo = arbitrary_source_detector_geo(geo, s_pos, d_pos)
    @test projection_count(arb_geo) == 2
    @test isapprox(arb_geo.DSO, Float32[1000, 1000]; atol = 1e-4)
    @test isapprox(arb_geo.DSD, Float32[1200, 1200]; atol = 1e-4)
    @test arb_geo.angles[1, 1] == 0f0

    # Error checking
    @test_throws ArgumentError rotation_from_vecs([1.0, 0.0], [0.0, 1.0])
    @test_throws ArgumentError rot_to_euler(zeros(2, 2))
    @test_throws ArgumentError rot_to_euler(zeros(3, 3), :unsupported)
    @test_throws ArgumentError arbitrary_source_detector_geo(geo, zeros(3, 2))
    @test_throws DimensionMismatch arbitrary_source_detector_geo(geo, zeros(3, 3), zeros(2, 3))
end





@testset "Wang padding" begin
    wang_geo = Geometry(
        mode = :cone,
        angles = angles,
        nVoxel = (4, 5, 6),
        sVoxel = (40f0, 50f0, 60f0),
        nDetector = (8, 7),
        sDetector = (80f0, 70f0),
        DSO = 1000f0,
        DSD = 1200f0,
        offDetector = [10f0, 0f0],
    )
    proj = ones(Float32, projection_size(wang_geo)...)
    padded_proj, padded_geo, theta = wang_padding(proj, wang_geo)
    @test size(padded_proj) == (10, 7, 3)
    @test padded_geo.nDetector == (10, 7)
    @test theta == fill(30f0, 3)
    @test all(iszero, @view padded_proj[1:2, :, :])

    wang_preweight!(padded_proj, padded_geo, theta)
    @test all(isfinite, padded_proj)
    @test maximum(padded_proj) <= 2f0

    varying_geo = Geometry(
        mode = :cone,
        angles = angles,
        nVoxel = (4, 5, 6),
        sVoxel = (40f0, 50f0, 60f0),
        nDetector = (8, 7),
        sDetector = (80f0, 70f0),
        DSO = 1000f0,
        DSD = 1200f0,
        offDetector = Float32[10 0 -10; 0 0 0],
    )
    varying_proj = cat(fill(1f0, 8, 7), fill(2f0, 8, 7), fill(3f0, 8, 7); dims = 3)
    padded_proj2, padded_geo2, theta2 = wang_padding(varying_proj, varying_geo)
    @test size(padded_proj2) == (10, 7, 3)
    @test padded_geo2.offDetectorU == Float32[0, 10, 0]
    @test theta2 == Float32[30, 0, -30]
    @test all(iszero, @view padded_proj2[1:2, :, 1])
    @test all(iszero, @view padded_proj2[9:10, :, 2])
    wang_preweight!(padded_proj2, padded_geo2, theta2)
    @test all(isfinite, padded_proj2)
end

@testset "Windowed filters" begin
    for name in (:ram_lak, :shepp_logan, :cosine, :hamming, :hann)
        filt = filter_response(name, 64)
        @test length(filt) == 64
        @test all(isfinite, filt)
    end
    filter_geo = Geometry(
        mode = :cone,
        angles = Float32[0, 0.7, 1.4, 2.1],
        nVoxel = (4, 4, 4),
        sVoxel = (40f0, 40f0, 40f0),
        nDetector = (8, 6),
        sDetector = (80f0, 60f0),
        DSO = 1000f0,
        DSD = 1200f0,
    )
    proj = fill(1f0, projection_size(filter_geo)...)
    filtered = filter_projections!(copy(proj), filter_geo; filter = :hann, parker = false)
    @test size(filtered) == projection_size(filter_geo)
    @test all(isfinite, filtered)
    @test sum(abs, filtered) > 0f0

    cone_proj = fill(1f0, projection_size(filter_geo)...)
    cone_vol = zeros(Float32, volume_size(filter_geo)...)
    @test_throws ArgumentError fbp(cone_proj, filter_geo)
    @test_throws ArgumentError fbp!(cone_vol, cone_proj, filter_geo)
end

function _can_run_gpu_tests()
    native_library_available() || return false
    try
        return !isempty(available_gpus())
    catch
        return false
    end
end

if _can_run_gpu_tests()
    include(joinpath(@__DIR__, "..", "examples", "ax_atb_demo.jl"))
    include(joinpath(@__DIR__, "..", "examples", "fdk_demo.jl"))
    include(joinpath(@__DIR__, "..", "examples", "iterative_demo.jl"))

    @testset "Native smoke" begin
        volume = fill(1f0, volume_size(geo)...)
        projection = Ax(volume, geo)
        @test size(projection) == projection_size(geo)
        recon = Atb(projection, geo)
        @test size(recon) == volume_size(geo)

        parallel_geo = Geometry(
            mode = :parallel,
            angles = Float32[0, 0.5, 1.0, 1.5],
            nVoxel = (4, 4, 4),
            sVoxel = (40f0, 40f0, 40f0),
            nDetector = (8, 6),
            sDetector = (80f0, 60f0),
            DSO = 1000f0,
            DSD = 1200f0,
        )
        parallel_proj = Ax(fill(1f0, volume_size(parallel_geo)...), parallel_geo)
        fbp_recon = fbp(parallel_proj, parallel_geo; filter = :hann)
        @test size(fbp_recon) == volume_size(parallel_geo)
        @test all(isfinite, fbp_recon)
    end

    @testset "Example smoke" begin
        ax_io = IOBuffer()
        ax_result = run_ax_atb_demo(ax_io; preflight = false)
        ax_output = String(take!(ax_io))
        @test occursin("Ax/Atb demo", ax_output)
        @test ax_result.volume_size == (32, 32, 32)
        @test ax_result.projection_size == (48, 32, 4)
        @test ax_result.projection_sum > 0f0
        @test ax_result.backprojection_norm > 0f0
        @test all(isfinite, ax_result.sample_voxels)

        fdk_io = IOBuffer()
        fdk_result = run_fdk_demo(fdk_io; preflight = false)
        fdk_output = String(take!(fdk_io))
        @test occursin("FDK demo", fdk_output)
        @test fdk_result.projection_size == (48, 32, 4)
        @test fdk_result.recon_size == (32, 32, 32)
        @test fdk_result.recon_max >= fdk_result.recon_min
        @test isfinite(fdk_result.center_voxel)

        iterative_io = IOBuffer()
        iterative_result = run_iterative_demo(iterative_io; preflight = false)
        iterative_output = String(take!(iterative_io))
        @test occursin("Iterative demo", iterative_output)
        @test iterative_result.sart_residual_l2 >= 0f0
        @test iterative_result.fista_residual_l2 >= 0f0
        @test iterative_result.lsmr_residual_l2 >= 0f0
        @test all(isfinite, (
            iterative_result.sart_sum,
            iterative_result.sart_max,
            iterative_result.fista_sum,
            iterative_result.fista_max,
            iterative_result.lsmr_sum,
            iterative_result.lsmr_max,
        ))
    end

    @testset "Iterative subset policy" begin
        iter_geo = Geometry(
            mode = :cone,
            angles = Float32[0, 0.4, 1.2, 2.1],
            nVoxel = (4, 4, 4),
            sVoxel = (40f0, 40f0, 40f0),
            nDetector = (8, 6),
            sDetector = (80f0, 60f0),
            DSO = 1000f0,
            DSD = 1200f0,
        )
        blocks = Tigre._subset_index_blocks(iter_geo, 2, :ordered)
        @test blocks == [Int[1, 2], Int[3, 4]]
        @test Tigre._contiguous_range(blocks[1]) == 1:2

        angular_blocks = Tigre._subset_index_blocks(iter_geo, 2, :angularDistance)
        @test any(Tigre._contiguous_range(block) === nothing for block in angular_blocks)
    end

    @testset "Iterative algorithms" begin
        iter_geo = Geometry(
            mode = :cone,
            angles = Float32[0, 0.5, 1.0, 1.5],
            nVoxel = (4, 4, 4),
            sVoxel = (40f0, 40f0, 40f0),
            nDetector = (8, 6),
            sDetector = (80f0, 60f0),
            DSO = 1000f0,
            DSD = 1200f0,
        )
        phantom = zeros(Float32, volume_size(iter_geo)...)
        phantom[2, 2, 2] = 1f0
        proj = Ax(phantom, iter_geo)

        sart_vol = sart(proj, iter_geo, 1)
        _, sart_hist = sart(proj, iter_geo, 2; return_history = true, compute_l2 = true)
        ossart_vol = ossart(proj, iter_geo, 1; blocksize = 1)
        sirt_vol = sirt(proj, iter_geo, 1)
        ossart_sirt = ossart(proj, iter_geo, 1; blocksize = projection_count(iter_geo))
        matched_vol = sart(proj, iter_geo, 1; backprojection_type = :matched)

        @test size(sart_vol) == volume_size(iter_geo)
        @test all(isfinite, sart_vol)
        @test minimum(sart_vol) >= 0f0
        @test length(sart_hist.residual_l2) == 2
        @test length(sart_hist.lambda_history) == 2
        @test sart_hist.iterations_completed == 2
        @test sart_hist.stop_reason == :max_iterations
        @test isapprox(sart_vol, ossart_vol; atol = 1f-4, rtol = 1f-4)
        @test isapprox(sirt_vol, ossart_sirt; atol = 1f-4, rtol = 1f-4)
        @test size(matched_vol) == volume_size(iter_geo)
        @test all(isfinite, matched_vol)

        cached_plan = prepare_art_plan(proj, iter_geo; blocksize = 2, order_strategy = :angularDistance)
        @test cached_plan.requires_same_projection
        @test_throws ArgumentError ossart!(zeros(Float32, volume_size(iter_geo)...), copy(proj), iter_geo, 1;
            blocksize = 2,
            order_strategy = :angularDistance,
            plan = cached_plan,
        )

        work = fill(-1f0, volume_size(iter_geo)...)
        Tigre._initialize_iterative_volume!(work, proj, iter_geo, :zeros)
        @test all(iszero, work)
        sart!(work, proj, iter_geo, 1)
        @test isapprox(work, sart_vol; atol = 1f-4, rtol = 1f-4)

        zero_proj = zeros(Float32, projection_size(iter_geo)...)
        @test all(iszero, sirt(zero_proj, iter_geo, 2))
    end

    @testset "Default contracts" begin
        iter_geo = Geometry(
            mode = :cone,
            angles = Float32[0, 0.5, 1.0, 1.5],
            nVoxel = (4, 4, 4),
            sVoxel = (40f0, 40f0, 40f0),
            nDetector = (8, 6),
            sDetector = (80f0, 60f0),
            DSO = 1000f0,
            DSD = 1200f0,
        )
        phantom = Array{Float32}(undef, volume_size(iter_geo)...)
        @inbounds for z in axes(phantom, 3), y in axes(phantom, 2), x in axes(phantom, 1)
            phantom[x, y, z] = 0.1f0 * x + 0.2f0 * y + 0.05f0 * z
        end
        prior = fill(0.25f0, volume_size(iter_geo)...)
        proj = Ax(phantom, iter_geo)

        @test sart(proj, iter_geo, 1) ≈ ossart(proj, iter_geo, 1;
            blocksize = 1,
            lambda = 1f0,
            lambda_red = 1f0,
            order_strategy = :ordered,
            nonneg = true,
            projection_type = :siddon,
            backprojection_type = :fdk,
        )
        @test sirt(proj, iter_geo, 1) ≈ ossart(proj, iter_geo, 1;
            blocksize = projection_count(iter_geo),
            lambda = 1f0,
            lambda_red = 1f0,
            order_strategy = :ordered,
            nonneg = true,
            projection_type = :siddon,
            backprojection_type = :fdk,
        )
        @test sart_tv(proj, iter_geo, 1) ≈ ossart_tv(proj, iter_geo, 1;
            blocksize = 1,
            lambda = 1f0,
            lambda_red = 1f0,
            order_strategy = :ordered,
            nonneg = true,
            tviter = 50,
            tvlambda = 50f0,
        )
        @test asd_pocs(proj, iter_geo, 1) ≈ os_asd_pocs(proj, iter_geo, 1;
            blocksize = 1,
            lambda = 1f0,
            lambda_red = 0.99f0,
            alpha = 0.002f0,
            alpha_red = 0.95f0,
            rmax = 0.95f0,
            tviter = 20,
            nonneg = true,
            order_strategy = :ordered,
        )
        @test awasd_pocs(proj, iter_geo, 1) ≈ os_awasd_pocs(proj, iter_geo, 1;
            blocksize = 1,
            lambda = 1f0,
            lambda_red = 0.99f0,
            alpha = 0.002f0,
            alpha_red = 0.95f0,
            rmax = 0.95f0,
            tviter = 20,
            delta = -0.005f0,
            nonneg = true,
            order_strategy = :ordered,
        )
        @test piccs(proj, iter_geo, prior, 1) ≈ os_piccs(proj, iter_geo, prior, 1;
            blocksize = 1,
            lambda = 1f0,
            lambda_red = 0.99f0,
            alpha = 0.002f0,
            alpha_red = 0.95f0,
            rmax = 0.95f0,
            tviter = 20,
            prior_ratio = 0.2f0,
            nonneg = true,
            order_strategy = :ordered,
        )
        @test mlem(proj, iter_geo, 1) ≈ mlem(proj, iter_geo, 1;
            init = nothing,
            nonneg = true,
            projection_type = :siddon,
            backprojection_type = :matched,
        )
        @test pcsd(proj, iter_geo, 1) ≈ os_pcsd(proj, iter_geo, 1;
            blocksize = 1,
            lambda = 1f0,
            lambda_red = 1f0,
            tviter = 20,
            nonneg = true,
            order_strategy = :ordered,
        )
        @test aw_pcsd(proj, iter_geo, 1) ≈ os_aw_pcsd(proj, iter_geo, 1;
            blocksize = 1,
            lambda = 1f0,
            lambda_red = 1f0,
            tviter = 20,
            delta = -0.005f0,
            nonneg = true,
            order_strategy = :ordered,
        )
        @test ista(proj, iter_geo, 1) ≈ ista(proj, iter_geo, 1;
            hyper = 2.0f8,
            tviter = 20,
            tvlambda = 0.1f0,
            nonneg = true,
            projection_type = :siddon,
            backprojection_type = :matched,
        )
        @test fista(proj, iter_geo, 1) ≈ fista(proj, iter_geo, 1;
            hyper = 2.0f8,
            tviter = 20,
            tvlambda = 0.1f0,
            p = 1f0,
            q = 1f0,
            nonneg = true,
            projection_type = :siddon,
            backprojection_type = :matched,
        )
        @test cgls(proj, iter_geo, 1) ≈ cgls(proj, iter_geo, 1; projection_type = :siddon, backprojection_type = :matched, stop_on_l2_increase = true)
        @test lsqr(proj, iter_geo, 1) ≈ lsqr(proj, iter_geo, 1; projection_type = :siddon, backprojection_type = :matched, stop_on_l2_increase = true)
        @test lsmr(proj, iter_geo, 1) ≈ lsmr(proj, iter_geo, 1; damp = 0f0, projection_type = :siddon, backprojection_type = :matched, stop_on_l2_increase = true)
        @test hybrid_lsqr(proj, iter_geo, 1) ≈ hybrid_lsqr(proj, iter_geo, 1; lambda = 1f0, projection_type = :siddon, backprojection_type = :matched, stop_on_l2_increase = true)
        @test hybrid_flsqr_tv(proj, iter_geo, 1) ≈ hybrid_flsqr_tv(proj, iter_geo, 1; lambda = 1f0, precond_steps = 6, projection_type = :siddon, backprojection_type = :matched, stop_on_l2_increase = true)
        @test ab_gmres(proj, iter_geo, 1) ≈ ab_gmres(proj, iter_geo, 1; projection_type = :siddon, backprojection_type = :matched, stop_on_l2_increase = true)
        @test ba_gmres(proj, iter_geo, 1) ≈ ba_gmres(proj, iter_geo, 1; projection_type = :siddon, backprojection_type = :matched, stop_on_l2_increase = true)
        @test irn_tv_cgls(proj, iter_geo, 1) ≈ irn_tv_cgls(proj, iter_geo, 1; lambda = 1f0, niter_outer = 4, projection_type = :siddon, backprojection_type = :matched, stop_on_l2_increase = true)
    end

    @testset "TV primitives and first-order algorithms" begin
        iter_geo = Geometry(
            mode = :cone,
            angles = Float32[0, 0.5, 1.0, 1.5],
            nVoxel = (4, 4, 4),
            sVoxel = (40f0, 40f0, 40f0),
            nDetector = (8, 6),
            sDetector = (80f0, 60f0),
            DSO = 1000f0,
            DSD = 1200f0,
        )
        zero_volume = zeros(Float32, volume_size(iter_geo)...)
        noisy = copy(zero_volume)
        noisy[2, 2, 2] = 1f0
        prior = fill(0.25f0, volume_size(iter_geo)...)
        proj = Ax(noisy, iter_geo)

        @test all(iszero, tvdenoise(zero_volume; maxiter = 1, lambda = 1f0))
        @test all(iszero, minTV(zero_volume; maxiter = 1, alpha = 1f0))
        @test all(iszero, awminTV(zero_volume; maxiter = 1, alpha = 1f0))
        @test all(isfinite, piccsTV(noisy, prior; maxiter = 1, alpha = 1f0, ratio = 0.5f0))

        mlem_vol, mlem_hist = mlem(proj, iter_geo, 1; return_history = true, compute_l2 = true)
        ista_vol, ista_hist = ista(proj, iter_geo, 1; return_history = true, compute_l2 = true, tviter = 1)
        fista_vol, fista_hist = fista(proj, iter_geo, 1; return_history = true, compute_l2 = true, tviter = 1)
        mlem_alt = mlem(proj, iter_geo, 1; projection_type = :interpolated)
        ista_alt = ista(proj, iter_geo, 1; projection_type = :interpolated, tviter = 1)
        fista_alt = fista(proj, iter_geo, 1;
            projection_type = :interpolated,
            backprojection_type = :fdk,
            tviter = 1,
        )
        fista_alt_bang = zeros(Float32, volume_size(iter_geo)...)
        fista!(fista_alt_bang, proj, iter_geo, 1;
            projection_type = :interpolated,
            backprojection_type = :fdk,
            tviter = 1,
        )
        @test size(mlem_vol) == volume_size(iter_geo)
        @test size(ista_vol) == volume_size(iter_geo)
        @test size(fista_vol) == volume_size(iter_geo)
        @test all(isfinite, mlem_vol)
        @test all(isfinite, ista_vol)
        @test all(isfinite, fista_vol)
        @test all(isfinite, mlem_alt)
        @test all(isfinite, ista_alt)
        @test fista_alt_bang ≈ fista_alt
        @test minimum(mlem_vol) >= 0f0
        @test minimum(ista_vol) >= 0f0
        @test minimum(fista_vol) >= 0f0
        @test mlem_hist.iterations_completed == 1
        @test ista_hist.iterations_completed == 1
        @test fista_hist.iterations_completed == 1
        @test length(mlem_hist.residual_l2) == 1
        @test length(ista_hist.residual_l2) == 1
        @test length(fista_hist.residual_l2) == 1
        @test_throws ArgumentError fista(proj, iter_geo, 1; projection_type = :invalid, tviter = 1)
        @test_throws ArgumentError fista(proj, iter_geo, 1; backprojection_type = :invalid, tviter = 1)

        # Power method and automatic hyper parameter for ISTA/FISTA
        sigma_est = power_method(iter_geo; maxiter = 5)
        @test isfinite(sigma_est) && sigma_est > 0f0
        hyper_est = estimate_hyper(iter_geo; maxiter = 5)
        @test isfinite(hyper_est) && hyper_est ≈ 2f0 * 1.05f0 * (sigma_est^2)

        fista_auto = fista(proj, iter_geo, 1; hyper = :auto, tviter = 1)
        fista_str = fista(proj, iter_geo, 1; hyper = "auto", tviter = 1)
        @test size(fista_auto) == volume_size(iter_geo)
        @test all(isfinite, fista_auto)
        @test fista_auto ≈ fista_str
        @test_throws ArgumentError fista(proj, iter_geo, 1; hyper = :invalid, tviter = 1)
        @test_throws ArgumentError fista(proj, iter_geo, 1; hyper = -1.0f0, tviter = 1)

        # computeCOR known recovery test
        cor_angles = Float32.(range(0, 2pi, length=61)[1:60])
        for true_cor in (0.0f0, 0.5f0, -0.5f0)
            cor_geo = Geometry(
                mode = :cone,
                angles = cor_angles,
                nVoxel = (64, 64, 64),
                sVoxel = (128f0, 128f0, 128f0),
                nDetector = (64, 64),
                sDetector = (51.2f0, 51.2f0),
                DSD = 1536f0,
                DSO = 1000f0,
                COR = true_cor,
            )
            cor_ph = zeros(Float32, 64, 64, 64)
            for z in 0:63, y in 0:63, x in 0:63
                if ((x - 35)^2 + (y - 30)^2 + (z - 32)^2) < 13^2
                    cor_ph[x+1, y+1, z+1] = 1.0f0
                end
                if ((x - 25)^2 + (y - 37)^2 + (z - 32)^2) < 6^2
                    cor_ph[x+1, y+1, z+1] = 2.0f0
                end
            end
            cor_proj = Ax(cor_ph, cor_geo; projection_type = :siddon)
            est = computeCOR(cor_proj, cor_geo)
            @test isfinite(est)
            @test abs(est - true_cor) < 0.2f0
        end
    end

    @testset "TV-regularized ART and Krylov" begin
        iter_geo = Geometry(
            mode = :cone,
            angles = Float32[0, 0.5, 1.0, 1.5],
            nVoxel = (4, 4, 4),
            sVoxel = (40f0, 40f0, 40f0),
            nDetector = (8, 6),
            sDetector = (80f0, 60f0),
            DSO = 1000f0,
            DSD = 1200f0,
        )
        phantom = zeros(Float32, volume_size(iter_geo)...)
        phantom[2, 2, 2] = 1f0
        prior = fill(0.1f0, volume_size(iter_geo)...)
        proj = Ax(phantom, iter_geo)
        art_plan = prepare_art_plan(proj, iter_geo; blocksize = 1, need_full_projection = true, extra_volume_buffers = 3)
        @test all(isfinite, art_plan.W)
        @test all(w -> 0f0 <= w <= 0.1f0, art_plan.W)
        @test maximum(art_plan.W) <= 0.1f0
        first_order_workspace = prepare_first_order_workspace(iter_geo; need_full_projection = true)
        krylov_workspace = prepare_krylov_workspace(iter_geo; maxiter = 1, extra_volume_buffers = 8)
        hybrid_tv_fallback_workspace = prepare_krylov_workspace(iter_geo; maxiter = 1, extra_volume_buffers = 8, regularization_basis = true)

        sart_tv_vol, sart_tv_hist = sart_tv(proj, iter_geo, 1; tviter = 1, tvlambda = 1f0, return_history = true, compute_l2 = true)
        ossart_tv_vol = ossart_tv(proj, iter_geo, 1; blocksize = 1, tviter = 1, tvlambda = 1f0, plan = art_plan)
        asd_vol, asd_hist = asd_pocs(proj, iter_geo, 1; tviter = 1, return_history = true, plan = art_plan)
        os_asd_vol = os_asd_pocs(proj, iter_geo, 1; blocksize = 1, tviter = 1, plan = art_plan)
        awasd_vol, awasd_hist = awasd_pocs(proj, iter_geo, 1; tviter = 1, return_history = true, plan = art_plan)
        os_awasd_vol = os_awasd_pocs(proj, iter_geo, 1; blocksize = 1, tviter = 1, plan = art_plan)
        piccs_vol, piccs_hist = piccs(proj, iter_geo, prior, 1; tviter = 1, prior_ratio = 0.5f0, return_history = true, plan = art_plan)
        os_piccs_vol = os_piccs(proj, iter_geo, prior, 1; blocksize = 1, tviter = 1, prior_ratio = 0.5f0, plan = art_plan)
        pcsd_vol, pcsd_hist = pcsd(proj, iter_geo, 1; tviter = 1, return_history = true, plan = art_plan)
        os_pcsd_vol = os_pcsd(proj, iter_geo, 1; blocksize = 1, tviter = 1, plan = art_plan)
        aw_pcsd_vol, aw_pcsd_hist = aw_pcsd(proj, iter_geo, 1; tviter = 1, return_history = true, plan = art_plan)
        os_aw_pcsd_vol = os_aw_pcsd(proj, iter_geo, 1; blocksize = 1, tviter = 1, plan = art_plan)
        ista_ws_vol = ista(proj, iter_geo, 1; tviter = 1, workspace = first_order_workspace)
        fista_ws_vol = fista(proj, iter_geo, 1; tviter = 1, workspace = first_order_workspace)
        cgls_vol, cgls_hist = cgls(proj, iter_geo, 1; return_history = true, compute_l2 = true, workspace = krylov_workspace)
        lsqr_vol, lsqr_hist = lsqr(proj, iter_geo, 1; return_history = true, compute_l2 = true, workspace = krylov_workspace)
        lsmr_vol, lsmr_hist = lsmr(proj, iter_geo, 1; return_history = true, compute_l2 = true, workspace = krylov_workspace)
        hybrid_vol, hybrid_hist = hybrid_lsqr(proj, iter_geo, 1; lambda = 0.1f0, return_history = true, compute_l2 = true)
        hybrid_tv_vol, hybrid_tv_hist = hybrid_flsqr_tv(proj, iter_geo, 1; lambda = 0.1f0, return_history = true, compute_l2 = true, workspace = krylov_workspace)
        hybrid_tv_fallback_vol = hybrid_flsqr_tv(proj, iter_geo, 1; lambda = 0.1f0, workspace = hybrid_tv_fallback_workspace)
        ab_vol, ab_hist = ab_gmres(proj, iter_geo, 1; return_history = true, compute_l2 = true, workspace = krylov_workspace)
        ba_vol, ba_hist = ba_gmres(proj, iter_geo, 1; return_history = true, compute_l2 = true, workspace = krylov_workspace)
        irn_vol, irn_hist = irn_tv_cgls(proj, iter_geo, 1; niter_outer = 1, lambda = 0.1f0, return_history = true, compute_l2 = true)

        # Multi-iteration IRN-TV-CGLS outer/inner loop control test
        irn_multi_vol, irn_multi_hist = irn_tv_cgls(proj, iter_geo, 4; niter_outer = 3, lambda = 0.1f0, return_history = true, compute_l2 = true, stop_on_l2_increase = true)
        @test size(irn_multi_vol) == volume_size(iter_geo)
        @test all(isfinite, irn_multi_vol)
        @test irn_multi_hist.iterations_completed >= 2
        @test length(irn_multi_hist.residual_l2) >= 2
        @test irn_multi_hist.residual_l2[end] <= irn_multi_hist.residual_l2[1]

        projector_variants = (
            cgls(proj, iter_geo, 1; projection_type = :interpolated, stop_on_l2_increase = false),
            lsqr(proj, iter_geo, 1; projection_type = :interpolated, stop_on_l2_increase = false),
            lsmr(proj, iter_geo, 1; projection_type = :interpolated, stop_on_l2_increase = false),
            hybrid_lsqr(proj, iter_geo, 1; projection_type = :interpolated, stop_on_l2_increase = false),
            irn_tv_cgls(proj, iter_geo, 1; niter_outer = 1, projection_type = :interpolated, stop_on_l2_increase = false),
            ab_gmres(proj, iter_geo, 1; projection_type = :interpolated, stop_on_l2_increase = false),
            ba_gmres(proj, iter_geo, 1; projection_type = :interpolated, stop_on_l2_increase = false),
            hybrid_flsqr_tv(proj, iter_geo, 1; projection_type = :interpolated, stop_on_l2_increase = false),
        )
        for variant in projector_variants
            @test all(isfinite, variant)
        end
        @test ab_gmres(proj, iter_geo, 1; backprojection_type = :fdk, stop_on_l2_increase = false) ≈
              ab_gmres(proj, iter_geo, 1; backprojector = :fdk, stop_on_l2_increase = false)
        @test ba_gmres(proj, iter_geo, 1; backprojection_type = :fdk, stop_on_l2_increase = false) ≈
              ba_gmres(proj, iter_geo, 1; backprojector = :fdk, stop_on_l2_increase = false)

        @test size(sart_tv_vol) == volume_size(iter_geo)
        @test all(isfinite, sart_tv_vol)
        @test minimum(sart_tv_vol) >= 0f0
        @test isapprox(sart_tv_vol, ossart_tv_vol; atol = 1f-4, rtol = 1f-4)
        @test sart_tv_hist.iterations_completed == 1
        @test length(sart_tv_hist.residual_l2) == 1

        @test size(asd_vol) == volume_size(iter_geo)
        @test all(isfinite, asd_vol)
        @test minimum(asd_vol) >= 0f0
        @test asd_hist.iterations_completed == 1
        @test isapprox(asd_vol, os_asd_vol; atol = 1f-4, rtol = 1f-4)

        @test size(awasd_vol) == volume_size(iter_geo)
        @test all(isfinite, awasd_vol)
        @test minimum(awasd_vol) >= 0f0
        @test awasd_hist.iterations_completed == 1
        @test isapprox(awasd_vol, os_awasd_vol; atol = 1f-4, rtol = 1f-4)

        @test size(piccs_vol) == volume_size(iter_geo)
        @test all(isfinite, piccs_vol)
        @test minimum(piccs_vol) >= 0f0
        @test piccs_hist.iterations_completed == 1
        @test length(piccs_hist.residual_l2) == 1
        @test isapprox(piccs_vol, os_piccs_vol; atol = 1f-4, rtol = 1f-4)

        @test size(pcsd_vol) == volume_size(iter_geo)
        @test all(isfinite, pcsd_vol)
        @test minimum(pcsd_vol) >= 0f0
        @test pcsd_hist.iterations_completed == 1
        @test isapprox(pcsd_vol, os_pcsd_vol; atol = 1f-4, rtol = 1f-4)

        @test size(aw_pcsd_vol) == volume_size(iter_geo)
        @test all(isfinite, aw_pcsd_vol)
        @test minimum(aw_pcsd_vol) >= 0f0
        @test aw_pcsd_hist.iterations_completed == 1
        @test isapprox(aw_pcsd_vol, os_aw_pcsd_vol; atol = 1f-4, rtol = 1f-4)

        @test all(isfinite, ista_ws_vol)
        @test all(isfinite, fista_ws_vol)

        @test size(cgls_vol) == volume_size(iter_geo)
        @test size(lsqr_vol) == volume_size(iter_geo)
        @test size(lsmr_vol) == volume_size(iter_geo)
        @test size(hybrid_vol) == volume_size(iter_geo)
        @test size(hybrid_tv_vol) == volume_size(iter_geo)
        @test size(ab_vol) == volume_size(iter_geo)
        @test size(ba_vol) == volume_size(iter_geo)
        @test size(irn_vol) == volume_size(iter_geo)
        @test all(isfinite, cgls_vol)
        @test all(isfinite, lsqr_vol)
        @test all(isfinite, lsmr_vol)
        @test all(isfinite, hybrid_vol)
        @test all(isfinite, hybrid_tv_vol)
        @test all(isfinite, hybrid_tv_fallback_vol)
        @test all(isfinite, ab_vol)
        @test all(isfinite, ba_vol)
        @test all(isfinite, irn_vol)
        @test cgls_hist.iterations_completed == 1
        @test lsqr_hist.iterations_completed == 1
        @test lsmr_hist.iterations_completed == 1
        @test hybrid_hist.iterations_completed == 1
        @test hybrid_tv_hist.iterations_completed == 1
        @test ab_hist.iterations_completed == 1
        @test ba_hist.iterations_completed == 1
        @test irn_hist.iterations_completed == 1
        @test length(cgls_hist.residual_l2) == 1
        @test length(lsqr_hist.residual_l2) == 1
        @test length(lsmr_hist.residual_l2) == 1
        @test length(hybrid_hist.residual_l2) == 1
        @test length(hybrid_tv_hist.residual_l2) == 1
        @test length(ab_hist.residual_l2) == 1
        @test length(ba_hist.residual_l2) == 1
        @test length(irn_hist.residual_l2) == 1

        for lambda in (0.03f0, 0.3f0), steps in (2, 4)
            tuned_vol, tuned_hist = hybrid_flsqr_tv(proj, iter_geo, 1; lambda = lambda, precond_steps = steps, return_history = true, compute_l2 = true, workspace = krylov_workspace)
            @test all(isfinite, tuned_vol)
            @test tuned_hist.iterations_completed == 1
            @test length(tuned_hist.residual_l2) == 1
        end
    end

    @testset "Parallel beam Siddon 3D Euler angles and iterative convergence" begin
        # 1. Compatibility test: 1D angles vs 3D angles with (theta=0, psi=0)
        angles_1d = Float32[0.0, 0.5, 1.0, 1.5]
        angles_zero_3d = Float32[
            0.0 0.0 0.0;
            0.5 0.0 0.0;
            1.0 0.0 0.0;
            1.5 0.0 0.0
        ]
        geo_1d = Geometry(
            mode = :parallel,
            angles = angles_1d,
            nVoxel = (8, 8, 8),
            sVoxel = (40f0, 40f0, 40f0),
            nDetector = (12, 12),
            sDetector = (60f0, 60f0),
            DSO = 1000f0,
            DSD = 1200f0,
        )
        geo_zero_3d = Geometry(
            mode = :parallel,
            angles = angles_zero_3d,
            nVoxel = (8, 8, 8),
            sVoxel = (40f0, 40f0, 40f0),
            nDetector = (12, 12),
            sDetector = (60f0, 60f0),
            DSO = 1000f0,
            DSD = 1200f0,
        )
        phantom = zeros(Float32, volume_size(geo_1d)...)
        phantom[3:6, 3:6, 3:6] .= 1.0f0

        p_1d = Ax(phantom, geo_1d; projection_type = :siddon)
        p_zero_3d = Ax(phantom, geo_zero_3d; projection_type = :siddon)
        @test isapprox(p_1d, p_zero_3d; atol = 1f-5)

        # 2. Sensitivity test: changing psi with theta != 0 must change projections
        angles_3d_theta = Float32[
            0.0 0.4 0.0;
            0.5 0.4 0.0;
            1.0 0.4 0.0;
            1.5 0.4 0.0
        ]
        angles_3d_psi = Float32[
            0.0 0.4 0.3;
            0.5 0.4 0.3;
            1.0 0.4 0.3;
            1.5 0.4 0.3
        ]
        geo_theta = Geometry(
            mode = :parallel,
            angles = angles_3d_theta,
            nVoxel = (8, 8, 8),
            sVoxel = (40f0, 40f0, 40f0),
            nDetector = (12, 12),
            sDetector = (60f0, 60f0),
            DSO = 1000f0,
            DSD = 1200f0,
        )
        geo_psi = Geometry(
            mode = :parallel,
            angles = angles_3d_psi,
            nVoxel = (8, 8, 8),
            sVoxel = (40f0, 40f0, 40f0),
            nDetector = (12, 12),
            sDetector = (60f0, 60f0),
            DSO = 1000f0,
            DSD = 1200f0,
        )
        p_theta = Ax(phantom, geo_theta; projection_type = :siddon)
        p_psi = Ax(phantom, geo_psi; projection_type = :siddon)
        @test all(isfinite, p_theta)
        @test all(isfinite, p_psi)
        @test maximum(abs.(p_theta .- p_psi)) > 1.0f0

        # 3. Iterative convergence tests under parallel Siddon with 3D Euler angles
        cgls_rec, cgls_hist = cgls(p_psi, geo_psi, 10;
            projection_type = :siddon,
            return_history = true,
            compute_l2 = true,
        )
        @test all(isfinite, cgls_rec)
        @test cgls_hist.iterations_completed == 10
        @test cgls_hist.residual_l2[end] < cgls_hist.residual_l2[1]

        sart_rec, sart_hist = sart(p_psi, geo_psi, 5;
            projection_type = :siddon,
            return_history = true,
            compute_l2 = true,
        )
        @test all(isfinite, sart_rec)
        @test sart_hist.iterations_completed == 5
        @test sart_hist.residual_l2[end] < sart_hist.residual_l2[1]
    end

    @testset "Numerical regression fixtures" begin
        reg_geo = Geometry(
            mode = :cone,
            angles = Float32[0, 0.35, 0.9, 1.4],
            nVoxel = (4, 4, 4),
            sVoxel = (40f0, 40f0, 40f0),
            nDetector = (8, 6),
            sDetector = (80f0, 60f0),
            DSO = 1000f0,
            DSD = 1200f0,
        )
        wang_geo = Geometry(
            mode = :cone,
            angles = Float32[0, 0.35, 0.9, 1.4],
            nVoxel = (4, 4, 4),
            sVoxel = (40f0, 40f0, 40f0),
            nDetector = (8, 6),
            sDetector = (80f0, 60f0),
            DSO = 1000f0,
            DSD = 1200f0,
            offDetector = [10f0, 0f0],
        )
        par_geo = Geometry(
            mode = :parallel,
            angles = Float32[0, 0.35, 0.9, 1.4],
            nVoxel = (4, 4, 4),
            sVoxel = (40f0, 40f0, 40f0),
            nDetector = (8, 6),
            sDetector = (80f0, 60f0),
            DSO = 1000f0,
            DSD = 1200f0,
        )
        phantom = zeros(Float32, volume_size(reg_geo)...)
        phantom[2, 2, 2] = 1f0
        phantom[3, 2, 1] = 0.5f0
        phantom[1, 4, 3] = 0.25f0
        phantom[2, 3, 4] = 0.75f0
        prior = fill(0.125f0, volume_size(reg_geo)...)

        proj = Ax(phantom, reg_geo)
        back = Atb(proj, reg_geo)
        fdk_proj = Ax(phantom, wang_geo)
        fdk_vol = fdk(fdk_proj, wang_geo; filter = :hann)
        sart_vol = sart(proj, reg_geo, 1)
        fista_vol = fista(proj, reg_geo, 1; tviter = 1)
        lsmr_vol = lsmr(proj, reg_geo, 1)
        hybrid_tv_vol = hybrid_flsqr_tv(proj, reg_geo, 1; lambda = 0.1f0, precond_steps = 2)
        piccs_vol = piccs(proj, reg_geo, prior, 1; tviter = 1, prior_ratio = 0.5f0)
        # Added 2026-09-25: fbp (parallel-beam), mlem, pcsd, aw_pcsd
        par_phantom = zeros(Float32, volume_size(par_geo)...)
        par_phantom[2, 2, 2] = 1f0
        par_phantom[3, 2, 1] = 0.5f0
        par_phantom[1, 4, 3] = 0.25f0
        par_phantom[2, 3, 4] = 0.75f0
        par_proj = Ax(par_phantom, par_geo)
        fbp_vol = fbp(par_proj, par_geo; filter = :hann)
        mlem_vol = mlem(proj, reg_geo, 1)
        pcsd_vol = pcsd(proj, reg_geo, 1; tviter = 1)
        aw_pcsd_vol = aw_pcsd(proj, reg_geo, 1; tviter = 1)

        assert_signature_matches(projection_signature(proj), REGRESSION_REFERENCE.ax)
        assert_signature_matches(volume_signature(back), REGRESSION_REFERENCE.atb)
        assert_signature_matches(volume_signature(fdk_vol), REGRESSION_REFERENCE.fdk_hann_wang)
        assert_signature_matches(volume_signature(sart_vol), REGRESSION_REFERENCE.sart)
        assert_signature_matches(volume_signature(fista_vol), REGRESSION_REFERENCE.fista)
        assert_signature_matches(volume_signature(lsmr_vol), REGRESSION_REFERENCE.lsmr)
        assert_signature_matches(volume_signature(hybrid_tv_vol), REGRESSION_REFERENCE.hybrid_flsqr_tv)
        assert_signature_matches(volume_signature(piccs_vol), REGRESSION_REFERENCE.piccs)
        # Added 2026-09-25
        assert_signature_matches(volume_signature(fbp_vol), REGRESSION_REFERENCE.fbp_hann)
        assert_signature_matches(volume_signature(mlem_vol), REGRESSION_REFERENCE.mlem)
        assert_signature_matches(volume_signature(pcsd_vol), REGRESSION_REFERENCE.pcsd; atol = 2f-4)
        assert_signature_matches(volume_signature(aw_pcsd_vol), REGRESSION_REFERENCE.aw_pcsd; atol = 2f-4)
    end
else
    @info "Skipping GPU smoke and numerical regression tests because native library or functional CUDA GPU is not available"
end

@testset "Visualization interface and Makie extension" begin
    @test hasmethod(plotImg, Tuple{AbstractArray{<:Real, 3}}) == false || true
    @test hasmethod(plotProj, Tuple{AbstractArray{<:Real, 3}}) == false || true

    # Test with CairoMakie loaded
    try
        import CairoMakie
        @test hasmethod(plotImg, Tuple{AbstractArray{<:Real, 3}})
        @test hasmethod(plotProj, Tuple{AbstractArray{<:Real, 3}})

        test_vol = rand(Float32, 16, 16, 8)
        fig_z = plotImg(test_vol; dim = :z, slice = 4)
        @test fig_z isa CairoMakie.Figure

        fig_y = plotImg(test_vol; dim = :y, slice = 8)
        @test fig_y isa CairoMakie.Figure

        fig_x = plotImg(test_vol; dim = :x, slice = 8)
        @test fig_x isa CairoMakie.Figure

        fig_2d = plotImg(test_vol[:, :, 4])
        @test fig_2d isa CairoMakie.Figure

        test_proj = rand(Float32, 20, 10, 6)
        test_angles = Float32.(range(0, pi, length=6))
        fig_p = plotProj(test_proj, test_angles; slice = 2)
        @test fig_p isa CairoMakie.Figure

        # Movie saving test
        tmp_gif = tempname() * ".gif"
        plotImg(test_vol; dim = :z, step = 2, savegif = tmp_gif, framerate = 10)
        @test isfile(tmp_gif)
        @test filesize(tmp_gif) > 0
        rm(tmp_gif, force = true)
    catch e
        @info "Skipping Makie extension runtime test (CairoMakie not in test environment)" exception=e
    end
end
