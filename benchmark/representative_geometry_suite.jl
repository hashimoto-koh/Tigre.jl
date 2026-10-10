using Tigre

include("reuse_bench_utils.jl")
include("native_preflight.jl")

function _int_env(name::AbstractString, default::Int)
    value = get(ENV, name, nothing)
    value === nothing && return default
    return parse(Int, value)
end

function _bool_env(name::AbstractString, default::Bool = false)
    value = get(ENV, name, nothing)
    value === nothing && return default
    return lowercase(strip(value)) in ("1", "true", "yes", "on")
end

function _float_env(name::AbstractString)
    value = get(ENV, name, nothing)
    value === nothing && return nothing
    return Float32(parse(Float64, value))
end

function _geometry_preset(name::AbstractString)
    if name == "moderate"
        return Geometry(
            mode = :cone,
            angles = collect(LinRange(0f0, 2f0 * Float32(pi), 96)),
            nVoxel = (128, 128, 96),
            sVoxel = (220f0, 220f0, 180f0),
            nDetector = (256, 192),
            sDetector = (420f0, 320f0),
            DSO = 740f0,
            DSD = 1200f0,
        )
    elseif name == "dense"
        return Geometry(
            mode = :cone,
            angles = collect(LinRange(0f0, 2f0 * Float32(pi), 144)),
            nVoxel = (160, 160, 128),
            sVoxel = (240f0, 240f0, 220f0),
            nDetector = (320, 256),
            sDetector = (430f0, 340f0),
            DSO = 740f0,
            DSD = 1200f0,
        )
    else
        error("unknown REPRESENTATIVE_GEOMETRY_PRESET: $name")
    end
end

function _stack_rows(rows::AbstractVector{Float32}...)
    out = Matrix{Float32}(undef, length(rows), length(rows[1]))
    for (i, row) in enumerate(rows)
        @views out[i, :] .= row
    end
    return out
end

function _with_optional_wang(geo::Geometry, enabled::Bool, offset_u_override)
    enabled || return geo
    offset_u = offset_u_override === nothing ? geo.dDetector[1] : Float32(offset_u_override)
    nangles = projection_count(geo)
    return Geometry(
        mode = geo.mode,
        angles = geo.angles,
        nVoxel = geo.nVoxel,
        sVoxel = geo.sVoxel,
        nDetector = geo.nDetector,
        sDetector = geo.sDetector,
        DSO = geo.DSO,
        DSD = geo.DSD,
        offOrigin = _stack_rows(geo.offOriginX, geo.offOriginY, geo.offOriginZ),
        offDetector = _stack_rows(fill(offset_u, nangles), geo.offDetectorV),
        rotDetector = _stack_rows(geo.rotDetectorRoll, geo.rotDetectorPitch, geo.rotDetectorYaw),
        COR = geo.COR,
        accuracy = geo.accuracy,
    )
end

function _make_phantom(geo::Geometry)
    vol = zeros(Float32, volume_size(geo)...)
    nx, ny, nz = size(vol)
    vol[(nx ÷ 2 - 8):(nx ÷ 2 + 8), (ny ÷ 2 - 8):(ny ÷ 2 + 8), (nz ÷ 2 - 6):(nz ÷ 2 + 6)] .= 1f0
    vol[(nx ÷ 3 - 4):(nx ÷ 3 + 4), (2ny ÷ 3 - 6):(2ny ÷ 3 + 6), (nz ÷ 3 - 4):(nz ÷ 3 + 4)] .= 0.5f0
    vol[(3nx ÷ 4 - 3):(3nx ÷ 4 + 3), (ny ÷ 4 - 5):(ny ÷ 4 + 5), (2nz ÷ 3 - 4):(2nz ÷ 3 + 4)] .= 0.25f0
    return vol
end

function _measure_case(f, label::AbstractString, samples::Int)
    f()
    GC.gc()
    times = Vector{Float64}(undef, samples)
    allocs = Vector{Int}(undef, samples)
    for i in 1:samples
        GC.gc()
        allocs[i] = @allocated times[i] = @elapsed f()
    end
    return (
        label = String(label),
        samples = samples,
        time_median_s = median(times),
        time_mean_s = mean(times),
        time_std_s = length(times) > 1 ? std(times) : 0.0,
        alloc_median_B = median(allocs),
        alloc_mean_B = mean(allocs),
    )
end

function _summary_header()
    return (
        "label",
        "samples",
        "time_median_s",
        "time_mean_s",
        "time_std_s",
        "alloc_median_B",
        "alloc_mean_B",
    )
end

function _summary_row(result)
    return (
        result.label,
        string(result.samples),
        string(result.time_median_s),
        string(result.time_mean_s),
        string(result.time_std_s),
        string(result.alloc_median_B),
        string(result.alloc_mean_B),
    )
end

function print_summary(results)
    println()
    println("Summary")
    println(join(_summary_header(), " | "))
    for result in results
        println(join(_summary_row(result), " | "))
    end
    return nothing
end

const PRESET = get(ENV, "REPRESENTATIVE_GEOMETRY_PRESET", "moderate")
const SAMPLES = _int_env("REPRESENTATIVE_GEOMETRY_SAMPLES", 5)
const FDK_FILTER = Symbol(get(ENV, "REPRESENTATIVE_FDK_FILTER", "hann"))
const ENABLE_WANG = _bool_env("REPRESENTATIVE_ENABLE_WANG", false)
const WANG_OFFSET_U = _float_env("REPRESENTATIVE_WANG_OFFSET_U")
const SART_ITERS = _int_env("REPRESENTATIVE_SART_ITERS", 1)
const OS_ASD_ITERS = _int_env("REPRESENTATIVE_OS_ASD_ITERS", 2)
const FISTA_ITERS = _int_env("REPRESENTATIVE_FISTA_ITERS", 5)
const LSMR_ITERS = _int_env("REPRESENTATIVE_LSMR_ITERS", 5)
const BLOCKSIZE = _int_env("REPRESENTATIVE_BLOCKSIZE", 8)
const TV_ITERS = _int_env("REPRESENTATIVE_TV_ITERS", 1)

geo = _with_optional_wang(_geometry_preset(PRESET), ENABLE_WANG, WANG_OFFSET_U)
ensure_native_benchmark_ready("representative_geometry_suite benchmark")

phantom = _make_phantom(geo)
projection = Ax(phantom, geo)

art_plan_sart = prepare_art_plan(projection, geo; blocksize = 1, need_full_projection = false)
art_plan_os_asd = prepare_art_plan(projection, geo; blocksize = BLOCKSIZE, need_full_projection = true, extra_volume_buffers = 3)
first_order_workspace = prepare_first_order_workspace(geo)
krylov_workspace = prepare_krylov_workspace(geo; maxiter = LSMR_ITERS)

fdk_volume = zeros(Float32, volume_size(geo)...)
sart_volume = zeros(Float32, volume_size(geo)...)
os_asd_volume = zeros(Float32, volume_size(geo)...)
fista_volume = zeros(Float32, volume_size(geo)...)
lsmr_volume = zeros(Float32, volume_size(geo)...)

geo_label = ENABLE_WANG ? string(PRESET, ", Wang") : PRESET

results = []

push!(results, _measure_case("FDK ($geo_label, filter=$FDK_FILTER)", SAMPLES) do
    fill!(fdk_volume, 0f0)
    fdk!(fdk_volume, projection, geo; filter = FDK_FILTER)
end)

push!(results, _measure_case("SART raw ($geo_label, niter=$SART_ITERS)", SAMPLES) do
    fill!(sart_volume, 0f0)
    sart!(sart_volume, projection, geo, SART_ITERS)
end)

push!(results, _measure_case("SART prepared ($geo_label, niter=$SART_ITERS)", SAMPLES) do
    fill!(sart_volume, 0f0)
    sart!(sart_volume, projection, geo, SART_ITERS; plan = art_plan_sart)
end)

push!(results, _measure_case("OS-ASD-POCS raw ($geo_label, niter=$OS_ASD_ITERS, blocksize=$BLOCKSIZE)", SAMPLES) do
    fill!(os_asd_volume, 0f0)
    os_asd_pocs!(os_asd_volume, projection, geo, OS_ASD_ITERS; blocksize = BLOCKSIZE, tviter = TV_ITERS)
end)

push!(results, _measure_case("OS-ASD-POCS prepared ($geo_label, niter=$OS_ASD_ITERS, blocksize=$BLOCKSIZE)", SAMPLES) do
    fill!(os_asd_volume, 0f0)
    os_asd_pocs!(os_asd_volume, projection, geo, OS_ASD_ITERS; blocksize = BLOCKSIZE, tviter = TV_ITERS, plan = art_plan_os_asd)
end)

push!(results, _measure_case("FISTA raw ($geo_label, niter=$FISTA_ITERS)", SAMPLES) do
    fill!(fista_volume, 0f0)
    fista!(fista_volume, projection, geo, FISTA_ITERS; tviter = TV_ITERS)
end)

push!(results, _measure_case("FISTA prepared ($geo_label, niter=$FISTA_ITERS)", SAMPLES) do
    fill!(fista_volume, 0f0)
    fista!(fista_volume, projection, geo, FISTA_ITERS; tviter = TV_ITERS, workspace = first_order_workspace)
end)

push!(results, _measure_case("LSMR raw ($geo_label, niter=$LSMR_ITERS)", SAMPLES) do
    fill!(lsmr_volume, 0f0)
    lsmr!(lsmr_volume, projection, geo, LSMR_ITERS)
end)

push!(results, _measure_case("LSMR prepared ($geo_label, niter=$LSMR_ITERS)", SAMPLES) do
    fill!(lsmr_volume, 0f0)
    lsmr!(lsmr_volume, projection, geo, LSMR_ITERS; workspace = krylov_workspace)
end)

println("preset = ", PRESET)
println("wang = ", ENABLE_WANG, ", wang_offset_u = ", ENABLE_WANG ? geo.offDetectorU[1] : 0f0)
println("angles = ", size(geo.angles, 2))
println("volume size = ", volume_size(geo))
println("projection size = ", projection_size(geo))
println("samples = ", SAMPLES)

for result in results
    println(result.label)
    println("  median/mean/std: ", result.time_median_s, " / ", result.time_mean_s, " / ", result.time_std_s, " s")
    println("  alloc median/mean: ", result.alloc_median_B, " / ", result.alloc_mean_B, " bytes")
end

print_summary(results)
maybe_write_table(_summary_header(), (_summary_row(result) for result in results);
    argv = ARGS,
    env_tsv = "REPRESENTATIVE_GEOMETRY_TSV",
    env_csv = "REPRESENTATIVE_GEOMETRY_CSV",
    label = "representative geometry summary",
)
