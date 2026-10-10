using Tigre
using Statistics

include("native_preflight.jl")

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

ensure_native_benchmark_ready("vertical_slice benchmark")

volume = rand(Float32, volume_size(geo)...)
projection = Array{Float32}(undef, projection_size(geo))
recon = Array{Float32}(undef, volume_size(geo))
fdk_volume = Array{Float32}(undef, volume_size(geo))
iter_volume = zeros(Float32, volume_size(geo)...)
ista_volume = zeros(Float32, volume_size(geo)...)
fista_volume = zeros(Float32, volume_size(geo)...)
sart_tv_volume = zeros(Float32, volume_size(geo)...)
os_asd_volume = zeros(Float32, volume_size(geo)...)
awasd_volume = zeros(Float32, volume_size(geo)...)
piccs_volume = zeros(Float32, volume_size(geo)...)
os_piccs_volume = zeros(Float32, volume_size(geo)...)
cgls_volume = zeros(Float32, volume_size(geo)...)
lsmr_volume = zeros(Float32, volume_size(geo)...)
hybrid_volume = zeros(Float32, volume_size(geo)...)
hybrid_tv_volume = zeros(Float32, volume_size(geo)...)
ab_volume = zeros(Float32, volume_size(geo)...)
ba_volume = zeros(Float32, volume_size(geo)...)
irn_volume = zeros(Float32, volume_size(geo)...)
prior_volume = copy(volume)

Ax!(projection, volume, geo)
art_plan = prepare_art_plan(projection, geo; blocksize = 4, need_full_projection = true, extra_volume_buffers = 3)
first_order_workspace = prepare_first_order_workspace(geo; need_full_projection = false)
krylov_workspace = prepare_krylov_workspace(geo; maxiter = 1, extra_volume_buffers = 8)
Atb!(recon, projection, geo)
fdk!(fdk_volume, projection, geo)
sart!(iter_volume, projection, geo, 1)
ista!(ista_volume, projection, geo, 1; tviter = 1)
fista!(fista_volume, projection, geo, 1; tviter = 1)
sart_tv!(sart_tv_volume, projection, geo, 1; tviter = 1, tvlambda = 1f0)
os_asd_pocs!(os_asd_volume, projection, geo, 1; blocksize = 4, tviter = 1)
awasd_pocs!(awasd_volume, projection, geo, 1; tviter = 1)
piccs!(piccs_volume, projection, geo, prior_volume, 1; tviter = 1, prior_ratio = 0.5f0)
os_piccs!(os_piccs_volume, projection, geo, prior_volume, 1; blocksize = 4, tviter = 1, prior_ratio = 0.5f0)
cgls!(cgls_volume, projection, geo, 1)
lsmr!(lsmr_volume, projection, geo, 1)
hybrid_lsqr!(hybrid_volume, projection, geo, 1; lambda = 0.1f0)
hybrid_flsqr_tv!(hybrid_tv_volume, projection, geo, 1; lambda = 0.1f0)
ab_gmres!(ab_volume, projection, geo, 1)
ba_gmres!(ba_volume, projection, geo, 1)
irn_tv_cgls!(irn_volume, projection, geo, 1; niter_outer = 1, lambda = 0.1f0)

function elapsed_stats(f; samples = 5)
    f()
    GC.gc()
    times = Vector{Float64}(undef, samples)
    for i in 1:samples
        GC.gc()
        times[i] = @elapsed f()
    end
    return (
        median = median(times),
        mean = mean(times),
        std = std(times),
        min = minimum(times),
        max = maximum(times),
    )
end

ax_time = elapsed_stats(() -> Ax!(projection, volume, geo))
atb_time = elapsed_stats(() -> Atb!(recon, projection, geo))
fdk_time = elapsed_stats(() -> fdk!(fdk_volume, projection, geo))
sart_time = elapsed_stats(() -> sart!(iter_volume, projection, geo, 1))
ista_time = elapsed_stats(() -> ista!(ista_volume, projection, geo, 1; tviter = 1))
fista_time = elapsed_stats(() -> fista!(fista_volume, projection, geo, 1; tviter = 1))
sart_tv_time = elapsed_stats(() -> sart_tv!(sart_tv_volume, projection, geo, 1; tviter = 1, tvlambda = 1f0))
os_asd_time = elapsed_stats(() -> os_asd_pocs!(os_asd_volume, projection, geo, 1; blocksize = 4, tviter = 1))
prepared_os_asd_time = elapsed_stats(() -> os_asd_pocs!(os_asd_volume, projection, geo, 1; blocksize = 4, tviter = 1, plan = art_plan))
awasd_time = elapsed_stats(() -> awasd_pocs!(awasd_volume, projection, geo, 1; tviter = 1))
piccs_time = elapsed_stats(() -> piccs!(piccs_volume, projection, geo, prior_volume, 1; tviter = 1, prior_ratio = 0.5f0))
os_piccs_time = elapsed_stats(() -> os_piccs!(os_piccs_volume, projection, geo, prior_volume, 1; blocksize = 4, tviter = 1, prior_ratio = 0.5f0))
cgls_time = elapsed_stats(() -> cgls!(cgls_volume, projection, geo, 1))
prepared_ista_time = elapsed_stats(() -> ista!(ista_volume, projection, geo, 1; tviter = 1, workspace = first_order_workspace))
lsmr_time = elapsed_stats(() -> lsmr!(lsmr_volume, projection, geo, 1))
prepared_lsmr_time = elapsed_stats(() -> lsmr!(lsmr_volume, projection, geo, 1; workspace = krylov_workspace))
hybrid_time = elapsed_stats(() -> hybrid_lsqr!(hybrid_volume, projection, geo, 1; lambda = 0.1f0))
hybrid_tv_time = elapsed_stats(() -> hybrid_flsqr_tv!(hybrid_tv_volume, projection, geo, 1; lambda = 0.1f0))
ab_time = elapsed_stats(() -> ab_gmres!(ab_volume, projection, geo, 1))
ba_time = elapsed_stats(() -> ba_gmres!(ba_volume, projection, geo, 1))
irn_time = elapsed_stats(() -> irn_tv_cgls!(irn_volume, projection, geo, 1; niter_outer = 1, lambda = 0.1f0))

function print_stats(label, stats)
    println(label, " median: ", stats.median, " s, mean: ", stats.mean, " s, std: ", stats.std, " s, min/max: ", stats.min, " / ", stats.max, " s")
end

print_stats("Ax!", ax_time)
print_stats("Atb!", atb_time)
print_stats("FDK!", fdk_time)
print_stats("SART!", sart_time)
print_stats("ISTA!", ista_time)
print_stats("FISTA!", fista_time)
print_stats("SART-TV!", sart_tv_time)
print_stats("OS-ASD-POCS!", os_asd_time)
print_stats("Prepared OS-ASD-POCS!", prepared_os_asd_time)
print_stats("AwASD-POCS!", awasd_time)
print_stats("PICCS!", piccs_time)
print_stats("OS-PICCS!", os_piccs_time)
print_stats("CGLS!", cgls_time)
print_stats("Prepared ISTA!", prepared_ista_time)
print_stats("LSMR!", lsmr_time)
print_stats("Prepared LSMR!", prepared_lsmr_time)
print_stats("Hybrid LSQR!", hybrid_time)
print_stats("Hybrid fLSQR-TV!", hybrid_tv_time)
print_stats("AB-GMRES!", ab_time)
print_stats("BA-GMRES!", ba_time)
print_stats("IRN-TV-CGLS!", irn_time)
