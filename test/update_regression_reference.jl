using Dates
using LinearAlgebra
using Tigre

function signature3d(x::AbstractArray{Float32, 3})
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

function signature_projection(x::AbstractArray{Float32, 3})
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

fdk_geo = Geometry(
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

par_phantom = zeros(Float32, volume_size(par_geo)...)
par_phantom[2, 2, 2] = 1f0
par_phantom[3, 2, 1] = 0.5f0
par_phantom[1, 4, 3] = 0.25f0
par_phantom[2, 3, 4] = 0.75f0

proj = Ax(phantom, reg_geo)
back = Atb(proj, reg_geo)
fdk_proj = Ax(phantom, fdk_geo)
fdk_vol = fdk(fdk_proj, fdk_geo; filter = :hann)
sart_vol = sart(proj, reg_geo, 1)
fista_vol = fista(proj, reg_geo, 1; tviter = 1)
lsmr_vol = lsmr(proj, reg_geo, 1)
hybrid_tv_vol = hybrid_flsqr_tv(proj, reg_geo, 1; lambda = 0.1f0, precond_steps = 2)
piccs_vol = piccs(proj, reg_geo, prior, 1; tviter = 1, prior_ratio = 0.5f0)
par_proj = Ax(par_phantom, par_geo)
fbp_vol = fbp(par_proj, par_geo; filter = :hann)
mlem_vol = mlem(proj, reg_geo, 1)
pcsd_vol = pcsd(proj, reg_geo, 1; tviter = 1)
aw_pcsd_vol = aw_pcsd(proj, reg_geo, 1; tviter = 1)

println("const REGRESSION_REFERENCE = (")
println("    ax = ", repr(signature_projection(proj)), ",")
println("    atb = ", repr(signature3d(back)), ",")
println("    fdk_hann_wang = ", repr(signature3d(fdk_vol)), ",")
println("    sart = ", repr(signature3d(sart_vol)), ",")
println("    fista = ", repr(signature3d(fista_vol)), ",")
println("    lsmr = ", repr(signature3d(lsmr_vol)), ",")
println("    hybrid_flsqr_tv = ", repr(signature3d(hybrid_tv_vol)), ",")
println("    piccs = ", repr(signature3d(piccs_vol)), ",")
println("    # Added: $(Dates.today())")
println("    fbp_hann = ", repr(signature3d(fbp_vol)), ",")
println("    mlem = ", repr(signature3d(mlem_vol)), ",")
println("    pcsd = ", repr(signature3d(pcsd_vol)), ",")
println("    aw_pcsd = ", repr(signature3d(aw_pcsd_vol)), ",")
println(")")
