using LinearAlgebra
using Tigre

include("native_preflight.jl")

function run_iterative_demo(io::IO = stdout; preflight::Bool = true)
    preflight && ensure_native_example_ready("iterative_demo")

    geo = Geometry(
        mode = :cone,
        angles = Float32[0, 0.4, 0.9, 1.3],
        nVoxel = (32, 32, 32),
        sVoxel = (160f0, 160f0, 160f0),
        nDetector = (48, 32),
        sDetector = (240f0, 160f0),
        DSO = 1000f0,
        DSD = 1200f0,
    )

    phantom = zeros(Float32, volume_size(geo)...)
    phantom[14:18, 14:18, 14:18] .= 1f0
    phantom[8:10, 20:22, 12:14] .= 0.4f0

    projection = Ax(phantom, geo)
    sart_vol, sart_hist = sart(projection, geo, 1; return_history = true, compute_l2 = true)
    fista_vol, fista_hist = fista(projection, geo, 1; tviter = 1, return_history = true, compute_l2 = true)
    lsmr_vol, lsmr_hist = lsmr(projection, geo, 1; return_history = true, compute_l2 = true)

    result = (
        sart_residual_l2 = only(sart_hist.residual_l2),
        fista_residual_l2 = only(fista_hist.residual_l2),
        lsmr_residual_l2 = only(lsmr_hist.residual_l2),
        sart_sum = Float32(sum(sart_vol)),
        sart_max = Float32(maximum(sart_vol)),
        fista_sum = Float32(sum(fista_vol)),
        fista_max = Float32(maximum(fista_vol)),
        lsmr_sum = Float32(sum(lsmr_vol)),
        lsmr_max = Float32(maximum(lsmr_vol)),
    )

    println(io, "Iterative demo")
    println(io, "  SART residual L2: ", result.sart_residual_l2)
    println(io, "  FISTA residual L2: ", result.fista_residual_l2)
    println(io, "  LSMR residual L2: ", result.lsmr_residual_l2)
    println(io, "  SART sum/max: ", result.sart_sum, " / ", result.sart_max)
    println(io, "  FISTA sum/max: ", result.fista_sum, " / ", result.fista_max)
    println(io, "  LSMR sum/max: ", result.lsmr_sum, " / ", result.lsmr_max)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_iterative_demo()
end
