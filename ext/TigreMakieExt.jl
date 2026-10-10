module TigreMakieExt

using Tigre
import Makie
using Makie: Figure, Axis, DataAspect, heatmap!, Colorbar, Observable, record, lift
using Printf: @sprintf

function _parse_slice_dim(dim::Union{Symbol, AbstractString, Integer})
    if dim isa Integer
        dim in (1, 2, 3) || throw(ArgumentError("Dimension index must be 1, 2, or 3, got $dim"))
        return Int(dim)
    end
    s = lowercase(string(dim))
    if s in ("1", "x")
        return 1
    elseif s in ("2", "y")
        return 2
    elseif s in ("3", "z")
        return 3
    else
        throw(ArgumentError("Invalid slice dimension: $dim. Expected :x, :y, :z, 1, 2, or 3"))
    end
end

function _compute_clims(data::AbstractArray{<:Real}, clims)
    if clims !== nothing
        return (Float32(clims[1]), Float32(clims[2]))
    end
    v_min, v_max = extrema(data)
    if v_min == v_max
        return (Float32(v_min - 1), Float32(v_max + 1))
    end
    return (Float32(v_min), Float32(v_max))
end

function _extract_2d_slice(volume::AbstractArray{<:Real, 3}, dim::Int, idx::Int)
    if dim == 3
        # Axial: (Nx, Ny) -> X horizontal, Y vertical
        return volume[:, :, idx]
    elseif dim == 2
        # Coronal: (Nx, Nz) -> X horizontal, Z vertical
        # In Makie heatmap(M), columns are Y-axis (Z here), rows are X-axis
        return volume[:, idx, :]
    else
        # Sagittal: (Ny, Nz) -> Y horizontal, Z vertical
        return volume[idx, :, :]
    end
end

function _dim_axis_labels(dim::Int)
    if dim == 3
        return ("X (voxel)", "Y (voxel)", "Z (Axial)")
    elseif dim == 2
        return ("X (voxel)", "Z (voxel)", "Y (Coronal)")
    else
        return ("Y (voxel)", "Z (voxel)", "X (Sagittal)")
    end
end

"""
    plotImg(img::AbstractArray{<:Real, 3}; ...)
    plotImg(img::AbstractArray{<:Real, 2}; ...)

Makie implementation of `plotImg`. Supports 2D slices, 3D interactive/static slices,
and saving animated slice movies (.gif, .mp4).
"""
function Tigre.plotImg(img::AbstractArray{<:Real, 3};
                       dim = :z,
                       step::Int = 1,
                       slice::Union{Int, Nothing} = nothing,
                       clims = nothing,
                       colormap = :grays,
                       savegif::Union{String, Nothing} = nothing,
                       framerate::Int = 15,
                       figure_size::Tuple{Int, Int} = (600, 500))

    step >= 1 || throw(ArgumentError("step must be >= 1, got $step"))
    dim_idx = _parse_slice_dim(dim)
    n_slices = size(img, dim_idx)
    xlabel_str, ylabel_str, dim_name = _dim_axis_labels(dim_idx)
    color_range = _compute_clims(img, clims)

    if savegif !== nothing
        # Movie / animation mode
        slice_indices = 1:step:n_slices
        isempty(slice_indices) && throw(ArgumentError("No slices found with step $step"))

        current_idx = Observable(first(slice_indices))
        slice_obs = lift(idx -> _extract_2d_slice(img, dim_idx, idx), current_idx)
        title_obs = lift(idx -> "$dim_name slice: $idx / $n_slices", current_idx)

        fig = Figure(size = figure_size)
        ax = Axis(fig[1, 1], title = title_obs, xlabel = xlabel_str, ylabel = ylabel_str, aspect = DataAspect())
        hm = heatmap!(ax, slice_obs; colorrange = color_range, colormap = colormap)
        Colorbar(fig[1, 2], hm)

        record(fig, savegif, slice_indices; framerate = framerate) do idx
            current_idx[] = idx
        end
        return fig
    else
        # Static display mode
        idx = slice === nothing ? (n_slices ÷ 2 == 0 ? 1 : n_slices ÷ 2) : slice
        1 <= idx <= n_slices || throw(BoundsError(img, (dim_idx, idx)))

        slice_data = _extract_2d_slice(img, dim_idx, idx)
        fig = Figure(size = figure_size)
        ax = Axis(fig[1, 1], title = "$dim_name slice: $idx / $n_slices", xlabel = xlabel_str, ylabel = ylabel_str, aspect = DataAspect())
        hm = heatmap!(ax, slice_data; colorrange = color_range, colormap = colormap)
        Colorbar(fig[1, 2], hm)
        return fig
    end
end

function Tigre.plotImg(img::AbstractArray{<:Real, 2};
                       clims = nothing,
                       colormap = :grays,
                       figure_size::Tuple{Int, Int} = (600, 500),
                       title::String = "2D Image")

    color_range = _compute_clims(img, clims)
    fig = Figure(size = figure_size)
    ax = Axis(fig[1, 1], title = title, xlabel = "X", ylabel = "Y", aspect = DataAspect())
    hm = heatmap!(ax, img; colorrange = color_range, colormap = colormap)
    Colorbar(fig[1, 2], hm)
    return fig
end

"""
    plotProj(proj::AbstractArray{<:Real, 3}, [angles]; ...)

Makie implementation of `plotProj`. Visualizes projection radiographs,
supports single angle display or animated rotation movie saving.
"""
function Tigre.plotProj(proj::AbstractArray{<:Real, 3},
                        angles::Union{AbstractVector{<:Real}, Nothing} = nothing;
                        step::Int = 1,
                        slice::Union{Int, Nothing} = nothing,
                        clims = nothing,
                        colormap = :viridis,
                        savegif::Union{String, Nothing} = nothing,
                        framerate::Int = 15,
                        figure_size::Tuple{Int, Int} = (650, 500))

    step >= 1 || throw(ArgumentError("step must be >= 1, got $step"))
    n_angles = size(proj, 3)
    color_range = _compute_clims(proj, clims)

    _angle_str(idx) = angles === nothing ? "Angle index: $idx / $n_angles" : @sprintf("Projection %d / %d (%.1f°)", idx, n_angles, rad2deg(Float64(angles[idx])))

    if savegif !== nothing
        # Animated projection movie
        angle_indices = 1:step:n_angles
        isempty(angle_indices) && throw(ArgumentError("No projection angles with step $step"))

        current_idx = Observable(first(angle_indices))
        proj_obs = lift(idx -> proj[:, :, idx], current_idx)
        title_obs = lift(_angle_str, current_idx)

        fig = Figure(size = figure_size)
        ax = Axis(fig[1, 1], title = title_obs, xlabel = "Detector U (channels)", ylabel = "Detector V (rows)", aspect = DataAspect())
        hm = heatmap!(ax, proj_obs; colorrange = color_range, colormap = colormap)
        Colorbar(fig[1, 2], hm)

        record(fig, savegif, angle_indices; framerate = framerate) do idx
            current_idx[] = idx
        end
        return fig
    else
        # Single projection display
        idx = slice === nothing ? 1 : slice
        1 <= idx <= n_angles || throw(BoundsError(proj, (3, idx)))

        fig = Figure(size = figure_size)
        ax = Axis(fig[1, 1], title = _angle_str(idx), xlabel = "Detector U (channels)", ylabel = "Detector V (rows)", aspect = DataAspect())
        hm = heatmap!(ax, proj[:, :, idx]; colorrange = color_range, colormap = colormap)
        Colorbar(fig[1, 2], hm)
        return fig
    end
end

end # module TigreMakieExt
