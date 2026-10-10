# Visualization interface stubs for Tigre.jl
# Real implementation is provided by TigreMakieExt when Makie is loaded.

"""
    plotImg(img::AbstractArray{<:Real, 3}; kwargs...)
    plotImg(img::AbstractArray{<:Real, 2}; kwargs...)

Visualize a 3D volumetric image or 2D slice, matching the upstream TIGRE `plotImg` interface.
This function requires a plotting backend such as `CairoMakie` or `GLMakie` to be loaded
(`using CairoMakie` or `using GLMakie`).

# Arguments
- `img`: 3D volume array of size `(Nx, Ny, Nz)` or 2D image of size `(Nx, Ny)`.

# Keyword Arguments
- `dim`: Slicing dimension/plane (`:z`, `:y`, `:x`, or `"Z"`, `"Y"`, `"X"`, default: `:z`).
  - `:z`: Axial planes (XY)
  - `:y`: Coronal planes (XZ)
  - `:x`: Sagittal planes (YZ)
- `step::Int`: Step size between successive slices when creating animation or displaying (default: `1`).
- `slice::Union{Int, Nothing}`: Index of a single slice to display. If provided, disables movie recording and displays this slice (default: `nothing`, displays central slice if `savegif=nothing`).
- `clims::Union{Tuple{<:Real, <:Real}, Nothing}`: Color limits `(cmin, cmax)`. Defaults to empirical `(1%, 99%)` percentiles of the data.
- `colormap`: Makie colormap name or symbol (default: `:grays`).
- `savegif::Union{String, Nothing}`: File path to save an animated movie (`.gif` or `.mp4`). If provided, iterates over all slices with step `step` (default: `nothing`).
- `framerate::Int`: Framerate in frames per second when saving movie (default: `15`).
- `figure_size::Tuple{Int, Int}`: Figure dimensions `(width, height)` in pixels (default: `(600, 500)`).

# Returns
- `Makie.Figure`: The generated Makie figure.

# Examples
```julia
using Tigre
using CairoMakie

vol = shepp_logan_3d(128)
# Display central axial slice
fig = plotImg(vol)

# Display specific coronal slice
fig = plotImg(vol; dim=:y, slice=64)

# Create and save animated slice movie
plotImg(vol; dim=:z, step=2, savegif="slices.gif", framerate=15)
```
"""
function plotImg(args...; kwargs...)
    error("`plotImg` requires a Makie backend. Please load CairoMakie or GLMakie (e.g., `using CairoMakie`) before calling `plotImg`.")
end

"""
    plotProj(proj::AbstractArray{<:Real, 3}; kwargs...)
    plotProj(proj::AbstractArray{<:Real, 3}, angles::AbstractVector{<:Real}; kwargs...)

Visualize 2D projection images over scan angles, matching the upstream TIGRE `plotProj` interface.
This function requires a plotting backend such as `CairoMakie` or `GLMakie` to be loaded
(`using CairoMakie` or `using GLMakie`).

# Arguments
- `proj`: 3D projection data array of size `(nDetU, nDetV, nAngles)`.
- `angles`: Optional vector of scan angles in radians or degrees (default: `1:nAngles`).

# Keyword Arguments
- `step::Int`: Step size between projection angles when creating animation (default: `1`).
- `slice::Union{Int, Nothing}`: Index of a single projection angle to display (default: `nothing`, displays 1st projection if `savegif=nothing`).
- `clims::Union{Tuple{<:Real, <:Real}, Nothing}`: Color limits `(cmin, cmax)`. Defaults to empirical `(1%, 99%)` percentiles of the data.
- `colormap`: Makie colormap name or symbol (default: `:viridis`).
- `savegif::Union{String, Nothing}`: File path to save an animated movie (`.gif` or `.mp4`) across all projection angles (default: `nothing`).
- `framerate::Int`: Framerate in frames per second when saving movie (default: `15`).
- `figure_size::Tuple{Int, Int}`: Figure dimensions `(width, height)` in pixels (default: `(600, 500)`).

# Returns
- `Makie.Figure`: The generated Makie figure.

# Examples
```julia
using Tigre
using CairoMakie

# Display first projection
fig = plotProj(projections)

# Save projection rotation movie
plotProj(projections; step=2, savegif="projections.gif", framerate=20)
```
"""
function plotProj(args...; kwargs...)
    error("`plotProj` requires a Makie backend. Please load CairoMakie or GLMakie (e.g., `using CairoMakie`) before calling `plotProj`.")
end
