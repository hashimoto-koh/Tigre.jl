# Visualization Guide

`Tigre.jl` includes high-performance visualization utilities matching the upstream TIGRE MATLAB/Python interface (`plotImg` and `plotProj`).

These functions are designed to streamline slice inspection, multi-planar reconstruction viewing, and animated slice movie creation with zero configuration.

---

## Package Extension Architecture

To keep `Tigre.jl` lightweight and avoid heavy mandatory graphics dependencies in headless compute environments, visualization functions are provided via **Julia Package Extensions** (`TigreMakieExt`).

To activate `plotImg` and `plotProj`, simply import your preferred Makie backend:

```julia
using Tigre
using CairoMakie  # or using GLMakie for interactive OpenGL windows
```

> [!NOTE]
> If you call `plotImg` or `plotProj` without loading a Makie backend, an informative error will guide you to load `CairoMakie` or `GLMakie` first.

---

## 1. Volumetric Image Visualization (`plotImg`)

`plotImg` visualizes 2D image slices or 3D volume cross-sections.

```julia
plotImg(img; kwargs...)
```

### Keyword Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dim` | `Symbol` / `String` | `:z` | Slicing dimension: `:z` (axial/XY), `:y` (coronal/XZ), `:x` (sagittal/YZ). |
| `slice` | `Int` / `nothing` | `nothing` | 1-based index of slice to display. Defaults to central slice (`size ÷ 2`). |
| `step` | `Int` | `1` | Step size between successive slices when generating animations. |
| `clims` | `Tuple` / `nothing` | `nothing` | Intensity display range `(min, max)`. Defaults to empirical `(minimum, maximum)` of data. |
| `colormap` | `Symbol` | `:grays` | Colormap symbol (e.g. `:grays`, `:viridis`, `:inferno`, `:plasma`). |
| `savegif` | `String` / `nothing` | `nothing` | Output file path (`.gif` or `.mp4`) to export an animated slice movie. |
| `framerate` | `Int` | `15` | Framerate in frames per second when exporting movie. |
| `figure_size` | `Tuple{Int, Int}` | `(600, 500)` | Output figure resolution `(width, height)` in pixels. |

### Examples

#### Inspecting Multi-Planar Orthogonal Slices

```julia
using Tigre
using CairoMakie

vol = shepp_logan_3d(256)

# Central axial slice (XY plane)
fig_axial = plotImg(vol; dim = :z)

# Coronal slice at Y = 128 (XZ plane)
fig_coronal = plotImg(vol; dim = :y, slice = 128)

# Sagittal slice at X = 128 (YZ plane) with custom contrast
fig_sagittal = plotImg(vol; dim = :x, slice = 128, clims = (0.0f0, 0.8f0), colormap = :inferno)
```

#### 2D Images and Matrices

`plotImg` also directly accepts 2D matrices:

```julia
img2d = vol[:, :, 128]
fig = plotImg(img2d; title = "2D Phantom Slice", clims = (0.0f0, 1.0f0))
```

---

## 2. Animated Slice Movies (`savegif`)

Creating slice-by-slice movies (through axial, coronal, or sagittal trajectories) is as simple as passing the `savegif` keyword argument.
Both `.gif` (animated GIF) and `.mp4` (H.264 video) formats are automatically supported:

```julia
# Export axial fly-through movie at 15 FPS
plotImg(vol; dim = :z, step = 1, savegif = "axial_scan.gif", framerate = 15)

# Export coronal video in MP4 format with step=2
plotImg(vol; dim = :y, step = 2, savegif = "coronal_flythrough.mp4", framerate = 20)
```

> [!TIP]
> Makie uses reactive `Observable` and in-place slice lifting during `record`, providing high throughput without memory bloat even for large volumes (e.g. $512^3$).

---

## 3. Projection Radiograph Visualization (`plotProj`)

`plotProj` displays planar x-ray projection radiographs acquired across gantry angles.

```julia
plotProj(projections, [angles]; kwargs...)
```

### Keyword Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `slice` | `Int` / `nothing` | `1` | 1-based gantry angle projection index to display. |
| `step` | `Int` | `1` | Angle step size when creating animations. |
| `clims` | `Tuple` / `nothing` | `nothing` | Detector intensity display range `(min, max)`. |
| `colormap` | `Symbol` | `:viridis` | Colormap symbol. |
| `savegif` | `String` / `nothing` | `nothing` | Output file path (`.gif` or `.mp4`) to export gantry rotation movie. |
| `framerate` | `Int` | `15` | Framerate in frames per second. |
| `figure_size` | `Tuple{Int, Int}` | `(650, 500)` | Output figure resolution. |

### Examples

```julia
# Forward project Shepp-Logan phantom
geo = Geometry(:cone; angles = Float32.(range(0, 2pi, length=180)))
projections = Ax(vol, geo)

# Display projection at gantry angle 0
fig = plotProj(projections, geo.angles; slice = 1)

# Export rotating gantry projection movie
plotProj(projections, geo.angles; step = 2, savegif = "rotation.gif", framerate = 20)
```
