# Quick Start

This guide demonstrates how to set up a CT scan geometry, simulate projection data (forward projection), and reconstruct the 3D volume using analytical and iterative methods.

---

## 1. 3D Cone-Beam CT (CBCT)

### Define Geometry and Phantom

```julia
using Tigre

# 1. Define circular cone-beam CT geometry
geo = Geometry(
    mode = :cone,
    angles = Float32.(range(0, 2pi, length=180)), # 180 projections over 360 degrees
    nVoxel = (128, 128, 128),                    # Volume dimensions (x, y, z)
    sVoxel = (128f0, 128f0, 128f0),              # Total volume physical size in mm
    nDetector = (192, 192),                      # Detector pixel matrix (u, v)
    sDetector = (192f0, 192f0),                  # Total detector size in mm
    DSO = 1000f0,                                # Source-to-Object distance in mm
    DSD = 1500f0,                                # Source-to-Detector distance in mm
)

# 2. Create a standard 3D Shepp-Logan phantom
phantom = shepp_logan_3d(volume_size(geo))
```

### Forward Projection (`Ax`)

Simulate projection sinograms through the volume:

```julia
# Forward projection (ray-tracing on GPU)
projections = Ax(phantom, geo)
# projections has dimensions: (192, 192, 180) -> (nDetU, nDetV, nAngles)
```

### Direct Reconstruction with FDK (`fdk`)

```julia
# Filtered Backprojection with Feldkamp-Davis-Kress (FDK)
rec_fdk = fdk(projections, geo; filter = :hann)
```

### Iterative Reconstruction with SART & FISTA

```julia
# 10 iterations of SART (Simultaneous Algebraic Reconstruction Technique)
rec_sart = sart(projections, geo, 10; lambda = 1.0f0, nonneg = true)

# 15 iterations of FISTA with Total Variation (TV) denoising
rec_fista = fista(projections, geo, 15; tviter = 10, tvlambda = 0.1f0, nonneg = true)
```

---

## 2. 3D Parallel-Beam CT

For synchrotron or parallel x-ray beam setups:

```julia
# Define parallel-beam geometry (angles typically span 180 degrees / π radians)
geo_par = Geometry(
    mode = :parallel,
    angles = Float32.(range(0, pi, length=120)),
    nVoxel = (128, 128, 128),
    sVoxel = (128f0, 128f0, 128f0),
    nDetector = (192, 192),
    sDetector = (192f0, 192f0),
    DSO = 1000f0,
    DSD = 1000f0, # Keep equal in parallel mode
)

# Forward projection
proj_par = Ax(phantom, geo_par)

# Direct parallel-beam Filtered Backprojection (FBP)
rec_fbp = fbp(proj_par, geo_par; filter = :hann)
```

---

## 3. In-Place Functions for High Efficiency

Every major operator and solver in `Tigre.jl` provides mutating in-place variants (`!` suffix) that write into pre-allocated memory buffers, avoiding GC overhead during repeated scans:

```julia
# Pre-allocate output volume
rec_buffer = Array{Float32}(undef, volume_size(geo))

# Run in-place FDK
fdk!(rec_buffer, projections, geo; filter = :hann)

# Run in-place iterative algorithms
fista!(rec_buffer, projections, geo, 20; tviter = 5)
```

---

## 4. Visualizing Projections and Reconstructions

`Tigre.jl` provides upstream-compatible visualization functions (`plotImg` and `plotProj`) powered by **Makie** via Package Extensions.

Load your preferred Makie backend (such as `CairoMakie` for high-quality static/vector output, or `GLMakie` for interactive inspection):

```julia
using CairoMakie

# 1. Visualize 3D volume orthogonal slice (default: central axial Z slice)
fig1 = plotImg(rec_fdk)

# 2. Inspect a specific coronal (Y) slice with custom contrast limits
fig2 = plotImg(rec_fista; dim = :y, slice = 64, clims = (0.0f0, 1.0f0), colormap = :viridis)

# 3. Save an animated slice movie (GIF or MP4) through all axial planes
plotImg(rec_fista; dim = :z, step = 2, savegif = "fista_reconstruction.gif", framerate = 15)

# 4. Visualize projection radiographs
fig3 = plotProj(projections, geo.angles; slice = 1)

# 5. Export rotating projection movie
plotProj(projections, geo.angles; step = 2, savegif = "projections.gif", framerate = 20)
```

For detailed options and multi-planar view techniques, see the [Visualization Guide](visualization.md).
