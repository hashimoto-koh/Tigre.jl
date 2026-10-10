# Geometry Specification

The `Geometry` struct encapsulates all physical dimensions, spatial transformations, detector properties, and trajectory angles required for forward projection and reconstruction.

---

## Coordinate Systems and Array Layout

`Tigre.jl` enforces a consistent, zero-overhead memory layout that interfaces directly with underlying CUDA thread blocks:

### Volume Arrays
- **Dimension ordering**: `(nVoxelX, nVoxelY, nVoxelZ)`
- In Julia's column-major format, indexing `volume[x, y, z]` means that the $X$-axis is contiguous in memory.
- Coordinates: $X$ and $Y$ span the axial (transverse) plane, while $Z$ aligns with the rotation axis (patient longitudinal axis).

### Projection Arrays
- **Dimension ordering**: `(nDetectorU, nDetectorV, nAngles)`
- Indexing `projection[u, v, angle]` means the detector horizontal row $U$ is contiguous in memory.
- $U$ represents the horizontal detector channel direction, and $V$ represents the vertical column direction.

---

## Constructing a `Geometry`

The constructor takes keyword arguments:

```julia
geo = Geometry(
    mode = :cone,                                 # :cone or :parallel
    angles = angles,                             # Vector or 3xN matrix of projection angles (radians)
    nVoxel = (nx, ny, nz),                       # Number of voxels in X, Y, Z
    sVoxel = (sx, sy, sz),                       # (Optional) Total volume extent in mm
    dVoxel = (dx, dy, dz),                       # (Optional) Individual voxel pitch in mm
    nDetector = (nu, nv),                        # Number of detector pixels in U, V
    sDetector = (su, sv),                        # (Optional) Total detector area in mm
    dDetector = (du, dv),                        # (Optional) Detector pixel pitch in mm
    DSO = 1000f0,                                # Distance Source-to-Origin in mm
    DSD = 1500f0,                                # Distance Source-to-Detector in mm
    offOrigin = (0f0, 0f0, 0f0),                 # Object center offset in mm
    offDetector = (0f0, 0f0),                    # Detector panel shift in mm
    rotDetector = (0f0, 0f0, 0f0),               # Detector rotation (roll, pitch, yaw) in radians
    COR = 0f0,                                   # Center of rotation offset in mm
    accuracy = 0.5f0,                            # Ray-tracing accuracy factor (default: 0.5)
)
```

> **Note**: For each length specification (`Voxel` and `Detector`), you must specify `s` (total extent), `d` (pitch), or both. If both are given, the consistency condition $s_i \approx n_i \times d_i$ is verified.

---

## Trajectory and Projection Angles

Angles are given in radians:
1. **Single-axis circular trajectory**:
   ```julia
   angles = Float32.(range(0, 2pi, length=360))
   ```
2. **Arbitrary 3D non-coplanar trajectory**:
   A $3 \times N$ matrix specifying Euler angles $(\theta, \phi, \eta)$ per projection:
   ```julia
   angles = Matrix{Float32}(undef, 3, nangles)
   angles[1, :] .= theta_vector # Primary rotation
   angles[2, :] .= roll_vector  # Gantry tilt
   angles[3, :] .= pitch_vector # Gantry pitch
   ```

---

## Dynamic Per-Projection Geometry

For complex trajectories (helical scans, non-circular orbits, moving gantry), distances and offsets can be supplied as vectors of length $N$:

- `DSO`: `Vector{Float32}` of length $N$
- `DSD`: `Vector{Float32}` of length $N$
- `offOrigin`: $3 \times N$ matrix or tuple of 3 vectors
- `offDetector`: $2 \times N$ matrix or tuple of 2 vectors
- `rotDetector`: $3 \times N$ matrix or tuple of 3 vectors
- `COR`: `Vector{Float32}` of length $N$

---

## Extended Field-of-View (Wang Weighting & Parker Weights)

### Asymmetric / Offset Detector (Wang Weighting)
When the detector panel is shifted horizontally (`offDetectorU != 0`) to cover an extended Field-of-View (FOV), redundancy weighting and lateral padding are required:

```julia
# 1. Apply padding for laterally shifted detector
padded_proj, padded_geo, theta = wang_padding(projection, geo)

# 2. Apply smooth transition weighting across overlap boundary
wang_preweight!(padded_proj, padded_geo, theta)
```

`fdk` handles Wang padding and preweighting automatically whenever `offDetectorU != 0`.

### Short-Scan Parker Weighting
For partial scans covering $[\pi + 2\gamma_m]$:

```julia
# In-place redundancy weighting for short scans
parkerweight!(projection, geo)

# Or pass parker=true directly to fdk
rec = fdk(projection, geo; parker = true)
```

---

## Convenience Introspection Functions

```julia
volume_size(geo)      # Tuple: (nx, ny, nz)
projection_size(geo)  # Tuple: (nu, nv, nangles)
projection_count(geo) # Integer: nangles
```
