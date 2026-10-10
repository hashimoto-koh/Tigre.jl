# Digital Phantoms

`Tigre.jl` includes built-in 3D digital phantoms and integrates seamlessly with external phantom synthesis packages such as [TomoPhantom.jl](https://github.com/hashimoto-koh/TomoPhantom.jl).

---

## Built-In 3D Phantoms

### Shepp-Logan 3D (`shepp_logan_3d`)

Generates an analytical 3D Shepp-Logan ellipsoid phantom:

```julia
using Tigre

# Cubic volume (128 x 128 x 128)
phantom = shepp_logan_3d(128)

# Arbitrary non-cubic dimensions (nx, ny, nz)
phantom_rect = shepp_logan_3d((128, 128, 64))

# Variant models (:kak_slaney, :yu_ye_wang, :toft_schabel)
phantom_yu = shepp_logan_3d(128; phantom_type = :yu_ye_wang)
```

### 2D Phantoms from 3D Shepp-Logan

Because `Tigre.jl`'s CUDA kernels are natively 3D, 2D geometries are treated as volumes with 1 slice along $z$ (`nVoxel = (N, N, 1)`). To create a standard 2D Shepp-Logan phantom slice, generate a 3D volume and extract its central axial slice:

```julia
# 1. Generate full 3D phantom
N = 256
vol_3d = shepp_logan_3d(N; phantom_type = :yu_ye_wang)

# 2. Extract central 2D slice and reshape to (N, N, 1) for Tigre
phantom_2d = vol_3d[:, :, N ÷ 2]
vol_phantom = reshape(phantom_2d, N, N, 1)
```

---

## Integration with TomoPhantom.jl

[TomoPhantom.jl](https://github.com/hashimoto-koh/TomoPhantom.jl) provides a wide variety of 2D and 3D digital models (geometric, cellular, foam, and realistic anatomical structures) that can be projected and reconstructed in `Tigre.jl`.

### 2D Phantoms with TomoPhantom.jl

Because `Tigre.jl` operates natively on 3D arrays, 2D phantoms are reshaped with a third dimension of size 1 (`nVoxel = (N, N, 1)`):

```julia
using Tigre
using TomoPhantom

# 1. Generate 2D phantom
core = NativeCore()
model = LibraryModel(1, default_2d_library_path())
phantom_2d = phantom2d(core, model, 256) # 256 x 256 Float32 array

# 2. Reshape to (256, 256, 1) for Tigre
vol_2d = reshape(phantom_2d, 256, 256, 1)

# 3. Setup 2D Geometry (1-pixel detector height)
geo_2d = Geometry(
    mode = :cone,
    angles = Float32.(range(0, 2pi, length=180)),
    nVoxel = (256, 256, 1),
    sVoxel = (256f0, 256f0, 1f0),
    nDetector = (384, 1),
    sDetector = (384f0, 1f0),
    DSO = 1000f0,
    DSD = 1500f0,
)

# 4. Project and reconstruct
proj = Ax(vol_2d, geo_2d)
rec_3d = fdk(proj, geo_2d)
rec_2d = dropdims(rec_3d, dims = 3) # Back to (256, 256)
```

### 3D Phantoms with TomoPhantom.jl

```julia
# Generate 3D phantom
model_3d = LibraryModel(1, default_3d_library_path())
phantom_3d = phantom3d(core, model_3d, 128) # 128 x 128 x 128 Float32 array

# Directly usable in Tigre.jl
geo_3d = Geometry(
    mode = :cone,
    angles = Float32.(range(0, 2pi, length=120)),
    nVoxel = (128, 128, 128),
    sVoxel = (128f0, 128f0, 128f0),
    nDetector = (192, 192),
    sDetector = (192f0, 192f0),
    DSO = 1000f0,
    DSD = 1500f0,
)

proj = Ax(phantom_3d, geo_3d)
rec = fdk(proj, geo_3d)
```
