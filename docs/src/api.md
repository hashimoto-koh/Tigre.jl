# API Reference

This page provides the complete, categorized reference of types, operators, reconstruction algorithms, and utilities in `Tigre.jl`.

Each item is documented below with its signatures, parameters, mathematical formulation, and usage examples.

---

## Geometry & Coordinate Transformations

```@docs
Geometry
normalize_angles
projection_count
projection_size
volume_size
```

---

## Projection & Backprojection Operators

```@docs
Ax
Ax!
Atb
Atb!
is_canonical_layout
validate_volume_array
validate_projection_array
```

---

## Analytical Reconstruction

```@docs
fdk
fdk!
fbp
fbp!
filter_response
filter_projections!
wang_padding
wang_preweight!
parkerweight!
```

---

## Algebraic & POCS Solvers

### SART & SIRT Family

```@docs
sart
sart!
sirt
sirt!
ossart
ossart!
sart_tv
sart_tv!
ossart_tv
ossart_tv!
```

### ASD-POCS & AwASD-POCS Family

```@docs
asd_pocs
asd_pocs!
os_asd_pocs
os_asd_pocs!
awasd_pocs
awasd_pocs!
os_awasd_pocs
os_awasd_pocs!
```

### PICCS & PCSD Family

```@docs
piccs
piccs!
os_piccs
os_piccs!
pcsd
pcsd!
aw_pcsd
aw_pcsd!
os_pcsd
os_pcsd!
os_aw_pcsd
os_aw_pcsd!
```

---

## Statistical & First-Order Solvers

```@docs
mlem
mlem!
ista
ista!
fista
fista!
power_method
estimate_hyper
```

---

## Krylov Subspace Solvers

```@docs
cgls
cgls!
lsqr
lsqr!
lsmr
lsmr!
hybrid_lsqr
hybrid_lsqr!
hybrid_flsqr_tv
hybrid_flsqr_tv!
ab_gmres
ab_gmres!
ba_gmres
ba_gmres!
irn_tv_cgls
irn_tv_cgls!
```

---

## Regularization & Total Variation

```@docs
tvdenoise
tvdenoise!
minTV
minTV!
awminTV
awminTV!
piccsTV
piccsTV!
```

---

## Preprocessing & FOV Utilities

```@docs
computeCOR
binning
subsample_angles
compute_fov_radius
out_of_fov_mask
crop_cbct
crop_cbct!
```

---

## Workspaces & Memory Management

```@docs
ARTPlan
FirstOrderWorkspace
KrylovWorkspace
IterativeHistory
prepare_art_plan
prepare_first_order_workspace
prepare_krylov_workspace
```

---

## Scanning Trajectories & Non-Standard Geometries

```@docs
static_detector_geo
static_det_linear_source_geo
arbitrary_source_detector_geo
rotation_from_vecs
rot_to_euler
euler_from_vecs
```

---

## Phantoms, Noise Simulation & Quality Assessment

### Phantoms
```@docs
shepp_logan_3d
```

### Noise Simulation
```@docs
add_ct_noise
add_ct_noise!
```

### Image Quality Metrics
```@docs
measure_quality
rmse
nrmse
psnr
ssim
uqi
ncc
```

---

## Visualization Utilities

Visualization utilities matching upstream TIGRE's `plotImg` and `plotProj`.
These functions are activated via Package Extension when `CairoMakie` or `GLMakie` is loaded (`using CairoMakie` or `using GLMakie`).

```@docs
plotImg
plotProj
```

---

## System & Low-Level API

```@docs
available_gpus
native_library_path
native_library_available
native_build_info_path
native_build_info
native_build_sources
native_build_stale
native_build_stale_reasons
library_status
build_native
```
