# Workspace & Plan Reuse

In iterative computed tomography, initializing projection weight matrices, subset indices, and intermediate volume buffers accounts for a notable fraction of runtime and memory allocations.

For high-throughput applications—such as dynamic (4D) CT, cardiac gating, parameter tuning sweeps, or real-time inspection—`Tigre.jl` provides dedicated **pre-allocation workspaces**. Passing a pre-built workspace allows algorithms to run with **zero heap allocations** inside the reconstruction loop.

---

## Workspace Pre-Allocation APIs

`Tigre.jl` categorizes workspaces into three core types:

### 1. `prepare_art_plan` (Algebraic & POCS Family)

Pre-computes subset normalization weights ($V, W$), projection slices, and auxiliary buffers for algebraic and POCS algorithms.

```julia
plan = prepare_art_plan(
    projection_template,
    geo;
    blocksize = 20,
    order_strategy = :ordered,       # :ordered, :random, or :angulardistance
    need_full_projection = false,    # Set true if compute_l2=true or return_history=true
    extra_volume_buffers = 0,
)
```

!!! note "Multi-Frame Projection Array Reuse"
    When reusing an `ARTPlan` across **different** projection arrays (e.g., dynamic CT time frames), the subset indices must be contiguous (`order_strategy = :ordered` or `blocksize = 1`). Non-contiguous strategies (`:angularDistance`, `:random` with `blocksize > 1`) cache internal projection slices bound to the specific projection array provided during `prepare_art_plan`.

**Compatible algorithms**:
`sart`, `sirt`, `ossart`, `sart_tv`, `ossart_tv`, `asd_pocs`, `os_asd_pocs`, `awasd_pocs`, `os_awasd_pocs`, `pcsd`, `aw_pcsd`, `os_pcsd`, `os_aw_pcsd`, `piccs`, `os_piccs`.

### 2. `prepare_first_order_workspace` (First-Order & Statistical)

Pre-allocates forward and backward projection temporary buffers and TV proximal work arrays.

```julia
workspace = prepare_first_order_workspace(geo; need_full_projection = false)
```

**Compatible algorithms**:
`mlem`, `ista`, `fista`.

### 3. `prepare_krylov_workspace` (Krylov Subspace Family)

Pre-allocates Lanczos bidiagonalization vectors, direction buffers, and subspace basis matrices.

```julia
workspace = prepare_krylov_workspace(
    geo;
    maxiter = 20,                    # Maximum iteration depth for subspace methods
    extra_volume_buffers = 0,
    regularization_basis = false,
)
```

**Compatible algorithms**:
`cgls`, `lsqr`, `lsmr`, `hybrid_lsqr`, `hybrid_flsqr_tv`, `ab_gmres`, `ba_gmres`, `irn_tv_cgls`.

---

## Example: Zero-Allocation Time-Series Reconstruction Loop

Consider reconstructing 50 consecutive time frames acquired with the same geometry:

```julia
using Tigre

# Setup geometry and workspace
geo = ...
rec_volume = Array{Float32}(undef, volume_size(geo))
workspace = prepare_first_order_workspace(geo)

# Dynamic CT reconstruction loop
for (frame_idx, frame_projection) in enumerate(time_series_projections)
    # Reconstruct directly into pre-allocated memory using pre-allocated workspace
    fista!(rec_volume, frame_projection, geo, 15; workspace = workspace, tviter = 5)
    
    # Process or save rec_volume for current frame
    save_frame(frame_idx, rec_volume)
end
```

By reusing `workspace` and `rec_volume`, Julia avoids garbage collection pauses, maximizing throughput across the entire dataset.
