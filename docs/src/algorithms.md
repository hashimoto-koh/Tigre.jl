# Reconstruction Algorithms

`Tigre.jl` provides an extensive suite of over 28 reconstruction algorithms. All iterative algorithms are implemented natively in Julia, ensuring high performance, memory efficiency, and easy extensibility.

---

## Algorithm Catalog Overview

| Algorithm | Category | Description | Primary Use Case |
|:---|:---|:---|:---|
| `fdk` | Analytical | Feldkamp-Davis-Kress filtered backprojection for cone-beam geometry | Standard CBCT, fast reconstruction |
| `fbp` | Analytical | Filtered backprojection for parallel-beam geometry | Synchrotron, parallel beam |
| `sart` | Algebraic | Simultaneous Algebraic Reconstruction Technique | General transmission CT |
| `sirt` | Algebraic | Simultaneous Iterative Reconstruction Technique | Robust convergence in noisy data |
| `ossart` | Algebraic (OS) | Ordered Subsets SART | Accelerated convergence via subset blocks |
| `sart_tv` | Algebraic + TV | SART with steepest descent Total Variation denoising | Low-dose CT, noise reduction |
| `ossart_tv`| Algebraic + TV | OS-SART with Total Variation denoising | Fast low-dose CT |
| `asd_pocs` | POCS / TV | Adaptive Steepest Descent POCS | Sparse-view CT, few-projection data |
| `os_asd_pocs`| POCS / TV | Ordered Subsets ASD-POCS | Accelerated sparse-view CT |
| `awasd_pocs`| POCS / TV | Edge-preserving Adaptive-Weighted ASD-POCS | Sharp edge preservation |
| `os_awasd_pocs`| POCS / TV | Ordered Subsets AW-ASD-POCS | Fast edge-preserving sparse-view CT |
| `mlem` | Statistical | Maximum Likelihood Expectation Maximization | Emission / Poisson counting statistics |
| `ista` | Proximal | Iterative Shrinkage-Thresholding Algorithm | L1 / TV regularized least-squares |
| `fista` | Proximal | Fast Iterative Shrinkage-Thresholding (Nesterov accelerated) | Accelerated sparse reconstruction |
| `piccs` | Prior-guided | Prior Image Constrained Compressed Sensing | Dynamic CT, motion-guided scanning |
| `os_piccs` | Prior-guided | Ordered Subsets PICCS | Accelerated dynamic CT |
| `pcsd` | TV POCS | Projection onto Convex Sets with Steepest Descent TV | Robust constrained TV minimization |
| `aw_pcsd` | TV POCS | Adaptive-Weighted PCSD | Adaptive edge-preserving constrained TV |
| `os_pcsd` | TV POCS | Ordered Subsets PCSD | Fast TV POCS |
| `os_aw_pcsd`| TV POCS | Ordered Subsets Adaptive-Weighted PCSD | Fast adaptive TV POCS |
| `cgls` | Krylov | Conjugate Gradient Least Squares | Fast convergence on linear least squares |
| `lsqr` | Krylov | Paige & Saunders Bidiagonalization LSQR | Stable least squares for ill-conditioned data |
| `lsmr` | Krylov | Fong & Saunders LSMR | Monotonic residual norm reduction |
| `hybrid_lsqr`| Krylov Subspace | Lanczos bidiagonalization with projected Tikhonov solve | Automatic regularization parameter selection |
| `hybrid_flsqr_tv`| Krylov Subspace | Subspace bidiagonalization with TV preconditioning | High-detail feature preservation |
| `ab_gmres` | Krylov GMRES | Unmatched backprojector GMRES (projector-first) | Non-adjoint backprojectors (e.g. FDK backprojector) |
| `ba_gmres` | Krylov GMRES | Unmatched backprojector GMRES (backprojector-first) | Non-adjoint backprojectors |
| `irn_tv_cgls`| Reweighted CG | Iteratively Reweighted Norm TV with inner CGLS | High-contrast edge recovery |

---

## Common Arguments and Options

All iterative solvers share consistent keyword conventions:

- `init`: Volume initialization (`:zeros` [default], `:fdk`, or a custom 3D array `Array{Float32, 3}`).
- `nonneg::Bool`: Enforces non-negativity constraint ($x \ge 0$) after each iteration (default: `true`).
- `gpu_ids`: Target GPU devices (e.g. `[0]`).
- `show_progress::Bool`: Displays a command-line progress bar (default: `false`).
- `return_history::Bool`: Returns `(volume, history)` instead of just `volume`.
- `compute_l2::Bool`: Evaluates projection residual error $\|Ax - b\|_2$ after each iteration.
- `stop_on_l2_increase::Bool`: Early-stops iterations if residual begins to diverge.

---

## Selected Detailed Examples

### 1. FISTA (`fista`, `fista!`)
Fast Iterative Shrinkage-Thresholding Algorithm with Total Variation proximal operator:

```julia
volume, history = fista(
    projections,
    geo,
    20;                          # Number of outer iterations
    tviter = 10,                 # TV denoising sub-iterations
    tvlambda = 0.1f0,            # Regularization strength
    hyper = :auto,               # Lipschitz gradient step constant (:auto, "auto", or numeric Float32)
    nonneg = true,
    return_history = true,
)

println("Iterations completed: ", history.iterations_completed)
println("Residual history: ", history.residual_l2)
```

Pass `hyper = :auto` (or `hyper = "auto"`) to automatically estimate the optimal problem-size-dependent gradient step using SVD power iteration (`estimate_hyper` / `power_method`), matching upstream TIGRE behaviour. Explicit numeric values (e.g. `hyper = 2.0f8`) can still be provided.

### 2. OS-SART (`ossart`, `ossart!`)
Ordered Subsets Simultaneous Algebraic Reconstruction Technique:

```julia
rec = ossart(
    projections,
    geo,
    10;
    blocksize = 20,                          # Number of angles per subset
    order_strategy = :angulardistance,       # :ordered, :random, or :angulardistance
    lambda = 1.0f0,                          # Relaxation factor
    lambda_red = 0.95f0,                     # Relaxation decay per iteration
)
```

### 3. ASD-POCS (`asd_pocs`, `asd_pocs!`)
Adaptive Steepest Descent Projection Onto Convex Sets for sparse-view CT:

```julia
rec = asd_pocs(
    projections,
    geo,
    15;
    tviter = 20,
    maxl2err = 0.05f0,                       # Target residual stopping tolerance
    alpha = 0.002f0,                         # TV gradient step size
)
```

### 4. Krylov Solvers: LSMR (`lsmr`) & Hybrid-fLSQR-TV (`hybrid_flsqr_tv`)

```julia
# LSMR for fast, monotonic least-squares convergence
rec_lsmr = lsmr(projections, geo, 20; damp = 0.1f0)

# Hybrid-fLSQR-TV combining subspace bidiagonalization and TV
rec_htv = hybrid_flsqr_tv(projections, geo, 10; lambda = 0.05f0, precond_steps = 4)
```

### 5. PICCS (`piccs`, `piccs!`)
Prior Image Constrained Compressed Sensing using a reference/prior volume:

```julia
# Reconstruct with prior constraint (e.g. from a prior full-dose phase)
rec_piccs = piccs(
    sparse_projections,
    geo,
    prior_volume,
    10;
    prior_ratio = 0.5f0, # Balance between prior sparsity and current data
    tviter = 10,
)
```

### 6. Centre-of-Rotation Calibration (`computeCOR`)

Automatic estimate of the rotation axis mis-centering offset directly from projection data (T. Liu 2009):

```julia
# Estimate COR offset (in mm at rotation axis) from projection data
cor_offset = computeCOR(projections, geo)

# Apply estimated COR to geometry for accurate reconstruction
geo_calibrated = Geometry(
    mode = geo.mode,
    angles = geo.angles,
    nVoxel = geo.nVoxel,
    sVoxel = geo.sVoxel,
    nDetector = geo.nDetector,
    sDetector = geo.sDetector,
    DSD = geo.DSD,
    DSO = geo.DSO,
    COR = cor_offset,
)
```

### 7. Iteratively Reweighted Norm TV CGLS (`irn_tv_cgls`, `irn_tv_cgls!`)

Two-level inner/outer solver combining CGLS with iteratively reweighted Total Variation:

```julia
# IRN-TV-CGLS with outer reweighting loops and inner CGLS iterations
rec_irn = irn_tv_cgls(
    projections,
    geo,
    5;              # Inner CGLS iterations per outer loop
    niter_outer = 4, # Outer reweighting iterations
    lambda = 0.1f0,  # TV regularization parameter
    stop_on_l2_increase = true,
)
```
When `stop_on_l2_increase = true`, inner CGLS orthogonality loss is handled gracefully: the divergent step is rolled back and the inner loop breaks to allow the next outer loop to recompute TV weights and continue. The algorithm terminates early only when an entire outer reweighting fails to improve the residual.

### 8. Low-Dose CT Simulation & Quality Assessment (`add_ct_noise`, `measure_quality`)

Simulate realistic Poisson quantum photon noise and Gaussian readout noise according to the Beer-Lambert law, then evaluate reconstruction fidelity using standard quantitative metrics:

```julia
using Tigre
using Random

# 1. Forward project ground truth phantom
phantom = shepp_logan_3d((128, 128, 128))
clean_projections = Ax(phantom, geo)

# 2. Add realistic CT transmission noise (low dose: I0 = 10,000 incident photons)
rng = Random.MersenneTwister(42)
noisy_projections = add_ct_noise(
    clean_projections;
    I0 = 10000,              # Incident photon count (lower = lower radiation dose)
    gaussian = (0f0, 0.5f0), # (mean, std) electronic readout noise
    poisson = true,
    rng = rng,
)

# 3. Reconstruct using analytical or iterative algorithms
rec_fdk  = fdk(noisy_projections, geo; filter = :hann)
rec_sart = sart(noisy_projections, geo, 20; lambda = 0.8f0)
rec_tv   = sart_tv(noisy_projections, geo, 20; lambda = 0.8f0, tviter = 10)

# 4. Quantify image quality against ground truth reference
q_fdk  = measure_quality(phantom, rec_fdk)
q_sart = measure_quality(phantom, rec_sart)
q_tv   = measure_quality(phantom, rec_tv)

println("FDK:     PSNR = $(round(q_fdk.psnr, digits=2)) dB, SSIM = $(round(q_fdk.ssim, digits=4))")
println("SART:    PSNR = $(round(q_sart.psnr, digits=2)) dB, SSIM = $(round(q_sart.ssim, digits=4))")
println("SART-TV: PSNR = $(round(q_tv.psnr, digits=2)) dB, SSIM = $(round(q_tv.ssim, digits=4))")
```



