struct IterativeSubset
    indices::Vector{Int}
    range::UnitRange{Int}
    contiguous::Bool
    geo::Geometry
    V::Matrix{Float32}
    projection_cache::Union{Nothing, Array{Float32, 3}}
    weight_cache::Union{Nothing, Array{Float32, 3}}
end

struct IterativeWorkspace
    proj_tmp::Array{Float32, 3}
    backproj_tmp::Array{Float32, 3}
    full_proj_tmp::Union{Nothing, Array{Float32, 3}}
    extra_volumes::Vector{Array{Float32, 3}}
end

"""
    ARTPlan

Precomputed geometry decomposition, backprojection weights, and reusable scratch buffers for ART / SART / OS-SART algorithms.

Created via `prepare_art_plan(projection, geo; ...)`. Reusing an `ARTPlan` across frames in dynamic CT eliminates repeated weight matrix computations and scratch buffer allocations.
"""
struct ARTPlan
    W::Array{Float32, 3}
    subsets::Vector{IterativeSubset}
    workspace::IterativeWorkspace
    projection_owner::UInt
    requires_same_projection::Bool
    residual_seed_cache::Base.RefValue{Float32}
end

"""
    FirstOrderWorkspace

Pre-allocated projection and volume work buffers for first-order proximal gradient algorithms (ISTA, FISTA).

Created via `prepare_first_order_workspace(geo; ...)`. Reusing across successive calls achieves zero heap allocation in iteration loops.
"""
struct FirstOrderWorkspace
    proj_tmp::Array{Float32, 3}
    backproj_tmp::Array{Float32, 3}
    prox_tmp::Array{Float32, 3}
    aux_tmp::Array{Float32, 3}
    full_proj_tmp::Union{Nothing, Array{Float32, 3}}
end

struct SubspaceWorkspace
    maxiter::Int
    proj_basis::Matrix{Float32}
    proj_basis_aux::Matrix{Float32}
    vol_basis::Matrix{Float32}
    vol_basis_aux::Matrix{Float32}
    mat_a::Matrix{Float32}
    mat_b::Matrix{Float32}
    mat_c::Matrix{Float32}
    mat_d::Matrix{Float32}
    rhs::Vector{Float32}
    coeffs::Vector{Float32}
    coeffs_aux::Vector{Float32}
end

"""
    KrylovWorkspace

Pre-allocated scratch buffers and optional subspace bases for Krylov-subspace solvers (CGLS, LSQR, LSMR, AB-GMRES, BA-GMRES, Hybrid methods).

Created via `prepare_krylov_workspace(geo; ...)`.
"""
struct KrylovWorkspace
    proj_a::Array{Float32, 3}
    proj_b::Array{Float32, 3}
    vol_a::Array{Float32, 3}
    vol_b::Array{Float32, 3}
    vol_c::Array{Float32, 3}
    vol_d::Array{Float32, 3}
    vol_e::Array{Float32, 3}
    extra_volumes::Vector{Array{Float32, 3}}
    subspace::Union{Nothing, SubspaceWorkspace}
end

"""
    IterativeHistory

Convergence history and termination metadata returned when `return_history = true` in iterative reconstruction algorithms.

# Fields
- `residual_l2::Vector{Float32}`: L2 data discrepancy norm `||Ax - b||₂` per iteration.
- `lambda_history::Vector{Float32}`: Relaxation parameter `λ` per iteration.
- `iterations_completed::Int`: Number of executed iterations.
- `stop_reason::Symbol`: Reason for termination (`:max_iterations`, `:l2_increase`, `:tol`, etc.).
"""
struct IterativeHistory
    residual_l2::Vector{Float32}
    lambda_history::Vector{Float32}
    iterations_completed::Int
    stop_reason::Symbol
end

function _make_progress(total::Integer, show_progress::Bool, desc::AbstractString)
    (!show_progress || total <= 0) && return nothing
    _ProgressMeter.ijulia_behavior(:clear)
    return _ProgressMeter.Progress(total; enabled = true, showspeed = true, desc)
end

_next_progress!(::Nothing, ::Integer, ::Integer) = nothing

function _next_progress!(progress::_ProgressMeter.Progress, current::Integer, total::Integer)
    _ProgressMeter.next!(progress; showvalues = [("iteration count", "$(current)/$(total)")])
    return nothing
end

_finish_progress!(::Nothing, ::Integer, ::Integer) = nothing

function _finish_progress!(progress::_ProgressMeter.Progress, completed::Integer, total::Integer)
    if completed < total
        _ProgressMeter.cancel(progress, "Stopped at $(completed)/$(total) iterations"; color = :yellow)
    else
        _ProgressMeter.finish!(progress)
    end
    return nothing
end

function _copy_projection_subset(projection::StridedArray{Float32, 3}, indices::AbstractVector{Int})
    out = Array{Float32}(undef, size(projection, 1), size(projection, 2), length(indices))
    @inbounds for local_angle in eachindex(indices)
        global_angle = indices[local_angle]
        out[:, :, local_angle] .= @view projection[:, :, global_angle]
    end
    return out
end

function _slice_geometry(geo::Geometry, indices::AbstractVector{Int})
    return _copy_geometry(
        geo;
        angles = copy(@view geo.angles[:, indices]),
        DSD = copy(@view geo.DSD[indices]),
        DSO = copy(@view geo.DSO[indices]),
        offOriginX = copy(@view geo.offOriginX[indices]),
        offOriginY = copy(@view geo.offOriginY[indices]),
        offOriginZ = copy(@view geo.offOriginZ[indices]),
        offDetectorU = copy(@view geo.offDetectorU[indices]),
        offDetectorV = copy(@view geo.offDetectorV[indices]),
        rotDetectorRoll = copy(@view geo.rotDetectorRoll[indices]),
        rotDetectorPitch = copy(@view geo.rotDetectorPitch[indices]),
        rotDetectorYaw = copy(@view geo.rotDetectorYaw[indices]),
        COR = copy(@view geo.COR[indices]),
    )
end

function _angular_distance_order(geo::Geometry)
    alpha = collect(@view geo.angles[1, :])
    sorted = sortperm(alpha)
    isempty(sorted) && return Int[]

    remaining_alpha = alpha[sorted[2:end]]
    remaining_index = sorted[2:end]
    ordered = Int[sorted[1]]
    ordered_alpha = Float32[alpha[sorted[1]]]

    for i in eachindex(remaining_alpha)
        comp = ordered_alpha[i]
        target = argmin(abs.(abs.(remaining_alpha .- comp) .- (Float32(pi) / 2f0)))
        push!(ordered, remaining_index[target])
        push!(ordered_alpha, remaining_alpha[target])
        deleteat!(remaining_alpha, target)
        deleteat!(remaining_index, target)
    end
    return ordered
end

function _subset_index_blocks(geo::Geometry, blocksize::Int, order_strategy)
    nangles = projection_count(geo)
    blocksize >= 1 || throw(ArgumentError("blocksize must be >= 1"))
    blocksize = min(blocksize, nangles)

    mode = order_strategy === nothing ? :ordered : Symbol(lowercase(String(order_strategy)))

    if blocksize == 1
        base = collect(1:nangles)
        if mode === :ordered
            return [Int[i] for i in base]
        elseif mode === :random || mode === :random2
            Random.shuffle!(base)
            return [Int[i] for i in base]
        elseif mode === :angulardistance
            return [Int[i] for i in _angular_distance_order(geo)]
        end
        throw(ArgumentError("unsupported order_strategy: $(order_strategy)"))
    end

    if mode === :random2
        shuffled = collect(1:nangles)
        Random.shuffle!(shuffled)
        return [shuffled[first:last] for (first, last) in ((i, min(i + blocksize - 1, nangles)) for i in 1:blocksize:nangles)]
    elseif mode === :angulardistance
        ordered = _angular_distance_order(geo)
        return [ordered[first:last] for (first, last) in ((i, min(i + blocksize - 1, nangles)) for i in 1:blocksize:nangles)]
    end

    blocks = [collect(first:last) for (first, last) in ((i, min(i + blocksize - 1, nangles)) for i in 1:blocksize:nangles)]
    if mode === :ordered
        return blocks
    elseif mode === :random
        Random.shuffle!(blocks)
        return blocks
    end
    throw(ArgumentError("unsupported order_strategy: $(order_strategy)"))
end

function _contiguous_range(indices::Vector{Int})
    start = first(indices)
    stop = last(indices)
    if stop - start + 1 == length(indices)
        @inbounds for i in eachindex(indices)
            indices[i] == start + i - 1 || return nothing
        end
        return start:stop
    end
    return nothing
end

function _compute_w(geo::Geometry; gpu_ids = nothing)
    geox = _copy_geometry(geo)
    sx = max(geo.sDetector[1], geo.sVoxel[1])
    sy = geo.sVoxel[2] * 1.1f0
    sz = geo.sVoxel[3] * 1.1f0
    geox = _copy_geometry(
        geox;
        nVoxel = (2, 2, 2),
        sVoxel = (sx, sy, sz),
        dVoxel = (sx / 2f0, sy / 2f0, sz / 2f0),
    )
    weights = Ax(fill(1f0, volume_size(geox)...), geox; projection_type = :siddon, gpu_ids)
    cutoff = minimum(Float32.(geo.dVoxel)) / 2f0
    @inbounds for i in eachindex(weights)
        if weights[i] <= cutoff
            weights[i] = 0f0
        else
            inv_w = inv(weights[i])
            weights[i] = inv_w > 0.1f0 ? 0.1f0 : inv_w
        end
    end
    return weights
end

function _compute_v(subset_geo::Geometry, blocklen::Int; gpu_ids = nothing)
    if subset_geo.mode === :parallel
        return fill(Float32(blocklen), subset_geo.nVoxel[1], subset_geo.nVoxel[2])
    end

    proj_one = fill(1f0, projection_size(subset_geo)...)
    backproj = Atb(proj_one, subset_geo; backprojection_type = :fdk, gpu_ids)
    V = Matrix{Float32}(undef, subset_geo.nVoxel[1], subset_geo.nVoxel[2])
    inv_nz = inv(Float32(subset_geo.nVoxel[3]))
    @inbounds for y in 1:subset_geo.nVoxel[2], x in 1:subset_geo.nVoxel[1]
        acc = 0f0
        for z in 1:subset_geo.nVoxel[3]
            acc += backproj[x, y, z]
        end
        value = acc * inv_nz
        V[x, y] = iszero(value) ? Inf32 : value
    end
    return V
end

function _prepare_subsets(projection::StridedArray{Float32, 3}, geo::Geometry, W::StridedArray{Float32, 3}, blocksize::Int, order_strategy; gpu_ids = nothing)
    subsets = IterativeSubset[]
    for indices in _subset_index_blocks(geo, blocksize, order_strategy)
        range = _contiguous_range(indices)
        subset_geo = _slice_geometry(geo, indices)
        V = _compute_v(subset_geo, length(indices); gpu_ids)
        if range === nothing
            projection_cache = _copy_projection_subset(projection, indices)
            weight_cache = _copy_projection_subset(W, indices)
            push!(subsets, IterativeSubset(indices, 1:0, false, subset_geo, V, projection_cache, weight_cache))
        else
            push!(subsets, IterativeSubset(indices, range, true, subset_geo, V, nothing, nothing))
        end
    end
    return subsets
end

function _iterative_workspace(geo::Geometry, blocksize::Int; need_full_projection::Bool = false, extra_volume_buffers::Int = 0)
    proj_tmp = Array{Float32}(undef, geo.nDetector[1], geo.nDetector[2], blocksize)
    backproj_tmp = Array{Float32}(undef, volume_size(geo)...)
    full_proj_tmp = need_full_projection ? Array{Float32}(undef, projection_size(geo)) : nothing
    extra_volume_buffers >= 0 || throw(ArgumentError("extra_volume_buffers must be >= 0"))
    extra_volumes = [Array{Float32}(undef, volume_size(geo)...) for _ in 1:extra_volume_buffers]
    return IterativeWorkspace(proj_tmp, backproj_tmp, full_proj_tmp, extra_volumes)
end

function _first_order_workspace(geo::Geometry; need_full_projection::Bool = false)
    proj_tmp = Array{Float32}(undef, projection_size(geo))
    backproj_tmp = Array{Float32}(undef, volume_size(geo)...)
    prox_tmp = Array{Float32}(undef, volume_size(geo)...)
    aux_tmp = Array{Float32}(undef, volume_size(geo)...)
    full_proj_tmp = need_full_projection ? Array{Float32}(undef, projection_size(geo)) : nothing
    return FirstOrderWorkspace(proj_tmp, backproj_tmp, prox_tmp, aux_tmp, full_proj_tmp)
end

function _subspace_workspace(geo::Geometry, maxiter::Int; regularization_basis::Bool = false)
    maxiter >= 0 || throw(ArgumentError("maxiter must be >= 0"))
    maxiter == 0 && return nothing
    nproj = prod(projection_size(geo))
    nvol = prod(volume_size(geo))
    return SubspaceWorkspace(
        maxiter,
        Matrix{Float32}(undef, maxiter + 1, nproj),
        Matrix{Float32}(undef, maxiter + 1, nproj),
        Matrix{Float32}(undef, maxiter + 1, nvol),
        Matrix{Float32}(undef, maxiter, 0),
        zeros(Float32, maxiter + 1, maxiter + 1),
        zeros(Float32, maxiter + 1, maxiter + 1),
        zeros(Float32, maxiter + 1, maxiter + 1),
        zeros(Float32, maxiter + 1, maxiter + 1),
        zeros(Float32, maxiter + 1),
        zeros(Float32, maxiter + 1),
        zeros(Float32, maxiter + 1),
    )
end

function _reset_subspace_scalars!(subspace::SubspaceWorkspace)
    fill!(subspace.mat_a, 0f0)
    fill!(subspace.mat_b, 0f0)
    fill!(subspace.mat_c, 0f0)
    fill!(subspace.mat_d, 0f0)
    fill!(subspace.rhs, 0f0)
    fill!(subspace.coeffs, 0f0)
    fill!(subspace.coeffs_aux, 0f0)
    return subspace
end

function _krylov_workspace(geo::Geometry; maxiter::Int = 0, extra_volume_buffers::Int = 0, regularization_basis::Bool = false)
    proj_a = Array{Float32}(undef, projection_size(geo))
    proj_b = Array{Float32}(undef, projection_size(geo))
    vol_a = Array{Float32}(undef, volume_size(geo)...)
    vol_b = Array{Float32}(undef, volume_size(geo)...)
    vol_c = Array{Float32}(undef, volume_size(geo)...)
    vol_d = Array{Float32}(undef, volume_size(geo)...)
    vol_e = Array{Float32}(undef, volume_size(geo)...)
    extra_volume_buffers >= 0 || throw(ArgumentError("extra_volume_buffers must be >= 0"))
    extra_volumes = [Array{Float32}(undef, volume_size(geo)...) for _ in 1:extra_volume_buffers]
    subspace = _subspace_workspace(geo, maxiter; regularization_basis)
    return KrylovWorkspace(proj_a, proj_b, vol_a, vol_b, vol_c, vol_d, vol_e, extra_volumes, subspace)
end

"""
    prepare_art_plan(projection::AbstractArray{Float32, 3}, geo::Geometry; blocksize::Integer = 20, order_strategy = :ordered, gpu_ids = nothing, need_full_projection::Bool = false, extra_volume_buffers::Integer = 0) -> ARTPlan

Precompute and allocate an `ARTPlan` containing subset geometries, backprojection weights, and scratch buffers.

Passing the returned plan to `ossart!`, `sart!`, etc. avoids re-allocating memory and re-calculating weights on every reconstruction call.

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `blocksize::Integer = 20`: Number of projection angles per subset (default: `20`).
- `order_strategy = :ordered`: Angle ordering scheme (`:ordered` or `:random`, default: `:ordered`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `need_full_projection::Bool = false`: Whether to preallocate full projection scratch buffer for residual computation (default: `false`).
- `extra_volume_buffers::Integer = 0`: Number of additional volume scratch buffers to allocate (default: `0`).

# Returns
- `ARTPlan`: Precomputed plan structure containing geometry subsets, weight matrices, and workspace buffers.

# Examples
```julia
plan = prepare_art_plan(projections, geo; blocksize = 20)
volume = zeros(Float32, volume_size(geo))
ossart!(volume, projections, geo, 10; plan = plan)
```
"""
function prepare_art_plan(projection::AbstractArray{Float32, 3}, geo::Geometry; blocksize::Integer = 20, order_strategy = :ordered, gpu_ids = nothing, need_full_projection::Bool = false, extra_volume_buffers::Integer = 0)
    projection = _require_canonical_array(projection, "projection", projection_size(geo))
    return _prepare_art_plan(projection, geo, Int(blocksize), order_strategy; gpu_ids, need_full_projection, extra_volume_buffers = Int(extra_volume_buffers))
end

"""
    prepare_first_order_workspace(geo::Geometry; need_full_projection::Bool = false) -> FirstOrderWorkspace

Allocate reusable scratch buffers for first-order reconstruction algorithms (`fista!`, `ista!`).

# Arguments
- `geo::Geometry`: CT acquisition geometry specification.
- `need_full_projection::Bool = false`: Whether to allocate full projection buffer for residual tracking (default: `false`).

# Returns
- `FirstOrderWorkspace`: Workspace holding pre-allocated projection, volume, proximal, and auxiliary scratch buffers.

# Examples
```julia
ws = prepare_first_order_workspace(geo)
volume = zeros(Float32, volume_size(geo))
fista!(volume, projections, geo, 50; workspace = ws)
```
"""
function prepare_first_order_workspace(geo::Geometry; need_full_projection::Bool = false)
    return _first_order_workspace(geo; need_full_projection)
end

"""
    prepare_krylov_workspace(geo::Geometry; maxiter::Integer = 0, extra_volume_buffers::Integer = 0, regularization_basis::Bool = false) -> KrylovWorkspace

Allocate reusable scratch buffers and subspace arrays for Krylov-subspace solvers (`cgls!`, `lsqr!`, `lsmr!`, `ab_gmres!`, `ba_gmres!`, `hybrid_lsqr!`, etc.).

# Arguments
- `geo::Geometry`: CT acquisition geometry specification.
- `maxiter::Integer = 0`: Maximum number of Krylov subspace iterations if subspace bases are needed (default: `0`).
- `extra_volume_buffers::Integer = 0`: Number of extra 3D volume scratch buffers to allocate (default: `0`).
- `regularization_basis::Bool = false`: Whether to allocate subspace bases for regularized Krylov methods (default: `false`).

# Returns
- `KrylovWorkspace`: Workspace containing projection vectors, volume vectors, and subspace structures.

# Examples
```julia
ws = prepare_krylov_workspace(geo)
volume = zeros(Float32, volume_size(geo))
cgls!(volume, projections, geo, 30; workspace = ws)
```
"""
function prepare_krylov_workspace(geo::Geometry; maxiter::Integer = 0, extra_volume_buffers::Integer = 0, regularization_basis::Bool = false)
    return _krylov_workspace(geo; maxiter = Int(maxiter), extra_volume_buffers = Int(extra_volume_buffers), regularization_basis)
end

function _resolve_art_plan(plan::Union{Nothing, ARTPlan}, projection::StridedArray{Float32, 3}, geo::Geometry, blocksize::Int, order_strategy; gpu_ids = nothing, need_full_projection::Bool = false)
    if plan === nothing
        return _prepare_art_plan(projection, geo, blocksize, order_strategy; gpu_ids, need_full_projection)
    end
    size(plan.W) == size(projection) || throw(ArgumentError("plan.W shape does not match projection"))
    size(plan.workspace.backproj_tmp) == volume_size(geo) || throw(ArgumentError("plan workspace volume shape does not match geometry"))
    for (i, extra) in pairs(plan.workspace.extra_volumes)
        size(extra) == volume_size(geo) || throw(ArgumentError("plan workspace.extra_volumes[$i] shape does not match geometry"))
    end
    if plan.requires_same_projection && plan.projection_owner != objectid(projection)
        throw(ArgumentError("ARTPlan with non-contiguous subsets can only be reused with the same projection array used during preparation"))
    end
    if need_full_projection && plan.workspace.full_proj_tmp === nothing
        throw(ArgumentError("residual computation requires an ARTPlan created with need_full_projection=true"))
    end
    return plan
end

function _resolve_first_order_workspace(workspace::Union{Nothing, FirstOrderWorkspace}, geo::Geometry; need_full_projection::Bool = false)
    if workspace === nothing
        return _first_order_workspace(geo; need_full_projection)
    end
    size(workspace.proj_tmp) == projection_size(geo) || throw(ArgumentError("workspace.proj_tmp shape does not match geometry"))
    size(workspace.backproj_tmp) == volume_size(geo) || throw(ArgumentError("workspace.backproj_tmp shape does not match geometry"))
    if need_full_projection && workspace.full_proj_tmp === nothing
        throw(ArgumentError("residual computation requires a FirstOrderWorkspace created with need_full_projection=true"))
    end
    return workspace
end

function _resolve_krylov_workspace(workspace::Union{Nothing, KrylovWorkspace}, geo::Geometry)
    if workspace === nothing
        return _krylov_workspace(geo)
    end
    size(workspace.proj_a) == projection_size(geo) || throw(ArgumentError("workspace.proj_a shape does not match geometry"))
    size(workspace.proj_b) == projection_size(geo) || throw(ArgumentError("workspace.proj_b shape does not match geometry"))
    size(workspace.vol_a) == volume_size(geo) || throw(ArgumentError("workspace.vol_a shape does not match geometry"))
    size(workspace.vol_e) == volume_size(geo) || throw(ArgumentError("workspace.vol_e shape does not match geometry"))
    for (i, extra) in pairs(workspace.extra_volumes)
        size(extra) == volume_size(geo) || throw(ArgumentError("workspace.extra_volumes[$i] shape does not match geometry"))
    end
    if workspace.subspace !== nothing
        subspace = workspace.subspace::SubspaceWorkspace
        size(subspace.proj_basis, 2) == prod(projection_size(geo)) || throw(ArgumentError("subspace.proj_basis shape does not match geometry"))
        size(subspace.vol_basis, 2) == prod(volume_size(geo)) || throw(ArgumentError("subspace.vol_basis shape does not match geometry"))
        regcols = size(subspace.vol_basis_aux, 2)
        (regcols == 0 || regcols == 3 * prod(volume_size(geo))) || throw(ArgumentError("subspace.vol_basis_aux shape does not match geometry"))
    end
    return workspace
end

function _resolve_subspace_workspace(workspace::KrylovWorkspace, geo::Geometry, niter::Int; require_regularization_basis::Bool = false)
    subspace = workspace.subspace
    if subspace !== nothing && subspace.maxiter >= niter
        _reset_subspace_scalars!(subspace)
        return subspace::SubspaceWorkspace
    end
    sub = _subspace_workspace(geo, niter; regularization_basis = require_regularization_basis)::SubspaceWorkspace
    _reset_subspace_scalars!(sub)
    return sub
end

function _extra_volume(workspace::KrylovWorkspace, idx::Int, geo::Geometry)
    if idx <= length(workspace.extra_volumes)
        return workspace.extra_volumes[idx]
    end
    return Array{Float32}(undef, volume_size(geo)...)
end

function _subset_projection_view(projection::StridedArray{Float32, 3}, subset::IterativeSubset)
    subset.contiguous && return @view projection[:, :, subset.range]
    return subset.projection_cache::Array{Float32, 3}
end

function _subset_weight_view(W::StridedArray{Float32, 3}, subset::IterativeSubset)
    subset.contiguous && return @view W[:, :, subset.range]
    return subset.weight_cache::Array{Float32, 3}
end

function _initialize_iterative_volume!(volume::StridedArray{Float32, 3}, projection::StridedArray{Float32, 3}, geo::Geometry, init; gpu_ids = nothing)
    if init === :zeros || init === nothing
        fill!(volume, 0f0)
        return volume
    elseif init === :fdk
        fdk!(volume, projection, geo; gpu_ids)
        clamp!(volume, 0f0, Inf32)
        return volume
    elseif init isa AbstractArray{<:Real, 3}
        size(init) == volume_size(geo) || throw(ArgumentError("init volume shape does not match geometry"))
        @inbounds for i in eachindex(volume, init)
            volume[i] = Float32(init[i])
        end
        return volume
    end
    throw(ArgumentError("init must be :zeros, :fdk, or a 3D array"))
end

function _apply_subset_update!(volume::StridedArray{Float32, 3}, backproj::StridedArray{Float32, 3}, V::Matrix{Float32}, lambda::Float32, nonneg::Bool)
    @inbounds for z in 1:size(volume, 3), y in 1:size(volume, 2), x in 1:size(volume, 1)
        updated = volume[x, y, z] + lambda * (backproj[x, y, z] / V[x, y])
        volume[x, y, z] = nonneg && updated < 0f0 ? 0f0 : updated
    end
    return volume
end

function _projection_residual_l2!(proj_tmp::StridedArray{Float32, 3}, projection::StridedArray{Float32, 3}, volume::StridedArray{Float32, 3}, geo::Geometry, projection_type, gpu_ids)
    Ax!(proj_tmp, volume, geo; projection_type, gpu_ids)
    accum = 0.0
    @inbounds for i in eachindex(proj_tmp, projection)
        diff = Float64(proj_tmp[i] - projection[i])
        accum += diff * diff
    end
    return Float32(sqrt(accum))
end

function _build_history(residual_l2::Vector{Float32}, lambda_history::Vector{Float32}, iterations_completed::Int, stop_reason::Symbol)
    return IterativeHistory(residual_l2, lambda_history, iterations_completed, stop_reason)
end

function _maybe_return_history(volume::StridedArray{Float32, 3}, history::IterativeHistory, return_history::Bool)
    return return_history ? (volume, history) : volume
end

function _diff_l2(a::StridedArray{Float32}, b::StridedArray{Float32})
    accum = 0.0
    @inbounds for i in eachindex(a, b)
        diff = Float64(a[i] - b[i])
        accum += diff * diff
    end
    return Float32(sqrt(accum))
end

function _update_dot(new_value::StridedArray{Float32}, middle_value::StridedArray{Float32}, old_value::StridedArray{Float32})
    accum = 0.0
    @inbounds for i in eachindex(new_value, middle_value, old_value)
        accum += Float64(new_value[i] - middle_value[i]) * Float64(middle_value[i] - old_value[i])
    end
    return Float32(accum)
end

function _prepare_art_plan(projection::StridedArray{Float32, 3}, geo::Geometry, blocksize::Int, order_strategy; gpu_ids = nothing, need_full_projection::Bool = false, extra_volume_buffers::Int = 0)
    W = _compute_w(geo; gpu_ids)
    subsets = _prepare_subsets(projection, geo, W, blocksize, order_strategy; gpu_ids)
    workspace = _iterative_workspace(geo, maximum(length(subset.indices) for subset in subsets); need_full_projection, extra_volume_buffers)
    requires_same_projection = any(!subset.contiguous for subset in subsets)
    return ARTPlan(W, subsets, workspace, objectid(projection), requires_same_projection, Ref(Float32(NaN)))
end

function _art_iteration!(volume::StridedArray{Float32, 3}, projection::StridedArray{Float32, 3}, plan::ARTPlan, current_lambda::Float32, nonneg::Bool, projection_type, backprojection_type, gpu_ids)
    for subset in plan.subsets
        b = length(subset.indices)
        proj_tmp = @view plan.workspace.proj_tmp[:, :, 1:b]
        proj_data = _subset_projection_view(projection, subset)
        weight_data = _subset_weight_view(plan.W, subset)
        Ax!(proj_tmp, volume, subset.geo; projection_type, gpu_ids)
        @inbounds for ia in 1:b, iv in 1:size(proj_tmp, 2), iu in 1:size(proj_tmp, 1)
            proj_tmp[iu, iv, ia] = weight_data[iu, iv, ia] * (proj_data[iu, iv, ia] - proj_tmp[iu, iv, ia])
        end
        Atb!(plan.workspace.backproj_tmp, proj_tmp, subset.geo; backprojection_type, gpu_ids)
        _apply_subset_update!(volume, plan.workspace.backproj_tmp, subset.V, current_lambda, nonneg)
    end
    return volume
end

"""
    ossart(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; blocksize::Integer = 20, init = :zeros, lambda::Real = 1f0, lambda_red::Real = 1f0, order_strategy = :ordered, nonneg::Bool = true, gpu_ids = nothing, projection_type = :siddon, backprojection_type = :fdk, compute_l2::Bool = false, stop_on_l2_increase::Bool = false, residual_projection_type = :siddon, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

Ordered Subsets Simultaneous Algebraic Reconstruction Technique (OS-SART).

Divides projection angles into subsets of size `blocksize`, accelerating convergence compared to standard SART.

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Number of complete epochs through all subsets.
- `blocksize::Integer = 20`: Number of projection angles per subset (default: `20`). Set `1` for SART, `projection_count(geo)` for SIRT.
- `init = :zeros`: Initial volume guess (`:zeros`, `:fdk`, or custom 3D array, default: `:zeros`).
- `lambda::Real = 1.0f0`: Initial relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 1.0f0`: Relaxation factor reduction per iteration (default: `1.0f0`).
- `order_strategy = :ordered`: Projection angle visiting order (`:ordered` or `:random`, default: `:ordered`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `projection_type = :siddon`: Forward projection ray-tracing kernel (`:siddon` or `:interpolated`, default: `:siddon`).
- `backprojection_type = :fdk`: Backprojection interpolation kernel (`:fdk` or `:voxel_ray`, default: `:fdk`).
- `compute_l2::Bool = false`: Compute residual L2 norm per iteration (default: `false`).
- `stop_on_l2_increase::Bool = false`: Terminate iterations early if residual L2 norm increases (default: `false`).
- `residual_projection_type = :siddon`: Ray-tracing kernel for residual evaluation (default: `:siddon`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `Array{Float32, 3}`: Reconstructed 3D volume (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
geo = default_geometry()
proj = Ax(phantom, geo)
vol = ossart(proj, geo, 20; blocksize = 20)
```
"""
function ossart(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; blocksize::Integer = 20, init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return ossart!(volume, projection_checked, geo, niter; blocksize, kwargs...)
end

"""
    ossart!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; blocksize::Integer = 20, lambda::Real = 1f0, lambda_red::Real = 1f0, order_strategy = :ordered, nonneg::Bool = true, gpu_ids = nothing, projection_type = :siddon, backprojection_type = :fdk, compute_l2::Bool = false, stop_on_l2_increase::Bool = false, residual_projection_type = :siddon, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

In-place Ordered Subsets SART reconstruction into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Number of complete epochs through all subsets.
- `blocksize::Integer = 20`: Number of projection angles per subset (default: `20`). Set `1` for SART, `projection_count(geo)` for SIRT.
- `lambda::Real = 1.0f0`: Initial relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 1.0f0`: Relaxation factor reduction per iteration (default: `1.0f0`).
- `order_strategy = :ordered`: Projection angle visiting order (`:ordered` or `:random`, default: `:ordered`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `projection_type = :siddon`: Forward projection ray-tracing kernel (`:siddon` or `:interpolated`, default: `:siddon`).
- `backprojection_type = :fdk`: Backprojection interpolation kernel (`:fdk` or `:voxel_ray`, default: `:fdk`).
- `compute_l2::Bool = false`: Compute residual L2 norm per iteration (default: `false`).
- `stop_on_l2_increase::Bool = false`: Terminate iterations early if residual L2 norm increases (default: `false`).
- `residual_projection_type = :siddon`: Ray-tracing kernel for residual evaluation (default: `:siddon`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `volume`: The modified reconstructed volume array (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = zeros(Float32, volume_size(geo))
ossart!(vol, proj, geo, 20; blocksize = 20)
```
"""
function ossart!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer;
    blocksize::Integer = 20,
    lambda::Real = 1f0,
    lambda_red::Real = 1f0,
    order_strategy = :ordered,
    nonneg::Bool = true,
    gpu_ids = nothing,
    projection_type = :siddon,
    backprojection_type = :fdk,
    compute_l2::Bool = false,
    stop_on_l2_increase::Bool = false,
    residual_projection_type = :siddon,
    return_history::Bool = false,
    show_progress::Bool = false,
    plan = nothing,
)
    niter >= 0 || throw(ArgumentError("niter must be >= 0"))
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))
    projection_count(geo) > 0 || throw(ArgumentError("angles is empty"))

    need_residual = compute_l2 || stop_on_l2_increase || return_history
    plan = _resolve_art_plan(plan, projection, geo, Int(blocksize), order_strategy; gpu_ids, need_full_projection = need_residual)
    current_lambda = Float32(lambda)
    lambda_scale = Float32(lambda_red)
    residual_l2 = Float32[]
    lambda_history = Float32[]
    prev_residual = Inf32
    stop_reason = :max_iterations
    iterations_completed = 0
    progress = _make_progress(niter, show_progress, "OS-SART: ")

    for iter in 1:niter
        push!(lambda_history, current_lambda)
        _art_iteration!(volume, projection, plan, current_lambda, nonneg, projection_type, backprojection_type, gpu_ids)

        iterations_completed = iter
        _next_progress!(progress, iter, niter)
        if need_residual
            residual = _projection_residual_l2!(plan.workspace.full_proj_tmp::Array{Float32, 3}, projection, volume, geo, residual_projection_type, gpu_ids)
            push!(residual_l2, residual)
            if stop_on_l2_increase && residual > prev_residual
                stop_reason = :l2_increase
                break
            end
            prev_residual = residual
        end

        current_lambda *= lambda_scale
    end

    _finish_progress!(progress, iterations_completed, niter)
    history = _build_history(residual_l2, lambda_history, iterations_completed, stop_reason)
    return _maybe_return_history(volume, history, return_history)
end

"""
    sart(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, lambda::Real = 1f0, lambda_red::Real = 1f0, order_strategy = :ordered, nonneg::Bool = true, gpu_ids = nothing, projection_type = :siddon, backprojection_type = :fdk, compute_l2::Bool = false, stop_on_l2_increase::Bool = false, residual_projection_type = :siddon, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

Simultaneous Algebraic Reconstruction Technique (SART). Equivalent to `ossart` with `blocksize = 1`.

Iteratively updates the 3D volume by backprojecting projection residuals one projection angle at a time.

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Number of complete iterations over all angles.
- `init = :zeros`: Initial volume guess (`:zeros`, `:fdk`, or custom 3D array, default: `:zeros`).
- `lambda::Real = 1.0f0`: Initial relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 1.0f0`: Relaxation factor reduction per iteration (default: `1.0f0`).
- `order_strategy = :ordered`: Projection angle visiting order (`:ordered` or `:random`, default: `:ordered`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `projection_type = :siddon`: Forward projection ray-tracing kernel (`:siddon` or `:interpolated`, default: `:siddon`).
- `backprojection_type = :fdk`: Backprojection interpolation kernel (`:fdk` or `:voxel_ray`, default: `:fdk`).
- `compute_l2::Bool = false`: Compute residual L2 norm per iteration (default: `false`).
- `stop_on_l2_increase::Bool = false`: Terminate iterations early if residual L2 norm increases (default: `false`).
- `residual_projection_type = :siddon`: Ray-tracing kernel for residual evaluation (default: `:siddon`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `Array{Float32, 3}`: Reconstructed 3D volume (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
geo = default_geometry()
proj = Ax(phantom, geo)
vol = sart(proj, geo, 10)
```
"""
function sart(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, kwargs...)
    return ossart(projection, geo, niter; blocksize = 1, init, kwargs...)
end

"""
    sart!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; lambda::Real = 1f0, lambda_red::Real = 1f0, order_strategy = :ordered, nonneg::Bool = true, gpu_ids = nothing, projection_type = :siddon, backprojection_type = :fdk, compute_l2::Bool = false, stop_on_l2_increase::Bool = false, residual_projection_type = :siddon, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

In-place Simultaneous Algebraic Reconstruction Technique (SART) into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Number of complete iterations over all angles.
- `lambda::Real = 1.0f0`: Initial relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 1.0f0`: Relaxation factor reduction per iteration (default: `1.0f0`).
- `order_strategy = :ordered`: Projection angle visiting order (`:ordered` or `:random`, default: `:ordered`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `projection_type = :siddon`: Forward projection ray-tracing kernel (`:siddon` or `:interpolated`, default: `:siddon`).
- `backprojection_type = :fdk`: Backprojection interpolation kernel (`:fdk` or `:voxel_ray`, default: `:fdk`).
- `compute_l2::Bool = false`: Compute residual L2 norm per iteration (default: `false`).
- `stop_on_l2_increase::Bool = false`: Terminate iterations early if residual L2 norm increases (default: `false`).
- `residual_projection_type = :siddon`: Ray-tracing kernel for residual evaluation (default: `:siddon`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `volume`: The modified reconstructed volume array (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = zeros(Float32, volume_size(geo))
sart!(vol, proj, geo, 10)
```
"""
function sart!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; kwargs...)
    return ossart!(volume, projection, geo, niter; blocksize = 1, kwargs...)
end

"""
    sirt(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, lambda::Real = 1f0, lambda_red::Real = 1f0, order_strategy = :ordered, nonneg::Bool = true, gpu_ids = nothing, projection_type = :siddon, backprojection_type = :fdk, compute_l2::Bool = false, stop_on_l2_increase::Bool = false, residual_projection_type = :siddon, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

Simultaneous Iterative Reconstruction Technique (SIRT). Equivalent to `ossart` with `blocksize = nangles`.

Updates the volume simultaneously using all projection views in a single batch update per iteration.

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Number of full iterations.
- `init = :zeros`: Initial volume guess (`:zeros`, `:fdk`, or custom 3D array, default: `:zeros`).
- `lambda::Real = 1.0f0`: Initial relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 1.0f0`: Relaxation factor reduction per iteration (default: `1.0f0`).
- `order_strategy = :ordered`: Projection angle visiting order (`:ordered` or `:random`, default: `:ordered`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `projection_type = :siddon`: Forward projection ray-tracing kernel (`:siddon` or `:interpolated`, default: `:siddon`).
- `backprojection_type = :fdk`: Backprojection interpolation kernel (`:fdk` or `:voxel_ray`, default: `:fdk`).
- `compute_l2::Bool = false`: Compute residual L2 norm per iteration (default: `false`).
- `stop_on_l2_increase::Bool = false`: Terminate iterations early if residual L2 norm increases (default: `false`).
- `residual_projection_type = :siddon`: Ray-tracing kernel for residual evaluation (default: `:siddon`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `Array{Float32, 3}`: Reconstructed 3D volume (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = sirt(proj, geo, 20)
```
"""
function sirt(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, kwargs...)
    return ossart(projection, geo, niter; blocksize = projection_count(geo), init, kwargs...)
end

"""
    sirt!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; lambda::Real = 1f0, lambda_red::Real = 1f0, order_strategy = :ordered, nonneg::Bool = true, gpu_ids = nothing, projection_type = :siddon, backprojection_type = :fdk, compute_l2::Bool = false, stop_on_l2_increase::Bool = false, residual_projection_type = :siddon, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

In-place Simultaneous Iterative Reconstruction Technique (SIRT) into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Number of full iterations.
- `lambda::Real = 1.0f0`: Initial relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 1.0f0`: Relaxation factor reduction per iteration (default: `1.0f0`).
- `order_strategy = :ordered`: Projection angle visiting order (`:ordered` or `:random`, default: `:ordered`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `projection_type = :siddon`: Forward projection ray-tracing kernel (`:siddon` or `:interpolated`, default: `:siddon`).
- `backprojection_type = :fdk`: Backprojection interpolation kernel (`:fdk` or `:voxel_ray`, default: `:fdk`).
- `compute_l2::Bool = false`: Compute residual L2 norm per iteration (default: `false`).
- `stop_on_l2_increase::Bool = false`: Terminate iterations early if residual L2 norm increases (default: `false`).
- `residual_projection_type = :siddon`: Ray-tracing kernel for residual evaluation (default: `:siddon`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `volume`: The modified reconstructed volume array (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = zeros(Float32, volume_size(geo))
sirt!(vol, proj, geo, 20)
```
"""
function sirt!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; kwargs...)
    return ossart!(volume, projection, geo, niter; blocksize = projection_count(geo), kwargs...)
end

function _run_tv_art!(volume::StridedArray{Float32, 3}, projection::StridedArray{Float32, 3}, geo::Geometry, niter::Integer;
    blocksize::Integer,
    lambda::Real,
    lambda_red::Real,
    order_strategy,
    nonneg::Bool,
    tviter::Integer,
    tvlambda::Real,
    gpu_ids,
    return_history::Bool,
    compute_l2::Bool,
    stop_on_l2_increase::Bool,
    residual_projection_type,
    show_progress::Bool,
    plan = nothing,
)
    need_residual = compute_l2 || stop_on_l2_increase || return_history
    plan = _resolve_art_plan(plan, projection, geo, Int(blocksize), order_strategy; gpu_ids, need_full_projection = need_residual)
    prox_tmp = _art_extra_volume(plan, 1, geo)
    current_lambda = Float32(lambda)
    lambda_scale = Float32(lambda_red)
    residual_l2 = Float32[]
    lambda_history = Float32[]
    prev_residual = Inf32
    stop_reason = :max_iterations
    iterations_completed = 0
    progress = _make_progress(niter, show_progress, "TV-ART: ")

    for iter in 1:niter
        push!(lambda_history, current_lambda)
        _art_iteration!(volume, projection, plan, current_lambda, nonneg, :siddon, :fdk, gpu_ids)
        tvdenoise!(prox_tmp, volume; lambda = tvlambda, maxiter = tviter, gpu_ids)
        copyto!(volume, prox_tmp)
        nonneg && clamp!(volume, 0f0, Inf32)
        iterations_completed = iter
        _next_progress!(progress, iter, niter)

        if need_residual
            residual = _projection_residual_l2!(plan.workspace.full_proj_tmp::Array{Float32, 3}, projection, volume, geo, residual_projection_type, gpu_ids)
            push!(residual_l2, residual)
            if stop_on_l2_increase && residual > prev_residual
                stop_reason = :l2_increase
                break
            end
            prev_residual = residual
        end

        current_lambda *= lambda_scale
    end

    _finish_progress!(progress, iterations_completed, niter)
    history = _build_history(residual_l2, lambda_history, iterations_completed, stop_reason)
    return _maybe_return_history(volume, history, return_history)
end

"""
    sart_tv(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, lambda::Real = 1f0, lambda_red::Real = 1f0, nonneg::Bool = true, tviter::Integer = 50, tvlambda::Real = 50f0, gpu_ids = nothing, return_history::Bool = false, compute_l2::Bool = false, stop_on_l2_increase::Bool = false, residual_projection_type = :siddon, show_progress::Bool = false, plan = nothing) -> volume

SART with Total Variation (TV) proximal denoising after each iteration.

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Number of complete iterations over all angles.
- `init = :zeros`: Initial volume guess (`:zeros`, `:fdk`, or custom 3D array, default: `:zeros`).
- `lambda::Real = 1.0f0`: Initial SART relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 1.0f0`: Relaxation factor reduction per iteration (default: `1.0f0`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `tviter::Integer = 50`: Number of TV proximal gradient denoising iterations per step (default: `50`).
- `tvlambda::Real = 50.0f0`: TV fidelity parameter passed to `tvdenoise!` on [0, 1]-normalized volume (default: `50.0f0`). Smaller values produce stronger TV smoothing, while larger values retain higher fidelity.
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Compute residual L2 norm per iteration (default: `false`).
- `stop_on_l2_increase::Bool = false`: Terminate iterations early if residual L2 norm increases (default: `false`).
- `residual_projection_type = :siddon`: Ray-tracing kernel for residual evaluation (default: `:siddon`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `Array{Float32, 3}`: Reconstructed 3D volume (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = sart_tv(proj, geo, 10; tvlambda = 50f0)
```
"""
function sart_tv(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return sart_tv!(volume, projection_checked, geo, niter; kwargs...)
end

"""
    sart_tv!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; lambda::Real = 1f0, lambda_red::Real = 1f0, nonneg::Bool = true, tviter::Integer = 50, tvlambda::Real = 50f0, gpu_ids = nothing, return_history::Bool = false, compute_l2::Bool = false, stop_on_l2_increase::Bool = false, residual_projection_type = :siddon, show_progress::Bool = false, plan = nothing) -> volume

In-place SART with TV regularization into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Number of complete iterations over all angles.
- `lambda::Real = 1.0f0`: Initial SART relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 1.0f0`: Relaxation factor reduction per iteration (default: `1.0f0`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `tviter::Integer = 50`: Number of TV proximal gradient denoising iterations per step (default: `50`).
- `tvlambda::Real = 50.0f0`: TV fidelity parameter passed to `tvdenoise!` on [0, 1]-normalized volume (default: `50.0f0`). Smaller values produce stronger TV smoothing, while larger values retain higher fidelity.
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Compute residual L2 norm per iteration (default: `false`).
- `stop_on_l2_increase::Bool = false`: Terminate iterations early if residual L2 norm increases (default: `false`).
- `residual_projection_type = :siddon`: Ray-tracing kernel for residual evaluation (default: `:siddon`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `volume`: The modified reconstructed volume array (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = zeros(Float32, volume_size(geo))
sart_tv!(vol, proj, geo, 10; tvlambda = 20f0)
```
"""
function sart_tv!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer;
    lambda::Real = 1f0,
    lambda_red::Real = 1f0,
    nonneg::Bool = true,
    tviter::Integer = 50,
    tvlambda::Real = 50f0,
    gpu_ids = nothing,
    return_history::Bool = false,
    compute_l2::Bool = false,
    stop_on_l2_increase::Bool = false,
    residual_projection_type = :siddon,
    show_progress::Bool = false,
    plan = nothing,
)
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))
    return _run_tv_art!(volume, projection, geo, niter;
        blocksize = 1,
        lambda,
        lambda_red,
        order_strategy = :ordered,
        nonneg,
        tviter,
        tvlambda,
        gpu_ids,
        return_history,
        compute_l2,
        stop_on_l2_increase,
        residual_projection_type,
        show_progress,
        plan,
    )
end

"""
    ossart_tv(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; blocksize::Integer = 20, init = :zeros, lambda::Real = 1f0, lambda_red::Real = 1f0, order_strategy = :ordered, nonneg::Bool = true, tviter::Integer = 50, tvlambda::Real = 50f0, gpu_ids = nothing, return_history::Bool = false, compute_l2::Bool = false, stop_on_l2_increase::Bool = false, residual_projection_type = :siddon, show_progress::Bool = false, plan = nothing) -> volume

Ordered Subsets SART with Total Variation (TV) proximal denoising after each epoch.

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Number of complete epochs through all subsets.
- `blocksize::Integer = 20`: Number of projection angles per subset (default: `20`).
- `init = :zeros`: Initial volume guess (`:zeros`, `:fdk`, or custom 3D array, default: `:zeros`).
- `lambda::Real = 1.0f0`: Initial relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 1.0f0`: Relaxation factor reduction per iteration (default: `1.0f0`).
- `order_strategy = :ordered`: Projection angle visiting order (`:ordered` or `:random`, default: `:ordered`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `tviter::Integer = 50`: Number of TV proximal gradient denoising iterations per epoch (default: `50`).
- `tvlambda::Real = 50.0f0`: TV fidelity parameter passed to `tvdenoise!` on [0, 1]-normalized volume (default: `50.0f0`). Smaller values produce stronger TV smoothing, while larger values retain higher fidelity.
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Compute residual L2 norm per iteration (default: `false`).
- `stop_on_l2_increase::Bool = false`: Terminate iterations early if residual L2 norm increases (default: `false`).
- `residual_projection_type = :siddon`: Ray-tracing kernel for residual evaluation (default: `:siddon`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `Array{Float32, 3}`: Reconstructed 3D volume (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = ossart_tv(proj, geo, 10; blocksize = 20, tvlambda = 50f0)
```
"""
function ossart_tv(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; blocksize::Integer = 20, init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return ossart_tv!(volume, projection_checked, geo, niter; blocksize, kwargs...)
end

"""
    ossart_tv!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; blocksize::Integer = 20, lambda::Real = 1f0, lambda_red::Real = 1f0, order_strategy = :ordered, nonneg::Bool = true, tviter::Integer = 50, tvlambda::Real = 50f0, gpu_ids = nothing, return_history::Bool = false, compute_l2::Bool = false, stop_on_l2_increase::Bool = false, residual_projection_type = :siddon, show_progress::Bool = false, plan = nothing) -> volume

In-place Ordered Subsets SART with TV regularization into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Number of complete epochs through all subsets.
- `blocksize::Integer = 20`: Number of projection angles per subset (default: `20`).
- `lambda::Real = 1.0f0`: Initial relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 1.0f0`: Relaxation factor reduction per iteration (default: `1.0f0`).
- `order_strategy = :ordered`: Projection angle visiting order (`:ordered` or `:random`, default: `:ordered`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `tviter::Integer = 50`: Number of TV proximal gradient denoising iterations per epoch (default: `50`).
- `tvlambda::Real = 50.0f0`: TV fidelity parameter passed to `tvdenoise!` on [0, 1]-normalized volume (default: `50.0f0`). Smaller values produce stronger TV smoothing, while larger values retain higher fidelity.
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Compute residual L2 norm per iteration (default: `false`).
- `stop_on_l2_increase::Bool = false`: Terminate iterations early if residual L2 norm increases (default: `false`).
- `residual_projection_type = :siddon`: Ray-tracing kernel for residual evaluation (default: `:siddon`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `volume`: The modified reconstructed volume array (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = zeros(Float32, volume_size(geo))
ossart_tv!(vol, proj, geo, 10; blocksize = 20, tvlambda = 20f0)
```
"""
function ossart_tv!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer;
    blocksize::Integer = 20,
    lambda::Real = 1f0,
    lambda_red::Real = 1f0,
    order_strategy = :ordered,
    nonneg::Bool = true,
    tviter::Integer = 50,
    tvlambda::Real = 50f0,
    gpu_ids = nothing,
    return_history::Bool = false,
    compute_l2::Bool = false,
    stop_on_l2_increase::Bool = false,
    residual_projection_type = :siddon,
    show_progress::Bool = false,
    plan = nothing,
)
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))
    return _run_tv_art!(volume, projection, geo, niter;
        blocksize,
        lambda,
        lambda_red,
        order_strategy,
        nonneg,
        tviter,
        tvlambda,
        gpu_ids,
        return_history,
        compute_l2,
        stop_on_l2_increase,
        residual_projection_type,
        show_progress,
        plan,
    )
end

function _min_tv_regularize!(dst::StridedArray{Float32, 3}, src::StridedArray{Float32, 3}, tvalpha::Float32, tviter::Integer, gpu_ids, prior, prior_ratio::Float32)
    minTV!(dst, src; alpha = tvalpha, maxiter = tviter, gpu_ids)
    return dst
end

function _piccs_regularize!(dst::StridedArray{Float32, 3}, src::StridedArray{Float32, 3}, tvalpha::Float32, tviter::Integer, gpu_ids, prior::StridedArray{Float32, 3}, prior_ratio::Float32)
    piccsTV!(dst, src, prior; alpha = tvalpha, ratio = prior_ratio, maxiter = tviter, gpu_ids)
    return dst
end

function _awmin_tv_regularize!(dst::StridedArray{Float32, 3}, src::StridedArray{Float32, 3}, tvalpha::Float32, tviter::Integer, gpu_ids, delta::Float32, prior_ratio::Float32)
    awminTV!(dst, src; alpha = tvalpha, maxiter = tviter, delta, gpu_ids)
    return dst
end

function _run_pocs!(
    volume::StridedArray{Float32, 3},
    projection::StridedArray{Float32, 3},
    geo::Geometry,
    niter::Integer,
    regularize!;
    blocksize::Integer,
    order_strategy,
    lambda::Real,
    lambda_red::Real,
    alpha::Real,
    alpha_red::Real,
    rmax::Real,
    maxl2err,
    tviter::Integer,
    nonneg::Bool,
    gpu_ids,
    return_history::Bool,
    show_progress::Bool,
    prior = nothing,
    prior_ratio::Real = 0.2f0,
    plan = nothing,
)
    niter >= 0 || throw(ArgumentError("niter must be >= 0"))
    projection_count(geo) > 0 || throw(ArgumentError("angles is empty"))

    plan = _resolve_art_plan(plan, projection, geo, Int(blocksize), order_strategy; gpu_ids, need_full_projection = true)
    residual_seed = if maxl2err === nothing
        cached = plan.residual_seed_cache[]
        if isnan(cached)
            seed = _art_extra_volume(plan, 3, geo)
            if geo.mode === :parallel
                fbp!(seed, projection, geo; gpu_ids)
            else
                fdk!(seed, projection, geo; gpu_ids)
            end
            cached = _projection_residual_l2!(plan.workspace.full_proj_tmp::Array{Float32, 3}, projection, seed, geo, :siddon, gpu_ids) * 0.2f0
            plan.residual_seed_cache[] = cached
        end
        cached
    else
        Float32(maxl2err)
    end

    prox_tmp = _art_extra_volume(plan, 1, geo)
    prev_tmp = _art_extra_volume(plan, 2, geo)
    residual_l2 = Float32[]
    lambda_history = Float32[]
    current_lambda = Float32(lambda)
    dtvg = 0f0
    beta = Float32(lambda)
    stop_reason = :max_iterations
    iterations_completed = 0
    prior_ratio32 = Float32(prior_ratio)
    progress = _make_progress(niter, show_progress, "POCS: ")

    for iter in 1:niter
        copyto!(prev_tmp, volume)
        push!(lambda_history, current_lambda)
        _art_iteration!(volume, projection, plan, current_lambda, nonneg, :siddon, :fdk, gpu_ids)
        copyto!(prox_tmp, volume)

        dd = _projection_residual_l2!(plan.workspace.full_proj_tmp::Array{Float32, 3}, projection, prox_tmp, geo, :siddon, gpu_ids)
        push!(residual_l2, dd)

        dp_sq = 0.0
        @inbounds for i in eachindex(prox_tmp, prev_tmp)
            diff = Float64(prox_tmp[i] - prev_tmp[i])
            dp_sq += diff * diff
        end
        dp = Float32(sqrt(dp_sq))
        iter == 1 && (dtvg = Float32(alpha) * dp)

        regularize!(volume, prox_tmp, dtvg, tviter, gpu_ids, prior, prior_ratio32)
        nonneg && clamp!(volume, 0f0, Inf32)

        dg_sq = 0.0
        dot_gd = 0.0
        @inbounds for i in eachindex(volume, prox_tmp)
            diff_g = Float64(volume[i] - prox_tmp[i])
            dg_sq += diff_g * diff_g
            dot_gd += diff_g * Float64(prox_tmp[i] - prev_tmp[i])
        end
        dg = Float32(sqrt(dg_sq))

        if dg > Float32(rmax) * dp && dd > residual_seed
            dtvg *= Float32(alpha_red)
        end

        beta *= Float32(lambda_red)
        iterations_completed = iter
        _next_progress!(progress, iter, niter)
        c = (dg > 0f0 && dp > 0f0) ? Float32(dot_gd / max(Float64(dg * dp), 1e-6)) : 0f0
        if (c < -0.99f0 && dd <= residual_seed) || beta < 0.005f0 || iter >= niter
            stop_reason = (c < -0.99f0 && dd <= residual_seed) ? :asd_pocs_criterion : :max_iterations
            break
        end
        current_lambda *= Float32(lambda_red)
    end

    _finish_progress!(progress, iterations_completed, niter)
    history = _build_history(residual_l2, lambda_history, iterations_completed, stop_reason)
    return _maybe_return_history(volume, history, return_history)
end

"""
    asd_pocs(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, lambda::Real = 1f0, lambda_red::Real = 0.99f0, alpha::Real = 0.002f0, alpha_red::Real = 0.95f0, rmax::Real = 0.95f0, maxl2err = nothing, tviter::Integer = 20, nonneg::Bool = true, gpu_ids = nothing, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

Adaptive Steepest Descent - Projection Onto Convex Sets (ASD-POCS).

Alternates between ART projection data consistency steps and TV-gradient steepest descent, adaptively balancing data error and total variation.

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Maximum number of iterations.
- `init = :zeros`: Initial volume guess (`:zeros`, `:fdk`, or custom 3D array, default: `:zeros`).
- `lambda::Real = 1.0f0`: Initial ART relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 0.99f0`: ART relaxation parameter reduction per iteration (default: `0.99f0`).
- `alpha::Real = 0.002f0`: TV gradient step size fraction (default: `0.002f0`).
- `alpha_red::Real = 0.95f0`: TV step size reduction factor (default: `0.95f0`).
- `rmax::Real = 0.95f0`: Ratio limit for TV step versus ART step (default: `0.95f0`).
- `maxl2err = nothing`: Residual error threshold for early stopping or `nothing` for auto-threshold (default: `nothing`).
- `tviter::Integer = 20`: Sub-iterations of TV gradient descent per ART step (default: `20`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `Array{Float32, 3}`: Reconstructed 3D volume (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Reference
- Sidky, E. Y., & Pan, X., "Constrained TV minimization in CBCT", Physics in Medicine & Biology, 53(17), 2008.

# Examples
```julia
vol = asd_pocs(proj, geo, 20)
```
"""
function asd_pocs(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return asd_pocs!(volume, projection_checked, geo, niter; kwargs...)
end

"""
    asd_pocs!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; lambda::Real = 1f0, lambda_red::Real = 0.99f0, alpha::Real = 0.002f0, alpha_red::Real = 0.95f0, rmax::Real = 0.95f0, maxl2err = nothing, tviter::Integer = 20, nonneg::Bool = true, gpu_ids = nothing, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

In-place ASD-POCS reconstruction into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Maximum number of iterations.
- `lambda::Real = 1.0f0`: Initial ART relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 0.99f0`: ART relaxation parameter reduction per iteration (default: `0.99f0`).
- `alpha::Real = 0.002f0`: TV gradient step size fraction (default: `0.002f0`).
- `alpha_red::Real = 0.95f0`: TV step size reduction factor (default: `0.95f0`).
- `rmax::Real = 0.95f0`: Ratio limit for TV step versus ART step (default: `0.95f0`).
- `maxl2err = nothing`: Residual error threshold for early stopping or `nothing` for auto-threshold (default: `nothing`).
- `tviter::Integer = 20`: Sub-iterations of TV gradient descent per ART step (default: `20`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `volume`: The modified reconstructed volume array (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = zeros(Float32, volume_size(geo))
asd_pocs!(vol, proj, geo, 20)
```
"""
function asd_pocs!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer;
    lambda::Real = 1f0,
    lambda_red::Real = 0.99f0,
    alpha::Real = 0.002f0,
    alpha_red::Real = 0.95f0,
    rmax::Real = 0.95f0,
    maxl2err = nothing,
    tviter::Integer = 20,
    nonneg::Bool = true,
    gpu_ids = nothing,
    return_history::Bool = false,
    show_progress::Bool = false,
    plan = nothing,
)
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))
    return _run_pocs!(volume, projection, geo, niter, _min_tv_regularize!;
        blocksize = 1,
        order_strategy = :ordered,
        lambda,
        lambda_red,
        alpha,
        alpha_red,
        rmax,
        maxl2err,
        tviter,
        nonneg,
        gpu_ids,
        return_history,
        show_progress,
        plan,
    )
end

"""
    os_asd_pocs(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; blocksize::Integer = 20, init = :zeros, lambda::Real = 1f0, lambda_red::Real = 0.99f0, alpha::Real = 0.002f0, alpha_red::Real = 0.95f0, rmax::Real = 0.95f0, maxl2err = nothing, tviter::Integer = 20, order_strategy = :ordered, nonneg::Bool = true, gpu_ids = nothing, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

Ordered Subsets ASD-POCS, speeding up data projection steps using angular subsets.

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Maximum number of iterations.
- `blocksize::Integer = 20`: Number of projection angles per subset (default: `20`).
- `init = :zeros`: Initial volume guess (`:zeros`, `:fdk`, or custom 3D array, default: `:zeros`).
- `lambda::Real = 1.0f0`: Initial ART relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 0.99f0`: ART relaxation parameter reduction per iteration (default: `0.99f0`).
- `alpha::Real = 0.002f0`: TV gradient step size fraction (default: `0.002f0`).
- `alpha_red::Real = 0.95f0`: TV step size reduction factor (default: `0.95f0`).
- `rmax::Real = 0.95f0`: Ratio limit for TV step versus ART step (default: `0.95f0`).
- `maxl2err = nothing`: Residual error threshold for early stopping or `nothing` for auto-threshold (default: `nothing`).
- `tviter::Integer = 20`: Sub-iterations of TV gradient descent per ART step (default: `20`).
- `order_strategy = :ordered`: Projection angle visiting order (`:ordered` or `:random`, default: `:ordered`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `Array{Float32, 3}`: Reconstructed 3D volume (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = os_asd_pocs(proj, geo, 20; blocksize = 20)
```
"""
function os_asd_pocs(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; blocksize::Integer = 20, init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return os_asd_pocs!(volume, projection_checked, geo, niter; blocksize, kwargs...)
end

"""
    os_asd_pocs!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; blocksize::Integer = 20, lambda::Real = 1f0, lambda_red::Real = 0.99f0, alpha::Real = 0.002f0, alpha_red::Real = 0.95f0, rmax::Real = 0.95f0, maxl2err = nothing, tviter::Integer = 20, order_strategy = :ordered, nonneg::Bool = true, gpu_ids = nothing, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

In-place Ordered Subsets ASD-POCS reconstruction into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Maximum number of iterations.
- `blocksize::Integer = 20`: Number of projection angles per subset (default: `20`).
- `lambda::Real = 1.0f0`: Initial ART relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 0.99f0`: ART relaxation parameter reduction per iteration (default: `0.99f0`).
- `alpha::Real = 0.002f0`: TV gradient step size fraction (default: `0.002f0`).
- `alpha_red::Real = 0.95f0`: TV step size reduction factor (default: `0.95f0`).
- `rmax::Real = 0.95f0`: Ratio limit for TV step versus ART step (default: `0.95f0`).
- `maxl2err = nothing`: Residual error threshold for early stopping or `nothing` for auto-threshold (default: `nothing`).
- `tviter::Integer = 20`: Sub-iterations of TV gradient descent per ART step (default: `20`).
- `order_strategy = :ordered`: Projection angle visiting order (`:ordered` or `:random`, default: `:ordered`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `volume`: The modified reconstructed volume array (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = zeros(Float32, volume_size(geo))
os_asd_pocs!(vol, proj, geo, 20; blocksize = 20)
```
"""
function os_asd_pocs!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer;
    blocksize::Integer = 20,
    lambda::Real = 1f0,
    lambda_red::Real = 0.99f0,
    alpha::Real = 0.002f0,
    alpha_red::Real = 0.95f0,
    rmax::Real = 0.95f0,
    maxl2err = nothing,
    tviter::Integer = 20,
    order_strategy = :ordered,
    nonneg::Bool = true,
    gpu_ids = nothing,
    return_history::Bool = false,
    show_progress::Bool = false,
    plan = nothing,
)
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))
    return _run_pocs!(volume, projection, geo, niter, _min_tv_regularize!;
        blocksize,
        order_strategy,
        lambda,
        lambda_red,
        alpha,
        alpha_red,
        rmax,
        maxl2err,
        tviter,
        nonneg,
        gpu_ids,
        return_history,
        show_progress,
        plan,
    )
end

"""
    awasd_pocs(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, lambda::Real = 1f0, lambda_red::Real = 0.99f0, alpha::Real = 0.002f0, alpha_red::Real = 0.95f0, rmax::Real = 0.95f0, maxl2err = nothing, tviter::Integer = 20, delta::Real = -0.005f0, nonneg::Bool = true, gpu_ids = nothing, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

Adaptive Weighted ASD-POCS (AwASD-POCS) with edge-preserving adaptive total variation.

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Maximum number of iterations.
- `init = :zeros`: Initial volume guess (`:zeros`, `:fdk`, or custom 3D array, default: `:zeros`).
- `lambda::Real = 1.0f0`: Initial ART relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 0.99f0`: ART relaxation parameter reduction per iteration (default: `0.99f0`).
- `alpha::Real = 0.002f0`: TV gradient step size fraction (default: `0.002f0`).
- `alpha_red::Real = 0.95f0`: TV step size reduction factor (default: `0.95f0`).
- `rmax::Real = 0.95f0`: Ratio limit for TV step versus ART step (default: `0.95f0`).
- `maxl2err = nothing`: Residual error threshold for early stopping or `nothing` for auto-threshold (default: `nothing`).
- `tviter::Integer = 20`: Sub-iterations of TV gradient descent per ART step (default: `20`).
- `delta::Real = -0.005f0`: Edge-weight scale parameter or negative for automated percentile estimation (default: `-0.005f0`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `Array{Float32, 3}`: Reconstructed 3D volume (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = awasd_pocs(proj, geo, 20)
```
"""
function awasd_pocs(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return awasd_pocs!(volume, projection_checked, geo, niter; kwargs...)
end

"""
    awasd_pocs!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; lambda::Real = 1f0, lambda_red::Real = 0.99f0, alpha::Real = 0.002f0, alpha_red::Real = 0.95f0, rmax::Real = 0.95f0, maxl2err = nothing, tviter::Integer = 20, delta::Real = -0.005f0, nonneg::Bool = true, gpu_ids = nothing, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

In-place Adaptive Weighted ASD-POCS reconstruction into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Maximum number of iterations.
- `lambda::Real = 1.0f0`: Initial ART relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 0.99f0`: ART relaxation parameter reduction per iteration (default: `0.99f0`).
- `alpha::Real = 0.002f0`: TV gradient step size fraction (default: `0.002f0`).
- `alpha_red::Real = 0.95f0`: TV step size reduction factor (default: `0.95f0`).
- `rmax::Real = 0.95f0`: Ratio limit for TV step versus ART step (default: `0.95f0`).
- `maxl2err = nothing`: Residual error threshold for early stopping or `nothing` for auto-threshold (default: `nothing`).
- `tviter::Integer = 20`: Sub-iterations of TV gradient descent per ART step (default: `20`).
- `delta::Real = -0.005f0`: Edge-weight scale parameter or negative for automated percentile estimation (default: `-0.005f0`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `volume`: The modified reconstructed volume array (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = zeros(Float32, volume_size(geo))
awasd_pocs!(vol, proj, geo, 20)
```
"""
function awasd_pocs!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer;
    lambda::Real = 1f0,
    lambda_red::Real = 0.99f0,
    alpha::Real = 0.002f0,
    alpha_red::Real = 0.95f0,
    rmax::Real = 0.95f0,
    maxl2err = nothing,
    tviter::Integer = 20,
    delta::Real = -0.005f0,
    nonneg::Bool = true,
    gpu_ids = nothing,
    return_history::Bool = false,
    show_progress::Bool = false,
    plan = nothing,
)
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))
    return _run_pocs!(volume, projection, geo, niter, _awmin_tv_regularize!;
        blocksize = 1,
        order_strategy = :ordered,
        lambda,
        lambda_red,
        alpha,
        alpha_red,
        rmax,
        maxl2err,
        tviter,
        nonneg,
        gpu_ids,
        return_history,
        show_progress,
        prior = Float32(delta),
        plan,
    )
end

"""
    os_awasd_pocs(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; blocksize::Integer = 20, init = :zeros, lambda::Real = 1f0, lambda_red::Real = 0.99f0, alpha::Real = 0.002f0, alpha_red::Real = 0.95f0, rmax::Real = 0.95f0, maxl2err = nothing, tviter::Integer = 20, delta::Real = -0.005f0, order_strategy = :ordered, nonneg::Bool = true, gpu_ids = nothing, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

Ordered Subsets Adaptive Weighted ASD-POCS.

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Maximum number of iterations.
- `blocksize::Integer = 20`: Number of projection angles per subset (default: `20`).
- `init = :zeros`: Initial volume guess (`:zeros`, `:fdk`, or custom 3D array, default: `:zeros`).
- `lambda::Real = 1.0f0`: Initial ART relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 0.99f0`: ART relaxation parameter reduction per iteration (default: `0.99f0`).
- `alpha::Real = 0.002f0`: TV gradient step size fraction (default: `0.002f0`).
- `alpha_red::Real = 0.95f0`: TV step size reduction factor (default: `0.95f0`).
- `rmax::Real = 0.95f0`: Ratio limit for TV step versus ART step (default: `0.95f0`).
- `maxl2err = nothing`: Residual error threshold for early stopping or `nothing` for auto-threshold (default: `nothing`).
- `tviter::Integer = 20`: Sub-iterations of TV gradient descent per ART step (default: `20`).
- `delta::Real = -0.005f0`: Edge-weight scale parameter or negative for automated percentile estimation (default: `-0.005f0`).
- `order_strategy = :ordered`: Projection angle visiting order (`:ordered` or `:random`, default: `:ordered`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `Array{Float32, 3}`: Reconstructed 3D volume (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = os_awasd_pocs(proj, geo, 20; blocksize = 20)
```
"""
function os_awasd_pocs(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; blocksize::Integer = 20, init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return os_awasd_pocs!(volume, projection_checked, geo, niter; blocksize, kwargs...)
end

"""
    os_awasd_pocs!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; blocksize::Integer = 20, lambda::Real = 1f0, lambda_red::Real = 0.99f0, alpha::Real = 0.002f0, alpha_red::Real = 0.95f0, rmax::Real = 0.95f0, maxl2err = nothing, tviter::Integer = 20, delta::Real = -0.005f0, order_strategy = :ordered, nonneg::Bool = true, gpu_ids = nothing, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

In-place Ordered Subsets Adaptive Weighted ASD-POCS reconstruction into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Maximum number of iterations.
- `blocksize::Integer = 20`: Number of projection angles per subset (default: `20`).
- `lambda::Real = 1.0f0`: Initial ART relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 0.99f0`: ART relaxation parameter reduction per iteration (default: `0.99f0`).
- `alpha::Real = 0.002f0`: TV gradient step size fraction (default: `0.002f0`).
- `alpha_red::Real = 0.95f0`: TV step size reduction factor (default: `0.95f0`).
- `rmax::Real = 0.95f0`: Ratio limit for TV step versus ART step (default: `0.95f0`).
- `maxl2err = nothing`: Residual error threshold for early stopping or `nothing` for auto-threshold (default: `nothing`).
- `tviter::Integer = 20`: Sub-iterations of TV gradient descent per ART step (default: `20`).
- `delta::Real = -0.005f0`: Edge-weight scale parameter or negative for automated percentile estimation (default: `-0.005f0`).
- `order_strategy = :ordered`: Projection angle visiting order (`:ordered` or `:random`, default: `:ordered`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `volume`: The modified reconstructed volume array (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = zeros(Float32, volume_size(geo))
os_awasd_pocs!(vol, proj, geo, 20; blocksize = 20)
```
"""
function os_awasd_pocs!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer;
    blocksize::Integer = 20,
    lambda::Real = 1f0,
    lambda_red::Real = 0.99f0,
    alpha::Real = 0.002f0,
    alpha_red::Real = 0.95f0,
    rmax::Real = 0.95f0,
    maxl2err = nothing,
    tviter::Integer = 20,
    delta::Real = -0.005f0,
    order_strategy = :ordered,
    nonneg::Bool = true,
    gpu_ids = nothing,
    return_history::Bool = false,
    show_progress::Bool = false,
    plan = nothing,
)
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))
    return _run_pocs!(volume, projection, geo, niter, _awmin_tv_regularize!;
        blocksize,
        order_strategy,
        lambda,
        lambda_red,
        alpha,
        alpha_red,
        rmax,
        maxl2err,
        tviter,
        nonneg,
        gpu_ids,
        return_history,
        show_progress,
        prior = Float32(delta),
        plan,
    )
end

"""
    piccs(projection::AbstractArray{Float32, 3}, geo::Geometry, prior::AbstractArray{Float32, 3}, niter::Integer; init = :zeros, lambda::Real = 1f0, lambda_red::Real = 0.99f0, alpha::Real = 0.002f0, alpha_red::Real = 0.95f0, rmax::Real = 0.95f0, maxl2err = nothing, tviter::Integer = 20, prior_ratio::Real = 0.2f0, nonneg::Bool = true, gpu_ids = nothing, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

Prior Image Constrained Compressed Sensing (PICCS).

Reconstructs few-view or noisy CT data constrained by a high-quality prior volume image (e.g. from an earlier prior scan).

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `prior::AbstractArray{Float32, 3}`: Prior volume array `(nx, ny, nz)`.
- `niter::Integer`: Number of iterations.
- `init = :zeros`: Initial volume guess (`:zeros`, `:fdk`, or custom 3D array, default: `:zeros`).
- `lambda::Real = 1.0f0`: Initial ART relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 0.99f0`: ART relaxation parameter reduction per iteration (default: `0.99f0`).
- `alpha::Real = 0.002f0`: TV gradient step size fraction (default: `0.002f0`).
- `alpha_red::Real = 0.95f0`: TV step size reduction factor (default: `0.95f0`).
- `rmax::Real = 0.95f0`: Ratio limit for TV step versus ART step (default: `0.95f0`).
- `maxl2err = nothing`: Residual error threshold for early stopping or `nothing` for auto-threshold (default: `nothing`).
- `tviter::Integer = 20`: Sub-iterations of TV gradient descent per ART step (default: `20`).
- `prior_ratio::Real = 0.2f0`: Fraction weight given to prior image TV versus standard TV (default: `0.2f0`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `Array{Float32, 3}`: Reconstructed 3D volume (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Reference
- Chen, G.-H., Tang, J., & Leng, S., "Prior image constrained compressed sensing (PICCS)", Medical Physics, 35(2), 2008.

# Examples
```julia
vol = piccs(sparse_proj, geo, prior_vol, 20; prior_ratio = 0.2f0)
```
"""
function piccs(projection::AbstractArray{Float32, 3}, geo::Geometry, prior::AbstractArray{Float32, 3}, niter::Integer; init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return piccs!(volume, projection_checked, geo, prior, niter; kwargs...)
end

"""
    piccs!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, prior::AbstractArray{Float32, 3}, niter::Integer; lambda::Real = 1f0, lambda_red::Real = 0.99f0, alpha::Real = 0.002f0, alpha_red::Real = 0.95f0, rmax::Real = 0.95f0, maxl2err = nothing, tviter::Integer = 20, prior_ratio::Real = 0.2f0, nonneg::Bool = true, gpu_ids = nothing, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

In-place Prior Image Constrained Compressed Sensing (PICCS) reconstruction into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `prior::AbstractArray{Float32, 3}`: Prior volume array `(nx, ny, nz)`.
- `niter::Integer`: Number of iterations.
- `lambda::Real = 1.0f0`: Initial ART relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 0.99f0`: ART relaxation parameter reduction per iteration (default: `0.99f0`).
- `alpha::Real = 0.002f0`: TV gradient step size fraction (default: `0.002f0`).
- `alpha_red::Real = 0.95f0`: TV step size reduction factor (default: `0.95f0`).
- `rmax::Real = 0.95f0`: Ratio limit for TV step versus ART step (default: `0.95f0`).
- `maxl2err = nothing`: Residual error threshold for early stopping or `nothing` for auto-threshold (default: `nothing`).
- `tviter::Integer = 20`: Sub-iterations of TV gradient descent per ART step (default: `20`).
- `prior_ratio::Real = 0.2f0`: Fraction weight given to prior image TV versus standard TV (default: `0.2f0`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `volume`: The modified reconstructed volume array (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = zeros(Float32, volume_size(geo))
piccs!(vol, sparse_proj, geo, prior_vol, 20; prior_ratio = 0.2f0)
```
"""
function piccs!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, prior::AbstractArray{Float32, 3}, niter::Integer;
    lambda::Real = 1f0,
    lambda_red::Real = 0.99f0,
    alpha::Real = 0.002f0,
    alpha_red::Real = 0.95f0,
    rmax::Real = 0.95f0,
    maxl2err = nothing,
    tviter::Integer = 20,
    prior_ratio::Real = 0.2f0,
    nonneg::Bool = true,
    gpu_ids = nothing,
    return_history::Bool = false,
    show_progress::Bool = false,
    plan = nothing,
)
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))
    prior = _require_canonical_array(prior, "prior", volume_size(geo))
    return _run_pocs!(volume, projection, geo, niter, _piccs_regularize!;
        blocksize = 1,
        order_strategy = :ordered,
        lambda,
        lambda_red,
        alpha,
        alpha_red,
        rmax,
        maxl2err,
        tviter,
        nonneg,
        gpu_ids,
        return_history,
        show_progress,
        prior,
        prior_ratio,
        plan,
    )
end

"""
    os_piccs(projection::AbstractArray{Float32, 3}, geo::Geometry, prior::AbstractArray{Float32, 3}, niter::Integer; blocksize::Integer = 20, init = :zeros, lambda::Real = 1f0, lambda_red::Real = 0.99f0, alpha::Real = 0.002f0, alpha_red::Real = 0.95f0, rmax::Real = 0.95f0, maxl2err = nothing, tviter::Integer = 20, prior_ratio::Real = 0.2f0, order_strategy = :ordered, nonneg::Bool = true, gpu_ids = nothing, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

Ordered Subsets Prior Image Constrained Compressed Sensing (OS-PICCS).

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `prior::AbstractArray{Float32, 3}`: Prior volume array `(nx, ny, nz)`.
- `niter::Integer`: Number of iterations.
- `blocksize::Integer = 20`: Number of projection angles per subset (default: `20`).
- `init = :zeros`: Initial volume guess (`:zeros`, `:fdk`, or custom 3D array, default: `:zeros`).
- `lambda::Real = 1.0f0`: Initial ART relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 0.99f0`: ART relaxation parameter reduction per iteration (default: `0.99f0`).
- `alpha::Real = 0.002f0`: TV gradient step size fraction (default: `0.002f0`).
- `alpha_red::Real = 0.95f0`: TV step size reduction factor (default: `0.95f0`).
- `rmax::Real = 0.95f0`: Ratio limit for TV step versus ART step (default: `0.95f0`).
- `maxl2err = nothing`: Residual error threshold for early stopping or `nothing` for auto-threshold (default: `nothing`).
- `tviter::Integer = 20`: Sub-iterations of TV gradient descent per ART step (default: `20`).
- `prior_ratio::Real = 0.2f0`: Fraction weight given to prior image TV versus standard TV (default: `0.2f0`).
- `order_strategy = :ordered`: Projection angle visiting order (`:ordered` or `:random`, default: `:ordered`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `Array{Float32, 3}`: Reconstructed 3D volume (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = os_piccs(sparse_proj, geo, prior_vol, 20; blocksize = 20, prior_ratio = 0.2f0)
```
"""
function os_piccs(projection::AbstractArray{Float32, 3}, geo::Geometry, prior::AbstractArray{Float32, 3}, niter::Integer; blocksize::Integer = 20, init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return os_piccs!(volume, projection_checked, geo, prior, niter; blocksize, kwargs...)
end

"""
    os_piccs!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, prior::AbstractArray{Float32, 3}, niter::Integer; blocksize::Integer = 20, lambda::Real = 1f0, lambda_red::Real = 0.99f0, alpha::Real = 0.002f0, alpha_red::Real = 0.95f0, rmax::Real = 0.95f0, maxl2err = nothing, tviter::Integer = 20, prior_ratio::Real = 0.2f0, order_strategy = :ordered, nonneg::Bool = true, gpu_ids = nothing, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

In-place Ordered Subsets Prior Image Constrained Compressed Sensing (OS-PICCS) reconstruction into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `prior::AbstractArray{Float32, 3}`: Prior volume array `(nx, ny, nz)`.
- `niter::Integer`: Number of iterations.
- `blocksize::Integer = 20`: Number of projection angles per subset (default: `20`).
- `lambda::Real = 1.0f0`: Initial ART relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 0.99f0`: ART relaxation parameter reduction per iteration (default: `0.99f0`).
- `alpha::Real = 0.002f0`: TV gradient step size fraction (default: `0.002f0`).
- `alpha_red::Real = 0.95f0`: TV step size reduction factor (default: `0.95f0`).
- `rmax::Real = 0.95f0`: Ratio limit for TV step versus ART step (default: `0.95f0`).
- `maxl2err = nothing`: Residual error threshold for early stopping or `nothing` for auto-threshold (default: `nothing`).
- `tviter::Integer = 20`: Sub-iterations of TV gradient descent per ART step (default: `20`).
- `prior_ratio::Real = 0.2f0`: Fraction weight given to prior image TV versus standard TV (default: `0.2f0`).
- `order_strategy = :ordered`: Projection angle visiting order (`:ordered` or `:random`, default: `:ordered`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `volume`: The modified reconstructed volume array (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = zeros(Float32, volume_size(geo))
os_piccs!(vol, sparse_proj, geo, prior_vol, 20; blocksize = 20, prior_ratio = 0.2f0)
```
"""
function os_piccs!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, prior::AbstractArray{Float32, 3}, niter::Integer;
    blocksize::Integer = 20,
    lambda::Real = 1f0,
    lambda_red::Real = 0.99f0,
    alpha::Real = 0.002f0,
    alpha_red::Real = 0.95f0,
    rmax::Real = 0.95f0,
    maxl2err = nothing,
    tviter::Integer = 20,
    prior_ratio::Real = 0.2f0,
    order_strategy = :ordered,
    nonneg::Bool = true,
    gpu_ids = nothing,
    return_history::Bool = false,
    show_progress::Bool = false,
    plan = nothing,
)
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))
    prior = _require_canonical_array(prior, "prior", volume_size(geo))
    return _run_pocs!(volume, projection, geo, niter, _piccs_regularize!;
        blocksize,
        order_strategy,
        lambda,
        lambda_red,
        alpha,
        alpha_red,
        rmax,
        maxl2err,
        tviter,
        nonneg,
        gpu_ids,
        return_history,
        show_progress,
        prior,
        prior_ratio,
        plan,
    )
end

function _initialize_mlem_volume!(volume::StridedArray{Float32, 3}, projection::StridedArray{Float32, 3}, geo::Geometry, init; gpu_ids = nothing)
    if init === nothing || init === :ones
        fill!(volume, 1f0)
    elseif init === :zeros
        fill!(volume, 0f0)
    elseif init === :fdk
        fdk!(volume, projection, geo; gpu_ids)
        clamp!(volume, 0f0, Inf32)
    elseif init isa AbstractArray{Float32, 3}
        _require_canonical_array(init, "init", volume_size(geo))
        copyto!(volume, init)
    else
        throw(ArgumentError("init must be nothing, :ones, :zeros, :fdk, or a 3D array"))
    end
    return volume
end

"""
    mlem(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = nothing, projection_type::Symbol = :siddon, backprojection_type::Symbol = :matched, gpu_ids = nothing, nonneg::Bool = true, return_history::Bool = false, compute_l2::Bool = false, stop_on_l2_increase::Bool = false, residual_projection_type::Symbol = projection_type, show_progress::Bool = false, workspace = nothing) -> volume

Maximum Likelihood Expectation Maximization (MLEM) reconstruction for emission tomography / non-negative Poisson data.

Multiplicative update rule:
```math
x^{(k+1)} = \\frac{x^{(k)}}{A^T \\mathbf{1}} A^T \\left( \\frac{b}{A x^{(k)}} \\right)
```

# Arguments
- `projection::AbstractArray{Float32, 3}`: Measured projection data (must be non-negative).
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Number of MLEM iterations.
- `init = nothing`: Initial volume guess (`nothing` or `:ones` defaults to 1.0, `:zeros`, `:fdk`, or custom 3D array, default: `nothing`).
- `projection_type::Symbol = :siddon`: Forward projection ray-tracing kernel (`:siddon` or `:interpolated`, default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:fdk`, default: `:matched`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Compute residual L2 norm per iteration (default: `false`).
- `stop_on_l2_increase::Bool = false`: Terminate iterations early if residual L2 norm increases (default: `false`).
- `residual_projection_type::Symbol = projection_type`: Kernel for residual evaluation (default: `projection_type`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `workspace = nothing`: Optional preallocated `FirstOrderWorkspace` (default: `nothing`).

# Returns
- `Array{Float32, 3}`: Reconstructed 3D volume (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Reference
- Shepp, L. A., & Vardi, Y., "Maximum likelihood reconstruction for emission tomography", IEEE TMI, 1(2), 1982.

# Examples
```julia
vol = mlem(proj, geo, 20)
```
"""
function mlem(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = nothing, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_mlem_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return mlem!(volume, projection_checked, geo, niter; kwargs...)
end

"""
    mlem!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; projection_type::Symbol = :siddon, backprojection_type::Symbol = :matched, gpu_ids = nothing, nonneg::Bool = true, return_history::Bool = false, compute_l2::Bool = false, stop_on_l2_increase::Bool = false, residual_projection_type::Symbol = projection_type, show_progress::Bool = false, workspace = nothing) -> volume

In-place Maximum Likelihood Expectation Maximization (MLEM) reconstruction into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: Measured projection data (must be non-negative).
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Number of MLEM iterations.
- `projection_type::Symbol = :siddon`: Forward projection ray-tracing kernel (`:siddon` or `:interpolated`, default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:fdk`, default: `:matched`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Compute residual L2 norm per iteration (default: `false`).
- `stop_on_l2_increase::Bool = false`: Terminate iterations early if residual L2 norm increases (default: `false`).
- `residual_projection_type::Symbol = projection_type`: Kernel for residual evaluation (default: `projection_type`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `workspace = nothing`: Optional preallocated `FirstOrderWorkspace` (default: `nothing`).

# Returns
- `volume`: The modified reconstructed volume array (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = ones(Float32, volume_size(geo))
mlem!(vol, proj, geo, 20)
```
"""
function mlem!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer;
    projection_type::Symbol = :siddon,
    backprojection_type::Symbol = :matched,
    gpu_ids = nothing,
    nonneg::Bool = true,
    return_history::Bool = false,
    compute_l2::Bool = false,
    stop_on_l2_increase::Bool = false,
    residual_projection_type::Symbol = projection_type,
    show_progress::Bool = false,
    workspace = nothing,
)
    niter >= 0 || throw(ArgumentError("niter must be >= 0"))
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))
    workspace = _resolve_first_order_workspace(workspace, geo; need_full_projection = compute_l2 || stop_on_l2_increase || return_history)

    weights = Atb(fill(1f0, projection_size(geo)), geo; backprojection_type, gpu_ids)
    @inbounds for i in eachindex(weights)
        weights[i] <= 0f0 && (weights[i] = Inf32)
    end

    residual_l2 = Float32[]
    lambda_history = fill(1f0, niter)
    prev_residual = Inf32
    stop_reason = :max_iterations
    iterations_completed = 0
    progress = _make_progress(niter, show_progress, "MLEM: ")

    for iter in 1:niter
        nonneg && clamp!(volume, 0f0, Inf32)
        Ax!(workspace.proj_tmp, volume, geo; projection_type, gpu_ids)
        @inbounds for i in eachindex(workspace.proj_tmp, projection)
            den = workspace.proj_tmp[i]
            workspace.proj_tmp[i] = iszero(den) ? 0f0 : projection[i] / den
        end
        Atb!(workspace.backproj_tmp, workspace.proj_tmp, geo; backprojection_type, gpu_ids)
        @inbounds for i in eachindex(volume, workspace.backproj_tmp, weights)
            volume[i] *= workspace.backproj_tmp[i] / weights[i]
        end
        nonneg && clamp!(volume, 0f0, Inf32)

        iterations_completed = iter
        _next_progress!(progress, iter, niter)
        if workspace.full_proj_tmp !== nothing
            residual = _projection_residual_l2!(workspace.full_proj_tmp::Array{Float32, 3}, projection, volume, geo, residual_projection_type, gpu_ids)
            push!(residual_l2, residual)
            if stop_on_l2_increase && residual > prev_residual
                stop_reason = :l2_increase
                break
            end
            prev_residual = residual
        end
    end

    _finish_progress!(progress, iterations_completed, niter)
    history = _build_history(residual_l2, lambda_history[1:iterations_completed], iterations_completed, stop_reason)
    return _maybe_return_history(volume, history, return_history)
end

function _run_pcsd!(
    volume::StridedArray{Float32, 3},
    projection::StridedArray{Float32, 3},
    geo::Geometry,
    niter::Integer,
    regularize!;
    blocksize::Integer,
    order_strategy,
    lambda::Real,
    lambda_red::Real,
    maxl2err,
    tviter::Integer,
    nonneg::Bool,
    gpu_ids,
    return_history::Bool,
    show_progress::Bool,
    delta::Real = -0.005f0,
    plan = nothing,
)
    niter >= 0 || throw(ArgumentError("niter must be >= 0"))
    projection_count(geo) > 0 || throw(ArgumentError("angles is empty"))
    plan = _resolve_art_plan(plan, projection, geo, Int(blocksize), order_strategy; gpu_ids, need_full_projection = true)

    epsilon = if maxl2err === nothing
        cached = plan.residual_seed_cache[]
        if isnan(cached)
            seed = _art_extra_volume(plan, 3, geo)
            if geo.mode === :parallel
                fbp!(seed, projection, geo; gpu_ids)
            else
                fdk!(seed, projection, geo; gpu_ids)
            end
            cached = _projection_residual_l2!(plan.workspace.full_proj_tmp::Array{Float32, 3}, projection, seed, geo, :siddon, gpu_ids) * 0.2f0
            plan.residual_seed_cache[] = cached
        end
        cached
    else
        Float32(maxl2err)
    end

    art_prev = _art_extra_volume(plan, 1, geo)
    tv_prev = _art_extra_volume(plan, 2, geo)
    residual_l2 = Float32[]
    lambda_history = Float32[]
    beta = Float32(lambda)
    beta_red = Float32(lambda_red)
    dtvg = 1f0
    d_p_first = 1f0
    stop_reason = :max_iterations
    iterations_completed = 0
    progress = _make_progress(niter, show_progress, "PCSD: ")

    for iter in 1:niter
        push!(lambda_history, beta)
        copyto!(art_prev, volume)
        d_p = _projection_residual_l2!(plan.workspace.full_proj_tmp::Array{Float32, 3}, projection, volume, geo, :siddon, gpu_ids)
        if d_p^2 > epsilon
            _art_iteration!(volume, projection, plan, beta, nonneg, :siddon, :fdk, gpu_ids)
        end

        dd = _projection_residual_l2!(plan.workspace.full_proj_tmp::Array{Float32, 3}, projection, volume, geo, :siddon, gpu_ids)
        push!(residual_l2, dd)
        dp = _diff_l2(volume, art_prev)
        iter == 1 ? (dtvg = 1f0) : (dtvg = d_p / d_p_first)

        copyto!(tv_prev, volume)
        regularize!(volume, tv_prev, dtvg, tviter, gpu_ids, Float32(delta), 0f0)
        nonneg && clamp!(volume, 0f0, Inf32)
        dg = _diff_l2(volume, tv_prev)
        iter == 1 && (d_p_first = dd)

        beta *= beta_red
        dot_gd = _update_dot(volume, tv_prev, art_prev)
        c = (dg > 0f0 && dp > 0f0) ? dot_gd / max(dg * dp, 1f-6) : 0f0
        iterations_completed = iter
        _next_progress!(progress, iter, niter)
        if (c < -0.99f0 && dd <= epsilon) || beta < 0.005f0 || iter >= niter
            stop_reason = (c < -0.99f0 && dd <= epsilon) ? :pcsd_criterion : :max_iterations
            break
        end
    end

    _finish_progress!(progress, iterations_completed, niter)
    history = _build_history(residual_l2, lambda_history, iterations_completed, stop_reason)
    return _maybe_return_history(volume, history, return_history)
end

"""
    pcsd(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, lambda::Real = 1f0, lambda_red::Real = 1f0, maxl2err = nothing, tviter::Integer = 20, nonneg::Bool = true, gpu_ids = nothing, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

Projections onto Convex Sets with Steepest Descent (PCSD).

A variation of ASD-POCS that updates the volume using ART only if the projection error exceeds a defined tolerance.

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Maximum number of iterations.
- `init = :zeros`: Initial volume guess (`:zeros`, `:fdk`, or custom 3D array, default: `:zeros`).
- `lambda::Real = 1.0f0`: Initial ART relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 1.0f0`: ART relaxation parameter reduction per iteration (default: `1.0f0`).
- `maxl2err = nothing`: Residual error threshold for ART updates or `nothing` for auto-threshold (default: `nothing`).
- `tviter::Integer = 20`: Sub-iterations of TV gradient descent per step (default: `20`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `Array{Float32, 3}`: Reconstructed 3D volume (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = pcsd(proj, geo, 20)
```
"""
function pcsd(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return pcsd!(volume, projection_checked, geo, niter; kwargs...)
end

"""
    pcsd!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; lambda::Real = 1f0, lambda_red::Real = 1f0, maxl2err = nothing, tviter::Integer = 20, nonneg::Bool = true, gpu_ids = nothing, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

In-place Projections onto Convex Sets with Steepest Descent (PCSD) reconstruction into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Maximum number of iterations.
- `lambda::Real = 1.0f0`: Initial ART relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 1.0f0`: ART relaxation parameter reduction per iteration (default: `1.0f0`).
- `maxl2err = nothing`: Residual error threshold for ART updates or `nothing` for auto-threshold (default: `nothing`).
- `tviter::Integer = 20`: Sub-iterations of TV gradient descent per step (default: `20`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `volume`: The modified reconstructed volume array (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = zeros(Float32, volume_size(geo))
pcsd!(vol, proj, geo, 20)
```
"""
function pcsd!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer;
    lambda::Real = 1f0,
    lambda_red::Real = 1f0,
    maxl2err = nothing,
    tviter::Integer = 20,
    nonneg::Bool = true,
    gpu_ids = nothing,
    return_history::Bool = false,
    show_progress::Bool = false,
    plan = nothing,
)
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))
    return _run_pcsd!(volume, projection, geo, niter, _min_tv_regularize!;
        blocksize = 1,
        order_strategy = :ordered,
        lambda,
        lambda_red,
        maxl2err,
        tviter,
        nonneg,
        gpu_ids,
        return_history,
        show_progress,
        plan,
    )
end

"""
    aw_pcsd(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, lambda::Real = 1f0, lambda_red::Real = 1f0, maxl2err = nothing, tviter::Integer = 20, delta::Real = -0.005f0, nonneg::Bool = true, gpu_ids = nothing, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

Adaptive Weighted PCSD (Aw-PCSD).

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Maximum number of iterations.
- `init = :zeros`: Initial volume guess (`:zeros`, `:fdk`, or custom 3D array, default: `:zeros`).
- `lambda::Real = 1.0f0`: Initial ART relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 1.0f0`: ART relaxation parameter reduction per iteration (default: `1.0f0`).
- `maxl2err = nothing`: Residual error threshold for ART updates or `nothing` for auto-threshold (default: `nothing`).
- `tviter::Integer = 20`: Sub-iterations of TV gradient descent per step (default: `20`).
- `delta::Real = -0.005f0`: Scale parameter for edge weights or negative for automatic estimation (default: `-0.005f0`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `Array{Float32, 3}`: Reconstructed 3D volume (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = aw_pcsd(proj, geo, 20)
```
"""
function aw_pcsd(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return aw_pcsd!(volume, projection_checked, geo, niter; kwargs...)
end

"""
    aw_pcsd!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; lambda::Real = 1f0, lambda_red::Real = 1f0, maxl2err = nothing, tviter::Integer = 20, delta::Real = -0.005f0, nonneg::Bool = true, gpu_ids = nothing, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

In-place Adaptive Weighted PCSD reconstruction into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Maximum number of iterations.
- `lambda::Real = 1.0f0`: Initial ART relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 1.0f0`: ART relaxation parameter reduction per iteration (default: `1.0f0`).
- `maxl2err = nothing`: Residual error threshold for ART updates or `nothing` for auto-threshold (default: `nothing`).
- `tviter::Integer = 20`: Sub-iterations of TV gradient descent per step (default: `20`).
- `delta::Real = -0.005f0`: Scale parameter for edge weights or negative for automatic estimation (default: `-0.005f0`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `volume`: The modified reconstructed volume array (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = zeros(Float32, volume_size(geo))
aw_pcsd!(vol, proj, geo, 20)
```
"""
function aw_pcsd!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer;
    lambda::Real = 1f0,
    lambda_red::Real = 1f0,
    maxl2err = nothing,
    tviter::Integer = 20,
    delta::Real = -0.005f0,
    nonneg::Bool = true,
    gpu_ids = nothing,
    return_history::Bool = false,
    show_progress::Bool = false,
    plan = nothing,
)
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))
    return _run_pcsd!(volume, projection, geo, niter, _awmin_tv_regularize!;
        blocksize = 1,
        order_strategy = :ordered,
        lambda,
        lambda_red,
        maxl2err,
        tviter,
        nonneg,
        gpu_ids,
        return_history,
        show_progress,
        delta,
        plan,
    )
end

"""
    os_pcsd(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; blocksize::Integer = 20, init = :zeros, lambda::Real = 1f0, lambda_red::Real = 1f0, maxl2err = nothing, tviter::Integer = 20, order_strategy = :ordered, nonneg::Bool = true, gpu_ids = nothing, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

Ordered Subsets PCSD (OS-PCSD).

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Maximum number of iterations.
- `blocksize::Integer = 20`: Number of projection angles per subset (default: `20`).
- `init = :zeros`: Initial volume guess (`:zeros`, `:fdk`, or custom 3D array, default: `:zeros`).
- `lambda::Real = 1.0f0`: Initial ART relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 1.0f0`: ART relaxation parameter reduction per iteration (default: `1.0f0`).
- `maxl2err = nothing`: Residual error threshold for ART updates or `nothing` for auto-threshold (default: `nothing`).
- `tviter::Integer = 20`: Sub-iterations of TV gradient descent per step (default: `20`).
- `order_strategy = :ordered`: Projection angle visiting order (`:ordered` or `:random`, default: `:ordered`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `Array{Float32, 3}`: Reconstructed 3D volume (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = os_pcsd(proj, geo, 20; blocksize = 20)
```
"""
function os_pcsd(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; blocksize::Integer = 20, init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return os_pcsd!(volume, projection_checked, geo, niter; blocksize, kwargs...)
end

"""
    os_pcsd!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; blocksize::Integer = 20, lambda::Real = 1f0, lambda_red::Real = 1f0, maxl2err = nothing, tviter::Integer = 20, order_strategy = :ordered, nonneg::Bool = true, gpu_ids = nothing, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

In-place Ordered Subsets PCSD reconstruction into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Maximum number of iterations.
- `blocksize::Integer = 20`: Number of projection angles per subset (default: `20`).
- `lambda::Real = 1.0f0`: Initial ART relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 1.0f0`: Relaxation factor reduction per iteration (default: `1.0f0`).
- `maxl2err = nothing`: Residual error threshold for ART updates or `nothing` for auto-threshold (default: `nothing`).
- `tviter::Integer = 20`: Sub-iterations of TV gradient descent per step (default: `20`).
- `order_strategy = :ordered`: Projection angle visiting order (`:ordered` or `:random`, default: `:ordered`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `volume`: The modified reconstructed volume array (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = zeros(Float32, volume_size(geo))
os_pcsd!(vol, proj, geo, 20; blocksize = 20)
```
"""
function os_pcsd!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer;
    blocksize::Integer = 20,
    lambda::Real = 1f0,
    lambda_red::Real = 1f0,
    maxl2err = nothing,
    tviter::Integer = 20,
    order_strategy = :ordered,
    nonneg::Bool = true,
    gpu_ids = nothing,
    return_history::Bool = false,
    show_progress::Bool = false,
    plan = nothing,
)
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))
    return _run_pcsd!(volume, projection, geo, niter, _min_tv_regularize!;
        blocksize,
        order_strategy,
        lambda,
        lambda_red,
        maxl2err,
        tviter,
        nonneg,
        gpu_ids,
        return_history,
        show_progress,
        plan,
    )
end

"""
    os_aw_pcsd(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; blocksize::Integer = 20, init = :zeros, lambda::Real = 1f0, lambda_red::Real = 1f0, maxl2err = nothing, tviter::Integer = 20, delta::Real = -0.005f0, order_strategy = :ordered, nonneg::Bool = true, gpu_ids = nothing, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

Ordered Subsets Adaptive Weighted PCSD (OS-Aw-PCSD).

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Maximum number of iterations.
- `blocksize::Integer = 20`: Number of projection angles per subset (default: `20`).
- `init = :zeros`: Initial volume guess (`:zeros`, `:fdk`, or custom 3D array, default: `:zeros`).
- `lambda::Real = 1.0f0`: Initial ART relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 1.0f0`: ART relaxation parameter reduction per iteration (default: `1.0f0`).
- `maxl2err = nothing`: Residual error threshold for ART updates or `nothing` for auto-threshold (default: `nothing`).
- `tviter::Integer = 20`: Sub-iterations of TV gradient descent per step (default: `20`).
- `delta::Real = -0.005f0`: Scale parameter for edge weights or negative for automatic estimation (default: `-0.005f0`).
- `order_strategy = :ordered`: Projection angle visiting order (`:ordered` or `:random`, default: `:ordered`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `Array{Float32, 3}`: Reconstructed 3D volume (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = os_aw_pcsd(proj, geo, 20; blocksize = 20)
```
"""
function os_aw_pcsd(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; blocksize::Integer = 20, init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return os_aw_pcsd!(volume, projection_checked, geo, niter; blocksize, kwargs...)
end

"""
    os_aw_pcsd!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; blocksize::Integer = 20, lambda::Real = 1f0, lambda_red::Real = 1f0, maxl2err = nothing, tviter::Integer = 20, delta::Real = -0.005f0, order_strategy = :ordered, nonneg::Bool = true, gpu_ids = nothing, return_history::Bool = false, show_progress::Bool = false, plan = nothing) -> volume

In-place Ordered Subsets Adaptive Weighted PCSD reconstruction into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Maximum number of iterations.
- `blocksize::Integer = 20`: Number of projection angles per subset (default: `20`).
- `lambda::Real = 1.0f0`: Initial ART relaxation parameter (default: `1.0f0`).
- `lambda_red::Real = 1f0`: ART relaxation parameter reduction per iteration (default: `1.0f0`).
- `maxl2err = nothing`: Residual error threshold for ART updates or `nothing` for auto-threshold (default: `nothing`).
- `tviter::Integer = 20`: Sub-iterations of TV gradient descent per step (default: `20`).
- `delta::Real = -0.005f0`: Scale parameter for edge weights or negative for automatic estimation (default: `-0.005f0`).
- `order_strategy = :ordered`: Projection angle visiting order (`:ordered` or `:random`, default: `:ordered`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `plan = nothing`: Optional precomputed `ARTPlan` for zero-allocation reuse (default: `nothing`).

# Returns
- `volume`: The modified reconstructed volume array (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = zeros(Float32, volume_size(geo))
os_aw_pcsd!(vol, proj, geo, 20; blocksize = 20)
```
"""
function os_aw_pcsd!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer;
    blocksize::Integer = 20,
    lambda::Real = 1f0,
    lambda_red::Real = 1f0,
    maxl2err = nothing,
    tviter::Integer = 20,
    delta::Real = -0.005f0,
    order_strategy = :ordered,
    nonneg::Bool = true,
    gpu_ids = nothing,
    return_history::Bool = false,
    show_progress::Bool = false,
    plan = nothing,
)
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))
    return _run_pcsd!(volume, projection, geo, niter, _awmin_tv_regularize!;
        blocksize,
        order_strategy,
        lambda,
        lambda_red,
        maxl2err,
        tviter,
        nonneg,
        gpu_ids,
        return_history,
        show_progress,
        delta,
        plan,
    )
end

function _gradient_step!(volume::StridedArray{Float32, 3}, backproj::StridedArray{Float32, 3}, step_scale::Float32, nonneg::Bool)
    @inbounds for i in eachindex(volume, backproj)
        updated = volume[i] + step_scale * backproj[i]
        volume[i] = nonneg && updated < 0f0 ? 0f0 : updated
    end
    return volume
end

"""
    power_method(geo::Geometry; maxiter::Integer = 100, tol::Real = 1e-3, gpu_ids = nothing, projection_type::Symbol = :siddon, backprojection_type::Symbol = :matched, seed::Integer = 0, verbose::Bool = false) -> Float32

Determine the largest singular value (operator norm) `σ_max(A)` of the forward-projection operator for the given geometry using the SVD power method.

# Arguments
- `geo::Geometry`: CT acquisition geometry specification.
- `maxiter::Integer = 100`: Maximum power method iterations (default: `100`).
- `tol::Real = 1e-3`: Convergence tolerance on Rayleigh quotient (default: `1e-3`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `projection_type::Symbol = :siddon`: Forward projection ray-tracing kernel (`:siddon` or `:interpolated`, default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:fdk`, default: `:matched`).
- `seed::Integer = 0`: Random seed for initial test volume (default: `0`).
- `verbose::Bool = false`: Print convergence progress per iteration (default: `false`).

# Returns
- `Float32`: Estimated largest singular value `σ_max(A)`.

# Examples
```julia
sigma = power_method(geo; maxiter = 50)
```
"""
function power_method(geo::Geometry;
    maxiter::Integer = 100,
    tol::Real = 1e-3,
    gpu_ids = nothing,
    projection_type::Symbol = :siddon,
    backprojection_type::Symbol = :matched,
    seed::Integer = 0,
    verbose::Bool = false,
)
    maxiter >= 1 || throw(ArgumentError("maxiter must be >= 1"))
    tol > 0 || throw(ArgumentError("tol must be > 0"))

    rng = Random.Xoshiro(seed)
    arr = randn(rng, Float32, volume_size(geo)...)
    norm_arr = sqrt(_squared_l2(arr))
    if norm_arr > 0f0
        arr ./= norm_arr
    end

    arr_old = copy(arr)
    u = Array{Float32}(undef, projection_size(geo)...)
    ax_arr = Array{Float32}(undef, projection_size(geo)...)
    z = Array{Float32}(undef, volume_size(geo)...)

    sigma = 0f0
    tol32 = Float32(tol)

    for iter in 1:maxiter
        Ax!(u, arr_old, geo; projection_type, gpu_ids)
        alpha = sqrt(_squared_l2(u))
        alpha > 0f0 || break
        u ./= alpha

        Atb!(z, u, geo; backprojection_type, gpu_ids)
        beta = sqrt(_squared_l2(z))
        beta > 0f0 || break
        arr .= z ./ beta

        Ax!(ax_arr, arr, geo; projection_type, gpu_ids)
        err_sq = 0.0
        @inbounds for i in eachindex(ax_arr, u)
            diff = Float64(ax_arr[i] - beta * u[i])
            err_sq += diff * diff
        end
        err = Float32(sqrt(err_sq))
        sigma = beta
        copyto!(arr_old, arr)

        if verbose
            println("power_method: iter $iter/$maxiter, sigma = $sigma, err = $err")
        end
        if err < tol32
            break
        end
    end
    return sigma
end

"""
    estimate_hyper(geo::Geometry; maxiter::Integer = 100, tol::Real = 1e-3, gpu_ids = nothing, projection_type::Symbol = :siddon, backprojection_type::Symbol = :matched, seed::Integer = 0, verbose::Bool = false) -> Float32

Estimate the Lipschitz constant / step-size hyperparameter for FISTA / ISTA:
`hyper = 2 * 1.05 * σ_max(A)^2`, matching the upstream TIGRE formula.

# Arguments
- `geo::Geometry`: CT acquisition geometry specification.
- `maxiter::Integer = 100`: Maximum power method iterations (default: `100`).
- `tol::Real = 1e-3`: Convergence tolerance on Rayleigh quotient (default: `1e-3`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `projection_type::Symbol = :siddon`: Forward projection ray-tracing kernel (`:siddon` or `:interpolated`, default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:fdk`, default: `:matched`).
- `seed::Integer = 0`: Random seed for power method initial volume (default: `0`).
- `verbose::Bool = false`: Print convergence progress per iteration (default: `false`).

# Returns
- `Float32`: Estimated hyperparameter for proximal step size scaling.

# Examples
```julia
hyper = estimate_hyper(geo)
```
"""
function estimate_hyper(geo::Geometry; kwargs...)
    sigma = power_method(geo; kwargs...)
    return 2f0 * 1.05f0 * (sigma * sigma)
end

"""
    computeCOR(projection, geo; slice = nothing, angles = @view(geo.angles[1, :])) -> Float32
    computeCOR(projection, geo, angles; slice = nothing) -> Float32

Estimate the centre-of-rotation (COR) offset in mm at the rotation axis, matching the sign convention expected by `geo.COR`.

`projection` can be a 3D array of size `(nu, nv, nangles)` or a 2D sinogram of size `(nu, nangles)`.
If `slice` is omitted, the central detector row is used (`floor(Int, nv / 2) + 1`).

# Arguments
- `projection`: Projection data array `(nu, nv, nangles)` or 2D sinogram `(nu, nangles)`.
- `geo::Geometry`: Corresponding CT geometry specification.
- `slice = nothing`: 1-based vertical slice index (default: `nothing`, uses central row).
- `angles = nothing`: Projection angles in radians (default: `@view(geo.angles[1, :])`).

# Returns
- `Float32`: Estimated centre-of-rotation offset in mm.

# Reference
- Liu, T., "Direct central ray determination in computed microtomography", Optical Engineering, 48(4), 2009.

# Examples
```julia
cor_offset = computeCOR(proj, geo)
```
"""
function computeCOR(
    projection::AbstractArray{<:Real, 3},
    geo::Geometry;
    slice::Union{Integer, Nothing} = nothing,
    angles::Union{AbstractVector{<:Real}, AbstractMatrix{<:Real}, Nothing} = nothing,
)
    nv = size(projection, 2)
    slc = slice === nothing ? (nv ÷ 2) + 1 : Int(slice)
    (1 <= slc <= nv) || throw(ArgumentError("slice index $slc is out of detector bounds 1:$nv"))
    sino = Matrix{Float64}(@view(projection[:, slc, :]))
    return _compute_cor_from_sinogram(sino, geo, angles)
end

function computeCOR(
    sinogram::AbstractMatrix{<:Real},
    geo::Geometry;
    slice::Union{Integer, Nothing} = nothing,
    angles::Union{AbstractVector{<:Real}, AbstractMatrix{<:Real}, Nothing} = nothing,
)
    sino = Matrix{Float64}(sinogram)
    return _compute_cor_from_sinogram(sino, geo, angles)
end

computeCOR(data::AbstractArray{<:Real}, geo::Geometry, angles; slice = nothing) =
    computeCOR(data, geo; slice = slice, angles = angles)

function _compute_cor_from_sinogram(
    sino::Matrix{Float64},
    geo::Geometry,
    angles::Union{AbstractVector{<:Real}, AbstractMatrix{<:Real}, Nothing},
)
    angs = if angles === nothing
        Float64.(vec(@view(geo.angles[1, :])))
    elseif angles isa AbstractVector
        Float64.(vec(angles))
    elseif angles isa AbstractMatrix
        if size(angles, 1) == 3
            Float64.(vec(@view(angles[1, :])))
        elseif size(angles, 2) == 3
            Float64.(vec(@view(angles[:, 1])))
        else
            Float64.(vec(angles))
        end
    else
        throw(ArgumentError("unsupported angles type"))
    end

    length(angs) == size(sino, 2) || throw(ArgumentError("angles count ($(length(angs))) does not match projection count ($(size(sino, 2)))"))

    if length(angs) > 1 && angs[1] > angs[end]
        angs = reverse(angs)
        sino = sino[:, end:-1:1]
    end

    DSD = Float64(sum(geo.DSD)) / length(geo.DSD)
    DSO = Float64(sum(geo.DSO)) / length(geo.DSO)

    n_u = geo.nDetector[1]
    s_u = Float64(geo.sDetector[1])
    d_u = Float64(geo.dDetector[1])
    det_min = -s_u / 2.0 + d_u / 2.0

    gamma = [atan((det_min + (i - 1) * d_u) / DSD) for i in 1:n_u]

    two_pi = 2.0 * π
    nangles = length(angs)
    ang3 = Vector{Float64}(undef, 3 * nangles)
    ang3[1:nangles] .= angs .- two_pi
    ang3[nangles+1:2*nangles] .= angs
    ang3[2*nangles+1:3*nangles] .= angs .+ two_pi

    sino3 = Matrix{Float64}(undef, n_u, 3 * nangles)
    sino3[:, 1:nangles] .= sino
    sino3[:, nangles+1:2*nangles] .= sino
    sino3[:, 2*nangles+1:3*nangles] .= sino

    midpoint = 0.0
    precisions = (1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001)

    COR = zeros(Float64, 21)
    M = fill(Inf, 21)

    for pr in precisions
        for j in 1:21
            COR[j] = midpoint + (j - 11) * pr
            M[j] = Inf

            gamma_c = atan(COR[j] / DSD)

            sse = 0.0
            n_valid = 0

            for u in 1:n_u
                s2 = DSD * tan(2.0 * gamma_c - gamma[u])
                t_u = (s2 - det_min) / d_u + 1.0
                (t_u < 1.0 || t_u > Float64(n_u)) && continue

                iu0 = clamp(floor(Int, t_u), 1, n_u - 1)
                iu1 = iu0 + 1
                wu = t_u - iu0

                beta_u = 2.0 * (gamma[u] - gamma_c) + π

                @inbounds for k in 1:nangles
                    ang_aux = angs[k] + beta_u
                    (ang_aux < ang3[1] || ang_aux > ang3[end]) && continue

                    ia0 = searchsortedlast(ang3, ang_aux)
                    if ia0 < 1
                        continue
                    elseif ia0 >= 3 * nangles
                        if ang_aux == ang3[end]
                            ia0 = 3 * nangles - 1
                        else
                            continue
                        end
                    end
                    ia1 = ia0 + 1
                    d_ang = ang3[ia1] - ang3[ia0]
                    wa = d_ang > 0.0 ? (ang_aux - ang3[ia0]) / d_ang : 0.0

                    v00 = sino3[iu0, ia0]
                    v10 = sino3[iu1, ia0]
                    v01 = sino3[iu0, ia1]
                    v11 = sino3[iu1, ia1]

                    test_val = (1.0 - wu) * (1.0 - wa) * v00 +
                               wu * (1.0 - wa) * v10 +
                               (1.0 - wu) * wa * v01 +
                               wu * wa * v11

                    if test_val > 0.0
                        diff = test_val - sino[u, k]
                        sse += diff * diff
                        n_valid += 1
                    end
                end
            end

            if n_valid > 0
                M[j] = sse / n_valid
            end
        end

        min_idx = argmin(M)
        midpoint = COR[min_idx]
    end

    return -midpoint * DSO / DSD
end



function _resolve_hyper(hyper, geo::Geometry; gpu_ids = nothing, projection_type::Symbol = :siddon, backprojection_type::Symbol = :matched)
    if hyper isa Real
        val = Float32(hyper)
        val > 0f0 || throw(ArgumentError("hyper must be > 0, got $hyper"))
        return val
    elseif hyper isa Symbol
        if hyper === :auto
            return estimate_hyper(geo; gpu_ids, projection_type, backprojection_type)
        else
            throw(ArgumentError("unsupported hyper symbol: :$hyper (expected :auto)"))
        end
    elseif hyper isa AbstractString
        s = lowercase(strip(hyper))
        if s == "auto"
            return estimate_hyper(geo; gpu_ids, projection_type, backprojection_type)
        else
            throw(ArgumentError("unsupported hyper string: \"$hyper\" (expected \"auto\")"))
        end
    else
        throw(ArgumentError("hyper must be a Real, :auto, or \"auto\", got $(typeof(hyper))"))
    end
end

"""
    ista(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, hyper = 2.0f8, tviter::Integer = 20, tvlambda::Real = 0.1f0, nonneg::Bool = true, projection_type::Symbol = :siddon, backprojection_type::Symbol = :matched, gpu_ids = nothing, return_history::Bool = false, compute_l2::Bool = false, stop_on_l2_increase::Bool = false, residual_projection_type::Symbol = projection_type, show_progress::Bool = false, workspace = nothing) -> volume

Iterative Shrinkage-Thresholding Algorithm (ISTA) with Total Variation (TV) proximal regularization.

Solves the penalized least-squares problem:
```math
\\min_{x \\ge 0} \\frac{1}{2} \\|Ax - b\\|_2^2 + \\lambda_{\\mathrm{TV}} \\mathrm{TV}(x)
```

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Number of ISTA iterations.
- `init = :zeros`: Initial volume guess (`:zeros`, `:fdk`, or custom 3D array, default: `:zeros`).
- `hyper = 2.0f8`: Lipschitz constant or step size scaling, or `:auto` to estimate via SVD power method (default: `2.0f8`).
- `tviter::Integer = 20`: Number of internal TV proximal iterations (default: `20`).
- `tvlambda::Real = 0.1f0`: TV regularization weight in objective `(1/2)||Ax - b||₂² + tvlambda * TV(x)` (default: `0.1f0`). Larger values produce stronger TV smoothing (internally mapped to proximal fidelity parameter).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `projection_type::Symbol = :siddon`: Forward projection ray-tracing kernel (`:siddon` or `:interpolated`, default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:fdk`, default: `:matched`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Compute residual L2 norm per iteration (default: `false`).
- `stop_on_l2_increase::Bool = false`: Terminate iterations early if residual L2 norm increases (default: `false`).
- `residual_projection_type::Symbol = projection_type`: Kernel for residual evaluation (default: `projection_type`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `workspace = nothing`: Optional preallocated `FirstOrderWorkspace` (default: `nothing`).

# Returns
- `Array{Float32, 3}`: Reconstructed 3D volume (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Reference
- Daubechies, I., Defrise, M., & De Mol, C., "An iterative thresholding algorithm for linear inverse problems", Comm. Pure Appl. Math., 57(11), 2004.

# Examples
```julia
vol = ista(proj, geo, 20; tvlambda = 0.1f0, hyper = :auto)
```
"""
function ista(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return ista!(volume, projection_checked, geo, niter; kwargs...)
end

"""
    ista!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; hyper = 2.0f8, tviter::Integer = 20, tvlambda::Real = 0.1f0, nonneg::Bool = true, projection_type::Symbol = :siddon, backprojection_type::Symbol = :matched, gpu_ids = nothing, return_history::Bool = false, compute_l2::Bool = false, stop_on_l2_increase::Bool = false, residual_projection_type::Symbol = projection_type, show_progress::Bool = false, workspace = nothing) -> volume

In-place Iterative Shrinkage-Thresholding Algorithm (ISTA) reconstruction into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Number of ISTA iterations.
- `hyper = 2.0f8`: Lipschitz constant or step size scaling, or `:auto` to estimate via SVD power method (default: `2.0f8`).
- `tviter::Integer = 20`: Number of internal TV proximal iterations (default: `20`).
- `tvlambda::Real = 0.1f0`: TV regularization weight in objective `(1/2)||Ax - b||₂² + tvlambda * TV(x)` (default: `0.1f0`). Larger values produce stronger TV smoothing (internally mapped to proximal fidelity parameter).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `projection_type::Symbol = :siddon`: Forward projection ray-tracing kernel (`:siddon` or `:interpolated`, default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:fdk`, default: `:matched`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Compute residual L2 norm per iteration (default: `false`).
- `stop_on_l2_increase::Bool = false`: Terminate iterations early if residual L2 norm increases (default: `false`).
- `residual_projection_type::Symbol = projection_type`: Kernel for residual evaluation (default: `projection_type`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `workspace = nothing`: Optional preallocated `FirstOrderWorkspace` (default: `nothing`).

# Returns
- `volume`: The modified reconstructed volume array (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = zeros(Float32, volume_size(geo))
ista!(vol, proj, geo, 20; tvlambda = 0.1f0, hyper = :auto)
```
"""
function ista!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer;
    hyper = 2.0f8,
    tviter::Integer = 20,
    tvlambda::Real = 0.1f0,
    nonneg::Bool = true,
    projection_type::Symbol = :siddon,
    backprojection_type::Symbol = :matched,
    gpu_ids = nothing,
    return_history::Bool = false,
    compute_l2::Bool = false,
    stop_on_l2_increase::Bool = false,
    residual_projection_type::Symbol = projection_type,
    show_progress::Bool = false,
    workspace = nothing,
)
    niter >= 0 || throw(ArgumentError("niter must be >= 0"))
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))
    workspace = _resolve_first_order_workspace(workspace, geo; need_full_projection = compute_l2 || stop_on_l2_increase || return_history)
    resolved_hyper = _resolve_hyper(hyper, geo; gpu_ids, projection_type, backprojection_type)
    step_scale = 2f0 / resolved_hyper
    tv_lambda = inv(2f0 * inv(resolved_hyper) * Float32(tvlambda))
    residual_l2 = Float32[]
    lambda_history = fill(step_scale, niter)
    prev_residual = Inf32
    stop_reason = :max_iterations
    iterations_completed = 0
    progress = _make_progress(niter, show_progress, "ISTA: ")

    for iter in 1:niter
        Ax!(workspace.proj_tmp, volume, geo; projection_type, gpu_ids)
        @inbounds for i in eachindex(workspace.proj_tmp, projection)
            workspace.proj_tmp[i] = projection[i] - workspace.proj_tmp[i]
        end
        Atb!(workspace.backproj_tmp, workspace.proj_tmp, geo; backprojection_type, gpu_ids)
        _gradient_step!(volume, workspace.backproj_tmp, step_scale, nonneg)
        tvdenoise!(workspace.prox_tmp, volume; lambda = tv_lambda, maxiter = tviter, gpu_ids)
        copyto!(volume, workspace.prox_tmp)

        iterations_completed = iter
        _next_progress!(progress, iter, niter)
        if workspace.full_proj_tmp !== nothing
            residual = _projection_residual_l2!(workspace.full_proj_tmp::Array{Float32, 3}, projection, volume, geo, residual_projection_type, gpu_ids)
            push!(residual_l2, residual)
            if stop_on_l2_increase && residual > prev_residual
                stop_reason = :l2_increase
                break
            end
            prev_residual = residual
        end
    end

    _finish_progress!(progress, iterations_completed, niter)
    history = _build_history(residual_l2, lambda_history[1:iterations_completed], iterations_completed, stop_reason)
    return _maybe_return_history(volume, history, return_history)
end

"""
    fista(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, hyper = 2.0f8, tviter::Integer = 20, tvlambda::Real = 0.1f0, p::Real = 1f0, q::Real = 1f0, nonneg::Bool = true, projection_type::Symbol = :siddon, backprojection_type::Symbol = :matched, gpu_ids = nothing, return_history::Bool = false, compute_l2::Bool = false, stop_on_l2_increase::Bool = false, residual_projection_type::Symbol = projection_type, show_progress::Bool = false, workspace = nothing) -> volume

Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) with Nesterov momentum acceleration and TV proximal regularization.

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Number of FISTA iterations.
- `init = :zeros`: Initial volume guess (`:zeros`, `:fdk`, or custom 3D array, default: `:zeros`).
- `hyper = 2.0f8`: Lipschitz constant or step size scaling, or `:auto` to estimate via SVD power method (default: `2.0f8`).
- `tviter::Integer = 20`: Number of internal TV proximal iterations (default: `20`).
- `tvlambda::Real = 0.1f0`: TV regularization weight in objective `(1/2)||Ax - b||₂² + tvlambda * TV(x)` (default: `0.1f0`). Larger values produce stronger TV smoothing (internally mapped to proximal fidelity parameter).
- `p::Real = 1.0f0`: Momentum coefficient parameter (default: `1.0f0`).
- `q::Real = 1.0f0`: Momentum coefficient parameter (default: `1.0f0`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `projection_type::Symbol = :siddon`: Forward projection ray-tracing kernel (`:siddon` or `:interpolated`, default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:fdk`, default: `:matched`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Compute residual L2 norm per iteration (default: `false`).
- `stop_on_l2_increase::Bool = false`: Terminate iterations early if residual L2 norm increases (default: `false`).
- `residual_projection_type::Symbol = projection_type`: Kernel for residual evaluation (default: `projection_type`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `workspace = nothing`: Optional preallocated `FirstOrderWorkspace` (default: `nothing`).

# Returns
- `Array{Float32, 3}`: Reconstructed 3D volume (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Reference
- Beck, A., & Teboulle, M., "A fast iterative shrinkage-thresholding algorithm for linear inverse problems", SIAM J. Imaging Sci., 2(1), 2009.

# Examples
```julia
vol = fista(proj, geo, 20; tvlambda = 0.1f0, hyper = :auto)
```
"""
function fista(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return fista!(volume, projection_checked, geo, niter; kwargs...)
end

"""
    fista!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; hyper = 2.0f8, tviter::Integer = 20, tvlambda::Real = 0.1f0, p::Real = 1f0, q::Real = 1f0, nonneg::Bool = true, projection_type::Symbol = :siddon, backprojection_type::Symbol = :matched, gpu_ids = nothing, return_history::Bool = false, compute_l2::Bool = false, stop_on_l2_increase::Bool = false, residual_projection_type::Symbol = projection_type, show_progress::Bool = false, workspace = nothing) -> volume

In-place Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) reconstruction into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Number of FISTA iterations.
- `hyper = 2.0f8`: Lipschitz constant or step size scaling, or `:auto` to estimate via SVD power method (default: `2.0f8`).
- `tviter::Integer = 20`: Number of internal TV proximal iterations (default: `20`).
- `tvlambda::Real = 0.1f0`: TV regularization weight in objective `(1/2)||Ax - b||₂² + tvlambda * TV(x)` (default: `0.1f0`). Larger values produce stronger TV smoothing (internally mapped to proximal fidelity parameter).
- `p::Real = 1.0f0`: Momentum coefficient parameter (default: `1.0f0`).
- `q::Real = 1.0f0`: Momentum coefficient parameter (default: `1.0f0`).
- `nonneg::Bool = true`: Enforce non-negativity constraint `volume >= 0` (default: `true`).
- `projection_type::Symbol = :siddon`: Forward projection ray-tracing kernel (`:siddon` or `:interpolated`, default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:fdk`, default: `:matched`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Compute residual L2 norm per iteration (default: `false`).
- `stop_on_l2_increase::Bool = false`: Terminate iterations early if residual L2 norm increases (default: `false`).
- `residual_projection_type::Symbol = projection_type`: Kernel for residual evaluation (default: `projection_type`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `workspace = nothing`: Optional preallocated `FirstOrderWorkspace` (default: `nothing`).

# Returns
- `volume`: The modified reconstructed volume array (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = zeros(Float32, volume_size(geo))
fista!(vol, proj, geo, 20; tvlambda = 0.1f0, hyper = :auto)
```
"""
function fista!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer;
    hyper = 2.0f8,
    tviter::Integer = 20,
    tvlambda::Real = 0.1f0,
    p::Real = 1f0,
    q::Real = 1f0,
    nonneg::Bool = true,
    projection_type::Symbol = :siddon,
    backprojection_type::Symbol = :matched,
    gpu_ids = nothing,
    return_history::Bool = false,
    compute_l2::Bool = false,
    stop_on_l2_increase::Bool = false,
    residual_projection_type::Symbol = projection_type,
    show_progress::Bool = false,
    workspace = nothing,
)
    niter >= 0 || throw(ArgumentError("niter must be >= 0"))
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))
    workspace = _resolve_first_order_workspace(workspace, geo; need_full_projection = compute_l2 || stop_on_l2_increase || return_history)
    copyto!(workspace.aux_tmp, volume)

    resolved_hyper = _resolve_hyper(hyper, geo; gpu_ids, projection_type, backprojection_type)
    step_scale = 2f0 / resolved_hyper
    tv_lambda = inv(2f0 * inv(resolved_hyper) * Float32(tvlambda))
    t = 1f0
    pf = Float32(p)
    qf = Float32(q)
    residual_l2 = Float32[]
    lambda_history = fill(step_scale, niter)
    prev_residual = Inf32
    stop_reason = :max_iterations
    iterations_completed = 0
    progress = _make_progress(niter, show_progress, "FISTA: ")

    for iter in 1:niter
        Ax!(workspace.proj_tmp, volume, geo; projection_type, gpu_ids)
        @inbounds for i in eachindex(workspace.proj_tmp, projection)
            workspace.proj_tmp[i] = projection[i] - workspace.proj_tmp[i]
        end
        Atb!(workspace.backproj_tmp, workspace.proj_tmp, geo; backprojection_type, gpu_ids)
        _gradient_step!(volume, workspace.backproj_tmp, step_scale, nonneg)

        copyto!(workspace.backproj_tmp, workspace.aux_tmp)
        tvdenoise!(workspace.prox_tmp, volume; lambda = tv_lambda, maxiter = tviter, gpu_ids)
        t_old = t
        t = (pf + sqrt(qf + 4f0 * t * t)) / 2f0
        coeff = (t_old - 1f0) / t
        @inbounds for i in eachindex(volume, workspace.prox_tmp, workspace.backproj_tmp)
            x_new = workspace.prox_tmp[i]
            accelerated = x_new + coeff * (x_new - workspace.backproj_tmp[i])
            volume[i] = nonneg && accelerated < 0f0 ? 0f0 : accelerated
            workspace.aux_tmp[i] = x_new
        end

        iterations_completed = iter
        _next_progress!(progress, iter, niter)
        if workspace.full_proj_tmp !== nothing
            residual = _projection_residual_l2!(workspace.full_proj_tmp::Array{Float32, 3}, projection, volume, geo, residual_projection_type, gpu_ids)
            push!(residual_l2, residual)
            if stop_on_l2_increase && residual > prev_residual
                stop_reason = :l2_increase
                break
            end
            prev_residual = residual
        end
    end

    _finish_progress!(progress, iterations_completed, niter)
    history = _build_history(residual_l2, lambda_history[1:iterations_completed], iterations_completed, stop_reason)
    return _maybe_return_history(volume, history, return_history)
end

function _squared_l2(A::StridedArray{Float32})
    accum = 0.0
    @inbounds for value in A
        v = Float64(value)
        accum += v * v
    end
    return Float32(accum)
end

function _dot_arrays(a::StridedArray{Float32}, b::StridedArray{Float32})
    accum = 0.0
    @inbounds for i in eachindex(a, b)
        accum += Float64(a[i]) * b[i]
    end
    return Float32(accum)
end

"""
    cgls(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, projection_type::Symbol = :siddon, backprojection_type::Symbol = :matched, gpu_ids = nothing, return_history::Bool = false, compute_l2::Bool = false, stop_on_l2_increase::Bool = true, show_progress::Bool = false, workspace = nothing) -> volume

Conjugate Gradient Least Squares (CGLS) reconstruction.

Solves the linear normal equations `Aᵀ A x = Aᵀ b` using conjugate gradients on the least-squares system.

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Maximum number of iterations.
- `init = :zeros`: Initial volume guess (`:zeros`, `:fdk`, or custom 3D array, default: `:zeros`).
- `projection_type::Symbol = :siddon`: Forward projection ray-tracing kernel (`:siddon` or `:interpolated`, default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:fdk`, default: `:matched`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Compute residual L2 norm per iteration (default: `false`).
- `stop_on_l2_increase::Bool = true`: If `true`, terminates and rolls back volume when residual increases (semi-convergence protection, default: `true`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `workspace = nothing`: Optional preallocated `KrylovWorkspace` (default: `nothing`).

# Returns
- `Array{Float32, 3}`: Reconstructed 3D volume (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Reference
- Hestenes, M. R., & Stiefel, E., "Methods of conjugate gradients for solving linear systems", J. Res. Natl. Bur. Stand., 49(6), 1952.

# Examples
```julia
vol = cgls(proj, geo, 20)
```
"""
function cgls(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return cgls!(volume, projection_checked, geo, niter; kwargs...)
end

"""
    cgls!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; projection_type::Symbol = :siddon, backprojection_type::Symbol = :matched, gpu_ids = nothing, return_history::Bool = false, compute_l2::Bool = false, stop_on_l2_increase::Bool = true, show_progress::Bool = false, workspace = nothing) -> volume

In-place Conjugate Gradient Least Squares (CGLS) reconstruction into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Maximum number of iterations.
- `projection_type::Symbol = :siddon`: Forward projection ray-tracing kernel (`:siddon` or `:interpolated`, default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:fdk`, default: `:matched`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Compute residual L2 norm per iteration (default: `false`).
- `stop_on_l2_increase::Bool = true`: If `true`, terminates and rolls back volume when residual increases (default: `true`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `workspace = nothing`: Optional preallocated `KrylovWorkspace` (default: `nothing`).

# Returns
- `volume`: The modified reconstructed volume array (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = zeros(Float32, volume_size(geo))
cgls!(vol, proj, geo, 20)
```
"""
function cgls!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer;
    projection_type::Symbol = :siddon,
    backprojection_type::Symbol = :matched,
    gpu_ids = nothing,
    return_history::Bool = false,
    compute_l2::Bool = false,
    stop_on_l2_increase::Bool = true,
    show_progress::Bool = false,
    workspace = nothing,
)
    niter >= 0 || throw(ArgumentError("niter must be >= 0"))
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))

    workspace = _resolve_krylov_workspace(workspace, geo)
    q = workspace.proj_a
    r = workspace.proj_b
    p = workspace.vol_a
    s = workspace.vol_b
    prev_volume = workspace.vol_c
    need_residual = compute_l2 || stop_on_l2_increase || return_history
    residual_l2 = Float32[]
    lambda_history = Float32[]
    prev_residual = Inf32
    stop_reason = :max_iterations
    iterations_completed = 0
    progress = _make_progress(niter, show_progress, "CGLS: ")

    Ax!(r, volume, geo; projection_type, gpu_ids)
    @inbounds for i in eachindex(r, projection)
        r[i] = projection[i] - r[i]
    end
    Atb!(p, r, geo; backprojection_type, gpu_ids)
    gamma = _squared_l2(p)

    for iter in 1:niter
        copyto!(prev_volume, volume)
        Ax!(q, p, geo; projection_type, gpu_ids)
        qnorm2 = _squared_l2(q)
        qnorm2 > 0f0 || break
        alpha = gamma / qnorm2
        push!(lambda_history, alpha)

        @inbounds for i in eachindex(volume, p)
            volume[i] += alpha * p[i]
        end
        @inbounds for i in eachindex(r, q)
            r[i] -= alpha * q[i]
        end

        iterations_completed = iter
        _next_progress!(progress, iter, niter)
        if need_residual
            residual = Float32(sqrt(_squared_l2(r)))
            push!(residual_l2, residual)
            if stop_on_l2_increase && residual > prev_residual
                copyto!(volume, prev_volume)
                stop_reason = :l2_increase
                break
            end
            prev_residual = residual
        end

        Atb!(s, r, geo; backprojection_type, gpu_ids)
        gamma1 = _squared_l2(s)
        gamma > 0f0 || break
        beta = gamma1 / gamma
        @inbounds for i in eachindex(p, s)
            p[i] = s[i] + beta * p[i]
        end
        gamma = gamma1
    end

    _finish_progress!(progress, iterations_completed, niter)
    history = _build_history(residual_l2, lambda_history, iterations_completed, stop_reason)
    return _maybe_return_history(volume, history, return_history)
end

function _weighted_tv_weights!(weights::StridedArray{Float32, 3}, volume::StridedArray{Float32, 3})
    fill!(weights, 1f0)
    @inbounds for z in 1:size(volume, 3)-1, y in 1:size(volume, 2)-1, x in 1:size(volume, 1)-1
        dx = volume[x, y, z] - volume[x + 1, y, z]
        dy = volume[x, y, z] - volume[x, y + 1, z]
        dz = volume[x, y, z] - volume[x, y, z + 1]
        weights[x, y, z] = Float32((Float64(dx) * dx + Float64(dy) * dy + Float64(dz) * dz + 1e-6)^(-0.25))
    end
    return weights
end

function _weighted_tv_grad!(gx::StridedArray{Float32, 3}, gy::StridedArray{Float32, 3}, gz::StridedArray{Float32, 3}, weights::StridedArray{Float32, 3}, volume::StridedArray{Float32, 3}, scale::Float32)
    fill!(gx, 0f0)
    fill!(gy, 0f0)
    fill!(gz, 0f0)
    @inbounds for z in 1:size(volume, 3)-1, y in 1:size(volume, 2)-1, x in 1:size(volume, 1)-1
        w = scale * weights[x, y, z]
        gx[x, y, z] = w * (volume[x, y, z] - volume[x + 1, y, z])
        gy[x, y, z] = w * (volume[x, y, z] - volume[x, y + 1, z])
        gz[x, y, z] = w * (volume[x, y, z] - volume[x, y, z + 1])
    end
    return gx, gy, gz
end

function _weighted_tv_divergence!(dst::StridedArray{Float32, 3}, gx::StridedArray{Float32, 3}, gy::StridedArray{Float32, 3}, gz::StridedArray{Float32, 3}, scale::Float32)
    fill!(dst, 0f0)
    @inbounds for z in 1:size(dst, 3), y in 1:size(dst, 2), x in 1:size(dst, 1)
        acc = 0f0
        acc += scale * gx[x, y, z]
        acc += scale * gy[x, y, z]
        acc += scale * gz[x, y, z]
        if x > 1
            acc -= scale * gx[x - 1, y, z]
        end
        if y > 1
            acc -= scale * gy[x, y - 1, z]
        end
        if z > 1
            acc -= scale * gz[x, y, z - 1]
        end
        dst[x, y, z] = acc
    end
    return dst
end

function _combined_norm2(proj::StridedArray{Float32, 3}, gx::StridedArray{Float32, 3}, gy::StridedArray{Float32, 3}, gz::StridedArray{Float32, 3})
    accum = 0.0
    @inbounds for v in proj
        fv = Float64(v)
        accum += fv * fv
    end
    @inbounds for v in gx
        fv = Float64(v)
        accum += fv * fv
    end
    @inbounds for v in gy
        fv = Float64(v)
        accum += fv * fv
    end
    @inbounds for v in gz
        fv = Float64(v)
        accum += fv * fv
    end
    return Float32(accum)
end

function _basis_dot(volume::StridedArray{Float32, 3}, basis::Matrix{Float32}, col::Int)
    accum = 0.0
    idx = 1
    @inbounds for z in 1:size(volume, 3), y in 1:size(volume, 2), x in 1:size(volume, 1)
        accum += Float64(volume[x, y, z]) * basis[col, idx]
        idx += 1
    end
    return Float32(accum)
end

function _store_basis_row!(basis::Matrix{Float32}, row::Int, volume::StridedArray{Float32, 3})
    idx = 1
    @inbounds for z in 1:size(volume, 3), y in 1:size(volume, 2), x in 1:size(volume, 1)
        basis[row, idx] = volume[x, y, z]
        idx += 1
    end
    return basis
end

function _weighted_tv_basis_dot(volume::StridedArray{Float32, 3}, basis::Matrix{Float32}, row::Int, weights::StridedArray{Float32, 3}, scale::Float32)
    accum = 0.0
    nx = size(volume, 1)
    nxy = nx * size(volume, 2)
    @inbounds for z in 1:size(volume, 3)-1, y in 1:size(volume, 2)-1, x in 1:size(volume, 1)-1
        base = x + (y - 1) * nx + (z - 1) * nxy
        w = Float64(scale * weights[x, y, z])
        w2 = w * w
        center = Float64(volume[x, y, z])
        dx = center - Float64(volume[x + 1, y, z])
        dy = center - Float64(volume[x, y + 1, z])
        dz = center - Float64(volume[x, y, z + 1])
        bcenter = Float64(basis[row, base])
        bdx = bcenter - Float64(basis[row, base + 1])
        bdy = bcenter - Float64(basis[row, base + nx])
        bdz = bcenter - Float64(basis[row, base + nxy])
        accum += w2 * (dx * bdx + dy * bdy + dz * bdz)
    end
    return Float32(accum)
end

function _apply_basis_combination!(dst::StridedArray{Float32, 3}, base::StridedArray{Float32, 3}, basis::Matrix{Float32}, coeffs::AbstractVector{<:Real}, ncoeff::Int)
    copyto!(dst, base)
    @inbounds for j in 1:ncoeff
        coeff = Float32(coeffs[j])
        iszero(coeff) && continue
        idx = 1
        for z in 1:size(dst, 3), y in 1:size(dst, 2), x in 1:size(dst, 1)
            dst[x, y, z] += coeff * basis[j, idx]
            idx += 1
        end
    end
    return dst
end

function _fill_normal_rhs!(dst::AbstractVector{Float32}, A::AbstractMatrix{Float32}, rhs::AbstractVector{Float32}, nrow::Int, ncol::Int)
    @inbounds for j in 1:ncol
        acc = 0f0
        for i in 1:nrow
            acc += A[i, j] * rhs[i]
        end
        dst[j] = acc
    end
    return dst
end

function _fill_normal_matrix!(dst::AbstractMatrix{Float32}, A::AbstractMatrix{Float32}, nrow::Int, ncol::Int)
    @inbounds for j in 1:ncol
        for i in 1:j
            acc = 0f0
            for k in 1:nrow
                acc += A[k, i] * A[k, j]
            end
            dst[i, j] = acc
            dst[j, i] = acc
        end
    end
    return dst
end

function _solve_small_system!(lhs::AbstractMatrix{Float32}, rhs::AbstractVector{Float32}, n::Int)
    n == 0 && return rhs
    lu_fact = lu!(@view lhs[1:n, 1:n])
    ldiv!(lu_fact, @view rhs[1:n])
    return rhs
end

function _append_cholesky_column!(L::AbstractMatrix{Float32}, A::AbstractMatrix{Float32}, n::Int)
    if n == 1
        L[1, 1] = sqrt(max(A[1, 1], eps(Float32)))
        return L
    end
    @inbounds for i in 1:n-1
        acc = A[n, i]
        for k in 1:i-1
            acc -= L[n, k] * L[i, k]
        end
        L[n, i] = acc / L[i, i]
    end
    diag = A[n, n]
    @inbounds for k in 1:n-1
        diag -= L[n, k] * L[n, k]
    end
    L[n, n] = sqrt(max(diag, eps(Float32)))
    return L
end

function _solve_cholesky!(x::AbstractVector{Float32}, L::AbstractMatrix{Float32}, n::Int)
    @inbounds for i in 1:n
        acc = x[i]
        for j in 1:i-1
            acc -= L[i, j] * x[j]
        end
        x[i] = acc / L[i, i]
    end
    @inbounds for i in n:-1:1
        acc = x[i]
        for j in i+1:n
            acc -= L[j, i] * x[j]
        end
        x[i] = acc / L[i, i]
    end
    return x
end

function _least_squares_givens!(coeffs::AbstractVector{Float32}, A::AbstractMatrix{Float32}, rhs::AbstractVector{Float32}, nrow::Int, ncol::Int,
    R::AbstractMatrix{Float32}, work_rhs::AbstractVector{Float32})
    @inbounds for j in 1:ncol
        for i in 1:nrow
            R[i, j] = A[i, j]
        end
    end
    @inbounds for i in 1:nrow
        work_rhs[i] = rhs[i]
    end

    for col in 1:ncol
        for row in nrow:-1:(col + 1)
            lower = R[row, col]
            if iszero(lower)
                continue
            end
            upper = R[row - 1, col]
            radius = hypot(upper, lower)
            c = upper / radius
            s = lower / radius
            @inbounds for k in col:ncol
                top = R[row - 1, k]
                bottom = R[row, k]
                R[row - 1, k] = c * top + s * bottom
                R[row, k] = -s * top + c * bottom
            end
            rhs_top = work_rhs[row - 1]
            rhs_bottom = work_rhs[row]
            work_rhs[row - 1] = c * rhs_top + s * rhs_bottom
            work_rhs[row] = -s * rhs_top + c * rhs_bottom
        end
    end

    @inbounds for i in ncol:-1:1
        acc = work_rhs[i]
        for j in (i + 1):ncol
            acc -= R[i, j] * coeffs[j]
        end
        coeffs[i] = acc / R[i, i]
    end
    return coeffs
end

function _update_hybrid_lsqr_system!(system::AbstractMatrix{Float32}, rhs_proj::AbstractVector{Float32}, Bmat::AbstractMatrix{Float32}, beta0::Float32, lambda::Float32, iter::Int)
    alpha = Bmat[iter, iter]
    beta = Bmat[iter, iter + 1]
    system[iter, iter] = alpha * alpha + beta * beta + lambda * lambda
    if iter > 1
        coupling = Bmat[iter - 1, iter] * alpha
        system[iter - 1, iter] = coupling
        system[iter, iter - 1] = coupling
    end
    rhs_proj[iter] = iter == 1 ? alpha * beta0 : 0f0
    return system, rhs_proj
end

function _update_hybrid_flsqr_data_system!(system::AbstractMatrix{Float32}, rhs_proj::AbstractVector{Float32}, Mmat::AbstractMatrix{Float32}, beta0::Float32, iter::Int)
    @inbounds for j in 1:iter
        acc = 0f0
        for col in 1:iter+1
            acc += Mmat[iter, col] * Mmat[j, col]
        end
        system[j, iter] = acc
        system[iter, j] = acc
    end
    rhs_proj[iter] = beta0 * Mmat[iter, 1]
    return system, rhs_proj
end

function _proj_basis_dot(projection::StridedArray{Float32, 3}, basis::Matrix{Float32}, row::Int)
    accum = 0.0
    idx = 1
    @inbounds for ia in 1:size(projection, 3), iv in 1:size(projection, 2), iu in 1:size(projection, 1)
        accum += Float64(projection[iu, iv, ia]) * basis[row, idx]
        idx += 1
    end
    return Float32(accum)
end

function _store_proj_basis_row!(basis::Matrix{Float32}, row::Int, projection::StridedArray{Float32, 3})
    idx = 1
    @inbounds for ia in 1:size(projection, 3), iv in 1:size(projection, 2), iu in 1:size(projection, 1)
        basis[row, idx] = projection[iu, iv, ia]
        idx += 1
    end
    return basis
end

function _apply_backproject!(dst::StridedArray{Float32, 3}, projection::StridedArray{Float32, 3}, geo::Geometry, backprojector, gpu_ids)
    mode = backprojector === nothing ? :matched : Symbol(lowercase(String(backprojector)))
    if mode === :matched
        Atb!(dst, projection, geo; backprojection_type = :matched, gpu_ids)
        return dst
    elseif mode === :fdk
        fdk!(dst, projection, geo; gpu_ids)
        return dst
    end
    throw(ArgumentError("unsupported backprojector: $(backprojector)"))
end

function _tv_system_mul!(dst::StridedArray{Float32, 3}, src::StridedArray{Float32, 3}, weights::StridedArray{Float32, 3}, lambda::Float32, gx::StridedArray{Float32, 3}, gy::StridedArray{Float32, 3}, gz::StridedArray{Float32, 3}, tmp::StridedArray{Float32, 3})
    _weighted_tv_grad!(gx, gy, gz, weights, src, 1f0)
    _weighted_tv_divergence!(tmp, gx, gy, gz, lambda)
    @inbounds for i in eachindex(dst, src, tmp)
        dst[i] = src[i] + tmp[i]
    end
    return dst
end

function _tv_inner_cg!(dst::StridedArray{Float32, 3}, rhs::StridedArray{Float32, 3}, weights::StridedArray{Float32, 3}, lambda::Float32, nsteps::Int,
    r::StridedArray{Float32, 3}, p::StridedArray{Float32, 3}, Ap::StridedArray{Float32, 3},
    gx::StridedArray{Float32, 3}, gy::StridedArray{Float32, 3}, gz::StridedArray{Float32, 3}, tmp::StridedArray{Float32, 3})
    fill!(dst, 0f0)
    copyto!(r, rhs)
    copyto!(p, rhs)
    rsold = _squared_l2(r)
    rsold > 0f0 || return dst
    for _ in 1:nsteps
        _tv_system_mul!(Ap, p, weights, lambda, gx, gy, gz, tmp)
        denom = _dot_arrays(p, Ap)
        denom > 0f0 || break
        alpha = rsold / denom
        @inbounds for i in eachindex(dst, p, r, Ap)
            dst[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
        end
        rsnew = _squared_l2(r)
        rsnew <= 1f-12 && break
        beta = rsnew / rsold
        @inbounds for i in eachindex(p, r)
            p[i] = r[i] + beta * p[i]
        end
        rsold = rsnew
    end
    return dst
end

function _art_extra_volume(plan::ARTPlan, idx::Int, geo::Geometry)
    if idx <= length(plan.workspace.extra_volumes)
        return plan.workspace.extra_volumes[idx]
    end
    return Array{Float32}(undef, volume_size(geo)...)
end

"""
    lsqr(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, projection_type::Symbol = :siddon, backprojection_type::Symbol = :matched, gpu_ids = nothing, return_history::Bool = false, compute_l2::Bool = false, stop_on_l2_increase::Bool = true, show_progress::Bool = false, workspace = nothing) -> volume

Least Squares QR (LSQR) reconstruction using Golub-Kahan bidiagonalization.

Numerically more stable than CGLS for ill-conditioned inverse problems.

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Maximum number of iterations.
- `init = :zeros`: Initial volume guess (`:zeros`, `:fdk`, or custom 3D array, default: `:zeros`).
- `projection_type::Symbol = :siddon`: Forward projection ray-tracing kernel (`:siddon` or `:interpolated`, default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:fdk`, default: `:matched`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Compute residual L2 norm per iteration (default: `false`).
- `stop_on_l2_increase::Bool = true`: If `true`, terminates and rolls back volume when residual increases (default: `true`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `workspace = nothing`: Optional preallocated `KrylovWorkspace` (default: `nothing`).

# Returns
- `Array{Float32, 3}`: Reconstructed 3D volume (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Reference
- Paige, C. C., & Saunders, M. A., "LSQR: An algorithm for sparse linear equations and sparse least squares", ACM TOMS, 8(1), 1982.

# Examples
```julia
vol = lsqr(proj, geo, 20)
```
"""
function lsqr(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return lsqr!(volume, projection_checked, geo, niter; kwargs...)
end

"""
    lsqr!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; projection_type::Symbol = :siddon, backprojection_type::Symbol = :matched, gpu_ids = nothing, return_history::Bool = false, compute_l2::Bool = false, stop_on_l2_increase::Bool = true, show_progress::Bool = false, workspace = nothing) -> volume

In-place Least Squares QR (LSQR) reconstruction into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Maximum number of iterations.
- `projection_type::Symbol = :siddon`: Forward projection ray-tracing kernel (`:siddon` or `:interpolated`, default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:fdk`, default: `:matched`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Compute residual L2 norm per iteration (default: `false`).
- `stop_on_l2_increase::Bool = true`: If `true`, terminates and rolls back volume when residual increases (default: `true`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `workspace = nothing`: Optional preallocated `KrylovWorkspace` (default: `nothing`).

# Returns
- `volume`: The modified reconstructed volume array (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = zeros(Float32, volume_size(geo))
lsqr!(vol, proj, geo, 20)
```
"""
function lsqr!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer;
    projection_type::Symbol = :siddon,
    backprojection_type::Symbol = :matched,
    gpu_ids = nothing,
    return_history::Bool = false,
    compute_l2::Bool = false,
    stop_on_l2_increase::Bool = true,
    show_progress::Bool = false,
    workspace = nothing,
)
    niter >= 0 || throw(ArgumentError("niter must be >= 0"))
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))

    workspace = _resolve_krylov_workspace(workspace, geo)
    u = workspace.proj_a
    u_prev = workspace.proj_b
    v = workspace.vol_a
    v_prev = workspace.vol_b
    w = workspace.vol_c
    prev_volume = workspace.vol_d
    need_residual = compute_l2 || stop_on_l2_increase || return_history
    residual_l2 = Float32[]
    lambda_history = Float32[]
    prev_residual = Inf32
    stop_reason = :max_iterations
    iterations_completed = 0
    progress = _make_progress(niter, show_progress, "LSQR: ")

    Ax!(u, volume, geo; projection_type, gpu_ids)
    @inbounds for i in eachindex(u, projection)
        u[i] = projection[i] - u[i]
    end
    beta = sqrt(_squared_l2(u))
    beta > 0f0 || return _maybe_return_history(volume, _build_history(residual_l2, lambda_history, 0, :zero_residual), return_history)
    beta0 = beta
    @inbounds for i in eachindex(u)
        u[i] /= beta
    end

    Atb!(v, u, geo; backprojection_type, gpu_ids)
    alpha = sqrt(_squared_l2(v))
    alpha > 0f0 || return _maybe_return_history(volume, _build_history(residual_l2, lambda_history, 0, :zero_residual), return_history)
    @inbounds for i in eachindex(v)
        v[i] /= alpha
        w[i] = v[i]
    end

    phibar = beta
    rhobar = alpha

    for iter in 1:niter
        copyto!(prev_volume, volume)
        copyto!(u_prev, u)
        Ax!(u, v, geo; projection_type, gpu_ids)
        @inbounds for i in eachindex(u)
            u[i] -= alpha * u_prev[i]
        end
        beta = sqrt(_squared_l2(u))
        beta > 0f0 || break
        @inbounds for i in eachindex(u)
            u[i] /= beta
        end

        copyto!(v_prev, v)
        Atb!(v, u, geo; backprojection_type, gpu_ids)
        @inbounds for i in eachindex(v, v_prev)
            v[i] -= beta * v_prev[i]
        end
        alpha = sqrt(_squared_l2(v))
        alpha > 0f0 || break
        @inbounds for i in eachindex(v)
            v[i] /= alpha
        end

        rho = sqrt(rhobar * rhobar + beta * beta)
        c = rhobar / rho
        srot = beta / rho
        theta = srot * alpha
        rhobar = -c * alpha
        phi = c * phibar
        phibar = srot * phibar
        push!(lambda_history, phi / rho)

        @inbounds for i in eachindex(volume, w, v)
            volume[i] += (phi / rho) * w[i]
            w[i] = v[i] - (theta / rho) * w[i]
        end

        iterations_completed = iter
        _next_progress!(progress, iter, niter)
        if need_residual
            residual = Float32(phibar)
            push!(residual_l2, residual)
            if stop_on_l2_increase && residual > prev_residual
                copyto!(volume, prev_volume)
                stop_reason = :l2_increase
                break
            end
            prev_residual = residual
        end
    end

    _finish_progress!(progress, iterations_completed, niter)
    history = _build_history(residual_l2, lambda_history, iterations_completed, stop_reason)
    return _maybe_return_history(volume, history, return_history)
end

"""
    lsmr(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, damp::Real = 0f0, projection_type::Symbol = :siddon, backprojection_type::Symbol = :matched, gpu_ids = nothing, return_history::Bool = false, compute_l2::Bool = false, stop_on_l2_increase::Bool = true, show_progress::Bool = false, workspace = nothing) -> volume

Least Squares Minimum Residual (LSMR) reconstruction.

Minimizes the residual norm `||Aᵀ (b - Ax)||₂` monotonically at each iteration.

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Maximum number of iterations.
- `init = :zeros`: Initial volume guess (`:zeros`, `:fdk`, or custom 3D array, default: `:zeros`).
- `damp::Real = 0.0f0`: Tikhonov damping regularization parameter `λ` solving `min ||Ax - b||₂² + λ² ||x||₂²` (default: `0.0f0`).
- `projection_type::Symbol = :siddon`: Forward projection ray-tracing kernel (`:siddon` or `:interpolated`, default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:fdk`, default: `:matched`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Compute residual L2 norm per iteration (default: `false`).
- `stop_on_l2_increase::Bool = true`: If `true`, terminates and rolls back volume when residual increases (default: `true`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `workspace = nothing`: Optional preallocated `KrylovWorkspace` (default: `nothing`).

# Returns
- `Array{Float32, 3}`: Reconstructed 3D volume (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Reference
- Fong, D. C.-L., & Saunders, M. A., "LSMR: An iterative algorithm for sparse least-squares problems", SIAM J. Sci. Comput., 33(5), 2011.

# Examples
```julia
vol = lsmr(proj, geo, 20; damp = 0.01f0)
```
"""
function lsmr(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return lsmr!(volume, projection_checked, geo, niter; kwargs...)
end

"""
    lsmr!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; damp::Real = 0f0, projection_type::Symbol = :siddon, backprojection_type::Symbol = :matched, gpu_ids = nothing, return_history::Bool = false, compute_l2::Bool = false, stop_on_l2_increase::Bool = true, show_progress::Bool = false, workspace = nothing) -> volume

In-place Least Squares Minimum Residual (LSMR) reconstruction into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT acquisition geometry specification.
- `niter::Integer`: Maximum number of iterations.
- `damp::Real = 0.0f0`: Tikhonov damping regularization parameter `λ` solving `min ||Ax - b||₂² + λ² ||x||₂²` (default: `0.0f0`).
- `projection_type::Symbol = :siddon`: Forward projection ray-tracing kernel (`:siddon` or `:interpolated`, default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:fdk`, default: `:matched`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) or `nothing` for default device (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Compute residual L2 norm per iteration (default: `false`).
- `stop_on_l2_increase::Bool = true`: If `true`, terminates and rolls back volume when residual increases (default: `true`).
- `show_progress::Bool = false`: Display interactive progress bar (default: `false`).
- `workspace = nothing`: Optional preallocated `KrylovWorkspace` (default: `nothing`).

# Returns
- `volume`: The modified reconstructed volume array (or tuple `(volume, history::IterativeHistory)` if `return_history = true`).

# Examples
```julia
vol = zeros(Float32, volume_size(geo))
lsmr!(vol, proj, geo, 20; damp = 0.01f0)
```
"""
function lsmr!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer;
    damp::Real = 0f0,
    projection_type::Symbol = :siddon,
    backprojection_type::Symbol = :matched,
    gpu_ids = nothing,
    return_history::Bool = false,
    compute_l2::Bool = false,
    stop_on_l2_increase::Bool = true,
    show_progress::Bool = false,
    workspace = nothing,
)
    niter >= 0 || throw(ArgumentError("niter must be >= 0"))
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))

    workspace = _resolve_krylov_workspace(workspace, geo)
    u = workspace.proj_a
    proj_tmp = workspace.proj_b
    v = workspace.vol_a
    h = workspace.vol_b
    hbar = workspace.vol_c
    prev_volume = workspace.vol_d
    vol_tmp = workspace.vol_e

    need_residual = compute_l2 || stop_on_l2_increase || return_history
    residual_l2 = Float32[]
    lambda_history = Float32[]
    prev_residual = Inf32
    stop_reason = :max_iterations
    iterations_completed = 0
    progress = _make_progress(niter, show_progress, "LSMR: ")

    Ax!(u, volume, geo; projection_type, gpu_ids)
    @inbounds for i in eachindex(u, projection)
        u[i] = projection[i] - u[i]
    end
    beta = sqrt(_squared_l2(u))
    beta > 0f0 || return _maybe_return_history(volume, _build_history(residual_l2, lambda_history, 0, :zero_residual), return_history)
    beta0 = beta
    @inbounds for i in eachindex(u)
        u[i] /= beta
    end

    Atb!(v, u, geo; backprojection_type, gpu_ids)
    alpha = sqrt(_squared_l2(v))
    alpha > 0f0 || return _maybe_return_history(volume, _build_history(residual_l2, lambda_history, 0, :zero_residual), return_history)
    @inbounds for i in eachindex(v)
        v[i] /= alpha
        h[i] = v[i]
        hbar[i] = 0f0
    end

    alphabar = alpha
    zetabar = alpha * beta
    rho = 1f0
    rhobar = 1f0
    cbar = 1f0
    sbar = 0f0
    betadd = beta
    betad = 0f0
    rhod = 1f0
    tautilda = 0f0
    thetatilda = 0f0
    zeta = 0f0
    damp32 = Float32(damp)
    dacc = 0f0

    for iter in 1:niter
        copyto!(prev_volume, volume)

        Ax!(proj_tmp, v, geo; projection_type, gpu_ids)
        @inbounds for i in eachindex(u, proj_tmp)
            u[i] = proj_tmp[i] - alpha * u[i]
        end
        beta = sqrt(_squared_l2(u))
        beta > 0f0 || break
        @inbounds for i in eachindex(u)
            u[i] /= beta
        end

        Atb!(vol_tmp, u, geo; backprojection_type, gpu_ids)
        @inbounds for i in eachindex(v, vol_tmp)
            v[i] = vol_tmp[i] - beta * v[i]
        end
        alpha = sqrt(_squared_l2(v))
        alpha > 0f0 || break
        @inbounds for i in eachindex(v)
            v[i] /= alpha
        end

        alphahat = sqrt(alphabar * alphabar + damp32 * damp32)
        chat = alphabar / alphahat
        shat = damp32 / alphahat

        rhopre = rho
        rho = sqrt(alphahat * alphahat + beta * beta)
        c = alphahat / rho
        s = beta / rho
        theta = s * alpha
        alphabar = c * alpha

        thetabar = sbar * rho
        rhobarpre = rhobar
        rhobar = sqrt((cbar * rho)^2 + theta^2)
        cbar = cbar * rho / rhobar
        sbar = theta / rhobar
        zetapre = zeta
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar
        push!(lambda_history, zeta / max(rho * rhobar, eps(Float32)))

        coeff = (thetabar * rho) / max(rhopre * rhobarpre, eps(Float32))
        update_scale = zeta / max(rho * rhobar, eps(Float32))
        @inbounds for i in eachindex(hbar, h, volume)
            hbar[i] = h[i] - coeff * hbar[i]
            volume[i] += update_scale * hbar[i]
        end
        @inbounds for i in eachindex(h, v)
            h[i] = v[i] - (theta / rho) * h[i]
        end

        betaacute = chat * betadd
        betacheck = -shat * betadd
        betahat = c * betaacute
        betadd = -s * betaacute

        rhotilda = sqrt(rhod * rhod + thetabar * thetabar)
        ctilda = rhod / rhotilda
        stilda = thetabar / rhotilda
        thetatildapre = thetatilda
        thetatilda = stilda * rhobar
        rhod = ctilda * rhobar
        betad = -stilda * betad + ctilda * betahat
        tautilda = (zetapre - thetatildapre * tautilda) / rhotilda
        taud = (zeta - thetatilda * tautilda) / rhod
        dacc += betacheck * betacheck
        gamma_var = dacc + (betad - taud)^2 + betadd^2
        isfinite(gamma_var) || break

        iterations_completed = iter
        _next_progress!(progress, iter, niter)
        if need_residual
            residual = _projection_residual_l2!(proj_tmp, projection, volume, geo, projection_type, gpu_ids)
            push!(residual_l2, residual)
            if stop_on_l2_increase && residual > prev_residual
                copyto!(volume, prev_volume)
                stop_reason = :l2_increase
                break
            end
            prev_residual = residual
        end
    end

    _finish_progress!(progress, iterations_completed, niter)
    history = _build_history(residual_l2, lambda_history, iterations_completed, stop_reason)
    return _maybe_return_history(volume, history, return_history)
end

"""
    hybrid_lsqr(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, lambda = 1f0, projection_type = :siddon, backprojection_type = :matched, gpu_ids = nothing, return_history = false, compute_l2 = false, stop_on_l2_increase = true, show_progress = false, workspace = nothing) -> volume
    hybrid_lsqr!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; lambda = 1f0, projection_type = :siddon, backprojection_type = :matched, gpu_ids = nothing, return_history = false, compute_l2 = false, stop_on_l2_increase = true, show_progress = false, workspace = nothing) -> volume

Hybrid LSQR with subspace projection and Tikhonov regularization in the projected Krylov subspace.

Solves large-scale tomographic linear inverse problems by generating a Golub-Kahan bidiagonalization
Krylov subspace, then solving a projected Tikhonov regularized subproblem at each iteration:
```
min_y || B_k y - beta e_1 ||_2^2 + lambda^2 || y ||_2^2,   x_k = x_0 + V_k y
```

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT geometry specification.
- `niter::Integer`: Number of Krylov subspace projection iterations.

# Keyword Arguments
- `init = :zeros`: Initial image volume or initialization strategy (`:zeros`, `:fdk`, or a 3D `Array{Float32, 3}`) (default: `:zeros`). (Only in `hybrid_lsqr`)
- `lambda::Real = 1f0`: Projected Tikhonov regularization parameter (default: `1f0`).
- `projection_type::Symbol = :siddon`: Forward projection algorithm (`:siddon` or `:ray_voxel`) (default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:voxel_cone`) (default: `:matched`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) to use (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Track residual L2 norm history across iterations (default: `false`).
- `stop_on_l2_increase::Bool = true`: Early stop if residual L2 norm strictly increases (default: `true`).
- `show_progress::Bool = false`: Display progress bar (default: `false`).
- `workspace::Union{Nothing, KrylovWorkspace} = nothing`: Preallocated `KrylovWorkspace` for zero-allocation loops (default: `nothing`).

# Returns
- Reconstructed 3D volume `Array{Float32, 3}` of size `(geo.nVoxel[1], geo.nVoxel[2], geo.nVoxel[3])`, or `(volume, history::IterativeHistory)` if `return_history=true`.

# Examples
```julia
# 30 iterations of Hybrid LSQR
recon = hybrid_lsqr(proj, geo, 30; lambda=0.5f0)

# In-place with workspace reuse
ws = prepare_krylov_workspace(geo)
hybrid_lsqr!(recon, proj, geo, 30; lambda=0.5f0, workspace=ws)
```

# Reference
- Chung, J. M., & Palmer, K., "A hybrid LSQR algorithm for large-scale inverse problems", SISC, 37(5), 2015.
"""
function hybrid_lsqr(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return hybrid_lsqr!(volume, projection_checked, geo, niter; kwargs...)
end

"""
    hybrid_lsqr!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; lambda = 1f0, projection_type = :siddon, backprojection_type = :matched, gpu_ids = nothing, return_history = false, compute_l2 = false, stop_on_l2_increase = true, show_progress = false, workspace = nothing) -> volume

In-place Hybrid LSQR reconstruction into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT geometry specification.
- `niter::Integer`: Number of Krylov subspace projection iterations.

# Keyword Arguments
- `lambda::Real = 1f0`: Projected Tikhonov regularization parameter (default: `1f0`).
- `projection_type::Symbol = :siddon`: Forward projection algorithm (`:siddon` or `:ray_voxel`) (default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:voxel_cone`) (default: `:matched`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) to use (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Track residual L2 norm history across iterations (default: `false`).
- `stop_on_l2_increase::Bool = true`: Early stop if residual L2 norm strictly increases (default: `true`).
- `show_progress::Bool = false`: Display progress bar (default: `false`).
- `workspace::Union{Nothing, KrylovWorkspace} = nothing`: Preallocated `KrylovWorkspace` (default: `nothing`).

# Returns
- The mutated `volume` array of size `(geo.nVoxel[1], geo.nVoxel[2], geo.nVoxel[3])`, or `(volume, history::IterativeHistory)` if `return_history=true`.

# Examples
```julia
ws = prepare_krylov_workspace(geo)
hybrid_lsqr!(recon, proj, geo, 30; lambda=0.5f0, workspace=ws)
```
"""
function hybrid_lsqr!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer;
    lambda::Real = 1f0,
    projection_type::Symbol = :siddon,
    backprojection_type::Symbol = :matched,
    gpu_ids = nothing,
    return_history::Bool = false,
    compute_l2::Bool = false,
    stop_on_l2_increase::Bool = true,
    show_progress::Bool = false,
    workspace = nothing,
)
    niter >= 0 || throw(ArgumentError("niter must be >= 0"))
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))

    workspace = _resolve_krylov_workspace(workspace, geo)
    u = workspace.proj_a
    proj_tmp = workspace.proj_b
    v = workspace.vol_a
    trial = workspace.vol_b
    base_volume = workspace.vol_c
    prev_volume = workspace.vol_d

    need_residual = compute_l2 || stop_on_l2_increase || return_history
    residual_l2 = Float32[]
    lambda_history = Float32[]
    prev_residual = Inf32
    stop_reason = :max_iterations
    iterations_completed = 0
    progress = _make_progress(niter, show_progress, "Hybrid LSQR: ")

    subspace = _resolve_subspace_workspace(workspace, geo, niter)
    Umat = subspace.proj_basis
    Vmat = subspace.vol_basis
    Bmat = subspace.mat_a
    normal = subspace.mat_b
    chol = subspace.mat_c
    coeffs = subspace.coeffs
    rhs_proj = subspace.coeffs_aux

    Ax!(u, volume, geo; projection_type, gpu_ids)
    @inbounds for i in eachindex(u, projection)
        u[i] = projection[i] - u[i]
    end
    beta = sqrt(_squared_l2(u))
    beta > 0f0 || return _maybe_return_history(volume, _build_history(residual_l2, lambda_history, 0, :zero_residual), return_history)
    beta0 = beta
    @inbounds for i in eachindex(u)
        u[i] /= beta
    end
    _store_proj_basis_row!(Umat, 1, u)
    copyto!(base_volume, volume)

    lambda32 = Float32(lambda)
    for iter in 1:niter
        copyto!(prev_volume, volume)
        Atb!(v, u, geo; backprojection_type, gpu_ids)
        if iter > 1
            @inbounds for i in eachindex(v)
                v[i] -= beta * Vmat[iter - 1, i]
            end
        end
        if iter > 2
            for j in 1:iter-2
                coeff = _basis_dot(v, Vmat, j)
                @inbounds for i in eachindex(v)
                    v[i] -= coeff * Vmat[j, i]
                end
            end
        end

        alpha = sqrt(_squared_l2(v))
        alpha > 0f0 || break
        @inbounds for i in eachindex(v)
            v[i] /= alpha
        end
        _store_basis_row!(Vmat, iter, v)

        Ax!(u, v, geo; projection_type, gpu_ids)
        @inbounds for i in eachindex(u)
            u[i] -= alpha * Umat[iter, i]
        end
        if iter > 2
            for j in 1:iter-2
                coeff = _proj_basis_dot(u, Umat, j)
                @inbounds for i in eachindex(u)
                    u[i] -= coeff * Umat[j, i]
                end
            end
        end

        beta = sqrt(_squared_l2(u))
        beta > 0f0 || break
        @inbounds for i in eachindex(u)
            u[i] /= beta
        end
        _store_proj_basis_row!(Umat, iter + 1, u)

        Bmat[iter, iter] = alpha
        Bmat[iter, iter + 1] = beta

        _update_hybrid_lsqr_system!(normal, rhs_proj, Bmat, beta0, lambda32, iter)
        _append_cholesky_column!(chol, normal, iter)
        @inbounds for i in 1:iter
            coeffs[i] = rhs_proj[i]
        end
        _solve_cholesky!(coeffs, chol, iter)
        _apply_basis_combination!(trial, base_volume, Vmat, coeffs, iter)
        copyto!(volume, trial)
        push!(lambda_history, lambda32)

        iterations_completed = iter
        _next_progress!(progress, iter, niter)
        if need_residual
            residual = _projection_residual_l2!(proj_tmp, projection, volume, geo, projection_type, gpu_ids)
            push!(residual_l2, residual)
            if stop_on_l2_increase && residual > prev_residual
                copyto!(volume, prev_volume)
                stop_reason = :l2_increase
                break
            end
            prev_residual = residual
        end
    end

    _finish_progress!(progress, iterations_completed, niter)
    history = _build_history(residual_l2, lambda_history, iterations_completed, stop_reason)
    return _maybe_return_history(volume, history, return_history)
end

"""
    irn_tv_cgls(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, lambda = 1f0, niter_outer = 4, projection_type = :siddon, backprojection_type = :matched, gpu_ids = nothing, return_history = false, compute_l2 = false, stop_on_l2_increase = true, show_progress = false, workspace = nothing) -> volume
    irn_tv_cgls!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; lambda = 1f0, niter_outer = 4, projection_type = :siddon, backprojection_type = :matched, gpu_ids = nothing, return_history = false, compute_l2 = false, stop_on_l2_increase = true, show_progress = false, workspace = nothing) -> volume

Iteratively Reweighted Norm Total Variation CGLS (IRN-TV-CGLS).

Solves the TV-regularized least-squares problem:
```
min_x || A x - b ||_2^2 + lambda || |grad x| ||_1
```
via outer iteratively reweighted norm loops (half-quadratic regularization) with inner CGLS iterations.

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT geometry specification.
- `niter::Integer`: Number of inner CGLS iterations per outer IRN loop.

# Keyword Arguments
- `init = :zeros`: Initial image volume or initialization strategy (`:zeros`, `:fdk`, or a 3D `Array{Float32, 3}`) (default: `:zeros`). (Only in `irn_tv_cgls`)
- `lambda::Real = 1f0`: TV regularization weight parameter (default: `1f0`).
- `niter_outer::Integer = 4`: Number of outer IRN weight recomputation iterations (default: `4`).
- `projection_type::Symbol = :siddon`: Forward projection algorithm (`:siddon` or `:ray_voxel`) (default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:voxel_cone`) (default: `:matched`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) to use (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Track residual L2 norm history across iterations (default: `false`).
- `stop_on_l2_increase::Bool = true`: Early stop if residual L2 norm strictly increases (default: `true`).
- `show_progress::Bool = false`: Display progress bar (default: `false`).
- `workspace::Union{Nothing, KrylovWorkspace} = nothing`: Preallocated `KrylovWorkspace` (default: `nothing`).

# Returns
- Reconstructed 3D volume `Array{Float32, 3}` of size `(geo.nVoxel[1], geo.nVoxel[2], geo.nVoxel[3])`, or `(volume, history::IterativeHistory)` if `return_history=true`.

# Examples
```julia
recon = irn_tv_cgls(proj, geo, 10; niter_outer=4, lambda=0.1f0)
```

# Reference
- Rodriguez, P., & Wohlberg, B., "An iteratively reweighted norm algorithm for total variation regularization", IEEE ACSSC, 2006.
"""
function irn_tv_cgls(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return irn_tv_cgls!(volume, projection_checked, geo, niter; kwargs...)
end

"""
    irn_tv_cgls!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; lambda = 1f0, niter_outer = 4, projection_type = :siddon, backprojection_type = :matched, gpu_ids = nothing, return_history = false, compute_l2 = false, stop_on_l2_increase = true, show_progress = false, workspace = nothing) -> volume

In-place Iteratively Reweighted Norm Total Variation CGLS (IRN-TV-CGLS) reconstruction into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT geometry specification.
- `niter::Integer`: Number of inner CGLS iterations per outer IRN loop.

# Keyword Arguments
- `lambda::Real = 1f0`: TV regularization weight parameter (default: `1f0`).
- `niter_outer::Integer = 4`: Number of outer IRN weight recomputation iterations (default: `4`).
- `projection_type::Symbol = :siddon`: Forward projection algorithm (`:siddon` or `:ray_voxel`) (default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:voxel_cone`) (default: `:matched`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) to use (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Track residual L2 norm history across iterations (default: `false`).
- `stop_on_l2_increase::Bool = true`: Early stop if residual L2 norm strictly increases (default: `true`).
- `show_progress::Bool = false`: Display progress bar (default: `false`).
- `workspace::Union{Nothing, KrylovWorkspace} = nothing`: Preallocated `KrylovWorkspace` (default: `nothing`).

# Returns
- The mutated `volume` array of size `(geo.nVoxel[1], geo.nVoxel[2], geo.nVoxel[3])`, or `(volume, history::IterativeHistory)` if `return_history=true`.

# Examples
```julia
irn_tv_cgls!(recon, proj, geo, 10; niter_outer=4, lambda=0.1f0)
```
"""
function irn_tv_cgls!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer;
    lambda::Real = 1f0,
    niter_outer::Integer = 4,
    projection_type::Symbol = :siddon,
    backprojection_type::Symbol = :matched,
    gpu_ids = nothing,
    return_history::Bool = false,
    compute_l2::Bool = false,
    stop_on_l2_increase::Bool = true,
    show_progress::Bool = false,
    workspace = nothing,
)
    niter >= 0 || throw(ArgumentError("niter must be >= 0"))
    niter_outer >= 0 || throw(ArgumentError("niter_outer must be >= 0"))
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))

    workspace = _resolve_krylov_workspace(workspace, geo)
    proj_r = workspace.proj_a
    proj_q = workspace.proj_b
    p = workspace.vol_a
    s = workspace.vol_b
    weights = workspace.vol_c
    prev_volume = workspace.vol_d
    tmp = workspace.vol_e

    gx = _extra_volume(workspace, 1, geo)
    gy = _extra_volume(workspace, 2, geo)
    gz = _extra_volume(workspace, 3, geo)
    qgx = _extra_volume(workspace, 4, geo)
    qgy = _extra_volume(workspace, 5, geo)
    qgz = _extra_volume(workspace, 6, geo)
    base_volume = _extra_volume(workspace, 7, geo)

    need_residual = compute_l2 || stop_on_l2_increase || return_history
    residual_l2 = Float32[]
    lambda_history = Float32[]
    stop_reason = :max_iterations
    iterations_completed = 0
    lambda_scale = sqrt(Float32(lambda))
    total_iterations = niter * niter_outer
    progress = _make_progress(total_iterations, show_progress, "IRN-TV-CGLS: ")

    copyto!(base_volume, volume)
    best_res = Inf32

    for outer in 1:niter_outer
        _weighted_tv_weights!(weights, volume)
        copyto!(volume, base_volume)

        Ax!(proj_q, volume, geo; projection_type, gpu_ids)
        @inbounds for i in eachindex(proj_r, projection, proj_q)
            proj_r[i] = projection[i] - proj_q[i]
        end
        _weighted_tv_grad!(gx, gy, gz, weights, volume, lambda_scale)
        @inbounds for i in eachindex(gx, gy, gz)
            gx[i] = -gx[i]
            gy[i] = -gy[i]
            gz[i] = -gz[i]
        end
        Atb!(p, proj_r, geo; backprojection_type, gpu_ids)
        _weighted_tv_divergence!(tmp, gx, gy, gz, lambda_scale)
        @inbounds for i in eachindex(p, tmp)
            p[i] += tmp[i]
        end
        gamma = _squared_l2(p)

        res_prev_inner = Inf32
        res_now = Inf32

        for inner in 1:niter
            copyto!(prev_volume, volume)
            Ax!(proj_q, p, geo; projection_type, gpu_ids)
            _weighted_tv_grad!(qgx, qgy, qgz, weights, p, lambda_scale)
            qnorm2 = _combined_norm2(proj_q, qgx, qgy, qgz)
            qnorm2 > 0f0 || break
            alpha = gamma / qnorm2
            push!(lambda_history, alpha)
            @inbounds for i in eachindex(volume, p)
                volume[i] += alpha * p[i]
            end
            @inbounds for i in eachindex(proj_r, proj_q)
                proj_r[i] -= alpha * proj_q[i]
            end
            @inbounds for i in eachindex(gx, qgx)
                gx[i] -= alpha * qgx[i]
                gy[i] -= alpha * qgy[i]
                gz[i] -= alpha * qgz[i]
            end

            iterations_completed += 1
            if need_residual
                residual = Float32(sqrt(_squared_l2(proj_r)))
                push!(residual_l2, residual)
                res_now = residual
                if inner > 1 && stop_on_l2_increase && residual > res_prev_inner
                    copyto!(volume, prev_volume)
                    residual_l2[end] = res_prev_inner
                    res_now = res_prev_inner
                    break
                end
                res_prev_inner = residual
            end
            _next_progress!(progress, iterations_completed, total_iterations)

            Atb!(s, proj_r, geo; backprojection_type, gpu_ids)
            _weighted_tv_divergence!(tmp, gx, gy, gz, lambda_scale)
            @inbounds for i in eachindex(s, tmp)
                s[i] += tmp[i]
            end
            gamma1 = _squared_l2(s)
            gamma > 0f0 || break
            beta = gamma1 / gamma
            @inbounds for i in eachindex(p, s)
                p[i] = s[i] + beta * p[i]
            end
            gamma = gamma1
        end

        copyto!(base_volume, volume)

        if need_residual && stop_on_l2_increase
            if res_now >= best_res
                stop_reason = :l2_increase
                break
            end
            best_res = res_now
        end
    end

    _finish_progress!(progress, iterations_completed, total_iterations)
    history = _build_history(residual_l2, lambda_history, iterations_completed, stop_reason)
    return _maybe_return_history(volume, history, return_history)
end

"""
    ab_gmres(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, projection_type = :siddon, backprojection_type = :matched, backprojector = nothing, gpu_ids = nothing, return_history = false, compute_l2 = false, stop_on_l2_increase = true, show_progress = false, workspace = nothing) -> volume
    ab_gmres!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; projection_type = :siddon, backprojection_type = :matched, backprojector = nothing, gpu_ids = nothing, return_history = false, compute_l2 = false, stop_on_l2_increase = true, show_progress = false, workspace = nothing) -> volume

AB-GMRES (Right-Preconditioned Generalized Minimal Residual) reconstruction.

Applies GMRES to solve `A B y = b` with `x = B y`, where `B` represents the backprojector
acting as a right preconditioner in projection space.

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT geometry specification.
- `niter::Integer`: Number of GMRES Krylov subspace iterations.

# Keyword Arguments
- `init = :zeros`: Initial image volume or initialization strategy (`:zeros`, `:fdk`, or a 3D `Array{Float32, 3}`) (default: `:zeros`). (Only in `ab_gmres`)
- `projection_type::Symbol = :siddon`: Forward projection algorithm (`:siddon` or `:ray_voxel`) (default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:voxel_cone`) (default: `:matched`).
- `backprojector = nothing`: Optional custom backprojector symbol (`:matched`, `:voxel_cone`, `:fdk`), overriding `backprojection_type` (default: `nothing`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) to use (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Track residual L2 norm history across iterations (default: `false`).
- `stop_on_l2_increase::Bool = true`: Early stop if residual L2 norm strictly increases (default: `true`).
- `show_progress::Bool = false`: Display progress bar (default: `false`).
- `workspace::Union{Nothing, KrylovWorkspace} = nothing`: Preallocated `KrylovWorkspace` (default: `nothing`).

# Returns
- Reconstructed 3D volume `Array{Float32, 3}` of size `(geo.nVoxel[1], geo.nVoxel[2], geo.nVoxel[3])`, or `(volume, history::IterativeHistory)` if `return_history=true`.

# Examples
```julia
recon = ab_gmres(proj, geo, 20)
```

# Reference
- Calvetti, D., Lewis, B., & Reichel, L., "GMRES-type methods for inconsistent systems", Linear Algebra Appl., 316(1-3), 2000.
"""
function ab_gmres(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return ab_gmres!(volume, projection_checked, geo, niter; kwargs...)
end

"""
    ab_gmres!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; projection_type = :siddon, backprojection_type = :matched, backprojector = nothing, gpu_ids = nothing, return_history = false, compute_l2 = false, stop_on_l2_increase = true, show_progress = false, workspace = nothing) -> volume

In-place AB-GMRES (Right-Preconditioned Generalized Minimal Residual) reconstruction into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT geometry specification.
- `niter::Integer`: Number of GMRES Krylov subspace iterations.

# Keyword Arguments
- `projection_type::Symbol = :siddon`: Forward projection algorithm (`:siddon` or `:ray_voxel`) (default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:voxel_cone`) (default: `:matched`).
- `backprojector = nothing`: Optional custom backprojector symbol (`:matched`, `:voxel_cone`, `:fdk`), overriding `backprojection_type` (default: `nothing`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) to use (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Track residual L2 norm history across iterations (default: `false`).
- `stop_on_l2_increase::Bool = true`: Early stop if residual L2 norm strictly increases (default: `true`).
- `show_progress::Bool = false`: Display progress bar (default: `false`).
- `workspace::Union{Nothing, KrylovWorkspace} = nothing`: Preallocated `KrylovWorkspace` (default: `nothing`).

# Returns
- The mutated `volume` array of size `(geo.nVoxel[1], geo.nVoxel[2], geo.nVoxel[3])`, or `(volume, history::IterativeHistory)` if `return_history=true`.

# Examples
```julia
ab_gmres!(recon, proj, geo, 20)
```
"""
function ab_gmres!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer;
    projection_type::Symbol = :siddon,
    backprojection_type::Symbol = :matched,
    backprojector = nothing,
    gpu_ids = nothing,
    return_history::Bool = false,
    compute_l2::Bool = false,
    stop_on_l2_increase::Bool = true,
    show_progress::Bool = false,
    workspace = nothing,
)
    niter >= 0 || throw(ArgumentError("niter must be >= 0"))
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))
    selected_backprojector = backprojector === nothing ? backprojection_type : backprojector

    workspace = _resolve_krylov_workspace(workspace, geo)
    r = workspace.proj_a
    q = workspace.proj_b
    z = workspace.vol_a
    trial = workspace.vol_b
    base_volume = workspace.vol_c
    prev_volume = workspace.vol_d

    need_residual = compute_l2 || stop_on_l2_increase || return_history
    residual_l2 = Float32[]
    lambda_history = Float32[]
    prev_residual = Inf32
    stop_reason = :max_iterations
    iterations_completed = 0
    progress = _make_progress(niter, show_progress, "AB-GMRES: ")

    subspace = _resolve_subspace_workspace(workspace, geo, niter)
    Wbasis = subspace.proj_basis
    Zbasis = subspace.vol_basis
    H = subspace.mat_a
    lsq_r = subspace.mat_c
    rhs = subspace.rhs
    coeffs = subspace.coeffs
    work_rhs = subspace.coeffs_aux
    fill!(H, 0f0)
    fill!(rhs, 0f0)

    Ax!(r, volume, geo; projection_type, gpu_ids)
    @inbounds for i in eachindex(r, projection)
        r[i] = projection[i] - r[i]
    end
    beta = sqrt(_squared_l2(r))
    beta > 0f0 || return _maybe_return_history(volume, _build_history(residual_l2, lambda_history, 0, :zero_residual), return_history)
    @inbounds for i in eachindex(r)
        r[i] /= beta
    end
    _store_proj_basis_row!(Wbasis, 1, r)
    copyto!(base_volume, volume)

    for k in 1:niter
        copyto!(prev_volume, volume)
        _apply_backproject!(z, r, geo, selected_backprojector, gpu_ids)
        _store_basis_row!(Zbasis, k, z)
        Ax!(q, z, geo; projection_type, gpu_ids)

        for j in 1:k
            H[j, k] = _proj_basis_dot(q, Wbasis, j)
            idx = 1
            @inbounds for ia in 1:size(q, 3), iv in 1:size(q, 2), iu in 1:size(q, 1)
                q[iu, iv, ia] -= H[j, k] * Wbasis[j, idx]
                idx += 1
            end
        end
        H[k + 1, k] = sqrt(_squared_l2(q))
        H[k + 1, k] > 0f0 || break
        @inbounds for i in eachindex(q)
            q[i] /= H[k + 1, k]
        end
        _store_proj_basis_row!(Wbasis, k + 1, q)

        rhs[1] = beta
        @inbounds for i in 2:k+1
            rhs[i] = 0f0
        end
        Hk = @view H[1:k+1, 1:k]
        _least_squares_givens!(coeffs, Hk, rhs, k + 1, k, lsq_r, work_rhs)
        _apply_basis_combination!(trial, base_volume, Zbasis, coeffs, k)
        copyto!(volume, trial)
        push!(lambda_history, one(Float32))
        iterations_completed = k
        _next_progress!(progress, k, niter)

        if need_residual
            residual = _projection_residual_l2!(r, projection, volume, geo, projection_type, gpu_ids)
            push!(residual_l2, residual)
            if stop_on_l2_increase && residual > prev_residual
                copyto!(volume, prev_volume)
                stop_reason = :l2_increase
                break
            end
            prev_residual = residual
        end

        copyto!(r, q)
    end

    _finish_progress!(progress, iterations_completed, niter)
    history = _build_history(residual_l2, lambda_history, iterations_completed, stop_reason)
    return _maybe_return_history(volume, history, return_history)
end

"""
    ba_gmres(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, projection_type = :siddon, backprojection_type = :matched, backprojector = nothing, gpu_ids = nothing, return_history = false, compute_l2 = false, stop_on_l2_increase = true, show_progress = false, workspace = nothing) -> volume
    ba_gmres!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; projection_type = :siddon, backprojection_type = :matched, backprojector = nothing, gpu_ids = nothing, return_history = false, compute_l2 = false, stop_on_l2_increase = true, show_progress = false, workspace = nothing) -> volume

BA-GMRES (Left-Preconditioned Generalized Minimal Residual) reconstruction.

Applies GMRES to solve `B A x = B b`, where `B` represents the backprojector
acting as a left preconditioner in image space.

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT geometry specification.
- `niter::Integer`: Number of GMRES Krylov subspace iterations.

# Keyword Arguments
- `init = :zeros`: Initial image volume or initialization strategy (`:zeros`, `:fdk`, or a 3D `Array{Float32, 3}`) (default: `:zeros`). (Only in `ba_gmres`)
- `projection_type::Symbol = :siddon`: Forward projection algorithm (`:siddon` or `:ray_voxel`) (default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:voxel_cone`) (default: `:matched`).
- `backprojector = nothing`: Optional custom backprojector symbol (`:matched`, `:voxel_cone`, `:fdk`), overriding `backprojection_type` (default: `nothing`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) to use (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Track residual L2 norm history across iterations (default: `false`).
- `stop_on_l2_increase::Bool = true`: Early stop if residual L2 norm strictly increases (default: `true`).
- `show_progress::Bool = false`: Display progress bar (default: `false`).
- `workspace::Union{Nothing, KrylovWorkspace} = nothing`: Preallocated `KrylovWorkspace` (default: `nothing`).

# Returns
- Reconstructed 3D volume `Array{Float32, 3}` of size `(geo.nVoxel[1], geo.nVoxel[2], geo.nVoxel[3])`, or `(volume, history::IterativeHistory)` if `return_history=true`.

# Examples
```julia
recon = ba_gmres(proj, geo, 20)
```
"""
function ba_gmres(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return ba_gmres!(volume, projection_checked, geo, niter; kwargs...)
end

"""
    ba_gmres!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; projection_type = :siddon, backprojection_type = :matched, backprojector = nothing, gpu_ids = nothing, return_history = false, compute_l2 = false, stop_on_l2_increase = true, show_progress = false, workspace = nothing) -> volume

In-place BA-GMRES (Left-Preconditioned Generalized Minimal Residual) reconstruction into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT geometry specification.
- `niter::Integer`: Number of GMRES Krylov subspace iterations.

# Keyword Arguments
- `projection_type::Symbol = :siddon`: Forward projection algorithm (`:siddon` or `:ray_voxel`) (default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:voxel_cone`) (default: `:matched`).
- `backprojector = nothing`: Optional custom backprojector symbol (`:matched`, `:voxel_cone`, `:fdk`), overriding `backprojection_type` (default: `nothing`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) to use (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Track residual L2 norm history across iterations (default: `false`).
- `stop_on_l2_increase::Bool = true`: Early stop if residual L2 norm strictly increases (default: `true`).
- `show_progress::Bool = false`: Display progress bar (default: `false`).
- `workspace::Union{Nothing, KrylovWorkspace} = nothing`: Preallocated `KrylovWorkspace` (default: `nothing`).

# Returns
- The mutated `volume` array of size `(geo.nVoxel[1], geo.nVoxel[2], geo.nVoxel[3])`, or `(volume, history::IterativeHistory)` if `return_history=true`.

# Examples
```julia
ba_gmres!(recon, proj, geo, 20)
```
"""
function ba_gmres!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer;
    projection_type::Symbol = :siddon,
    backprojection_type::Symbol = :matched,
    backprojector = nothing,
    gpu_ids = nothing,
    return_history::Bool = false,
    compute_l2::Bool = false,
    stop_on_l2_increase::Bool = true,
    show_progress::Bool = false,
    workspace = nothing,
)
    niter >= 0 || throw(ArgumentError("niter must be >= 0"))
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))
    selected_backprojector = backprojector === nothing ? backprojection_type : backprojector

    workspace = _resolve_krylov_workspace(workspace, geo)
    proj_tmp = workspace.proj_a
    qproj = workspace.proj_b
    r = workspace.vol_a
    q = workspace.vol_b
    trial = workspace.vol_c
    base_volume = workspace.vol_d
    prev_volume = workspace.vol_e

    need_residual = compute_l2 || stop_on_l2_increase || return_history
    residual_l2 = Float32[]
    lambda_history = Float32[]
    prev_residual = Inf32
    stop_reason = :max_iterations
    iterations_completed = 0
    progress = _make_progress(niter, show_progress, "BA-GMRES: ")

    subspace = _resolve_subspace_workspace(workspace, geo, niter)
    Wbasis = subspace.vol_basis
    H = subspace.mat_a
    lsq_r = subspace.mat_c
    rhs = subspace.rhs
    coeffs = subspace.coeffs
    work_rhs = subspace.coeffs_aux
    Ax!(proj_tmp, volume, geo; projection_type, gpu_ids)
    @inbounds for i in eachindex(proj_tmp, projection)
        proj_tmp[i] = projection[i] - proj_tmp[i]
    end
    _apply_backproject!(r, proj_tmp, geo, selected_backprojector, gpu_ids)
    beta = sqrt(_squared_l2(r))
    beta > 0f0 || return _maybe_return_history(volume, _build_history(residual_l2, lambda_history, 0, :zero_residual), return_history)
    @inbounds for i in eachindex(r)
        r[i] /= beta
    end
    _store_basis_row!(Wbasis, 1, r)
    copyto!(base_volume, volume)

    for k in 1:niter
        copyto!(prev_volume, volume)
        Ax!(qproj, r, geo; projection_type, gpu_ids)
        _apply_backproject!(q, qproj, geo, selected_backprojector, gpu_ids)

        for j in 1:k
            H[j, k] = _basis_dot(q, Wbasis, j)
            idx = 1
            @inbounds for zidx in 1:size(q, 3), y in 1:size(q, 2), x in 1:size(q, 1)
                q[x, y, zidx] -= H[j, k] * Wbasis[j, idx]
                idx += 1
            end
        end
        H[k + 1, k] = sqrt(_squared_l2(q))
        H[k + 1, k] > 0f0 || break
        @inbounds for i in eachindex(q)
            q[i] /= H[k + 1, k]
        end
        _store_basis_row!(Wbasis, k + 1, q)

        rhs[1] = beta
        @inbounds for i in 2:k+1
            rhs[i] = 0f0
        end
        Hk = @view H[1:k+1, 1:k]
        _least_squares_givens!(coeffs, Hk, rhs, k + 1, k, lsq_r, work_rhs)
        _apply_basis_combination!(trial, base_volume, Wbasis, coeffs, k)
        copyto!(volume, trial)
        push!(lambda_history, one(Float32))
        iterations_completed = k
        _next_progress!(progress, k, niter)

        if need_residual
            residual = _projection_residual_l2!(proj_tmp, projection, volume, geo, projection_type, gpu_ids)
            push!(residual_l2, residual)
            if stop_on_l2_increase && residual > prev_residual
                copyto!(volume, prev_volume)
                stop_reason = :l2_increase
                break
            end
            prev_residual = residual
        end

        copyto!(r, q)
    end

    _finish_progress!(progress, iterations_completed, niter)
    history = _build_history(residual_l2, lambda_history, iterations_completed, stop_reason)
    return _maybe_return_history(volume, history, return_history)
end

"""
    hybrid_flsqr_tv(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, lambda = 1f0, precond_steps = 6, projection_type = :siddon, backprojection_type = :matched, gpu_ids = nothing, return_history = false, compute_l2 = false, stop_on_l2_increase = true, show_progress = false, workspace = nothing) -> volume
    hybrid_flsqr_tv!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; lambda = 1f0, precond_steps = 6, projection_type = :siddon, backprojection_type = :matched, gpu_ids = nothing, return_history = false, compute_l2 = false, stop_on_l2_increase = true, show_progress = false, workspace = nothing) -> volume

Hybrid Flexible LSQR with Total Variation regularization (Hybrid-fLSQR-TV).

Combines flexible Krylov subspace projection with inner TV regularization steps to achieve fast
convergence with edge-preserving regularization.

# Arguments
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT geometry specification.
- `niter::Integer`: Number of outer flexible LSQR iterations.

# Keyword Arguments
- `init = :zeros`: Initial image volume or initialization strategy (`:zeros`, `:fdk`, or a 3D `Array{Float32, 3}`) (default: `:zeros`). (Only in `hybrid_flsqr_tv`)
- `lambda::Real = 1f0`: Regularization weight parameter (default: `1f0`).
- `precond_steps::Integer = 6`: Number of inner weighted TV conjugate gradient steps (default: `6`).
- `projection_type::Symbol = :siddon`: Forward projection algorithm (`:siddon` or `:ray_voxel`) (default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:voxel_cone`) (default: `:matched`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) to use (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Track residual L2 norm history across iterations (default: `false`).
- `stop_on_l2_increase::Bool = true`: Early stop if residual L2 norm strictly increases (default: `true`).
- `show_progress::Bool = false`: Display progress bar (default: `false`).
- `workspace::Union{Nothing, KrylovWorkspace} = nothing`: Preallocated `KrylovWorkspace` (default: `nothing`).

# Returns
- Reconstructed 3D volume `Array{Float32, 3}` of size `(geo.nVoxel[1], geo.nVoxel[2], geo.nVoxel[3])`, or `(volume, history::IterativeHistory)` if `return_history=true`.

# Examples
```julia
recon = hybrid_flsqr_tv(proj, geo, 15; lambda=0.1f0, precond_steps=6)
```

# Reference
- Chung, J., & Palmer, K., "Flexible Krylov subspace methods for ℓ_p regularization", SISC, 2015.
"""
function hybrid_flsqr_tv(projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; init = :zeros, kwargs...)
    volume = Array{Float32}(undef, volume_size(geo))
    projection_checked = _require_canonical_array(projection, "projection", projection_size(geo))
    _initialize_iterative_volume!(volume, projection_checked, geo, init; gpu_ids = get(kwargs, :gpu_ids, nothing))
    return hybrid_flsqr_tv!(volume, projection_checked, geo, niter; kwargs...)
end

# Upstream Python hybrid_fLSQR_TV uses SciPy lsqr for the inner solve.
# Tigre.jl replaces this with a weighted-TV small CG to avoid Python dependency.
# Therefore, while the API and outer-loop roles match, the inner solve is an intentional difference rather than an exact replica.
"""
    hybrid_flsqr_tv!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer; lambda = 1f0, precond_steps = 6, projection_type = :siddon, backprojection_type = :matched, gpu_ids = nothing, return_history = false, compute_l2 = false, stop_on_l2_increase = true, show_progress = false, workspace = nothing) -> volume

In-place Hybrid Flexible LSQR with TV regularization reconstruction into `volume`.

# Arguments
- `volume::AbstractArray{Float32, 3}`: Output 3D volume array `(nx, ny, nz)` modified in-place.
- `projection::AbstractArray{Float32, 3}`: 3D projection data array `(nu, nv, nangles)`.
- `geo::Geometry`: CT geometry specification.
- `niter::Integer`: Number of outer flexible LSQR iterations.

# Keyword Arguments
- `lambda::Real = 1f0`: Regularization weight parameter (default: `1f0`).
- `precond_steps::Integer = 6`: Number of inner weighted TV conjugate gradient steps (default: `6`).
- `projection_type::Symbol = :siddon`: Forward projection algorithm (`:siddon` or `:ray_voxel`) (default: `:siddon`).
- `backprojection_type::Symbol = :matched`: Backprojection kernel (`:matched` or `:voxel_cone`) (default: `:matched`).
- `gpu_ids = nothing`: Zero-based GPU device ID(s) to use (default: `nothing`).
- `return_history::Bool = false`: Return `(volume, history::IterativeHistory)` when `true` (default: `false`).
- `compute_l2::Bool = false`: Track residual L2 norm history across iterations (default: `false`).
- `stop_on_l2_increase::Bool = true`: Early stop if residual L2 norm strictly increases (default: `true`).
- `show_progress::Bool = false`: Display progress bar (default: `false`).
- `workspace::Union{Nothing, KrylovWorkspace} = nothing`: Preallocated `KrylovWorkspace` (default: `nothing`).

# Returns
- The mutated `volume` array of size `(geo.nVoxel[1], geo.nVoxel[2], geo.nVoxel[3])`, or `(volume, history::IterativeHistory)` if `return_history=true`.

# Examples
```julia
hybrid_flsqr_tv!(recon, proj, geo, 15; lambda=0.1f0, precond_steps=6)
```
"""
function hybrid_flsqr_tv!(volume::AbstractArray{Float32, 3}, projection::AbstractArray{Float32, 3}, geo::Geometry, niter::Integer;
    lambda::Real = 1f0,
    precond_steps::Integer = 6,
    projection_type::Symbol = :siddon,
    backprojection_type::Symbol = :matched,
    gpu_ids = nothing,
    return_history::Bool = false,
    compute_l2::Bool = false,
    stop_on_l2_increase::Bool = true,
    show_progress::Bool = false,
    workspace = nothing,
)
    niter >= 0 || throw(ArgumentError("niter must be >= 0"))
    precond_steps >= 1 || throw(ArgumentError("precond_steps must be >= 1"))
    volume = _require_canonical_array(volume, "volume", volume_size(geo))
    projection = _require_canonical_array(projection, "projection", projection_size(geo))

    workspace = _resolve_krylov_workspace(workspace, geo)
    u = workspace.proj_a
    proj_tmp = workspace.proj_b
    v = workspace.vol_a
    z = workspace.vol_b
    trial = workspace.vol_c
    base_volume = workspace.vol_d
    weights = workspace.vol_e

    cg_r = _extra_volume(workspace, 1, geo)
    cg_p = _extra_volume(workspace, 2, geo)
    cg_Ap = _extra_volume(workspace, 3, geo)
    gx = _extra_volume(workspace, 4, geo)
    gy = _extra_volume(workspace, 5, geo)
    gz = _extra_volume(workspace, 6, geo)
    tmp = _extra_volume(workspace, 7, geo)
    prev_volume = _extra_volume(workspace, 8, geo)

    need_residual = compute_l2 || stop_on_l2_increase || return_history
    residual_l2 = Float32[]
    lambda_history = Float32[]
    prev_residual = Inf32
    stop_reason = :max_iterations
    iterations_completed = 0
    progress = _make_progress(niter, show_progress, "Hybrid FLSQR-TV: ")

    subspace = _resolve_subspace_workspace(workspace, geo, niter)
    Umat = subspace.proj_basis
    Zmat = subspace.vol_basis
    Mmat = subspace.mat_a
    reg_gram = subspace.mat_b
    system = subspace.mat_c
    chol = subspace.mat_d
    rhs = subspace.rhs
    coeffs = subspace.coeffs
    data_rhs = subspace.coeffs_aux
    fill!(Mmat, 0f0)
    fill!(reg_gram, 0f0)
    fill!(system, 0f0)
    fill!(chol, 0f0)
    fill!(rhs, 0f0)
    fill!(data_rhs, 0f0)

    if maximum(abs, volume) == 0f0
        fill!(weights, 1f0)
    else
        _weighted_tv_weights!(weights, volume)
    end

    Ax!(u, volume, geo; projection_type, gpu_ids)
    @inbounds for i in eachindex(u, projection)
        u[i] = projection[i] - u[i]
    end
    beta = sqrt(_squared_l2(u))
    beta > 0f0 || return _maybe_return_history(volume, _build_history(residual_l2, lambda_history, 0, :zero_residual), return_history)
    @inbounds for i in eachindex(u)
        u[i] /= beta
    end
    _store_proj_basis_row!(Umat, 1, u)
    rhs[1] = beta
    copyto!(base_volume, volume)
    lambda32 = Float32(lambda)

    for k in 1:niter
        copyto!(prev_volume, volume)
        Atb!(v, u, geo; backprojection_type, gpu_ids)
        for j in 1:k-1
            coeff = _basis_dot(v, Zmat, j)
            @inbounds for i in eachindex(v)
                v[i] -= coeff * Zmat[j, i]
            end
        end
        vnorm = sqrt(_squared_l2(v))
        vnorm > 0f0 || break
        @inbounds for i in eachindex(v)
            v[i] /= vnorm
        end

        _tv_inner_cg!(z, v, weights, lambda32, Int(precond_steps), cg_r, cg_p, cg_Ap, gx, gy, gz, tmp)
        _store_basis_row!(Zmat, k, z)
        @inbounds for j in 1:k
            reg_gram[j, k] = _weighted_tv_basis_dot(z, Zmat, j, weights, 1f0)
            reg_gram[k, j] = reg_gram[j, k]
        end

        Ax!(u, z, geo; projection_type, gpu_ids)
        for j in 1:k
            Mmat[k, j] = _proj_basis_dot(u, Umat, j)
            idx = 1
            @inbounds for ia in 1:size(u, 3), iv in 1:size(u, 2), iu in 1:size(u, 1)
                u[iu, iv, ia] -= Mmat[k, j] * Umat[j, idx]
                idx += 1
            end
        end
        Mmat[k, k + 1] = sqrt(_squared_l2(u))
        Mmat[k, k + 1] > 0f0 || break
        @inbounds for i in eachindex(u)
            u[i] /= Mmat[k, k + 1]
        end
        _store_proj_basis_row!(Umat, k + 1, u)

        _update_hybrid_flsqr_data_system!(system, data_rhs, Mmat, rhs[1], k)
        @inbounds for j in 1:k-1
            system[j, k] += lambda32 * lambda32 * reg_gram[j, k]
            system[k, j] = system[j, k]
        end
        system[k, k] += lambda32 * lambda32 * reg_gram[k, k]
        _append_cholesky_column!(chol, system, k)
        @inbounds for i in 1:k
            coeffs[i] = data_rhs[i]
        end
        _solve_cholesky!(coeffs, chol, k)
        _apply_basis_combination!(trial, base_volume, Zmat, coeffs, k)
        copyto!(volume, trial)
        push!(lambda_history, lambda32)
        iterations_completed = k
        _next_progress!(progress, k, niter)

        if need_residual
            residual = _projection_residual_l2!(proj_tmp, projection, volume, geo, projection_type, gpu_ids)
            push!(residual_l2, residual)
            if stop_on_l2_increase && residual > prev_residual
                copyto!(volume, prev_volume)
                stop_reason = :l2_increase
                break
            end
            prev_residual = residual
        end
    end

    _finish_progress!(progress, iterations_completed, niter)
    history = _build_history(residual_l2, lambda_history, iterations_completed, stop_reason)
    return _maybe_return_history(volume, history, return_history)
end
