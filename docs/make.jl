push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter
using Tigre

makedocs(
    sitename = "Tigre.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://hashimoto-koh.github.io/Tigre.jl",
        assets = String[],
        size_threshold = nothing,
        size_threshold_warn = 500 * 1024,
    ),
    modules = [Tigre],
    authors = "Koh Hashimoto",
    pages = [
        "Home" => "index.md",
        "About TIGRE & Upstream" => "about_tigre.md",
        "Quick Start" => "quickstart.md",
        "Installation" => "installation.md",
        "Geometry Specification" => "geometry.md",
        "Operators & Weights" => "operators.md",
        "Algorithms" => "algorithms.md",
        "Workspace & Plan Reuse" => "workspace_reuse.md",
        "Phantoms" => "phantoms.md",
        "Demonstrations" => [
            "01: 2D Parallel-Beam" => "demos/01_2D_parallel_beam.md",
            "02: 2D Cone-Beam (Fan-Beam)" => "demos/02_2D_cone_beam.md",
            "03: 3D Parallel-Beam" => "demos/03_3D_parallel_beam.md",
            "04: 3D Cone-Beam (CBCT)" => "demos/04_3D_cone_beam.md",
            "05: Iterative Algorithms Comparison" => "demos/05_iterative_comparison.md",
            "06: Krylov Subspace Solvers" => "demos/06_krylov_solvers.md",
            "07: Projection Preprocessing & COR" => "demos/07_preprocessing.md",
            "08: Offset Detector & Extended FOV" => "demos/08_offset_detector_fov.md",
            "09: Computed Laminography (CBCL)" => "demos/09_laminography_conebeam.md",
            "10: Emission Tomography (PET/SPECT)" => "demos/10_emission_mlem_osem.md",
            "11: Low-Dose CT & Regularization" => "demos/11_low_dose_regularization.md",
            "12: Sparse-View CT & Compressed Sensing" => "demos/12_sparse_view_cs.md",
            "13: Workspace & Plan Reuse" => "demos/13_workspace_reuse.md",
        ],
        "Visualization" => "visualization.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/hashimoto-koh/Tigre.jl.git",
    devbranch = "master",
    push_preview = true,
)
