using Tigre

function run_geometry_layout_demo(io::IO = stdout)
    geo = Geometry(
        mode = :cone,
        angles = Float32[0, 0.5, 1.0],
        nVoxel = (4, 5, 6),
        sVoxel = (40f0, 50f0, 60f0),
        nDetector = (8, 7),
        sDetector = (80f0, 70f0),
        DSO = 1000f0,
        DSD = 1200f0,
        offOrigin = [1f0, -2f0, 3f0],
        offDetector = [4f0, -5f0],
    )

    volume = zeros(Float32, volume_size(geo)...)
    projection = zeros(Float32, projection_size(geo)...)
    status = library_status()

    result = (
        projection_count = projection_count(geo),
        volume_size = volume_size(geo),
        projection_size = projection_size(geo),
        angles_size = size(geo.angles),
        canonical_volume = is_canonical_layout(volume),
        canonical_projection = is_canonical_layout(projection),
        detector_spacing = geo.dDetector,
        voxel_spacing = geo.dVoxel,
        library_available = status.available,
        build_info_available = status.build_info_available,
    )

    println(io, "Geometry/layout demo")
    println(io, "  projection count: ", result.projection_count)
    println(io, "  volume size: ", result.volume_size)
    println(io, "  projection size: ", result.projection_size)
    println(io, "  angles size: ", result.angles_size)
    println(io, "  canonical volume/projection: ", result.canonical_volume, " / ", result.canonical_projection)
    println(io, "  voxel spacing: ", result.voxel_spacing)
    println(io, "  detector spacing: ", result.detector_spacing)
    println(io, "  library available/build-info: ", result.library_available, " / ", result.build_info_available)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_geometry_layout_demo()
end
