using Tigre

function run_geometry_validation_demo(io::IO = stdout)
    geo = Geometry(
        mode = :cone,
        angles = Float32[0, 0.5, 1.0],
        nVoxel = (4, 5, 6),
        sVoxel = (40f0, 50f0, 60f0),
        nDetector = (8, 7),
        sDetector = (80f0, 70f0),
        DSO = 1000f0,
        DSD = 1200f0,
    )

    volume = zeros(Float32, volume_size(geo)...)
    projection = zeros(Float32, projection_size(geo)...)
    bad_volume_shape = zeros(Float32, 4, 5, 5)
    bad_projection_layout = PermutedDimsArray(projection, (1, 2, 3))

    volume_ok = validate_volume_array(volume, geo)
    projection_ok = validate_projection_array(projection, geo)

    bad_shape_error = try
        validate_volume_array(bad_volume_shape, geo)
        ""
    catch err
        sprint(showerror, err)
    end

    bad_layout_error = try
        validate_projection_array(bad_projection_layout, geo)
        ""
    catch err
        sprint(showerror, err)
    end

    result = (
        volume_valid = volume_ok === volume,
        projection_valid = projection_ok === projection,
        bad_shape_error = bad_shape_error,
        bad_layout_error = bad_layout_error,
    )

    println(io, "Geometry validation demo")
    println(io, "  valid volume/projection: ", result.volume_valid, " / ", result.projection_valid)
    println(io, "  bad shape error: ", result.bad_shape_error)
    println(io, "  bad layout/non-dense error: ", result.bad_layout_error)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_geometry_validation_demo()
end
