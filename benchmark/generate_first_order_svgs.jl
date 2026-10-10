module GenerateBenchmarkSVGDriver
include(joinpath(@__DIR__, "generate_benchmark_svgs.jl"))
end

function main(args = copy(ARGS); io::IO = stdout)
    println(io, "warning: generate_first_order_svgs.jl is a compatibility wrapper; please use generate_benchmark_svgs.jl in the future.")
    return GenerateBenchmarkSVGDriver.main(args; io = io)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
