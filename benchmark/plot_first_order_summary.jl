using Printf

abstract type AbstractFirstOrderRow end

struct SingleShotRow <: AbstractFirstOrderRow
    source::String
    label::String
    build_time_median_s::Float64
    prepared_time_median_s::Float64
    speedup_median::Float64
    build_alloc_median_B::Float64
    prepared_alloc_median_B::Float64
end

struct VarianceRow <: AbstractFirstOrderRow
    source::String
    label::String
    cold_time_s::Float64
    warm_gc_on_median_s::Float64
    warm_gc_off_median_s::Float64
    cold_alloc_B::Float64
    warm_gc_on_alloc_median_B::Float64
    warm_gc_off_alloc_median_B::Float64
end

struct RealisticRow <: AbstractFirstOrderRow
    source::String
    label::String
    build_avg_median_s::Float64
    reuse_avg_median_s::Float64
    speedup_median::Float64
    build_alloc_median_B::Float64
    reuse_alloc_median_B::Float64
end

struct FirstOrderPlotOptions
    show_single_shot::Bool
    show_variance::Bool
    show_realistic::Bool
    show_source::Bool
    single_sort::Symbol
    variance_sort::Symbol
    realistic_sort::Symbol
end

function _xml_escape(s::AbstractString)
    escaped = replace(s, '&' => "&amp;")
    escaped = replace(escaped, '<' => "&lt;")
    escaped = replace(escaped, '>' => "&gt;")
    return replace(escaped, '"' => "&quot;")
end

function _parse_csv_line(line::AbstractString)
    fields = String[]
    buf = IOBuffer()
    in_quotes = false
    i = firstindex(line)
    while i <= lastindex(line)
        c = line[i]
        if c == '"'
            next_i = nextind(line, i)
            if in_quotes && next_i <= lastindex(line) && line[next_i] == '"'
                print(buf, '"')
                i = next_i
            else
                in_quotes = !in_quotes
            end
        elseif c == ',' && !in_quotes
            push!(fields, String(take!(buf)))
        else
            print(buf, c)
        end
        i = nextind(line, i)
    end
    push!(fields, String(take!(buf)))
    return fields
end

function _split_line(path::AbstractString, line::AbstractString)
    return endswith(lowercase(path), ".csv") ? _parse_csv_line(line) : split(line, '\t')
end

function _read_summary(path::AbstractString)
    lines = filter(line -> !isempty(line) && !startswith(line, "#"), strip.(readlines(path)))
    isempty(lines) && error("summary file is empty: $path")
    header = _split_line(path, lines[1])
    source = splitext(basename(path))[1]
    rows = AbstractFirstOrderRow[]
    if header == ["label", "samples", "build_time_median_s", "build_time_mean_s", "build_time_std_s", "build_alloc_median_B", "build_alloc_mean_B", "prepared_time_median_s", "prepared_time_mean_s", "prepared_time_std_s", "prepared_alloc_median_B", "prepared_alloc_mean_B", "speedup_median"]
        for line in lines[2:end]
            fields = _split_line(path, line)
            push!(rows, SingleShotRow(
                source,
                fields[1],
                parse(Float64, fields[3]),
                parse(Float64, fields[8]),
                parse(Float64, fields[13]),
                parse(Float64, fields[6]),
                parse(Float64, fields[11]),
            ))
        end
    elseif header == ["label", "samples", "cold_time_s", "cold_alloc_B", "warm_gc_on_median_s", "warm_gc_on_mean_s", "warm_gc_on_std_s", "warm_gc_on_alloc_median_B", "warm_gc_off_median_s", "warm_gc_off_mean_s", "warm_gc_off_std_s", "warm_gc_off_alloc_median_B"]
        for line in lines[2:end]
            fields = _split_line(path, line)
            push!(rows, VarianceRow(
                source,
                fields[1],
                parse(Float64, fields[3]),
                parse(Float64, fields[5]),
                parse(Float64, fields[9]),
                parse(Float64, fields[4]),
                parse(Float64, fields[8]),
                parse(Float64, fields[12]),
            ))
        end
    elseif header == ["label", "samples", "build_avg_median_s", "build_avg_std_s", "reuse_avg_median_s", "reuse_avg_std_s", "prepare_once_s", "build_alloc_median_B", "reuse_alloc_median_B", "speedup_median"]
        for line in lines[2:end]
            fields = _split_line(path, line)
            push!(rows, RealisticRow(
                source,
                fields[1],
                parse(Float64, fields[3]),
                parse(Float64, fields[5]),
                parse(Float64, fields[10]),
                parse(Float64, fields[8]),
                parse(Float64, fields[9]),
            ))
        end
    else
        error("unsupported first-order summary header in $path")
    end
    return rows
end

function _parse_args(args::Vector{String})
    output = "first_order_summary.svg"
    title = "First-order Benchmark Summary"
    show_single_shot = true
    show_variance = true
    show_realistic = true
    show_source = true
    single_sort = :input
    variance_sort = :input
    realistic_sort = :input
    inputs = String[]
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--output=")
            output = split(arg, "=", limit = 2)[2]
        elseif arg == "--output" && i < length(args)
            i += 1
            output = args[i]
        elseif startswith(arg, "--title=")
            title = split(arg, "=", limit = 2)[2]
        elseif arg == "--title" && i < length(args)
            i += 1
            title = args[i]
        elseif arg == "--hide-single-shot"
            show_single_shot = false
        elseif arg == "--show-single-shot"
            show_single_shot = true
        elseif arg == "--hide-variance"
            show_variance = false
        elseif arg == "--show-variance"
            show_variance = true
        elseif arg == "--hide-realistic"
            show_realistic = false
        elseif arg == "--show-realistic"
            show_realistic = true
        elseif arg == "--hide-source"
            show_source = false
        elseif arg == "--show-source"
            show_source = true
        elseif startswith(arg, "--single-sort=")
            single_sort = Symbol(split(arg, "=", limit = 2)[2])
        elseif arg == "--single-sort" && i < length(args)
            i += 1
            single_sort = Symbol(args[i])
        elseif startswith(arg, "--variance-sort=")
            variance_sort = Symbol(split(arg, "=", limit = 2)[2])
        elseif arg == "--variance-sort" && i < length(args)
            i += 1
            variance_sort = Symbol(args[i])
        elseif startswith(arg, "--realistic-sort=")
            realistic_sort = Symbol(split(arg, "=", limit = 2)[2])
        elseif arg == "--realistic-sort" && i < length(args)
            i += 1
            realistic_sort = Symbol(args[i])
        else
            push!(inputs, arg)
        end
        i += 1
    end
    isempty(inputs) && error("at least one TSV/CSV summary path is required")
    single_sort in (:input, :label, :source, :build_desc, :build_asc, :prepared_desc, :prepared_asc, :speedup_desc, :speedup_asc) || error("unsupported --single-sort mode: $single_sort")
    variance_sort in (:input, :label, :source, :cold_desc, :cold_asc, :warm_on_desc, :warm_on_asc, :warm_off_desc, :warm_off_asc) || error("unsupported --variance-sort mode: $variance_sort")
    realistic_sort in (:input, :label, :source, :build_desc, :build_asc, :prepared_desc, :prepared_asc, :speedup_desc, :speedup_asc) || error("unsupported --realistic-sort mode: $realistic_sort")
    options = FirstOrderPlotOptions(show_single_shot, show_variance, show_realistic, show_source, single_sort, variance_sort, realistic_sort)
    return output, title, options, inputs
end

function _bar(x, y, w, h, fill)
    return @sprintf("<rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\" rx=\"3\" ry=\"3\" fill=\"%s\" />", x, y, w, h, fill)
end

function _text(x, y, value; size = 12, anchor = "start", fill = "#1f2328", weight = "400")
    return @sprintf(
        "<text x=\"%.1f\" y=\"%.1f\" font-family=\"monospace\" font-size=\"%d\" text-anchor=\"%s\" fill=\"%s\" font-weight=\"%s\">%s</text>",
        x, y, size, anchor, fill, weight, _xml_escape(value),
    )
end

function _single_sorted_rows(rows::Vector{SingleShotRow}, sort_mode::Symbol)
    sort_mode === :input && return rows
    ordered = copy(rows)
    if sort_mode === :label
        sort!(ordered; by = row -> (lowercase(row.label), lowercase(row.source), row.build_time_median_s))
    elseif sort_mode === :source
        sort!(ordered; by = row -> (lowercase(row.source), lowercase(row.label), row.build_time_median_s))
    elseif sort_mode === :build_desc
        sort!(ordered; by = row -> (-row.build_time_median_s, row.label, row.source))
    elseif sort_mode === :build_asc
        sort!(ordered; by = row -> (row.build_time_median_s, row.label, row.source))
    elseif sort_mode === :prepared_desc
        sort!(ordered; by = row -> (-row.prepared_time_median_s, row.label, row.source))
    elseif sort_mode === :prepared_asc
        sort!(ordered; by = row -> (row.prepared_time_median_s, row.label, row.source))
    elseif sort_mode === :speedup_desc
        sort!(ordered; by = row -> (-row.speedup_median, row.label, row.source))
    elseif sort_mode === :speedup_asc
        sort!(ordered; by = row -> (row.speedup_median, row.label, row.source))
    end
    return ordered
end

function _variance_sorted_rows(rows::Vector{VarianceRow}, sort_mode::Symbol)
    sort_mode === :input && return rows
    ordered = copy(rows)
    if sort_mode === :label
        sort!(ordered; by = row -> (lowercase(row.label), lowercase(row.source), row.cold_time_s))
    elseif sort_mode === :source
        sort!(ordered; by = row -> (lowercase(row.source), lowercase(row.label), row.cold_time_s))
    elseif sort_mode === :cold_desc
        sort!(ordered; by = row -> (-row.cold_time_s, row.label, row.source))
    elseif sort_mode === :cold_asc
        sort!(ordered; by = row -> (row.cold_time_s, row.label, row.source))
    elseif sort_mode === :warm_on_desc
        sort!(ordered; by = row -> (-row.warm_gc_on_median_s, row.label, row.source))
    elseif sort_mode === :warm_on_asc
        sort!(ordered; by = row -> (row.warm_gc_on_median_s, row.label, row.source))
    elseif sort_mode === :warm_off_desc
        sort!(ordered; by = row -> (-row.warm_gc_off_median_s, row.label, row.source))
    elseif sort_mode === :warm_off_asc
        sort!(ordered; by = row -> (row.warm_gc_off_median_s, row.label, row.source))
    end
    return ordered
end

function _realistic_sorted_rows(rows::Vector{RealisticRow}, sort_mode::Symbol)
    sort_mode === :input && return rows
    ordered = copy(rows)
    if sort_mode === :label
        sort!(ordered; by = row -> (lowercase(row.label), lowercase(row.source), row.build_avg_median_s))
    elseif sort_mode === :source
        sort!(ordered; by = row -> (lowercase(row.source), lowercase(row.label), row.build_avg_median_s))
    elseif sort_mode === :build_desc
        sort!(ordered; by = row -> (-row.build_avg_median_s, row.label, row.source))
    elseif sort_mode === :build_asc
        sort!(ordered; by = row -> (row.build_avg_median_s, row.label, row.source))
    elseif sort_mode === :prepared_desc
        sort!(ordered; by = row -> (-row.reuse_avg_median_s, row.label, row.source))
    elseif sort_mode === :prepared_asc
        sort!(ordered; by = row -> (row.reuse_avg_median_s, row.label, row.source))
    elseif sort_mode === :speedup_desc
        sort!(ordered; by = row -> (-row.speedup_median, row.label, row.source))
    elseif sort_mode === :speedup_asc
        sort!(ordered; by = row -> (row.speedup_median, row.label, row.source))
    end
    return ordered
end

_show_source(multi_source::Bool, options::FirstOrderPlotOptions) = multi_source && options.show_source

function _label_for(row, multi_source::Bool, options::FirstOrderPlotOptions)
    return _show_source(multi_source, options) ? "$(row.source): $(row.label)" : row.label
end

function _section_subtitle(base::String, sort_mode::Symbol, multi_source::Bool, options::FirstOrderPlotOptions)
    parts = String[base]
    _show_source(multi_source, options) && push!(parts, "label=source: metric")
    sort_mode !== :input && push!(parts, "sort=$(String(sort_mode))")
    return join(parts, ", ")
end

function _render_single_shot!(lines, rows::Vector{SingleShotRow}, width, y0, options::FirstOrderPlotOptions)
    isempty(rows) && return y0
    rows = _single_sorted_rows(rows, options.single_sort)
    max_time = maximum(max(row.build_time_median_s, row.prepared_time_median_s) for row in rows)
    multi_source = length(unique(row.source for row in rows)) > 1
    label_width = _show_source(multi_source, options) ? 360.0 : 290.0
    chart_width = 420.0
    info_x = 24.0 + label_width + chart_width + 24.0
    row_height = 54.0
    chart_x0 = 24.0 + label_width

    push!(lines, _text(24, y0, "Single-shot"; size = 18, weight = "600"))
    push!(lines, _text(24, y0 + 20, _section_subtitle("blue=build median, green=prepared median, right text=speedup + alloc", options.single_sort, multi_source, options); size = 11, fill = "#57606a"))
    top = y0 + 44.0
    for i in 0:4
        t = chart_x0 + chart_width * i / 4
        value = max_time * i / 4
        push!(lines, @sprintf("<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"#d0d7de\" stroke-width=\"1\" />", t, top - 10, t, top + row_height * length(rows)))
        push!(lines, _text(t, top - 16, @sprintf("%.3fs", value); size = 10, anchor = "middle", fill = "#57606a"))
    end
    for (idx, row) in enumerate(rows)
        y = top + row_height * (idx - 1)
        label = _label_for(row, multi_source, options)
        push!(lines, _text(24, y + 18, label; size = 12))
        build_w = chart_width * row.build_time_median_s / max(max_time, eps(Float64))
        prepared_w = chart_width * row.prepared_time_median_s / max(max_time, eps(Float64))
        push!(lines, _bar(chart_x0, y, build_w, 14, "#2f6fed"))
        push!(lines, _bar(chart_x0, y + 20, prepared_w, 14, "#17a34a"))
        push!(lines, _text(chart_x0 + build_w + 6, y + 12, @sprintf("%.4fs", row.build_time_median_s); size = 10))
        push!(lines, _text(chart_x0 + prepared_w + 6, y + 32, @sprintf("%.4fs", row.prepared_time_median_s); size = 10))
        push!(lines, _text(info_x, y + 14, @sprintf("speedup %.3fx", row.speedup_median); size = 11, weight = "600"))
        push!(lines, _text(info_x, y + 30, @sprintf("alloc %.0f -> %.0f B", row.build_alloc_median_B, row.prepared_alloc_median_B); size = 10, fill = "#57606a"))
    end
    return top + row_height * length(rows) + 24.0
end

function _render_variance!(lines, rows::Vector{VarianceRow}, width, y0, options::FirstOrderPlotOptions)
    isempty(rows) && return y0
    rows = _variance_sorted_rows(rows, options.variance_sort)
    max_time = maximum(max(row.cold_time_s, row.warm_gc_on_median_s, row.warm_gc_off_median_s) for row in rows)
    multi_source = length(unique(row.source for row in rows)) > 1
    label_width = _show_source(multi_source, options) ? 360.0 : 290.0
    chart_width = 420.0
    info_x = 24.0 + label_width + chart_width + 24.0
    row_height = 72.0
    chart_x0 = 24.0 + label_width

    push!(lines, _text(24, y0, "Variance"; size = 18, weight = "600"))
    push!(lines, _text(24, y0 + 20, _section_subtitle("orange=cold-first, blue=warm-gc-on median, green=warm-gc-off median", options.variance_sort, multi_source, options); size = 11, fill = "#57606a"))
    top = y0 + 44.0
    for i in 0:4
        t = chart_x0 + chart_width * i / 4
        value = max_time * i / 4
        push!(lines, @sprintf("<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"#d0d7de\" stroke-width=\"1\" />", t, top - 10, t, top + row_height * length(rows)))
        push!(lines, _text(t, top - 16, @sprintf("%.3fs", value); size = 10, anchor = "middle", fill = "#57606a"))
    end
    for (idx, row) in enumerate(rows)
        y = top + row_height * (idx - 1)
        label = _label_for(row, multi_source, options)
        push!(lines, _text(24, y + 18, label; size = 12))
        cold_w = chart_width * row.cold_time_s / max(max_time, eps(Float64))
        on_w = chart_width * row.warm_gc_on_median_s / max(max_time, eps(Float64))
        off_w = chart_width * row.warm_gc_off_median_s / max(max_time, eps(Float64))
        push!(lines, _bar(chart_x0, y, cold_w, 12, "#d97706"))
        push!(lines, _bar(chart_x0, y + 18, on_w, 12, "#2f6fed"))
        push!(lines, _bar(chart_x0, y + 36, off_w, 12, "#17a34a"))
        push!(lines, _text(chart_x0 + cold_w + 6, y + 10, @sprintf("cold %.4fs", row.cold_time_s); size = 10))
        push!(lines, _text(chart_x0 + on_w + 6, y + 28, @sprintf("gc-on %.4fs", row.warm_gc_on_median_s); size = 10))
        push!(lines, _text(chart_x0 + off_w + 6, y + 46, @sprintf("gc-off %.4fs", row.warm_gc_off_median_s); size = 10))
        push!(lines, _text(info_x, y + 14, @sprintf("alloc cold %.0f B", row.cold_alloc_B); size = 10, fill = "#57606a"))
        push!(lines, _text(info_x, y + 30, @sprintf("alloc gc-on %.0f B", row.warm_gc_on_alloc_median_B); size = 10, fill = "#57606a"))
        push!(lines, _text(info_x, y + 46, @sprintf("alloc gc-off %.0f B", row.warm_gc_off_alloc_median_B); size = 10, fill = "#57606a"))
    end
    return top + row_height * length(rows) + 24.0
end

function _render_realistic!(lines, rows::Vector{RealisticRow}, width, y0, options::FirstOrderPlotOptions)
    isempty(rows) && return y0
    rows = _realistic_sorted_rows(rows, options.realistic_sort)
    max_time = maximum(max(row.build_avg_median_s, row.reuse_avg_median_s) for row in rows)
    multi_source = length(unique(row.source for row in rows)) > 1
    label_width = _show_source(multi_source, options) ? 380.0 : 320.0
    chart_width = 420.0
    info_x = 24.0 + label_width + chart_width + 24.0
    row_height = 54.0
    chart_x0 = 24.0 + label_width

    push!(lines, _text(24, y0, "Realistic"; size = 18, weight = "600"))
    push!(lines, _text(24, y0 + 20, _section_subtitle("blue=build avg/outer median, green=prepared avg/outer median", options.realistic_sort, multi_source, options); size = 11, fill = "#57606a"))
    top = y0 + 44.0
    for i in 0:4
        t = chart_x0 + chart_width * i / 4
        value = max_time * i / 4
        push!(lines, @sprintf("<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"#d0d7de\" stroke-width=\"1\" />", t, top - 10, t, top + row_height * length(rows)))
        push!(lines, _text(t, top - 16, @sprintf("%.3fs", value); size = 10, anchor = "middle", fill = "#57606a"))
    end
    for (idx, row) in enumerate(rows)
        y = top + row_height * (idx - 1)
        label = _label_for(row, multi_source, options)
        push!(lines, _text(24, y + 18, label; size = 12))
        build_w = chart_width * row.build_avg_median_s / max(max_time, eps(Float64))
        reuse_w = chart_width * row.reuse_avg_median_s / max(max_time, eps(Float64))
        push!(lines, _bar(chart_x0, y, build_w, 14, "#2f6fed"))
        push!(lines, _bar(chart_x0, y + 20, reuse_w, 14, "#17a34a"))
        push!(lines, _text(chart_x0 + build_w + 6, y + 12, @sprintf("%.4fs", row.build_avg_median_s); size = 10))
        push!(lines, _text(chart_x0 + reuse_w + 6, y + 32, @sprintf("%.4fs", row.reuse_avg_median_s); size = 10))
        push!(lines, _text(info_x, y + 14, @sprintf("speedup %.3fx", row.speedup_median); size = 11, weight = "600"))
        push!(lines, _text(info_x, y + 30, @sprintf("alloc %.0f -> %.0f B", row.build_alloc_median_B, row.reuse_alloc_median_B); size = 10, fill = "#57606a"))
    end
    return top + row_height * length(rows) + 24.0
end

function _render_svg(rows::Vector{AbstractFirstOrderRow}, output::AbstractString, title::AbstractString, options::FirstOrderPlotOptions)
    single_rows = SingleShotRow[row for row in rows if row isa SingleShotRow]
    variance_rows = VarianceRow[row for row in rows if row isa VarianceRow]
    realistic_rows = RealisticRow[row for row in rows if row isa RealisticRow]
    !options.show_single_shot && (single_rows = SingleShotRow[])
    !options.show_variance && (variance_rows = VarianceRow[])
    !options.show_realistic && (realistic_rows = RealisticRow[])
    (isempty(single_rows) && isempty(variance_rows) && isempty(realistic_rows)) && error("no visible sections to plot")

    width = 980
    height = 110 + 54 * length(single_rows) + 72 * length(variance_rows) + 54 * length(realistic_rows) + 140
    lines = String[]
    push!(lines, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
    push!(lines, @sprintf("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\">", width, height, width, height))
    push!(lines, @sprintf("<rect width=\"%d\" height=\"%d\" fill=\"#f6f8fa\" />", width, height))
    push!(lines, _text(24, 34, title; size = 22, weight = "600"))
    visible_sections = String[]
    options.show_single_shot && push!(visible_sections, "single-shot")
    options.show_variance && push!(visible_sections, "variance")
    options.show_realistic && push!(visible_sections, "realistic")
    push!(lines, _text(24, 56, "visible sections: " * join(visible_sections, ", "); size = 12, fill = "#57606a"))

    y = 88.0
    y = _render_single_shot!(lines, single_rows, width, y, options)
    y = _render_variance!(lines, variance_rows, width, y, options)
    y = _render_realistic!(lines, realistic_rows, width, y, options)

    push!(lines, "</svg>")
    open(output, "w") do io
        write(io, join(lines, '\n'))
    end
    return output
end

function main(args = copy(ARGS))
    output, title, options, inputs = _parse_args(args)
    rows = AbstractFirstOrderRow[]
    for path in inputs
        append!(rows, _read_summary(path))
    end
    output_path = _render_svg(rows, output, title, options)
    println("SVG plot written to ", output_path)
    return nothing
end

main()
