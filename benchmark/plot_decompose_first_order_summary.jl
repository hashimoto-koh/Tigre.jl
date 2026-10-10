using Printf

struct DecomposeRow
    source::String
    label::String
    time_median_s::Float64
    alloc_median_B::Float64
end

struct DecomposePlotOptions
    show_source::Bool
    sort_mode::Symbol
    top_n::Union{Nothing, Int}
    show_ratio::Bool
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
    expected = ["label", "samples", "time_median_s", "time_mean_s", "time_std_s", "alloc_median_B", "alloc_mean_B"]
    header == expected || error("unsupported decompose summary header in $path")
    source = splitext(basename(path))[1]
    rows = DecomposeRow[]
    for line in lines[2:end]
        fields = _split_line(path, line)
        push!(rows, DecomposeRow(
            source,
            fields[1],
            parse(Float64, fields[3]),
            parse(Float64, fields[6]),
        ))
    end
    return rows
end

function _parse_args(args::Vector{String})
    output = "decompose_first_order_summary.svg"
    title = "Decompose First-order Summary"
    show_source = true
    sort_mode = :input
    top_n = nothing
    show_ratio = true
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
        elseif arg == "--hide-source"
            show_source = false
        elseif arg == "--show-source"
            show_source = true
        elseif arg == "--hide-ratio"
            show_ratio = false
        elseif arg == "--show-ratio"
            show_ratio = true
        elseif startswith(arg, "--sort=")
            sort_mode = Symbol(split(arg, "=", limit = 2)[2])
        elseif arg == "--sort" && i < length(args)
            i += 1
            sort_mode = Symbol(args[i])
        elseif startswith(arg, "--top=")
            top_n = parse(Int, split(arg, "=", limit = 2)[2])
        elseif arg == "--top" && i < length(args)
            i += 1
            top_n = parse(Int, args[i])
        else
            push!(inputs, arg)
        end
        i += 1
    end
    isempty(inputs) && error("at least one TSV/CSV summary path is required")
    sort_mode in (:input, :time_desc, :time_asc, :alloc_desc, :alloc_asc, :label, :source) || error("unsupported --sort mode: $sort_mode")
    top_n === nothing || top_n > 0 || error("--top must be positive")
    return output, title, DecomposePlotOptions(show_source, sort_mode, top_n, show_ratio), inputs
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

function _hex_byte(v::Real)
    n = clamp(round(Int, v), 0, 255)
    return lowercase(string(n, base = 16, pad = 2))
end

function _blend_color(low::NTuple{3, Int}, high::NTuple{3, Int}, t::Float64)
    tt = clamp(t, 0.0, 1.0)
    r = low[1] + (high[1] - low[1]) * tt
    g = low[2] + (high[2] - low[2]) * tt
    b = low[3] + (high[3] - low[3]) * tt
    return "#" * _hex_byte(r) * _hex_byte(g) * _hex_byte(b)
end

function _alloc_fill(row::DecomposeRow, max_alloc::Float64)
    max_alloc <= 0 && return "#2da44e"
    return _blend_color((45, 164, 78), (207, 34, 46), row.alloc_median_B / max_alloc)
end

function _alloc_ratio_b_per_ms(row::DecomposeRow)
    ms = max(row.time_median_s * 1000.0, eps(Float64))
    return row.alloc_median_B / ms
end

function _sorted_rows(rows::Vector{DecomposeRow}, sort_mode::Symbol)
    sort_mode === :input && return rows
    ordered = copy(rows)
    if sort_mode === :time_desc
        sort!(ordered; by = row -> (-row.time_median_s, row.label, row.source))
    elseif sort_mode === :time_asc
        sort!(ordered; by = row -> (row.time_median_s, row.label, row.source))
    elseif sort_mode === :alloc_desc
        sort!(ordered; by = row -> (-row.alloc_median_B, row.label, row.source))
    elseif sort_mode === :alloc_asc
        sort!(ordered; by = row -> (row.alloc_median_B, row.label, row.source))
    elseif sort_mode === :label
        sort!(ordered; by = row -> (lowercase(row.label), lowercase(row.source), row.time_median_s))
    elseif sort_mode === :source
        sort!(ordered; by = row -> (lowercase(row.source), lowercase(row.label), row.time_median_s))
    end
    return ordered
end

function _render_svg(rows::Vector{DecomposeRow}, output::AbstractString, title::AbstractString, options::DecomposePlotOptions)
    isempty(rows) && error("no rows to plot")
    rows = _sorted_rows(rows, options.sort_mode)
    options.top_n !== nothing && (rows = rows[1:min(options.top_n, length(rows))])
    multi_source = length(unique(row.source for row in rows)) > 1
    max_time = maximum(row.time_median_s for row in rows)
    max_time = max(max_time, eps(Float64))
    max_alloc = maximum(row.alloc_median_B for row in rows)

    width = 980
    label_width = multi_source ? 340.0 : 300.0
    chart_width = 420.0
    info_width = options.show_ratio ? 220.0 : 180.0
    row_height = 40.0
    top = 88.0
    bottom = 28.0
    height = Int(round(top + bottom + row_height * length(rows)))

    lines = String[]
    push!(lines, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
    push!(lines, @sprintf("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\">", width, height, width, height))
    push!(lines, @sprintf("<rect width=\"%d\" height=\"%d\" fill=\"#f6f8fa\" />", width, height))
    push!(lines, _text(24, 34, title; size = 22, weight = "600"))
    subtitle = "bar width=time median, bar color=alloc median, right text=alloc median"
    options.show_ratio && (subtitle *= " + alloc/time")
    multi_source && options.show_source && (subtitle *= ", label=source: primitive")
    options.sort_mode !== :input && (subtitle *= ", sort=$(String(options.sort_mode))")
    options.top_n !== nothing && (subtitle *= ", top=$(options.top_n)")
    push!(lines, _text(24, 56, subtitle; size = 12, fill = "#57606a"))

    chart_x0 = 24.0 + label_width
    chart_y0 = top
    for i in 0:4
        t = chart_x0 + chart_width * i / 4
        value = max_time * i / 4
        push!(lines, @sprintf("<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"#d0d7de\" stroke-width=\"1\" />", t, chart_y0 - 10, t, height - bottom))
        push!(lines, _text(t, chart_y0 - 16, @sprintf("%.3fs", value); size = 10, anchor = "middle", fill = "#57606a"))
    end

    info_x = chart_x0 + chart_width + 24.0
    for (idx, row) in enumerate(rows)
        y = chart_y0 + row_height * (idx - 1)
        label = (multi_source && options.show_source) ? "$(row.source): $(row.label)" : row.label
        push!(lines, _text(24, y + 18, label; size = 12))
        bar_w = chart_width * row.time_median_s / max_time
        push!(lines, _bar(chart_x0, y, bar_w, 14, _alloc_fill(row, max_alloc)))
        push!(lines, _text(chart_x0 + bar_w + 6, y + 12, @sprintf("%.4fs", row.time_median_s); size = 10))
        ratio = _alloc_ratio_b_per_ms(row)
        info = options.show_ratio ? @sprintf("alloc %.0f B | %.1f B/ms", row.alloc_median_B, ratio) : @sprintf("alloc %.0f B", row.alloc_median_B)
        push!(lines, _text(info_x, y + 12, info; size = 10, fill = "#57606a"))
    end

    push!(lines, "</svg>")
    open(output, "w") do io
        write(io, join(lines, '\n'))
    end
    return output
end

function main(args = copy(ARGS))
    output, title, options, inputs = _parse_args(args)
    rows = DecomposeRow[]
    for path in inputs
        append!(rows, _read_summary(path))
    end
    output_path = _render_svg(rows, output, title, options)
    println("SVG plot written to ", output_path)
    return nothing
end

main()
