using Printf

struct SummaryRow
    source::String
    label::String
    build_avg_s::Float64
    reuse_avg_s::Float64
    prepare_once_s::Float64
    build_alloc_B::Int
    reuse_alloc_B::Int
    speedup::Float64
end

struct PlotOptions
    show_prepare::Bool
    show_alloc::Bool
    show_alloc_ratio::Bool
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

function _read_summary(path::AbstractString)
    lines = filter(line -> !isempty(line) && !startswith(line, "#"), strip.(readlines(path)))
    isempty(lines) && error("summary file is empty: $path")
    delim = endswith(lowercase(path), ".csv") ? :csv : :tsv
    split_line = delim === :csv ? _parse_csv_line : line -> split(line, '\t')
    header = split_line(lines[1])
    expected = ["label", "build_avg_s", "reuse_avg_s", "prepare_once_s", "build_alloc_B", "reuse_alloc_B", "speedup"]
    header == expected || error("unexpected header in $path")
    source = splitext(basename(path))[1]
    rows = SummaryRow[]
    for line in lines[2:end]
        fields = split_line(line)
        length(fields) == 7 || error("unexpected column count in $path")
        push!(rows, SummaryRow(
            source,
            fields[1],
            parse(Float64, fields[2]),
            parse(Float64, fields[3]),
            parse(Float64, fields[4]),
            parse(Int, fields[5]),
            parse(Int, fields[6]),
            parse(Float64, fields[7]),
        ))
    end
    return rows
end

function _parse_args(args::Vector{String})
    output = "reuse_summary.svg"
    title = "Reuse Benchmark Summary"
    show_prepare = true
    show_alloc = true
    show_alloc_ratio = true
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
        elseif arg == "--hide-prepare"
            show_prepare = false
        elseif arg == "--hide-alloc"
            show_alloc = false
        elseif arg == "--hide-alloc-ratio"
            show_alloc_ratio = false
        elseif arg == "--show-prepare"
            show_prepare = true
        elseif arg == "--show-alloc"
            show_alloc = true
        elseif arg == "--show-alloc-ratio"
            show_alloc_ratio = true
        else
            push!(inputs, arg)
        end
        i += 1
    end
    isempty(inputs) && error("at least one TSV/CSV summary path is required")
    options = PlotOptions(show_prepare, show_alloc, show_alloc_ratio)
    return output, title, options, inputs
end

function _row_label(row::SummaryRow, multi_source::Bool)
    return multi_source ? "$(row.source): $(row.label)" : row.label
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

function _render_svg(rows::Vector{SummaryRow}, output::AbstractString, title::AbstractString, options::PlotOptions)
    n = length(rows)
    n > 0 || error("no rows to plot")
    multi_source = length(unique(row.source for row in rows)) > 1
    max_time = maximum(max(row.build_avg_s, row.reuse_avg_s) for row in rows)
    max_time = max(max_time, eps(Float64))

    label_width = multi_source ? 340.0 : 280.0
    chart_width = 540.0
    right_width = 180.0
    row_height = 54.0
    top = 86.0
    bottom = 28.0
    width = Int(round(label_width + chart_width + right_width + 60.0))
    height = Int(round(top + bottom + row_height * n))

    build_color = "#2f6fed"
    reuse_color = "#17a34a"
    bg = "#f6f8fa"
    grid = "#d0d7de"

    lines = String[]
    push!(lines, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
    push!(lines, @sprintf("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\">", width, height, width, height))
    push!(lines, @sprintf("<rect width=\"%d\" height=\"%d\" fill=\"%s\" />", width, height, bg))
    push!(lines, _text(24, 34, title; size = 22, weight = "600"))
    subtitle_parts = ["blue=build avg", "green=reuse avg", "right text=speedup"]
    options.show_prepare && push!(subtitle_parts, "prep")
    options.show_alloc && push!(subtitle_parts, options.show_alloc_ratio ? "alloc + ratio" : "alloc")
    push!(lines, _text(24, 56, join(subtitle_parts, ", "); size = 12, fill = "#57606a"))

    chart_x0 = 24.0 + label_width
    chart_y0 = top
    ticks = 4
    for i in 0:ticks
        t = chart_x0 + chart_width * i / ticks
        value = max_time * i / ticks
        push!(lines, @sprintf("<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"%s\" stroke-width=\"1\" />", t, chart_y0 - 10, t, height - bottom, grid))
        push!(lines, _text(t, chart_y0 - 16, @sprintf("%.3fs", value); size = 11, anchor = "middle", fill = "#57606a"))
    end

    for (idx, row) in enumerate(rows)
        y = chart_y0 + row_height * (idx - 1)
        label = _row_label(row, multi_source)
        push!(lines, _text(24, y + 18, label; size = 12))

        build_w = chart_width * row.build_avg_s / max_time
        reuse_w = chart_width * row.reuse_avg_s / max_time
        push!(lines, _bar(chart_x0, y, build_w, 14, build_color))
        push!(lines, _bar(chart_x0, y + 20, reuse_w, 14, reuse_color))
        push!(lines, _text(chart_x0 + build_w + 6, y + 12, @sprintf("%.4fs", row.build_avg_s); size = 11))
        push!(lines, _text(chart_x0 + reuse_w + 6, y + 32, @sprintf("%.4fs", row.reuse_avg_s); size = 11))

        info_x = chart_x0 + chart_width + 24
        line_y = y + 14
        push!(lines, _text(info_x, line_y, @sprintf("speedup %.3fx", row.speedup); size = 11, weight = "600"))
        if options.show_prepare
            line_y += 16
            push!(lines, _text(info_x, line_y, @sprintf("prep %.4fs", row.prepare_once_s); size = 11, fill = "#57606a"))
        end
        if options.show_alloc
            line_y += 16
            alloc_text = @sprintf("alloc %d -> %d B", row.build_alloc_B, row.reuse_alloc_B)
            if options.show_alloc_ratio
                ratio = row.build_alloc_B / max(row.reuse_alloc_B, 1)
                alloc_text *= @sprintf(" (%.2fx)", ratio)
            end
            push!(lines, _text(info_x, line_y, alloc_text; size = 11, fill = "#57606a"))
        end
    end

    push!(lines, "</svg>")
    open(output, "w") do io
        write(io, join(lines, '\n'))
    end
    return output
end

function main(args = copy(ARGS))
    output, title, options, inputs = _parse_args(args)
    rows = SummaryRow[]
    for path in inputs
        append!(rows, _read_summary(path))
    end
    output_path = _render_svg(rows, output, title, options)
    println("SVG plot written to ", output_path)
    return nothing
end

main()
