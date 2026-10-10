import json
import os
import sys

def build_runner(ipynb_path):
    stem = os.path.splitext(os.path.basename(ipynb_path))[0]
    root_dir = os.path.abspath(".")
    cell_dir = os.path.abspath(os.path.join("scratch", stem))
    os.makedirs(cell_dir, exist_ok=True)
    
    with open(ipynb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    jl_lines = []
    jl_lines.append('using LinearAlgebra, Printf, Random, CairoMakie, Tigre')
    jl_lines.append(f'md_path = joinpath("{root_dir}", "docs/src/demos/{stem}.md")')
    jl_lines.append('mkpath(dirname(md_path))')
    jl_lines.append('md_io = open(md_path, "w")')
    jl_lines.append('last_saved_fig = nothing')
    jl_lines.append('fig_counter = 0')
    jl_lines.append('')
    jl_lines.append('function run_and_log_cell(code_file, md_code_file)')
    jl_lines.append('    global last_saved_fig, fig_counter')
    jl_lines.append('    md_source_code = read(md_code_file, String)')
    jl_lines.append('    write(md_io, "```julia\\n" * md_source_code * "\\n```\\n\\n")')
    jl_lines.append('    ')
    jl_lines.append('    orig_stdout = stdout')
    jl_lines.append('    rd, wr = redirect_stdout()')
    jl_lines.append('    ')
    jl_lines.append('    res = try')
    jl_lines.append('        Base.include(Main, code_file)')
    jl_lines.append('    catch err')
    jl_lines.append('        redirect_stdout(orig_stdout)')
    jl_lines.append('        close(wr)')
    jl_lines.append('        rethrow(err)')
    jl_lines.append('    end')
    jl_lines.append('    ')
    jl_lines.append('    redirect_stdout(orig_stdout)')
    jl_lines.append('    close(wr)')
    jl_lines.append('    out = read(rd, String)')
    jl_lines.append('    clean_out = strip(out)')
    jl_lines.append('    if !isempty(clean_out)')
    jl_lines.append('        write(md_io, "```text\\n" * clean_out * "\\n```\\n\\n")')
    jl_lines.append('    end')
    jl_lines.append('    ')
    jl_lines.append('    current_fig = if res isa CairoMakie.Figure')
    jl_lines.append('        res')
    jl_lines.append('    elseif isdefined(Main, :fig) && Main.fig isa CairoMakie.Figure')
    jl_lines.append('        Main.fig')
    jl_lines.append('    else')
    jl_lines.append('        nothing')
    jl_lines.append('    end')
    jl_lines.append('    ')
    jl_lines.append('    if current_fig !== nothing && current_fig !== last_saved_fig')
    jl_lines.append('        fig_counter += 1')
    jl_lines.append(f'        fig_name = @sprintf("{stem}_%02d.png", fig_counter)')
    jl_lines.append(f'        fig_path = joinpath("{root_dir}", "docs/src/assets/tutorials", fig_name)')
    jl_lines.append('        save(fig_path, current_fig)')
    jl_lines.append('        write(md_io, "![](" * "../assets/tutorials/" * fig_name * ")\\n\\n")')
    jl_lines.append('        last_saved_fig = current_fig')
    jl_lines.append('    end')
    jl_lines.append('end')
    jl_lines.append('')

    code_idx = 0
    md_idx = 0
    for cell in nb.get("cells", []):
        ctype = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        if ctype == "markdown":
            md_idx += 1
            md_file = os.path.join(cell_dir, f"md_{md_idx:02d}.md")
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(source)
            jl_lines.append(f'write(md_io, read("{md_file}", String) * "\\n\\n")')
        elif ctype == "code":
            code_idx += 1
            code_file = os.path.join(cell_dir, f"code_{code_idx:02d}.jl")
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(source)
            jl_lines.append(f'println("[Cell {code_idx}] Running {stem}...")')
            jl_lines.append(f'run_and_log_cell("{code_file}", "{code_file}")')

    jl_lines.append('close(md_io)')
    jl_lines.append(f'println("Completed {stem}: $(fig_counter) figures saved.")')

    runner_path = os.path.join("scratch", f"run_{stem}.jl")
    with open(runner_path, "w", encoding="utf-8") as f:
        f.write("\n".join(jl_lines) + "\n")
    return runner_path

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        rp = build_runner(path)
        print("Generated runner:", rp)
    else:
        print("Usage: python3 tools/build_demo_runner.py <path_to_ipynb>")
