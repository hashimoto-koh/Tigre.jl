# Third-Party Notices

This repository, `Tigre.jl`, is distributed under the BSD 3-Clause License. See [LICENSE.md](LICENSE.md).

## Upstream Project

`Tigre.jl` is an unofficial Julia port and interface built upon the upstream `TIGRE` project:

- **Upstream Repository**: <https://github.com/CERN/TIGRE>
- **Upstream Documentation**: <https://github.com/CERN/TIGRE#readme>
- **Upstream Authors**: Ander Biguri, Manjit Dosanjh, Steven Hancock, Manuchehr Soleimani, Reuben Lindroos, and CERN / University of Bath contributors.
- **Upstream License**: BSD 3-Clause License

The maintainers of this repository gratefully acknowledge the work of Ander Biguri and all contributors to the original `TIGRE` project.

---

## TIGRE

This project includes, ports, adapts, or interfaces with code derived from:

TIGRE: Tomographic Iterative GPU-based Reconstruction Toolbox  
https://github.com/CERN/TIGRE

Copyright (c) 2015,
University of Bath and CERN-European Organization for Nuclear Research.

Licensed under the BSD 3-Clause License:

```
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## How This Repository Uses Upstream Material

This repository does **not** vendor the full upstream `TIGRE` source tree in Git.

Instead, during `deps/build.jl`, the build script fetches selected upstream C++/CUDA source files from a pinned upstream revision (`deps/upstream.toml`), applies local integration patches and C ABI shims (`deps/patches/`, `deps/wrapper/`), and builds the native shared library (`libtigre_julia`) locally using the system's CUDA compiler (`nvcc`).

The Julia package then interfaces directly with the compiled native library via Julia's native `ccall` mechanism, while implementing analytical, algebraic, statistical, proximal, and Krylov reconstruction algorithms natively in pure Julia.

---

## Attribution and Redistribution Notes

This repository is intended to comply fully with the BSD 3-Clause License obligations applicable to upstream-derived material, including the retention of copyright notices, conditions, and disclaimers.

If you redistribute this repository or derivative works thereof (in source or binary form), review:

- [LICENSE.md](LICENSE.md)
- [README.md](README.md)
- [deps/upstream.toml](deps/upstream.toml)
- [deps/build.jl](deps/build.jl)

and ensure that all applicable upstream attribution and license obligations continue to be satisfied.
