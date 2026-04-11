# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from typing import Any, Dict, Sequence

from tvm_ffi.dataclasses import py_class

from tilus.hidet.ir.expr import Var
from tilus.hidet.ir.func import Function
from tilus.hidet.ir.node import Node
from tilus.hidet.ir.type import FuncType


@py_class
class IRModule(Node):
    """
    The intermediate representation of tensor programs.

    An IRModule contains one or more functions. It is the basic compilation unit of hidet.
    """

    functions: Any = None
    global_vars: Any = None
    namespace: str = ""
    extern_functions: Any = None
    include_headers: Any = None
    include_dirs: Any = None
    linking_dirs: Any = None
    linking_libs: Any = None
    object_files: Any = None
    task: Any = None

    def __post_init__(self):
        if self.functions is None:
            self.functions = {}
        if self.global_vars is None:
            self.global_vars = {}
        if self.extern_functions is None:
            self.extern_functions = {}
        if self.include_headers is None:
            self.include_headers = []
        if self.include_dirs is None:
            self.include_dirs = []
        if self.linking_dirs is None:
            self.linking_dirs = []
        if self.linking_libs is None:
            self.linking_libs = []
        if self.object_files is None:
            self.object_files = []

        assert all(isinstance(func, Function) for func in self.functions.values()) and all(
            isinstance(var, Var) for var in self.global_vars.values()
        )

    def lookup_var(self, name):
        assert name in self.functions, (name, self.functions.keys())
        if name not in self.global_vars:
            func = self.functions[name]
            if isinstance(func, Function):
                updated = dict(self.global_vars)
                updated[name] = Var(hint=None, type=FuncType.from_func(func), name=name)
                self.global_vars = updated
            else:
                raise ValueError()

        return self.global_vars[name]

    def add_function(self, name, func: Function):
        if name in self.functions:
            raise ValueError("Function {} has already existed in module.".format(name))
        else:
            updated = dict(self.functions)
            updated[name] = func
            self.functions = updated

    def copy(self):
        return IRModule(
            functions=dict(self.functions),
            global_vars=dict(self.global_vars),
            namespace=self.namespace,
            extern_functions=dict(self.extern_functions),
            include_headers=list(self.include_headers),
            include_dirs=list(self.include_dirs),
            linking_dirs=list(self.linking_dirs),
            linking_libs=list(self.linking_libs),
            object_files=list(self.object_files),
            task=self.task,
        )

    def build(self):
        """Build the IR module into a compiled module.

        Uses tilus's own lowering, codegen, and compilation pipeline.
        Returns a callable CompiledModule backed by the compiled .so file.
        """
        import os
        import tempfile

        from tilus.drivers import build_ir_module
        from tilus.runtime.compiled_program import CompiledModule

        output_dir = tempfile.mkdtemp(prefix="tilus_build_")
        build_ir_module(self, output_dir)
        return CompiledModule(os.path.join(output_dir, "lib.so"))

    def reset_funcs(self, functions: Dict[str, Function] = None, global_vars: Dict[str, Var] = None):
        self.functions = functions if functions else {}
        self.global_vars = global_vars if global_vars else {}
        return self


def merge_ir_modules(modules: Sequence[IRModule]) -> IRModule:
    if len(modules) == 0:
        return IRModule()
    first = modules[0]
    functions = dict(first.functions)
    global_vars = dict(first.global_vars)
    extern_functions = dict(first.extern_functions)
    include_headers = list(first.include_headers)
    include_dirs = list(first.include_dirs)
    linking_dirs = list(first.linking_dirs)
    linking_libs = list(first.linking_libs)
    object_files = list(first.object_files)

    for module in modules[1:]:
        if module.namespace != first.namespace:
            raise ValueError("Cannot merge IRModules with different namespaces")
        for name, var in module.global_vars.items():
            if name in global_vars:
                raise ValueError("Global variable {} has already existed in module.".format(name))
            global_vars[name] = var
        for name, func in module.functions.items():
            if name in functions:
                raise ValueError("Function {} has already existed in module.".format(name))
            functions[name] = func
        for name, var in module.extern_functions.items():
            if name in extern_functions:
                continue
            extern_functions[name] = var
        include_headers.extend([h for h in module.include_headers if h not in include_dirs])
        include_dirs.extend([d for d in module.include_dirs if d not in include_dirs])
        linking_dirs.extend([d for d in module.linking_dirs if d not in linking_dirs])
        linking_libs.extend([l for l in module.linking_libs if l not in linking_libs])
        object_files.extend([f for f in module.object_files if f not in object_files])

    return IRModule(
        functions=functions,
        global_vars=global_vars,
        namespace=first.namespace,
        extern_functions=extern_functions,
        include_headers=include_headers,
        include_dirs=include_dirs,
        linking_dirs=linking_dirs,
        linking_libs=linking_libs,
        object_files=object_files,
        task=first.task,
    )
