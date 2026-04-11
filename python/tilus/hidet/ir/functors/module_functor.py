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
# pylint: disable=bad-staticmethod-argument
from tilus.hidet.ir.func import Function
from tilus.hidet.ir.module import IRModule
from tilus.hidet.utils import same_list

from .base_functor import BaseFunctor, BaseRewriter, BaseVisitor


def _unchanged(a, b):
    """Check if a is the same as b, using same_as for tvm_ffi Objects."""
    return a is b or (hasattr(a, "same_as") and a.same_as(b))


class ModuleFunctor(BaseFunctor):
    _type_dispatch = {
        IRModule: "visit_IRModule",
        Function: "visit_Function",
    }

    def visit_dispatch(self, node):
        method_name = ModuleFunctor._type_dispatch.get(type(node))
        if method_name is not None:
            return getattr(self, method_name)(node)
        return NotImplemented

    def visit_IRModule(self, module: IRModule):
        raise NotImplementedError()

    def visit_Function(self, func: Function):
        raise NotImplementedError()


class ModuleVisitor(ModuleFunctor, BaseVisitor):
    def visit_IRModule(self, module: IRModule):
        self.visit(module.global_vars)
        self.visit(module.functions)

    def visit_Function(self, func: Function):
        self.visit(func.params)
        self.visit(func.ret_type)
        self.visit(func.body)
        self.visit(func.attrs)


class ModuleRewriter(ModuleFunctor, BaseRewriter):
    def visit_IRModule(self, module: IRModule):
        orig_global_vars = module.global_vars
        orig_functions = module.functions
        global_vars = self.visit(orig_global_vars)
        functions = self.visit(orig_functions)
        if same_list(global_vars, orig_global_vars) and _unchanged(functions, orig_functions):
            return module
        else:
            return module.copy().reset_funcs(functions, global_vars)

    def visit_Function(self, func: Function):
        orig_params = func.params
        orig_ret_type = func.ret_type
        orig_body = func.body
        orig_attrs = func.attrs
        params = self.visit(orig_params)
        ret_type = self.visit(orig_ret_type)
        body = self.visit(orig_body)
        attrs = self.visit(orig_attrs)
        if (
            same_list(params, orig_params)
            and _unchanged(ret_type, orig_ret_type)
            and _unchanged(body, orig_body)
            and _unchanged(attrs, orig_attrs)
        ):
            return func
        else:
            return Function(func.name, params, body, ret_type, func.kind, attrs)
