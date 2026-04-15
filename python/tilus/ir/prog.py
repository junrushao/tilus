# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from __future__ import annotations

from typing import Any

from tvm_ffi import ir_traits as tr
from tvm_ffi.dataclasses import py_class

from tilus.ir.func import Function
from tilus.ir.node import IRNode
from tilus.ir.utils import frozendict


@py_class
class Program(IRNode):
    """A program is a collection of named functions."""

    __ffi_ir_traits__ = tr.ModuleTraits("$field:functions")

    functions: Any

    @staticmethod
    def create(functions: dict[str, Function]) -> Program:
        return Program(frozendict(functions))

    def with_function(self, new_function: Function) -> Program:
        new_functions = dict(self.functions)
        new_functions[new_function.name] = new_function
        return Program(frozendict(new_functions))
