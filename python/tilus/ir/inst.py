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

from typing import Any, Optional

from tvm_ffi import pyast
from tvm_ffi.access_path import AccessPath
from tvm_ffi.dataclasses import py_class

from tilus.ir.node import IRNode
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedTensor, Tensor, TMemoryTensor


@py_class
class Instruction(IRNode):
    output: Optional[Tensor]
    inputs: tuple[Tensor, ...]

    def __ffi_text_print__(self, printer: pyast.IRPrinter, path: AccessPath):
        inst_name = type(self).__name__.removesuffix("Inst")
        args = [printer(inp, path.attr("inputs").array_item(i)) for i, inp in enumerate(self.inputs)]
        kwargs_keys = []
        kwargs_values = []
        for k, v in self.attributes.items():
            if v is None:
                continue
            kwargs_keys.append(k)
            kwargs_values.append(printer(v, path.attr(k)))
        if kwargs_keys:
            rhs = pyast.Call(pyast.Id(inst_name), args, kwargs_keys, kwargs_values)
        else:
            rhs = pyast.Call(pyast.Id(inst_name), args)
        if self.output is not None:
            lhs = printer(self.output, path.attr("output"))
            return pyast.Assign(lhs, rhs)
        return rhs

    @property
    def shared_output(self) -> SharedTensor:
        assert isinstance(self.output, SharedTensor), self.output
        return self.output

    @property
    def register_output(self) -> RegisterTensor:
        assert isinstance(self.output, RegisterTensor), self.output
        return self.output

    @property
    def register_or_shared_output(self) -> SharedTensor | RegisterTensor:
        assert isinstance(self.output, SharedTensor) or isinstance(self.output, RegisterTensor), self.output
        return self.output

    @property
    def global_output(self) -> GlobalTensor:
        assert isinstance(self.output, GlobalTensor), self.output
        return self.output

    @property
    def tmemory_output(self) -> TMemoryTensor:
        assert isinstance(self.output, TMemoryTensor), self.output
        return self.output

    @property
    def register_input(self) -> RegisterTensor:
        assert len(self.inputs) == 1
        x = self.inputs[0]
        assert isinstance(x, RegisterTensor)
        return x

    @property
    def shared_input(self) -> SharedTensor:
        assert len(self.inputs) == 1
        x = self.inputs[0]
        assert isinstance(x, SharedTensor)
        return x

    @property
    def global_input(self) -> GlobalTensor:
        assert len(self.inputs) == 1
        x = self.inputs[0]
        assert isinstance(x, GlobalTensor)
        return x

    @property
    def tmemory_input(self) -> TMemoryTensor:
        assert len(self.inputs) == 1
        x = self.inputs[0]
        assert isinstance(x, TMemoryTensor)
        return x

    @property
    def register_or_shared_input(self) -> RegisterTensor | SharedTensor:
        assert len(self.inputs) == 1
        x = self.inputs[0]
        assert isinstance(x, RegisterTensor) or isinstance(x, SharedTensor)
        return x

    @property
    def attributes(self) -> dict[str, Any]:
        attrs = {}
        skip = {"output", "inputs"}
        for cls in type(self).__mro__:
            if cls is object:
                break
            for name, anno in getattr(cls, "__annotations__", {}).items():
                if name in skip or name in attrs:
                    continue
                # Skip ClassVar fields (with from __future__ import annotations, they're strings)
                if isinstance(anno, str) and "ClassVar" in anno:
                    continue
                attrs[name] = getattr(self, name)
        return attrs


@py_class
class InstructionConfig(IRNode):
    pass


class InstructionError(Exception):
    """Exception raised when the parameters of an instruction are invalid."""
