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
from typing import Any, Dict, Hashable, List, Mapping, Tuple, TypeVar, Union

import tvm_ffi

from tilus.hidet.ir.expr import Expr
from tilus.hidet.ir.type import BaseType
from tilus.ir._replace import replace
from tilus.ir.func import Function
from tilus.ir.inst import Instruction, InstructionConfig
from tilus.ir.layout import GlobalLayout, RegisterLayout, SharedLayout, TMemoryLayout
from tilus.ir.prog import Program
from tilus.ir.stmt import (
    AssignStmt,
    BreakStmt,
    DeclareStmt,
    EvaluateStmt,
    ForStmt,
    IfStmt,
    InstStmt,
    LetStmt,
    ReturnStmt,
    SeqStmt,
    Stmt,
    TensorItemPtrStmt,
    TensorItemValueStmt,
    ThreadGroupStmt,
    WhileStmt,
)
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedTensor, TMemoryTensor
from tilus.utils import same_list

InstClsVar = TypeVar("InstClsVar", bound=Instruction)


def _unchanged(a, b):
    """Check if a is the same as b, using same_as for tvm_ffi Objects."""
    return a is b or (hasattr(a, "same_as") and a.same_as(b))


class IRFunctor:
    def __init__(self):
        self.memo = {}

    def __call__(self, node):
        return self.visit(node)

    # Type dispatch table: maps concrete types to visitor method names.
    # Built lazily on first use to avoid import-order issues.
    _dispatch_table: Dict[type, str] = {}

    @classmethod
    def _build_dispatch_table(cls) -> Dict[type, str]:
        if cls._dispatch_table:
            return cls._dispatch_table
        table: Dict[type, str] = {
            InstStmt: "visit_InstStmt",
            Program: "visit_Program",
            Function: "visit_Function",
            SeqStmt: "visit_SeqStmt",
            ForStmt: "visit_ForStmt",
            ThreadGroupStmt: "visit_ThreadGroupStmt",
            IfStmt: "visit_IfStmt",
            WhileStmt: "visit_WhileStmt",
            BreakStmt: "visit_BreakStmt",
            ReturnStmt: "visit_ReturnStmt",
            DeclareStmt: "visit_DeclareStmt",
            LetStmt: "visit_LetStmt",
            AssignStmt: "visit_AssignStmt",
            EvaluateStmt: "visit_EvaluateStmt",
            TensorItemPtrStmt: "visit_TensorItemPtrStmt",
            TensorItemValueStmt: "visit_TensorItemValueStmt",
            RegisterTensor: "visit_RegisterTensor",
            SharedTensor: "visit_SharedTensor",
            GlobalTensor: "visit_GlobalTensor",
            TMemoryTensor: "visit_TMemoryTensor",
            RegisterLayout: "visit_RegisterLayout",
            SharedLayout: "visit_SharedLayout",
            GlobalLayout: "visit_GlobalLayout",
            TMemoryLayout: "visit_TMemoryLayout",
            list: "visit_list",
            tvm_ffi.Array: "visit_list",
            tuple: "visit_tuple",
            dict: "visit_dict",
            tvm_ffi.Map: "visit_dict",
            int: "visit_PyConstant",
            float: "visit_PyConstant",
            bool: "visit_PyConstant",
            str: "visit_PyConstant",
            type(None): "visit_PyConstant",
        }
        cls._dispatch_table = table
        return table

    def visit(self, node):
        key: Hashable
        node_type = type(node)
        if node_type in (list, tuple, dict):
            key = id(node)
        elif node_type in (tvm_ffi.Array, tvm_ffi.Map):
            key = ("__container__", node.__chandle__())
        elif node_type in (str, int, float, bool):
            key = (node_type, node)
        else:
            key = node
        if key in self.memo:
            return self.memo[key]

        # Fast path: exact type match in dispatch table
        table = self._build_dispatch_table()
        method_name = table.get(node_type)
        if method_name is not None:
            ret = getattr(self, method_name)(node)
        # Instruction subclasses: check class name for specific visitor, fall back to generic
        elif isinstance(node, Instruction):
            visit_method = getattr(self.__class__, "visit_" + node_type.__name__, None)
            ret = visit_method(self, node) if visit_method else self.visit_Instruction(node)
        elif isinstance(node, InstructionConfig):
            visit_method = getattr(self.__class__, "visit_" + node_type.__name__, None)
            ret = visit_method(self, node) if visit_method else self.visit_InstructionConfig(node)
        # Fallback for subclasses not in the table
        elif isinstance(node, Stmt):
            # Try dispatch by name first
            visit_method = getattr(self.__class__, "visit_" + node_type.__name__, None)
            if visit_method:
                ret = visit_method(self, node)
            else:
                raise NotImplementedError(node_type.__name__)
        elif isinstance(node, Expr):
            ret = self.visit_Expr(node)
        elif isinstance(node, BaseType):
            ret = self.visit_BaseType(node)
        else:
            raise NotImplementedError(node_type.__name__)

        self.memo[key] = ret
        return ret

    def visit_Instruction(self, inst: Instruction) -> Any:
        raise NotImplementedError()

    def visit_InstructionConfig(self, inst_config: InstructionConfig) -> Any:
        raise NotImplementedError()

    def visit_list(self, lst: List) -> Any:
        raise NotImplementedError()

    def visit_tuple(self, lst: Tuple) -> Any:
        raise NotImplementedError()

    def visit_dict(self, node: Dict) -> Any:
        raise NotImplementedError()

    def visit_PyConstant(self, node: Union[int, float, bool, str, None]) -> Any:
        raise NotImplementedError()

    def visit_Expr(self, expr: Expr) -> Any:
        raise NotImplementedError()

    def visit_BaseType(self, tp: BaseType) -> Any:
        raise NotImplementedError()

    def visit_Program(self, prog: Program) -> Any:
        raise NotImplementedError()

    def visit_Function(self, func: Function) -> Any:
        raise NotImplementedError()

    # statements

    def visit_InstStmt(self, stmt: InstStmt) -> Any:
        raise NotImplementedError()

    def visit_SeqStmt(self, stmt: SeqStmt) -> Any:
        raise NotImplementedError()

    def visit_ForStmt(self, stmt: ForStmt) -> Any:
        raise NotImplementedError()

    def visit_ThreadGroupStmt(self, stmt: ThreadGroupStmt) -> Any:
        raise NotImplementedError()

    def visit_IfStmt(self, stmt: IfStmt) -> Any:
        raise NotImplementedError()

    def visit_WhileStmt(self, stmt: WhileStmt) -> Any:
        raise NotImplementedError()

    def visit_BreakStmt(self, stmt: BreakStmt) -> Any:
        raise NotImplementedError()

    def visit_ReturnStmt(self, stmt: ReturnStmt) -> Any:
        raise NotImplementedError()

    def visit_DeclareStmt(self, stmt: DeclareStmt) -> Any:
        raise NotImplementedError()

    def visit_LetStmt(self, stmt: LetStmt) -> Any:
        raise NotImplementedError()

    def visit_AssignStmt(self, stmt: AssignStmt) -> Any:
        raise NotImplementedError()

    def visit_EvaluateStmt(self, stmt: EvaluateStmt) -> Any:
        raise NotImplementedError()

    def visit_TensorItemPtrStmt(self, stmt: TensorItemPtrStmt) -> Any:
        raise NotImplementedError()

    def visit_TensorItemValueStmt(self, stmt: TensorItemValueStmt) -> Any:
        raise NotImplementedError()

    # tensors and layouts

    def visit_RegisterTensor(self, tensor: RegisterTensor) -> Any:
        raise NotImplementedError()

    def visit_SharedTensor(self, tensor: SharedTensor) -> Any:
        raise NotImplementedError()

    def visit_GlobalTensor(self, tensor: GlobalTensor) -> Any:
        raise NotImplementedError()

    def visit_TMemoryTensor(self, tensor: TMemoryTensor) -> Any:
        raise NotImplementedError()

    def visit_RegisterLayout(self, layout: RegisterLayout) -> Any:
        raise NotImplementedError()

    def visit_SharedLayout(self, node: SharedLayout) -> Any:
        raise NotImplementedError()

    def visit_GlobalLayout(self, node: GlobalLayout) -> Any:
        raise NotImplementedError()

    def visit_TMemoryLayout(self, layout: TMemoryLayout) -> Any:
        raise NotImplementedError()


class IRRewriter(IRFunctor):
    def visit_list(self, lst: List) -> List:
        updated = [self.visit(item) for item in lst]
        if same_list(lst, updated):
            return lst
        else:
            return updated

    def visit_tuple(self, lst: Tuple) -> Tuple:
        updated = tuple(self.visit(item) for item in lst)
        if same_list(lst, updated):
            return lst
        else:
            return updated

    def visit_dict(self, node: Dict) -> Dict:
        updated = type(node)({key: self.visit(value) for key, value in node.items()})
        if same_list(list(node.values()), list(updated.values())):
            return node
        else:
            return updated

    def visit_PyConstant(self, node: Union[int, float, bool, str, None]) -> Union[int, float, bool, str, None]:
        return node

    def visit_Expr(self, expr: Expr) -> Expr:
        from tilus.hidet.ir.tools import rewrite

        return rewrite(expr, rewrite_map=self.memo)

    def visit_BaseType(self, tp: BaseType) -> BaseType:
        return tp

    def visit_Program(self, prog: Program) -> Program:
        functions = self.visit(prog.functions)
        if same_list([functions], [prog.functions]):
            return prog
        else:
            return Program(functions=functions)

    def visit_Function(self, func: Function) -> Function:
        orig_body = func.body
        body = self.visit(orig_body)
        if _unchanged(body, orig_body):
            return func
        else:
            return Function(
                name=func.name,
                params=func.params,
                body=body,
                metadata=func.metadata,
            )

    def visit_InstStmt(self, stmt: InstStmt) -> Stmt:
        inst_or_stmt = self.visit(stmt.inst)
        if isinstance(inst_or_stmt, Stmt):
            return inst_or_stmt
        elif isinstance(inst_or_stmt, Instruction):
            return InstStmt(inst_or_stmt)
        elif inst_or_stmt is None:
            return SeqStmt(())
        else:
            raise ValueError(f"An instruction should be rewritten to an instruction or a statement, got {inst_or_stmt}")

    def visit_SeqStmt(self, stmt: SeqStmt) -> Stmt:
        orig_seq = stmt.seq
        seq = self.visit(orig_seq)
        if _unchanged(seq, orig_seq):
            return stmt
        else:
            return SeqStmt(seq)

    def visit_ForStmt(self, stmt: ForStmt) -> Stmt:
        orig_extent = stmt.extent
        orig_body = stmt.body
        extent = self.visit(orig_extent)
        body = self.visit(orig_body)
        if _unchanged(extent, orig_extent) and _unchanged(body, orig_body):
            return stmt
        else:
            return ForStmt(stmt.iter_var, extent, body, stmt.unroll_factor)

    def visit_ThreadGroupStmt(self, stmt: ThreadGroupStmt) -> Stmt:
        orig_body = stmt.body
        body = self.visit(orig_body)
        if _unchanged(body, orig_body):
            return stmt
        else:
            return ThreadGroupStmt(stmt.thread_begin, stmt.num_threads, body)

    def visit_IfStmt(self, stmt: IfStmt) -> Stmt:
        orig_cond = stmt.cond
        orig_then = stmt.then_body
        orig_else = stmt.else_body
        cond = self.visit(orig_cond)
        then_body = self.visit(orig_then)
        else_body = self.visit(orig_else)
        if _unchanged(cond, orig_cond) and _unchanged(then_body, orig_then) and _unchanged(else_body, orig_else):
            return stmt
        else:
            return IfStmt(cond, then_body, else_body)

    def visit_BreakStmt(self, stmt: BreakStmt) -> Stmt:
        return stmt

    def visit_ReturnStmt(self, stmt: ReturnStmt) -> Stmt:
        return stmt

    def visit_DeclareStmt(self, stmt: DeclareStmt) -> Stmt:
        orig_init = stmt.init
        init = self.visit(orig_init)
        if _unchanged(init, orig_init):
            return stmt
        else:
            return DeclareStmt(stmt.var, init)

    def visit_LetStmt(self, stmt: LetStmt) -> Stmt:
        orig_bind_values = stmt.bind_values
        orig_body = stmt.body
        bind_values = self.visit(orig_bind_values)
        body = self.visit(orig_body)
        if _unchanged(bind_values, orig_bind_values) and _unchanged(body, orig_body):
            return stmt
        else:
            return LetStmt(stmt.bind_vars, bind_values, body)

    def visit_AssignStmt(self, stmt: AssignStmt) -> Stmt:
        orig_value = stmt.value
        value = self.visit(orig_value)
        if _unchanged(value, orig_value):
            return stmt
        else:
            return AssignStmt(stmt.var, value)

    def visit_EvaluateStmt(self, stmt):
        orig_expr = stmt.expr
        orig_pred = stmt.pred
        expr = self.visit(orig_expr)
        pred = self.visit(orig_pred)
        if _unchanged(expr, orig_expr) and _unchanged(pred, orig_pred):
            return stmt
        else:
            return EvaluateStmt(expr=expr, pred=pred)

    def visit_TensorItemPtrStmt(self, stmt: TensorItemPtrStmt) -> Stmt:
        orig_tensor = stmt.tensor
        tensor = self.visit(orig_tensor)
        if _unchanged(tensor, orig_tensor):
            return stmt
        else:
            return TensorItemPtrStmt(stmt.ptr_var, tensor, stmt.space)

    def visit_TensorItemValueStmt(self, stmt: TensorItemValueStmt) -> Stmt:
        orig_tensor = stmt.tensor
        tensor = self.visit(orig_tensor)
        if _unchanged(tensor, orig_tensor):
            return stmt
        else:
            return TensorItemValueStmt(stmt.var, tensor)

    def visit_WhileStmt(self, stmt: WhileStmt) -> Stmt:
        orig_cond = stmt.cond
        orig_body = stmt.body
        cond = self.visit(orig_cond)
        body = self.visit(orig_body)
        if _unchanged(cond, orig_cond) and _unchanged(body, orig_body):
            return stmt
        else:
            return WhileStmt(cond, body)

    def visit_RegisterTensor(self, tensor: RegisterTensor) -> RegisterTensor:
        orig_layout = tensor.optional_layout
        optional_layout = self.visit(orig_layout)
        if _unchanged(optional_layout, orig_layout):
            return tensor
        else:
            return RegisterTensor.create(dtype=tensor.dtype, shape=tensor.shape, optional_layout=optional_layout)

    def visit_SharedTensor(self, tensor: SharedTensor) -> SharedTensor:
        orig_layout = tensor.optional_layout
        optional_layout = self.visit(orig_layout)
        if _unchanged(optional_layout, orig_layout):
            return tensor
        else:
            return SharedTensor.create(dtype=tensor.dtype, shape=tensor.shape, optional_layout=optional_layout)

    def visit_GlobalTensor(self, tensor: GlobalTensor) -> GlobalTensor:
        orig_layout = tensor.layout
        layout = self.visit(orig_layout)
        if _unchanged(layout, orig_layout):
            return tensor
        else:
            return GlobalTensor.create(dtype=tensor.dtype, layout=layout)

    def visit_TMemoryTensor(self, tensor: TMemoryTensor) -> TMemoryTensor:
        orig_layout = tensor.optional_layout
        optional_layout = self.visit(orig_layout)
        if _unchanged(optional_layout, orig_layout):
            return tensor
        else:
            return TMemoryTensor.create(dtype=tensor.dtype, shape=tensor.shape, optional_layout=optional_layout)

    def visit_RegisterLayout(self, layout: RegisterLayout) -> RegisterLayout:
        return layout

    def visit_SharedLayout(self, layout: SharedLayout) -> SharedLayout:
        return layout

    def visit_GlobalLayout(self, layout: GlobalLayout) -> GlobalLayout:
        orig_shape = layout.shape
        orig_size = layout.size
        orig_offset = layout.offset
        shape = self.visit(orig_shape)
        size = self.visit(orig_size)
        offset = self.visit(orig_offset)

        if _unchanged(shape, orig_shape) and _unchanged(offset, orig_offset) and _unchanged(size, orig_size):
            return layout
        else:
            return GlobalLayout(shape=shape, size=size, axes=layout.axes, offset=offset)

    def visit_TMemoryLayout(self, layout: TMemoryLayout) -> TMemoryLayout:
        return layout

    # instructions
    def visit_Instruction(self, inst: InstClsVar) -> InstClsVar:
        orig_output = inst.output
        orig_inputs = inst.inputs
        orig_attributes = inst.attributes
        output = self.visit(orig_output)
        inputs = self.visit(orig_inputs)
        attributes: Mapping[str, Any] = {key: self.visit(value) for key, value in orig_attributes.items()}

        if (
            _unchanged(output, orig_output)
            and _unchanged(inputs, orig_inputs)
            and all(_unchanged(a, b) for a, b in zip(attributes.values(), orig_attributes.values()))
        ):
            return inst
        else:
            return replace(inst, output=output, inputs=inputs, **attributes)

    # instruction configs
    def visit_InstructionConfig(self, inst_config: InstructionConfig) -> Any:
        return inst_config


class IRVisitor(IRFunctor):
    def visit_list(self, lst: List) -> None:
        for item in lst:
            self.visit(item)

    def visit_tuple(self, lst: Tuple) -> None:
        for item in lst:
            self.visit(item)

    def visit_dict(self, node: Dict) -> None:
        for k, v in node.items():
            self.visit(v)

    def visit_PyConstant(self, node: Union[int, float, bool, str, None]) -> None:
        pass

    def visit_Expr(self, expr: Expr) -> None:
        pass

    def visit_BaseType(self, tp: BaseType) -> None:
        pass

    def visit_Program(self, prog: Program) -> None:
        self.visit(prog.functions)

    def visit_Function(self, func: Function) -> None:
        self.visit(func.body)

    def visit_InstStmt(self, stmt: InstStmt) -> None:
        self.visit(stmt.inst)

    def visit_SeqStmt(self, stmt: SeqStmt) -> None:
        for sub_stmt in stmt.seq:
            self.visit(sub_stmt)

    def visit_ForStmt(self, stmt: ForStmt) -> None:
        self.visit(stmt.extent)
        self.visit(stmt.body)

    def visit_ThreadGroupStmt(self, stmt: ThreadGroupStmt) -> None:
        self.visit(stmt.body)

    def visit_IfStmt(self, stmt: IfStmt) -> None:
        self.visit(stmt.cond)
        self.visit(stmt.then_body)
        if stmt.else_body is not None:
            self.visit(stmt.else_body)

    def visit_BreakStmt(self, stmt: BreakStmt) -> None:
        pass

    def visit_ReturnStmt(self, stmt: ReturnStmt) -> None:
        pass

    def visit_WhileStmt(self, stmt: WhileStmt) -> None:
        self.visit(stmt.cond)
        self.visit(stmt.body)

    def visit_DeclareStmt(self, stmt: DeclareStmt) -> None:
        self.visit(stmt.init)

    def visit_LetStmt(self, stmt: LetStmt) -> Any:
        self.visit(stmt.bind_values)
        self.visit(stmt.body)

    def visit_AssignStmt(self, stmt: AssignStmt) -> None:
        self.visit(stmt.var)
        self.visit(stmt.value)

    def visit_EvaluateStmt(self, stmt: EvaluateStmt) -> None:
        self.visit(stmt.expr)
        self.visit(stmt.pred)

    def visit_TensorItemPtrStmt(self, stmt: TensorItemPtrStmt) -> None:
        self.visit(stmt.ptr_var)
        self.visit(stmt.tensor)

    def visit_TensorItemValueStmt(self, stmt: TensorItemValueStmt) -> None:
        self.visit(stmt.var)
        self.visit(stmt.tensor)

    # values

    def visit_RegisterTensor(self, tensor: RegisterTensor) -> None:
        self.visit(tensor.optional_layout)

    def visit_SharedTensor(self, tensor: SharedTensor) -> None:
        self.visit(tensor.optional_layout)

    def visit_GlobalTensor(self, tensor: GlobalTensor) -> None:
        self.visit(tensor.layout)

    def visit_TMemoryTensor(self, tensor: TMemoryTensor) -> None:
        self.visit(tensor.optional_layout)

    def visit_RegisterLayout(self, layout: RegisterLayout) -> None:
        pass

    def visit_SharedLayout(self, layout: SharedLayout) -> None:
        pass

    def visit_GlobalLayout(self, layout: GlobalLayout) -> None:
        self.visit(layout.shape)
        self.visit(layout.offset)

    def visit_TMemoryLayout(self, layout: TMemoryLayout) -> None:
        pass

    # instructions
    def visit_Instruction(self, inst: Instruction) -> None:
        self.visit(inst.output)
        self.visit(inst.inputs)
        self.visit(inst.attributes)

    def visit_InstructionConfig(self, inst_config: InstructionConfig) -> None:
        pass
