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
from tilus.hidet.ir.dialects.pattern import PlaceholderExpr
from tilus.hidet.ir.expr import (
    Add,
    Address,
    BinaryExpr,
    BitwiseAnd,
    BitwiseNot,
    BitwiseOr,
    BitwiseXor,
    Call,
    Cast,
    Constant,
    Dereference,
    Div,
    Equal,
    Expr,
    FloorDiv,
    IfThenElse,
    LeftShift,
    LessEqual,
    LessThan,
    Let,
    LogicalAnd,
    LogicalNot,
    LogicalOr,
    Mod,
    Multiply,
    Neg,
    NotEqual,
    Reference,
    RightShift,
    Sub,
    SymbolVar,
    TensorElement,
    TensorSlice,
    Var,
    cast,
)
from tilus.hidet.utils import same_list

from .base_functor import BaseFunctor, BaseRewriter, BaseVisitor


def _unchanged(a, b):
    """Check if a is the same as b, using same_as for tvm_ffi Objects."""
    return a is b or (hasattr(a, "same_as") and a.same_as(b))


class ExprFunctor(BaseFunctor):
    _type_dispatch = {
        Var: "visit_Var",
        SymbolVar: "visit_Var",
        Add: "visit_Add",
        Sub: "visit_Sub",
        Multiply: "visit_Multiply",
        Div: "visit_Div",
        Mod: "visit_Mod",
        FloorDiv: "visit_FloorDiv",
        Neg: "visit_Neg",
        LessThan: "visit_LessThan",
        LessEqual: "visit_LessEqual",
        Equal: "visit_Equal",
        NotEqual: "visit_NotEqual",
        LogicalAnd: "visit_And",
        LogicalOr: "visit_Or",
        LogicalNot: "visit_Not",
        BitwiseAnd: "visit_BitwiseAnd",
        BitwiseOr: "visit_BitwiseOr",
        BitwiseNot: "visit_BitwiseNot",
        BitwiseXor: "visit_BitwiseXor",
        LeftShift: "visit_LeftShift",
        RightShift: "visit_RightShift",
        TensorElement: "visit_TensorElement",
        TensorSlice: "visit_TensorSlice",
        IfThenElse: "visit_IfThenElse",
        Call: "visit_Call",
        Let: "visit_Let",
        Constant: "visit_Constant",
        Cast: "visit_Cast",
        Dereference: "visit_Dereference",
        Address: "visit_Address",
        Reference: "visit_Reference",
        PlaceholderExpr: "visit_PlaceholderExpr",
    }

    def visit_dispatch(self, node):
        method_name = ExprFunctor._type_dispatch.get(type(node))
        if method_name is not None:
            return getattr(self, method_name)(node)
        return NotImplemented

    def visit_Add(self, e: Add):
        raise NotImplementedError()

    def visit_Sub(self, e: Sub):
        raise NotImplementedError()

    def visit_Multiply(self, e: Multiply):
        raise NotImplementedError()

    def visit_Div(self, e: Div):
        raise NotImplementedError()

    def visit_Mod(self, e: Mod):
        raise NotImplementedError()

    def visit_FloorDiv(self, e: FloorDiv):
        raise NotImplementedError()

    def visit_LessThan(self, e: LessThan):
        raise NotImplementedError()

    def visit_LessEqual(self, e: LessEqual):
        raise NotImplementedError()

    def visit_Equal(self, e: Equal):
        raise NotImplementedError()

    def visit_NotEqual(self, e: NotEqual):
        raise NotImplementedError()

    def visit_And(self, e: LogicalAnd):
        raise NotImplementedError()

    def visit_Or(self, e: LogicalOr):
        raise NotImplementedError()

    def visit_Neg(self, e: Neg):
        raise NotImplementedError()

    def visit_Not(self, e: LogicalNot):
        raise NotImplementedError()

    def visit_BitwiseAnd(self, e: BitwiseAnd):
        raise NotImplementedError()

    def visit_BitwiseOr(self, e: BitwiseOr):
        raise NotImplementedError()

    def visit_BitwiseNot(self, e: BitwiseNot):
        raise NotImplementedError()

    def visit_BitwiseXor(self, e: BitwiseXor):
        raise NotImplementedError()

    def visit_LeftShift(self, e: LeftShift):
        raise NotImplementedError()

    def visit_RightShift(self, e: RightShift):
        raise NotImplementedError()

    def visit_TensorElement(self, e: TensorElement):
        raise NotImplementedError()

    def visit_TensorSlice(self, e: TensorSlice):
        raise NotImplementedError()

    def visit_IfThenElse(self, e: IfThenElse):
        raise NotImplementedError()

    def visit_Cast(self, e: Cast):
        raise NotImplementedError()

    def visit_Dereference(self, e: Dereference):
        raise NotImplementedError()

    def visit_Address(self, e: Address):
        raise NotImplementedError()

    def visit_Reference(self, e: Reference):
        raise NotImplementedError()

    def visit_Call(self, e: Call):
        raise NotImplementedError()

    def visit_Let(self, e: Let):
        raise NotImplementedError()

    def visit_Var(self, e: Var):
        raise NotImplementedError()

    def visit_Constant(self, e: Constant):
        raise NotImplementedError()

    def visit_PlaceholderExpr(self, e: PlaceholderExpr):
        raise NotImplementedError()


class ExprVisitor(ExprFunctor, BaseVisitor):
    def visit_Add(self, e: Add):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Sub(self, e: Sub):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Multiply(self, e: Multiply):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Div(self, e: Div):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Mod(self, e: Mod):
        self.visit(e.a)
        self.visit(e.b)

    def visit_FloorDiv(self, e: FloorDiv):
        self.visit(e.a)
        self.visit(e.b)

    def visit_LessThan(self, e: LessThan):
        self.visit(e.a)
        self.visit(e.b)

    def visit_LessEqual(self, e: LessEqual):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Equal(self, e: Equal):
        self.visit(e.a)
        self.visit(e.b)

    def visit_NotEqual(self, e: NotEqual):
        self.visit(e.a)
        self.visit(e.b)

    def visit_And(self, e: LogicalAnd):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Or(self, e: LogicalOr):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Neg(self, e: Neg):
        self.visit(e.a)

    def visit_Not(self, e: LogicalNot):
        self.visit(e.a)

    def visit_BitwiseAnd(self, e: BitwiseAnd):
        self.visit(e.a)
        self.visit(e.b)

    def visit_BitwiseOr(self, e: BitwiseOr):
        self.visit(e.a)
        self.visit(e.b)

    def visit_BitwiseXor(self, e: BitwiseXor):
        self.visit(e.a)
        self.visit(e.b)

    def visit_BitwiseNot(self, e: BitwiseNot):
        self.visit(e.a)

    def visit_LeftShift(self, e: LeftShift):
        self.visit(e.a)
        self.visit(e.b)

    def visit_RightShift(self, e: RightShift):
        self.visit(e.a)
        self.visit(e.b)

    def visit_TensorElement(self, e: TensorElement):
        self.visit(e.base)
        for idx in e.indices:
            self.visit(idx)

    def visit_TensorSlice(self, e: TensorSlice):
        self.visit(e.base)
        for idx, start, end in zip(e.starts, e.indices, e.ends):
            for obj in [idx, start, end]:
                if obj is not None:
                    self.visit(obj)

    def visit_IfThenElse(self, e: IfThenElse):
        self.visit(e.cond)
        self.visit(e.then_expr)
        self.visit(e.else_expr)

    def visit_Call(self, e: Call):
        self.visit(e.func_var)
        for arg in e.args:
            self.visit(arg)

    def visit_Let(self, e: Let):
        self.visit(e.value)
        self.visit(e.var)
        self.visit(e.body)

    def visit_Var(self, e: Var):
        pass

    def visit_Constant(self, e: Constant):
        pass

    # low level dialect
    def visit_Cast(self, e: Cast):
        self.visit(e.expr)

    def visit_Dereference(self, e: Dereference):
        self.visit(e.expr)

    def visit_Address(self, e: Address):
        self.visit(e.expr)

    def visit_Reference(self, e: Reference):
        self.visit(e.expr)

    def visit_PlaceholderExpr(self, e: PlaceholderExpr):
        pass


class ExprRewriter(ExprFunctor, BaseRewriter):
    def rewrite(self, e):
        return self.visit(e)

    def visit_Binary(self, e: BinaryExpr):
        orig_a = e.a
        orig_b = e.b
        a = self(orig_a)
        b = self(orig_b)
        if _unchanged(a, orig_a) and _unchanged(b, orig_b):
            return e
        else:
            return Expr._binary(e.__class__, a, b)  # pylint: disable=protected-access

    def visit_Add(self, e: Add):
        return self.visit_Binary(e)

    def visit_Sub(self, e: Sub):
        return self.visit_Binary(e)

    def visit_Multiply(self, e: Multiply):
        return self.visit_Binary(e)

    def visit_Div(self, e: Div):
        return self.visit_Binary(e)

    def visit_Mod(self, e: Mod):
        return self.visit_Binary(e)

    def visit_FloorDiv(self, e: FloorDiv):
        return self.visit_Binary(e)

    def visit_LessThan(self, e: LessThan):
        return self.visit_Binary(e)

    def visit_LessEqual(self, e: LessEqual):
        return self.visit_Binary(e)

    def visit_Equal(self, e: Equal):
        return self.visit_Binary(e)

    def visit_NotEqual(self, e: NotEqual):
        return self.visit_Binary(e)

    def visit_And(self, e: LogicalAnd):
        return self.visit_Binary(e)

    def visit_Or(self, e: LogicalOr):
        return self.visit_Binary(e)

    def visit_Neg(self, e: Neg):
        orig_a = e.a
        a = self(orig_a)
        if _unchanged(a, orig_a):
            return e
        else:
            return Neg(a)

    def visit_Not(self, e: LogicalNot):
        orig_a = e.a
        a = self(orig_a)
        if _unchanged(a, orig_a):
            return e
        else:
            return LogicalNot(a)

    def visit_BitwiseAnd(self, e: BitwiseAnd):
        return self.visit_Binary(e)

    def visit_BitwiseOr(self, e: BitwiseOr):
        return self.visit_Binary(e)

    def visit_BitwiseXor(self, e: BitwiseXor):
        return self.visit_Binary(e)

    def visit_BitwiseNot(self, e: BitwiseNot):
        orig_a = e.a
        base = self.visit(orig_a)
        if _unchanged(base, orig_a):
            return e
        else:
            return BitwiseNot(base)

    def visit_LeftShift(self, e: LeftShift):
        orig_a = e.a
        orig_b = e.b
        base = self.visit(orig_a)
        cnt = self.visit(orig_b)
        if _unchanged(base, orig_a) and _unchanged(cnt, orig_b):
            return e
        else:
            return LeftShift(base, cnt)

    def visit_RightShift(self, e: RightShift):
        orig_a = e.a
        orig_b = e.b
        base = self.visit(orig_a)
        cnt = self.visit(orig_b)
        if _unchanged(base, orig_a) and _unchanged(cnt, orig_b):
            return e
        else:
            return RightShift(base, cnt)

    def visit_TensorElement(self, e: TensorElement):
        orig_base = e.base
        orig_indices = e.indices
        base = self(orig_base)
        indices = tuple(self(idx) if idx is not None else None for idx in orig_indices)
        if _unchanged(base, orig_base) and same_list(indices, orig_indices):
            return e
        else:
            return TensorElement(base, indices, e.protected)

    def visit_TensorSlice(self, e: TensorSlice):
        orig_base = e.base
        orig_indices = e.indices
        orig_starts = e.starts
        orig_ends = e.ends
        base = self(orig_base)
        indices = tuple(self(idx) if idx is not None else None for idx in orig_indices)
        starts = tuple(self(start) if start is not None else None for start in orig_starts)
        ends = tuple(self(end) if end is not None else None for end in orig_ends)
        if (
            _unchanged(base, orig_base)
            and same_list(indices, orig_indices)
            and same_list(starts, orig_starts)
            and same_list(ends, orig_ends)
        ):
            return e
        else:
            return TensorSlice(base, indices, starts, ends)

    def visit_IfThenElse(self, e: IfThenElse):
        orig_cond = e.cond
        orig_then = e.then_expr
        orig_else = e.else_expr
        cond = self(orig_cond)
        then_expr = self(orig_then)
        else_expr = self(orig_else)
        if _unchanged(cond, orig_cond) and _unchanged(then_expr, orig_then) and _unchanged(else_expr, orig_else):
            return e
        else:
            return IfThenElse(cond, then_expr, else_expr)

    def visit_Cast(self, e: Cast):
        orig_expr = e.expr
        expr = self(orig_expr)
        if _unchanged(expr, orig_expr):
            return e
        else:
            return cast(expr, e.target_type)

    def visit_Dereference(self, e: Dereference):
        orig_expr = e.expr
        expr = self(orig_expr)
        if _unchanged(expr, orig_expr):
            return e
        else:
            return Dereference(expr)

    def visit_Address(self, e: Address):
        orig_expr = e.expr
        expr = self(orig_expr)
        if _unchanged(expr, orig_expr):
            return e
        else:
            return Address(expr)

    def visit_Reference(self, e: Reference):
        orig_expr = e.expr
        expr = self(orig_expr)
        if _unchanged(expr, orig_expr):
            return e
        else:
            return Reference(expr)

    def visit_Call(self, e: Call):
        orig_func_var = e.func_var
        orig_args = e.args
        func_var = self(orig_func_var)
        args = tuple(self(arg) for arg in orig_args)
        if _unchanged(func_var, orig_func_var) and same_list(args, orig_args):
            return e
        else:
            return Call(func_var, args)

    def visit_Let(self, e: Let):
        orig_var = e.var
        orig_value = e.value
        orig_body = e.body
        var = self(orig_var)
        value = self(orig_value)
        body = self(orig_body)
        if same_list([var, value, body], [orig_var, orig_value, orig_body]):
            return e
        else:
            return Let(var, value, body)

    def visit_Var(self, e: Var):
        orig_tp = e.type
        tp = self(orig_tp)
        if tp == orig_tp:
            return e
        else:
            assert not isinstance(tp, SymbolVar)
            return Var(e.hint, tp, e.name)

    def visit_Constant(self, e: Constant):
        return e

    def visit_PlaceholderExpr(self, e: PlaceholderExpr):
        return e
