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
# pylint: disable=import-outside-toplevel, useless-parent-delegation, redefined-outer-name, redefined-builtin
# pylint: disable=useless-super-delegation, protected-access
from __future__ import annotations

import operator
import string
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import tvm_ffi
from tvm_ffi import ir_traits as tr
from tvm_ffi import pyast
from tvm_ffi.access_path import AccessPath
from tvm_ffi.dataclasses import field, py_class

from tilus.hidet.ir.dtypes import IntegerType, boolean, int32, int64, promote_type, uint64

from .dtypes import default_float_dtype, default_int_dtype
from .node import Node
from .type import (
    ArrayType,
    BaseType,
    DataType,
    FuncType,
    PointerType,
    StringType,
    TensorPointerType,
    TensorType,
    data_type,
    string_type,
    tensor_pointer_type,
    tensor_type,
    type_equal,
)

PyScalar = Union[bool, int, float, complex, str]


@py_class
class Expr(Node):
    def __bool__(self) -> bool:
        raise TypeError(
            "hidet.ir.Expr does not support pythonic logical operations (e.g., and, or, not, if(...)). "
            "Please use hidet.ir.if_then_else, hidet.ir.logical_and, hidet.ir.logical_or, hidet.ir.logical_or "
            "explicitly."
        )

    def __call__(self, *args, **kwargs):
        if len(kwargs) > 0:
            raise ValueError("Keyword arguments are not supported in hidet function calls.")
        if isinstance(self, Var) and self.type.is_func_type():
            return call(self, args)
        else:
            raise ValueError("Only function variable can be called.")

    def __neg__(self):
        return self._unary(Neg, self)

    def __add__(self, other):
        return self._binary(Add, self, other)

    def __radd__(self, other):
        return self._binary(Add, other, self)

    def __sub__(self, other):
        return self._binary(Sub, self, other)

    def __rsub__(self, other):
        return self._binary(Sub, other, self)

    def __mul__(self, other):
        return self._binary(Multiply, self, other)

    def __rmul__(self, other):
        return self._binary(Multiply, other, self)

    def __truediv__(self, other):
        return self._binary(Div, self, other)

    def __rtruediv__(self, other):
        return self._binary(Div, other, self)

    def __floordiv__(self, other):
        return self._binary(Div, self, other)

    def __rfloordiv__(self, other):
        return self._binary(Div, other, self)

    def __mod__(self, other):
        return self._binary(Mod, self, other)

    def __rmod__(self, other):
        return self._binary(Mod, other, self)

    def __lt__(self, other):
        return self._binary(LessThan, self, other)

    def __le__(self, other):
        return self._binary(LessEqual, self, other)

    def __eq__(self, other):
        # In hidet, we override the comparison operators: '<', '<=', '==', '!=', '>', '>=', which will return
        # the corresponding comparison expression. For example, `a < b` will return a LessThan expression.
        # An error will be raised if the result of the comparison is used in any place to evaluate the value in python:
        # `if a < b: ...` will raise an error.
        # This is because these comparison operators are used to construct the comparison expression in hidet IR.
        # However, there are two exceptions:
        #  1. if the comparison result is a constant (e.g., `int32(1) < int32(2)`), the result can be evaluated.
        #  2. if it is a `==` comparison, the result can be evaluated in python, and True if the two expressions are
        #     identical (e.g., `a is b`), False otherwise. We need this feature to use Expr as the keys in dict.
        # See `hidet.ir.expr.Constant.__bool__` and `hidet.ir.expr.Equal.__bool__` for the implementation details.
        return self._binary(Equal, self, other)

    def __ne__(self, other):
        return self._binary(NotEqual, self, other)

    def __lshift__(self, other):
        return self._binary(LeftShift, self, other)

    def __rshift__(self, other):
        return self._binary(RightShift, self, other)

    def __rlshift__(self, other):
        return self._binary(LeftShift, other, self)

    def __rrshift__(self, other):
        return self._binary(RightShift, other, self)

    def __gt__(self, other):
        return self._binary(LessThan, other, self)

    def __ge__(self, other):
        return self._binary(LessEqual, other, self)

    def __invert__(self):
        return Address(self)  # We override the invert operator `~a` as the addressing operator.

    def __or__(self, other):
        return self._binary(BitwiseOr, self, other)

    def __ror__(self, other):
        return self._binary(BitwiseOr, other, self)

    def __and__(self, other):
        return self._binary(BitwiseAnd, self, other)

    def __rand__(self, other):
        return self._binary(BitwiseAnd, other, self)

    def __xor__(self, other):
        return self._binary(BitwiseXor, self, other)

    def __rxor__(self, other):
        return self._binary(BitwiseXor, other, self)

    def __getitem__(self, items):
        if not isinstance(items, (tuple, list, tvm_ffi.Array)):
            items = [items]
        indices = []
        starts = []
        ends = []
        for item in items:
            if isinstance(item, slice):
                indices.append(None)
                starts.append(item.start)
                ends.append(item.stop)
                assert item.step is None, "do not support step slice"
            else:
                indices.append(item)
                starts.append(None)
                ends.append(None)
        rank = tensor_rank(self)
        if len(items) < rank or any(i is None for i in indices):
            while len(indices) < rank:
                indices.append(None)
                starts.append(None)
                ends.append(None)
            return tensor_slice(base=self, indices=indices, starts=starts, ends=ends)
        else:
            return tensor_element(base=self, indices=indices)

    def __setitem__(self, key, value):
        raise ValueError()

    def __str__(self):
        from tvm_ffi import pyast  # pylint: disable=import-outside-toplevel

        return pyast.to_python(self)

    def __int__(self):
        raise TypeError("Cannot convert hidet.ir.Expr to int.")

    def __float__(self):
        raise TypeError("Cannot convert hidet.ir.Expr to float.")

    def __complex__(self):
        raise TypeError("Cannot convert hidet.ir.Expr to complex.")

    # Use tvm_ffi Object's handle-based hash so dict lookups work
    # across different Python wrappers of the same underlying C handle.
    __hash__ = tvm_ffi.Object.__hash__

    def read(self, items, protected=True):
        te = self[items]
        if not isinstance(te, TensorElement):
            raise ValueError("expect element indexing, but got slicing.")
        te.protected = protected
        return te

    def write(self, items, value, protected=True):
        from tilus.hidet.ir.stmt import BufferStoreStmt

        te = self[items]
        if not isinstance(te, TensorElement):
            raise ValueError("expect element indexing, but got slicing.")
        return BufferStoreStmt(self, te.indices, value, protected)

    @staticmethod
    def _unary(cls, a):  # pylint: disable=bad-staticmethod-argument
        if not isinstance(a, Expr):
            a = convert(a)
        if isinstance(a, Constant):
            if cls is Neg:
                return constant(-a.value, a.type)
            elif cls is LogicalNot:
                return constant(not a.value, a.type)
            elif cls is BitwiseNot:
                return constant(~a.value, a.type)
            else:
                raise ValueError("unknown unary operator {}".format(cls))
        else:
            return cls(a)

    @staticmethod
    def _binary(cls, a: Expr, b: Expr):  # pylint: disable=bad-staticmethod-argument
        if not isinstance(a, Expr):
            a = convert(a)
        if not isinstance(b, Expr):
            b = convert(b)
        if isinstance(a, Constant) and isinstance(b, Constant):
            if a.type.is_data_type() and b.type.is_data_type():
                value = operator_dict[cls](a.value, b.value)
                if cls in [Equal, NotEqual, LessThan, LessEqual, LogicalAnd, LogicalOr]:
                    return constant(value, const_type="bool")
                elif cls in [LeftShift, RightShift]:
                    return constant(value, a.type)
                else:
                    return constant(value, promote_type(a.type, b.type))
            elif a.type.is_pointer() and b.type.is_pointer():
                if cls is Sub:
                    return constant(a.value - b.value, "uint64")
                elif cls in [Equal, NotEqual]:
                    return constant(operator_dict[cls](a.value, b.value), "bool")
                else:
                    raise ValueError("unknown binary operator {}".format(cls))
            elif a.type.is_pointer() and b.type.is_data_type():
                return constant(a.value + b.value, a.type)
            elif a.type.is_data_type() and b.type.is_pointer():
                return constant(a.value + b.value, b.type)
            elif a.type.is_string_type() and b.type.is_string_type():
                if cls is Add:
                    return constant(a.value + b.value, a.type)
                elif cls in [Equal, NotEqual]:
                    return constant(operator_dict[cls](a.value, b.value), "bool")
                else:
                    raise ValueError("unknown binary operator {}".format(cls))
            else:
                raise ValueError("unknown binary operator {}".format(cls))
        elif isinstance(b, Constant) and b.type.is_data_type():
            from tilus.hidet.ir.tools import infer_type

            if b == 0:
                if cls in [Add, Sub, BitwiseXor, BitwiseOr]:
                    return a
                elif cls is Multiply:
                    return promote_type(infer_type(a), b.type)(0)
            elif b == 1 and cls in [Multiply, Div]:
                return a
        elif isinstance(a, Constant):
            from tilus.hidet.ir.tools import infer_type

            if a == 0:
                if cls in [Add, BitwiseXor, BitwiseOr]:
                    return b
                elif cls is Multiply:
                    return promote_type(infer_type(b), a.type)(0)
            elif a == 1 and cls is Multiply:
                return b
        elif a is b and isinstance(a, Var) and cls == LessThan:
            return constant(False, "bool")

        return cls(a, b)


@py_class
class BinaryExpr(Expr):
    a: Any
    b: Any


@py_class
class UnaryExpr(Expr):
    a: Any


@py_class
class Condition(Expr):
    """Marker base class for condition expressions. Kept for backward compatibility."""

    pass


@py_class
class LessThan(BinaryExpr):
    __ffi_ir_traits__ = tr.BinOpTraits("$field:a", "$field:b", "<", None, None)


@py_class
class LessEqual(BinaryExpr):
    __ffi_ir_traits__ = tr.BinOpTraits("$field:a", "$field:b", "<=", None, None)


@py_class
class Equal(BinaryExpr):
    __ffi_ir_traits__ = tr.BinOpTraits("$field:a", "$field:b", "==", None, None)

    def __bool__(self):
        a, b = self.a, self.b
        # Use same_as for tvm_ffi Objects (handle-based identity) since
        # py_class fields return different Python wrappers for the same handle.
        if isinstance(a, tvm_ffi.Object) and isinstance(b, tvm_ffi.Object):
            return a.same_as(b)
        return a is b


@py_class
class NotEqual(BinaryExpr):
    __ffi_ir_traits__ = tr.BinOpTraits("$field:a", "$field:b", "!=", None, None)


@py_class
class LogicalAnd(BinaryExpr):
    __ffi_ir_traits__ = tr.BinOpTraits("$field:a", "$field:b", "and", None, None)


@py_class
class LogicalOr(BinaryExpr):
    __ffi_ir_traits__ = tr.BinOpTraits("$field:a", "$field:b", "or", None, None)


@py_class
class LogicalNot(UnaryExpr):
    __ffi_ir_traits__ = tr.UnaryOpTraits("$field:a", "not")


@py_class
class Neg(UnaryExpr):
    __ffi_ir_traits__ = tr.UnaryOpTraits("$field:a", "-")


@py_class
class Add(BinaryExpr):
    __ffi_ir_traits__ = tr.BinOpTraits("$field:a", "$field:b", "+", None, None)


@py_class
class Sub(BinaryExpr):
    __ffi_ir_traits__ = tr.BinOpTraits("$field:a", "$field:b", "-", None, None)


@py_class
class Multiply(BinaryExpr):
    __ffi_ir_traits__ = tr.BinOpTraits("$field:a", "$field:b", "*", None, None)


@py_class
class Div(BinaryExpr):
    __ffi_ir_traits__ = tr.BinOpTraits("$field:a", "$field:b", "/", None, None)


@py_class
class FloorDiv(BinaryExpr):
    __ffi_ir_traits__ = tr.BinOpTraits("$field:a", "$field:b", "//", None, None)

    def __post_init__(self):
        raise ValueError("FloorDiv is not supported in hidet by design from now on.")


@py_class
class Mod(BinaryExpr):
    __ffi_ir_traits__ = tr.BinOpTraits("$field:a", "$field:b", "%", None, None)


@py_class
class BitwiseNot(UnaryExpr):
    __ffi_ir_traits__ = tr.UnaryOpTraits("$field:a", "~")


@py_class
class BitwiseAnd(BinaryExpr):
    __ffi_ir_traits__ = tr.BinOpTraits("$field:a", "$field:b", "&", None, None)


@py_class
class BitwiseOr(BinaryExpr):
    __ffi_ir_traits__ = tr.BinOpTraits("$field:a", "$field:b", "|", None, None)


@py_class
class BitwiseXor(BinaryExpr):
    __ffi_ir_traits__ = tr.BinOpTraits("$field:a", "$field:b", "^", None, None)


@py_class
class LeftShift(BinaryExpr):
    __ffi_ir_traits__ = tr.BinOpTraits("$field:a", "$field:b", "<<", None, None)


@py_class
class RightShift(BinaryExpr):
    __ffi_ir_traits__ = tr.BinOpTraits("$field:a", "$field:b", ">>", None, None)


@py_class
class TensorElement(Expr):
    __ffi_ir_traits__ = tr.LoadTraits("$field:base", "$field:indices", None)

    base: Any
    indices: Any
    protected: bool = False


@py_class
class TensorSlice(Expr):
    base: Any
    indices: Any
    starts: Any
    ends: Any


@py_class
class Call(Expr):
    __ffi_ir_traits__ = tr.CallTraits("$field:func_var", "$field:args", None, None, None, None)

    func_var: Any
    args: Any


@py_class
class Let(Expr):
    var: Any
    value: Any
    body: Any


@py_class
class Cast(Expr):
    expr: Any
    target_type: Any

    def __ffi_text_print__(self, printer: pyast.IRPrinter, path: AccessPath):
        # Reason: cast(target_type, expr) — standard function-call syntax
        target = printer(self.target_type, path.attr("target_type"))
        expr = printer(self.expr, path.attr("expr"))
        return pyast.Call(pyast.Id("cast"), [target, expr])


@py_class
class Constant(Expr):
    value: Any
    type: Any

    # reuse commonly-used constant objects
    _constant_pool: ClassVar[Dict[Tuple[Union[int, float, bool], str], Constant]] = {}

    def is_scalar(self) -> bool:
        return isinstance(self.type, DataType)

    def is_tensor(self) -> bool:
        return isinstance(self.type, TensorType)

    def is_string(self) -> bool:
        return isinstance(self.type, StringType)

    def __ffi_text_print__(self, printer: pyast.IRPrinter, path: AccessPath):
        if self.value is None:
            tp = printer(self.type, path.attr("type"))
            return pyast.Call(pyast.Id("Constant"), [pyast.Literal(None)], ["type"], [tp])
        if self.is_tensor():
            return pyast.Call(
                pyast.Id("ConstTensor"),
                [pyast.Id(str(self.value.shape)), printer(self.type, path.attr("type"))],
            )
        elif self.is_string():
            return pyast.Literal(str(self.value))
        elif self.is_scalar():
            dtype = self.type.name
            if dtype == "float32":
                return pyast.Id("{}f".format(float(self.value)))
            elif dtype == "float16":
                return pyast.Call(pyast.Id("half"), [pyast.Literal(float(self.value))])
            elif dtype == "bfloat16":
                return pyast.Call(pyast.Id("bfloat16"), [pyast.Literal(float(self.value))])
            elif dtype == "int32":
                return pyast.Literal(int(self.value))
            elif dtype == "bool":
                return pyast.Literal(True if self.value else False)
            else:
                return pyast.Call(pyast.Id(dtype), [pyast.Literal(self.value)])
        elif isinstance(self.type, PointerType):
            tp = printer(self.type, path.attr("type"))
            val = printer(self.value, path.attr("value"))
            return pyast.Call(tp, [val])
        else:
            return pyast.Id(repr(self.value))

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __bool__(self):
        return bool(self.value)

    def __complex__(self):
        return complex(self.value)

    def array(self) -> np.ndarray:
        return self.value

    # This speciallisation of Expr._binary is done for speedup purposes only
    # and fully equvivalent to Expr._binary by functionality
    @staticmethod
    def _binary(cls, a: Expr, b: Expr):  # pylint: disable=bad-staticmethod-argument
        def _binary_internal(cls, a: int, b: int, a_type, res_type):
            value = operator_dict[cls](a, b)
            if cls in [Equal, NotEqual, LessThan, LessEqual]:
                return boolean.true if value else boolean.false
            elif cls in [LeftShift, RightShift]:
                return constant_int(value, a_type)
            else:
                return constant_int(value, res_type)

        if (
            isinstance(a, Constant)
            and isinstance(b, Constant)
            and isinstance(a.type, IntegerType)
            and isinstance(b.type, IntegerType)
        ):
            res_type = a.type if a.type is b.type else promote_type(a.type, b.type)
            return _binary_internal(cls, a.value, b.value, a.type, res_type)
        if isinstance(b, int) and isinstance(a.type, IntegerType):
            res_type = int64 if (a.type is int64 or a.type is uint64) else int32
            return _binary_internal(cls, a.value, b, a.type, res_type)
        if isinstance(a, int) and isinstance(b.type, IntegerType):
            res_type = int64 if (b.type is int64 or b.type is uint64) else int32
            return _binary_internal(cls, a, b.value, int32, res_type)
        # 2.5% cases fall here
        return super(Constant, Constant)._binary(cls, a, b)


@py_class
class IfThenElse(Expr):
    """
    The if-then-else expression.

    Parameters
    ----------
    cond: Expr
        The condition of the if-then-else expression.
    then_expr: Expr
        The expression to be evaluated if the condition is true.
    else_expr: Expr
        The expression to be evaluated if the condition is false.
    """

    cond: Any
    then_expr: Any
    else_expr: Any

    def __ffi_text_print__(self, printer: pyast.IRPrinter, path: AccessPath):
        # Reason: ternary expression `then_expr if cond else else_expr`
        # Operation IfThenElse operand order: [cond, then, else]
        cond = printer(self.cond, path.attr("cond"))
        then_expr = printer(self.then_expr, path.attr("then_expr"))
        else_expr = printer(self.else_expr, path.attr("else_expr"))
        return pyast.Operation(pyast.OperationKind.IfThenElse, [cond, then_expr, else_expr])


@py_class
class Dereference(Expr):
    __ffi_ir_traits__ = tr.UnaryOpTraits("$field:expr", "*")

    expr: Any


@py_class
class Address(Expr):
    __ffi_ir_traits__ = tr.UnaryOpTraits("$field:expr", "&")

    expr: Any


@py_class
class Reference(Expr):
    expr: Any


@py_class
class Var(Expr):
    __ffi_ir_traits__ = tr.ValueTraits("$field:hint", "$field:type", None)

    hint: Any
    type: Any
    name: Any = None
    id: int = field(default_factory=lambda: Var.new_id())

    id_clock: ClassVar[int] = 0

    def __ffi_text_print__(self, printer: pyast.IRPrinter, path: AccessPath):
        # Reason: `name` field is used by codegen for actual C variable names, so we
        # cannot mutate it in __post_init__. The display name (name ?? hint ?? "v")
        # must be a printer-only concern. `$method:` doesn't resolve Python methods.
        if not printer.var_is_defined(self):
            name = self.name if self.name else (self.hint if self.hint else "v")
            printer.var_def(name, self, None)
        return printer.var_get(self)

    @staticmethod
    def new_id():
        Var.id_clock += 1
        return Var.id_clock

    @staticmethod
    def reset_id_counter():
        Var.id_clock = 0


@py_class
class SymbolVar(Var):
    name2symbol: ClassVar[Dict[str, SymbolVar]] = {}


# the following are used as type hints
# if a primitive function expect an int8 expression, we should use ExprInt8 instead of Expr
# to explicit tell the reader that this function expect an int8 expression
# but this is not enforced by the type system of python
# ExprInt8 should be read as "expression with int8 type"
# Usage example:
#
#   def cuda_i64_to_f16(a: ExprInt64) -> ExprFloat16:
#       ...
# Above function expects an int64 expression and returns a float16 value.

ExprInt8 = ExprInt16 = ExprInt32 = ExprInt64 = Expr
ExprUInt8 = ExprUInt16 = ExprUInt32 = ExprUInt64 = Expr
ExprFloat16 = ExprFloat32 = ExprFloat64 = ExprBFloat16 = ExprTFloat32 = Expr
Int = Union[int, Expr]

"""
The following are the mapping from hidet expression class to corresponding python operator.
Used by compilation-time constant folding.
"""
operator_dict: Dict[Type[Expr], Callable] = {
    # unary arithmetic
    Neg: operator.neg,
    LogicalNot: operator.not_,
    BitwiseNot: operator.invert,
    # binary arithmetic
    Add: operator.add,
    Sub: operator.sub,
    Multiply: operator.mul,
    Div: lambda a, b: a // b if isinstance(a, int) and isinstance(b, int) else a / b,
    Mod: operator.mod,
    BitwiseOr: operator.or_,
    BitwiseAnd: operator.and_,
    BitwiseXor: operator.xor,
    Equal: operator.eq,
    NotEqual: operator.ne,
    LessThan: operator.lt,
    LessEqual: operator.le,
    LogicalAnd: lambda a, b: a and b,
    LogicalOr: lambda a, b: a or b,
    LeftShift: operator.lshift,
    RightShift: operator.rshift,
}


def convert(
    obj: Optional[Union[Expr, PyScalar, tuple, Sequence]], dtype: Optional[Union[str, DataType]] = None
) -> Optional[Union[Expr, tuple]]:
    if isinstance(obj, Expr):
        return obj

    if dtype is not None:
        if isinstance(obj, (bool, int, float)):
            return constant(obj, dtype)
        else:
            raise ValueError("Can not convert {} to {}.".format(obj, dtype))

    if isinstance(obj, bool):
        return constant(obj, data_type("bool"))
    elif isinstance(obj, int):
        return constant(obj, data_type("int32"))
    elif isinstance(obj, float):
        return constant(obj, data_type("float32"))
    elif isinstance(obj, str):
        return constant(obj, string_type())
    elif isinstance(obj, (tuple, list, tvm_ffi.Array)):
        return tuple(convert(v) for v in obj)
    elif obj is None:
        return None
    else:
        raise NotImplementedError(type(obj))


def as_expr(obj: Union[float, bool, int, str, Expr]) -> Expr:
    if isinstance(obj, Expr):
        return obj
    elif isinstance(obj, bool):
        return boolean.constant(obj)
    elif isinstance(obj, int):
        assert default_int_dtype.min_value <= obj <= default_int_dtype.max_value, obj
        return default_int_dtype.constant(obj)
    elif isinstance(obj, float):
        return default_float_dtype.constant(obj)
    elif isinstance(obj, str):
        return Constant(obj, string_type())
    else:
        raise ValueError(obj)


def var(hint: str = None, dtype="int32"):
    if isinstance(hint, str):
        assert set(hint) <= set(string.ascii_letters + "_." + string.digits)
    if isinstance(dtype, str):
        dtype = data_type(dtype)
    return Var(hint, dtype)


def scalar_var(hint: str, dtype: Union[str, DataType] = "float32") -> Var:
    dtype = dtype if isinstance(dtype, DataType) else data_type(dtype)
    return Var(hint, dtype)


def tensor_var(hint: str, shape=None, dtype: Union[str, DataType] = "float32", layout=None) -> Var:
    return Var(hint, tensor_type(dtype, shape, layout))


def is_one(v: Expr) -> bool:
    return isinstance(v, Constant) and v.value == 1


def is_zero(v: Expr) -> bool:
    return isinstance(v, Constant) and v.value == 0


def is_true(v: Union[Expr, bool]) -> bool:
    if isinstance(v, (Constant, bool)):
        return bool(v) is True
    return False


def is_false(v: Union[Expr, bool]) -> bool:
    if isinstance(v, (Constant, bool)):
        return bool(v) is False
    return False


def if_then_else(
    cond: Union[Expr, PyScalar], then_expr: Union[Expr, PyScalar], else_expr: Union[Expr, PyScalar]
) -> Expr:
    """
    Create an if-then-else expression.

    Parameters
    ----------
    cond: Expr or PyScalar
        The condition of the if-then-else expression.

    then_expr: Expr or PyScalar
        The expression to be evaluated if the condition is true.

    else_expr: Expr or PyScalar
        The expression to be evaluated if the condition is false.

    Returns
    -------
    ret: Expr
        The if-then-else expression.
    """
    cond = convert(cond)
    then_expr = convert(then_expr)
    else_expr = convert(else_expr)
    if is_constant(cond):
        if bool(cond):
            return then_expr
        else:
            return else_expr
    else:
        return IfThenElse(cond, then_expr, else_expr)


def tensor_rank(v: Expr) -> int:
    try:
        from tilus.hidet.ir.compute import TensorNode
    except ImportError:
        TensorNode = None

    if isinstance(v, Var):
        if isinstance(v.type, TensorType):
            return len(v.type.shape)
        elif isinstance(v.type, TensorPointerType):
            return len(v.type.tensor_type.shape)
        elif isinstance(v.type, PointerType):
            return 1
        elif isinstance(v.type, ArrayType):
            return 1
        else:
            raise ValueError(v)
    elif isinstance(v, TensorSlice):
        return sum(1 if i is None else 0 for i in v.indices)
    elif TensorNode is not None and isinstance(v, TensorNode):
        return len(v.type.shape)
    elif isinstance(v, Constant) and isinstance(v.type, TensorType):
        return len(v.type.shape)
    elif isinstance(v, Cast) and isinstance(v.target_type, PointerType):
        return 1
    else:
        raise ValueError('Can not infer the tensor rank of "{}"'.format(v))


def tensor_slice(
    base: Expr, indices: Sequence[Optional[Int]], starts: Sequence[Optional[Int]], ends: Sequence[Optional[Int]]
) -> TensorSlice:
    """
    Create a tensor slice expression.

    Parameters
    ----------
    base: Expr
        The base tensor expression.

    indices: Sequence[Optional[Int]]
        The indices of the tensor slice.

    starts: Sequence[Optional[Int]]
        The start indices of the tensor slice.

    ends: Sequence[Optional[Int]]
        The end indices of the tensor slice.

    Returns
    -------
    ret: TensorSlice
        The tensor slice expression.
    """
    if len(indices) != len(starts) or len(indices) != len(ends):
        raise ValueError("The length of indices, starts and ends should be the same.")

    indices = tuple(convert(i) for i in indices)
    starts = tuple(convert(i) for i in starts)
    ends = tuple(convert(i) for i in ends)
    return TensorSlice(base, indices, starts, ends)


def tensor_element(base: Expr, indices: Sequence[Int], protected=False):
    indices = tuple(convert(i) for i in indices)
    return TensorElement(base, indices, protected)


def _chain_binary_op(op: Type[BinaryExpr], operands, default):
    # pylint: disable=protected-access
    if len(operands) == 0:
        return convert(default)
    elif len(operands) == 1:
        return convert(operands[0])
    else:
        a = _chain_binary_op(op, operands[:-1], default)
        b = convert(operands[-1])
        return Expr._binary(op, a, b)


def logical_and(*args: Union[Expr, bool]) -> LogicalAnd:
    return _chain_binary_op(LogicalAnd, args, True)


def logical_or(*args: Union[Expr, bool]) -> LogicalOr:
    return _chain_binary_op(LogicalOr, args, False)


def logical_not(a: Union[Expr, PyScalar]):
    # pylint: disable=protected-access
    a = convert(a)
    return Expr._unary(LogicalNot, a)


def bitwise_not(a: Union[Expr, PyScalar]):
    a = convert(a)
    return Expr._unary(BitwiseNot, a)


def equal(a: Union[Expr, PyScalar], b: Union[Expr, PyScalar]):
    a = convert(a)
    b = convert(b)
    return a == b


def less_than(a: Union[Expr, PyScalar], b: Union[Expr, PyScalar]):
    a = convert(a)
    b = convert(b)
    return a < b


def less_equal(a: Union[Expr, PyScalar], b: Union[Expr, PyScalar]):
    a = convert(a)
    b = convert(b)
    return a <= b


def not_equal(a: Union[Expr, PyScalar], b: Union[Expr, PyScalar]):
    a = convert(a)
    b = convert(b)
    return a != b


def left_shift(a: Union[Expr, int], b: Union[Expr, int]) -> LeftShift:
    a = convert(a)
    b = convert(b)
    return a << b


def right_shift(a: Union[Expr, int], b: Union[Expr, int]) -> RightShift:
    a = convert(a)
    b = convert(b)
    return a >> b


def bitwise_and(*args: Union[Expr, int]) -> BitwiseAnd:
    return _chain_binary_op(BitwiseAnd, args, -1)


def bitwise_or(*args: Union[Expr, int]) -> BitwiseOr:
    return _chain_binary_op(BitwiseOr, args, 0)


def bitwise_xor(*args: Union[Expr, int]) -> BitwiseXor:
    return _chain_binary_op(BitwiseXor, args, 0)


def cast(v: Union[Expr, int, bool, float], dtype: Union[str, DataType, BaseType]):
    if isinstance(v, (bool, int, float)):
        v = convert(v)
    if not isinstance(v, Expr):
        raise ValueError("Expect an expression, got {}".format(type(v).__name__))

    if isinstance(dtype, str):
        dtype = data_type(dtype)

    if isinstance(v, Constant) and v.is_scalar():
        if dtype.is_vector():
            raise ValueError("Can not cast a scalar {} to a vector type {}.".format(v, dtype))
        return constant(v.value, dtype)
    elif isinstance(v, Var) and v.type.is_data_type() and dtype.is_data_type() and v.type == dtype:
        return v
    else:
        return Cast(v, dtype)


def call(func: Var, args: Sequence[Union[Expr, PyScalar]]) -> Call:
    args = tuple(convert(a) for a in args)
    return Call(func, args)


def const_tensor(value: np.ndarray) -> Constant:
    from tilus.hidet.ir.utils.type_utils import from_numpy_dtype

    return constant(value=value, const_type=tensor_type(dtype=from_numpy_dtype(value.dtype), shape=list(value.shape)))


def tensor_pointer_var(hint: str, shape=None, dtype: Union[str, DataType] = "float32", layout=None):
    return Var(hint, tensor_pointer_type(dtype=dtype, shape=shape, layout=layout))


def view(ptr: Expr, tp: TensorType) -> Expr:
    if not isinstance(tp, TensorType):
        raise ValueError("Expect a tensor type, got {}".format(type(tp).__name__))
    return cast(ptr, TensorPointerType.from_tensor_type(tp))


def address(v: Expr) -> Expr:
    return Address(v)


def deref(v: Expr, derefed_type: Optional[BaseType] = None) -> Expr:
    if derefed_type is not None:
        v = cast(v, ~derefed_type)
    return Dereference(v)


def is_constant(e: Union[Expr, PyScalar], *other: Union[Expr, PyScalar]) -> bool:
    if isinstance(e, Expr) and not isinstance(e, Constant):
        return False
    if len(other) > 0:
        return is_constant(*other)
    return True


def constant(value, const_type: Union[str, BaseType]) -> Constant:
    if const_type and isinstance(const_type, str):
        const_type = data_type(const_type)

    # normalize value
    if isinstance(const_type, DataType):
        if const_type.is_complex():
            value = complex(value)
        elif const_type.is_float():
            value = float(value)
        elif const_type.is_integer():
            if const_type == boolean:
                value = bool(value)
            else:
                value = int(value)
        elif const_type.is_vector():
            value = tuple(value)
        else:
            raise ValueError(f"Invalid data const_type {const_type}")
    elif isinstance(const_type, TensorType):
        value = np.array(value)
    elif isinstance(const_type, PointerType):
        value = int(value)
    elif isinstance(const_type, StringType):
        value = str(value)
    else:
        raise ValueError(f"Invalid const_type {const_type}")

    if const_type.is_data_type() and (
        (isinstance(value, int) and -128 <= value <= 128) or (isinstance(value, float) and value in [-1.0, 0.0, 1.0])
    ):
        # pylint: disable=protected-access
        if (value, const_type.name) not in Constant._constant_pool:
            Constant._constant_pool[(value, const_type.name)] = Constant(value, const_type)
        return Constant._constant_pool[(value, const_type.name)]
    else:
        return Constant(value, const_type)


def constant_int(value: int, const_type: IntegerType) -> Constant:
    if -128 <= value <= 128:
        # pylint: disable=protected-access
        if (value, const_type.name) not in Constant._constant_pool:
            Constant._constant_pool[(value, const_type.name)] = Constant(value, const_type)
        return Constant._constant_pool[(value, const_type.name)]
    else:
        return Constant(value, const_type)


def symbol_var(name: str, dtype: Union[DataType, PointerType, str] = "int32") -> SymbolVar:
    if isinstance(dtype, str):
        dtype = data_type(dtype)
    assert isinstance(dtype, (DataType, PointerType)), (
        "symbolic variable must be with either a data type or a pointer type"
    )
    if name not in SymbolVar.name2symbol:
        if not name.isidentifier():
            raise ValueError('Invalid symbol name "{}", must be a valid identifier'.format(name))
        SymbolVar.name2symbol[name] = SymbolVar(hint=None, type=dtype, name=name)
    else:
        if not type_equal(SymbolVar.name2symbol[name].type, dtype):
            raise ValueError(
                'SymbolVar "{}" already exists with dtype {}, new dtype is {}'.format(
                    name, SymbolVar.name2symbol[name].type, dtype
                )
            )
    return SymbolVar.name2symbol[name]


def index_vars(num_vars: int) -> List[Var]:
    """Create a list of index variables with given number of variables.

    Parameters
    ----------
    num_vars: int
        The number of index variables to create.

    Returns
    -------
    ret: List[Var]
        The list of created index variables.
    """
    default_names = ["i", "j", "k", "p", "q", "r", "s", "t", "u", "v"]
    if num_vars < len(default_names):
        return [var(default_names[i]) for i in range(num_vars)]
    else:
        return [var("i") for _ in range(num_vars)]


def reinterpret(value: Expr, target_type: BaseType) -> Expr:
    return cast(~value, ~target_type)[0]
