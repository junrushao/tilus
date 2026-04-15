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
# pylint: disable=import-outside-toplevel
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import tvm_ffi
from tvm_ffi import ir_traits as tr
from tvm_ffi import pyast
from tvm_ffi.access_path import AccessPath
from tvm_ffi.dataclasses import py_class

from tilus.hidet.ir.node import Node

# typing forward declaration
Expr = "Expr"
Int = Union[int, Expr]


@py_class
class BaseType(Node):
    def __invert__(self) -> BaseType:
        # get the pointer type that points to current type
        if isinstance(self, TensorType):
            return TensorPointerType.from_tensor_type(self)
        elif isinstance(self, DataType):
            return PointerType(base_type=self)
        elif isinstance(self, (PointerType, TensorPointerType)):
            return PointerType(base_type=self)
        else:
            raise ValueError("Can not recognize type {}".format(self))

    def __getitem__(self, item):
        if isinstance(item, (tuple, list, tvm_ffi.Array)):
            if len(item) == 1:
                item = item[0]
            else:
                raise ValueError("Currently, only support 1-d array, but got {}".format(item))
        return array_type(self, int(item))

    def is_void(self):
        return isinstance(self, VoidType)

    def is_tensor(self):
        return isinstance(self, TensorType)

    def is_pointer(self):
        return isinstance(self, (PointerType, TensorPointerType))

    def is_data_type(self):
        return isinstance(self, DataType)

    def is_func_type(self):
        return isinstance(self, FuncType)

    def is_string_type(self):
        return isinstance(self, StringType)

    def as_data_type(self) -> Optional[DataType]:
        if not isinstance(self, DataType):
            return None
        return self


@py_class
class DataType(BaseType):
    """
    The data type that defines how to interpret the data in memory.

    """

    _name: str
    _short_name: str
    _nbytes: int

    __ffi_ir_traits__ = tr.PrimTyTraits("$field:_name")

    def __str__(self):
        return "hidet.{}".format(self.name)

    def __eq__(self, other):
        return isinstance(other, DataType) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __call__(self, value: Any):
        """
        Create a constant of current data type, or convert an existing Expr to current data type with cast expression.

        Parameters
        ----------
        value: Union[int, float, bool, list, tuple, Constant, Expr]
            The value of the constant or the value to be casted.

        Returns
        -------
        ret: Constant or Cast
            The constant or cast expression.
        """
        from tilus.hidet.ir import expr

        built_types = (int, float, bool, complex)

        if (
            isinstance(value, built_types)
            or isinstance(value, (list, tuple, tvm_ffi.Array))
            and all(isinstance(v, built_types) for v in value)
        ):
            return self.constant(value)
        elif isinstance(value, expr.Constant):
            return self.constant(value.value)
        elif isinstance(value, expr.Expr):
            return expr.cast(value, self)
        else:
            raise ValueError("Can not convert {} to {}".format(value, self))

    def __getitem__(self, item):
        if not isinstance(item, (tuple, list, tvm_ffi.Array)):
            item = (item,)
        return tensor_type(dtype=self, shape=list(item))

    @property
    def name(self) -> str:
        return self._name

    @property
    def short_name(self) -> str:
        return self._short_name

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def nbits(self) -> int:
        """
        Get the bit length of the data type

        Note:
        1. The bit length of the data type itself other than the bit length of its storage.
        2. For regular data types, the nbits can be computed from its nbytes property.
        3. For subbyte data types, the nbits is defined when constructing the data type,
        and this method will also be overridden for subbyte data types.
        4. In addition, we cannot access the nbytes for a subbyte data type, otherwise
        a type error will be raised.
        """
        return self._nbytes * 8

    @property
    def storage(self) -> DataType:
        """
        Get the actual storage type of the data type

        Note:
        1. The storage of a regular data type is the data type itself, while the storage
        of a subbyte type is the type of its actual storage. e.g., the storage of int4b is uint8
        2. The property will be overridden in the subclass of subbyte types.
        """
        return self

    def is_integer_subbyte(self) -> bool:
        return self.is_integer() and self.is_subbyte()

    def is_float_subbyte(self) -> bool:
        return self.is_float() and self.is_subbyte()

    def is_subbyte(self):
        return self.nbits < 8

    def is_any_float16(self) -> bool:
        return self.is_float() and self.nbits == 16

    def is_float(self) -> bool:
        raise NotImplementedError()

    def is_integer(self) -> bool:
        raise NotImplementedError()

    def is_complex(self) -> bool:
        raise NotImplementedError()

    def is_vector(self) -> bool:
        raise NotImplementedError()

    def is_boolean(self) -> bool:
        raise NotImplementedError()

    def constant(self, value: Any):
        raise NotImplementedError()

    @property
    def one(self):
        raise NotImplementedError()

    @property
    def zero(self):
        raise NotImplementedError()

    @property
    def min_value(self):
        raise NotImplementedError()

    @property
    def max_value(self):
        raise NotImplementedError()


@py_class
class TensorType(BaseType):
    """
    A tensor type.

    Parameters
    ----------
    dtype: DataType
        The data type of the tensor.
    shape: Tuple[Expr, ...]
        The shape of the tensor.
    layout: hidet.ir.layout.DataLayout
        The layout of the tensor.
    """

    dtype: Any = None
    shape: Any = None
    layout: Any = None

    __ffi_ir_traits__ = tr.TensorTyTraits("$field:shape", "$field:dtype", None)

    def __invert__(self):
        return TensorPointerType.from_tensor_type(self)

    def storage_bytes(self) -> Expr:
        if self.dtype.is_integer_subbyte():
            return self.layout.size * self.dtype.nbits // 8
        else:
            return self.layout.size * self.dtype.nbytes

    def const_shape(self) -> List[int]:
        return [int(v) for v in self.shape]


@py_class
class VoidType(BaseType):
    __ffi_ir_traits__ = tr.PrimTyTraits("void")


@py_class
class StringType(BaseType):
    __ffi_ir_traits__ = tr.PrimTyTraits("char*")


@py_class
class PointerType(BaseType):
    base_type: Any
    specifiers: Any = None
    use_bracket: bool = False

    def __post_init__(self):
        if isinstance(self.base_type, str):
            self.base_type = data_type(self.base_type)
        # todo: move the following attributes to DeclareStmt
        self.specifiers = list(self.specifiers) if self.specifiers else []

    def __ffi_text_print__(self, printer: pyast.IRPrinter, path: AccessPath):
        # Reason: pointer types print as just the base type for readability
        return printer(self.base_type, path.attr("base_type"))

    def __call__(self, x):
        from tilus.hidet.ir.expr import Constant, Expr, cast, constant  # pylint: disable=redefined-outer-name

        if isinstance(x, int):
            return constant(x, self)
        elif isinstance(x, Constant):
            return constant(x.value, self)
        elif isinstance(x, Expr):
            return cast(x, self)
        else:
            raise ValueError("Can not convert {} to {}".format(x, self))


@py_class
class ReferenceType(BaseType):
    base_type: BaseType


@py_class
class TensorPointerType(BaseType):
    """
    A pointer type that points to tensor.
    """

    tensor_type: TensorType

    def __ffi_text_print__(self, printer: pyast.IRPrinter, path: AccessPath):
        # Reason: tensor pointer types print as just the tensor type for readability
        return printer(self.tensor_type, path.attr("tensor_type"))

    @staticmethod
    def from_tensor_type(tp: TensorType) -> TensorPointerType:
        return TensorPointerType(tensor_type=tp)


@py_class
class ArrayType(BaseType):
    base_type: BaseType
    size: int

    def __post_init__(self):
        assert isinstance(self.base_type, BaseType) and not isinstance(self.base_type, (ArrayType, TensorType))
        assert isinstance(self.size, int) and self.size >= 0


TypeLike = Union[str, BaseType]


@py_class
class FuncType(BaseType):
    param_types: Any = None
    ret_type: Any = None
    type_infer_func: Any = None  # string name registered via tvm_ffi.register_global_func

    def __post_init__(self):
        if self.param_types is not None:
            self.param_types = [self._convert_type(tp) for tp in self.param_types]
        if self.ret_type is not None:
            self.ret_type = self._convert_type(self.ret_type)
        if self.type_infer_func is not None and not isinstance(self.type_infer_func, str):
            raise TypeError(
                f"type_infer_func must be a string name registered via tvm_ffi.register_global_func(), "
                f"got {type(self.type_infer_func).__name__}"
            )
        msg = "Please provide either a static type or a type infer func"
        assert not all(v is None for v in [self.ret_type, self.type_infer_func]), msg

    def ret_type_on(self, arg_types: List[BaseType]) -> BaseType:
        if self.ret_type is not None:
            assert isinstance(self.ret_type, BaseType)
            return self.ret_type
        else:
            return tvm_ffi.get_global_func(self.type_infer_func)(arg_types)

    def _convert_type(self, tp: Union[str, BaseType]):
        if isinstance(tp, str):
            return data_type(tp)
        else:
            return tp

    @staticmethod
    def from_func(func):
        return FuncType(param_types=[param.type for param in func.params], ret_type=func.ret_type)


@py_class
class OpaqueType(BaseType):
    cpp_name: str
    modifiers: Any = ()


def tensor_type(dtype, shape: Optional[Sequence[Union[int, Expr]]] = None, layout=None):
    """
    Construct a tensor type.

    One of shape and layout must be given.

    Parameters
    ----------
    dtype: str or DataType
        The scalar type of this tensor.

    shape: Sequence[Union[int, Expr]] or none
        The shape of the tensor. If not given, the shape in layout will be used.

    layout: hidet.ir.layout.DataLayout or none
        The layout of the tensor. If not given, the row major layout of given shape will
        be used.

    Returns
    -------
    ret: TensorType
        The constructed tensor type
    """
    from tilus.hidet.ir.expr import convert
    from tilus.hidet.ir.layout import DataLayout, row_major
    from tilus.hidet.ir.tools import simplify

    if isinstance(dtype, str):
        dtype = data_type(dtype)
    if not isinstance(dtype, DataType):
        raise ValueError('Scalar type expect a "str" or "ScalarType", but got {}'.format(type(dtype)))
    if shape is None and layout is None:
        raise ValueError("Tensor type must give either shape or layout")
    elif shape is None:
        assert isinstance(layout, DataLayout)
        shape = layout.shape
    elif layout is None:
        layout = row_major(*shape)
        if not all(isinstance(s, int) for s in shape):
            layout = simplify(layout, enable_rules=True)
    else:
        assert isinstance(layout, DataLayout)
        assert isinstance(shape, (list, tuple, tvm_ffi.Array))
        assert len(shape) == len(layout.shape)
    shape = convert(shape)
    return TensorType(dtype, shape, layout)


def array_type(base_type: BaseType, size: int):
    return ArrayType(base_type, size)


def pointer_type(base_type):
    return PointerType(base_type)


def tensor_pointer_type(dtype, shape=None, layout=None):
    return TensorPointerType(tensor_type(dtype, shape, layout))


def string_type():
    return StringType()


def func_type(param_types, ret_type) -> FuncType:
    return FuncType(param_types, ret_type)


def data_type(dtype: Union[str, DataType]) -> DataType:
    from tilus.hidet.ir.dtypes import name2dtype, sname2dtype

    if isinstance(dtype, DataType):
        return dtype
    elif isinstance(dtype, str):
        if dtype in name2dtype:
            return name2dtype[dtype]
        elif dtype in sname2dtype:
            return sname2dtype[dtype]
        else:
            raise ValueError("Unknown data type: {}, candidates:\n{}".format(dtype, "\n".join(name2dtype.keys())))
    else:
        raise ValueError("Expect a string or a DataType, but got {}".format(type(dtype)))


def type_equal(lhs: BaseType, rhs: BaseType) -> bool:
    """
    Check whether the two types are equal or not.

    Parameters
    ----------
    lhs: BaseType
        The first type to compare.
    rhs: BaseType
        The second type to compare.

    Returns
    -------
    ret: bool
        Whether the two types are equal or not.
    """
    if type(lhs) is not type(rhs):
        return False
    if isinstance(lhs, DataType) and isinstance(rhs, DataType):
        return lhs.name == rhs.name
    elif isinstance(lhs, PointerType) and isinstance(rhs, PointerType):
        return type_equal(lhs.base_type, rhs.base_type)
    elif isinstance(lhs, VoidType) and isinstance(rhs, VoidType):
        return True
    elif isinstance(lhs, TensorPointerType) and isinstance(rhs, TensorPointerType):
        return type_equal(lhs.tensor_type, rhs.tensor_type)
    elif isinstance(lhs, TensorType) and isinstance(rhs, TensorType):
        from tilus.hidet.ir.expr import is_constant

        if not type_equal(lhs.dtype, rhs.dtype):
            return False
        if len(lhs.shape) != len(rhs.shape):
            return False
        for a, b in zip(lhs.shape, rhs.shape):
            if is_constant(a) ^ is_constant(b):
                return False
            elif is_constant(a) and is_constant(b):
                if int(a) != int(b):
                    return False
            else:
                # we do not have equivalence checking for symbolic expression
                pass
        # do not check layout
        return True
    elif isinstance(lhs, FuncType) and isinstance(rhs, FuncType):
        assert lhs.param_types is not None and lhs.ret_type is not None
        assert rhs.param_types is not None and rhs.ret_type is not None
        if len(lhs.param_types) != len(rhs.param_types):
            return False
        if not type_equal(lhs.ret_type, rhs.ret_type):
            return False
        for a, b in zip(lhs.param_types, rhs.param_types):
            if not type_equal(a, b):
                return False
        return True
    elif isinstance(lhs, ReferenceType) and isinstance(rhs, ReferenceType):
        return type_equal(lhs.base_type, rhs.base_type)
    elif isinstance(lhs, OpaqueType) and isinstance(rhs, OpaqueType):
        return lhs.cpp_name == rhs.cpp_name
    else:
        raise NotImplementedError("type_equal not implemented for {} and {}".format(type(lhs), type(rhs)))


def sizeof(tp: BaseType) -> int:
    """
    Get the size of the given type in bytes.

    Parameters
    ----------
    tp: BaseType
        The type to get the size.

    Returns
    -------
    ret: int
        The size of the type in bytes.
    """
    from tilus.hidet.utils import prod

    if isinstance(tp, DataType):
        return tp.nbytes
    elif isinstance(tp, (PointerType, TensorPointerType)):
        # we assume we work on 64-bit system
        return 8
    elif isinstance(tp, TensorType):
        return sizeof(tp.dtype) * prod(tp.shape)
    else:
        raise NotImplementedError(type(tp))


void_p = PointerType(base_type=VoidType())
byte_p = PointerType(base_type=data_type("uint8"))
void = VoidType()


def is_addressable(tp_or_var):
    from tilus.hidet.ir.expr import Var

    if isinstance(tp_or_var, Var):
        tp = tp_or_var.type
    else:
        tp = tp_or_var
    return isinstance(tp, (PointerType, TensorPointerType, TensorType))


def get_base_type(tp: BaseType) -> BaseType:
    if isinstance(tp, PointerType):
        return tp.base_type
    elif isinstance(tp, TensorPointerType):
        return tp.tensor_type.dtype
    elif isinstance(tp, TensorType):
        return tp.dtype
    else:
        assert False
