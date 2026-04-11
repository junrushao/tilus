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
import warnings
from dataclasses import dataclass
from typing import Any

import tvm_ffi
from tvm_ffi.dataclasses import field, py_class

from tilus.hidet.ir.type import DataType


@dataclass
class IntInfo:
    bits: int
    max: int
    min: int
    dtype: DataType


@py_class
class _IntBox(tvm_ffi.Object):
    """Wraps an int as two halves so the FFI can serialize values outside int64 range (e.g. int64 min/max).

    The value is split into ``hi`` (upper 32 bits, signed) and ``lo`` (lower 32 bits, unsigned)
    so both halves fit in a signed int64 FFI field without overflow.
    """

    hi: Any  # upper 32 bits (signed)
    lo: Any  # lower 32 bits (unsigned, 0..2^32-1)

    @staticmethod
    def create(v: int) -> "_IntBox":
        lo = v & 0xFFFFFFFF
        hi = v >> 32
        return _IntBox(hi=hi, lo=lo)

    @property
    def value(self) -> int:
        return (int(self.hi) << 32) | int(self.lo)

    def __le__(self, other):
        return self.value <= other

    def __ge__(self, other):
        return self.value >= other

    def __lt__(self, other):
        return self.value < other

    def __gt__(self, other):
        return self.value > other

    def __eq__(self, other):
        if isinstance(other, _IntBox):
            return self.value == other.value
        return self.value == other

    def __hash__(self):
        return hash(self.value)

    def __int__(self):
        return self.value

    def __repr__(self):
        return repr(self.value)


@py_class
class IntegerType(DataType):
    _min_value: Any
    _max_value: Any

    def is_float(self) -> bool:
        return False

    def is_integer(self) -> bool:
        return True

    def is_complex(self) -> bool:
        return False

    def is_vector(self) -> bool:
        return False

    def is_boolean(self) -> bool:
        return False

    def constant(self, value: Any):
        from tilus.hidet.ir.expr import Constant, constant

        if isinstance(value, Constant):
            value = value.value
        if isinstance(value, _IntBox):
            value = value.value
        if isinstance(value, float):
            warnings.warn("Converting float to integer when creating {} constant: {}.".format(self.name, value))
        value = int(value)

        if not self._min_value <= value <= self._max_value:
            raise ValueError("Value {} is out of range for {}.".format(value, self.name))
        return constant(value, self)

    def signedness(self):
        return self._min_value < 0

    @property
    def one(self):
        return self.constant(1)

    @property
    def zero(self):
        return self.constant(0)

    @property
    def min_value(self):
        return self.constant(self._min_value)

    @property
    def max_value(self):
        return self.constant(self._max_value)

    def iinfo(self) -> IntInfo:
        return IntInfo(self.nbytes * 8, int(self._max_value), int(self._min_value), self)


def _make_integer_type(name: str, short_name: str, nbytes: int, min_value: int, max_value: int) -> IntegerType:
    return IntegerType(
        _name=name,
        _short_name=short_name,
        _nbytes=nbytes,
        _min_value=_IntBox.create(min_value),
        _max_value=_IntBox.create(max_value),
    )


int8 = _make_integer_type("int8", "i8", 1, -128, 127)
int16 = _make_integer_type("int16", "i16", 2, -32768, 32767)
int32 = _make_integer_type("int32", "i32", 4, -2147483648, 2147483647)
int64 = _make_integer_type("int64", "i64", 8, -9223372036854775808, 9223372036854775807)

uint8 = _make_integer_type("uint8", "u8", 1, 0, 255)
uint16 = _make_integer_type("uint16", "u16", 2, 0, 65535)
uint32 = _make_integer_type("uint32", "u32", 4, 0, 4294967295)
uint64 = _make_integer_type("uint64", "u64", 8, 0, 18446744073709551615)

i8 = int8
i16 = int16
i32 = int32
i64 = int64

u8 = uint8
u16 = uint16
u32 = uint32
u64 = uint64
