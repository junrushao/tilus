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
from typing import Any

import tvm_ffi
from tvm_ffi.dataclasses import py_class

from tilus.hidet.ir.type import DataType

from .integer import IntegerType, IntInfo, uint8, uint32


@py_class
class IntegerSubbyteType(IntegerType):
    _storage: Any  # DataType
    _nbits: int
    _bits_mask: int
    _sign_mask: int

    @property
    def storage(self):
        return self._storage

    @property
    def nbytes(self):
        raise TypeError(f"Cannot access nbytes property for the type({self}")

    @property
    def nbits(self):
        return self._nbits

    @property
    def bits_mask(self):
        return self._bits_mask

    @property
    def sign_mask(self):
        return self._sign_mask

    def iinfo(self) -> IntInfo:
        return IntInfo(self._nbits, self._max_value, self._min_value, self)


def _make_integer_subbyte_type(
    name: str, short_name: str, storage: DataType, nbits: int, signed: bool, min_value: int, max_value: int
) -> IntegerSubbyteType:
    nbytes = storage.nbytes
    bits_mask = (1 << nbits) - 1
    sign_mask = (1 << (nbits - 1)) if (min_value < 0) else 0
    return IntegerSubbyteType(
        _name=name,
        _short_name=short_name,
        _nbytes=nbytes,
        _min_value=min_value,
        _max_value=max_value,
        _storage=storage,
        _nbits=nbits,
        _bits_mask=bits_mask,
        _sign_mask=sign_mask,
    )


int4b = _make_integer_subbyte_type("int4b", "i4", uint8, 4, True, -8, 7)
int3b = _make_integer_subbyte_type("int3b", "i3", uint32, 3, True, -4, 3)
int2b = _make_integer_subbyte_type("int2b", "i2", uint8, 2, True, -2, 1)
int1b = _make_integer_subbyte_type("int1b", "i1", uint8, 1, True, -1, 0)

uint4b = _make_integer_subbyte_type("uint4b", "u4", uint8, 4, False, 0, 16)
uint3b = _make_integer_subbyte_type("uint3b", "u3", uint32, 3, False, 0, 8)
uint2b = _make_integer_subbyte_type("uint2b", "u2", uint8, 2, False, 0, 4)
uint1b = _make_integer_subbyte_type("uint1b", "u1", uint8, 1, False, 0, 1)

i4 = int4b
i3 = int3b
i2 = int2b
i1 = int1b

u4 = uint4b
u3 = uint3b
u2 = uint2b
u1 = uint1b

int7b = _make_integer_subbyte_type("int7b", "i7", uint32, 7, True, -64, 63)
int6b = _make_integer_subbyte_type("int6b", "i6", uint32, 6, True, -32, 31)
int5b = _make_integer_subbyte_type("int5b", "i5", uint32, 5, True, -16, 15)

uint7b = _make_integer_subbyte_type("uint7b", "u7", uint32, 7, False, 0, 127)
uint6b = _make_integer_subbyte_type("uint6b", "u6", uint32, 6, False, 0, 63)
uint5b = _make_integer_subbyte_type("uint5b", "u5", uint32, 5, False, 0, 31)

i7 = int7b
i6 = int6b
i5 = int5b

u7 = uint7b
u6 = uint6b
u5 = uint5b
