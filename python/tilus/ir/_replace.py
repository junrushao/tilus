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


def _get_field_names(cls: type) -> list[str]:
    """Get all py_class field names from the class hierarchy, in MRO order."""
    names: list[str] = []
    seen: set[str] = set()
    for klass in cls.__mro__:
        if klass is object:
            break
        for name, anno in getattr(klass, "__annotations__", {}).items():
            if name in seen:
                continue
            seen.add(name)
            if isinstance(anno, str) and "ClassVar" in anno:
                continue
            names.append(name)
    return names


def replace(obj: Any, **changes: Any) -> Any:
    """Create a copy of a py_class object with some fields replaced.

    This is a drop-in replacement for ``dataclasses.replace()`` that works
    with ``@tvm_ffi.dataclasses.py_class`` decorated classes.
    """
    field_values = {name: getattr(obj, name) for name in _get_field_names(type(obj))}
    field_values.update(changes)
    return type(obj)(**field_values)
