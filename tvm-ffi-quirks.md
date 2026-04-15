# TVM-FFI Printer System: Design Quirks

| # | Quirk | Severity | Impact on Tilus migration |
|---|-------|----------|--------------------------|
| **1** | `__ffi_text_print__` can't return multiple statements | **High** | `LetStmt` loses bindings, multi-function modules print only first function |
| **2** | Tier 3 default can't handle Statement-valued fields | **High** | Forces `__ffi_text_print__` on every node with a `body: Stmt` field (LetStmt, ThreadGroupStmt, etc.) -- the #1 driver of escape-hatch usage |
| **3** | `pyast.Call` requires all 4 positional args | Low | Every call site needs trailing `[], []` for empty kwargs |
| **4** | No trait for module/program containers | Medium | No way to declaratively print `dict[str, Function]` -- forces `__ffi_text_print__` boilerplate |
| **5** | `$field:` can't compose or transform values | **High** | Any field needing even `removesuffix("Inst")` or a conditional requires escape-hatch. This is why `Instruction` (80+ subclasses) needs `__ffi_text_print__` |
| **6** | `WithTraits(no_frame=True)` is the only inline sequence mechanism | Low | SeqStmt uses "with" semantics for a plain sequence -- confusing to read |
| **7** | Inherited field resolution undocumented | Low | Works correctly, just surprising that `$field:a` on `Add` finds `BinaryExpr.a` |
| **8** | No Expr trait for `*`/`&` operators | Low | `Dereference`/`Address` fall to verbose Tier 3 instead of printing `*x`/`&x` |

## 1. `__ffi_text_print__` cannot return multiple statements

**Problem**: `__ffi_text_print__` must return a single `pyast.Node`. When an IR node
(e.g., `LetStmt` with multiple bindings + body) naturally maps to multiple statements,
there is no way to return them. Returning a Python list gets converted to `ffi.Array`
and causes `TypeError: Cannot convert from type 'ffi.Array' to 'ffi.pyast.Node'`.

**Impact**: `LetStmt` loses binding information in text output because only the body
can be returned, not the bindings + body. `IRModule`/`Program` with multiple functions
can only print the first function.

**Suggested fix**: Allow `__ffi_text_print__` to return a `list[pyast.Node]`. The printer
should emit each element sequentially into the current statement scope. Alternatively,
expose `pyast.StmtBlock(stmts: list[Stmt])` as a Python-constructible node.

## 2. Tier 3 default printing can't handle Statement-valued fields

**Problem**: Tier 3 default renders `TypeKey(field1=val1, field2=val2, ...)` by printing
each field value as an expression. When a field value is a Statement node (e.g.,
`body: SeqStmt`), the printer produces a `StmtBlock` and tries to coerce it to `Expr`:
`TypeError: Cannot convert from type 'ffi.pyast.StmtBlock' to 'ffi.pyast.Expr'`.

**Impact**: ANY node with a `body: Stmt` field **must** have `__ffi_ir_traits__` or
`__ffi_text_print__` — it cannot fall through to Tier 3. This forces `__ffi_text_print__`
on nodes that would otherwise be fine with default printing (e.g., `ThreadGroupStmt`,
`LetStmt`). It's the single biggest reason for escape-hatch proliferation.

**Suggested fix**: In Tier 3 default, detect statement-level AST results and render them
as inline blocks (e.g., `body={...}` or indented sub-block) instead of trying Expr coercion.

## 3. `pyast.Call` requires all 4 positional arguments

**Problem**: `pyast.Call(callee, args, kwargs_keys, kwargs_values)` makes `kwargs_keys`
and `kwargs_values` mandatory. Every call site must pass `[], []` for empty kwargs.

**Impact**: Every `pyast.Call(...)` in `__ffi_text_print__` methods needs the trailing
`[], []`. Easy to forget — the error message (`missing required argument: 'kwargs_keys'`)
doesn't hint that `[]` is the fix.

**Suggested fix**: Make kwargs optional with default `[]`.

## 4. No trait for "container of functions" (module/program)

**Problem**: `FuncTraits` handles a single function. There's no trait for a module/class
that contains multiple functions. `IRModule` and `Program` hold `dict[str, Function]`
but no trait accepts a dict-valued body.

**Impact**: Combined with quirk #1, multi-function containers can only print their first
function. We need `__ffi_text_print__` just for dict iteration, which is boilerplate.

**Suggested fix**: Add a `ModuleTraits` (or extend `FuncTraits`) that accepts a dict/list
body and iterates over it, rendering each child as a top-level statement.

## 5. `$field:` references can't compose or transform values

**Problem**: Trait field references (`$field:name`, `$method:name`) are purely lookup-based.
There's no way to do simple transformations like:
- Strip a suffix: `type(self).__name__.removesuffix("Inst")`
- Conditional: `"elect_any" if self.x == -1 else str(self.x)`
- Combine fields: `[self.target_type, self.expr]` as a single args list

**Impact**: Any node needing even trivial field transformation requires `__ffi_text_print__`.
For example, `Instruction` needs escape-hatch solely because it strips `"Inst"` from the
class name and iterates a dynamic `attributes` dict.

**Suggested fix**: Support `$expr:` references or allow traits to accept small lambdas
for value transformation. Even just `$field:name:removesuffix("Inst")` would help.

## 6. `WithTraits(no_frame=True)` is the only way to inline a statement sequence

**Problem**: `SeqStmt` (a tuple of statements that should be emitted inline) has no
natural trait. The workaround is `WithTraits(no_frame=True)`, which is semantically
"a with-block without a frame" — not intuitive for "just expand this list of statements".

**Impact**: Every `SeqStmt`-like node uses `WithTraits` with a non-obvious `no_frame=True`
flag. Reading the code, it's unclear why a sequence is printed via "with" traits.

**Suggested fix**: Add a dedicated `SeqTraits` or `InlineTraits` for statement sequences.

## 7. Trait printing of inherited fields requires knowledge of the reflection system

**Problem**: When `Add` inherits `a, b` from `BinaryExpr`, using `$field:a` on `Add`
works because the C++ `FindField` walks the type hierarchy. But this isn't documented
and requires understanding TVM-FFI's reflection internals.

**Impact**: Not a bug, but a documentation gap. Users writing traits for classes with
inherited fields may be unsure whether `$field:` works across the hierarchy.

**Suggested fix**: Document that `$field:` resolves through the full type hierarchy.

## 8. No Expr-level trait for pointer/address operations

**Problem**: `UnaryOpTraits` supports `-`, `~`, `not`, `+` but not `*` (dereference)
or `&` (address-of). These are common in systems-level IR.

**Impact**: `Dereference` and `Address` nodes fall through to Tier 3 default, producing
verbose output like `Dereference(expr=x)` instead of `*x`.

**Suggested fix**: Extend `UnaryOpTraits` to accept `"*"` and `"&"` as operator strings,
or add a `PrefixOpTraits` for arbitrary prefix operators.
