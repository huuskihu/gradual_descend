# Bug report: PyCharm Quick Documentation shows wrong element for np.bincount

**Title (for the issue tracker):**

Quick Documentation for np.bincount shows `__all__` declaration from numpy/\_\_init\_\_.pyi instead of the function's docstring (numpy 2.x)

---

## Environment

- **IDE**: PyCharm Professional, build PY-262.10968.92 (exact version shown in Help > About)
- **OS**: Windows 11, build 10.0.26200
- **Python**: 3.14.7 (`D:\Program Files\python\python.exe`, system interpreter, not a virtualenv)
- **NumPy**: 2.5.3
- **Project type**: plain Python project; `src/` marked as Sources Root

## Steps to reproduce

1. Create any Python file with these two lines:

   ```python
   import numpy as np
   np.bincount([0, 1, 1, 2])
   ```

2. Place the caret on `bincount` and press `Ctrl+Q` (or simply hover over it).

## Expected result

The popup shows the real docstring of `numpy.bincount`, for example:

    bincount(x, /, weights=None, minlength=0)
    Count number of occurrences of each value in array of non-negative ints.
    ...

## Actual result

The popup instead shows the module-level `__all__` declaration from the stub file
`numpy/__init__.pyi`:

    D:\Program Files\python\Lib\site-packages\numpy\__init__.pyi
    __all__: Any = [...]

(Screenshot attached.)

## Additional observations

1. **Only a few names are affected.** `np.bincount`, `np.array` and `np.i0` behave this way.
   Most other numpy names (for example `np.absolute`, `np.add`, `np.acos`, `np.int8`)
   show a correct documentation popup.

2. **The completion popup shows the same pattern.** After typing `np.`, the affected names
   (`bincount`, `array`, `i0`) have *no symbol icon at all*, while unaffected names show the
   expected icon — e.g. `absolute`, which is declared as
   `absolute: _UFunc_Nin1_Nout1[...]` at `numpy/__init__.pyi:7551`.

3. **Navigation works correctly.** `Ctrl+B` on `np.bincount` jumps to
   `numpy/_core/multiarray.pyi:811`. So the symbol *is* resolvable — only the
   Quick Documentation popup renders the wrong element.

4. **Likely mechanism.** In `numpy/__init__.pyi`, `bincount` and `array` appear only inside
   the re-export import list (lines 302 and 315):

   ```python
   from numpy._core.multiarray import (
       array,
       ...
       bincount,
       ...
   )
   ```

   while the actual `def bincount(...)` lives in `numpy/_core/multiarray.pyi:811`.
   PyCharm appears to fail to follow this re-export chain when rendering Quick Documentation,
   and falls back to the nearest declaration in `__init__.pyi` — the `__all__: Any = [...]`
   assignment.

## Already tried (no effect)

- `File > Invalidate Caches... > Invalidate and Restart`, then waited for full re-indexing.
- Verified the project interpreter is the same interpreter where numpy is installed.
- Verified `src/` is marked as Sources Root and the `.iml` references the correct SDK.
- Verified there are no custom file-type mappings (`filetypes.xml` contains only one unrelated
  entry), so the file is definitely recognized as Python: syntax highlighting, code completion
  and Go to Definition all work normally.

## Comparison

In VS Code with Pylance, hovering `np.bincount` renders the correct documentation, and
Go to Definition resolves to the same stub file. This suggests the problem is specific to
PyCharm's handling of stub re-exports rather than to the numpy installation.

## Note

This is **not** PY-48166 ("numpy 1.20.x stubs are breaking PyCharm code insight"), which is
Fixed as of PyCharm 2021.1.1. That issue affected nearly all top-level numpy names; here, with
numpy 2.5.3, only a small set of re-exported C functions is affected.

## Attachments

- Screenshot 1: hover popup showing `__all__: Any = [...]`
- Screenshot 2: completion popup where `bincount` / `array` / `i0` have no symbol icon

---

## Where to submit

- Easiest: in PyCharm, `Help > Submit Feedback...` (automatically attaches environment info)
- Or on the web: `https://youtrack.jetbrains.com/newIssue?project=PY` (free JetBrains account needed)
