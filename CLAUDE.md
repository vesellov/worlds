# Python code style

Every Python file written or edited (modules, tools, one-off scripts) is laid out in this order:

1. shebang and license comments, if any
2. module docstring
3. `from __future__ import ...`, if any (Python requires it right after the docstring)
4. `_Debug = False` (keep the existing value in files which already have it)
5. stdlib imports, then a blank line, then imports of third-party packages from the venv
6. imports of the project's own modules: the base/shared package first (for example `from lib import ...`),
   then the packages built on top of it (for example `from client ...`, `from server ...`)
7. module constants (UPPER_CASE)
8. the rest of the code

## Naming

- A variable, argument or attribute which holds a path to a file or folder on the disk ends with `_path`:
  `file_path`, `dest_dir_path`, `heightmap_file_path`, `source_paths` for a list of paths.
- A variable which holds only the name of a file or folder, not a path, ends with `_name`: `file_name`, `archive_name`.
- When one value starts as a name and later becomes a path, two variables are used: `mod_file_name` and `mod_file_path`.
- Command line options keep their flags, the attribute gets the suffix through `dest=` (and `metavar=` for the help).

## Imports

- When code needs a built-in Python module, its import always goes into the header of the file,
  never into the body of a function.
- No `sys.path` manipulation in modules. Code is organized in packages with `__init__.py` where needed,
  and modules are imported the classic way: `from lib import jsn`, `from client import loader`.
  Entry points are run with the source folder on the path, for example `PYTHONPATH=src python3 -m server.main`.

Sections are split with this separator line (`#` and 78 `-`):

```python
"""
Module description.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import os
import sys

import requests

#------------------------------------------------------------------------------

from lib import res
from client import loader

#------------------------------------------------------------------------------

TIMEOUT = 10

#------------------------------------------------------------------------------


class First(object):
    pass

#------------------------------------------------------------------------------


class Second(object):
    pass

#------------------------------------------------------------------------------


def helper():
    pass
```

One blank line around a separator, two blank lines between it and a following `def` or `class`.
The same separator goes between top-level classes and the code around them.
Sections which are empty are left out together with their separator.
Vendored third-party code copied into a project is left as it is.
