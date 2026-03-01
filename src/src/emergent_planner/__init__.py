"""Compatibility shim for deprecated `src.emergent_planner` imports."""
from __future__ import annotations

import warnings

import emergent_planner as _new_pkg

warnings.warn(
    "`src.emergent_planner` is deprecated and will be removed in the next minor release; "
    "use `emergent_planner` instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Make legacy submodule imports (for example `src.emergent_planner.config`) resolve
# against the modern package path without duplicating files.
__path__ = list(getattr(_new_pkg, "__path__", []))

__all__ = list(getattr(_new_pkg, "__all__", []))
for _name in __all__:
    globals()[_name] = getattr(_new_pkg, _name)
