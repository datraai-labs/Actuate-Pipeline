"""Load `MANO_RIGHT.pkl` on a modern NumPy/SciPy, without `chumpy`.

The MANO pickle was written in 2017 against Python 2, chumpy, an old SciPy and an old NumPy.
Reading it today needs four names that no longer exist where the pickle expects them. Found
by reading the pickle's own opcodes rather than guessing:

    chumpy.ch.Ch                        chumpy is abandonware -- it does `from numpy import
                                        int`, removed in NumPy 1.24, so it cannot even be
                                        installed against a modern NumPy.
    chumpy.reordering.Select            a LAZY op: shapedirs is stored as
                                        Select(a, idxs) == a.ravel()[idxs].reshape(shape)
    scipy.sparse.csc.csc_matrix         moved to scipy.sparse._csc
    numpy.core.multiarray._reconstruct  moved to numpy._core in NumPy 2

The alternative is pinning NumPy back to 1.23 to satisfy one dead dependency, dragging the
whole project backwards. Instead we install exactly those names, then MATERIALISE the lazy
chumpy objects into plain NumPy before smplx ever sees them -- smplx calls `.shape` on
`shapedirs` immediately, so a lazy object would blow up there.

This is a shim to read a file. Nothing computes with chumpy semantics; we only read values.

--------------------------------------------------------------------------------------
LICENCE -- READ BEFORE SHIPPING ANYTHING DERIVED FROM THIS
--------------------------------------------------------------------------------------
MANO is a Max Planck (MPI) body model whose standard grant is non-commercial. WiLoR's
published models are CC-BY-NC-ND-4.0. This combined path is cleared for INTERNAL RESEARCH
ONLY unless separate commercial agreements have been signed.

Shipping MANO parameters -- or anything derived from them, which includes the Stage-I
retargeted-reference-hand action the Master Spec makes the delivered pretraining target --
requires commercial-rights review with MPI. Tracked in docs/COMMERCIAL_LICENSE_READINESS.md.
"""

from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


class _Ch:
    """`chumpy.ch.Ch`: a lazy array. Its payload lives under `x` in the pickled state."""

    def __setstate__(self, state: Any) -> None:
        self.__dict__.update(state if isinstance(state, dict) else {})

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.__dict__.get("x", []), dtype=np.float64)


class _Select:
    """`chumpy.reordering.Select`: gather + reshape.

        result == asarray(a).ravel()[idxs].reshape(preferred_shape)

    MANO stores `shapedirs` this way -- 23340 gathered floats that reshape to (778, 3, 10).
    """

    def __setstate__(self, state: Any) -> None:
        self.__dict__.update(state if isinstance(state, dict) else {})

    def to_numpy(self) -> np.ndarray:
        a = _materialise(self.__dict__.get("a"))
        idxs = np.asarray(self.__dict__["idxs"], dtype=np.int64)
        flat = np.asarray(a).ravel()[idxs]
        shape = self.__dict__.get("preferred_shape")
        return flat.reshape(shape) if shape else flat


def _materialise(v: Any) -> Any:
    """Turn any lazy chumpy object into plain NumPy. Leaves everything else alone."""
    if isinstance(v, (_Ch, _Select)):
        return v.to_numpy()
    return v


def install_shims() -> None:
    """Register the names the MANO pickle expects. Idempotent."""
    if "chumpy" not in sys.modules:
        chumpy = types.ModuleType("chumpy")
        chumpy.__path__ = []  # declare a package so `chumpy.ch` resolves
        ch = types.ModuleType("chumpy.ch")
        reordering = types.ModuleType("chumpy.reordering")

        ch.Ch = _Ch
        reordering.Select = _Select
        chumpy.Ch, chumpy.ch, chumpy.reordering = _Ch, ch, reordering

        sys.modules["chumpy"] = chumpy
        sys.modules["chumpy.ch"] = ch
        sys.modules["chumpy.reordering"] = reordering

    if "scipy.sparse.csc" not in sys.modules:
        import scipy.sparse

        csc = types.ModuleType("scipy.sparse.csc")
        csc.csc_matrix = scipy.sparse.csc_matrix
        sys.modules["scipy.sparse.csc"] = csc

    if "numpy.core.multiarray" not in sys.modules:
        try:
            import numpy.core.multiarray  # noqa: F401
        except (ImportError, AttributeError):  # pragma: no cover -- NumPy 2 path
            import numpy._core.multiarray as _ma

            core = types.ModuleType("numpy.core")
            core.__path__ = []
            core.multiarray = _ma
            sys.modules.setdefault("numpy.core", core)
            sys.modules["numpy.core.multiarray"] = _ma


def load_mano_pkl(path: Path) -> dict:
    """Read MANO_RIGHT.pkl into plain NumPy, lazy objects resolved."""
    install_shims()
    with Path(path).open("rb") as fh:
        raw = pickle.load(fh, encoding="latin1")
    return {k: _materialise(v) for k, v in raw.items()}


def patch_smplx() -> None:
    """Make `smplx` read MANO through our resolving loader.

    smplx does `pickle.load(...)` and then immediately touches `shapedirs.shape`. A lazy
    chumpy `Select` has no `.shape`, so it must be materialised before smplx sees it.

    We swap only the `pickle` name inside `smplx.body_models` -- patching the stdlib
    `pickle.load` itself would be a global side effect on every unpickle in the process,
    which is exactly the kind of action-at-a-distance that is impossible to debug later.
    """
    install_shims()
    import smplx.body_models as bm

    if getattr(bm, "_actuate_patched", False):
        return

    real_load = pickle.load

    def _load(fh, **kwargs):  # noqa: ANN001, ANN202
        obj = real_load(fh, **kwargs)
        return {k: _materialise(v) for k, v in obj.items()} if isinstance(obj, dict) else obj

    shim = types.ModuleType("pickle")
    shim.load = _load
    shim.loads = pickle.loads
    shim.dump = pickle.dump
    shim.dumps = pickle.dumps

    bm.pickle = shim
    bm._actuate_patched = True
