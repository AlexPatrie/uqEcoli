"""
Flatten a SimulationDataEcoli pickle into a Polars DataFrame.

Usage::

    from uq.pipeline.param_loader import ParameterDataset

    ds = ParameterDataset(sim_data_path="sim_data/violacein/kb/simData.cPickle")
    space = ds.to_parameter_space()          # XSpaceVecoli from real sim_data
    df = ds.to_dataframe()                   # long-format DataFrame

    # Multi-condition merge:
    vio = ParameterDataset(sim_data_path="sim_data/violacein/kb/simData.cPickle")
    mec = ParameterDataset(sim_data_path="sim_data/mecillinam/kb/simData.cPickle")
    space = ParameterDataset.merge_to_parameter_space(vio, mec)
"""
import dataclasses
import os
import pickle
import signal
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import polars as pl
from ecoli.library.sim_data import LoadSimData
from scipy import sparse

from reconstruction.ecoli.simulation_data import SimulationDataEcoli

from uq.common import get_repo_root

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DEFAULT_PICKLE = (get_repo_root() / "sim_data" / "baseline" / "kb" / "simData.cPickle").__str__()

# Attribute names that are known to be expensive to access or not useful data
_SKIP_ATTRS = frozenset({
    "derivatives_jit", "derivatives", "jit", "jacobian",
})

# Sim-data subdirectory → condition label
_CONDITION_LABELS: dict[str, str] = {
    "baseline": "baseline",
    "violacein": "violacein",
    "mecillinam": "mecillinam",
}


@dataclasses.dataclass
class SimDataPayload:
    """Nested dict of serializable values plus any callables found."""
    data: dict[str, Any]
    callbacks: dict[str, Callable]
    sim_data_path: Path


@dataclasses.dataclass
class ParametersDataframe:
    df: pl.DataFrame
    callbacks: dict[str, Callable]


@dataclasses.dataclass
class ParameterDataset:
    """Wraps a ``SimulationDataEcoli`` and exposes methods for
    programmatic parameter space construction.

    Provide *either* ``sim_data_path`` (a path to a ``simData.cPickle``)
    *or* an already-loaded ``sim_data`` instance.

    Properties derived from the sim_data:
        ``has_violacein``  — True if the vio pathway is active
        ``vio_baselines``  — dict of 5 new-gene expression baseline floats
        ``condition``      — inferred condition label (baseline/violacein/mecillinam)

    Parameter-space construction:
        ``to_parameter_space(...)`` — builds an ``XSpaceVecoli`` from this dataset
        ``merge_to_parameter_space(...)`` — class method to combine multiple datasets
    """

    sim_data_path: Path | str | None = None
    sim_data: SimulationDataEcoli | None = None
    experiment_id: str | None = None

    def __post_init__(self):
        if self.sim_data is None:
            if self.sim_data_path is None:
                raise ValueError(
                    "Must pass either sim_data_path or sim_data instance."
                )
            self.sim_data = LoadSimData(str(self.sim_data_path)).sim_data
        if self.sim_data_path is not None:
            self.sim_data_path = Path(self.sim_data_path)

    # ── Condition detection ──────────────────────────────────────────────

    @property
    def condition(self) -> str:
        """Infer condition label from the sim_data_path or sim_data contents."""
        if self.sim_data_path is not None:
            for part in Path(self.sim_data_path).parts:
                if part in _CONDITION_LABELS:
                    return _CONDITION_LABELS[part]
        # Fall back to content inspection
        if self.has_violacein:
            return "violacein"
        return "baseline"

    @property
    def has_violacein(self) -> bool:
        """True if this sim_data includes the violacein pathway."""
        try:
            return bool(self.sim_data.process.metabolism.include_violacein_reactions)
        except AttributeError:
            return False

    # ── Parameter extraction ─────────────────────────────────────────────

    @property
    def vio_baselines(self) -> dict[str, float]:
        """Extract the 5 new-gene expression baselines from sim_data.

        Reads ``sim_data.process.transcription.new_gene_expression_baselines``,
        which is a dict with keys:
            new_gene_rna_synth_prob_baseline
            new_gene_rna_expression_baseline
            new_gene_exp_free_baseline
            new_gene_exp_ppgpp_baseline
            new_gene_reg_basal_prob_baseline

        Returns empty dict if the attribute is absent.
        """
        ts = getattr(self.sim_data, "process", None)
        ts = getattr(ts, "transcription", None) if ts else None
        if ts is None:
            return {}
        raw = getattr(ts, "new_gene_expression_baselines", None)
        if isinstance(raw, dict):
            return {k: float(v) for k, v in raw.items()}
        return {}

    # ── Parameter space construction ─────────────────────────────────────

    def to_parameter_space(
        self,
        include_vio: bool | None = None,
        include_mecillinam: bool | None = None,
        vio_expression_bounds: tuple[float, float] | None = None,
        vio_trl_eff_bounds: tuple[float, float] = (0.0, 2.0),
        mecillinam_conc_bounds: tuple[float, float] = (0.0, 10.0),
    ) -> "XSpaceVecoli":
        """Build an ``XSpaceVecoli`` from this dataset.

        When ``include_vio`` is *None* (default), it is auto-detected from
        ``self.has_violacein``.  When ``vio_expression_bounds`` is *None*,
        bounds are derived from the sim_data baselines: [0, 5× baseline].

        Args:
            include_vio: Include vio pathway parameters.  Auto-detected if None.
            include_mecillinam: Include mecillinam parameters.  Defaults to
                True only when condition is "mecillinam".
            vio_expression_bounds: (lo, hi) for vio expression factor.
                Derived from sim_data baselines if None.
            vio_trl_eff_bounds: (lo, hi) for translation efficiency multiplier.
            mecillinam_conc_bounds: (lo, hi) for mecillinam concentration.

        Returns:
            Configured ``XSpaceVecoli`` instance.
        """
        from uq.inputs import XSpaceVecoli

        # Auto-detect flags from sim_data content
        if include_vio is None:
            include_vio = self.has_violacein
        if include_mecillinam is None:
            include_mecillinam = self.condition == "mecillinam"

        # Derive vio expression bounds from baselines
        if vio_expression_bounds is None and include_vio:
            baselines = self.vio_baselines
            base_expr = baselines.get("new_gene_rna_expression_baseline")
            if base_expr is not None and base_expr > 0:
                # UQ sweep: 0 to 5× the baseline expression level
                vio_expression_bounds = (0.0, 5.0 * base_expr / base_expr)  # normalized to multiplier
            else:
                vio_expression_bounds = (0.0, 5.0)

        return XSpaceVecoli(
            experiment_id=self.condition,
            include_vio=include_vio,
            include_mecillinam=include_mecillinam,
            vio_expression_bounds=vio_expression_bounds or (0.0, 5.0),
            vio_trl_eff_bounds=vio_trl_eff_bounds,
            mecillinam_conc_bounds=mecillinam_conc_bounds,
        )

    @classmethod
    def merge_to_parameter_space(
        cls,
        *datasets: "ParameterDataset",
        vio_expression_bounds: tuple[float, float] | None = None,
        vio_trl_eff_bounds: tuple[float, float] = (0.0, 2.0),
        mecillinam_conc_bounds: tuple[float, float] = (0.0, 10.0),
    ) -> "XSpaceVecoli":
        """Merge parameters from multiple condition datasets into one ``XSpaceVecoli``.

        OR's the include flags across all datasets: if *any* dataset has
        violacein, the merged space includes vio parameters.  Same for
        mecillinam.

        If ``vio_expression_bounds`` is None and any dataset has vio
        baselines, bounds are derived from the first vio dataset.

        Example::

            vio = ParameterDataset(sim_data_path="sim_data/violacein/kb/simData.cPickle")
            mec = ParameterDataset(sim_data_path="sim_data/mecillinam/kb/simData.cPickle")
            space = ParameterDataset.merge_to_parameter_space(vio, mec)
            # → XSpaceVecoli with include_vio=True, include_mecillinam=True
        """
        if not datasets:
            raise ValueError("Need at least one ParameterDataset.")

        include_vio = any(ds.has_violacein for ds in datasets)
        include_mec = any(ds.condition == "mecillinam" for ds in datasets)

        # Derive bounds from the first vio dataset
        if vio_expression_bounds is None and include_vio:
            for ds in datasets:
                if ds.has_violacein and ds.vio_baselines:
                    vio_expression_bounds = ds.to_parameter_space(
                        include_vio=True,
                        include_mecillinam=False,
                    ).parameter_bounds[0]  # first param = vio_expression
                    break

        from uq.inputs import XSpaceVecoli

        experiment_id = "+".join(ds.condition for ds in datasets)

        return XSpaceVecoli(
            experiment_id=experiment_id,
            include_vio=include_vio,
            include_mecillinam=include_mec,
            vio_expression_bounds=vio_expression_bounds or (0.0, 5.0),
            vio_trl_eff_bounds=vio_trl_eff_bounds,
            mecillinam_conc_bounds=mecillinam_conc_bounds,
        )

    # ── Serialization ────────────────────────────────────────────────────

    def to_payload(self) -> SimDataPayload:
        return sim_data_to_dict(self.sim_data)

    def to_dataframe(self) -> ParametersDataframe:
        return load_sim_data_df(path=str(self.sim_data_path))

    @classmethod
    def from_payload(cls, serialized: SimDataPayload) -> "ParameterDataset":
        return ParameterDataset(
            sim_data_path=serialized.sim_data_path,
            sim_data=dict_to_sim_data(serialized),
        )


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout()


def _safe_getattr(obj, name, timeout_s=2):
    """getattr with a signal-based timeout to skip expensive computed attrs."""
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_s)
    try:
        val = getattr(obj, name)
    except _Timeout:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        return None, True  # (value, timed_out)
    except Exception:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        return None, True
    signal.alarm(0)
    signal.signal(signal.SIGALRM, old)
    return val, False


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------

def _is_unum(obj: Any) -> bool:
    return type(obj).__name__ == "Unum"


def _is_unit_struct_array(obj: Any) -> bool:
    return type(obj).__name__ == "UnitStructArray"


def _is_data_container(obj: Any) -> bool:
    """True for vEcoli data-class objects that should be recursed into."""
    mod = getattr(type(obj), "__module__", "") or ""
    return (
        mod.startswith("reconstruction.ecoli")
        or mod.startswith("wholecell")
        or mod.startswith("ecoli")
    )


def _resolve_value(obj: Any) -> tuple[Any, str | None, str]:
    """Return (value, unit_string, dtype_label) for a leaf object."""
    if _is_unum(obj):
        try:
            val = obj.asNumber()
        except Exception:
            val = str(obj)
        unit = obj.strUnit()
        if isinstance(val, np.ndarray):
            return val.tolist(), unit, f"ndarray({val.shape},{val.dtype})"
        return val, unit, type(val).__name__

    if _is_unit_struct_array(obj):
        sa = obj.struct_array
        return sa.tolist(), str(obj.units), f"struct_array({sa.shape},{sa.dtype})"

    if sparse.issparse(obj):
        dense = obj.toarray()
        return dense.tolist(), None, f"sparse({dense.shape},{dense.dtype})"

    if isinstance(obj, np.ndarray):
        return obj.tolist(), None, f"ndarray({obj.shape},{obj.dtype})"

    if isinstance(obj, dict):
        return str(obj)[:500], None, "dict"

    if isinstance(obj, (set, frozenset)):
        return sorted(str(x) for x in obj), None, "set"

    if isinstance(obj, (list, tuple)):
        return obj, None, type(obj).__name__

    # Scalars: int, float, str, bool, None
    return obj, None, type(obj).__name__


# ---------------------------------------------------------------------------
# Tree walker
# ---------------------------------------------------------------------------

def _instance_attrs(obj: Any) -> list[str]:
    """Get attribute names from __dict__ (instance attrs only, no methods)."""
    d = getattr(obj, "__dict__", None)
    if d is None:
        return []
    return sorted(n for n in d if not n.startswith("_"))


def _walk(
    obj: Any,
    prefix: str,
    rows: list[dict],
    callbacks: dict[str, Callable],
    max_depth: int = 3,
    depth: int = 0,
):
    """Recursively walk *obj* collecting (path, value, unit, dtype) rows.

    Callables are stored in *callbacks* instead of being discarded.
    """
    if depth > max_depth:
        return

    names = _instance_attrs(obj)

    for name in names:
        if name in _SKIP_ATTRS:
            continue
        path = f"{prefix}.{name}" if prefix else name

        attr, timed_out = _safe_getattr(obj, name)
        if timed_out:
            rows.append({"path": path, "value": "<timed_out>", "unit": None, "dtype": "skipped"})
            continue

        # Recurse into vEcoli data containers
        if _is_data_container(attr) and depth < max_depth:
            _walk(attr, path, rows, callbacks, max_depth, depth + 1)
        elif callable(attr) and not isinstance(attr, type):
            callbacks[path] = attr
        else:
            val, unit, dtype_label = _resolve_value(attr)
            rows.append({
                "path": path,
                "value": _to_str(val),
                "unit": unit,
                "dtype": dtype_label,
            })


def _to_str(val: Any) -> str:
    """Convert any value to a string representation for the DataFrame."""
    if val is None:
        return ""
    if isinstance(val, (list, tuple)):
        if len(val) > 200:
            return f"[{len(val)} elements] {str(val[:5])}..."
        return str(val)
    return str(val)


def _serialize_key(k: Any) -> str | int | float | bool:
    """Coerce a dict key to a JSON-compatible type."""
    if isinstance(k, np.integer):
        return int(k)
    if isinstance(k, np.floating):
        return float(k)
    if isinstance(k, (str, int, float, bool)):
        return k
    return str(k)


def _serialize_leaf(obj: Any) -> Any:
    """Convert a leaf value to a JSON-serializable, tagged form.

    Tagged values use a ``__type__`` key so that ``_deserialize_leaf`` can
    reconstruct the original Python/numpy type losslessly.
    """
    if _is_unum(obj):
        try:
            val = obj.asNumber()
        except Exception:
            val = str(obj)
        unit = obj.strUnit()
        if isinstance(val, np.ndarray):
            return {"__type__": "unum", "value": val.tolist(),
                    "dtype": str(val.dtype), "unit": unit}
        return {"__type__": "unum", "value": val, "unit": unit}

    if _is_unit_struct_array(obj):
        sa = obj.struct_array
        return {
            "__type__": "unit_struct_array",
            "value": sa.tolist(),
            "dtype_descr": sa.dtype.descr,
            "units": {k: str(v) if v is not None else None
                      for k, v in obj.units.items()},
        }

    if sparse.issparse(obj):
        fmt = obj.format if hasattr(obj, "format") else "csr"
        dense = obj.toarray()
        return {"__type__": "sparse", "data": dense.tolist(),
                "dtype": str(dense.dtype), "format": fmt}

    if isinstance(obj, np.ndarray):
        return {"__type__": "ndarray", "data": obj.tolist(),
                "dtype": str(obj.dtype)}

    if isinstance(obj, (set, frozenset)):
        return {"__type__": "set", "data": sorted(str(x) for x in obj)}

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, dict):
        return {_serialize_key(k): _serialize_leaf(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_serialize_leaf(x) for x in obj]

    # Scalars: int, float, str, bool, None — already serializable
    return obj


def _to_nested_dict(
    obj: Any,
    prefix: str,
    callbacks: dict[str, Callable],
    max_depth: int = 3,
    depth: int = 0,
) -> dict[str, Any]:
    """Recursively convert *obj* into a nested dict of serializable values.

    Each intermediate container node is tagged with ``__class__`` and
    ``__module__`` so ``dict_to_sim_data`` can reconstruct the original
    object types.  Callables are separated into *callbacks* keyed by dot-path.
    """
    result: dict[str, Any] = {}
    if depth > max_depth:
        return result

    # Tag this node so reconstruction knows the class
    cls = type(obj)
    result["__class__"] = cls.__qualname__
    result["__module__"] = cls.__module__

    names = _instance_attrs(obj)

    for name in names:
        if name in _SKIP_ATTRS:
            continue
        path = f"{prefix}.{name}" if prefix else name

        attr, timed_out = _safe_getattr(obj, name)
        if timed_out:
            continue

        if _is_data_container(attr) and depth < max_depth:
            child = _to_nested_dict(attr, path, callbacks, max_depth, depth + 1)
            if child:
                result[name] = child
        elif callable(attr) and not isinstance(attr, type):
            callbacks[path] = attr
        else:
            result[name] = _serialize_leaf(attr)

    return result


# ---------------------------------------------------------------------------
# Deserialization helpers
# ---------------------------------------------------------------------------

def _import_class(module: str, qualname: str) -> type:
    """Import and return a class given its module and qualname."""
    import importlib
    mod = importlib.import_module(module)
    obj = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


_UNIT_ALIASES = {
    "nucleotide": "nt",
    "amino_acid": "aa",
}


def _parse_unit_str(unit_str: str | None) -> Any:
    """Parse a Unum unit string like ``[umol/L]`` back into a Unum unit.

    Returns ``None`` if the string is empty or ``[]`` (dimensionless).
    """
    if not unit_str or unit_str.strip() in ("", "[]"):
        return None
    from wholecell.utils import units as units_pkg
    # Strip brackets: "[umol/L]" -> "umol/L"
    s = unit_str.strip("[] ")
    if not s:
        return None
    # Apply aliases
    for alias, canonical in _UNIT_ALIASES.items():
        s = s.replace(alias, canonical)
    # Split numerator / denominator on "/"
    parts = s.split("/")
    try:
        numer = _parse_unit_product(parts[0], units_pkg)
        result = numer
        for denom_part in parts[1:]:
            result = result / _parse_unit_product(denom_part, units_pkg)
        return result
    except (AttributeError, ValueError):
        return None


def _parse_unit_product(expr: str, units_pkg: Any) -> Any:
    """Parse a product of unit tokens like ``umol``, ``s.umol``, etc.

    Multiplication can be ``*`` or ``.`` (Unum uses ``.``).
    """
    import re
    tokens = re.split(r"[.*]", expr.strip())
    result = None
    for tok in tokens:
        tok = tok.strip()
        if not tok or tok == "1":
            continue
        unit = getattr(units_pkg, tok, None)
        if unit is None:
            raise AttributeError(f"Unknown unit: {tok}")
        result = unit if result is None else result * unit
    if result is None:
        # "1" or empty — dimensionless
        return 1
    return result


def _deserialize_leaf(obj: Any) -> Any:
    """Reverse of ``_serialize_leaf``: reconstruct original Python types."""
    if not isinstance(obj, dict) or "__type__" not in obj:
        # Plain scalar, or a dict/list that was serialized recursively
        if isinstance(obj, dict):
            return {k: _deserialize_leaf(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_deserialize_leaf(x) for x in obj]
        return obj

    t = obj["__type__"]

    if t == "unum":
        import unum as unum_mod
        val = obj["value"]
        if "dtype" in obj:
            val = np.array(val, dtype=np.dtype(obj["dtype"]))
        unit_str = obj["unit"]
        unit = _parse_unit_str(unit_str)
        if unit is not None:
            return val * unit
        # Unknown unit string — wrap as dimensionless Unum to preserve type
        return unum_mod.Unum.uniform(val)

    if t == "unit_struct_array":
        from wholecell.utils.unit_struct_array import UnitStructArray
        dtype = np.dtype(obj["dtype_descr"])
        sa = np.array([tuple(row) for row in obj["value"]], dtype=dtype)
        unit_map = {}
        for field, unit_str in obj["units"].items():
            if unit_str is None or unit_str == "None":
                unit_map[field] = None
            else:
                unit_map[field] = _parse_unit_str(unit_str)
        return UnitStructArray(sa, unit_map)

    if t == "ndarray":
        dtype_spec = obj["dtype"]
        try:
            dtype = np.dtype(dtype_spec)
        except TypeError:
            # Structured dtype stored as string repr — eval is safe here
            # since it only contains numpy dtype descriptors
            dtype = np.dtype(eval(dtype_spec))  # noqa: S307
        data = obj["data"]
        if dtype.names is not None:
            # Structured array: each element must be a tuple
            return np.array([tuple(row) for row in data], dtype=dtype)
        return np.array(data, dtype=dtype)

    if t == "sparse":
        from scipy import sparse as sp
        dense = np.array(obj["data"], dtype=np.dtype(obj["dtype"]))
        fmt = obj.get("format", "csr")
        if fmt == "csc":
            return sp.csc_matrix(dense)
        return sp.csr_matrix(dense)

    if t == "set":
        return set(obj["data"])

    return obj


def _from_nested_dict(
    d: dict[str, Any],
    callbacks: dict[str, Callable],
    prefix: str = "",
) -> Any:
    """Reconstruct an object tree from a tagged nested dict.

    Uses ``__class__`` / ``__module__`` tags to instantiate the correct
    container classes, then sets deserialized attributes on them.
    Callables from *callbacks* are re-attached at their original dot-paths.
    """
    module = d.get("__module__")
    qualname = d.get("__class__")

    # Instantiate the container without calling __init__
    cls = _import_class(module, qualname)
    obj = object.__new__(cls)

    for key, val in d.items():
        if key in ("__class__", "__module__"):
            continue
        path = f"{prefix}.{key}" if prefix else key

        if isinstance(val, dict) and "__class__" in val and "__module__" in val:
            # Nested container — recurse
            child = _from_nested_dict(val, callbacks, prefix=path)
            setattr(obj, key, child)
        else:
            setattr(obj, key, _deserialize_leaf(val))

    # Re-attach any callbacks that belong at this level
    for cb_path, cb_func in callbacks.items():
        parts = cb_path.split(".")
        if prefix:
            prefix_parts = prefix.split(".")
            if parts[:-1] == prefix_parts:
                setattr(obj, parts[-1], cb_func)
        else:
            if len(parts) == 1:
                setattr(obj, parts[0], cb_func)

    return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_sim_data(path: str = _DEFAULT_PICKLE):
    """Load and return the raw SimulationDataEcoli object."""
    with open(path, "rb") as f:
        return pickle.load(f)


def dict_to_sim_data(serialized: SimDataPayload) -> Any:
    """Reconstruct a ``SimulationDataEcoli`` from a ``SerializedSimData``.

    Inverse of ``sim_data_to_dict``.  Uses the ``__class__`` / ``__module__``
    tags embedded in the nested dict to instantiate the correct container
    classes, deserializes tagged leaves back to their original types
    (ndarray, Unum, UnitStructArray, sparse matrix, set), and re-attaches
    callables from ``serialized.callbacks``.
    """
    return _from_nested_dict(serialized.data, serialized.callbacks)


def sim_data_to_dict(
    sim_data: SimulationDataEcoli
) -> SimDataPayload:
    """Convert a ``SimulationDataEcoli`` instance to a nested dict.

    Returns a ``SerializedSimData`` with:
        data      – nested dict mirroring the object tree; every leaf is
                    JSON-serializable (ndarrays → lists, Unum → {value, unit},
                    UnitStructArray → {value, units}, sparse → dense list, etc.)
        callbacks – flat dict mapping dot-paths to callable attributes
    """
    callbacks: dict[str, Callable] = {}
    data = _to_nested_dict(sim_data, "", callbacks)
    return SimDataPayload(data=data, callbacks=callbacks)


def load_sim_data_df(
    path: str = _DEFAULT_PICKLE,
    max_depth: int = 3,
) -> ParametersDataframe:
    """Load ``SimulationDataEcoli`` and flatten into a ``ParametersDataset``.

    Returns a ``ParametersDataset`` with:
        df        – long-format DataFrame with columns (path, value, unit, dtype)
        callbacks – dict mapping dot-paths to callable attributes found during
                    the walk (methods, functions, lambdas stored on the object)

    Unum objects are unwrapped via ``asNumber()`` / ``strUnit()``.
    UnitStructArrays and sparse matrices are converted to dense representations.
    """
    sim_data = load_sim_data(path)
    rows: list[dict] = []
    callbacks: dict[str, Callable] = {}
    _walk(sim_data, "", rows, callbacks, max_depth=max_depth)
    return ParametersDataframe(df=pl.DataFrame(rows), callbacks=callbacks)


if __name__ == "__main__":
    ds = load_sim_data_df()
    print(f"Loaded {len(ds.df)} data attributes from SimulationDataEcoli")
    print(f"Captured {len(ds.callbacks)} callbacks")
    print(ds.df.head(20))
    print(f"\nDtype distribution:")
    print(ds.df.group_by("dtype").len().sort("len", descending=True))
    print(f"\nCallback paths (first 20):")
    for i, path in enumerate(sorted(ds.callbacks)):
        if i >= 20:
            print(f"  ... and {len(ds.callbacks) - 20} more")
            break
        print(f"  {path}")

