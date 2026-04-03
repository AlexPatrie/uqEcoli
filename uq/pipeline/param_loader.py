"""
Flatten a SimulationDataEcoli pickle into a Polars DataFrame.

Usage::

    from uq.pipeline.param_loader import ParameterDataset

    ds = ParameterDataset(sim_data_path="sim_data/baseline/kb/simData.cPickle")
    space = ds.to_parameter_space()          # XSpaceVecoli from real sim_data
    df = ds.to_dataframe()                   # long-format DataFrame
"""

import dataclasses
import pickle
import signal
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl
from ecoli.library.sim_data import LoadSimData
from reconstruction.ecoli.simulation_data import SimulationDataEcoli
from scipy import sparse

from uq.common import get_repo_root
from uq.pipeline.models import SimDataParameter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DEFAULT_PICKLE = (get_repo_root() / "sim_data" / "baseline" / "kb" / "simData.cPickle").__str__()


# ── Default generic sim_data parameters for UQ ────────────────────────────

DEFAULT_SIM_DATA_PARAMETERS: list[SimDataParameter] = [
    # Bounds are ±30% around typical baseline values to avoid crashing
    # the simulation.  These span the three axes of the growth rate
    # control loop: transcription (RNAP), translation (ribosome speed),
    # and metabolism (FBA tuning + mass).
    #
    # -- Transcription: ppGpp-mediated RNAP regulation --
    # (Ahn-Horst et al. 2022, Fig. 1b)
    SimDataParameter(
        name="fraction_active_rnap_free",
        attr_path="process.transcription.fraction_active_rnap_free",
        bounds=(0.25, 0.47),
        description=(
            "Fraction of RNAP actively transcribing when ppGpp-free. "
            "Controls rRNA/mRNA/tRNA synthesis rates. Experimentally "
            "tunable via ppGpp circuit mutations (relA/spoT). Baseline ~0.36."
        ),
    ),
    SimDataParameter(
        name="fraction_active_rnap_bound",
        attr_path="process.transcription.fraction_active_rnap_bound",
        bounds=(0.12, 0.22),
        description=(
            "Fraction of RNAP actively transcribing when ppGpp-bound. "
            "ppGpp destabilizes open complex formation, reducing this "
            "fraction. Tunable via relA/spoT knockouts. Baseline ~0.17."
        ),
    ),
    # -- Translation: ribosome elongation speed --
    # (experimentally affected by antibiotics, temperature, nutrient quality)
    SimDataParameter(
        name="basal_elongation_rate",
        attr_path="process.translation.basal_elongation_rate",
        bounds=(15.0, 28.0),
        description=(
            "Ribosome elongation rate for non-ribosomal proteins (aa/s). "
            "Experimentally tunable via sub-inhibitory chloramphenicol, "
            "fusidic acid, or growth temperature. Baseline ~22 aa/s."
        ),
    ),
    # -- Metabolism: FBA solver parameters --
    # (control metabolic flux distribution; secretion_penalty_coeff
    #  affects acetate overflow, experimentally observable via media)
    SimDataParameter(
        name="kinetic_objective_weight",
        attr_path="process.metabolism.kinetic_objective_weight",
        bounds=(5e-8, 5e-7),
        description=(
            "Weight on kinetic vs homeostatic objective in FBA. Controls "
            "balance between matching enzyme kinetics and maintaining "
            "metabolite homeostasis. Baseline 1e-7 (linear solver)."
        ),
    ),
    SimDataParameter(
        name="secretion_penalty_coeff",
        attr_path="process.metabolism.secretion_penalty_coeff",
        bounds=(5e-4, 5e-3),
        description=(
            "Penalty on metabolite secretion fluxes in FBA. Higher values "
            "force the cell to retain metabolites. Controls acetate overflow "
            "metabolism, experimentally tunable via media composition. "
            "Baseline 0.001."
        ),
    ),
    # -- Cell composition --
    SimDataParameter(
        name="cell_dry_mass_fraction",
        attr_path="mass.cell_dry_mass_fraction",
        bounds=(0.25, 0.35),
        description=(
            "Fraction of total cell mass that is dry mass. Determines "
            "relationship between cell volume and biosynthetic capacity. "
            "Constrained by buoyant density measurements. Baseline ~0.30."
        ),
    ),
]

# Attribute names that are known to be expensive to access or not useful data
_SKIP_ATTRS = frozenset({
    "derivatives_jit",
    "derivatives",
    "jit",
    "jacobian",
})


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

    Attributes:
        sim_data_path: Path | str | None
        sim_data: SimulationDataEcoli | None
        experiment_id: str | None

    Provide *either* ``sim_data_path`` (a path to a ``simData.cPickle``)
    *or* an already-loaded ``sim_data`` instance.

    Properties:
        ``condition`` — inferred condition label from sim_data_path

    Parameter-space construction:
        ``to_parameter_space(...)`` — builds an ``XSpaceVecoli`` from generic
            ``SimDataParameter`` specs (defaults to ``DEFAULT_SIM_DATA_PARAMETERS``)
    """

    sim_data_path: Path | str | None = None
    sim_data: SimulationDataEcoli | None = None
    experiment_id: str | None = None

    def __post_init__(self):
        if self.sim_data is None:
            if self.sim_data_path is None:
                raise ValueError("Must pass either sim_data_path or sim_data instance.")
            self.sim_data = LoadSimData(str(self.sim_data_path)).sim_data
        if self.sim_data_path is not None:
            self.sim_data_path = Path(self.sim_data_path)

    # ── Condition detection ──────────────────────────────────────────────

    @property
    def condition(self) -> str:
        """Infer condition label from the sim_data_path directory name."""
        if self.sim_data_path is not None:
            for part in Path(self.sim_data_path).parts:
                if part in ("baseline", "violacein", "mecillinam"):
                    return part
        return "baseline"

    # ── Parameter space construction ─────────────────────────────────────

    def to_parameter_space(
        self,
        parameters: list[SimDataParameter] | None = None,
    ) -> "XSpaceVecoli":
        """Build an ``XSpaceVecoli`` from this dataset.

        Uses a list of ``SimDataParameter`` specs identifying arbitrary
        scalar attributes in sim_data by dot-path. Each attr_path is
        validated against ``self.sim_data``. If ``parameters`` is not
        provided, defaults to ``DEFAULT_SIM_DATA_PARAMETERS``.

        Args:
            parameters: List of ``SimDataParameter`` specs. Defaults to
                ``DEFAULT_SIM_DATA_PARAMETERS`` when None.

        Returns:
            Configured ``XSpaceVecoli`` instance.
        """
        from uq.inputs import XSpaceVecoli

        if parameters is None:
            parameters = list(DEFAULT_SIM_DATA_PARAMETERS)
        validated = self._validate_sim_data_parameters(parameters)
        return XSpaceVecoli(
            experiment_id=self.condition,
            parameters=validated,
        )

    def _validate_sim_data_parameters(
        self,
        parameters: list[SimDataParameter],
    ) -> list[SimDataParameter]:
        """Validate that each attr_path exists on self.sim_data."""
        validated = []
        for p in parameters:
            obj = self.sim_data
            parts = p.attr_path.split(".")
            for part in parts:
                if not hasattr(obj, part):
                    raise ValueError(
                        f"SimDataParameter '{p.name}': attr_path "
                        f"'{p.attr_path}' not found on sim_data "
                        f"(failed at '{part}')"
                    )
                obj = getattr(obj, part)
            validated.append(p)
        return validated

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
    return mod.startswith("reconstruction.ecoli") or mod.startswith("wholecell") or mod.startswith("ecoli")


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
            return f"[{len(val)} elements] {val[:5]!s}..."
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
            return {"__type__": "unum", "value": val.tolist(), "dtype": str(val.dtype), "unit": unit}
        return {"__type__": "unum", "value": val, "unit": unit}

    if _is_unit_struct_array(obj):
        sa = obj.struct_array
        return {
            "__type__": "unit_struct_array",
            "value": sa.tolist(),
            "dtype_descr": sa.dtype.descr,
            "units": {k: str(v) if v is not None else None for k, v in obj.units.items()},
        }

    if sparse.issparse(obj):
        fmt = obj.format if hasattr(obj, "format") else "csr"
        dense = obj.toarray()
        return {"__type__": "sparse", "data": dense.tolist(), "dtype": str(dense.dtype), "format": fmt}

    if isinstance(obj, np.ndarray):
        return {"__type__": "ndarray", "data": obj.tolist(), "dtype": str(obj.dtype)}

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


def sim_data_to_dict(sim_data: SimulationDataEcoli) -> SimDataPayload:
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
    print("\nDtype distribution:")
    print(ds.df.group_by("dtype").len().sort("len", descending=True))
    print("\nCallback paths (first 20):")
    for i, path in enumerate(sorted(ds.callbacks)):
        if i >= 20:
            print(f"  ... and {len(ds.callbacks) - 20} more")
            break
        print(f"  {path}")
