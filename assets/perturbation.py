"""Variant function that sets arbitrary sim_data attributes by dot-path.

Used by the UQ framework to encode LHS parameter samples as variants.
Each variant receives a ``mutations`` dict mapping dot-paths to values.

Config example::

    {
        "variants": {
            "sim_data_setattr": {
                "mutations": {
                    "value": [
                        {"process.transcription.fraction_active_rnap_free": 0.36},
                        {"process.transcription.fraction_active_rnap_free": 0.40}
                    ]
                }
            }
        }
    }

Each entry in the ``value`` list becomes one variant.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli


def apply_variant(
    sim_data: "SimulationDataEcoli",
    params: dict[str, Any],
) -> "SimulationDataEcoli":
    """Apply attribute mutations to sim_data via dot-path traversal.

    Args:
        sim_data: The simulation data object to mutate.
        params: Dict with a ``mutations`` key containing
            ``{dot_path: value}`` pairs. Values can be scalars
            or ``{"__index__": idx, "__value__": val}`` for
            indexed array assignment.

    Returns:
        The mutated sim_data.
    """
    mutations = params.get("mutations", params)
    for attr_path, value in mutations.items():
        parts = attr_path.split(".")
        obj = sim_data
        for part in parts[:-1]:
            obj = getattr(obj, part)
        if isinstance(value, dict) and "__index__" in value:
            arr = getattr(obj, parts[-1])
            arr[value["__index__"]] = value["__value__"]
        else:
            setattr(obj, parts[-1], value)
    return sim_data