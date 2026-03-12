"""
UQ Pipeline → Musical Score encoding.

This module maps the outputs of the RFC006 UQ pipeline onto Western musical
notation, providing an alternative "reading" of sensitivity analysis results.

The existing apollo package maps Koopman modes → individual notes (one mode,
one note). This module operates at a higher level: it maps the *entire pipeline
output* — Sobol indices, variance decomposition, per-stage sensitivity — into
a score that a user can read to understand "which parameters matter, where does
uncertainty come from, and how does it change across the cell cycle?"

Structural Mapping (not metaphor — these are genuine mathematical parallels)
═══════════════════════════════════════════════════════════════════════════════

    ┌──────────────────────────────┬──────────────────────────────────────────┐
    │ UQ Pipeline Concept          │ Musical Concept                          │
    ├──────────────────────────────┼──────────────────────────────────────────┤
    │ Input parameter              │ Voice / Instrument                       │
    │   (vio_expression, etc.)     │   Each is an independent contributor     │
    │                              │   to the combined output "sound"         │
    ├──────────────────────────────┼──────────────────────────────────────────┤
    │ Sobol S_i (first-order)      │ Dynamic marking (pp → ff)               │
    │                              │   How loud this voice is when soloing    │
    ├──────────────────────────────┼──────────────────────────────────────────┤
    │ Sobol S_T - S_i              │ Harmony indicator                        │
    │   (interaction component)    │   How much this voice depends on playing │
    │                              │   with others (consonance vs dissonance) │
    ├──────────────────────────────┼──────────────────────────────────────────┤
    │ Variance decomposition       │ Register / Voicing                       │
    │   generation fraction        │   Bass register (low, foundational)      │
    │   seed fraction              │   Tenor register (middle, structural)    │
    │   residual (cell cycle)      │   Treble register (high, expressive)     │
    ├──────────────────────────────┼──────────────────────────────────────────┤
    │ Cell cycle stages            │ Beats within a measure                   │
    │   (bins 0..N)                │   The periodic temporal framework;       │
    │                              │   one full cycle = one measure           │
    ├──────────────────────────────┼──────────────────────────────────────────┤
    │ PCE surrogate                │ The score itself                         │
    │                              │   Compact symbolic notation that lets    │
    │                              │   you "replay" instantly without the     │
    │                              │   full orchestra (simulation)            │
    ├──────────────────────────────┼──────────────────────────────────────────┤
    │ Morris screening             │ Audition / Rehearsal                     │
    │                              │   Cheap pass to decide which voices      │
    │                              │   to feature in the final performance    │
    ├──────────────────────────────┼──────────────────────────────────────────┤
    │ Aggregation strategy         │ Listening perspective                    │
    │   UNIFORM                    │   Tutti (whole ensemble blend)           │
    │   BY_GENERATION              │   Listening across movements (time)      │
    │   BY_LINEAGE_SEED            │   Comparing different performances       │
    │   BY_CELL_CYCLE              │   Listening within a single measure      │
    └──────────────────────────────┴──────────────────────────────────────────┘

Why this works mathematically
─────────────────────────────
PCE coefficients *are* a spectral decomposition — they expand the model response
in an orthogonal polynomial basis. Sobol indices partition total output variance
into per-parameter contributions, exactly as Fourier power spectrum partitions
total signal energy into per-frequency contributions. The cell cycle variable IS
a fundamental frequency (via Koopman). This isn't analogy — it's the same math
in different notation.

Usage
─────
    from apollo.uq_score import encode_sensitivity, SensitivityScore

    score = encode_sensitivity(
        sobol_indices=sobol,
        variance_decomposition=decomp,
        cell_cycle_time=3600.0,
        per_stage_sobol=per_stage_sobol,  # optional
    )

    print(score.to_ascii())    # Human-readable score
    print(score.read_aloud())  # Natural language "program notes"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from apollo.mappings import cycle_time_to_tempo
from apollo.types import Dynamic, Tempo

# ═══════════════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════════════


class Instrument(str, Enum):
    """
    Voice/instrument assignment for a parameter.

    Higher Sobol total-order index → more prominent instrument.
    The mapping is by rank: the most influential parameter gets the
    most prominent instrument in its register.
    """

    # Solo instruments (top 1-2 parameters)
    VIOLIN_1 = "Violin I"
    CELLO = "Cello"

    # Section leaders (parameters 3-5)
    VIOLIN_2 = "Violin II"
    VIOLA = "Viola"
    OBOE = "Oboe"

    # Ensemble (parameters 6+)
    FLUTE = "Flute"
    CLARINET = "Clarinet"
    BASSOON = "Bassoon"
    HORN = "Horn"
    TRUMPET = "Trumpet"

    # Negligible parameters
    REST = "—"


class InteractionQuality(str, Enum):
    """
    How a parameter interacts with others.

    Derived from the gap between total-order and first-order Sobol indices:
        interaction_strength = (S_T - S_i) / S_T

    Mapped to consonance/dissonance in the harmonic sense.
    """

    UNISON = "unison"  # interaction_strength < 0.05: pure solo effect
    CONSONANT = "consonant"  # 0.05 - 0.20: mild cooperative interactions
    DISSONANT = "dissonant"  # 0.20 - 0.50: moderate cross-parameter coupling
    CLUSTER = "cluster"  # > 0.50: dominated by interactions, not solo


class VarianceRegister(str, Enum):
    """
    Register assignment for variance components.

    Variance decomposition maps onto the registral space of the score:
    low frequencies (bass) = slow-moving foundational effects,
    high frequencies (treble) = fast-moving, cycle-scale variation.
    """

    BASS = "bass"  # Generation variance (slow convergence effect)
    TENOR = "tenor"  # Seed variance (stochastic, between-lineage)
    TREBLE = "treble"  # Residual variance (cell-cycle-related, within-cell)


# ═══════════════════════════════════════════════════════════════════════════════
# Score Data Structures
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ParameterVoice:
    """
    A single parameter's representation as a musical voice.

    This is the core unit: one parameter = one voice in the ensemble.
    """

    name: str
    instrument: Instrument

    # Sensitivity (how loud this voice is)
    first_order_sobol: float  # S_i: solo contribution
    total_order_sobol: float  # S_T: total contribution including interactions
    dynamic: Dynamic  # Derived from S_T

    # Interaction character
    interaction_strength: float  # (S_T - S_i) / S_T if S_T > 0
    interaction_quality: InteractionQuality

    # Morris screening (if available)
    mu_star: Optional[float] = None  # Mean absolute elementary effect
    passed_audition: bool = True  # Whether Morris selected this parameter

    def __str__(self) -> str:
        audit = " (auditioned)" if self.passed_audition else " (cut)"
        return (
            f"{self.instrument.value:12s} {self.dynamic.value:5s} "
            f"S_i={self.first_order_sobol:.3f} S_T={self.total_order_sobol:.3f} "
            f"[{self.interaction_quality.value}]{audit} — {self.name}"
        )


@dataclass
class VarianceVoicing:
    """
    The registral distribution of variance — which "register" dominates the sound.

    If generation_fraction is 0.6 and seed_fraction is 0.3, the voicing is
    "bass-heavy" — most of the uncertainty comes from the slow, foundational
    convergence effects, not from stochastic or cell-cycle variation.
    """

    generation_fraction: float  # Bass register share
    seed_fraction: float  # Tenor register share
    residual_fraction: float  # Treble register share (= 1 - gen - seed)

    dominant_register: VarianceRegister = field(init=False)
    balance_description: str = field(init=False)

    def __post_init__(self):
        fracs = {
            VarianceRegister.BASS: self.generation_fraction,
            VarianceRegister.TENOR: self.seed_fraction,
            VarianceRegister.TREBLE: self.residual_fraction,
        }
        self.dominant_register = max(fracs, key=fracs.get)

        # Describe the balance
        if max(fracs.values()) > 0.6:
            self.balance_description = f"dominated by {self.dominant_register.value} register"
        elif max(fracs.values()) < 0.4:
            self.balance_description = "balanced across all registers"
        else:
            top_two = sorted(fracs.items(), key=lambda x: -x[1])[:2]
            self.balance_description = (
                f"split between {top_two[0][0].value} ({top_two[0][1]:.0%}) "
                f"and {top_two[1][0].value} ({top_two[1][1]:.0%})"
            )


@dataclass
class BeatSensitivity:
    """
    Sensitivity profile at one beat (cell cycle stage) within the measure.

    If we have 10 cell cycle bins, we have 10 beats per measure. Each beat
    has its own dynamic profile — some parameters may be forte during the
    C period and piano during the B period.
    """

    stage_index: int
    stage_label: str  # e.g., "B period", "C period", "beat 3/10"
    parameter_dynamics: dict[str, Dynamic]  # param_name → dynamic at this beat

    def loudest_voice(self) -> Optional[str]:
        """Which parameter is loudest at this beat?"""
        if not self.parameter_dynamics:
            return None
        # Dynamic enum ordering: PPPPP is quietest, FFFFF is loudest
        dynamic_order = list(Dynamic)
        return max(
            self.parameter_dynamics.items(),
            key=lambda x: dynamic_order.index(x[1]),
        )[0]


@dataclass
class SensitivityScore:
    """
    The complete UQ sensitivity analysis encoded as a musical score.

    Reading this score tells you everything the pipeline discovered:
    - Which parameters matter (loud voices)
    - How they interact (consonance/dissonance)
    - Where uncertainty comes from (registral balance)
    - How sensitivity changes across the cell cycle (dynamics per beat)

    This is the "program notes + score" for the UQ analysis.
    """

    # Ensemble: one voice per parameter
    voices: list[ParameterVoice]

    # Variance voicing: the registral balance
    variance_voicing: VarianceVoicing

    # Temporal structure
    tempo: Tempo
    tempo_bpm: int
    cell_cycle_time: float  # seconds
    n_beats: int  # cell cycle bins = beats per measure

    # Per-beat dynamics (optional, from per-stage Sobol)
    beat_sensitivities: list[BeatSensitivity] = field(default_factory=list)

    # Metadata
    output_name: str = ""  # Which output observable this score describes

    def soloists(self) -> list[ParameterVoice]:
        """Voices with S_T >= 0.1 (forte or louder)."""
        return [v for v in self.voices if v.total_order_sobol >= 0.1]

    def ensemble_players(self) -> list[ParameterVoice]:
        """Voices with 0.01 <= S_T < 0.1."""
        return [v for v in self.voices if 0.01 <= v.total_order_sobol < 0.1]

    def resting(self) -> list[ParameterVoice]:
        """Voices with S_T < 0.01 (negligible)."""
        return [v for v in self.voices if v.total_order_sobol < 0.01]

    def to_ascii(self) -> str:
        """Render the score as ASCII art."""
        lines = []
        tempo_str = f"{self.tempo.value} ({self.tempo_bpm} BPM)"
        cycle_str = f"{self.cell_cycle_time:.0f}s"

        lines.append("=" * 72)
        lines.append("  SENSITIVITY SCORE")
        if self.output_name:
            lines.append(f'  Output: "{self.output_name}"')
        lines.append(f"  Tempo: {tempo_str}  |  Cell cycle: {cycle_str}")
        lines.append(f"  Time signature: {self.n_beats}/4  (one measure = one cell cycle)")
        lines.append("=" * 72)

        # Variance voicing
        lines.append("")
        lines.append("  VOICING (variance decomposition):")
        lines.append(
            f"    Bass (generation):   {'█' * _bar(self.variance_voicing.generation_fraction)} "
            f"{self.variance_voicing.generation_fraction:.0%}"
        )
        lines.append(
            f"    Tenor (seed):        {'█' * _bar(self.variance_voicing.seed_fraction)} "
            f"{self.variance_voicing.seed_fraction:.0%}"
        )
        lines.append(
            f"    Treble (cell cycle): {'█' * _bar(self.variance_voicing.residual_fraction)} "
            f"{self.variance_voicing.residual_fraction:.0%}"
        )
        lines.append(f"    → {self.variance_voicing.balance_description}")

        # Voices (sorted by S_T, loudest first)
        sorted_voices = sorted(self.voices, key=lambda v: -v.total_order_sobol)

        lines.append("")
        lines.append("  ENSEMBLE:")
        lines.append("  " + "-" * 68)

        for v in sorted_voices:
            if v.total_order_sobol < 0.001:
                lines.append(f"    {v.instrument.value:12s}  —  (tacet)  {v.name}")
            else:
                bar = "█" * _bar(v.total_order_sobol)
                interaction_marker = {
                    InteractionQuality.UNISON: "  ",
                    InteractionQuality.CONSONANT: " ~",
                    InteractionQuality.DISSONANT: " #",
                    InteractionQuality.CLUSTER: "##",
                }[v.interaction_quality]
                lines.append(
                    f"    {v.instrument.value:12s}  {v.dynamic.value:5s}  "
                    f"{bar:20s} {v.total_order_sobol:.3f}{interaction_marker}  {v.name}"
                )

        # Per-beat dynamics (if available)
        if self.beat_sensitivities:
            lines.append("")
            lines.append("  ACROSS THE MEASURE (cell cycle dynamics):")
            lines.append("  " + "-" * 68)

            # Header
            param_names = list(self.beat_sensitivities[0].parameter_dynamics.keys())
            header = "    Beat  "
            for name in param_names:
                short = name[:8]
                header += f" {short:>8s}"
            lines.append(header)

            for beat in self.beat_sensitivities:
                row = f"    {beat.stage_label:6s}"
                for name in param_names:
                    dyn = beat.parameter_dynamics.get(name, Dynamic.PPP)
                    row += f" {dyn.value:>8s}"

                loudest = beat.loudest_voice()
                if loudest:
                    row += f"  ← {loudest}"
                lines.append(row)

        lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)

    def read_aloud(self) -> str:
        """
        Generate natural-language "program notes" for this score.

        This is the explanation a conductor would give the audience before
        the performance: what to listen for, what matters, what's interesting.
        """
        paragraphs = []

        # Opening
        output_desc = f' for "{self.output_name}"' if self.output_name else ""
        paragraphs.append(
            f"This score describes the sensitivity landscape{output_desc}. "
            f"The cell divides every {self.cell_cycle_time:.0f} seconds "
            f"(tempo: {self.tempo.value}, {self.tempo_bpm} BPM), "
            f"and one full measure spans one cell cycle divided into "
            f"{self.n_beats} beats."
        )

        # Soloists
        soloists = self.soloists()
        if soloists:
            solo_parts = []
            for v in sorted(soloists, key=lambda x: -x.total_order_sobol):
                pct = f"{v.total_order_sobol:.0%}"
                solo_parts.append(f"{v.name} ({v.instrument.value}, {v.dynamic.value}, {pct} of variance)")
            paragraphs.append(
                f"The dominant voices are: {'; '.join(solo_parts)}. "
                f"These parameters explain the majority of output variability."
            )
        else:
            paragraphs.append(
                "No single parameter dominates — the output is shaped by many small, roughly equal contributions."
            )

        # Interactions
        dissonant = [
            v
            for v in self.voices
            if v.interaction_quality in (InteractionQuality.DISSONANT, InteractionQuality.CLUSTER)
        ]
        if dissonant:
            names = [v.name for v in dissonant]
            paragraphs.append(
                f"Notable interactions (dissonance): {', '.join(names)}. "
                f"These parameters' effects depend strongly on other parameters' "
                f"values — their solo contribution (S_i) is much smaller than "
                f"their total contribution (S_T). You cannot understand them in isolation."
            )

        # Variance voicing
        vv = self.variance_voicing
        paragraphs.append(
            f"The uncertainty is {vv.balance_description}. "
            f"Generation effects account for {vv.generation_fraction:.0%} "
            f"(bass register — the slow convergence toward steady state), "
            f"lineage seed stochasticity accounts for {vv.seed_fraction:.0%} "
            f"(tenor register — different random realizations), and the residual "
            f"{vv.residual_fraction:.0%} is attributable to cell-cycle-scale dynamics "
            f"(treble register — variation within a single cell's lifetime)."
        )

        # Per-beat narrative
        if self.beat_sensitivities:
            # Find if any parameter's dynamic changes significantly across beats
            param_names = list(self.beat_sensitivities[0].parameter_dynamics.keys())
            dynamic_order = list(Dynamic)

            for param in param_names:
                dynamics_across_beats = [
                    dynamic_order.index(b.parameter_dynamics.get(param, Dynamic.PPP)) for b in self.beat_sensitivities
                ]
                if max(dynamics_across_beats) - min(dynamics_across_beats) >= 3:
                    loud_beats = [
                        self.beat_sensitivities[i].stage_label
                        for i, d in enumerate(dynamics_across_beats)
                        if d >= max(dynamics_across_beats) - 1
                    ]
                    quiet_beats = [
                        self.beat_sensitivities[i].stage_label
                        for i, d in enumerate(dynamics_across_beats)
                        if d <= min(dynamics_across_beats) + 1
                    ]
                    paragraphs.append(
                        f"{param} varies across the cell cycle: "
                        f"loudest at {', '.join(loud_beats)} and "
                        f"quietest at {', '.join(quiet_beats)}. "
                        f"This means its influence on the output is "
                        f"cell-cycle-stage-dependent."
                    )

        return "\n\n".join(paragraphs)


# ═══════════════════════════════════════════════════════════════════════════════
# Encoding Functions
# ═══════════════════════════════════════════════════════════════════════════════


def _sobol_to_dynamic(s_t: float) -> Dynamic:
    """
    Map total-order Sobol index to dynamic marking.

    The mapping is designed so that:
    - S_T < 0.01 → ppp (barely audible, negligible parameter)
    - S_T ~ 0.05 → mp (present but not dominant)
    - S_T ~ 0.20 → f (clearly important)
    - S_T > 0.50 → fff (dominant parameter)
    """
    if s_t >= 0.70:
        return Dynamic.FFFFF
    elif s_t >= 0.50:
        return Dynamic.FFFF
    elif s_t >= 0.35:
        return Dynamic.FFF
    elif s_t >= 0.20:
        return Dynamic.FF
    elif s_t >= 0.10:
        return Dynamic.F
    elif s_t >= 0.05:
        return Dynamic.MF
    elif s_t >= 0.03:
        return Dynamic.MP
    elif s_t >= 0.01:
        return Dynamic.P
    elif s_t >= 0.005:
        return Dynamic.PP
    elif s_t >= 0.001:
        return Dynamic.PPP
    else:
        return Dynamic.PPPPP


def _interaction_quality(s_i: float, s_t: float) -> InteractionQuality:
    """Classify interaction strength from first-order vs total-order gap."""
    if s_t < 1e-10:
        return InteractionQuality.UNISON
    strength = (s_t - s_i) / s_t
    if strength < 0.05:
        return InteractionQuality.UNISON
    elif strength < 0.20:
        return InteractionQuality.CONSONANT
    elif strength < 0.50:
        return InteractionQuality.DISSONANT
    else:
        return InteractionQuality.CLUSTER


_INSTRUMENT_RANKING = [
    Instrument.VIOLIN_1,
    Instrument.CELLO,
    Instrument.VIOLIN_2,
    Instrument.VIOLA,
    Instrument.OBOE,
    Instrument.FLUTE,
    Instrument.CLARINET,
    Instrument.BASSOON,
    Instrument.HORN,
    Instrument.TRUMPET,
]


def _assign_instrument(rank: int, s_t: float) -> Instrument:
    """Assign instrument based on parameter importance rank."""
    if s_t < 0.001:
        return Instrument.REST
    if rank < len(_INSTRUMENT_RANKING):
        return _INSTRUMENT_RANKING[rank]
    return Instrument.REST


def _bar(fraction: float, max_width: int = 20) -> int:
    """Convert a fraction [0, 1] to a bar width for ASCII display."""
    return max(0, min(max_width, round(fraction * max_width)))


def encode_sensitivity(
    sobol_first_order: np.ndarray,
    sobol_total_order: np.ndarray,
    parameter_names: list[str],
    variance_decomposition: dict[str, np.ndarray],
    cell_cycle_time: float = 3600.0,
    n_cell_cycle_bins: int = 10,
    per_stage_sobol: Optional[list[dict]] = None,
    morris_mu_star: Optional[np.ndarray] = None,
    morris_selected: Optional[list[str]] = None,
    output_name: str = "",
) -> SensitivityScore:
    """
    Encode UQ pipeline sensitivity results as a musical score.

    This is the main entry point. Takes the raw numerical outputs of the
    RFC006 pipeline and returns a SensitivityScore that can be rendered
    as ASCII, read aloud as natural language, or (future) exported as MusicXML.

    Args:
        sobol_first_order: First-order Sobol indices, shape (n_params,)
        sobol_total_order: Total-order Sobol indices, shape (n_params,)
        parameter_names: Names of the input parameters
        variance_decomposition: Dict with 'generation_fraction', 'seed_fraction' keys
            (values are arrays; we take the mean across observables)
        cell_cycle_time: Cell cycle time in seconds
        n_cell_cycle_bins: Number of cell cycle bins (= beats per measure)
        per_stage_sobol: Optional list of dicts with per-stage Sobol indices,
            each dict has 'stage': int, 'sobol': SobolIndices
        morris_mu_star: Optional Morris mu_star values for each parameter
        morris_selected: Optional list of parameter names that passed screening
        output_name: Name of the output observable this score describes

    Returns:
        SensitivityScore ready for rendering
    """
    n_params = len(parameter_names)

    # Handle multi-output variance decomposition by averaging
    gen_frac = float(np.mean(variance_decomposition.get("generation_fraction", np.array([0.33]))))
    seed_frac = float(np.mean(variance_decomposition.get("seed_fraction", np.array([0.33]))))
    residual_frac = max(0.0, 1.0 - gen_frac - seed_frac)

    # Rank parameters by S_T for instrument assignment
    rank_order = np.argsort(-sobol_total_order)

    # Build voices
    voices = []
    for i in range(n_params):
        rank = int(np.where(rank_order == i)[0][0])
        s_i = float(sobol_first_order[i])
        s_t = float(sobol_total_order[i])
        interaction = (s_t - s_i) / s_t if s_t > 1e-10 else 0.0

        voice = ParameterVoice(
            name=parameter_names[i],
            instrument=_assign_instrument(rank, s_t),
            first_order_sobol=s_i,
            total_order_sobol=s_t,
            dynamic=_sobol_to_dynamic(s_t),
            interaction_strength=interaction,
            interaction_quality=_interaction_quality(s_i, s_t),
            mu_star=float(morris_mu_star[i]) if morris_mu_star is not None else None,
            passed_audition=(parameter_names[i] in morris_selected if morris_selected is not None else True),
        )
        voices.append(voice)

    # Tempo from cell cycle time
    tempo, tempo_bpm = cycle_time_to_tempo(cell_cycle_time)

    # Variance voicing
    voicing = VarianceVoicing(
        generation_fraction=gen_frac,
        seed_fraction=seed_frac,
        residual_fraction=residual_frac,
    )

    # Per-beat sensitivities
    beat_sensitivities = []
    if per_stage_sobol is not None:
        for stage_data in per_stage_sobol:
            stage_idx = stage_data["stage"]
            stage_sobol = stage_data["sobol"]

            param_dynamics = {}
            for j, name in enumerate(parameter_names):
                if j < len(stage_sobol.total_order):
                    s_t_stage = float(stage_sobol.total_order[j])
                    param_dynamics[name] = _sobol_to_dynamic(s_t_stage)

            beat = BeatSensitivity(
                stage_index=stage_idx,
                stage_label=f"{stage_idx + 1}/{n_cell_cycle_bins}",
                parameter_dynamics=param_dynamics,
            )
            beat_sensitivities.append(beat)

    return SensitivityScore(
        voices=voices,
        variance_voicing=voicing,
        tempo=tempo,
        tempo_bpm=tempo_bpm,
        cell_cycle_time=cell_cycle_time,
        n_beats=n_cell_cycle_bins,
        beat_sensitivities=beat_sensitivities,
        output_name=output_name,
    )
