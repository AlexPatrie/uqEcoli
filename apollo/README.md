# Apollo: Musical Notation for Cellular Dynamics

Apollo encodes computational biology results as Western musical scores.
It contains two layers, each operating at a different level of abstraction.

---

## Layer 1: Koopman Mode → Note (existing)

Maps individual Koopman spectral modes to individual musical notes.
One mode = one note. The mapping is bijective (lossless round-trip possible).

| Koopman Concept | Musical Concept | Mapping |
|---|---|---|
| Eigenvalue frequency | Pitch (C4, A♭3, …) | Logarithmic: `12 · log₂(f / 440)` → semitones from A4 |
| Growth rate (Re(λ)) | Duration + Articulation | Stable → long notes (whole/half); decaying → staccato |
| Amplitude | Dynamic (pp → ff) | Linear on [0, 1] |
| Cell cycle period | Tempo (BPM) | `60 / T_cycle` |
| Observable class | Clef | Mass → bass, transcriptome → treble, etc. |

**Entry points:** `encode_spectrum()`, `decode_score()`, `score_to_musicxml()`

---

## Layer 2: UQ Pipeline → Score (`uq_score`)

Maps the *entire* RFC006 sensitivity analysis output into a musical score.
This is the new module. Instead of encoding raw spectra, it encodes
what the UQ pipeline *discovered about the system*: which parameters
matter, how they interact, where uncertainty comes from, and how
sensitivity changes across the cell cycle.

### The core idea

PCE coefficients are a spectral decomposition — they expand the model
response in an orthogonal polynomial basis. Sobol indices partition
total output variance into per-parameter contributions, exactly as a
Fourier power spectrum partitions total signal energy into per-frequency
contributions. The cell cycle variable IS a fundamental frequency (via
Koopman). This is structural correspondence, not metaphor.

### The mapping

Not every row here is equally deep. Some are genuine mathematical
isomorphisms (the same operation in a different notation system); others
are useful structural analogies but not identical problems. The
"Structural basis" column is honest about which is which.

| UQ Pipeline Concept | Musical Concept | Structural basis |
|---|---|---|
| **Sobol index (variance fraction)** | Dynamic marking (pp → ff) | **Isomorphic.** Both are energy partition problems. Sobol indices decompose total output variance into per-source fractions: `Var(Y) = Σ V_i + Σ V_ij + …`. A Fourier power spectrum decomposes total signal energy into per-frequency fractions: `‖x‖² = Σ |X_k|²`. Dynamic markings encode amplitude = energy contribution of one voice to the total sound. In both cases you are asking "what fraction of the total energy does this component carry?" — the math is Parseval's identity in different basis sets. |
| **PCE coefficient vector** | The score itself | **Isomorphic.** A PCE surrogate expands the model response in an orthogonal polynomial basis: `f(x) ≈ Σ cₐ Ψₐ(x)`. A musical score is a symbolic representation of a signal in a basis of pitched, timed events. Both are compact symbolic encodings that allow exact reconstruction ("replay") without re-running the generative process (simulation / orchestra). The relationship is: basis expansion → symbolic notation → reconstruction. |
| **Cell cycle period** | Measure / Tempo | **Isomorphic.** The cell cycle is a biological oscillator with a measurable fundamental period T. A musical measure is a temporal container defined by a period (bar length = beats × beat duration). Both are periodic time frames that impose structure on events within them. Tempo (BPM = 60/T) is literally the reciprocal of the period — the same quantity in inverse units. The Koopman eigenfunction whose phase advances by 2π per division *is* the fundamental frequency. |
| **Cell cycle stages (bins)** | Beats within a measure | **Isomorphic.** Discretizing one period of a periodic signal into N equal phase bins is the same operation whether the signal is a cell cycle or a musical measure. Bin k/N corresponds to phase 2πk/N. In music, beat k of an N-beat measure is at phase 2πk/N of the bar. Both are uniform phase quantizations of a single period. |
| **Variance decomposition (generation / seed / residual)** | Registral balance (bass / tenor / treble) | **Structural analogy, not isomorphism.** The correspondence is between timescale and frequency register: generation variance is slow (low-frequency, many cell cycles to converge), seed variance is intermediate, residual/cell-cycle variance is fast (within one cycle). In music, bass = low frequency = slow oscillation, treble = high frequency = fast oscillation. The mapping leverages the universal association between timescale and frequency, but the variance components are not literally frequency bands — they are ANOVA-style decompositions across grouping factors. |
| **S_T − S_i (interaction gap)** | Consonance / dissonance | **Structural analogy.** When S_T ≈ S_i, the parameter acts independently — its effect is the same regardless of other parameters' values. This is analogous to unison or simple consonance: voices that sound the same whether played alone or together. When S_T ≫ S_i, the parameter's effect is dominated by interactions — it sounds fundamentally different in combination. Dissonance in music also arises from interference between simultaneous frequencies. Both describe the degree to which components are independent vs coupled, but the underlying mechanisms differ (variance algebra vs acoustic beating). |
| **Input parameter** | Voice / Instrument | **Structural analogy.** Both are independent degrees of freedom whose contributions superpose to produce the observed output. In UQ, the model output is a function of N input parameters. In an ensemble, the sound is a superposition of N voices. The analogy holds because linear superposition of variance (Sobol) mirrors linear superposition of sound energy. It breaks down for strongly nonlinear interactions (where UQ "superposition" is only approximate via ANOVA decomposition). |
| **Morris screening** | Audition / rehearsal cut | **Functional analogy.** Both are cheap, coarse evaluations used to reduce the ensemble before an expensive main analysis. Morris uses O(N) model evaluations to rank parameters; an audition uses brief performances to select players. The function is identical (screen → rank → cut) but the underlying processes are unrelated. |
| **Aggregation strategy** | Listening perspective | **Functional analogy.** Choosing to aggregate by generation vs by seed vs uniformly is choosing which axis of variation to collapse — analogous to choosing to listen to the full ensemble blend vs isolating a section vs following one voice across movements. Both are choices about which dimension of a multi-dimensional signal to marginalize over, but "listening perspective" is a human metaphor, not a mathematical operation with the same formalism. |

### Reading a score

A rendered score tells you:

1. **Voicing** — bar chart of variance decomposition. If bass-heavy,
   most uncertainty comes from generational convergence effects. If
   treble-heavy, cell-cycle-scale dynamics dominate.

2. **Ensemble** — each parameter listed with its instrument assignment,
   dynamic marking, Sobol index, and interaction quality marker:
   - `(space)` = unison (pure solo effect)
   - `~` = consonant (mild interactions)
   - `#` = dissonant (moderate coupling)
   - `##` = cluster (interaction-dominated)

3. **Across the measure** — if per-stage Sobol indices are available,
   a grid showing how each parameter's dynamic changes beat by beat
   through the cell cycle. A parameter that is `ff` during the C period
   but `pp` during the B period has cell-cycle-stage-dependent influence.

### Example output

```
========================================================================
  SENSITIVITY SCORE
  Output: "listeners__mass__dry_mass"
  Tempo: Largo (17 BPM)  |  Cell cycle: 3600s
  Time signature: 10/4  (one measure = one cell cycle)
========================================================================

  VOICING (variance decomposition):
    Bass (generation):   ████████████         60%
    Tenor (seed):        ██████               30%
    Treble (cell cycle): ██                   10%
    → dominated by bass register

  ENSEMBLE:
  --------------------------------------------------------------------
    Violin I       fff   ████████████████     0.420    vio_expression
    Cello          f     ██████               0.150 #  mecA_kcat
    Violin II      mf    ████                 0.080 ~  ppGpp_hill_n
    Viola          mp    ██                   0.035    ftsZ_threshold
    ...
========================================================================
```

### Usage

```python
from apollo.uq_score import encode_sensitivity

score = encode_sensitivity(
    sobol_first_order=sobol.first_order,
    sobol_total_order=sobol.total_order,
    parameter_names=["vio_expression", "mecA_kcat", "ppGpp_hill_n", ...],
    variance_decomposition={
        "generation_fraction": gen_frac_array,
        "seed_fraction": seed_frac_array,
    },
    cell_cycle_time=3600.0,
    n_cell_cycle_bins=10,
    per_stage_sobol=per_stage_data,  # optional
    morris_mu_star=morris.mu_star,   # optional
    morris_selected=top_k_params,    # optional
    output_name="listeners__mass__dry_mass",
)

# ASCII rendering
print(score.to_ascii())

# Natural language "program notes"
print(score.read_aloud())

# Inspect programmatically
for voice in score.soloists():
    print(f"{voice.name}: {voice.dynamic.value}, interactions={voice.interaction_quality.value}")
```

### Key types

| Type | Role |
|---|---|
| `SensitivityScore` | Complete encoded score — the top-level container |
| `ParameterVoice` | One parameter as one voice: instrument, dynamic, interaction quality |
| `VarianceVoicing` | Registral distribution of variance (bass/tenor/treble fractions) |
| `BeatSensitivity` | Per-cell-cycle-stage dynamics for each parameter |
| `Instrument` | Enum: Violin I, Cello, Viola, … , Rest (by importance rank) |
| `InteractionQuality` | Enum: unison, consonant, dissonant, cluster |
| `VarianceRegister` | Enum: bass (generation), tenor (seed), treble (residual) |

---

## How the two layers connect

Layer 1 operates on raw Koopman spectra — the eigenvalues themselves.
Layer 2 operates on what the UQ pipeline learns *about* the system by
running sensitivity analysis on top of those simulations.

```
Simulation outputs
    │
    ├── Koopman DMD ──→ eigenvalues ──→ Layer 1 (note-level encoding)
    │
    └── UQ Pipeline ──→ Sobol indices ──→ Layer 2 (score-level encoding)
                        variance decomp
                        cell cycle stages
```

Both layers share the same `apollo.types` and `apollo.mappings`
primitives (Dynamic, Tempo, cycle_time_to_tempo, etc.), ensuring
consistent musical semantics across levels.

---

## Toward Full Isomorphism

Four of the nine mappings in the table above are currently isomorphic
(Sobol/dynamics, PCE/score, cell cycle period/tempo, cell cycle
bins/beats). The remaining five are structural or functional analogies.
This section explores whether a richer musical formalism could promote
each non-isomorphic row to a true isomorphism — and is honest about
where that's feasible vs. where the correspondence is irreducibly
metaphorical.

### Variance decomposition → Registral balance

**Current status:** Structural analogy. Generation variance is "slow"
(bass), seed variance is "intermediate" (tenor), residual is "fast"
(treble). But the ANOVA fractions are not literally frequency bands.

**Path to isomorphism:** The variance components *are* separated by
timescale. Generation effects unfold over many cell cycles (low
frequency), seed effects are fixed at initialization and visible
across the full trajectory (DC-to-low), and residual/cell-cycle
variance operates within one period (fundamental frequency and above).
If we define registral energy as the power spectral density integrated
over frequency bands — bass: [0, 1/(2T_gen)], tenor: [1/(2T_gen),
1/(2T_cycle)], treble: [1/(2T_cycle), Nyquist] — then the variance
fractions *become* frequency-band energy fractions, and the mapping is
literally a sub-band energy decomposition. This requires:

1. A spectral representation of the model output (available via
   Koopman DMD or FFT of the aggregated time series)
2. Defining the band boundaries from the aggregation group sizes
   (number of generations, number of seeds, cell cycle period)
3. Computing band energy as `∫ S(f) df` over each band

With this formulation, the ANOVA variance fractions would approximate
the sub-band energy ratios (they are related through Wiener-Khinchin),
and the musical register assignment becomes a genuine frequency-domain
decomposition — the same math as an audio equalizer applied to the
model output's power spectrum.

**Verdict:** Achievable. The key insight is that the ANOVA grouping
factors (generation, seed) implicitly define timescale boundaries.
Formalizing those boundaries as frequency cutoffs makes the mapping
invertible.

### S_T − S_i (interaction gap) → Consonance / dissonance

**Current status:** Structural analogy. Both describe "independence vs.
coupling" between components, but Sobol interactions are variance
ratios while acoustic consonance is determined by frequency ratios.

**Path to isomorphism:** Consider the second-order Sobol indices
S_ij, which quantify the pairwise interaction between parameters i
and j. In music, the consonance of an interval between two notes is
determined by the simplicity of their frequency ratio (octave = 2:1,
fifth = 3:2, tritone = 45:32). Could S_ij be mapped to interval
quality?

The problem: Sobol interactions live in [0, 1] and are symmetric
(S_ij = S_ji). Musical intervals are determined by *frequency ratios*,
not by coupling strength. There is no natural frequency associated
with a parameter's Sobol index — the Sobol decomposition is in
parameter space, not frequency space.

However, if the PCE basis functions Ψₐ(x) are viewed as "frequencies"
in the polynomial spectral expansion, then a multi-index α that
involves parameters i and j simultaneously *is* a "combination
frequency" analogous to a musical interval. The PCE coefficient c_α
for a mixed multi-index is the amplitude of that "interaction
harmonic." The ratio of the individual basis function degrees could
define an interval quality:

- α = (1, 1, 0, ...): both parameters at degree 1 → unison (same
  "frequency") → consonant
- α = (2, 1, 0, ...): degrees 2 and 1 → octave-like ratio → consonant
- α = (3, 2, 0, ...): degrees 3 and 2 → fifth-like ratio → consonant
- α = (5, 3, 0, ...): more complex ratio → dissonant

This is speculative but mathematically grounded: the PCE multi-index
structure does define a lattice of "interaction harmonics" where
the degree ratios play a role analogous to frequency ratios. Whether
this produces musically meaningful intervals depends on the specific
PCE expansion, and it would require a custom consonance metric based
on multi-index structure rather than acoustic frequency ratios.

**Verdict:** Partially achievable. The multi-index interpretation
provides genuine structure, but the resulting "consonance" metric
would be a PCE-specific construct, not identical to acoustic
consonance. It's a deeper analogy than the current S_T − S_i gap,
but not a full isomorphism.

### Input parameter → Voice / Instrument

**Current status:** Structural analogy. Parameters superpose via
ANOVA decomposition; voices superpose via acoustic addition.

**Path to isomorphism:** In the PCE expansion
`f(x) = Σ cₐ Ψₐ(x)`, each parameter x_i has an associated set of
univariate basis functions {Ψ_k(x_i)}. These are literally spectral
components — Legendre or Hermite polynomials are orthogonal bases,
just as sinusoidal harmonics are. A parameter's "voice" in the PCE
expansion is the set of all terms where x_i appears:

```
Voice_i = Σ_{α: α_i > 0} c_α Ψ_α(x)
```

This voice has a spectral content (the distribution of energy across
polynomial degrees), a fundamental "pitch" (degree-1 term), harmonics
(higher-degree terms), and an amplitude profile (the coefficients).
If each parameter's voice is rendered as a time-domain signal by
evaluating the basis functions along a parameter sweep, then the
voices literally *are* audio-like signals that superpose to form the
total model response.

The instrument assignment (violin, cello, etc.) could then be based
on the spectral content of the voice: voices dominated by low-degree
terms (smooth, broad effects) get bass instruments; voices with
significant high-degree terms (sharp, localized effects) get treble
instruments.

**Verdict:** Achievable with the PCE spectral interpretation. The
mapping becomes: parameter → set of PCE basis functions → spectral
voice → instrument assignment by spectral content. This is genuine
signal processing applied to the polynomial expansion.

### Morris screening → Audition

**Current status:** Functional analogy. Same workflow pattern
(cheap screen → rank → cut) but no shared mathematical object.

**Path to isomorphism:** Morris elementary effects are finite
differences: EE_i = [f(x + Δe_i) − f(x)] / Δ. An audition is a
brief performance evaluated by a judge. The mathematical object is
the same in both cases — a low-resolution sample of the full signal
used for binary classification (keep/cut).

But the evaluation criterion differs fundamentally: Morris ranks by
|μ*| (mean absolute effect), while auditions evaluate by subjective
quality against a standard. There is no natural musical quantity
that corresponds to "mean absolute elementary effect."

One could define a musical audition metric: have each parameter
"play" its PCE voice (as defined above) in isolation, and measure
the RMS amplitude. Parameters below a threshold are "cut." This
would make the audition a literal amplitude screening — but it
would not be Morris screening; it would be a different, simpler
screening method (just checking first-order variance contribution).

**Verdict:** Not achievable as isomorphism. The correspondence is
irreducibly procedural. Morris screening's specific algorithm
(OAT perturbation along random trajectories) has no musical
counterpart. The best we can do is a parallel workflow pattern.

### Aggregation strategy → Listening perspective

**Current status:** Functional analogy. Both are choices about which
dimension to marginalize over.

**Path to isomorphism:** Aggregation strategies project the full
(cell × generation × seed × time) data tensor onto lower-dimensional
summaries by averaging over selected axes. In audio signal processing,
the equivalent operation is spatial or channel mixing: stereo-to-mono
(marginalize over channels), time averaging (marginalize over time
windows), or spectral averaging (marginalize over frequency bands).

If the simulation output is represented as a multi-dimensional signal
S(cell, generation, seed, time), then:

- UNIFORM = average over all dimensions → mono mix (tutti)
- BY_GENERATION = average within generations → one track per movement
- BY_LINEAGE_SEED = average within seeds → one track per performer
- BY_CELL_CYCLE = average within cycle stages → one track per beat

These are literally different mixing strategies applied to the same
multi-track recording. In a DAW (digital audio workstation), the
equivalent operations are bus routing and submixing. The
mathematical operation is the same: conditional expectation
E[S | group], which is a projection in L² function space.

**Verdict:** Achievable if we commit to the multi-track signal
interpretation. The aggregation strategies become mixing/routing
decisions on a multi-dimensional recording, and the conditional
expectation is the same operator in both domains. The barrier to
full isomorphism is that standard musical notation doesn't have
explicit notation for mixing operations — it's implicit in the
orchestration. A richer formalism (DAW project file, or spatial
audio metadata) would make this fully invertible.

### Summary

| Row | Current | Could become isomorphic? | Required formalism |
|-----|---------|--------------------------|-------------------|
| Variance decomposition → Register | Structural analogy | Yes | Sub-band energy decomposition of model output PSD |
| Interaction gap → Consonance | Structural analogy | Partially | PCE multi-index degree ratios as interval quality |
| Parameter → Voice | Structural analogy | Yes | PCE per-parameter spectral voice extraction |
| Morris → Audition | Functional analogy | No | Irreducibly procedural parallel |
| Aggregation → Listening perspective | Functional analogy | Yes | Multi-track mixing / conditional expectation in L² |

Three of five non-isomorphic rows can be promoted to genuine
isomorphisms by adopting a richer formalism (sub-band energy,
PCE spectral voices, multi-track mixing). One (interaction/consonance)
can be deepened but not fully bridged. One (Morris/audition) is
irreducibly a workflow parallel.

The practical implication: if these formalisms were implemented, the
musical score would carry enough information to recover 7 of 9
pipeline quantities exactly, with the remaining 2 (interaction
quality and screening decisions) being lossy projections. That's
a substantially more informative encoding than standard sensitivity
analysis tables or bar charts — and it leverages the human auditory
system's evolved capacity for parsing complex multi-voice signals.
