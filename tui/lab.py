import marimo

__generated_with = "0.19.4"
app = marimo.App(width="full")


@app.cell
def _():
    import dotenv 
    from huggingface_hub import login

    dotenv.load_dotenv("assets/dev/config/.dev_env")
    return


@app.cell
def _():
    """
    1. Sonification for Real-Time Monitoring
    Concept: Convert simulation state variables to audio streams during execution, allowing researchers to "hear" the cell's behavior.
    Applications:

    Map metabolite concentrations to pitch, allowing you to detect metabolic oscillations or steady-state transitions by ear
    Encode gene expression levels as timbres—each gene family gets an instrument, transcription events trigger notes
    Use stereo/spatial audio to represent compartmentalization (cytoplasm left, periplasm right, membrane center)

    Why it's useful: Human auditory perception excels at detecting patterns, anomalies, and periodicity that might be missed in visual dashboards. You could literally hear a simulation going wrong.
    """

    import numpy as np
    from scipy.io import wavfile

    def sonify_metabolite_timeseries(concentrations: np.ndarray, 
                                      sample_rate: int = 44100,
                                      base_freq: float = 220.0):
        """
        Convert metabolite concentration timeseries to audio.
        """
        # Normalize to 0-1 range
        normalized = (concentrations - concentrations.min()) / (concentrations.max() - concentrations.min() + 1e-9)
    
        # Map to frequency range (220 Hz to 880 Hz - two octaves)
        frequencies = base_freq * (2 ** (normalized * 2))
    
        # Generate audio samples
        duration_per_step = 0.05  # 50ms per simulation step
        audio = []
    
        for freq in frequencies:
            t = np.linspace(0, duration_per_step, int(sample_rate * duration_per_step))
            # Use sine wave with harmonics for richer sound
            wave = (0.6 * np.sin(2 * np.pi * freq * t) +
                    0.3 * np.sin(4 * np.pi * freq * t) +
                    0.1 * np.sin(6 * np.pi * freq * t))
            audio.extend(wave)
    
        return np.array(audio)


    def sonify_multiple_channels(data_dict: dict, sample_rate: int = 44100):
        """
        Create multi-channel sonification for different molecular species.
        """
        channels = []
    
        # Assign different base frequencies to different species
        base_frequencies = {
            'atp': 220,      # A3 - low for energy currency
            'glucose': 330,  # E4
            'lactate': 440,  # A4
            'nadh': 554,     # C#5
        }
    
        for species, concentrations in data_dict.items():
            base_freq = base_frequencies.get(species, 440)
            channel = sonify_metabolite_timeseries(concentrations, sample_rate, base_freq)
            channels.append(channel)
    
        # Mix channels
        min_len = min(len(c) for c in channels)
        mixed = sum(c[:min_len] for c in channels) / len(channels)
    
        return (mixed * 32767).astype(np.int16)
    return (np,)


@app.cell
def _(np):
    """
    3. Audio Compression Techniques for State Snapshots
    Concept: Adapt audio codecs to compress simulation state efficiently.
    Applications:

    Use delta encoding (like FLAC) for incremental state storage—most variables change slowly
    Apply perceptual coding concepts: compress "unimportant" state variables more aggressively
    Use predictive coding (like ADPCM) to store only prediction residuals
    """
    from typing import List, Tuple

    class DeltaStateCompressor:
        """
        Audio-inspired delta encoding for WCM state compression.
        """
    
        def __init__(self, quantization_bits: int = 16):
            self.quantization_bits = quantization_bits
            self.previous_state = None
            self.scale_factors = {}
    
        def compress_state(self, state: dict) -> bytes:
            """
            Compress state using delta encoding with adaptive quantization.
            """
            compressed_data = []
        
            for key, value in state.items():
                arr = np.asarray(value, dtype=np.float64)
            
                if self.previous_state is not None and key in self.previous_state:
                    # Delta encode
                    prev = np.asarray(self.previous_state[key], dtype=np.float64)
                    delta = arr - prev
                
                    # Adaptive quantization based on delta range
                    delta_range = np.abs(delta).max() + 1e-10
                    self.scale_factors[key] = delta_range
                
                    # Quantize deltas
                    quantized = np.round(delta / delta_range * (2**(self.quantization_bits-1))).astype(np.int16)
                    compressed_data.append((key, 'delta', quantized.tobytes(), delta_range))
                else:
                    # Full state for first frame
                    compressed_data.append((key, 'full', arr.tobytes(), arr.dtype))
        
            self.previous_state = state.copy()
            return compressed_data
    
        def compute_compression_ratio(self, original_state: dict, compressed: list) -> float:
            """Estimate compression ratio."""
            original_size = sum(np.asarray(v).nbytes for v in original_state.values())
            compressed_size = sum(len(item[2]) for item in compressed)
            return original_size / compressed_size


    class PredictiveStateEncoder:
        """
        Linear predictive coding for simulation state.
        Similar to how speech codecs predict the next sample.
        """
    
        def __init__(self, order: int = 4):
            self.order = order
            self.history = []
            self.coefficients = {}
    
        def fit_predictor(self, key: str, history: np.ndarray):
            """Fit LPC coefficients from recent history."""
            if len(history) < self.order + 1:
                return None
        
            # Autocorrelation method for LPC
            from scipy.linalg import toeplitz, solve_toeplitz
        
            r = np.correlate(history, history, mode='full')
            r = r[len(r)//2:]  # Positive lags only
        
            # Levinson-Durbin recursion approximation
            try:
                coeffs = solve_toeplitz(r[:self.order], r[1:self.order+1])
                self.coefficients[key] = coeffs
                return coeffs
            except:
                return None
    
        def predict(self, key: str, recent_values: np.ndarray) -> float:
            """Predict next value using LPC coefficients."""
            if key not in self.coefficients:
                return recent_values[-1] if len(recent_values) > 0 else 0
        
            coeffs = self.coefficients[key]
            prediction = np.dot(coeffs, recent_values[-self.order:][::-1])
            return prediction
    
        def encode_residual(self, actual: float, predicted: float) -> int:
            """Encode prediction residual with adaptive quantization."""
            residual = actual - predicted
            # Mu-law companding (as in telephony)
            mu = 255
            compressed = np.sign(residual) * np.log1p(mu * np.abs(residual)) / np.log1p(mu)
            return int(compressed * 127)
    return


@app.cell
def _(np):
    """
    5. Cross-Correlation for Subsystem Coupling Analysis
    Concept: Use audio-style cross-correlation and coherence analysis to identify how different cellular subsystems influence each other.
    """
    from scipy import signal

    def analyze_subsystem_coupling(signals: dict, dt: float) -> dict:
        """
        Analyze coupling between simulation subsystems using audio correlation techniques.
    
        Args:
            signals: Dict mapping subsystem names to their output timeseries
            dt: Time step
    
        Returns:
            Coupling analysis results
        """
        names = list(signals.keys())
        n_systems = len(names)
    
        results = {
            'cross_correlations': {},
            'time_lags': {},
            'coherence': {},
            'transfer_functions': {}
        }
    
        for i in range(n_systems):
            for j in range(i+1, n_systems):
                name_i, name_j = names[i], names[j]
                sig_i = signals[name_i]
                sig_j = signals[name_j]
            
                # Normalize signals
                sig_i = (sig_i - np.mean(sig_i)) / (np.std(sig_i) + 1e-10)
                sig_j = (sig_j - np.mean(sig_j)) / (np.std(sig_j) + 1e-10)
            
                # Cross-correlation
                correlation = signal.correlate(sig_i, sig_j, mode='full')
                lags = signal.correlation_lags(len(sig_i), len(sig_j), mode='full') * dt
            
                # Find peak correlation and its lag
                peak_idx = np.argmax(np.abs(correlation))
                peak_lag = lags[peak_idx]
                peak_corr = correlation[peak_idx] / len(sig_i)
            
                pair_key = f"{name_i}_{name_j}"
                results['cross_correlations'][pair_key] = peak_corr
                results['time_lags'][pair_key] = peak_lag
            
                # Coherence analysis (frequency-domain correlation)
                f, Cxy = signal.coherence(sig_i, sig_j, fs=1/dt, nperseg=min(256, len(sig_i)//4))
                results['coherence'][pair_key] = {'frequencies': f, 'coherence': Cxy}
            
                # Transfer function estimation (how does i drive j?)
                f, Txy = signal.csd(sig_i, sig_j, fs=1/dt, nperseg=min(256, len(sig_i)//4))
                f, Pxx = signal.welch(sig_i, fs=1/dt, nperseg=min(256, len(sig_i)//4))
                H = Txy / (Pxx + 1e-10)  # Transfer function estimate
                results['transfer_functions'][pair_key] = {'frequencies': f, 'H': H}
    
        return results


    def detect_causal_relationships(signals: dict, dt: float, max_lag: int = 50) -> dict:
        """
        Granger-causality-inspired analysis using cross-correlation directionality.
        """
        names = list(signals.keys())
        causality_matrix = np.zeros((len(names), len(names)))
    
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if i == j:
                    continue
            
                sig_i = signals[name_i]
                sig_j = signals[name_j]
            
                # Does i lead j? (positive lag means i precedes j)
                corr = signal.correlate(sig_j, sig_i, mode='full')
                lags = signal.correlation_lags(len(sig_j), len(sig_i), mode='full')
            
                # Focus on positive lags (i preceding j)
                positive_mask = (lags > 0) & (lags < max_lag)
                if not np.any(positive_mask):
                    continue
                
                max_positive_corr = np.max(np.abs(corr[positive_mask]))
            
                # Compare to negative lags
                negative_mask = (lags < 0) & (lags > -max_lag)
                max_negative_corr = np.max(np.abs(corr[negative_mask])) if np.any(negative_mask) else 0
            
                # Asymmetry suggests directionality
                causality_matrix[i, j] = max_positive_corr - max_negative_corr
    
        return {
            'names': names,
            'causality_matrix': causality_matrix
        }
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
