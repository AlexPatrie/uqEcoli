import numpy as np


def generate_signal(
    params: np.ndarray,
    param_names: list[str],
    baseline_value: float,
    n_timesteps: int,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Default timeseries generator: exponential growth + damped oscillation + noise.

    Parameter effects (generic interpretation):
    - First parameter: affects growth rate (higher = faster growth)
    - Second parameter: affects oscillation frequency (if present)
    - Third+ parameters: damping effects (reduce growth)

    Customize this function for your specific use case!
    """
    t = np.arange(n_timesteps)
    n_params = len(params)

    # Normalize parameters to [0, 1] for generic effects
    # (In practice, you'd use domain-specific logic)

    # Growth rate: first param drives growth
    growth_rate = 0.002 * params[0] if n_params > 0 else 0.002

    # Oscillation frequency: second param affects frequency
    osc_freq = 0.005 + 0.01 * params[1] if n_params > 1 else 0.01

    # Damping: remaining params contribute to damping
    damping = 0.0
    if n_params > 2:
        for i in range(2, n_params):
            damping += 0.0005 * params[i]

    # Build timeseries components
    trend = baseline_value * np.exp((growth_rate - damping) * t)
    oscillation = 0.15 * baseline_value * np.sin(2 * np.pi * osc_freq * t)
    oscillation *= np.exp(-0.001 * t)  # Damped oscillation

    # Add reproducible noise
    np.random.seed(random_seed)
    noise = 0.03 * baseline_value * np.random.randn(n_timesteps)

    return trend + oscillation + noise
