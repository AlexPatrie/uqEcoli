import numpy as np


def initialize_background(max_x: int = 1111, max_y: int = 1111) -> np.ndarray:
    brightness = np.random.randint(0, 2, size=(max_x, max_y)) * 255
    bg = np.stack([brightness, brightness, brightness], axis=-1)
    return bg


def set_pixels(bg, max_x, max_y):
    pass