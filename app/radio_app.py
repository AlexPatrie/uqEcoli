"""
RADIODYNE — Modulation Workbench
Interactive AM/FM transmission simulator with channel modelling and signal quality metrics.
"""

import tkinter as tk
from tkinter import ttk, font, filedialog, messagebox
import numpy as np
import math, time, os, threading, wave, struct

try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

# ═══════════════════════════════════════════════════════════════════
# SIGNAL ENGINE
# ═══════════════════════════════════════════════════════════════════

def generate_source_wave(waveform="sine", frequency=440.0, amplitude=1.0,
                         duration=1.0, sample_rate=8000, phase=0.0,
                         duty_cycle=0.5, harmonics=5):
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    w = 2 * np.pi * frequency
    match waveform:
        case "sine":
            wave = np.sin(w * t + phase)
        case "square":
            wave = sum((1/k)*np.sin(k*w*t+phase) for k in range(1, harmonics*2, 2)) * (4/np.pi)
        case "sawtooth":
            wave = sum(((-1)**(k+1))*(1/k)*np.sin(k*w*t+phase) for k in range(1, harmonics+1)) * (2/np.pi)
        case "triangle":
            wave = sum(((-1)**i)*(1/n2**2)*np.sin(n2*w*t+phase)
                       for i, n2 in enumerate(range(1, harmonics*2, 2))) * (8/np.pi**2)
        case "pulse":
            wave = np.where((t*frequency % 1.0) < duty_cycle, 1.0, -1.0).astype(float)
        case "noise":
            wave = np.random.default_rng().uniform(-1.0, 1.0, n)
        case "chirp":
            f0, f1 = frequency/2, frequency
            wave = np.sin(phase + 2*np.pi*(f0*t + (f1-f0)/(2*duration)*t**2))
        case _:
            wave = np.sin(w * t + phase)
    peak = np.max(np.abs(wave)) or 1.0
    return (amplitude * wave / peak).astype(np.float64)


def amplitude_modulation(source, carrier_freq, sample_rate=8000, duration=1.0, mod_index=1.0):
    t = np.linspace(0, duration, int(sample_rate*duration), endpoint=False)
    carrier = np.cos(2*np.pi*carrier_freq*t)
    src = source / (np.max(np.abs(source)) + 1e-9)
    return (1 + mod_index * src) * carrier


def frequency_modulation(source, carrier_freq, freq_deviation=50.0, sample_rate=8000, duration=1.0):
    t = np.linspace(0, duration, int(sample_rate*duration), endpoint=False)
    src = source / (np.max(np.abs(source)) + 1e-9)
    integrated = np.cumsum(src) / sample_rate
    return np.cos(2*np.pi*carrier_freq*t + 2*np.pi*freq_deviation*integrated)


def apply_channel(signal, distance_km, tx_power_w, freq_mhz, noise_floor_dbm,
                  multipath=False, sample_rate=8000):
    """
    Free-space path loss (Friis) + AWGN + optional 2-ray multipath.
    Returns (rx_signal, snr_db, signal_strength_pct 0-100)
    """
    d_m   = max(distance_km * 1000, 1.0)
    f_hz  = freq_mhz * 1e6
    c     = 3e8
    fspl_db = 20*math.log10(d_m) + 20*math.log10(f_hz) + 20*math.log10(4*math.pi/c)
    tx_dbm  = 10*math.log10(tx_power_w * 1000)
    rx_dbm  = tx_dbm - fspl_db
    snr_db  = rx_dbm - noise_floor_dbm
    snr_lin = 10 ** (snr_db / 10)

    rx = signal.copy() * (10 ** ((rx_dbm - tx_dbm) / 20))

    sig_pwr   = np.mean(rx**2) + 1e-30
    noise_var = sig_pwr / max(snr_lin, 1e-6)
    rx += np.random.normal(0, math.sqrt(noise_var), len(rx))

    if multipath:
        delay = max(1, min(int(sample_rate * d_m / c * 0.1), len(rx)//4))
        echo  = np.zeros_like(rx)
        echo[delay:] = rx[:-delay] * 0.4
        rx += echo

    strength = float(np.clip(snr_db / 40.0 * 100, 0, 100))
    return rx, snr_db, strength, fspl_db, rx_dbm


# ═══════════════════════════════════════════════════════════════════
# DEMODULATION (coherent recovery of baseband from received signal)
# ═══════════════════════════════════════════════════════════════════

def demodulate_am(rx_signal, carrier_freq, sample_rate):
    """
    Envelope detection via analytic signal (Hilbert transform).
    Physically: rectification + low-pass filtering of the AM waveform
    recovers the original baseband modulation envelope.
    """
    # Analytic signal → instantaneous envelope
    N = len(rx_signal)
    spectrum = np.fft.fft(rx_signal)
    h = np.zeros(N)
    h[0] = 1
    h[1:N//2] = 2
    if N % 2 == 0:
        h[N//2] = 1
    analytic = np.fft.ifft(spectrum * h)
    envelope = np.abs(analytic)

    # Remove DC (carrier) component — what remains is the message signal
    envelope -= np.mean(envelope)

    # Low-pass filter: brick-wall at 2× source bandwidth to reject
    # carrier harmonics while preserving the baseband content
    cutoff_bin = int(carrier_freq * 0.8 * N / sample_rate)
    if cutoff_bin > 0 and cutoff_bin < N // 2:
        spec = np.fft.fft(envelope)
        spec[cutoff_bin:N - cutoff_bin] = 0
        envelope = np.real(np.fft.ifft(spec))

    peak = np.max(np.abs(envelope)) or 1.0
    return (envelope / peak).astype(np.float64)


def demodulate_fm(rx_signal, sample_rate):
    """
    FM discriminator: instantaneous frequency via analytic signal.
    Physically: the rate of change of the instantaneous phase
    is proportional to the original message signal.
    d/dt[phase(t)] = 2π·[f_c + k_f·m(t)]  →  m(t) ∝ d(phase)/dt - f_c
    """
    N = len(rx_signal)
    spectrum = np.fft.fft(rx_signal)
    h = np.zeros(N)
    h[0] = 1
    h[1:N//2] = 2
    if N % 2 == 0:
        h[N//2] = 1
    analytic = np.fft.ifft(spectrum * h)

    # Instantaneous phase and its derivative (= instantaneous frequency)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) / (2 * np.pi / sample_rate)

    # Remove carrier offset (DC component of frequency)
    demod = inst_freq - np.mean(inst_freq)
    demod = np.append(demod, demod[-1])  # match length

    peak = np.max(np.abs(demod)) or 1.0
    return (demod / peak).astype(np.float64)


def load_wav_file(path):
    """Load a .wav file, return (samples_float64_mono, sample_rate)."""
    with wave.open(path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        fmt = f"{n_frames * n_channels}B"
        samples = np.array(struct.unpack(fmt, raw), dtype=np.float64) - 128.0
        samples /= 128.0
    elif sampwidth == 2:
        fmt = f"<{n_frames * n_channels}h"
        samples = np.array(struct.unpack(fmt, raw), dtype=np.float64)
        samples /= 32768.0
    elif sampwidth == 3:
        # 24-bit: unpack manually
        samples = np.zeros(n_frames * n_channels, dtype=np.float64)
        for i in range(n_frames * n_channels):
            b = raw[i*3:(i+1)*3]
            val = int.from_bytes(b, byteorder='little', signed=True)
            samples[i] = val / 8388608.0
    elif sampwidth == 4:
        fmt = f"<{n_frames * n_channels}i"
        samples = np.array(struct.unpack(fmt, raw), dtype=np.float64)
        samples /= 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    # Mix to mono
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples, framerate


# ═══════════════════════════════════════════════════════════════════
# PALETTE  —  retro phosphor green + cartoonish pops
# ═══════════════════════════════════════════════════════════════════
BG      = "#050a05"      # deep CRT black-green
BG2     = "#081208"      # panel background
BG3     = "#0e1a0e"      # recessed panel
BORDER  = "#1a5a2a"      # bright green border
TEAL    = "#33ff66"      # primary phosphor green (vivid)
TEAL2   = "#22cc55"      # secondary green
TEAL3   = "#0a3318"      # dim green fill
AMBER   = "#ffcc00"      # warm yellow — cartoonish pop
AMBER2  = "#ff9900"      # orange highlight
DIMTEAL = "#0d2a12"      # dim bar segments
TEXT    = "#88ee88"       # readable green text
TEXTDIM = "#2a6630"      # dim label green
RED     = "#ff4466"       # hot pink-red — cartoonish
GREEN   = "#33ff99"       # bright mint
GREEN2  = "#00ee66"       # signal-good green
MAGENTA = "#ff44ff"       # bonus cartoon accent
CYAN    = "#00ffee"       # bonus cartoon accent
VIOLET  = "#aa66ff"       # bonus cartoon accent


# ═══════════════════════════════════════════════════════════════════
# REUSABLE WIDGETS
# ═══════════════════════════════════════════════════════════════════

class SliderRow(tk.Frame):
    def __init__(self, parent, label, var, lo, hi, res, lw=18, fmt="{:.2f}", **kw):
        super().__init__(parent, bg=BG2, **kw)
        f8b = font.Font(family="Courier", size=8, weight="bold")
        f8  = font.Font(family="Courier", size=8)
        tk.Label(self, text=label, width=lw, anchor="w",
                 font=f8, fg=TEXT, bg=BG2).pack(side="left")
        self.val_lbl = tk.Label(self, text=fmt.format(float(var.get())),
                                width=9, font=f8b, fg=AMBER, bg=BG2)
        self.val_lbl.pack(side="right")
        tk.Scale(self, variable=var, from_=lo, to=hi, resolution=res,
                 orient="horizontal", showvalue=False,
                 bg=BG3, fg=TEAL, troughcolor=TEAL3,
                 activebackground=GREEN, highlightthickness=0,
                 bd=1, sliderrelief="raised", width=14, sliderlength=22,
                 command=lambda v: self.val_lbl.config(text=fmt.format(float(v)))
                 ).pack(side="left", fill="x", expand=True)


class SectionLabel(tk.Frame):
    def __init__(self, parent, title, **kw):
        super().__init__(parent, bg=BG2, **kw)
        tk.Canvas(self, height=2, bg=TEAL2, highlightthickness=0).pack(fill="x")
        tk.Label(self, text=title,
                 font=font.Font(family="Courier", size=8, weight="bold"),
                 fg=GREEN, bg=BG2).pack(anchor="w", pady=(2,0))


class SignalMeter(tk.Frame):
    """Segmented vertical bar signal strength indicator."""
    def __init__(self, parent, width=30, height=110, segs=16):
        super().__init__(parent, bg=BG2)
        self._mw, self._mh, self._segs = width, height, segs
        self._pct = 0.0
        self._c = tk.Canvas(self, width=width, height=height,
                            bg=BG2, highlightthickness=0)
        self._c.pack()
        self._draw()

    def set(self, pct):
        self._pct = max(0.0, min(100.0, float(pct)))
        self._draw()

    def _draw(self):
        self._c.delete("all")
        sh     = max(1, (self._mh - self._segs) // self._segs)
        active = int(self._pct / 100 * self._segs)
        for i in range(self._segs):
            y2 = self._mh - i*(sh+1)
            y1 = y2 - sh
            if i < active:
                ratio = i / self._segs
                col = RED if ratio > 0.75 else (AMBER if ratio > 0.5 else GREEN2)
            else:
                col = DIMTEAL
            self._c.create_rectangle(4, y1, self._mw-4, y2, fill=col, outline="")


class SNRGauge(tk.Frame):
    """Half-arc SNR gauge."""
    def __init__(self, parent, size=100):
        super().__init__(parent, bg=BG2)
        self._size = size
        self._snr  = 0.0
        self._c = tk.Canvas(self, width=size, height=size//2+14,
                            bg=BG2, highlightthickness=0)
        self._c.pack()
        self._draw()

    def set(self, snr_db):
        self._snr = float(snr_db)
        self._draw()

    def _draw(self):
        self._c.delete("all")
        s  = self._size
        cx = s//2
        cy = s//2 + 4
        r  = s//2 - 7
        self._c.create_arc(cx-r, cy-r, cx+r, cy+r,
                           start=0, extent=180, style="arc",
                           outline=DIMTEAL, width=6)
        pct   = float(np.clip(self._snr / 40.0, 0, 1))
        sweep = int(pct * 180)
        col   = GREEN2 if pct > 0.6 else (AMBER if pct > 0.3 else RED)
        if sweep > 0:
            self._c.create_arc(cx-r, cy-r, cx+r, cy+r,
                               start=0, extent=sweep, style="arc",
                               outline=col, width=6)
        self._c.create_text(cx, cy+6,
                            text=f"{self._snr:.1f} dB",
                            fill=GREEN,
                            font=font.Font(family="Courier", size=8, weight="bold"))


# ═══════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════

class RadioApp(tk.Tk):
    SCOPE_W = 920

    def __init__(self):
        super().__init__()
        self.title("RADIODYNE — Modulation Workbench")
        self.configure(bg=BG)
        self.resizable(False, False)

        # ── Signal vars ──────────────────────────────────────────
        self.modulation   = tk.StringVar(value="AM")
        self.waveform     = tk.StringVar(value="sine")
        self.src_freq     = tk.DoubleVar(value=5.0)
        self.carrier_freq = tk.DoubleVar(value=40.0)
        self.amplitude    = tk.DoubleVar(value=1.0)
        self.duty_cycle   = tk.DoubleVar(value=0.5)
        self.harmonics    = tk.IntVar(value=5)
        self.freq_dev     = tk.DoubleVar(value=20.0)
        self.mod_index    = tk.DoubleVar(value=1.0)
        self.sample_rate  = tk.IntVar(value=8000)
        self.duration     = tk.DoubleVar(value=1.0)
        self.phase        = tk.DoubleVar(value=0.0)

        # ── Channel vars ─────────────────────────────────────────
        self.distance_km  = tk.DoubleVar(value=10.0)
        self.tx_power_w   = tk.DoubleVar(value=10.0)
        self.carrier_mhz  = tk.DoubleVar(value=100.0)
        self.noise_floor  = tk.DoubleVar(value=-100.0)
        self.multipath    = tk.BooleanVar(value=False)
        self.distance_km.trace_add("write", self._on_distance_slider)

        # ── Runtime state ────────────────────────────────────────
        self.transmitting   = False
        self.signal_data    = np.zeros(self.SCOPE_W)
        self.rx_data        = np.zeros(self.SCOPE_W)
        self.scroll_offset  = 0
        self._tx_frame      = 0
        self._uploaded_wave = None
        self._uploaded_name = tk.StringVar(value="none")
        self._snr_db        = 0.0
        self._strength_pct  = 0.0
        self._show_rx       = tk.BooleanVar(value=False)

        # ── Audio state ────────────────────────────────────────────
        self._uploaded_sr     = None   # native sample rate of uploaded file
        self._audio_playing   = False
        self._audio_previewing = False
        self._audio_stream    = None
        self._audio_lock      = threading.Lock()
        self._demod_buffer    = np.zeros(0)
        self._demod_read_pos  = 0
        self._audio_loop      = tk.BooleanVar(value=True)
        self._audio_volume    = tk.DoubleVar(value=0.8)

        self._build_ui()
        self._start_idle_animation()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self._stop_audio()
        self.destroy()

    # ─────────────────────────────────────────────────────────────
    # BUILD UI
    # ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = tk.Frame(self, bg=TEAL2, padx=2, pady=2)
        outer.pack(padx=10, pady=10)
        root = tk.Frame(outer, bg=BG2)
        root.pack()
        self._main_frame = root

        self._build_header(root)

        body = tk.Frame(root, bg=BG2)
        body.pack(fill="both")

        self._build_scene(body)

        ctrl = tk.Frame(body, bg=BG2)
        ctrl.pack(side="left", fill="y", padx=(4,6), pady=6)

        # Scrollable control panel
        self._build_ctrl_scroll(ctrl)

        self._build_scope(root)
        self._build_statusbar(root)

    def _build_header(self, parent):
        hdr = tk.Frame(parent, bg=TEAL3, height=48)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        self.blink_canvas = tk.Canvas(hdr, width=80, height=48,
                                       bg=TEAL3, highlightthickness=0)
        self.blink_canvas.pack(side="left", padx=10)
        self._blink_dots = []
        for i, col in enumerate([GREEN, AMBER, RED, MAGENTA, CYAN]):
            d = self.blink_canvas.create_oval(4+i*15, 18, 16+i*15, 30,
                                               fill=col, outline="")
            self._blink_dots.append(d)

        tk.Label(hdr,
                 text="R A D I O D Y N E   //   M O D U L A T I O N   W O R K B E N C H",
                 font=font.Font(family="Courier", size=14, weight="bold"),
                 fg=GREEN, bg=TEAL3).pack(side="left", expand=True)

        tk.Label(hdr, text="REV 3.0  //  AM/FM CHANNEL SIM\nFREE-SPACE PATH + AWGN MODEL",
                 font=font.Font(family="Courier", size=7),
                 fg=TEXTDIM, bg=TEAL3, justify="right").pack(side="right", padx=12)

    def _build_scene(self, parent):
        sf = tk.Frame(parent, bg=BG)
        sf.pack(side="left", padx=(6,0), pady=6)
        self.scene = tk.Canvas(sf, width=500, height=290,
                               bg=BG, highlightthickness=2,
                               highlightbackground=TEAL2)
        self.scene.pack()
        self._draw_scene_bg()
        self._draw_towers()

    def _draw_scene_bg(self):
        c = self.scene
        # Night sky with green-tinted gradient
        for i in range(30):
            ratio = i/29
            r = int(4 + 3*ratio)
            g = int(10 + 18*ratio)
            b = int(6 + 8*ratio)
            c.create_rectangle(0, i*10, 500, i*10+11,
                               fill=f"#{r:02x}{g:02x}{b:02x}", outline="")
        c.create_rectangle(0, 248, 500, 290, fill="#040a04", outline="")
        c.create_line(0, 248, 500, 248, fill=TEAL3, width=1)
        pts = [0,248, 40,228, 80,218, 130,232, 170,210,
               220,224, 260,205, 310,218, 360,208, 420,225, 465,215, 500,230, 500,248]
        c.create_polygon(pts, fill="#061006", outline="")
        # Stars — mix of green, amber, white
        rng = np.random.default_rng(7)
        star_cols = [TEXT, AMBER, "#ffffff", GREEN, CYAN, MAGENTA]
        for _ in range(80):
            x = int(rng.integers(0,500))
            y = int(rng.integers(0,130))
            s = int(rng.choice([1,1,2]))
            col = star_cols[int(rng.integers(0, len(star_cols)))]
            c.create_oval(x, y, x+s, y+s, fill=col, outline="")
        # Moon
        c.create_oval(410, 20, 440, 50, fill="#1a2a1a", outline=TEAL3, width=1)
        c.create_oval(417, 22, 445, 46, fill=BG, outline="")
        # Ground hatching
        for x in range(0, 500, 35):
            c.create_line(x, 248, x+12, 260, fill=TEAL3, width=1)

    def _draw_towers(self):
        c = self.scene
        # TX tower is fixed at x=90; RX tower is draggable
        self._tx_cx = 90
        self._tx_top = 83      # antenna tip y for TX
        self._rx_cx = 410      # initial RX x position
        self._rx_top = 110     # antenna tip y for RX
        self._scene_base_y = 248
        self._rx_height = 138

        # Scene pixel range for RX tower: TX+40 .. 480
        self._rx_min_px = self._tx_cx + 50
        self._rx_max_px = 480

        # Distance range mapped to pixel range
        self._dist_min_km = 0.1
        self._dist_max_km = 1000.0

        self._draw_tower(c, cx=self._tx_cx, base_y=248, height=165)
        lf = font.Font(family="Courier", size=7, weight="bold")
        c.create_text(self._tx_cx, 252, text="TX", fill=TEXTDIM, font=lf, anchor="n")

        # RX tower drawn as a tagged group so we can move it
        self._draw_rx_tower()

        self.signal_arcs = []
        self.signal_dots = []
        self.tower_glows = []

        for cx, ty in [(self._tx_cx, self._tx_top), (self._rx_cx, self._rx_top)]:
            g = c.create_oval(cx-18, ty-18, cx+18, ty+18,
                              outline=TEAL, width=0, fill="")
            self.tower_glows.append(g)

        for _ in range(5):
            arc = c.create_arc(50, 105, 130, 175,
                               start=200, extent=140,
                               outline=TEAL, width=1, style="arc")
            self.signal_arcs.append(arc)

        self.tx_beam = c.create_line(self._tx_cx, self._tx_top,
                                     self._rx_cx, self._rx_top,
                                     fill=TEAL, width=0, dash=(4,4))
        mid_x = (self._tx_cx + self._rx_cx) // 2
        self.dist_label = c.create_text(mid_x, 88, text="", fill=TEXTDIM,
                                        font=font.Font(family="Courier", size=7))
        self.str_label = c.create_text(mid_x, 100, text="", fill=GREEN,
                                       font=font.Font(family="Courier", size=8, weight="bold"))

        for _ in range(7):
            d = c.create_oval(0,0,0,0, fill=TEAL, outline="")
            self.signal_dots.append(d)

        # Drag bindings
        self._dragging_rx = False
        c.tag_bind("rx_tower", "<ButtonPress-1>", self._rx_drag_start)
        c.bind("<B1-Motion>", self._rx_drag_move)
        c.bind("<ButtonRelease-1>", self._rx_drag_end)

    def _draw_rx_tower(self):
        """Draw (or redraw) the RX tower at self._rx_cx using canvas tags."""
        c = self.scene
        c.delete("rx_tower")
        cx = self._rx_cx
        base_y = self._scene_base_y
        height = self._rx_height
        top_y = base_y - height
        bw = 26
        # Main truss
        c.create_polygon(cx-bw, base_y, cx+bw, base_y,
                         cx+4, top_y, cx-4, top_y,
                         fill=BG3, outline=TEAL2, width=1, tags="rx_tower")
        steps = 7
        for i in range(steps):
            y0 = base_y - height*i//steps
            y1 = base_y - height*(i+1)//steps
            w0 = bw - (bw-4)*i//steps
            w1 = bw - (bw-4)*(i+1)//steps
            c.create_line(cx-w0, y0, cx+w1, y1, fill=TEAL2, width=1, tags="rx_tower")
            c.create_line(cx+w0, y0, cx-w1, y1, fill=TEAL2, width=1, tags="rx_tower")
            c.create_line(cx-w0, y0, cx+w0, y0, fill=TEAL2, width=1, tags="rx_tower")
        c.create_line(cx, top_y, cx, top_y-22, fill=GREEN, width=2, tags="rx_tower")
        c.create_line(cx-12, top_y-10, cx+12, top_y-10, fill=GREEN, width=1, tags="rx_tower")
        c.create_line(cx-7, top_y-16, cx+7, top_y-16, fill=GREEN, width=1, tags="rx_tower")
        c.create_oval(cx-4, top_y-26, cx+4, top_y-18,
                      fill=RED, outline=AMBER, width=1, tags="rx_tower")
        lf = font.Font(family="Courier", size=7, weight="bold")
        c.create_text(cx, base_y+4, text="RX", fill=TEXTDIM, font=lf, anchor="n",
                      tags="rx_tower")
        self._rx_top = top_y

    def _px_to_dist(self, px):
        """Map pixel x position to distance in km (log scale for physical realism)."""
        frac = (px - self._rx_min_px) / (self._rx_max_px - self._rx_min_px)
        frac = max(0.0, min(1.0, frac))
        # Log-scale: close towers = small distance, far towers = large distance
        return self._dist_min_km * (self._dist_max_km / self._dist_min_km) ** frac

    def _dist_to_px(self, km):
        """Map distance in km to pixel x position."""
        km = max(self._dist_min_km, min(self._dist_max_km, km))
        frac = math.log(km / self._dist_min_km) / math.log(self._dist_max_km / self._dist_min_km)
        return self._rx_min_px + frac * (self._rx_max_px - self._rx_min_px)

    def _rx_drag_start(self, event):
        self._dragging_rx = True
        self.scene.config(cursor="sb_h_double_arrow")

    def _rx_drag_move(self, event):
        if not self._dragging_rx:
            return
        new_x = max(self._rx_min_px, min(self._rx_max_px, event.x))
        self._rx_cx = new_x
        self._draw_rx_tower()
        dist = self._px_to_dist(new_x)
        self.distance_km.set(round(dist, 2))
        # Update beam and labels immediately
        self.scene.coords(self.tx_beam, self._tx_cx, self._tx_top,
                          self._rx_cx, self._rx_top)
        mid_x = (self._tx_cx + self._rx_cx) // 2
        self.scene.coords(self.dist_label, mid_x, 88)
        self.scene.coords(self.str_label, mid_x, 100)
        if self.transmitting:
            self._compute_signal()
            self._update_metrics()
            self._update_scene_labels()

    def _rx_drag_end(self, event):
        self._dragging_rx = False
        self.scene.config(cursor="")

    def _on_distance_slider(self, *args):
        """Sync RX tower position when the distance slider changes."""
        if self._dragging_rx:
            return  # drag is driving the slider, not the other way
        if not hasattr(self, '_rx_min_px'):
            return  # UI not built yet
        new_px = self._dist_to_px(self.distance_km.get())
        self._rx_cx = int(new_px)
        self._draw_rx_tower()
        self.scene.coords(self.tx_beam, self._tx_cx, self._tx_top,
                          self._rx_cx, self._rx_top)
        mid_x = (self._tx_cx + self._rx_cx) // 2
        self.scene.coords(self.dist_label, mid_x, 88)
        self.scene.coords(self.str_label, mid_x, 100)

    def _draw_tower(self, c, cx, base_y, height):
        top_y = base_y - height
        bw = 26
        c.create_polygon(cx-bw, base_y, cx+bw, base_y,
                         cx+4, top_y, cx-4, top_y,
                         fill=BG3, outline=TEAL2, width=1)
        steps = 7
        for i in range(steps):
            y0 = base_y - height*i//steps
            y1 = base_y - height*(i+1)//steps
            w0 = bw - (bw-4)*i//steps
            w1 = bw - (bw-4)*(i+1)//steps
            c.create_line(cx-w0, y0, cx+w1, y1, fill=TEAL2, width=1)
            c.create_line(cx+w0, y0, cx-w1, y1, fill=TEAL2, width=1)
            c.create_line(cx-w0, y0, cx+w0, y0, fill=TEAL2, width=1)
        c.create_line(cx, top_y, cx, top_y-22, fill=GREEN, width=2)
        c.create_line(cx-12, top_y-10, cx+12, top_y-10, fill=GREEN, width=1)
        c.create_line(cx-7,  top_y-16, cx+7,  top_y-16, fill=GREEN, width=1)
        c.create_oval(cx-4, top_y-26, cx+4, top_y-18,
                      fill=RED, outline=AMBER, width=1)

    def _build_ctrl_scroll(self, parent):
        """All controls in a scrollable canvas column."""
        container = tk.Frame(parent, bg=BG2)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, bg=BG2, highlightthickness=0, width=370)
        vsb    = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=BG2)
        win   = canvas.create_window((0,0), window=inner, anchor="nw")

        def _on_config(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(win, width=canvas.winfo_width())
        inner.bind("<Configure>", _on_config)

        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Fix canvas height to scene height
        canvas.configure(height=290)

        self._build_source_section(inner)
        self._build_signal_section(inner)
        self._build_am_section(inner)
        self._build_fm_section(inner)
        self._build_channel_section(inner)
        self._build_metrics_section(inner)
        self._build_tx_button(inner)

        self._on_mode_change()

    def _build_source_section(self, parent):
        SectionLabel(parent, "DATA SOURCE").pack(fill="x", pady=(6,2))
        upf = tk.Frame(parent, bg=BG2)
        upf.pack(fill="x", pady=(0,2))
        bf = font.Font(family="Courier", size=8, weight="bold")
        tk.Button(upf, text="UPLOAD .npy",
                  font=bf, fg=BG, bg=TEAL2, activebackground=TEAL,
                  relief="flat", bd=0, padx=8, pady=3,
                  cursor="hand2", command=self._upload_npy).pack(side="left")
        tk.Button(upf, text="UPLOAD .wav",
                  font=bf, fg=BG, bg=TEAL2, activebackground=TEAL,
                  relief="flat", bd=0, padx=8, pady=3,
                  cursor="hand2", command=self._upload_wav).pack(side="left", padx=2)
        tk.Button(upf, text="CLEAR",
                  font=font.Font(family="Courier", size=8),
                  fg=TEXTDIM, bg=BG3, activebackground=BG2,
                  relief="flat", bd=0, padx=6, pady=3,
                  cursor="hand2", command=self._clear_upload).pack(side="left", padx=4)
        tk.Label(upf, textvariable=self._uploaded_name,
                 font=font.Font(family="Courier", size=7),
                 fg=AMBER, bg=BG2, padx=4).pack(side="left")

        # Audio playback controls
        SectionLabel(parent, "AUDIO MONITOR").pack(fill="x", pady=(6,2))
        af = tk.Frame(parent, bg=BG2)
        af.pack(fill="x", pady=(0,2))
        abf = font.Font(family="Courier", size=9, weight="bold")
        self._preview_btn = tk.Button(
            af, text="🎧 SOURCE",
            font=abf, fg=BG, bg=VIOLET, activebackground=MAGENTA,
            relief="flat", bd=0, padx=8, pady=3,
            cursor="hand2", command=self._toggle_preview,
            state="disabled" if not HAS_AUDIO else "normal")
        self._preview_btn.pack(side="left")
        self._audio_btn = tk.Button(
            af, text="🔊 TX/RX",
            font=abf, fg=BG, bg=GREEN2, activebackground=GREEN,
            relief="flat", bd=0, padx=8, pady=3,
            cursor="hand2", command=self._toggle_audio,
            state="disabled" if not HAS_AUDIO else "normal")
        self._audio_btn.pack(side="left", padx=4)
        self._audio_status = tk.Label(
            af, text="IDLE" if HAS_AUDIO else "NO AUDIO (pip install sounddevice)",
            font=font.Font(family="Courier", size=7),
            fg=TEXTDIM, bg=BG2, padx=6)
        self._audio_status.pack(side="left")

        af2 = tk.Frame(parent, bg=BG2)
        af2.pack(fill="x", pady=(0,2))
        SliderRow(af2, "VOLUME", self._audio_volume, 0.0, 1.0, 0.05).pack(fill="x", pady=1)
        lpf = tk.Frame(parent, bg=BG2)
        lpf.pack(fill="x", pady=1)
        tk.Label(lpf, text="LOOP PLAYBACK",
                 font=font.Font(family="Courier", size=7), width=18, anchor="w",
                 fg=TEXTDIM, bg=BG2).pack(side="left")
        tk.Checkbutton(lpf, variable=self._audio_loop,
                       bg=BG2, fg=TEAL, selectcolor=TEAL3,
                       activebackground=BG2, relief="flat",
                       highlightthickness=0).pack(side="left")

        # Preset load/save
        SectionLabel(parent, "PRESETS").pack(fill="x", pady=(6,2))
        pf = tk.Frame(parent, bg=BG2)
        pf.pack(fill="x", pady=(0,4))
        tk.Button(pf, text="LOAD CONFIG",
                  font=font.Font(family="Courier", size=8, weight="bold"),
                  fg=BG, bg=AMBER, activebackground=AMBER2,
                  relief="flat", bd=0, padx=8, pady=3,
                  cursor="hand2", command=self._load_config).pack(side="left")
        tk.Button(pf, text="SAVE CONFIG",
                  font=font.Font(family="Courier", size=8),
                  fg=TEXTDIM, bg=BG3, activebackground=BG2,
                  relief="flat", bd=0, padx=6, pady=3,
                  cursor="hand2", command=self._save_config).pack(side="left", padx=4)

        SectionLabel(parent, "MODULATION MODE").pack(fill="x", pady=(6,2))
        modf = tk.Frame(parent, bg=BG2)
        modf.pack(fill="x", pady=(0,4))
        self._mod_buttons = {}
        for mode, col in [("AM", CYAN), ("FM", MAGENTA)]:
            b = tk.Button(modf, text=mode,
                          font=font.Font(family="Courier", size=11, weight="bold"),
                          fg=BG, bg=col if self.modulation.get() == mode else BG3,
                          activebackground=col, activeforeground=BG,
                          relief="raised" if self.modulation.get() == mode else "flat",
                          bd=2, padx=16, pady=4, width=4,
                          cursor="hand2",
                          command=lambda m=mode: self._set_modulation(m))
            b.pack(side="left", padx=3)
            self._mod_buttons[mode] = (b, col)

        SectionLabel(parent, "WAVEFORM  (synthetic source)").pack(fill="x", pady=(6,2))
        self.wave_frame = tk.Frame(parent, bg=BG2)
        self.wave_frame.pack(fill="x", pady=(0,4))
        waves = ["sine","square","sawtooth","triangle","pulse","chirp","noise"]
        wave_cols = [GREEN, CYAN, AMBER, MAGENTA, RED, VIOLET, TEXT]
        for i, (w, wc) in enumerate(zip(waves, wave_cols)):
            tk.Radiobutton(self.wave_frame, text=w.upper(),
                           variable=self.waveform, value=w,
                           font=font.Font(family="Courier", size=8),
                           fg=wc, bg=BG2, selectcolor=TEAL3,
                           activebackground=BG2, activeforeground=wc,
                           indicatoron=True, relief="flat"
                           ).grid(row=i//2, column=i%2, sticky="w", padx=4, pady=1)

    def _build_signal_section(self, parent):
        SectionLabel(parent, "SIGNAL PARAMETERS").pack(fill="x", pady=(6,2))
        params = [
            ("SRC FREQ (Hz)",  self.src_freq,    0.5, 500,   0.5),
            ("CARRIER (Hz)",   self.carrier_freq, 2,  500,   1.0),
            ("AMPLITUDE",      self.amplitude,    0.1, 3.0,  0.05),
            ("PHASE (rad)",    self.phase,        0,   6.28, 0.01),
            ("DUTY CYCLE",     self.duty_cycle,   0.05, 0.95, 0.05),
            ("HARMONICS",      self.harmonics,    1,   16,   1),
            ("SAMPLE RATE",    self.sample_rate,  1000, 44100, 500),
            ("DURATION (s)",   self.duration,     0.1,  10.0, 0.1),
        ]
        for args in params:
            SliderRow(parent, *args).pack(fill="x", pady=1)

    def _build_am_section(self, parent):
        SectionLabel(parent, "AM PARAMETERS").pack(fill="x", pady=(6,2))
        self.am_frame = tk.Frame(parent, bg=BG2)
        self.am_frame.pack(fill="x")
        SliderRow(self.am_frame, "MOD INDEX",
                  self.mod_index, 0.1, 2.0, 0.05).pack(fill="x", pady=1)

    def _build_fm_section(self, parent):
        SectionLabel(parent, "FM PARAMETERS").pack(fill="x", pady=(6,2))
        self.fm_frame = tk.Frame(parent, bg=BG2)
        self.fm_frame.pack(fill="x")
        SliderRow(self.fm_frame, "FREQ DEV (Hz)",
                  self.freq_dev, 1, 1000, 1).pack(fill="x", pady=1)

    def _build_channel_section(self, parent):
        SectionLabel(parent, "CHANNEL MODEL").pack(fill="x", pady=(6,2))
        SliderRow(parent, "DISTANCE (km)",     self.distance_km,  0.1, 1000,  0.5).pack(fill="x", pady=1)
        SliderRow(parent, "TX POWER (W)",      self.tx_power_w,   0.1, 10000, 1.0).pack(fill="x", pady=1)
        SliderRow(parent, "CARRIER FREQ(MHz)", self.carrier_mhz,  0.1, 10000, 1.0).pack(fill="x", pady=1)
        SliderRow(parent, "NOISE FLOOR(dBm)",  self.noise_floor, -160, -40,   1.0,
                  fmt="{:.0f}").pack(fill="x", pady=1)
        mpf = tk.Frame(parent, bg=BG2)
        mpf.pack(fill="x", pady=2)
        tk.Label(mpf, text="MULTIPATH FADING",
                 font=font.Font(family="Courier", size=7), width=18, anchor="w",
                 fg=TEXTDIM, bg=BG2).pack(side="left")
        tk.Checkbutton(mpf, variable=self.multipath,
                       bg=BG2, fg=TEAL, selectcolor=TEAL3,
                       activebackground=BG2, relief="flat",
                       highlightthickness=0).pack(side="left")

    def _build_metrics_section(self, parent):
        SectionLabel(parent, "SIGNAL METRICS").pack(fill="x", pady=(8,2))
        row = tk.Frame(parent, bg=BG2)
        row.pack(fill="x")

        # Strength meter
        lm = tk.Frame(row, bg=BG2)
        lm.pack(side="left", padx=(2,8))
        tk.Label(lm, text="STRENGTH", font=font.Font(family="Courier", size=6),
                 fg=TEXTDIM, bg=BG2).pack()
        self.sig_meter = SignalMeter(lm, width=30, height=110)
        self.sig_meter.pack()
        self.strength_lbl = tk.Label(lm, text="0%",
                                      font=font.Font(family="Courier", size=7, weight="bold"),
                                      fg=GREEN, bg=BG2)
        self.strength_lbl.pack()

        # SNR gauge
        rm = tk.Frame(row, bg=BG2)
        rm.pack(side="left")
        tk.Label(rm, text="SNR", font=font.Font(family="Courier", size=6),
                 fg=TEXTDIM, bg=BG2).pack()
        self.snr_gauge = SNRGauge(rm, size=100)
        self.snr_gauge.pack()

        # Readouts
        ro = tk.Frame(parent, bg=BG2)
        ro.pack(fill="x", pady=(4,0))
        f7b = font.Font(family="Courier", size=7, weight="bold")
        f7  = font.Font(family="Courier", size=7)
        self._metric_labels = {}
        for label, key in [("SNR (dB)",       "snr"),
                            ("PATH LOSS (dB)", "fspl"),
                            ("RX PWR (dBm)",   "rxpwr"),
                            ("BANDWIDTH (Hz)", "bw"),
                            ("STRENGTH",       "str")]:
            r = tk.Frame(ro, bg=BG2)
            r.pack(fill="x")
            tk.Label(r, text=label, width=16, anchor="w",
                     font=f7, fg=TEXTDIM, bg=BG2).pack(side="left")
            lbl = tk.Label(r, text="—", width=12, anchor="e",
                           font=f7b, fg=TEAL, bg=BG2)
            lbl.pack(side="right")
            self._metric_labels[key] = lbl

        # Scope TX/RX toggle
        tg = tk.Frame(parent, bg=BG2)
        tg.pack(fill="x", pady=(4,0))
        tk.Label(tg, text="SCOPE VIEW:",
                 font=font.Font(family="Courier", size=7),
                 fg=TEXTDIM, bg=BG2).pack(side="left")
        for label, val in [("TX (modulated)", False), ("RX (received)", True)]:
            tk.Radiobutton(tg, text=label, variable=self._show_rx, value=val,
                           font=font.Font(family="Courier", size=8),
                           fg=GREEN, bg=BG2, selectcolor=TEAL3,
                           activebackground=BG2, activeforeground=AMBER,
                           indicatoron=True,
                           relief="flat").pack(side="left", padx=4)

    def _build_tx_button(self, parent):
        tk.Frame(parent, bg=TEAL2, height=2).pack(fill="x", pady=8)
        self.tx_btn = tk.Button(parent, text="▶   TRANSMIT",
                                font=font.Font(family="Courier", size=13, weight="bold"),
                                fg=BG, bg=GREEN, activebackground=GREEN2,
                                activeforeground=BG, relief="raised",
                                cursor="hand2", bd=2, padx=10, pady=10,
                                command=self._toggle_transmit)
        self.tx_btn.pack(fill="x", pady=2, padx=2)

    def _build_scope(self, parent):
        pnl = tk.Frame(parent, bg=BG)
        pnl.pack(fill="x", padx=6, pady=(0,4))
        hdr = tk.Frame(pnl, bg=TEAL3)
        hdr.pack(fill="x")
        f8b = font.Font(family="Courier", size=8, weight="bold")
        tk.Label(hdr, text="◈  O S C I L L O S C O P E",
                 font=f8b, fg=GREEN, bg=TEAL3, pady=3).pack(side="left", padx=8)
        self.scope_lbl = tk.Label(hdr, text="[ STANDBY ]",
                                   font=f8b, fg=AMBER, bg=TEAL3, pady=3)
        self.scope_lbl.pack(side="right", padx=8)
        W = self.SCOPE_W
        self.wave_canvas = tk.Canvas(pnl, width=W, height=120,
                                      bg=BG, highlightthickness=1,
                                      highlightbackground=BORDER)
        self.wave_canvas.pack()
        # CRT scanline grid
        for y in range(0, 120, 3):
            self.wave_canvas.create_line(0, y, W, y, fill="#081008", width=1)
        self.wave_canvas.create_line(0, 60, W, 60, fill=TEAL3, width=1, dash=(4,4))
        for x in range(0, W, 92):
            self.wave_canvas.create_line(x, 0, x, 120, fill=TEAL3, width=1, dash=(2,6))
        self.wave_ghost = self.wave_canvas.create_line(0,60,W,60, fill="#0a3318", width=5, smooth=True)
        self.wave_line  = self.wave_canvas.create_line(0,60,W,60, fill=GREEN,    width=2, smooth=True)

    def _build_statusbar(self, parent):
        bar = tk.Frame(parent, bg=TEAL3, height=24)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        sf = font.Font(family="Courier", size=8)
        self.status_lbl = tk.Label(bar,
                                    text="● STANDBY  //  CONFIGURE PARAMETERS AND TRANSMIT",
                                    font=sf, fg=GREEN, bg=TEAL3, anchor="w")
        self.status_lbl.pack(side="left", padx=10)
        self.clock_lbl = tk.Label(bar, text="", font=sf, fg=AMBER, bg=TEAL3)
        self.clock_lbl.pack(side="right", padx=10)
        self._tick_clock()

    # ─────────────────────────────────────────────────────────────
    # LOGIC
    # ─────────────────────────────────────────────────────────────

    def _set_modulation(self, mode):
        self.modulation.set(mode)
        self._on_mode_change()

    def _on_mode_change(self):
        is_fm = self.modulation.get() == "FM"
        # Update toggle button visuals
        if hasattr(self, '_mod_buttons'):
            for m, (btn, col) in self._mod_buttons.items():
                active = (m == self.modulation.get())
                btn.config(bg=col if active else BG3,
                           fg=BG if active else TEXTDIM,
                           relief="raised" if active else "flat")
        # Enable/disable AM vs FM parameter sections
        def _set_tree(widget, state):
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass
            for child in widget.winfo_children():
                _set_tree(child, state)
        for w in self.fm_frame.winfo_children():
            _set_tree(w, "normal" if is_fm else "disabled")
        for w in self.am_frame.winfo_children():
            _set_tree(w, "disabled" if is_fm else "normal")

    def _upload_npy(self):
        path = filedialog.askopenfilename(
            title="Select NumPy .npy wave file",
            filetypes=[("NumPy array", "*.npy"), ("All files", "*.*")])
        if not path:
            return
        try:
            arr = np.load(path).astype(np.float64).ravel()
            if len(arr) < 4:
                raise ValueError("Array too short (need >= 4 samples)")
            peak = np.max(np.abs(arr)) or 1.0
            self._uploaded_wave = arr / peak
            self._uploaded_sr = None  # unknown native SR for .npy
            self._uploaded_name.set(os.path.basename(path))
            for w in self.wave_frame.winfo_children():
                try: w.configure(state="disabled")
                except: pass
            self.status_lbl.config(
                text=f"● FILE: {os.path.basename(path)}  //  {len(arr)} samples")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _upload_wav(self):
        path = filedialog.askopenfilename(
            title="Select WAV audio file",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")])
        if not path:
            return
        try:
            samples, native_sr = load_wav_file(path)
            if len(samples) < 4:
                raise ValueError("Audio too short (need >= 4 samples)")
            peak = np.max(np.abs(samples)) or 1.0
            self._uploaded_wave = samples / peak
            self._uploaded_sr = native_sr
            self._uploaded_name.set(os.path.basename(path))
            dur = len(samples) / native_sr
            for w in self.wave_frame.winfo_children():
                try: w.configure(state="disabled")
                except: pass
            self.status_lbl.config(
                text=f"● WAV: {os.path.basename(path)}  //  "
                     f"{native_sr} Hz, {dur:.2f}s, {len(samples)} samples")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _clear_upload(self):
        self._stop_audio()
        self._uploaded_wave = None
        self._uploaded_sr = None
        self._uploaded_name.set("none")
        for w in self.wave_frame.winfo_children():
            try: w.configure(state="normal")
            except: pass
        self.status_lbl.config(text="● STANDBY  //  USING SYNTHETIC SOURCE")

    # ─────────────────────────────────────────────────────────────
    # CONFIG LOAD / SAVE
    # ─────────────────────────────────────────────────────────────

    _CONFIG_KEYS = [
        ('modulation', 'str'), ('waveform', 'str'),
        ('src_freq', 'float'), ('carrier_freq', 'float'),
        ('amplitude', 'float'), ('phase', 'float'),
        ('duty_cycle', 'float'), ('harmonics', 'int'),
        ('sample_rate', 'int'), ('duration', 'float'),
        ('mod_index', 'float'), ('freq_dev', 'float'),
        ('distance_km', 'float'), ('tx_power_w', 'float'),
        ('carrier_mhz', 'float'), ('noise_floor_dbm', 'float'),
        ('multipath', 'bool'), ('volume', 'float'), ('loop', 'bool'),
    ]

    def _load_config(self):
        import json as _json
        path = filedialog.askopenfilename(
            title="Load Radio Config",
            filetypes=[("JSON config", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path) as f:
                cfg = _json.load(f)
            var_map = {
                'modulation': self.modulation, 'waveform': self.waveform,
                'src_freq': self.src_freq, 'carrier_freq': self.carrier_freq,
                'amplitude': self.amplitude, 'phase': self.phase,
                'duty_cycle': self.duty_cycle, 'harmonics': self.harmonics,
                'sample_rate': self.sample_rate, 'duration': self.duration,
                'mod_index': self.mod_index, 'freq_dev': self.freq_dev,
                'distance_km': self.distance_km, 'tx_power_w': self.tx_power_w,
                'carrier_mhz': self.carrier_mhz,
                'noise_floor_dbm': self.noise_floor, 'noise_floor': self.noise_floor,
                'multipath': self.multipath,
                'volume': self._audio_volume, 'loop': self._audio_loop,
            }
            for key, var in var_map.items():
                if key in cfg:
                    var.set(cfg[key])
            # Auto-load source_file relative to config dir
            if 'source_file' in cfg:
                src_path = os.path.join(os.path.dirname(path), cfg['source_file'])
                if os.path.exists(src_path):
                    if src_path.endswith('.wav'):
                        samples, native_sr = load_wav_file(src_path)
                        peak = np.max(np.abs(samples)) or 1.0
                        self._uploaded_wave = samples / peak
                        self._uploaded_sr = native_sr
                    else:
                        arr = np.load(src_path).astype(np.float64).ravel()
                        peak = np.max(np.abs(arr)) or 1.0
                        self._uploaded_wave = arr / peak
                        self._uploaded_sr = None
                    self._uploaded_name.set(cfg['source_file'])
                    for w in self.wave_frame.winfo_children():
                        try: w.configure(state="disabled")
                        except: pass
            self._on_mode_change()
            desc = cfg.get('description', os.path.basename(path))
            self.status_lbl.config(text=f"● CONFIG: {desc}")
        except Exception as e:
            messagebox.showerror("Config Error", str(e))

    def _save_config(self):
        import json as _json
        path = filedialog.asksaveasfilename(
            title="Save Radio Config",
            defaultextension=".json",
            filetypes=[("JSON config", "*.json")])
        if not path:
            return
        cfg = {
            'modulation': self.modulation.get(),
            'waveform': self.waveform.get(),
            'src_freq': self.src_freq.get(),
            'carrier_freq': self.carrier_freq.get(),
            'amplitude': self.amplitude.get(),
            'phase': self.phase.get(),
            'duty_cycle': self.duty_cycle.get(),
            'harmonics': self.harmonics.get(),
            'sample_rate': self.sample_rate.get(),
            'duration': self.duration.get(),
            'mod_index': self.mod_index.get(),
            'freq_dev': self.freq_dev.get(),
            'distance_km': self.distance_km.get(),
            'tx_power_w': self.tx_power_w.get(),
            'carrier_mhz': self.carrier_mhz.get(),
            'noise_floor_dbm': self.noise_floor.get(),
            'multipath': self.multipath.get(),
            'volume': self._audio_volume.get(),
            'loop': self._audio_loop.get(),
        }
        if self._uploaded_name.get() != "none":
            cfg['source_file'] = self._uploaded_name.get()
        with open(path, 'w') as f:
            _json.dump(cfg, f, indent=2)
        self.status_lbl.config(text=f"● SAVED: {os.path.basename(path)}")

    # ─────────────────────────────────────────────────────────────
    # AUDIO PLAYBACK (demodulated received signal)
    # ─────────────────────────────────────────────────────────────

    def _toggle_preview(self):
        """Play/stop the raw source audio (no modulation or channel)."""
        if self._audio_previewing:
            self._stop_audio()
        else:
            self._start_preview()

    def _start_preview(self):
        if not HAS_AUDIO:
            self._audio_status.config(text="NO AUDIO BACKEND", fg=RED)
            return
        # Stop any existing playback first
        if self._audio_playing or self._audio_previewing:
            self._stop_audio()

        src, SR, DUR = self._get_source()
        if len(src) < 4:
            self._audio_status.config(text="NO SOURCE", fg=RED)
            return

        playback_sr = self._uploaded_sr or SR
        # Resample source to playback rate if needed
        if len(src) != int(playback_sr * DUR):
            n_out = int(playback_sr * DUR)
            src = np.interp(np.linspace(0, len(src)-1, n_out),
                            np.arange(len(src)), src)

        peak = np.max(np.abs(src)) or 1.0
        buf = (src / peak).astype(np.float32)
        self._demod_read_pos = 0
        self._audio_previewing = True
        self._cached_volume = float(self._audio_volume.get())
        self._cached_loop = bool(self._audio_loop.get())

        with self._audio_lock:
            self._demod_buffer = buf

        def _callback(outdata, frames, time_info, status):
            with self._audio_lock:
                if not self._audio_previewing:
                    outdata[:] = 0
                    raise sd.CallbackAbort
                b = self._demod_buffer
                vol = self._cached_volume
                pos = self._demod_read_pos
                nb = len(b)
                if nb == 0:
                    outdata[:] = 0
                    return
                remaining = frames
                written = 0
                out = np.zeros(frames, dtype=np.float32)
                while remaining > 0:
                    avail = min(remaining, nb - pos)
                    out[written:written + avail] = b[pos:pos + avail]
                    written += avail
                    pos += avail
                    remaining -= avail
                    if pos >= nb:
                        if self._cached_loop:
                            pos = 0
                        else:
                            self._audio_previewing = False
                            break
                self._demod_read_pos = pos
                outdata[:, 0] = out * vol

        try:
            self._audio_stream = sd.OutputStream(
                samplerate=int(playback_sr), channels=1, dtype='float32',
                callback=_callback, blocksize=1024)
            self._audio_stream.start()
            self._preview_btn.config(text="🎧 STOP", bg=RED, activebackground="#ff6666")
            self._audio_status.config(text=f"PREVIEW @ {int(playback_sr)} Hz", fg=VIOLET)
            self._preview_refresh_loop()
        except Exception as e:
            self._audio_previewing = False
            self._audio_status.config(text=f"ERR: {e}", fg=RED)

    def _preview_refresh_loop(self):
        if not self._audio_previewing:
            self._stop_audio()
            return
        self._cached_volume = float(self._audio_volume.get())
        self._cached_loop = bool(self._audio_loop.get())
        self.after(250, self._preview_refresh_loop)

    def _toggle_audio(self):
        if self._audio_playing:
            self._stop_audio()
        else:
            self._start_audio()

    def _start_audio(self):
        if not HAS_AUDIO or not self.transmitting:
            self._audio_status.config(
                text="TRANSMIT FIRST" if HAS_AUDIO else "NO AUDIO BACKEND",
                fg=RED)
            return
        # Stop preview if running
        if self._audio_previewing:
            self._stop_audio()
        self._recompute_demod()
        if len(self._demod_buffer) < 4:
            self._audio_status.config(text="NO SIGNAL", fg=RED)
            return

        playback_sr = self._uploaded_sr or self.sample_rate.get()
        self._demod_read_pos = 0
        self._audio_playing = True
        # Cache tkinter var values for thread-safe access
        self._cached_volume = float(self._audio_volume.get())
        self._cached_loop = bool(self._audio_loop.get())

        def _callback(outdata, frames, time_info, status):
            with self._audio_lock:
                if not self._audio_playing:
                    outdata[:] = 0
                    raise sd.CallbackAbort
                buf = self._demod_buffer
                vol = self._cached_volume
                pos = self._demod_read_pos
                n = len(buf)
                if n == 0:
                    outdata[:] = 0
                    return
                # Vectorized copy with wraparound
                remaining = frames
                written = 0
                out = np.zeros(frames, dtype=np.float32)
                while remaining > 0:
                    avail = min(remaining, n - pos)
                    out[written:written + avail] = buf[pos:pos + avail]
                    written += avail
                    pos += avail
                    remaining -= avail
                    if pos >= n:
                        if self._cached_loop:
                            pos = 0
                        else:
                            self._audio_playing = False
                            break
                self._demod_read_pos = pos
                outdata[:, 0] = out * vol

        try:
            self._audio_stream = sd.OutputStream(
                samplerate=int(playback_sr),
                channels=1,
                dtype='float32',
                callback=_callback,
                blocksize=1024)
            self._audio_stream.start()
            self._audio_playing = True
            self._audio_btn.config(text="🔇 STOP", bg=RED, activebackground="#ff6666")
            self._audio_status.config(
                text=f"PLAYING @ {int(playback_sr)} Hz", fg=GREEN)
            self._audio_refresh_loop()
        except Exception as e:
            self._audio_playing = False
            self._audio_status.config(text=f"ERR: {e}", fg=RED)

    def _stop_audio(self):
        self._audio_playing = False
        self._audio_previewing = False
        if self._audio_stream is not None:
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None
        self._audio_btn.config(text="🔊 TX/RX", bg=GREEN2, activebackground=GREEN)
        self._preview_btn.config(text="🎧 SOURCE", bg=VIOLET, activebackground=MAGENTA)
        self._audio_status.config(text="IDLE", fg=TEXTDIM)

    def _recompute_demod(self):
        """Recompute demodulated audio from current channel parameters."""
        src, SR, DUR = self._get_source()
        cf = self.carrier_freq.get()
        if self.modulation.get() == "AM":
            tx = amplitude_modulation(src, cf, SR, DUR, self.mod_index.get())
        else:
            tx = frequency_modulation(src, cf, self.freq_dev.get(), SR, DUR)

        rx, _, _, _, _ = apply_channel(
            tx,
            distance_km=self.distance_km.get(),
            tx_power_w=self.tx_power_w.get(),
            freq_mhz=self.carrier_mhz.get(),
            noise_floor_dbm=self.noise_floor.get(),
            multipath=self.multipath.get(),
            sample_rate=SR,
        )

        if self.modulation.get() == "AM":
            demod = demodulate_am(rx, cf, SR)
        else:
            demod = demodulate_fm(rx, SR)

        with self._audio_lock:
            self._demod_buffer = demod.astype(np.float32)

    def _audio_refresh_loop(self):
        """Periodically recompute demod buffer so slider changes are heard live."""
        if not self._audio_playing or not self.transmitting:
            self._stop_audio()
            return
        # Sync cached values from tkinter vars (main thread safe)
        self._cached_volume = float(self._audio_volume.get())
        self._cached_loop = bool(self._audio_loop.get())
        self._recompute_demod()
        self.after(500, self._audio_refresh_loop)

    def _tick_clock(self):
        self.clock_lbl.config(text=time.strftime("%H:%M:%S  UTC"))
        self.after(1000, self._tick_clock)

    def _toggle_transmit(self):
        if not self.transmitting:
            self.transmitting = True
            self.tx_btn.config(text="■   ABORT",
                               bg=RED, activebackground="#ff6666")
            self.status_lbl.config(text="▶ TRANSMITTING  //  CHANNEL ACTIVE")
            self.scope_lbl.config(text=f"[ {self.modulation.get()} LIVE ]")
            self._compute_signal()
            self._update_metrics()
            self._update_scene_labels()
            self._start_tx_anim()
        else:
            self.transmitting = False
            self._stop_audio()
            self.tx_btn.config(text="▶   TRANSMIT", bg=GREEN, activebackground=GREEN2)
            self.status_lbl.config(text="■ HALTED  //  STANDBY")
            self.scope_lbl.config(text="[ STANDBY ]")
            self._stop_tx_anim()

    def _get_source(self):
        SR  = self.sample_rate.get()
        DUR = self.duration.get()
        if self._uploaded_wave is not None:
            n   = int(SR * DUR)
            src = np.interp(np.linspace(0, len(self._uploaded_wave)-1, n),
                            np.arange(len(self._uploaded_wave)),
                            self._uploaded_wave)
            return src, SR, DUR
        src = generate_source_wave(
            waveform    = self.waveform.get(),
            frequency   = self.src_freq.get(),
            amplitude   = self.amplitude.get(),
            duration    = DUR,
            sample_rate = SR,
            phase       = self.phase.get(),
            duty_cycle  = self.duty_cycle.get(),
            harmonics   = int(self.harmonics.get()),
        )
        return src, SR, DUR

    def _compute_signal(self):
        src, SR, DUR = self._get_source()
        cf = self.carrier_freq.get()
        if self.modulation.get() == "AM":
            tx = amplitude_modulation(src, cf, SR, DUR, self.mod_index.get())
        else:
            tx = frequency_modulation(src, cf, self.freq_dev.get(), SR, DUR)

        rx, snr, strength, fspl, rxpwr = apply_channel(
            tx,
            distance_km     = self.distance_km.get(),
            tx_power_w      = self.tx_power_w.get(),
            freq_mhz        = self.carrier_mhz.get(),
            noise_floor_dbm = self.noise_floor.get(),
            multipath       = self.multipath.get(),
            sample_rate     = SR,
        )
        self._snr_db       = snr
        self._strength_pct = strength
        self._fspl         = fspl
        self._rxpwr        = rxpwr

        W    = self.SCOPE_W
        idxs = np.linspace(0, len(tx)-1, W).astype(int)
        self.signal_data = tx[idxs]
        self.rx_data     = rx[idxs]

    def _update_metrics(self):
        # Carson's rule bandwidth
        sr = self.src_freq.get()
        bw = 2*(self.freq_dev.get() + sr) if self.modulation.get()=="FM" else 2*sr
        self._metric_labels["snr" ].config(text=f"{self._snr_db:.1f} dB")
        self._metric_labels["fspl"].config(text=f"{self._fspl:.1f} dB")
        self._metric_labels["rxpwr"].config(text=f"{self._rxpwr:.1f} dBm")
        self._metric_labels["bw"  ].config(text=f"{bw:.1f} Hz")
        self._metric_labels["str" ].config(text=f"{self._strength_pct:.1f} %")
        self.sig_meter.set(self._strength_pct)
        self.snr_gauge.set(self._snr_db)
        col = GREEN if self._strength_pct > 60 else (AMBER if self._strength_pct > 25 else RED)
        self.strength_lbl.config(text=f"{self._strength_pct:.0f}%", fg=col)

    def _update_scene_labels(self):
        d = self.distance_km.get()
        self.scene.itemconfig(self.dist_label, text=f"{d:.1f} km")
        col = GREEN if self._strength_pct > 60 else (AMBER if self._strength_pct > 25 else RED)
        self.scene.itemconfig(self.str_label,
                              text=f"{self._strength_pct:.0f}% SIG", fill=col)

    # ─────────────────────────────────────────────────────────────
    # ANIMATION
    # ─────────────────────────────────────────────────────────────

    def _start_idle_animation(self):
        self._idle_loop()

    def _idle_loop(self):
        t = time.time()
        # Cycle through blink dots with staggered phases
        dot_colors = [GREEN, AMBER, RED, MAGENTA, CYAN]
        for i, base_col in enumerate(dot_colors):
            phase = (math.sin(t * 1.5 + i * 1.2) + 1) / 2
            # Pulse brightness
            br = int(0x30 + 0xcf * phase)
            r = int(int(base_col[1:3], 16) * br / 255)
            g = int(int(base_col[3:5], 16) * br / 255)
            b = int(int(base_col[5:7], 16) * br / 255)
            if i < len(self._blink_dots):
                self.blink_canvas.itemconfig(self._blink_dots[i],
                                             fill=f"#{r:02x}{g:02x}{b:02x}")
        if not self.transmitting:
            self._draw_wave(np.random.uniform(-0.07, 0.07, self.SCOPE_W))
        self.after(60, self._idle_loop)

    def _start_tx_anim(self):
        self.scroll_offset = 0
        self._tx_frame     = 0
        self._tx_loop()

    def _stop_tx_anim(self):
        self.scene.itemconfig(self.tx_beam, width=0)
        for d in self.signal_dots:
            self.scene.coords(d, 0,0,0,0)
        for a in self.signal_arcs:
            self.scene.itemconfig(a, outline=BG)
        for g in self.tower_glows:
            self.scene.itemconfig(g, outline=BG, width=0)
        self.scene.itemconfig(self.dist_label, text="")
        self.scene.itemconfig(self.str_label,  text="")

    def _tx_loop(self):
        if not self.transmitting:
            return
        t = time.time()
        f = self._tx_frame
        self._tx_frame += 1

        if f % 8 == 0:
            self._compute_signal()
            self._update_metrics()
            self._update_scene_labels()

        # Scope scroll
        self.scroll_offset = (self.scroll_offset + 3) % self.SCOPE_W
        data = self.rx_data if self._show_rx.get() else self.signal_data
        self._draw_wave(np.roll(data, -self.scroll_offset))

        # Tower glow
        gp = (math.sin(t*4)+1)/2
        gw = int(1 + 3*gp)
        tx_cx, tx_ty = self._tx_cx, self._tx_top
        rx_cx, rx_ty = self._rx_cx, self._rx_top
        for i, g in enumerate(self.tower_glows):
            r  = int(18 + 18*gp)
            cx = tx_cx if i == 0 else rx_cx
            ty = tx_ty if i == 0 else rx_ty
            self.scene.coords(g, cx-r, ty-r, cx+r, ty+r)
            self.scene.itemconfig(g, outline=GREEN, width=gw)

        # Beam
        self.scene.coords(self.tx_beam, tx_cx, tx_ty, rx_cx, rx_ty)
        self.scene.itemconfig(self.tx_beam, width=int(1+2*gp), fill=GREEN)

        # Labels
        mid_x = (tx_cx + rx_cx) // 2
        self.scene.coords(self.dist_label, mid_x, 88)
        self.scene.coords(self.str_label, mid_x, 100)

        # Radiating arcs
        for i, arc in enumerate(self.signal_arcs):
            ps     = (f*0.08 - i*0.5)
            radius = 20 + ((ps*18) % 90)
            alpha  = max(0, 1 - radius/90)
            col    = GREEN if alpha>0.5 else (TEAL2 if alpha>0.2 else TEAL3)
            self.scene.coords(arc, tx_cx-radius, tx_ty-radius*0.7,
                              tx_cx+radius, tx_ty+radius*0.7)
            self.scene.itemconfig(arc, outline=col, width=max(1, int(2*alpha)))

        # Packets — colour reflects signal quality
        str_col = GREEN if self._strength_pct>60 else (AMBER if self._strength_pct>25 else RED)
        for i, dot in enumerate(self.signal_dots):
            progress = ((f*0.012 + i/7) % 1.0)
            x = tx_cx + (rx_cx - tx_cx) * progress
            y_mid = (tx_ty + rx_ty) / 2
            y = y_mid + (-28 * math.sin(math.pi * progress))
            sz = 3
            self.scene.coords(dot, x-sz, y-sz, x+sz, y+sz)
            self.scene.itemconfig(dot, fill=(CYAN if progress<0.5 else str_col))

        self.after(40, self._tx_loop)

    def _draw_wave(self, data):
        W, H, MID = self.SCOPE_W, 120, 60
        scale = H * 0.42
        pg, pm = [], []
        for x, v in enumerate(data):
            y = max(4, min(H-4, MID - float(v)*scale))
            pg += [x, y+1]
            pm += [x, y]
        if len(pm) >= 4:
            self.wave_canvas.coords(self.wave_ghost, *pg)
            self.wave_canvas.coords(self.wave_line,  *pm)


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = RadioApp()
    app.mainloop()