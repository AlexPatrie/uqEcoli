# eq.py
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal


class MultiBandEQ:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        # 5-band EQ: bass, low-mid, mid, high-mid, treble
        self.bands = {
            "Bass": (20, 250),
            "Low-Mid": (250, 500),
            "Mid": (500, 2000),
            "High-Mid": (2000, 4000),
            "Treble": (4000, 20000),
        }
        self.gains = dict.fromkeys(self.bands, 0.0)  # dB

    def set_gain(self, band, gain_db):
        """Set gain for a specific band in dB (-12 to +12)"""
        self.gains[band] = np.clip(gain_db, -12, 12)

    def apply_eq(self, audio):
        """Apply EQ to audio signal"""
        if len(audio.shape) == 1:
            # Mono
            return self._process_channel(audio)
        else:
            # Stereo - process each channel
            left = self._process_channel(audio[:, 0])
            right = self._process_channel(audio[:, 1])
            return np.column_stack([left, right])

    def _process_channel(self, channel):
        """Process a single channel"""
        output = np.zeros_like(channel)

        for band_name, (low_freq, high_freq) in self.bands.items():
            gain_db = self.gains[band_name]
            if gain_db == 0:
                continue

            # Design bandpass filter
            nyquist = self.sample_rate / 2
            low = low_freq / nyquist
            high = high_freq / nyquist

            # Clamp frequencies to valid range
            low = np.clip(low, 0.001, 0.999)
            high = np.clip(high, 0.001, 0.999)

            if low >= high:
                continue

            # Butterworth bandpass filter
            sos = signal.butter(4, [low, high], btype="band", output="sos")
            filtered = signal.sosfilt(sos, channel)

            # Apply gain
            gain_linear = 10 ** (gain_db / 20)
            output += filtered * (gain_linear - 1)

        # Add original signal
        return channel + output


class ColoredButton(tk.Frame):
    """Custom colored button that works on macOS"""

    def __init__(self, parent, text, command, bg_color, fg_color="white", width=100):
        super().__init__(parent, bg=bg_color, relief="raised", borderwidth=2)

        self.command = command
        self.bg_color = bg_color
        self.darker_color = self._darken_color(bg_color)

        self.label = tk.Label(self, text=text, bg=bg_color, fg=fg_color, font=("Arial", 10, "bold"), cursor="hand2")
        self.label.pack(padx=15, pady=8)

        # Bind click events
        self.label.bind("<Button-1>", self._on_click)
        self.label.bind("<Enter>", self._on_enter)
        self.label.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _darken_color(self, hex_color):
        """Darken a hex color by 20%"""
        hex_color = hex_color.lstrip("#")
        r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        r, g, b = int(r * 0.8), int(g * 0.8), int(b * 0.8)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _on_click(self, event):
        self.command()

    def _on_enter(self, event):
        self.config(bg=self.darker_color)
        self.label.config(bg=self.darker_color)

    def _on_leave(self, event):
        self.config(bg=self.bg_color)
        self.label.config(bg=self.bg_color)


class WaveformCanvas(tk.Canvas):
    """Custom canvas for drawing waveforms DAW-style"""

    def __init__(self, parent, position_callback=None, **kwargs):
        super().__init__(parent, bg="#1a1a1a", highlightthickness=0, **kwargs)
        self.bind("<Configure>", self._on_resize)
        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)

        self.audio = None
        self.sample_rate = None
        self.cursor_line = None
        self.current_position = 0.0  # 0.0 to 1.0
        self.position_callback = position_callback

    def _on_resize(self, event):
        """Redraw when canvas is resized"""
        if self.audio is not None:
            self.draw_waveform(self.audio, self.sample_rate)
            self.update_cursor(self.current_position)

    def _on_click(self, event):
        """Handle mouse click - set position"""
        width = self.winfo_width()
        position = max(0.0, min(1.0, event.x / width))
        self.current_position = position
        self.update_cursor(position)

        # Notify the app of position change
        if self.position_callback:
            self.position_callback(position)

    def _on_drag(self, event):
        """Handle mouse drag - update position"""
        width = self.winfo_width()
        position = max(0.0, min(1.0, event.x / width))
        self.current_position = position
        self.update_cursor(position)

        # Notify the app of position change
        if self.position_callback:
            self.position_callback(position)

    def draw_waveform(self, audio, sample_rate, title="Waveform"):
        """Draw waveform on canvas"""
        self.delete("all")
        self.cursor_line = None
        self.audio = audio
        self.sample_rate = sample_rate

        width = self.winfo_width()
        height = self.winfo_height()

        if width <= 1 or height <= 1:
            return

        # Downsample for display
        max_points = width * 2
        if len(audio) > max_points:
            step = len(audio) // max_points
            audio_display = audio[::step]
        else:
            audio_display = audio

        # Check if stereo or mono
        if len(audio.shape) == 1:
            # Mono - use full height
            self._draw_channel(audio_display, 0, height, width, "#00ff88", "MONO")
        else:
            # Stereo - split height
            channel_height = height // 2
            self._draw_channel(audio_display[:, 0], 0, channel_height, width, "#00ff88", "LEFT")
            self._draw_channel(audio_display[:, 1], channel_height, channel_height, width, "#00aaff", "RIGHT")

        # Draw title
        self.create_text(width // 2, 20, text=title, fill="white", font=("Arial", 12, "bold"))

    def _draw_channel(self, channel_data, y_offset, channel_height, width, color, label):
        """Draw a single audio channel"""
        if len(channel_data) == 0:
            return

        # Normalize to fit in channel height
        max_val = np.max(np.abs(channel_data))
        if max_val > 0:
            normalized = channel_data / max_val
        else:
            normalized = channel_data

        # Scale to canvas coordinates
        center_y = y_offset + channel_height // 2
        scale = (channel_height // 2) * 0.9  # 90% of available height

        # Draw centerline
        self.create_line(0, center_y, width, center_y, fill="#444444", width=1)

        # Draw waveform
        points = []
        num_samples = len(normalized)
        for i, sample in enumerate(normalized):
            x = int((i / num_samples) * width)
            y = int(center_y - sample * scale)
            points.append((x, y))

        # Draw filled polygon
        if len(points) > 0:
            # Create filled area
            polygon_points = [(0, center_y)] + points + [(width, center_y)]
            flat_points = [coord for point in polygon_points for coord in point]
            self.create_polygon(flat_points, fill=color, outline=color, stipple="gray50")

            # Draw outline
            flat_line = [coord for point in points for coord in point]
            self.create_line(flat_line, fill=color, width=1, smooth=True)

        # Draw label
        self.create_text(10, y_offset + 20, text=label, fill="white", font=("Arial", 9, "bold"), anchor="w")

        # Draw grid lines
        for i in range(5):
            y = y_offset + (channel_height * i // 4)
            self.create_line(0, y, width, y, fill="#2a2a2a", width=1)

    def update_cursor(self, position):
        """Update playback cursor position (0.0 to 1.0)"""
        self.current_position = position
        width = self.winfo_width()
        height = self.winfo_height()

        # Remove old cursor
        if self.cursor_line is not None:
            self.delete(self.cursor_line)

        # Draw new cursor
        x = int(position * width)
        self.cursor_line = self.create_line(x, 0, x, height, fill="#ff0000", width=2, tags="cursor")

    def clear_cursor(self):
        """Remove playback cursor"""
        if self.cursor_line is not None:
            self.delete(self.cursor_line)
            self.cursor_line = None
        self.current_position = 0.0


class EQApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Band EQ")
        self.root.geometry("900x700")

        self.eq = MultiBandEQ()
        self.audio = None
        self.sample_rate = None
        self.processed_audio = None

        # Playback tracking
        self.is_playing = False
        self.playback_start_time = None
        self.audio_duration = 0
        self.playback_position = 0.0  # Current position in seconds

        self._create_widgets()

    def _create_widgets(self):
        # Main container with two columns
        main_container = ttk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Left column - Controls
        left_column = ttk.Frame(main_container)
        left_column.pack(side="left", fill="both", expand=False, padx=(0, 10))

        # Right column - Waveform
        right_column = ttk.Frame(main_container)
        right_column.pack(side="right", fill="both", expand=True)

        # File controls
        file_frame = ttk.Frame(left_column, padding=10)
        file_frame.pack(fill="x")

        ttk.Button(file_frame, text="Load Audio File", command=self.load_file).pack(side="left", padx=5)
        ttk.Button(file_frame, text="Save EQ'd", command=self.save_file).pack(side="left", padx=5)

        # Playback controls
        playback_frame = ttk.LabelFrame(left_column, text="Playback", padding=10)
        playback_frame.pack(fill="x", padx=10, pady=5)

        ColoredButton(playback_frame, "▶ Play", self.play, "#4CAF50").pack(side="left", padx=5)
        ColoredButton(playback_frame, "⏹ Stop", self.stop, "#f44336").pack(side="left", padx=5)

        # EQ sliders
        eq_frame = ttk.Frame(left_column, padding=10)
        eq_frame.pack(fill="both", expand=True)

        ttk.Label(eq_frame, text="EQ Controls (-12 to +12 dB)", font=("Arial", 12, "bold")).pack(pady=10)

        self.sliders = {}
        for band_name in self.eq.bands.keys():
            self._create_band_slider(eq_frame, band_name)

        # Reset button
        ttk.Button(left_column, text="Reset All Bands", command=self.reset_eq).pack(pady=10)

        # Waveform visualization
        waveform_frame = ttk.LabelFrame(right_column, text="Waveform", padding=10)
        waveform_frame.pack(fill="both", expand=True)

        self.waveform_canvas = WaveformCanvas(
            waveform_frame, position_callback=self.on_position_change, width=500, height=600
        )
        self.waveform_canvas.pack(fill="both", expand=True)

        # Status
        self.status_label = ttk.Label(self.root, text="Load an audio file to start", relief="sunken")
        self.status_label.pack(fill="x", side="bottom")

    def _create_band_slider(self, parent, band_name):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=5)

        label = ttk.Label(frame, text=f"{band_name}:", width=12)
        label.pack(side="left")

        slider = tk.Scale(
            frame,
            from_=-12,
            to=12,
            orient="horizontal",
            resolution=0.5,
            length=200,
            command=lambda v, b=band_name: self.on_slider_change(b, v),
        )
        slider.set(0)
        slider.pack(side="left", padx=5)

        value_label = ttk.Label(frame, text="0.0 dB", width=8)
        value_label.pack(side="left")

        self.sliders[band_name] = (slider, value_label)

    def on_slider_change(self, band, value):
        value = float(value)
        self.eq.set_gain(band, value)
        self.sliders[band][1].config(text=f"{value:+.1f} dB")

        if self.audio is not None:
            self.process_audio()
            self.waveform_canvas.draw_waveform(self.processed_audio, self.sample_rate, "EQ'd Waveform")
            # Restore cursor position after redraw
            self.waveform_canvas.update_cursor(
                self.playback_position / self.audio_duration if self.audio_duration > 0 else 0
            )

    def on_position_change(self, position):
        """Callback when user clicks/drags the waveform"""
        self.playback_position = position * self.audio_duration
        time_str = f"{self.playback_position:.2f}s / {self.audio_duration:.2f}s"
        self.status_label.config(text=f"Position: {time_str}")

    def load_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Audio File", filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg"), ("All Files", "*.*")]
        )

        if filepath:
            try:
                self.audio, self.sample_rate = sf.read(filepath)
                self.eq.sample_rate = self.sample_rate
                self.process_audio()
                self.stop()
                self.audio_duration = len(self.processed_audio) / self.sample_rate
                self.playback_position = 0.0
                self.waveform_canvas.draw_waveform(self.processed_audio, self.sample_rate, Path(filepath).name)
                self.waveform_canvas.update_cursor(0.0)
                self.status_label.config(
                    text=f"Loaded: {Path(filepath).name} ({self.sample_rate} Hz, {self.audio_duration:.2f}s)"
                )
            except Exception as e:
                self.status_label.config(text=f"Error loading file: {e}")

    def process_audio(self):
        if self.audio is not None:
            self.processed_audio = self.eq.apply_eq(self.audio)

    def _update_playback_cursor(self):
        """Update the playback cursor position"""
        if not self.is_playing:
            return

        # Calculate elapsed time from start position
        elapsed = time.time() - self.playback_start_time
        current_time = self.playback_position + elapsed

        # Calculate position (0.0 to 1.0)
        if self.audio_duration > 0:
            position = current_time / self.audio_duration
        else:
            position = 0

        # Update cursor
        if position <= 1.0:
            self.waveform_canvas.update_cursor(position)
            time_str = f"{current_time:.2f}s / {self.audio_duration:.2f}s"
            self.status_label.config(text=f"Playing: {time_str}")
            # Schedule next update (60 FPS)
            self.root.after(16, self._update_playback_cursor)
        else:
            # Playback finished
            self.is_playing = False
            self.playback_position = 0.0
            self.waveform_canvas.update_cursor(0.0)
            self.status_label.config(text="Playback finished")

    def play(self):
        """Play the current EQ'd audio from current position"""
        if self.processed_audio is None:
            self.status_label.config(text="No audio loaded")
            return

        # Calculate start sample
        start_sample = int(self.playback_position * self.sample_rate)

        # Get audio from current position to end
        audio_to_play = self.processed_audio[start_sample:]

        if len(audio_to_play) == 0:
            # Already at the end, restart from beginning
            self.playback_position = 0.0
            audio_to_play = self.processed_audio

        sd.stop()
        sd.play(audio_to_play, self.sample_rate)

        # Start cursor animation
        self.is_playing = True
        self.playback_start_time = time.time()
        self._update_playback_cursor()

    def stop(self):
        """Stop playback"""
        sd.stop()
        self.is_playing = False
        self.playback_position = 0.0
        self.waveform_canvas.update_cursor(0.0)
        self.status_label.config(text="Stopped")

    def save_file(self):
        if self.processed_audio is not None:
            filepath = filedialog.asksaveasfilename(
                title="Save EQ'd Audio",
                defaultextension=".wav",
                filetypes=[("WAV File", "*.wav"), ("FLAC File", "*.flac")],
            )
            if filepath:
                sf.write(filepath, self.processed_audio, self.sample_rate)
                self.status_label.config(text=f"Saved: {Path(filepath).name}")
        else:
            self.status_label.config(text="No processed audio to save")

    def reset_eq(self):
        for band_name, (slider, label) in self.sliders.items():
            slider.set(0)
            self.eq.set_gain(band_name, 0)
            label.config(text="0.0 dB")

        if self.audio is not None:
            self.process_audio()
            self.waveform_canvas.draw_waveform(self.processed_audio, self.sample_rate, "Waveform (No EQ)")
            # Restore cursor position after redraw
            self.waveform_canvas.update_cursor(
                self.playback_position / self.audio_duration if self.audio_duration > 0 else 0
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = EQApp(root)
    root.mainloop()
