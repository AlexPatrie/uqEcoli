import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import pprint
    import dataclasses as dc, typing as ty 
    from enum import StrEnum

    import marimo as mo
    import numpy as np 
    import polars as pl

    return StrEnum, dc, mo, np, pl, pprint


@app.cell
def _(StrEnum, dc, np, pl, pprint):
    class BinBand(StrEnum):
        LOW = 'low'
        MID = 'mid'
        HIGH = 'high'

    @dc.dataclass
    class Bins:
        # TODO: instead of low, mid, high have continuous spectrum (with granularity)
        low: np.ndarray
        mid: np.ndarray
        high: np.ndarray

        def vectorize(self):
            bins = [getattr(self, binType) for binType in [BinBand.LOW, BinBand.MID, BinBand.HIGH]]
            vec = []
            for row in bins:
                for val in row:
                    vec.append(val)
            # self.vector = np.ndarray(vec)
            return np.arange(self.low.min(), self.high.max(), 1)
        
    class Spectrum:
        data: np.ndarray
        dt: float

        def __init__(self, data, dt: float):
            self.data = data
            self.dt = dt 

        def __repr__(self) -> str:
            bins = self.bins
            def get_range(r: str):
                v = getattr(bins, r)
                return f"{int(v.min())} - {int(v.max())}"
            low, mid, high = list(map(
                lambda c: get_range(c), 
                ["low", "mid", "high"]
            ))
            return f"Spectrum:\n\n{pprint.pformat(self.data)}\n=====\nBins:\n\nlow: {low}\nmid: {mid}\nhigh: {high}"

        @property
        def frequencies(self):
            # freqs is now in Hz, same length as spectrum
            sample_rate: int = 44100
            # d=1/sample_rate
            return np.fft.rfftfreq(len(self.data), d=1/sample_rate)

        @property
        def bins(self):
            mid_thresh = 300
            high_thresh = mid_thresh * 10
            freqs = self.frequencies
            low_bins  = np.where(freqs < mid_thresh)[0]
            mid_bins  = np.where((freqs >= mid_thresh) & (freqs < high_thresh))[0]
            high_bins = np.where(freqs >= high_thresh)[0]
            return Bins(low=low_bins, mid=mid_bins, high=high_bins)

        # def adjust(self, band: BinBand, dg: float, i_bin: int | None = None):
        #     selected_bins = getattr(self.bins, band)
        #     if i_bin is not None:
        #         selected_bins = selected_bins[i_bin]
        #     self.data[selected_bins] *= dg

        def adjust(self, dg: float, band: BinBand | None = None, i_bin: int | None = None):
            if i_bin is not None:
                selected_bin = self.bins.vectorize()[i_bin]
                self.data[selected_bin] *= dg
            else:
                selected_bins = getattr(self.bins, band)
                self.data[selected_bins] *= dg

        @property
        def magnitude(self):
            return 20 * np.log10(np.abs(self.data[1:len(self.frequencies)]) + 1e-10)

        def plot(self, obs: str):
            import plotly.graph_objects as go
            freqs = self.frequencies[1:]
            magnitude_db = self.magnitude

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=freqs, y=magnitude_db, fill='tozeroy',
                                   line=dict(color='cyan', width=1)))
            fig.update_xaxes(type='log', range=[np.log10(20), np.log10(20000)])
            fig.update_layout(template='plotly_dark', title=f'Spectrum Analyzer: {obs}')
            return fig

    class TimeseriesDataset:
        df: pl.DataFrame
        t: np.ndarray[float]
        observables: list[str]

        def __init__(self, data: pl.DataFrame, t_colname: str = "time"):
            self.df = data
            self.t = self.df.select(t_colname).to_numpy()
            self.dt = self.t[1] - self.t[0]
            self.observables = [c for c in self.df.columns if not c == t_colname]

        @property
        def timeseries(self) -> np.ndarray:
            return self.df.select(self.observables)

        def to_spectrum(self, observable_name: str) -> Spectrum:
            if observable_name not in self.observables:
                raise ValueError(f"{observable_name} not found in data cols. Valid options: {self.observables}")
            ts = self.df.select(observable_name).to_numpy().flatten()
            return Spectrum(data=np.fft.fft(ts), dt=self.dt)  # shape (T, N)

        def from_spectrum(self, s: Spectrum) -> np.ndarray:
            return np.fft.ifft(s.data).real

        def __repr__(self) -> str:
            return pprint.pformat(self.df)

    def test_spectral_transform():
        _T = 1111  # total duration
        _dt = 1.0  # global timestep/interval
        _t = (lambda T, dt: np.arange(start=0.0, stop=float(T), step=dt))(T, dt)

        _N = 3  # num observables 
        _observable_names = ['a', 'b', 'c']
        _timeseries_dataset = pl.DataFrame({
            "time": t, 
            **dict(zip(
                observable_names,
                np.random.random((N, T))
            ))
        })
        ds = TimeseriesDataset(data=_timeseries_dataset)
        _spectrum = ds.to_spectrum('a')
        _ts_a = ds.df.select('a').to_numpy()
        ds.from_spectrum(_spectrum) == _ts_a
        assert np.all(_ts_a == ds.df.select('a').to_numpy())


    T = 1111  # total duration
    dt = 1.0  # global timestep/interval
    t = (lambda T, dt: np.arange(start=0.0, stop=float(T), step=dt))(T, dt)

    N = 3  # num observables 
    observable_names = ['a', 'b', 'c']
    timeseries_dataset = pl.DataFrame({
        "time": t, 
        **dict(zip(
            observable_names,
            np.random.random((N, T))
        ))
    })

    timeseries = TimeseriesDataset(data=timeseries_dataset)
    timeseries
    return BinBand, Spectrum, timeseries


@app.cell
def _(timeseries):
    spectrum = timeseries.to_spectrum('a')
    spectrum
    return (spectrum,)


@app.cell
def _(spectrum, timeseries):
    ts_a = timeseries.from_spectrum(spectrum)
    ts_a
    return


@app.cell
def _(mo):
    low_slider = mo.ui.slider(label="LOW", value=1.0, start=0.1, stop=10.0, step=0.1, show_value=True)
    mid_slider = mo.ui.slider(label="MID", value=1.0, start=0.1, stop=10.0, step=0.1, show_value=True)
    high_slider = mo.ui.slider(label="HIGH", value=1.0, start=0.1, stop=10.0, step=0.1, show_value=True)
    return high_slider, low_slider, mid_slider


@app.cell
def _(BinBand, Spectrum, mo, spectrum):
    class AllSliders(dict):
        def flatten(self):
            s = []
            for row in self.values():
                for r in row:
                    s.append(r)
            return s 
        
    class BinSliders:
        def generate_range_sliders(self, band: BinBand, spectrum: Spectrum) -> list[mo.ui.slider]:
                sliders = []
                bins = getattr(spectrum.bins, band)
                for i, b in enumerate(bins):
                    freqs = spectrum.frequencies[bins]
                    bin_min = -freqs.min()
                    bin_max = freqs.max() * 4
                    on_change = (lambda val: spectrum)
                    b_slider = mo.ui.slider(full_width=True, label=f"     {band.value}:{b}     ", start=bin_min, stop=bin_max, step=1.0, show_value=True, value=spectrum.frequencies[bins][i])
                    sliders.append(b_slider)
                return sliders

        def get_all_sliders(self):
            bands = [BinBand.LOW, BinBand.MID, BinBand.HIGH]
            return dict(zip(
                bands, 
                [self.generate_range_sliders(band, spectrum) for band in bands],
            
            ))

        @property
        def all(self):
            return AllSliders(self.get_all_sliders())

        def get_ui(self):
            return list(map(
                # lambda band: mo.accordion({band: self.generate_range_sliders(band, spectrum)}),
                lambda band: self.generate_range_sliders(band, spectrum),
                [BinBand.LOW, BinBand.MID, BinBand.HIGH]
            ))

        @property
        def ui(self):
            return mo.vstack(self.get_ui(), justify="start")


    sliders = BinSliders()
    return


@app.cell
def _(BinBand, high_slider, low_slider, mid_slider, mo, timeseries):
    def generate_spectrum_analyzer(observable: str):
        # Create a fresh spectrum each time the slider changes
        _spectrum = timeseries.to_spectrum(observable)
        _spectrum.adjust(band=BinBand.LOW, dg=low_slider.value)
        _spectrum.adjust(band=BinBand.MID, dg=mid_slider.value)
        _spectrum.adjust(dg=high_slider.value, i_bin=100)
    
        return mo.vstack([
            mo.hstack([low_slider, mid_slider, high_slider], justify="start"), 
            _spectrum.plot(obs=observable)
        ], justify="start")

    observable = 'b'
    generate_spectrum_analyzer(observable)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
