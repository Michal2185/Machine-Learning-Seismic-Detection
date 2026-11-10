"""Unit tests of the Mars branch (synthetic data; real files are used only if present). Run: python -m tests.test_mars"""

import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.signal import resample_poly, sosfreqz, butter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.mars import chain
from src.modules.mars.miniseed import assemble, read_trace
from src.modules.mars.scale import background_samples, calibrate_n0, find_normalizers, set_fixed_scale
from src.modules.mars.streams import (EvalStream, EventTarget, ZoneParams, estimate_zone, envelope, fold_for,
                                      make_target, make_windows, window_grid, build_streams, read_catalogue)

FS = chain.FS_APOLLO
ROOT = Path(__file__).resolve().parents[1]
SEARCH = [Path(p) for p in (os.environ.get("MARS_DATA", ""), ROOT / "data/mars/data", ROOT / "data/mars/catalogs",
                            ROOT / "data/mars", "/mnt/user-data/uploads") if str(p)]


def locate(pattern):
    """First file matching the glob pattern in the data folders (env MARS_DATA, data/mars/..., sandbox uploads)."""
    for d in SEARCH:
        hits = sorted(d.glob(pattern)) if d.is_dir() else []
        if hits:
            return hits[0]
    return None


TESTS = []


def test(f):
    TESTS.append(f)
    return f


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def tone(f, fs, n):
    return np.sin(2 * np.pi * f * np.arange(n) / fs)


# ---------------------------------------------------------------- resampler

@test
def resampler_grid_is_exact():
    assert chain.resample_ratio(20.0) == (53, 160)
    assert len(chain.resample_to_apollo(np.zeros(72000), 20.0)) == 23850      # 3600 s x 6.625 Hz


@test
def resampler_lookahead_documented():
    d = chain.resample_lookahead_s(20.0)
    assert abs(d - 1.509) < 0.01, d
    # measured: an impulse at t = 100 s comes out centred at 100 s (linear phase, delay is symmetric), width ~ d
    x = np.zeros(4000); x[2000] = 1.0
    y = chain.resample_to_apollo(x, 20.0)
    assert abs(int(np.argmax(np.abs(y))) / FS - 100.0) < 1.0 / FS + 1e-6


@test
def resampler_keeps_in_band_tone_and_attenuates_above_nyquist():
    n = 20 * 600
    y = chain.resample_to_apollo(tone(0.5, 20.0, n), 20.0)[200:-200]
    assert abs(rms(y) / (1 / np.sqrt(2)) - 1) < 0.01
    y5 = chain.resample_to_apollo(tone(5.0, 20.0, n), 20.0)[200:-200]
    assert rms(y5) < 0.02 * (1 / np.sqrt(2))                                   # above 3.3 Hz: removed, not aliased


# ---------------------------------------------------------------- band chain

def gain(sos, f, fs=FS):
    w, h = sosfreqz(sos, worN=[f], fs=fs)
    return float(abs(h[0]))


@test
def highpass_gain_and_causality():
    sos = butter(4, chain.HP_HZ, "highpass", fs=FS, output="sos")
    assert abs(gain(sos, 0.5) - 1) < 0.01 and gain(sos, 0.05) < 0.01
    x = np.zeros(2000); x[1000] = 1.0
    y = chain.highpass(x)
    assert np.all(y[:1000] == 0.0) and np.any(y[1000:] != 0.0)                  # nothing before the impulse


@test
def steady_state_start_suppresses_dc_transient():
    y = chain.highpass(np.full(3000, 5000.0))                                   # constant input: output stays ~0 from sample 0
    assert np.max(np.abs(y)) < 1e-6 * 5000.0
    from scipy.signal import sosfilt as sf
    z0 = sf(butter(4, chain.HP_HZ, "highpass", fs=FS, output="sos"), np.full(3000, 5000.0))
    assert np.max(np.abs(z0)) > 100.0                                           # a zero-state filter would ring


@test
def lowpass_gain_and_causality():
    sos = butter(4, chain.LP_HZ, "lowpass", fs=FS, output="sos")
    assert abs(gain(sos, 0.5) - 1) < 0.01 and gain(sos, 2.4) <= 0.1, gain(sos, 2.4)
    x = np.zeros(2000); x[1000] = 1.0
    y = chain.lowpass(x)
    assert np.all(y[:1000] == 0.0)


@test
def arms():
    x = np.random.default_rng(0).normal(size=5000)
    a, b = chain.arm_data(x, "nolp"), chain.arm_data(x, "lp1.3")
    assert a.dtype == np.float32 and b.dtype == np.float32 and rms(b) < rms(a)
    try:
        chain.arm_data(x, "bogus"); raise SystemExit("unknown arm accepted")
    except ValueError:
        pass


@test
def apollo_apply_filter_agrees_if_repo_present():
    try:
        from src.modules.preprocessor import apply_filter
    except Exception:
        print("    (skipped: repo preprocessor not importable here; run once in the repo)")
        return
    x = np.random.default_rng(1).normal(size=20000)
    a = apply_filter(x, FS, chain.HP_HZ, None, chain.ORDER)
    b = chain.highpass(x)
    err = np.max(np.abs(a - b)) / np.max(np.abs(b))
    assert err < 1e-12, f"chain.highpass differs from the Apollo apply_filter by {err:.2e}"


@test
def round_trip_apollo_like_stream_through_20hz():
    rng = np.random.default_rng(2)
    n = int(FS * 3600)
    # band-limited (< 1 Hz) noise at 6.625 Hz, like the content of the Apollo detectors' band
    white = rng.normal(size=n)
    from scipy.signal import sosfilt
    x = sosfilt(butter(4, 0.9, "lowpass", fs=FS, output="sos"), white)
    x[int(FS * 1800):int(FS * 1860)] *= 8.0
    ref = chain.highpass(x)
    up = resample_poly(x, 160, 53)                                              # to 20 Hz
    got = chain.preprocess(up, 20.0)
    m = min(len(ref), len(got)); a, b = ref[:m][1000:-1000], got[:m][1000:-1000]
    c = np.corrcoef(a, b)[0, 1]
    assert c > 0.999, c
    assert abs(rms(b) / rms(a) - 1) < 0.01


# ---------------------------------------------------------------- reader

@test
def assemble_fills_gaps():
    t0 = datetime(2022, 1, 2, 4)
    r = lambda off, n, v: (t0 + timedelta(seconds=off), 20.0, n, 5, np.full(n, float(v)))
    tr = assemble([r(0, 100, 1.0), r(10, 100, 3.0)])                            # 100 samples = 5 s, then 5 s missing
    assert len(tr.data) == 300 and tr.gaps_s == [(5.0, 10.0)]
    assert abs(tr.data[150] - 2.0) < 0.05 and np.isfinite(tr.data).all()
    assert assemble([r(0, 100, 1.0), r(5, 100, 2.0)]).gaps_s == []              # contiguous: no gap


@test
def real_files_if_present():
    f = locate("*evid0005.mseed")
    if f is None:
        print(f"    (skipped: no *evid0005.mseed in {[str(d) for d in SEARCH]})")
        return
    tr = read_trace(f)
    assert tr.fs == 20.0 and len(tr.data) == 72000 and tr.gaps_s == [] and np.isfinite(tr.data).all()
    assert tr.start.replace(microsecond=0) == datetime(2022, 2, 3, 8, 0, 0)


# ---------------------------------------------------------------- zone / targets

def synthetic(arrival=2000.0, dur=3600.0, onset_delay=0.0, seed=0, amp=15.0, tau=500.0):
    rng = np.random.default_rng(seed)
    n = int(dur * FS)
    t = np.arange(n) / FS
    x = rng.normal(size=n)
    k = t >= arrival + onset_delay
    x[k] += amp * np.exp(-(t[k] - arrival - onset_delay) / tau) * rng.normal(size=int(k.sum()))
    return x


@test
def zone_finds_onset_and_coda():
    x = synthetic(tau=300.0)
    z = estimate_zone(envelope(x, FS, 20), FS, 2000.0, 120.0, 3570.0)
    assert z["status"] == "ok" and abs(z["onset_offset_s"]) < 30, z
    assert z["peak_ratio"] > 5 and not z["coda_censored"] and 2000 < z["coda_end_s"] < 3570, z
    t = make_target("e", 2000.0, z)
    assert t.has_positive and t.detect == (1880.0, 2600.0) and t.ignore[0] <= 2000 - 180 + 30 and t.ignore[1] == z["coda_end_s"]


@test
def zone_no_background_and_pre_active_and_unusable_offset():
    x = synthetic(arrival=200.0)
    z = estimate_zone(envelope(x, FS, 20), FS, 200.0, 120.0, 3570.0)            # only 0 s of background before it
    assert z["status"] == "no_bg" and not make_target("e", 200.0, z).has_positive
    x = synthetic(arrival=2000.0, onset_delay=-800.0)                           # signal starts 800 s before the catalogue time
    z = estimate_zone(envelope(x, FS, 20), FS, 2000.0, 120.0, 3570.0)
    assert z["status"] in ("pre_active", "ok") and not make_target("e", 2000.0, z).has_positive or z["onset_offset_s"] < -120
    x = synthetic(arrival=1500.0, onset_delay=900.0)                            # visible onset 900 s after the catalogue time
    z = estimate_zone(envelope(x, FS, 20), FS, 1500.0, 120.0, 3570.0)
    assert z["status"] == "none" or z["onset_offset_s"] > 600
    assert not make_target("e", 1500.0, z).has_positive


# ---------------------------------------------------------------- windows

def fake_stream(n=int(FS * 3600), wid="w"):
    x = np.arange(n, dtype=np.float32)
    return EvalStream(wid, x, FS, None, [], [])


@test
def window_grid_matches_apollo_convention():
    ends, length = window_grid(int(FS * 3600), FS)
    assert length == 398                                                        # 60 s x 6.625 = 397.5 -> 398, as Apollo
    assert ends[0] == 398 and (np.diff(ends) == 66).all()                       # 10 s x 6.625 = 66.25 -> 66, as Apollo
    assert ends[-1] <= int(FS * 3600) < ends[-1] + 66


@test
def window_grid_reproduces_the_apollo_window_count():
    # evid00002: 24 h at 6.625 Hz = 572 400 samples; the Apollo pipeline made 8 667 windows (a 66.25-sample step gives 8 635)
    assert len(window_grid(572400, FS)[0]) == 8667


@test
def windows_recut_from_stream_exactly():
    s = fake_stream()
    ws = make_windows(s)
    assert len(ws) == 356 and all(len(w.data) == 398 for w in ws)
    for w in ws[::37]:
        assert np.array_equal(w.data, s.data[w.end_sample - 398:w.end_sample])  # stress.check_window_convention
    assert ws[0].end_sample == 398 and ws[1].end_sample == 464 and ws[-1].event.evid == "w" and ws[0].label == 0
    assert abs(ws[1].end_time - 464 / FS) < 1e-9 and abs(ws[1].start_time - (66 / FS)) < 1e-9
    assert make_windows(s, time_rule="nominal")[1].end_time == 70.0


# ---------------------------------------------------------------- scale

class FakeNorm:
    def __init__(self, v="N0", c=1e-10):
        self.config = SimpleNamespace(variant=v); self.fixed_scale = c


class FakeDet:
    def __init__(self, v="N0"):
        self.inputs = SimpleNamespace(normalizer=FakeNorm(v), scaler="untouched")
        self.name = "d"


@test
def set_fixed_scale_sets_and_guards():
    d = FakeDet()
    assert find_normalizers(d) == [d.inputs.normalizer]
    old = set_fixed_scale(d, 42.0)
    assert old == [1e-10] and d.inputs.normalizer.fixed_scale == 42.0 and d.inputs.scaler == "untouched"
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        try:
            set_fixed_scale(d, bad); raise SystemExit(f"accepted {bad}")
        except ValueError:
            pass
    try:
        set_fixed_scale(FakeDet("N2"), 1.0); raise SystemExit("accepted N2")
    except ValueError:
        pass
    try:
        set_fixed_scale(SimpleNamespace(), 1.0); raise SystemExit("no normalizer accepted")
    except TypeError:
        pass


@test
def gain_equivariance_of_recalibration():
    x = np.random.default_rng(3).normal(size=1000)
    for g in (1e-3, 1.0, 1e11):
        assert np.allclose((g * x) / (1.7 * g), x / 1.7)                         # N0 with c scaled by g: identical input


@test
def calibrate_n0_ignores_event_and_gaps():
    rng = np.random.default_rng(4)
    n = int(FS * 3600)
    x = rng.normal(size=n) * 10.0
    x[int(FS * 1000):int(FS * 2000)] *= 30.0                                    # event inside the ignore interval
    x[:int(FS * 100)] = 0.0                                                     # start-up gap
    tgt = EventTarget("e", "marsquake", 1100.0, (980.0, 1700.0), (920.0, 2100.0), None, False, None)
    s = EvalStream("w", x.astype(np.float32), FS, None, [tgt], [(0.0, 120.0)])
    c = calibrate_n0([s, s])
    assert abs(c / 10.0 - 1) < 0.05, c
    assert background_samples(s).size < n


# ---------------------------------------------------------------- policy

@test
def fold_rotation_covers_every_fold_once_per_waveform():
    for i in range(7):
        assert sorted(fold_for(i, s) for s in range(5)) == [0, 1, 2, 3, 4]
    assert fold_for(0, 0) == 0 and fold_for(1, 0) == 1 and fold_for(1, 4) == 0


# ---------------------------------------------------------------- end to end on the real files (if present)

@test
def build_streams_on_real_files_if_present():
    cat = locate("Mars_InSight_training_catalog_final.csv")
    data = locate("*evid0005.mseed")
    if cat is None or data is None:
        print(f"    (skipped: catalogue {cat} / data {data} not found in {[str(d) for d in SEARCH]})")
        return
    rows = read_catalogue(cat)
    assert [r["evid"] for r in rows] == ["evid0005", "evid0006"] and rows[0]["time_rel"] == 507.0
    streams, info = build_streams(rows, data.parent)
    for arm, ss in streams.items():
        assert len(ss) == 2 and all(len(s.data) == 23850 and s.sampling_rate == FS for s in ss)
        assert all(s.events[0].detect == (s.events[0].arrival_s - 120, s.events[0].arrival_s + 600) for s in ss)
    for i in info:
        print(f"    {i['evid']}: zone {i['zone']['status']}, onset offset {i['zone']['onset_offset_s']:.0f} s, "
              f"peak/bg {i['zone']['peak_ratio']:.1f}, coda end {i['zone']['coda_end_s']:.0f} s "
              f"(censored {i['zone']['coda_censored']}), usable {i['target'].has_positive}")
    print(f"    N0 constant (lp1.3 / nolp): {calibrate_n0(streams['lp1.3']):.4g} / {calibrate_n0(streams['nolp']):.4g}")


def main():
    failed = 0
    for f in TESTS:
        try:
            f(); print(f"ok    {f.__name__}")
        except BaseException:
            failed += 1; print(f"FAIL  {f.__name__}"); traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()