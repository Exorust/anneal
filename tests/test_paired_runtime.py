"""measure_runtime_paired + get_timing_stats: interleaving, median-of-ratios, cv, roofline. No GPU needed."""

import itertools

from kernelfoundry.eval_pipeline.utils import performance
from kernelfoundry.algorithm.utils.score import get_timing_stats


def test_paired_record_and_stats(monkeypatch):
    # fake clock: every call to perf_counter advances by the duration the last-called fn declared
    clock = itertools.count()
    now = [0.0]
    pending = [0.0]

    def perf_counter():
        now[0] += pending[0]
        pending[0] = 0.0
        return now[0]

    monkeypatch.setattr(performance.time, "perf_counter", perf_counter)
    calls = []

    def make(name, seconds):
        def fn(x, k=None):
            calls.append(name)
            pending[0] += seconds
            return x

        return fn

    out = {}
    rec = performance.measure_runtime_paired(
        make("cand", 0.001),
        {"champion": make("champ", 0.002), "scan": make("scan", 0.004)},
        sync_fn=lambda: None,
        args=(1,),
        kwargs={"k": 2},
        warmup_min_iters=2,
        warmup_min_time=0,
        perf_trials_min_iters=5,
        perf_trials_min_time=0,
        inner_loop_min_time=0,
        bytes_per_call=100_000_000,  # 100 MB per call at 1 ms -> 100 GB/s
        peak_bandwidth_gbs=200.0,
        output=out,
    )
    assert out == rec and rec["paired"] and rec["primary"] == "champion"
    assert len(rec["candidate"]) == 5 and set(rec["baselines"]) == {"champion", "scan"}
    # interleaved: trial order alternates cand,champ,scan / scan,champ,cand after probe+warmup
    timed = calls[3 + 2 * 3 :]
    assert timed[:6] == ["cand", "champ", "scan", "scan", "champ", "cand"]

    stats = get_timing_stats(rec)
    assert abs(stats["speedup"] - 2.0) < 1e-9 and abs(stats["baselines"]["scan"]["speedup"] - 4.0) < 1e-9
    assert stats["ref_speed"] == 2.0 and stats["median"] == 1.0 and stats["cv"] == 0.0
    assert abs(stats["roofline_pct"] - 50.0) < 1e-6
    assert stats["num_trials"] == 5


def test_plain_list_unchanged():
    stats = get_timing_stats([1.0, 2.0, 3.0])
    assert stats["speedup"] == -1 and "paired" not in stats and stats["median"] == 2.0


def _fake_clock(monkeypatch):
    now, pending = [0.0], [0.0]

    def perf_counter():
        now[0] += pending[0]
        pending[0] = 0.0
        return now[0]

    monkeypatch.setattr(performance.time, "perf_counter", perf_counter)
    return pending


def test_arg_sets_rotate_and_every_series_sees_same_sequence(monkeypatch):
    pending = _fake_clock(monkeypatch)
    seen = {"cand": [], "base": []}

    def make(name, seconds):
        def fn(x):
            seen[name].append(x)
            pending[0] += seconds

        return fn

    rec = performance.measure_runtime_paired(
        make("cand", 0.001),
        make("base", 0.002),
        sync_fn=lambda: None,
        arg_sets=[("A",), ("B",), ("C",)],
        warmup_min_iters=1,
        warmup_min_time=0,
        perf_trials_min_iters=6,
        perf_trials_min_time=0,
        inner_loop_min_time=0,
        max_time=0,
    )
    assert rec["n_arg_sets"] == 3 and len(rec["candidate"]) == 6
    timed_c, timed_b = seen["cand"][2:], seen["base"][2:]  # drop probe + 1 warm-up
    assert timed_c == ["A", "B", "C", "A", "B", "C"] and timed_b == timed_c


def test_adaptive_trials_stop_when_ratio_settles(monkeypatch):
    pending = _fake_clock(monkeypatch)
    n = [0]

    def cand(x):
        n[0] += 1
        # probe + 1 warm-up are calls 1-2; the first 10 timed calls cycle 2/3/1 ms, then 1 ms steady
        pending[0] += [0.002, 0.003, 0.001][n[0] % 3] if n[0] <= 12 else 0.001

    def base(x):
        pending[0] += 0.001

    rec = performance.measure_runtime_paired(
        cand, base, sync_fn=lambda: None, args=(0,),
        warmup_min_iters=1, warmup_min_time=0, perf_trials_min_iters=5, perf_trials_min_time=0,
        inner_loop_min_time=0, cv_target=0.05, max_time=1e9, perf_trials_max_iters=50,
    )
    assert 5 < len(rec["candidate"]) < 50, rec["candidate"]
    assert rec["ratio_cv"] <= 0.05


def test_adaptive_trials_respect_round_cap(monkeypatch):
    pending = _fake_clock(monkeypatch)
    n = [0]

    def cand(x):
        n[0] += 1
        pending[0] += 0.001 * (1 + 0.05 * n[0])  # steady drift: the ratio never settles

    def base(x):
        pending[0] += 0.001

    rec = performance.measure_runtime_paired(
        cand, base, sync_fn=lambda: None, args=(0,),
        warmup_min_iters=1, warmup_min_time=0, perf_trials_min_iters=5, perf_trials_min_time=0,
        inner_loop_min_time=0, cv_target=0.01, max_time=1e9, perf_trials_max_iters=20,
    )
    assert len(rec["candidate"]) == 20


def test_paired_screen_caps_and_flags():
    from kernelfoundry.algorithm.evaluator import paired_screen, SPEEDUP_IF_TIMEOUT

    s = {"speedup": 1.3, "roofline_pct": 150.0}
    msg = paired_screen(s, 100.0, 2.0)
    assert s["implausible"] and s["speedup"] == SPEEDUP_IF_TIMEOUT and "IMPLAUSIBLE" in msg
    s = {"speedup": 2.5, "roofline_pct": 40.0}
    assert "verify by hand" in paired_screen(s, 100.0, 2.0) and s["speedup"] == 2.5
    s = {"speedup": 1.05, "roofline_pct": None}
    assert paired_screen(s, 100.0, 2.0) == ""
