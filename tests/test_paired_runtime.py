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
