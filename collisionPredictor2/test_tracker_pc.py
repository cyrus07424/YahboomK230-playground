# test_tracker_pc.py - Offline verification of tracker.py with synthetic data.
# Run on a PC:  python3 test_tracker_pc.py
# No hardware, no model, no video needed. Each scenario feeds fake YOLO
# detections (x, y, w, h, class_id) at 20 fps into CollisionTracker and
# checks whether a warning is / is not produced.

import math
import sys

sys.path.insert(0, ".")
from tracker import CollisionTracker

W, H = 640, 480
FPS = 20.0
CAR = 2


def jitter(i, amp=0.06):
    # deterministic pseudo-noise, mimics YOLO box jitter
    return 1.0 + amp * math.sin(i * 12.9898) * math.cos(i * 3.7)


def run(scenario_fn, seconds=3.0):
    tracker = CollisionTracker(W, H)
    levels = []
    first_alert_t = None
    n = int(seconds * FPS)
    for i in range(n):
        t = i / FPS
        dets = scenario_fn(i, t)
        _results, max_level = tracker.update(dets, t)
        levels.append(max_level)
        if max_level == 2 and first_alert_t is None:
            first_alert_t = t
    return levels, first_alert_t


def box(cx, bottom, w, i, cls=CAR):
    j = jitter(i)
    w2 = w * j
    h2 = w2 * 0.45 * jitter(i + 50, 0.04)
    return (cx - w2 / 2.0, bottom - h2, w2, h2, cls)


def scenario_stationary_large(i, t):
    # Big truck right in front, but distance never changes (same speed).
    return [box(320, 450, 260, i)]


def scenario_approaching(ttc_start=4.0):
    # Closing at constant speed: Z(t) = Z0 * (1 - t/ttc_start) in effect,
    # width grows as 1/Z. "Collision" would occur at t = ttc_start.
    def fn(i, t):
        frac = max(0.05, 1.0 - t / ttc_start)
        w = 70.0 / frac
        if w > 560:
            w = 560
        return [box(320, 450, w, i)]
    return fn


def scenario_adjacent_lane(i, t):
    # Fast approach, but in the next lane (centre far from image centre).
    frac = max(0.05, 1.0 - t / 4.0)
    w = min(560, 70.0 / frac)
    return [box(90, 450, w, i)]


def scenario_receding(i, t):
    w = max(40.0, 260.0 - t * 60.0)
    return [box(320, 450, w, i)]


def scenario_flicker(i, t):
    # Large box appears for only 3 frames, then gone (false detection).
    if i < 3:
        return [box(320, 450, 300, i)]
    return []


def scenario_two_cars(i, t):
    # Stationary large car in front + approaching car in path centre.
    # The approaching one must win; the stationary one must not alert alone
    # (checked separately), here we only require an alert happens.
    return [box(320, 450, 200, i)] + scenario_approaching(4.0)(i, t)


def check(name, levels, first_alert_t, expect_alert, expect_before=None):
    got_alert = first_alert_t is not None
    ok = (got_alert == expect_alert)
    detail = "alert_at=%s" % (("%.2fs" % first_alert_t) if got_alert else "-")
    if ok and expect_alert and expect_before is not None:
        ok = first_alert_t <= expect_before
        detail += " (must be <= %.2fs)" % expect_before
    print("%-28s %s  %s" % (name, "PASS" if ok else "FAIL", detail))
    return ok


def main():
    results = []
    levels, at = run(scenario_stationary_large)
    results.append(check("stationary large vehicle", levels, at, False))

    levels, at = run(scenario_approaching(4.0), seconds=3.5)
    # True TTC reaches 1.8s at t=2.2s; alert must come by ~2.8s,
    # i.e. at least ~1.2s before the (virtual) collision at 4.0s.
    results.append(check("approaching in path", levels, at, True, expect_before=2.8))

    levels, at = run(scenario_adjacent_lane)
    results.append(check("approaching adjacent lane", levels, at, False))

    levels, at = run(scenario_receding)
    results.append(check("receding vehicle", levels, at, False))

    levels, at = run(scenario_flicker, seconds=1.5)
    results.append(check("3-frame flicker", levels, at, False))

    levels, at = run(scenario_two_cars, seconds=3.5)
    results.append(check("stationary + approaching", levels, at, True, expect_before=2.8))

    print()
    if all(results):
        print("ALL %d SCENARIOS PASS" % len(results))
        return 0
    print("FAILED: %d/%d" % (results.count(False), len(results)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
