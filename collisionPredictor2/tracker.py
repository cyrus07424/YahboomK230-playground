# tracker.py - Time-series collision risk tracker for Yahboom K230
#
# Pure Python, no device APIs, no numpy. Runs identically on:
#   - CPython (PC) for offline verification
#   - MicroPython / CanMV on the K230
#
# Idea
# ----
# A per-frame detector (YOLO) only tells us "a car is here, this big".
# Collision risk is NOT "big" - it is "in my path AND closing fast".
#
# For an object at distance Z closing at constant speed:
#     width  w ~ 1/Z        area A ~ 1/Z^2
#     d/dt log(A) = 2 * (closing speed) / Z = 2 / TTC
# so
#     TTC = 2 / slope,  slope = d/dt log(A)   [seconds]
#
# slope is estimated by least-squares regression of log(area) over a
# sliding time window (default 1.0 s), which is robust against the
# frame-to-frame bounding-box jitter of YOLO (+/-10% is normal).
# A single previous-frame comparison must NOT be used.
#
# A warning is issued only when ALL of these hold at the same time:
#   1. the same object has been tracked long enough (>= min_track_sec
#      and >= min_hits frames)
#   2. TTC is below the threshold AND the smoothed area really grew
#      (growth_ratio >= growth_min) over the window
#   3. the object is inside the ego-path trapezoid (narrow far away,
#      wide near the bottom of the image)
#   4. the condition persisted for alert_frames consecutive updates
# Release uses a larger TTC (hysteresis) so the buzzer does not chatter.

import math


def _clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _iou(a, b):
    # a, b: (x, y, w, h)
    ax2 = a[0] + a[2]
    ay2 = a[1] + a[3]
    bx2 = b[0] + b[2]
    by2 = b[1] + b[3]
    ix = min(ax2, bx2) - max(a[0], b[0])
    iy = min(ay2, by2) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    if union <= 0:
        return 0.0
    return inter / union


class TrackResult:
    __slots__ = ("track_id", "class_id", "level", "ttc", "slope",
                 "growth_ratio", "in_path", "box", "age_sec", "hits")

    def __init__(self, track_id, class_id):
        self.track_id = track_id
        self.class_id = class_id
        self.level = 0            # 0=safe 1=caution 2=alert
        self.ttc = None           # seconds, None = not closing / unknown
        self.slope = 0.0          # d/dt log(area), 1/sec
        self.growth_ratio = 1.0   # smoothed area now / area at window start
        self.in_path = False
        self.box = None           # (x, y, w, h)
        self.age_sec = 0.0
        self.hits = 0


class _Track:
    __slots__ = ("track_id", "class_id", "box", "history", "first_t",
                 "last_t", "hits", "alert_streak", "level")

    def __init__(self, track_id, class_id, box, t):
        self.track_id = track_id
        self.class_id = class_id
        self.box = box
        # history entries: (t, log_area, area, cx, bottom)
        self.history = []
        self.first_t = t
        self.last_t = t
        self.hits = 0
        self.alert_streak = 0
        self.level = 0


class CollisionTracker:
    def __init__(self, frame_w, frame_h,
                 window_sec=0.8,
                 min_track_sec=0.5,
                 min_hits=8,
                 ttc_caution=3.5,
                 ttc_alert=2.2,
                 ttc_release=4.5,
                 growth_caution=1.15,
                 growth_min=1.30,
                 alert_frames=3,
                 max_missed_sec=0.3,
                 iou_min=0.20,
                 horizon_ratio=0.35):
        self.frame_w = float(frame_w)
        self.frame_h = float(frame_h)
        self.window_sec = window_sec
        self.min_track_sec = min_track_sec
        self.min_hits = min_hits
        self.ttc_caution = ttc_caution
        self.ttc_alert = ttc_alert
        self.ttc_release = ttc_release
        self.growth_caution = growth_caution
        self.growth_min = growth_min
        self.alert_frames = alert_frames
        self.max_missed_sec = max_missed_sec
        self.iou_min = iou_min
        self.horizon_ratio = horizon_ratio
        self._tracks = {}
        self._next_id = 1

    # ---- ego path -------------------------------------------------
    def in_ego_path(self, cx, bottom):
        """Trapezoid: narrow at the horizon, wide at the image bottom."""
        frame_w = self.frame_w
        frame_h = self.frame_h
        progress = _clamp((bottom / frame_h - self.horizon_ratio) /
                          (1.0 - self.horizon_ratio), 0.0, 1.0)
        # Ego lane at the image bottom is roughly the central 40% of the
        # frame for a centred dashcam; wider than that and adjacent-lane
        # vehicles (centre far off-axis) start to count as "in path".
        half_width = frame_w * (0.05 + 0.17 * progress)
        return abs(cx - frame_w * 0.5) <= half_width

    # ---- regression ----------------------------------------------
    @staticmethod
    def _log_area_slope(history):
        """Least-squares slope of log(area) vs time. None if not enough data."""
        n = len(history)
        if n < 4:
            return None
        st = 0.0
        sy = 0.0
        stt = 0.0
        sty = 0.0
        for (t, log_a, _a, _cx, _b) in history:
            st += t
            sy += log_a
            stt += t * t
            sty += t * log_a
        denom = n * stt - st * st
        if denom <= 1e-9:
            return None
        return (n * sty - st * sy) / denom

    @staticmethod
    def _smoothed_growth(history):
        """Mean area of the last 3 samples / mean area of the first 3."""
        n = len(history)
        if n < 6:
            if n >= 2 and history[0][2] > 0:
                return history[-1][2] / history[0][2]
            return 1.0
        head = (history[0][2] + history[1][2] + history[2][2]) / 3.0
        tail = (history[-1][2] + history[-2][2] + history[-3][2]) / 3.0
        if head <= 0:
            return 1.0
        return tail / head

    # ---- main update ----------------------------------------------
    def update(self, detections, now_sec):
        """detections: iterable of (x, y, w, h, class_id).
        now_sec: monotonic seconds (float).
        Returns (results, max_level): list of TrackResult, int 0..2."""
        dets = []
        for d in detections:
            x, y, w, h, cid = d[0], d[1], d[2], d[3], d[4]
            if w > 0 and h > 0:
                dets.append((float(x), float(y), float(w), float(h), int(cid)))

        # --- association: greedy by IoU, same class only ---
        pairs = []
        track_ids = list(self._tracks.keys())
        for tid in track_ids:
            tr = self._tracks[tid]
            for di in range(len(dets)):
                if dets[di][4] != tr.class_id:
                    continue
                v = _iou(tr.box, dets[di])
                if v >= self.iou_min:
                    pairs.append((v, tid, di))
        pairs.sort(key=lambda p: -p[0])
        used_tracks = set()
        used_dets = set()
        match = {}
        for _v, tid, di in pairs:
            if tid in used_tracks or di in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(di)
            match[tid] = di

        # --- update matched tracks, spawn new ones ---
        for tid, di in match.items():
            tr = self._tracks[tid]
            self._append(tr, dets[di], now_sec)
        for di in range(len(dets)):
            if di in used_dets:
                continue
            d = dets[di]
            tr = _Track(self._next_id, d[4], (d[0], d[1], d[2], d[3]), now_sec)
            self._tracks[self._next_id] = tr
            self._next_id += 1
            self._append(tr, d, now_sec)

        # --- drop stale tracks, evaluate the rest ---
        results = []
        max_level = 0
        dead = []
        for tid, tr in self._tracks.items():
            if now_sec - tr.last_t > self.max_missed_sec and tid not in match \
                    and tr.hits > 0 and (now_sec - tr.last_t) > self.max_missed_sec:
                # keep tracks that were just created this frame (last_t == now)
                if tr.last_t < now_sec - 1e-9:
                    dead.append(tid)
                    continue
            res = self._evaluate(tr, now_sec)
            results.append(res)
            if res.level > max_level:
                max_level = res.level
        for tid in dead:
            del self._tracks[tid]
        return results, max_level

    def _append(self, tr, det, now_sec):
        x, y, w, h = det[0], det[1], det[2], det[3]
        tr.box = (x, y, w, h)
        tr.last_t = now_sec
        tr.hits += 1
        area = w * h
        cx = x + w * 0.5
        bottom = y + h
        tr.history.append((now_sec, math.log(area), area, cx, bottom))
        cutoff = now_sec - self.window_sec
        while tr.history and tr.history[0][0] < cutoff:
            tr.history.pop(0)

    def _evaluate(self, tr, now_sec):
        res = TrackResult(tr.track_id, tr.class_id)
        res.box = tr.box
        res.hits = tr.hits
        res.age_sec = now_sec - tr.first_t
        if not tr.history:
            return res
        _t, _la, _a, cx, bottom = tr.history[-1]
        res.in_path = self.in_ego_path(cx, bottom)

        enough = (res.age_sec >= self.min_track_sec and
                  tr.hits >= self.min_hits and
                  len(tr.history) >= 4 and
                  (tr.history[-1][0] - tr.history[0][0]) >= self.min_track_sec * 0.8)
        if not enough:
            tr.level = 0
            tr.alert_streak = 0
            return res

        slope = self._log_area_slope(tr.history)
        if slope is None:
            return res
        res.slope = slope
        res.growth_ratio = self._smoothed_growth(tr.history)
        ttc = None
        if slope > 0.02:
            ttc = 2.0 / slope
        res.ttc = ttc

        raw = 0
        if res.in_path and ttc is not None:
            if ttc < self.ttc_alert and res.growth_ratio >= self.growth_min:
                raw = 2
            elif ttc < self.ttc_caution and res.growth_ratio >= self.growth_caution:
                raw = 1

        # persistence + hysteresis
        if raw == 2:
            tr.alert_streak += 1
        else:
            tr.alert_streak = 0
        if tr.level == 2:
            # stay in alert until clearly safe again
            if ttc is None or ttc > self.ttc_release or not res.in_path:
                tr.level = raw
            # else keep level 2
        elif raw == 2 and tr.alert_streak >= self.alert_frames:
            tr.level = 2
        elif raw == 2:
            tr.level = 1  # suspect: show caution until confirmed
        else:
            tr.level = raw
        res.level = tr.level
        return res
