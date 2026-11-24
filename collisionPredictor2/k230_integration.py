# k230_integration.py - How to wire tracker.py into collision_predictor_1.py
#
# This is an integration fragment, not a standalone program.
# Do NOT replace the working ver1 file. Keep its camera / YOLO / buzzer /
# LED code exactly as-is and make only the three changes below.
# tracker.py itself goes on the SD card next to the main script
# (or in /sdcard/) and contains no device APIs.

# ----------------------------------------------------------------------
# Change 1: import + create the tracker (once, before the main loop)
# ----------------------------------------------------------------------
# from tracker import CollisionTracker
#
# tracker = CollisionTracker(display_size[0], display_size[1])
# start_ms = utime.ticks_ms()
#
# # Level -> LED colour mapping (replaces the single-frame risk colour):
# #   no target : off
# #   level 0   : green  (vehicle in view, not closing)
# #   level 1   : yellow (caution: closing, TTC < 3.5 s)
# #   level 2   : red    (alert:   closing, TTC < 2.2 s, confirmed)

# ----------------------------------------------------------------------
# Change 2: in the main loop, replace the risk calculation
# ----------------------------------------------------------------------
# Remove / stop using these single-frame calls for the buzzer decision:
#     alert_risk_score   = seg.get_max_alert_risk(seg_res)
#     vehicle_risk_score = seg.get_max_vehicle_risk(seg_res)
#     collision_risk     = alert_risk_score >= seg.alert_risk_threshold
#
# Use the tracker instead:
#
#     now_sec = utime.ticks_diff(utime.ticks_ms(), start_ms) / 1000.0
#     dets = []
#     if seg_res[0]:
#         for det, cid in zip(seg_res[0], seg_res[1]):
#             cid = int(cid)
#             if cid in seg.alert_label_ids:
#                 x, y, w, h = det
#                 dets.append((x, y, w, h, cid))
#     track_results, max_level = tracker.update(dets, now_sec)
#
#     collision_risk   = (max_level == 2)
#     vehicle_detected = len(dets) > 0
#     buzzer.update(alerts_enabled and collision_risk)
#
#     if max_level == 2:
#         led.show((255, 0, 0))
#     elif max_level == 1:
#         led.show((255, 200, 0))
#     elif vehicle_detected:
#         led.show((0, 255, 0))
#     else:
#         led.off()
#
# Note: LedIndicator in ver1 takes (visible, risk_score). Either keep
# calling it with a score mapped from max_level (0 -> 0.0, 1 -> 0.5,
# 2 -> 1.0), or call led.led.show_rgb(...) via a small helper as above.
# The mapping version is the smaller change:
#     level_score = (0.0, 0.5, 1.0)[max_level]
#     led.update(vehicle_detected or collision_risk, level_score)

# ----------------------------------------------------------------------
# Change 3 (strongly recommended): draw TTC on screen for tuning
# ----------------------------------------------------------------------
# In draw_result(), next to each label, also draw the track info.
# Store the latest results on the app object in the main loop:
#     seg.last_tracks = track_results
# and in the per-detection drawing loop, match by class + position is
# unreliable, so simplest useful version: draw a summary line at the
# top-left of the OSD with the most dangerous track:
#
#     worst = None
#     for r in track_results:
#         if r.ttc is not None and (worst is None or r.ttc < worst.ttc):
#             worst = r
#     if worst is not None:
#         pl.osd_img.draw_string_advanced(
#             10, 10, 24,
#             "TTC %.1fs x%.2f L%d" % (worst.ttc, worst.growth_ratio, worst.level),
#             color=(255, 255, 80, 80))
#
# Tune thresholds ONLY while watching these numbers on real drives:
# if alerts fire too late, raise ttc_alert (2.2 -> 2.6);
# if adjacent lanes fire, narrow the ego path in tracker.in_ego_path();
# if parked cars fire, raise growth_min (1.30 -> 1.45).
