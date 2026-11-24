# run_tracker_on_csv.py - Evaluate tracker.py on real detection logs.
#
# Usage:
#   python3 run_tracker_on_csv.py detections.csv
#
# CSV format (one detection per row, header required):
#   time_sec,x,y,w,h,class_id
#   0.00,280,300,80,36,2
#   ...
#
# Produce such a CSV by running any YOLO model over a dashcam video on a
# PC (e.g. Ultralytics on Nexar / CCD / your own recordings), then check
# here - before any hardware exists - at what time a warning would have
# fired and how many false warnings a normal-driving video produces.
#
# Output: one line per level change, plus a summary count of alerts.

import csv
import sys

sys.path.insert(0, ".")
from tracker import CollisionTracker

FRAME_W, FRAME_H = 640, 480  # must match the coordinate scale of the CSV


def main(path):
    by_time = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["time_sec"])
            det = (float(row["x"]), float(row["y"]),
                   float(row["w"]), float(row["h"]), int(row["class_id"]))
            by_time.setdefault(t, []).append(det)

    tracker = CollisionTracker(FRAME_W, FRAME_H)
    prev = 0
    alerts = 0
    for t in sorted(by_time.keys()):
        _res, level = tracker.update(by_time[t], t)
        if level != prev:
            print("t=%7.2fs  level %d -> %d" % (t, prev, level))
            if level == 2:
                alerts += 1
            prev = level
    print("alert episodes: %d" % alerts)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
