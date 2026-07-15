#!/usr/bin/env python3
"""Render the trot-MPC benchmark's predicted horizon (trot_mpc_horizon.csv,
from srbd_mpc::tests::mpc_backend_bench with MPC_BENCH_CSV_OUT set) as an
MP4: the SRBD body as a simple box (this is what the model *is* --
single rigid body + point-mass GRFs, no leg kinematics at all, so a box
is the honest representation, not a simplification of a fancier one),
the 4 feet as dots, and GRF vectors as arrows, over the 10-step horizon
the benchmarked QP snapshot predicts.
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--trace", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames-dir", required=True)
args = ap.parse_args()

rows = list(csv.DictReader(open(args.trace)))
rows = [{k: float(v) for k, v in r.items()} for r in rows]
n = len(rows)

LEGS = ["FL", "FR", "RL", "RR"]
LEG_COLOR = {"FL": "#3987e5", "FR": "#e69f00", "RL": "#009e73", "RR": "#d55e00"}

# Body box half-extents (typical Cheetah-class trunk, matches the
# hip-offset fixtures the benchmark itself uses: 0.18 x 0.10).
HX, HY, HZ = 0.20, 0.11, 0.05

BOX_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),  # bottom face
    (4, 5), (5, 7), (7, 6), (6, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]


def rot_zyx(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def box_corners(center, r):
    local = np.array([
        [-HX, -HY, -HZ], [HX, -HY, -HZ], [-HX, HY, -HZ], [HX, HY, -HZ],
        [-HX, -HY, HZ], [HX, -HY, HZ], [-HX, HY, HZ], [HX, HY, HZ],
    ])
    return (r @ local.T).T + center


FRAMES_DIR = Path(args.frames_dir)
FRAMES_DIR.mkdir(exist_ok=True, parents=True)
for f in FRAMES_DIR.glob("*.png"):
    f.unlink()

path_pts = np.array([[row["pos_x"], row["pos_y"], row["pos_z"]] for row in rows])
all_feet = []
for row in rows:
    for leg in LEGS:
        p = np.array([row["pos_x"], row["pos_y"], row["pos_z"]]) + np.array(
            [row[f"r_{leg}_x"], row[f"r_{leg}_y"], row[f"r_{leg}_z"]]
        )
        all_feet.append(p)
all_feet = np.array(all_feet)
all_pts = np.vstack([path_pts, all_feet])
xmin, ymin, zmin = all_pts.min(axis=0) - 0.15
xmax, ymax, zmax = all_pts.max(axis=0) + 0.15
zmin = min(zmin, -0.02)

SUBSTEPS = 6  # interpolate between horizon steps for smoother playback
FPS = 15

frame_i = 0
for k in range(n - 1):
    r0, r1 = rows[k], rows[k + 1]
    for s in range(SUBSTEPS):
        a = s / SUBSTEPS
        row = {key: r0[key] * (1 - a) + r1[key] * a for key in r0}
        # Stance flags don't interpolate -- hold the earlier step's
        # until the switch (a discrete gait event).
        for leg in LEGS:
            row[f"stance_{leg}"] = r0[f"stance_{leg}"]

        fig = plt.figure(figsize=(7.2, 7.2), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("#0e1420")
        ax.set_facecolor("#0e1420")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.06, 0.08, 0.13, 1.0))
            axis._axinfo["grid"]["color"] = (1, 1, 1, 0.08)
            axis.label.set_color("#7e8b9b")
        ax.tick_params(colors="#4d5a6b", labelsize=7)

        # Ground grid.
        gx = np.linspace(xmin, xmax, 6)
        gy = np.linspace(ymin, ymax, 6)
        for x in gx:
            ax.plot([x, x], [ymin, ymax], [0, 0], color="#2a3446", lw=0.6)
        for y in gy:
            ax.plot([xmin, xmax], [y, y], [0, 0], color="#2a3446", lw=0.6)

        # Predicted CoM path (full horizon, faint).
        ax.plot(path_pts[:, 0], path_pts[:, 1], path_pts[:, 2],
                color="#5b6ee8", alpha=0.4, lw=1.4, linestyle="--")

        center = np.array([row["pos_x"], row["pos_y"], row["pos_z"]])
        r = rot_zyx(row["roll"], row["pitch"], row["yaw"])
        corners = box_corners(center, r)
        for i, j in BOX_EDGES:
            ax.plot(*zip(corners[i], corners[j]), color="#e4eaf0", lw=1.8)

        for leg in LEGS:
            foot = center + np.array([row[f"r_{leg}_x"], row[f"r_{leg}_y"], row[f"r_{leg}_z"]])
            stance = row[f"stance_{leg}"] > 0.5
            color = LEG_COLOR[leg]
            ax.scatter([foot[0]], [foot[1]], [foot[2]], color=color,
                       s=60 if stance else 28, alpha=1.0 if stance else 0.4,
                       depthshade=False)
            if stance:
                fvec = np.array([row[f"F_{leg}_x"], row[f"F_{leg}_y"], row[f"F_{leg}_z"]])
                scale = 1.0 / 250.0  # N -> metres, tuned for a readable arrow length
                tip = foot + fvec * scale
                ax.plot(*zip(foot, tip), color=color, lw=2.4)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_box_aspect([xmax - xmin, ymax - ymin, zmax - zmin])
        ax.view_init(elev=18, azim=-60)
        ax.set_title(
            f"quadruped-gait SRBD trot MPC -- predicted horizon step {k + a:4.1f}/9",
            color="#e4eaf0", fontsize=11, family="monospace",
        )

        fig.tight_layout()
        fig.savefig(FRAMES_DIR / f"f{frame_i:05d}.png", facecolor=fig.get_facecolor())
        plt.close(fig)
        frame_i += 1

# Hold the last real step for a beat.
for _ in range(SUBSTEPS):
    import shutil
    shutil.copy(FRAMES_DIR / f"f{frame_i - 1:05d}.png", FRAMES_DIR / f"f{frame_i:05d}.png")
    frame_i += 1

print(f"Rendered {frame_i} frames, encoding...", file=sys.stderr)
subprocess.run([
    "ffmpeg", "-y", "-framerate", str(FPS),
    "-i", str(FRAMES_DIR / "f%05d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
    str(args.out),
], check=True)
print(f"Done: {args.out}", file=sys.stderr)
