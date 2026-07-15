#!/usr/bin/env python3
"""Render the trot-MPC benchmark's predicted horizon
(trot_mpc_horizon.csv) using the REAL Go2 trunk mesh (base_0..4.obj,
via the mesh manifest go2_leg_singularity_demo.rs already writes)
instead of a generic box -- the SRBD model itself has no leg
kinematics (single rigid body + point-mass GRFs), so legs are still
not drawn, but the body itself is now recognizably Go2, not an
abstract placeholder.
"""
import argparse
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np
import vtk

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--trace", required=True)
ap.add_argument("--manifest", required=True, help="go2_mesh_manifest.csv (reused from go2_leg_singularity_demo)")
ap.add_argument("--out", required=True)
ap.add_argument("--frames-dir", required=True)
args = ap.parse_args()

rows = list(csv.DictReader(open(args.trace)))
rows = [{k: float(v) for k, v in r.items()} for r in rows]
n = len(rows)

LEGS = ["FL", "FR", "RL", "RR"]
LEG_COLOR = {"FL": (0.22, 0.53, 0.9), "FR": (0.9, 0.62, 0.0), "RL": (0.0, 0.62, 0.45), "RR": (0.84, 0.33, 0.0)}

SUBSTEPS = 6
FPS = 15


def rot_zyx(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def vtk_matrix(t, r):
    m = vtk.vtkMatrix4x4()
    m.Identity()
    for i in range(3):
        for j in range(3):
            m.SetElement(i, j, r[i, j])
        m.SetElement(i, 3, t[i])
    return m


# ---- load only the base (trunk) meshes from the manifest ----
base_meshes = []
with open(args.manifest) as f:
    for row in csv.DictReader(f):
        if int(row["parent_joint"]) != 0:
            continue
        base_meshes.append(row["mesh_path"])

renderer = vtk.vtkRenderer()
renderer.SetBackground(0x0e / 255, 0x14 / 255, 0x20 / 255)
render_window = vtk.vtkRenderWindow()
render_window.SetOffScreenRendering(1)
render_window.AddRenderer(renderer)
render_window.SetSize(1000, 1000)
render_window.SetMultiSamples(8)

body_actors = []
for path in base_meshes:
    reader = vtk.vtkOBJReader()
    reader.SetFileName(path)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(reader.GetOutputPort())
    normals.ConsistencyOn()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.62, 0.63, 0.66)
    actor.GetProperty().SetAmbient(0.28)
    actor.GetProperty().SetDiffuse(0.72)
    actor.GetProperty().SetSpecular(0.2)
    actor.GetProperty().SetSpecularPower(12)
    renderer.AddActor(actor)
    body_actors.append(actor)
print(f"loaded {len(body_actors)} trunk mesh pieces", file=sys.stderr)

# ---- predicted CoM path (static, full horizon) ----
path_pts = np.array([[r["pos_x"], r["pos_y"], r["pos_z"]] for r in rows])
path_vtk_pts = vtk.vtkPoints()
for p in path_pts:
    path_vtk_pts.InsertNextPoint(*p)
path_lines = vtk.vtkCellArray()
path_lines.InsertNextCell(len(path_pts))
for i in range(len(path_pts)):
    path_lines.InsertCellPoint(i)
path_poly = vtk.vtkPolyData()
path_poly.SetPoints(path_vtk_pts)
path_poly.SetLines(path_lines)
path_tube = vtk.vtkTubeFilter()
path_tube.SetInputData(path_poly)
path_tube.SetRadius(0.003)
path_tube.SetNumberOfSides(8)
path_mapper = vtk.vtkPolyDataMapper()
path_mapper.SetInputConnection(path_tube.GetOutputPort())
path_actor = vtk.vtkActor()
path_actor.SetMapper(path_mapper)
path_actor.GetProperty().SetColor(0.36, 0.43, 0.91)
path_actor.GetProperty().SetOpacity(0.5)
renderer.AddActor(path_actor)

# ---- per-leg: foot marker + GRF arrow (line + cone tip) ----
foot_actors = {}
arrow_actors = {}
for leg in LEGS:
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(0.018)
    sphere.SetThetaResolution(16)
    sphere.SetPhiResolution(16)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*LEG_COLOR[leg])
    actor.GetProperty().SetAmbient(0.5)
    renderer.AddActor(actor)
    foot_actors[leg] = actor

    line_src = vtk.vtkLineSource()
    mapper2 = vtk.vtkPolyDataMapper()
    mapper2.SetInputConnection(line_src.GetOutputPort())
    line_actor = vtk.vtkActor()
    line_actor.SetMapper(mapper2)
    line_actor.GetProperty().SetColor(*LEG_COLOR[leg])
    line_actor.GetProperty().SetLineWidth(3.5)
    renderer.AddActor(line_actor)
    arrow_actors[leg] = (line_src, line_actor)

# ---- ground grid ----
grid = vtk.vtkPlaneSource()
grid.SetOrigin(-0.5, -0.4, 0.0)
grid.SetPoint1(0.5, -0.4, 0.0)
grid.SetPoint2(-0.5, 0.4, 0.0)
grid.SetXResolution(10)
grid.SetYResolution(8)
grid_mapper = vtk.vtkPolyDataMapper()
grid_mapper.SetInputConnection(grid.GetOutputPort())
grid_actor = vtk.vtkActor()
grid_actor.SetMapper(grid_mapper)
grid_actor.GetProperty().SetRepresentationToWireframe()
grid_actor.GetProperty().SetColor(0.25, 0.3, 0.38)
grid_actor.GetProperty().SetOpacity(0.35)
renderer.AddActor(grid_actor)

# ---- lighting (matches render_go2_vtk.py) ----
key_light = vtk.vtkLight()
key_light.SetPosition(1.0, -1.2, 1.4)
key_light.SetFocalPoint(0.0, 0, 0.1)
key_light.SetIntensity(0.9)
key_light.SetColor(1.0, 1.0, 0.98)
renderer.AddLight(key_light)
fill_light = vtk.vtkLight()
fill_light.SetPosition(-1.0, 1.2, 0.8)
fill_light.SetFocalPoint(0.0, 0, 0.1)
fill_light.SetIntensity(0.35)
fill_light.SetColor(0.75, 0.82, 1.0)
renderer.AddLight(fill_light)

text_actor = vtk.vtkTextActor()
text_actor.SetPosition(24, 950)
text_actor.GetTextProperty().SetFontSize(20)
text_actor.GetTextProperty().SetColor(0.9, 0.92, 0.94)
text_actor.GetTextProperty().SetFontFamilyToCourier()
renderer.AddActor2D(text_actor)

# ---- camera ----
all_feet = []
for r in rows:
    for leg in LEGS:
        all_feet.append([r["pos_x"] + r[f"r_{leg}_x"], r["pos_y"] + r[f"r_{leg}_y"], r["pos_z"] + r[f"r_{leg}_z"]])
all_pts = np.vstack([path_pts, np.array(all_feet)])
center = all_pts.mean(axis=0)
center[2] += 0.12  # bias upward: the trunk mesh sits mostly above the CoM point
extent = max((all_pts.max(axis=0) - all_pts.min(axis=0)).max(), 0.5)
# The extent above only samples path/foot points, not the trunk mesh's
# own vertices -- pad generously so the body (roughly 0.4m long) isn't
# clipped/dominant relative to the feet.
extent += 0.35
elev = np.radians(18)
azim = np.radians(-55)
distance = extent * 1.6 + 0.5
direction = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
cam_pos = center + distance * direction
camera = renderer.GetActiveCamera()
camera.SetFocalPoint(*center)
camera.SetPosition(*cam_pos)
camera.SetViewUp(0, 0, 1)
camera.SetViewAngle(35)

w2i = vtk.vtkWindowToImageFilter()
w2i.SetInput(render_window)
w2i.SetInputBufferTypeToRGB()
writer = vtk.vtkPNGWriter()

FRAMES_DIR = Path(args.frames_dir)
FRAMES_DIR.mkdir(exist_ok=True, parents=True)
for f in FRAMES_DIR.glob("*.png"):
    f.unlink()

frame_i = 0
FORCE_SCALE = 1.0 / 250.0
for k in range(n - 1):
    r0, r1 = rows[k], rows[k + 1]
    for s in range(SUBSTEPS):
        a = s / SUBSTEPS
        row = {key: r0[key] * (1 - a) + r1[key] * a for key in r0}
        for leg in LEGS:
            row[f"stance_{leg}"] = r0[f"stance_{leg}"]

        body_t = np.array([row["pos_x"], row["pos_y"], row["pos_z"]])
        body_r = rot_zyx(row["roll"], row["pitch"], row["yaw"])
        mat = vtk_matrix(body_t, body_r)
        for actor in body_actors:
            actor.SetUserMatrix(mat)

        for leg in LEGS:
            foot = body_t + np.array([row[f"r_{leg}_x"], row[f"r_{leg}_y"], row[f"r_{leg}_z"]])
            stance = row[f"stance_{leg}"] > 0.5
            foot_actors[leg].SetPosition(*foot)
            foot_actors[leg].GetProperty().SetOpacity(1.0 if stance else 0.35)

            line_src, line_actor = arrow_actors[leg]
            if stance:
                fvec = np.array([row[f"F_{leg}_x"], row[f"F_{leg}_y"], row[f"F_{leg}_z"]])
                tip = foot + fvec * FORCE_SCALE
                line_src.SetPoint1(*foot)
                line_src.SetPoint2(*tip)
                line_actor.VisibilityOn()
            else:
                line_actor.VisibilityOff()

        text_actor.SetInput(f"quadruped-gait SRBD trot MPC (Go2 trunk) -- predicted horizon step {k + a:4.1f}/9")

        render_window.Render()
        w2i.Modified()
        w2i.Update()
        writer.SetFileName(str(FRAMES_DIR / f"f{frame_i:05d}.png"))
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        frame_i += 1

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
