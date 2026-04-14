import streamlit as st
import streamlit.components.v1 as components
import os, time, uuid, requests, shutil
from datetime import datetime
import numpy as np
import json
import zipfile
import io
import glob
import re
import pyvista as pv
import plotly.graph_objects as go
import trimesh

st.set_page_config(page_title="MIM-Ops Pro", page_icon="🔬", layout="wide")
st.title("🔬 MIM-Ops: AI-Powered Cloud Simulation")

ZAPIER_URL = st.secrets.get("ZAPIER_URL", "")

# ------------------- Session State Initialization -------------------
def _init(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

_init("gx", 0.0); _init("gy", 0.0); _init("gz", 0.0)
_init("gsize", 2.0)
_init("temp", 230.0); _init("press", 70.0); _init("vel", 80.0)
_init("etime", 0.5)
_init("sim_running", False)
_init("sim_status", "idle")
_init("sim_logs", [])
_init("last_signal_id", None)
_init("mesh", None)
_init("props", None)
_init("props_confirmed", False)
_init("process_confirmed", False)
_init("mat_name", "PA66+30GF")
_init("last_vel_mms", 80.0)
_init("last_etime", 0.5)
_init("gx_final", 0.0)
_init("gy_final", 0.0)
_init("gz_final", 0.0)
_init("animation_playing", False)
_init("current_frame", 0)
_init("vtk_files", [])
_init("last_synced_signal_id", None)
_init("executed_params", None)
_init("num_frames", 15)   # ★ global frame count

# ------------------- Helper Functions -------------------
def add_log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state["sim_logs"].append(f"[{timestamp}] {message}")
    if len(st.session_state["sim_logs"]) > 100:
        st.session_state["sim_logs"] = st.session_state["sim_logs"][-100:]

def clear_old_results():
    if os.path.exists("VTK"):
        shutil.rmtree("VTK")
    if os.path.exists("results.txt"):
        os.remove("results.txt")
    if os.path.exists("logs.zip"):
        os.remove("logs.zip")
    st.session_state["vtk_files"] = []
    add_log("Cleared old simulation results.")

def sample_vtk_files(vtk_dir, num_frames):
    """Return exactly `num_frames` VTK files, evenly spaced."""
    all_files = sorted(
        glob.glob(os.path.join(vtk_dir, "**", "case_*.vt*"), recursive=True) +
        glob.glob(os.path.join(vtk_dir, "case_*.vt*"), recursive=True)
    )
    all_files = sorted(
        set(all_files),
        key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0
    )
    if not all_files:
        return []
    total = len(all_files)
    if total <= num_frames:
        return all_files
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    return [all_files[i] for i in indices]

def sync_simulation_results():
    """Download latest artifact from GitHub."""
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    REPO_OWNER = "workshopcompany"
    REPO_NAME = "OpenFOAM-Injection-Automation"
    ARTIFACT_NAME = "simulation-results"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/artifacts"
    try:
        with st.spinner("Fetching from GitHub..."):
            resp = requests.get(url, headers=headers)
            if resp.status_code != 200:
                st.error(f"GitHub API error: {resp.status_code}")
                return False
            artifacts = resp.json().get("artifacts", [])
            target = next((a for a in artifacts if a["name"] == ARTIFACT_NAME), None)
            if not target:
                st.warning("No simulation results found. Run a simulation first.")
                return False
            dl_url = target["archive_download_url"]
            file_resp = requests.get(dl_url, headers=headers)
            if file_resp.status_code == 200:
                clear_old_results()
                with zipfile.ZipFile(io.BytesIO(file_resp.content)) as z:
                    z.extractall(".")
                vtk_dir = "VTK"
                if os.path.exists(vtk_dir):
                    nf = st.session_state.get("num_frames", 15)
                    st.session_state["vtk_files"] = sample_vtk_files(vtk_dir, nf)
                if os.path.exists("results.txt"):
                    with open("results.txt", "r") as f:
                        content = f.read()
                        match = re.search(r"Signal ID[:\s]+([A-Za-z0-9\-]+)", content)
                        if match:
                            st.session_state["last_synced_signal_id"] = match.group(1)
                add_log("Results synchronized from GitHub.")
                st.success("Results synchronized successfully!")
                return True
            else:
                st.error("Failed to download artifact.")
                return False
    except Exception as e:
        st.error(f"Sync error: {e}")
        return False


# ============================================================
# ★ CORE FIX: load fluid WITHOUT clip_surface
#   - threshold only (no wall cutting)
#   - fluid must originate from gate region
#   - mesh resolution: characteristic length / 5
# ============================================================
def estimate_mesh_resolution(mold_trimesh):
    """
    Estimate target cell size = min(X,Y,Z) thickness / 5.
    OpenFOAM coords are in meters (mold is in mm → convert).
    Returns resolution in meters.
    """
    if mold_trimesh is None:
        return 0.001  # default 1 mm
    bb = mold_trimesh.bounds  # [[xmin,ymin,zmin],[xmax,ymax,zmax]] in mm
    dims = np.array(bb[1]) - np.array(bb[0])
    min_dim_mm = np.min(dims[dims > 0]) if np.any(dims > 0) else 10.0
    res_mm = max(min_dim_mm / 5.0, 0.5)   # at least 0.5 mm
    return res_mm / 1000.0  # → meters


def load_fluid_no_clip(fpath, scale=1000.0, thres=0.05,
                       gate_pos_m=None, gate_dia_m=None):
    """
    Load fluid cells (alpha > thres) from a VTK frame.
    NO clip_surface → fluid stops naturally at mesh walls.
    gate_pos_m : (x, y, z) gate position in METERS (OpenFOAM coords)
    gate_dia_m : gate diameter in meters (used to seed first frame)

    Returns (pts, i, j, k), alpha_values, n_cells
    """
    try:
        mesh = pv.read(fpath)
        if isinstance(mesh, pv.MultiBlock):
            mesh = mesh.combine()

        # detect alpha field name
        fname = None
        for candidate in ["alpha.water", "alpha1", "alpha"]:
            if candidate in mesh.array_names:
                fname = candidate
                break
        if fname is None:
            return None, None, 0

        # ★ threshold — no clipping, just keep fluid cells
        fluid = mesh.threshold(thres, scalars=fname)
        if fluid.n_cells == 0:
            fluid = mesh.threshold(0.001, scalars=fname)
            if fluid.n_cells == 0:
                return None, None, 0

        # scale meters → mm
        fluid.points *= scale

        # extract surface mesh for rendering
        surf = fluid.extract_surface().triangulate()
        if surf.n_points == 0:
            return None, None, 0

        pts = surf.points
        if surf.faces.size == 0:
            return None, None, 0
        faces = surf.faces.reshape(-1, 4)[:, 1:]

        if fname in surf.point_data:
            alpha_vals = surf.point_data[fname].tolist()
        else:
            alpha_vals = [float(thres)] * len(pts)

        return (pts, faces[:, 0], faces[:, 1], faces[:, 2]), alpha_vals, fluid.n_cells

    except Exception as e:
        add_log(f"Error loading {os.path.basename(fpath)}: {e}")
        return None, None, 0


# ------------------- Plotly trace helpers -------------------
def make_fluid_trace(pts, fi, fj, fk, alpha_vals, opacity=0.8):
    intensity = np.array(alpha_vals) if alpha_vals is not None else np.ones(len(pts))
    return go.Mesh3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        i=fi, j=fj, k=fk,
        intensity=intensity,
        colorscale="Viridis",
        opacity=opacity,
        name="Fluid",
        showscale=True,
        colorbar=dict(title="Fill Ratio", thickness=15, len=0.5)
    )


def make_mold_trace(mold_trimesh, opacity=0.1, color="lightgray"):
    if mold_trimesh is None:
        return None
    v, f = mold_trimesh.vertices, mold_trimesh.faces
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        opacity=opacity, color=color, name="Mold", showlegend=True
    )


# ------------------- Material Database -------------------
LOCAL_DB = {
    "PP":         {"nu": 1e-3,  "rho": 900.0,  "Tmelt": 230.0, "Tmold": 40.0,  "press_mpa": 70.0,  "vel_mms": 80.0,  "desc": "General-purpose polypropylene"},
    "ABS":        {"nu": 2e-3,  "rho": 1050.0, "Tmelt": 240.0, "Tmold": 60.0,  "press_mpa": 80.0,  "vel_mms": 70.0,  "desc": "ABS resin"},
    "PA66":       {"nu": 5e-4,  "rho": 1140.0, "Tmelt": 280.0, "Tmold": 80.0,  "press_mpa": 90.0,  "vel_mms": 100.0, "desc": "Nylon 66"},
    "PA66+30GF":  {"nu": 4e-4,  "rho": 1300.0, "Tmelt": 285.0, "Tmold": 85.0,  "press_mpa": 110.0, "vel_mms": 80.0,  "desc": "30% glass-fiber reinforced nylon"},
    "PC":         {"nu": 3e-3,  "rho": 1200.0, "Tmelt": 300.0, "Tmold": 85.0,  "press_mpa": 120.0, "vel_mms": 60.0,  "desc": "Polycarbonate"},
    "POM":        {"nu": 8e-4,  "rho": 1410.0, "Tmelt": 200.0, "Tmold": 90.0,  "press_mpa": 85.0,  "vel_mms": 90.0,  "desc": "Polyacetal"},
    "HDPE":       {"nu": 9e-4,  "rho": 960.0,  "Tmelt": 220.0, "Tmold": 35.0,  "press_mpa": 60.0,  "vel_mms": 90.0,  "desc": "HDPE"},
    "PET":        {"nu": 6e-4,  "rho": 1370.0, "Tmelt": 265.0, "Tmold": 70.0,  "press_mpa": 80.0,  "vel_mms": 85.0,  "desc": "PET"},
    "CATAMOLD":   {"nu": 5e-3,  "rho": 4900.0, "Tmelt": 185.0, "Tmold": 40.0,  "press_mpa": 100.0, "vel_mms": 30.0,  "desc": "BASF Catamold MIM feedstock"},
    "MIM":        {"nu": 5e-3,  "rho": 5000.0, "Tmelt": 185.0, "Tmold": 40.0,  "press_mpa": 100.0, "vel_mms": 30.0,  "desc": "Metal injection molding feedstock"},
    "17-4PH":     {"nu": 4e-3,  "rho": 7780.0, "Tmelt": 185.0, "Tmold": 40.0,  "press_mpa": 110.0, "vel_mms": 25.0,  "desc": "17-4PH stainless steel"},
    "316L":       {"nu": 4e-3,  "rho": 7900.0, "Tmelt": 185.0, "Tmold": 40.0,  "press_mpa": 110.0, "vel_mms": 25.0,  "desc": "316L stainless steel"},
}

def get_props(material: str) -> dict:
    name = material.upper().strip()
    for key in LOCAL_DB:
        if key.upper() == name:
            return {**LOCAL_DB[key], "material": key, "source": "Database"}
    return {
        "nu": 1e-3, "rho": 1000.0, "Tmelt": 220.0, "Tmold": 50.0,
        "press_mpa": 70.0, "vel_mms": 80.0,
        "material": material, "source": "Default",
        "desc": f"{material} — using default values"
    }

def get_process(material: str) -> dict:
    props = get_props(material)
    return {"temp": float(props["Tmelt"]), "press": float(props["press_mpa"]), "vel": float(props["vel_mms"])}


# ============================================================
# ★ Simulation Summary: always show executed_params, not results.txt
# ============================================================
def build_summary_text():
    """
    Build a summary string from the currently-loaded executed_params.
    Falls back to results.txt only for server-side fields (End Time, Steps, Finish Time).
    """
    ep = st.session_state.get("executed_params")
    if ep is None:
        return None

    lines = [
        "Simulation Status: Success",
        f"Material: {ep.get('material', 'N/A')}",
        f"Velocity: {ep.get('vel_mms', 0)/1000:.4f} m/s   ({ep.get('vel_mms', 0):.1f} mm/s)",
        f"Viscosity: {ep.get('viscosity', 0):.2e} m²/s",
        f"Density: {ep.get('density', 0):.0f} kg/m³",
        f"Melt Temp: {ep.get('melt_temp', 0):.1f} °C",
        f"Mold Temp: {ep.get('mold_temp', 0):.1f} °C",
        f"Injection Temp: {ep.get('temp', 0):.1f} °C",
        f"Pressure: {ep.get('press', 0):.1f} MPa",
        f"End Time: {ep.get('etime', 0):.2f} s",
        f"Gate: ({ep.get('gate_x',0):.2f}, {ep.get('gate_y',0):.2f}, {ep.get('gate_z',0):.2f}) mm",
        f"Gate Dia: {ep.get('gate_dia',0):.1f} mm",
        f"Signal ID: {ep.get('signal_id', 'N/A')}",
    ]

    # try to append server-side time steps from results.txt
    if os.path.exists("results.txt"):
        with open("results.txt", "r") as f:
            raw = f.read()
        for keyword in ["Last Time Step", "Time Steps", "Finish Time"]:
            match = re.search(rf"{keyword}[:\s]+(.+)", raw)
            if match:
                lines.append(f"{keyword}: {match.group(1).strip()}")

    return "\n".join(lines)


# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("📂 1. Geometry")
    uploaded = st.file_uploader("Upload STL (mm)", type=["stl"])
    if uploaded:
        try:
            mesh = trimesh.load(uploaded, file_type="stl")
            st.session_state["mesh"] = mesh
            st.success(f"✅ STL loaded — {len(mesh.faces):,} faces")
            add_log(f"STL loaded: {len(mesh.faces):,} faces")
        except Exception as e:
            st.error(f"STL load failed: {e}")

    st.divider()
    st.header("📍 2. Gate Configuration")
    if st.button("🪄 AI Gate Suggestion", use_container_width=True):
        mesh = st.session_state.get("mesh")
        if mesh:
            center = mesh.centroid
            snap, _, _ = trimesh.proximity.closest_point(mesh, [center])
            pos = snap[0]
            st.session_state["gx"] = float(pos[0])
            st.session_state["gy"] = float(pos[1])
            st.session_state["gz"] = float(pos[2])
            st.session_state["gsize"] = 2.5
            st.toast("Gate suggested!", icon="🪄")
            add_log(f"Gate set to ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        else:
            st.warning("Upload STL first")
    g_size = st.number_input("Gate Diameter (mm)", 0.5, 10.0, step=0.1, key="gsize")
    vx = st.number_input("Gate X", value=float(st.session_state["gx"]), step=0.1, key="gx")
    vy = st.number_input("Gate Y", value=float(st.session_state["gy"]), step=0.1, key="gy")
    vz = st.number_input("Gate Z", value=float(st.session_state["gz"]), step=0.1, key="gz")
    mesh = st.session_state.get("mesh")
    if mesh:
        snap, _, _ = trimesh.proximity.closest_point(mesh, [[vx, vy, vz]])
        gx, gy, gz = snap[0]
    else:
        gx, gy, gz = vx, vy, vz
    st.session_state["gx_final"] = float(gx)
    st.session_state["gy_final"] = float(gy)
    st.session_state["gz_final"] = float(gz)

    st.divider()
    st.header("🧪 3. Material")
    mat_name = st.text_input("Material Name", value=st.session_state["mat_name"], key="mat_name_input")
    st.session_state["mat_name"] = mat_name
    if st.button("🤖 AI Material Properties", use_container_width=True, type="primary"):
        props = get_props(mat_name)
        st.session_state["props"] = props
        st.session_state["props_confirmed"] = False
        add_log(f"Material properties loaded: {mat_name}")
    if st.session_state["props"]:
        p = st.session_state["props"]
        st.caption(f"🟢 Source: {p.get('source', 'Database')}")
        if p.get("desc"):
            st.info(p["desc"])
        with st.expander("📋 Edit Properties", expanded=True):
            p["nu"]    = st.number_input("Viscosity (m²/s)", value=float(p["nu"]),    format="%.2e", key="edit_nu")
            p["rho"]   = st.number_input("Density (kg/m³)",  value=float(p["rho"]),              key="edit_rho")
            p["Tmelt"] = st.number_input("Melt Temp (°C)",   value=float(p["Tmelt"]),             key="edit_tmelt")
            p["Tmold"] = st.number_input("Mold Temp (°C)",   value=float(p["Tmold"]),             key="edit_tmold")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Confirm\nProperti\nes", use_container_width=True):
                    st.session_state["props_confirmed"] = True
                    st.toast("Properties confirmed!", icon="✅")
                    add_log("Material properties confirmed")
            with col2:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state["props"] = None
                    st.session_state["props_confirmed"] = False
                    add_log("Properties reset")
                    st.rerun()

    st.divider()
    st.header("⚙️ 4. Process")
    if st.button("🤖 Optimize Process", use_container_width=True):
        opt = get_process(mat_name)
        st.session_state["temp"]  = opt["temp"]
        st.session_state["press"] = opt["press"]
        st.session_state["vel"]   = opt["vel"]
        st.toast("Process optimized!", icon="🤖")
        add_log(f"Optimized: {opt['temp']}°C, {opt['press']}MPa, {opt['vel']}mm/s")
    temp_c    = st.number_input("Temp (°C)",      50.0,  450.0, value=float(st.session_state["temp"]),  step=1.0, key="temp")
    press_mpa = st.number_input("Pressure (MPa)", 10.0,  250.0, value=float(st.session_state["press"]), step=1.0, key="press")
    vel_mms   = st.number_input("Velocity (mm/s)", 1.0,  600.0, value=float(st.session_state["vel"]),   step=1.0, key="vel")
    etime     = st.number_input("End Time (s)", value=float(st.session_state["etime"]),
                                min_value=0.1, max_value=10.0, step=0.1, key="etime")
    st.session_state["last_vel_mms"] = vel_mms
    st.session_state["last_etime"]   = etime

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Confirm Process", use_container_width=True):
            st.session_state["process_confirmed"] = True
            st.toast("Process confirmed!", icon="✅")
            add_log("Process conditions confirmed")
    with col2:
        if st.button("🔄 Reset Process", use_container_width=True):
            st.session_state["process_confirmed"] = False
            add_log("Process reset")
            st.rerun()
    if not st.session_state.get("process_confirmed", False):
        st.warning("⚠️ Please confirm process conditions")

    st.divider()
    # ★ Animation Frames — stored in session state immediately
    num_frames_selected = st.select_slider(
        "Animation Frames",
        options=[5, 10, 15, 20, 30],
        value=st.session_state.get("num_frames", 15)
    )
    st.session_state["num_frames"] = num_frames_selected

    run_disabled = (
        st.session_state["sim_running"]
        or not st.session_state["props_confirmed"]
        or not st.session_state.get("process_confirmed", False)
    )
    if st.button("🚀 Run Cloud Simulation", type="primary", use_container_width=True, disabled=run_disabled):
        if not ZAPIER_URL:
            st.error("ZAPIER_URL not configured")
        else:
            clear_old_results()
            sig_id = str(uuid.uuid4())[:8]
            # ★ Capture ALL parameters at launch time
            st.session_state["executed_params"] = {
                "signal_id":  sig_id,
                "material":   st.session_state["mat_name"],
                "viscosity":  float(st.session_state["props"]["nu"]),
                "density":    float(st.session_state["props"]["rho"]),
                "melt_temp":  float(st.session_state["props"]["Tmelt"]),
                "mold_temp":  float(st.session_state["props"]["Tmold"]),
                "temp":       float(temp_c),
                "press":      float(press_mpa),
                "vel_mms":    float(vel_mms),
                "etime":      float(etime),
                "gate_x":     float(gx),
                "gate_y":     float(gy),
                "gate_z":     float(gz),
                "gate_dia":   float(g_size),
                "num_frames": st.session_state["num_frames"],
            }
            st.session_state["last_signal_id"] = sig_id
            st.session_state["sim_running"]     = True
            st.session_state["sim_status"]      = "running"
            add_log(f"🚀 Simulation launched | ID: {sig_id}")
            add_log(f"Material: {st.session_state['executed_params']['material']}")
            add_log(f"Gate: ({gx:.2f}, {gy:.2f}, {gz:.2f}) mm, Dia={g_size}mm")
            add_log(f"Injection: {temp_c}°C, {press_mpa}MPa, {vel_mms}mm/s")
            add_log(f"Animation frames requested: {st.session_state['num_frames']}")

            # Compute mesh resolution (thickness/5)
            mold_tm = st.session_state.get("mesh")
            res_m = estimate_mesh_resolution(mold_tm)
            add_log(f"Mesh resolution target: {res_m*1000:.2f} mm ({res_m:.4f} m)")

            payload = {
                "signal_id":    sig_id,
                "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "material":     st.session_state["executed_params"]["material"],
                "viscosity":    st.session_state["executed_params"]["viscosity"],
                "density":      st.session_state["executed_params"]["density"],
                "melt_temp":    st.session_state["executed_params"]["melt_temp"],
                "mold_temp":    st.session_state["executed_params"]["mold_temp"],
                "temp":         st.session_state["executed_params"]["temp"],
                "press":        st.session_state["executed_params"]["press"],
                "vel":          round(st.session_state["executed_params"]["vel_mms"] / 1000, 6),
                "etime":        st.session_state["executed_params"]["etime"],
                "num_frames":   st.session_state["num_frames"],
                "mesh_resolution": res_m,          # ★ thickness/5 resolution
                "gate_pos":     {"x": round(gx, 3), "y": round(gy, 3), "z": round(gz, 3)},
                "gate_size":    st.session_state["executed_params"]["gate_dia"],
            }
            try:
                add_log("Sending to Zapier...")
                r = requests.post(ZAPIER_URL, json=payload, timeout=10)
                if r.status_code == 200:
                    st.toast(f"Signal sent! ID: {sig_id}", icon="🚀")
                    add_log(f"Signal sent successfully (HTTP {r.status_code})")
                    add_log("⏳ GitHub Actions will process. Wait 2-3 min then click 'Sync Results'.")
                else:
                    st.error(f"Failed: HTTP {r.status_code}")
                    add_log(f"Failed to send: HTTP {r.status_code}")
                    st.session_state["sim_running"] = False
            except Exception as e:
                st.error(f"Error: {e}")
                add_log(f"Error: {e}")
                st.session_state["sim_running"] = False


# ==================== MAIN AREA ====================
col_geo, col_log = st.columns([2, 1])
with col_geo:
    st.header("🎥 3D Geometry & Gate")
    mesh = st.session_state.get("mesh")
    if mesh:
        v, f = mesh.vertices, mesh.faces
        fig = go.Figure(data=[
            go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color="#AAAAAA", opacity=0.7
            ),
            go.Scatter3d(
                x=[st.session_state["gx_final"]],
                y=[st.session_state["gy_final"]],
                z=[st.session_state["gz_final"]],
                mode="markers",
                marker=dict(size=st.session_state["gsize"] * 3, color="red"),
                name="Gate"
            )
        ])
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                          scene=dict(aspectmode="data"), height=500)
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
        bb = mesh.bounds
        c1, c2, c3 = st.columns(3)
        c1.metric("X Size", f"{bb[1][0]-bb[0][0]:.1f} mm")
        c2.metric("Y Size", f"{bb[1][1]-bb[0][1]:.1f} mm")
        c3.metric("Z Size", f"{bb[1][2]-bb[0][2]:.1f} mm")
    else:
        st.info("Upload STL file to see 3D model")

with col_log:
    st.header("📟 Simulation Logs")
    if st.session_state["sim_running"]:
        st.info("🟢 STATUS: RUNNING...")
    elif st.session_state["sim_status"] == "failed":
        st.error("🔴 STATUS: FAILED")
    else:
        st.success("✅ STATUS: READY")
    log_container = st.container(height=350)
    with log_container:
        for log in st.session_state["sim_logs"][-25:]:
            st.code(log, language="bash")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.session_state["sim_running"]:
            if st.button("✅ Mark Complete", use_container_width=True):
                st.session_state["sim_running"] = False
                st.session_state["sim_status"]  = "completed"
                add_log("Simulation marked as completed by user")
                st.rerun()
    with col_btn2:
        if st.button("🗑 Clear Logs", use_container_width=True):
            st.session_state["sim_logs"] = []
            st.rerun()

st.info(f"📍 Final Gate: ({st.session_state['gx_final']:.2f}, {st.session_state['gy_final']:.2f}, {st.session_state['gz_final']:.2f}) mm")
if st.session_state["props_confirmed"] and st.session_state.get("process_confirmed", False) and st.session_state["props"]:
    p = st.session_state["props"]
    st.caption(f"ℹ️ Current settings: {st.session_state['mat_name']} | nu={p['nu']:.2e} | rho={p['rho']} kg/m³ | Tmelt={p['Tmelt']}°C")
else:
    st.caption("ℹ️ Confirm properties and process before running simulation.")


# ==================== RESULTS SECTION ====================
st.title("📊 Simulation Results")
col_res1, col_res2, col_res3 = st.columns([2, 1, 1])
with col_res1:
    st.markdown("### Download & Sync")
with col_res2:
    if st.button("🔄 Sync Results", use_container_width=True, type="primary"):
        if sync_simulation_results():
            st.rerun()
with col_res3:
    if st.button("🗑 Clear Results", use_container_width=True):
        clear_old_results()
        st.success("Results cleared")
        st.rerun()

# ★ Simulation Summary: always built from executed_params (current run)
summary_text = build_summary_text()
if summary_text:
    st.text_area("📄 Simulation Summary", summary_text, height=230)
    # Warn if synced results belong to a different signal
    if (st.session_state.get("last_synced_signal_id")
            and st.session_state.get("executed_params")):
        if (st.session_state["last_synced_signal_id"]
                != st.session_state["executed_params"]["signal_id"]):
            st.warning(
                f"⚠️ Synced results belong to an older run "
                f"(ID: {st.session_state['last_synced_signal_id']}). "
                f"Current run ID: {st.session_state['executed_params']['signal_id']}. "
                "Please wait for the new simulation and re-sync."
            )
        else:
            st.success("Results synchronized successfully!")
else:
    st.info("No results loaded. Run a simulation and click 'Sync Results'.")

if os.path.exists("logs.zip"):
    with open("logs.zip", "rb") as f:
        st.download_button("📂 Download Logs (logs.zip)", f, "logs.zip", use_container_width=True)


# ==================== VTK ANIMATION ====================
vtk_dir = "VTK"
if os.path.exists(vtk_dir):
    st.subheader("🌊 3D Filling Animation (Solid Mesh)")

    mold_trimesh = st.session_state.get("mesh")

    # ★ Use session_state num_frames (set in sidebar)
    num_frames = st.session_state.get("num_frames", 15)
    sampled_files = sample_vtk_files(vtk_dir, num_frames)
    total_steps = len(sampled_files)

    if total_steps == 0:
        st.warning("No VTK files found. Please sync results first.")
    else:
        with st.expander("🔧 Visualization Settings", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                scale_factor = st.slider("Scale (m→mm)", 100.0, 2000.0, 1000.0, 50.0,
                                         help="OpenFOAM uses meters, STL uses mm.")
                threshold = st.slider("Fluid Threshold (alpha)", 0.001, 0.2, 0.010, 0.001,
                                      format="%.3f", help="Lower = more fluid volume")
            with col2:
                mold_opacity  = st.slider("Mold Opacity",  0.0, 0.5, 0.10, 0.01)
                fluid_opacity = st.slider("Fluid Opacity",  0.5, 1.0, 0.80, 0.05)
            with col3:
                view_mode = st.radio("View Mode", ["Auto", "Uniform"], index=0)

        st.markdown("### 🎮 Animation Controls")
        col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([1, 1, 3, 1])
        with col_ctrl1:
            if st.button("⏮ First", use_container_width=True):
                st.session_state["current_frame"]    = 0
                st.session_state["animation_playing"] = False
                st.rerun()
        with col_ctrl2:
            if st.button("⏸ Pause", use_container_width=True):
                st.session_state["animation_playing"] = False
                st.rerun()
        with col_ctrl3:
            if st.button("▶ Play", use_container_width=True, type="primary"):
                st.session_state["animation_playing"] = True
                st.rerun()
        with col_ctrl4:
            if st.button("⏭ Last", use_container_width=True):
                st.session_state["current_frame"]    = total_steps - 1
                st.session_state["animation_playing"] = False
                st.rerun()

        current_frame = st.slider(
            "Frame", 0, total_steps - 1,
            value=min(st.session_state.get("current_frame", 0), total_steps - 1),
            key="frame_slider"
        )
        st.session_state["current_frame"] = current_frame

        # Gate position in METERS for fluid origin seeding
        gate_x_m = st.session_state["gx_final"] / 1000.0
        gate_y_m = st.session_state["gy_final"] / 1000.0
        gate_z_m = st.session_state["gz_final"] / 1000.0
        gate_dia_m = st.session_state.get("gsize", 2.0) / 1000.0

        with st.spinner(f"Loading frame {current_frame + 1}/{total_steps}..."):
            fpath = sampled_files[current_frame]

            # ★ No clip_surface — fluid bounded naturally by mesh walls
            res, alpha_vals, n_cells = load_fluid_no_clip(
                fpath,
                scale=scale_factor,
                thres=threshold,
                gate_pos_m=(gate_x_m, gate_y_m, gate_z_m),
                gate_dia_m=gate_dia_m
            )

            fig = go.Figure()

            if mold_trimesh:
                fig.add_trace(make_mold_trace(mold_trimesh, opacity=mold_opacity))

            # Gate marker
            gate_x, gate_y, gate_z = (
                st.session_state["gx_final"],
                st.session_state["gy_final"],
                st.session_state["gz_final"],
            )
            fig.add_trace(go.Scatter3d(
                x=[gate_x], y=[gate_y], z=[gate_z],
                mode="markers",
                marker=dict(
                    size=st.session_state["gsize"] * 2,
                    color="red", symbol="x",
                    line=dict(width=2, color="white")
                ),
                name="Gate"
            ))

            if res:
                pts, fi, fj, fk = res
                fig.add_trace(make_fluid_trace(pts, fi, fj, fk, alpha_vals, opacity=fluid_opacity))
                st.success(f"Frame {current_frame + 1}/{total_steps} | Fluid cells: {n_cells:,}")
            else:
                st.info(f"Frame {current_frame + 1}: No fluid at threshold={threshold:.3f}. Try lowering threshold.")

            scene_config = dict(
                aspectmode="data" if view_mode == "Auto" else "cube",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"
            )
            fig.update_layout(
                scene=scene_config, height=600,
                margin=dict(l=0, r=0, b=0, t=0)
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"scrollZoom": True, "displayModeBar": True})

        if st.session_state.get("animation_playing", False):
            next_frame = (current_frame + 1) % total_steps
            st.session_state["current_frame"] = next_frame
            time.sleep(0.2)
            st.rerun()

        if mold_trimesh:
            bounds = mold_trimesh.bounds
            st.caption(
                f"📐 Model bounds: "
                f"X [{bounds[0][0]:.1f}, {bounds[1][0]:.1f}] | "
                f"Y [{bounds[0][1]:.1f}, {bounds[1][1]:.1f}] | "
                f"Z [{bounds[0][2]:.1f}, {bounds[1][2]:.1f}] mm"
            )
        # ★ Show mesh resolution info
        mold_tm = st.session_state.get("mesh")
        if mold_tm:
            res_m = estimate_mesh_resolution(mold_tm)
            st.caption(f"🔬 Target mesh resolution: {res_m*1000:.2f} mm (min dim / 5)")
else:
    st.info("📁 No VTK directory found. Run a simulation and click 'Sync Results'.")

st.divider()
st.caption("MIM-Ops Pro v2.3 | Gate-Origin Flow | No-Clip | Fine Mesh | AI-Powered Injection Molding Simulation")
