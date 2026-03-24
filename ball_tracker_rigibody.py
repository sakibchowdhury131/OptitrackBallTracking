"""
OptiTrack Live Ball Tracker - Fixed-Count Sample Mode
======================================================
Press P  -> immediately starts collecting N_SAMPLES points, then fits.
Press R  -> reset.
Press Q  -> quit.

No flight detection, no speed thresholds. Just: press P, move ball, get fit.

Tune at the top of CONFIG:
  N_SAMPLES  = 20   # how many points to collect after P is pressed
  GRAVITY_AXIS = 1  # 1=Y-up, 2=Z-up  (auto-detected if wrong)

Requirements:  pip install numpy scipy matplotlib
"""

from __future__ import annotations

import socket
import struct
import threading
import time
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from collections import deque
from typing import Optional, List, Tuple

# ── CONFIG ────────────────────────────────────────────────────────────────────
SERVER_IP        = "10.156.84.112"
CLIENT_IP        = "0.0.0.0"
MULTICAST_ADDR   = "239.255.42.99"
DATA_PORT        = 1511

NAT_FRAMEOFDATA  = 7

G                = 9.81
GRAVITY_AXIS     = 1      # 1=Y-up, 2=Z-up. Auto-detected if wrong.

N_SAMPLES        = 20     # ← TUNE: how many frames to collect after pressing P
TRACKING_MODE    = "rigid_body"   # "rigid_body"  or  "single_marker"
BUFFER_SIZE      = 2000   # live trail buffer
TRAIL_LEN        = 80
PLOT_INTERVAL_MS = 50    # fast poll, but only redraws when data changed

# Axis limits — adjust to your capture volume
X_LIM = (-1.5, 1.5)
Y_LIM = (0.0,  2.5)
Z_LIM = (-1.5, 1.5)

# ── STATE MACHINE ─────────────────────────────────────────────────────────────
class Mode:
    IDLE      = "IDLE"       # live view only
    RECORDING = "RECORDING"  # collecting N_SAMPLES points
    FITTING   = "FITTING"    # running curve_fit
    DONE      = "DONE"       # results shown

# ── SHARED STATE ──────────────────────────────────────────────────────────────
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()

        # live stream
        self.buffer    = deque(maxlen=BUFFER_SIZE)
        self.connected = False
        self.rb_name   = "unknown"
        self.hw_fps    = 0.0
        self._fps_ts   = deque(maxlen=200)

        # recording
        self.mode           = Mode.IDLE
        self.arm_time       = None   # when P was pressed
        self.record_start_t = None   # timestamp of first collected sample
        self.record_end_t   = None   # timestamp of last collected sample
        self.fit_done_t     = None

        # collected samples (appended inline in UDP thread)
        self.samples_t  = []   # timestamps
        self.samples_p  = []   # positions (np.ndarray each)
        self._target_n  = 0    # how many we still need

        # results
        self.fit_result = None
        self.timing     = None
        self.accuracy   = None

        # plot dirty flag: True = something changed, redraw needed
        self.plot_dirty = True

state = SharedState()

# ── NATNET PARSER ─────────────────────────────────────────────────────────────
def unpack_string(data: bytes, offset: int) -> Tuple[str, int]:
    end = data.index(b'\x00', offset)
    return data[offset:end].decode('utf-8', errors='ignore'), end + 1

def parse_rigid_bodies(data: bytes) -> List[Tuple[int, np.ndarray]]:
    try:
        offset = 8
        n_sets = struct.unpack_from('<I', data, offset)[0]; offset += 4
        for _ in range(n_sets):
            _, offset = unpack_string(data, offset)
            n_m = struct.unpack_from('<I', data, offset)[0]; offset += 4
            offset += n_m * 12
        n_unlabeled = struct.unpack_from('<I', data, offset)[0]; offset += 4
        offset += n_unlabeled * 12
        n_rb = struct.unpack_from('<I', data, offset)[0]; offset += 4
        if n_rb == 0 or n_rb > 50:
            return []
        results = []
        for _ in range(n_rb):
            rb_id  = struct.unpack_from('<i',    data, offset)[0]; offset += 4
            x,y,z  = struct.unpack_from('<fff',  data, offset);    offset += 12
            _q     = struct.unpack_from('<ffff', data, offset);    offset += 16
            _err   = struct.unpack_from('<f',    data, offset)[0]; offset += 4
            params = struct.unpack_from('<h',    data, offset)[0]; offset += 2
            if bool(params & 0x01):
                results.append((rb_id, np.array([x, y, z], dtype=float)))
        return results
    except Exception:
        return []

def parse_unlabeled_markers(data: bytes):
    """
    Parse NatNet FrameOfData and return list of unlabeled marker positions.
    These are raw XYZ floats with no ID or tracking flag.
    """
    try:
        offset = 8  # skip msg_id(2) + packet_size(2) + frame_number(4)

        n_sets = struct.unpack_from('<I', data, offset)[0]; offset += 4
        for _ in range(n_sets):
            _, offset = unpack_string(data, offset)
            n_m = struct.unpack_from('<I', data, offset)[0]; offset += 4
            offset += n_m * 12

        n_unlabeled = struct.unpack_from('<I', data, offset)[0]; offset += 4
        markers = []
        for _ in range(n_unlabeled):
            x, y, z = struct.unpack_from('<fff', data, offset); offset += 12
            markers.append(np.array([x, y, z], dtype=float))
        return markers
    except Exception:
        return []

def parse_rigid_body_descriptions(data: bytes) -> dict:
    id_to_name = {}
    try:
        offset = 4
        n_defs = struct.unpack_from('<I', data, offset)[0]; offset += 4
        for _ in range(n_defs):
            def_type = struct.unpack_from('<I', data, offset)[0]; offset += 4
            if def_type == 1:
                name, offset = unpack_string(data, offset)
                rb_id        = struct.unpack_from('<i', data, offset)[0]; offset += 4
                _parent      = struct.unpack_from('<i', data, offset)[0]; offset += 4
                offset      += 12
                n_m          = struct.unpack_from('<I', data, offset)[0]; offset += 4
                offset      += n_m * (12 + 4 + 4)
                id_to_name[rb_id] = name
            elif def_type == 0:
                name, offset = unpack_string(data, offset)
                n_m = struct.unpack_from('<I', data, offset)[0]; offset += 4
                for _ in range(n_m):
                    _, offset = unpack_string(data, offset)
            else:
                break
    except Exception:
        pass
    return id_to_name

# ── UDP LISTENER ──────────────────────────────────────────────────────────────
rb_id_filter: Optional[int] = None
id_to_name: dict = {}

def request_descriptions(sock):
    try:
        sock.sendto(struct.pack('<HH', 4, 0), (SERVER_IP, DATA_PORT))
    except Exception:
        pass

def udp_listener():
    global rb_id_filter, id_to_name

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((CLIENT_IP, DATA_PORT))
    except Exception as e:
        print(f"[UDP] Bind failed: {e}"); return

    try:
        mreq = struct.pack("4sL", socket.inet_aton(MULTICAST_ADDR), socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        print(f"[UDP] Joined multicast {MULTICAST_ADDR}:{DATA_PORT}")
    except Exception:
        print(f"[UDP] Multicast failed – trying unicast on :{DATA_PORT}")

    sock.settimeout(2.0)
    if TRACKING_MODE == "rigid_body":
        request_descriptions(sock)
    last_req = time.time()

    while True:
        try:
            data, _ = sock.recvfrom(65536)
        except socket.timeout:
            if TRACKING_MODE == "rigid_body" and rb_id_filter is None and (time.time() - last_req) > 3.0:
                request_descriptions(sock); last_req = time.time()
            continue
        except Exception as e:
            print(f"[UDP] Error: {e}"); break

        msg_id = struct.unpack_from('<H', data, 0)[0]

        if msg_id == 5:
            new_map = parse_rigid_body_descriptions(data)
            if new_map:
                id_to_name.update(new_map)
                print(f"[UDP] Rigid bodies: { {v:k for k,v in id_to_name.items()} }")
                for rid, name in id_to_name.items():
                    if name.lower() == "ball":
                        rb_id_filter = rid
                        with state.lock: state.rb_name = name
                        print(f"[UDP] Tracking '{name}' (id={rid})")
                        break
                else:
                    print(f"[UDP] 'ball' not found. Available: {list(id_to_name.values())}")
                    if id_to_name:
                        rb_id_filter = list(id_to_name.keys())[0]
                        name = id_to_name[rb_id_filter]
                        with state.lock: state.rb_name = name
                        print(f"[UDP] Falling back to '{name}' (id={rb_id_filter})")
            continue

        if msg_id != NAT_FRAMEOFDATA:
            continue

        now    = time.time()
        bodies = parse_rigid_bodies(data)

        pos = None

        if TRACKING_MODE == "single_marker":
            markers = parse_unlabeled_markers(data)
            if markers:
                if len(markers) == 1:
                    pos = markers[0]
                else:
                    # Multiple markers: pick closest to last known position
                    with state.lock:
                        last = state.buffer[-1][1] if state.buffer else None
                    if last is not None:
                        pos = min(markers, key=lambda m: np.linalg.norm(m - last))
                    else:
                        pos = markers[0]   # first frame: just take first marker
        else:
            bodies = parse_rigid_bodies(data)
            for rb_id, p in bodies:
                if rb_id_filter is not None and rb_id != rb_id_filter:
                    continue
                pos = p; break

        if pos is None:
            continue

        # ── Always: update display buffer + FPS ──────────────────────────────
        with state.lock:
            state.buffer.append((now, pos))
            state.connected  = True
            state.plot_dirty = True
            state._fps_ts.append(now)
            if len(state._fps_ts) >= 2:
                span = state._fps_ts[-1] - state._fps_ts[0]
                if span > 0:
                    state.hw_fps = (len(state._fps_ts) - 1) / span
            cur_mode = state.mode
            need     = state._target_n

        # ── RECORDING: collect exactly N_SAMPLES points ───────────────────────
        if cur_mode == Mode.RECORDING and need > 0:
            with state.lock:
                # First sample: record start time
                if not state.samples_t:
                    state.record_start_t = now
                    print(f"[UDP] Collecting {N_SAMPLES} samples...")

                state.samples_t.append(now)
                state.samples_p.append(pos.copy())
                state._target_n -= 1
                remaining = state._target_n

                if remaining == 0:
                    state.record_end_t = now
                    state.mode         = Mode.FITTING
                    # grab data for fit thread
                    f_ts  = list(state.samples_t)
                    f_pts = list(state.samples_p)
                    t_arm = state.arm_time
                    t_rec = state.record_start_t

            if remaining == 0:
                print(f"[UDP] {N_SAMPLES} samples collected. Fitting...")
                threading.Thread(
                    target=_run_fit,
                    args=(f_ts, f_pts, t_arm, t_rec),
                    daemon=True
                ).start()

# ── FIT THREAD ────────────────────────────────────────────────────────────────
def _run_fit(f_ts, f_pts, t_arm, t_rec_start):
    fit     = fit_parabola(np.array(f_ts), np.array(f_pts))
    fit_end = time.time()

    if not fit:
        print("[Fit] Fit failed.")
        with state.lock: state.mode = Mode.IDLE
        return

    t_land = predict_landing(fit)
    g_axis = fit['gravity_axis']

    timing = {
        "arm_to_first_sample_s" : round(t_rec_start - t_arm,  3) if t_arm else None,
        "collection_duration_s" : round(fit['duration_s'],     3),
        "fit_compute_s"         : round(fit_end - (t_rec_start + fit['duration_s']), 3),
        "total_arm_to_fit_s"    : round(fit_end - t_arm,       3) if t_arm else None,
        "n_samples"             : fit['n_points'],
        "sample_rate_hz"        : round(fit['sample_rate'],    1),
        "hw_fps"                : round(state.hw_fps,          1),
    }
    accuracy = {
        "rms_3d_mm"   : round(fit['rms_3d_mm'],       2),
        "mean_err_mm" : round(fit['mean_err_mm'],      2),
        "max_err_mm"  : round(fit['max_err_mm'],       2),
        "res_x_mm"    : round(fit['residuals_mm'][0],  2),
        "res_y_mm"    : round(fit['residuals_mm'][1],  2),
        "res_z_mm"    : round(fit['residuals_mm'][2],  2),
        "speed_mps"   : round(fit['speed'],            3),
        "speed_kmh"   : round(fit['speed'] * 3.6,      2),
        "gravity_axis": g_axis,
    }
    if fit['t_apex']:
        ap = fit['predict'](fit['t_apex'])
        accuracy['apex_height_m'] = round(float(ap[g_axis]), 3)
        accuracy['apex_time_s']   = round(fit['t_apex'],     3)
    if t_land:
        lp   = fit['predict'](t_land)
        non_g = [i for i in range(3) if i != g_axis]
        dist = float(np.sqrt(sum((lp[i] - [fit['x0'],fit['y0'],fit['z0']][i])**2
                                  for i in non_g)))
        accuracy['landing_pos'] = tuple(round(float(lp[i]), 3) for i in range(3))
        accuracy['range_m']     = round(dist, 3)

    with state.lock:
        state.fit_result = fit
        state.timing     = timing
        state.accuracy   = accuracy
        state.fit_done_t = fit_end
        state.mode       = Mode.DONE
        state.plot_dirty = True

    _print_report(timing, accuracy)

# ── PARABOLA FIT ──────────────────────────────────────────────────────────────
def fit_parabola(times: np.ndarray, points: np.ndarray) -> Optional[dict]:
    if len(times) < 3:
        return None

    t = times - times[0]

    # Auto-detect gravity axis: most curved axis = gravity direction
    curvatures = [abs(np.polyfit(t, points[:, ax], 2)[0]) for ax in range(3)]
    detected   = int(np.argmax(curvatures))
    used_axis  = GRAVITY_AXIS

    if detected != GRAVITY_AXIS:
        ratio = curvatures[detected] / max(curvatures[GRAVITY_AXIS], 1e-9)
        if ratio > 2.0:
            used_axis = detected
            print(f"[Fit] Auto-override: GRAVITY_AXIS={GRAVITY_AXIS} -> using axis {detected} "
                  f"(curvatures X={curvatures[0]:.3f} Y={curvatures[1]:.3f} Z={curvatures[2]:.3f}). "
                  f"Set GRAVITY_AXIS={detected} in CONFIG to silence.")

    acc = [0.0, 0.0, 0.0]
    acc[used_axis] = -G
    fitted, residuals = [], []

    for axis in range(3):
        p_obs   = points[:, axis]
        a_fixed = acc[axis]

        def model(t, p0, v0, _a=a_fixed):
            return p0 + v0 * t + 0.5 * _a * t**2

        try:
            v_init  = (p_obs[-1] - p_obs[0]) / (t[-1] + 1e-9)
            popt, _ = curve_fit(model, t, p_obs, p0=[p_obs[0], v_init], maxfev=3000)
        except Exception:
            c    = np.polyfit(t, p_obs - 0.5 * a_fixed * t**2, 1)
            popt = [c[1], c[0]]

        fitted.append(popt)
        residuals.append(np.sqrt(np.mean((p_obs - model(t, *popt))**2)))

    (x0, vx), (y0, vy), (z0, vz) = fitted
    _acc = acc

    def predict(t_query):
        tq = np.atleast_1d(np.array(t_query, dtype=float))
        return np.stack([
            x0 + vx*tq + 0.5*_acc[0]*tq**2,
            y0 + vy*tq + 0.5*_acc[1]*tq**2,
            z0 + vz*tq + 0.5*_acc[2]*tq**2,
        ], axis=-1).squeeze()

    v_up   = [vx, vy, vz][used_axis]
    t_apex = v_up / G if v_up > 0 else None

    pred_pts   = predict(t)
    per_pt_err = np.linalg.norm(points - pred_pts, axis=1) * 1000  # mm

    return dict(
        x0=x0, y0=y0, z0=z0, vx=vx, vy=vy, vz=vz,
        speed        = float(np.sqrt(vx**2 + vy**2 + vz**2)),
        residuals_mm = [r * 1000 for r in residuals],
        per_pt_err   = per_pt_err,
        max_err_mm   = float(np.max(per_pt_err)),
        mean_err_mm  = float(np.mean(per_pt_err)),
        rms_3d_mm    = float(np.sqrt(np.mean(per_pt_err**2))),
        n_points     = len(times),
        duration_s   = float(t[-1]),
        sample_rate  = float(len(times) / (t[-1] + 1e-9)),
        t_apex       = t_apex,
        predict      = predict,
        t0           = float(times[0]),
        gravity_axis = used_axis,
    )

def predict_landing(fit: dict, floor_height=0.0) -> Optional[float]:
    g   = fit['gravity_axis']
    y0  = [fit['x0'], fit['y0'], fit['z0']][g]
    vy  = [fit['vx'], fit['vy'], fit['vz']][g]
    a   = -0.5 * G; b = vy; c = y0 - floor_height
    disc = b**2 - 4*a*c
    if disc < 0: return None
    t1 = (-b + np.sqrt(disc)) / (2*a)
    t2 = (-b - np.sqrt(disc)) / (2*a)
    cands = [t for t in [t1, t2] if t > 0]
    return min(cands) if cands else None

# ── PRINT REPORT ──────────────────────────────────────────────────────────────
def _print_report(timing, accuracy):
    print(f"\n{'='*58}")
    print(f"  TRAJECTORY REPORT  ({timing['n_samples']} samples)")
    print(f"{'─'*58}")
    print(f"  {'P press -> 1st sample':<32}: {timing['arm_to_first_sample_s']} s")
    print(f"  {'Collection window':<32}: {timing['collection_duration_s']} s")
    print(f"  {'Fit compute time':<32}: {timing['fit_compute_s']} s")
    print(f"  {'Total (P press -> result)':<32}: {timing['total_arm_to_fit_s']} s")
    print(f"  {'Sample rate':<32}: {timing['sample_rate_hz']} Hz  "
          f"(Motive: {timing['hw_fps']} Hz)")
    print(f"{'─'*58}")
    print(f"  {'3D RMS error':<32}: {accuracy['rms_3d_mm']} mm")
    print(f"  {'Mean / Max error':<32}: {accuracy['mean_err_mm']} / {accuracy['max_err_mm']} mm")
    print(f"  {'Residuals X / Y / Z':<32}: "
          f"{accuracy['res_x_mm']} / {accuracy['res_y_mm']} / {accuracy['res_z_mm']} mm")
    print(f"  {'Gravity axis':<32}: {'XYZ'[accuracy['gravity_axis']]} (axis {accuracy['gravity_axis']})")
    print(f"{'─'*58}")
    print(f"  {'Launch speed':<32}: {accuracy['speed_mps']} m/s  ({accuracy['speed_kmh']} km/h)")
    if 'apex_height_m' in accuracy:
        print(f"  {'Apex height':<32}: {accuracy['apex_height_m']} m  "
              f"at t={accuracy['apex_time_s']} s")
    if 'range_m' in accuracy:
        lp = accuracy['landing_pos']
        print(f"  {'Predicted landing':<32}: {lp} m")
        print(f"  {'Horizontal range':<32}: {accuracy['range_m']} m")
    print(f"{'='*58}\n")

# ── ANIMATION RESUME WATCHDOG ────────────────────────────────────────────────
def _wait_and_resume(ani_ref):
    """
    Waits until recording/fitting is done, then resumes matplotlib animation.
    Runs in its own daemon thread so it never blocks UDP.
    """
    while True:
        time.sleep(0.05)
        with state.lock:
            m = state.mode
        if m in (Mode.DONE, Mode.IDLE):
            time.sleep(0.1)   # small grace period for fit results to be written
            try:
                if ani_ref[0] is not None:
                    ani_ref[0].resume()
                    print("[Viz] Animation RESUMED.")
            except Exception:
                pass
            return

# ── LIVE PLOT ─────────────────────────────────────────────────────────────────
MODE_COLORS = {
    Mode.IDLE:      ("#888888", "IDLE  --  Press P to start collecting"),
    Mode.RECORDING: ("#ff4400", "● RECORDING"),
    Mode.FITTING:   ("#00aaff", "⚙ Fitting..."),
    Mode.DONE:      ("#44ff88", "✔ Done  --  P for new capture  |  R to reset"),
}

def start_live_plot():
    fig = plt.figure(figsize=(16, 8), facecolor='#0a0a0a')
    try:
        fig.canvas.manager.set_window_title("OptiTrack Ball Tracker")
    except Exception:
        pass

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.05, right=0.97, top=0.90, bottom=0.08,
                           wspace=0.38, hspace=0.48)

    ax3d = fig.add_subplot(gs[:, 0], projection='3d', facecolor='#0a0a0a')
    ax_h = fig.add_subplot(gs[0, 1], facecolor='#0a0a0a')
    ax_e = fig.add_subplot(gs[1, 1], facecolor='#0a0a0a')
    ax_r = fig.add_subplot(gs[:, 2], facecolor='#0a0a0a')

    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor('#2a2a2a')
    ax3d.tick_params(colors='#666', labelsize=7)
    ax3d.set_xlabel('X (m)', color='#888', fontsize=8)
    ax3d.set_ylabel('Z (m)', color='#888', fontsize=8)
    ax3d.set_zlabel('Height (m)', color='#888', fontsize=8)
    ax3d.set_title('Live 3-D Position', color='#ccc', pad=4, fontsize=9)
    ax3d.set_xlim(*X_LIM); ax3d.set_ylim(*Z_LIM); ax3d.set_zlim(*Y_LIM)

    def _style(ax, title, xlabel, ylabel):
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='#888', labelsize=8)
        for sp in ax.spines.values(): sp.set_color('#333')
        ax.set_title(title,   color='#ccc', fontsize=9)
        ax.set_xlabel(xlabel, color='#888', fontsize=8)
        ax.set_ylabel(ylabel, color='#888', fontsize=8)
        ax.grid(True, color='#1a1a1a', linestyle='--', linewidth=0.6)

    _style(ax_h, 'Height vs Time',      'Time (s)',    'Height (m)')
    _style(ax_e, 'Per-Point 3-D Error', 'Point index', 'Error (mm)')
    ax_h.set_xlim(-0.05, 0.5); ax_h.set_ylim(*Y_LIM)
    ax_r.axis('off')
    ax_r.set_title('Analysis Report', color='#ccc', fontsize=9, pad=4)

    # Persistent artists
    trail_sc  = ax3d.scatter([], [], [], s=16, alpha=0.55, cmap='cool',  label='Trail')
    cur_sc    = ax3d.scatter([], [], [], c='#ff3333', s=90, zorder=6,    label='Current')
    samp_sc   = ax3d.scatter([], [], [], c='#ff8800', s=50, zorder=5,    label='Samples')
    fit_ln,   = ax3d.plot([], [], [], color='#00aaff', lw=2,             label='Fit')
    apex_sc   = ax3d.scatter([], [], [], c='gold',   s=160, marker='^',  zorder=7, label='Apex')
    land_sc   = ax3d.scatter([], [], [], c='#44ff44',s=160, marker='*',  zorder=7, label='Landing')
    ax3d.legend(fontsize=7, loc='upper left',
                facecolor='#111', edgecolor='#333', labelcolor='#bbb')

    obs_ln, = ax_h.plot([], [], 'o', color='#ff8800', ms=5,  label='Samples')
    fit_h,  = ax_h.plot([], [], color='#00aaff', lw=2,       label='Fit')
    apex_vl = ax_h.axvline(0, color='gold',    lw=1, ls='--', alpha=0, label='Apex')
    land_vl = ax_h.axvline(0, color='#44ff44', lw=1, ls='--', alpha=0, label='Landing')
    cur_hl  = ax_h.axhline(0, color='#ff3333', lw=0.8, ls=':', alpha=0)
    ax_h.legend(fontsize=7, facecolor='#111', edgecolor='#333', labelcolor='#bbb')

    # Progress ring: text showing N collected / N_SAMPLES
    prog_txt = ax3d.text2D(0.02, 0.02, '', transform=ax3d.transAxes,
                           color='#ffdd00', fontsize=10, fontfamily='monospace')

    report_txt = ax_r.text(0.04, 0.97, '', transform=ax_r.transAxes,
                           color='#cccccc', fontsize=8.5,
                           verticalalignment='top', fontfamily='monospace')
    status_bar = fig.text(0.5, 0.95, '', ha='center', va='top',
                          fontsize=12, fontweight='bold')

    ani_ref = [None]   # set after FuncAnimation is created

    # ── Key handler ───────────────────────────────────────────────────────────
    def on_key(event):
        key = (event.key or '').lower()

        if key == 'p':
            with state.lock:
                m = state.mode
            if m in (Mode.IDLE, Mode.DONE):
                with state.lock:
                    state.mode           = Mode.RECORDING
                    state.arm_time       = time.time()
                    state.record_start_t = None
                    state.record_end_t   = None
                    state.fit_done_t     = None
                    state.samples_t      = []
                    state.samples_p      = []
                    state._target_n      = N_SAMPLES
                    state.fit_result     = None
                    state.timing         = None
                    state.accuracy       = None
                # Pause animation so UDP gets full GIL during collection
                if ani_ref[0] is not None:
                    ani_ref[0].pause()
                    print(f"[Key] Animation PAUSED. Collecting {N_SAMPLES} samples at full rate...")
                # Start a watchdog that resumes animation when done
                threading.Thread(target=_wait_and_resume, args=(ani_ref,), daemon=True).start()

        elif key == 'r':
            with state.lock:
                state.mode       = Mode.IDLE
                state.fit_result = None
                state.timing     = None
                state.accuracy   = None
                state.samples_t  = []
                state.samples_p  = []
                state._target_n  = 0
            if ani_ref[0] is not None:
                try: ani_ref[0].resume()
                except Exception: pass
            print("[Key] Reset.")

        elif key == 'q':
            plt.close('all')

    fig.canvas.mpl_connect('key_press_event', on_key)

    # ── Animation update ──────────────────────────────────────────────────────
    def update(_frame):
        # Only redraw if something actually changed -- this lets mouse
        # rotation work smoothly without being interrupted by redraws
        with state.lock:
            dirty = state.plot_dirty
            if dirty:
                state.plot_dirty = False

        if not dirty:
            return

        with state.lock:
            buf      = list(state.buffer)
            mode     = state.mode
            fit      = state.fit_result
            s_pts    = list(state.samples_p)
            s_ts     = list(state.samples_t)
            n_so_far = len(s_pts)
            need     = state._target_n
            timing   = state.timing
            accuracy = state.accuracy
            rb       = state.rb_name
            conn     = state.connected
            hw_fps   = state.hw_fps

        # Status bar
        color, label = MODE_COLORS[mode]
        fps_str  = f"  |  {hw_fps:.1f} Hz" if hw_fps > 0 else ""
        mode_str = "marker" if TRACKING_MODE == "single_marker" else f"RB: '{rb}'"
        rb_str   = f"  |  {mode_str}{fps_str}" if conn else "  -- waiting for stream..."
        status_bar.set_text(f"{label}{rb_str}")
        status_bar.set_color(color)

        # Live trail
        if buf:
            trail = np.array([b[1] for b in buf[-TRAIL_LEN:]])
            n     = len(trail)
            trail_sc._offsets3d = (trail[:, 0], trail[:, 2], trail[:, 1])
            trail_sc.set_color(plt.cm.cool(np.linspace(0, 1, n)))
            cur = trail[-1]
            cur_sc._offsets3d = ([cur[0]], [cur[2]], [cur[1]])
            cur_hl.set_ydata([cur[1], cur[1]]); cur_hl.set_alpha(0.4)
        else:
            trail_sc._offsets3d = cur_sc._offsets3d = ([], [], [])

        # Recording progress indicator
        if mode == Mode.RECORDING:
            collected = n_so_far
            prog_txt.set_text(f"{collected}/{N_SAMPLES}")
        else:
            prog_txt.set_text('')

        g_ax = fit['gravity_axis'] if fit else GRAVITY_AXIS

        # Collected sample points
        if s_pts:
            sp = np.array(s_pts)
            st = np.array(s_ts) - s_ts[0]
            samp_sc._offsets3d = (sp[:, 0], sp[:, 2], sp[:, 1])
            obs_ln.set_data(st, sp[:, g_ax])
            # Auto-scale height plot x-axis to fit the samples
            if len(st) > 1:
                ax_h.set_xlim(-0.02, st[-1] * 1.3 + 0.01)
        else:
            samp_sc._offsets3d = ([], [], [])
            obs_ln.set_data([], [])

        # Fitted trajectory
        if fit:
            t_end   = max((fit['t_apex'] or 0) * 2.2, fit['duration_s'] * 1.5, 0.3)
            t_range = np.linspace(0, t_end, 300)
            pred    = fit['predict'](t_range)
            fit_ln.set_data_3d(pred[:, 0], pred[:, 2], pred[:, 1])
            fit_h.set_data(t_range, pred[:, g_ax])
            ax_h.set_xlim(-0.02, t_end * 1.1)

            if fit['t_apex'] and 0 < fit['t_apex'] < t_end:
                ap = fit['predict'](fit['t_apex'])
                apex_sc._offsets3d = ([ap[0]], [ap[2]], [ap[1]])
                apex_vl.set_xdata([fit['t_apex']] * 2); apex_vl.set_alpha(0.85)
            else:
                apex_sc._offsets3d = ([], [], []); apex_vl.set_alpha(0)

            t_land = predict_landing(fit)
            if t_land and 0 < t_land < t_end + 0.5:
                lp = fit['predict'](t_land)
                land_sc._offsets3d = ([lp[0]], [lp[2]], [lp[1]])
                land_vl.set_xdata([t_land] * 2); land_vl.set_alpha(0.85)
            else:
                land_sc._offsets3d = ([], [], []); land_vl.set_alpha(0)

            # Per-point error bars
            ax_e.cla(); _style(ax_e, 'Per-Point 3-D Error', 'Point index', 'Error (mm)')
            errs = fit['per_pt_err']
            norm = errs / max(errs.max(), 1e-9)
            ax_e.bar(range(len(errs)), errs,
                     color=plt.cm.RdYlGn_r(norm), alpha=0.85, width=0.8)
            ax_e.axhline(fit['mean_err_mm'], color='#ffdd00', lw=1.2, ls='--',
                         label=f"mean {fit['mean_err_mm']:.1f} mm")
            ax_e.axhline(fit['rms_3d_mm'],   color='#00aaff', lw=1.0, ls=':',
                         label=f"RMS  {fit['rms_3d_mm']:.1f} mm")
            ax_e.legend(fontsize=7, facecolor='#111', edgecolor='#333', labelcolor='#bbb')
            # Label each bar with its index
            for i, e in enumerate(errs):
                ax_e.text(i, e + errs.max()*0.02, str(i),
                          ha='center', va='bottom', color='#888', fontsize=6)
        else:
            fit_ln.set_data_3d([], [], [])
            fit_h.set_data([], [])
            apex_sc._offsets3d = land_sc._offsets3d = ([], [], [])
            apex_vl.set_alpha(0); land_vl.set_alpha(0)

        # Report panel
        if timing and accuracy:
            g_label = f"{'XYZ'[accuracy['gravity_axis']]} (axis {accuracy['gravity_axis']})"
            lines = [
                f"N_SAMPLES = {timing['n_samples']}",
                "",
                "TIMING",
                f"  P press -> 1st sample : {timing['arm_to_first_sample_s']} s",
                f"  Collection window     : {timing['collection_duration_s']} s",
                f"  Fit compute           : {timing['fit_compute_s']} s",
                f"  Total (P -> result)   : {timing['total_arm_to_fit_s']} s",
                f"  Sample rate           : {timing['sample_rate_hz']} Hz",
                f"  Motive stream         : {timing['hw_fps']} Hz",
                "",
                "ACCURACY",
                f"  3D RMS error          : {accuracy['rms_3d_mm']} mm",
                f"  Mean error            : {accuracy['mean_err_mm']} mm",
                f"  Max error             : {accuracy['max_err_mm']} mm",
                f"  Residual X            : {accuracy['res_x_mm']} mm",
                f"  Residual Y            : {accuracy['res_y_mm']} mm",
                f"  Residual Z            : {accuracy['res_z_mm']} mm",
                f"  Gravity axis          : {g_label}",
                "",
                "KINEMATICS",
                f"  Launch speed          : {accuracy['speed_mps']} m/s",
                f"                        : {accuracy['speed_kmh']} km/h",
            ]
            if 'apex_height_m' in accuracy:
                lines += [f"  Apex height           : {accuracy['apex_height_m']} m",
                          f"  Apex time             : {accuracy['apex_time_s']} s"]
            if 'range_m' in accuracy:
                lp = accuracy['landing_pos']
                lines += [f"  Landing               : {lp} m",
                          f"  Horizontal range      : {accuracy['range_m']} m"]
            report_txt.set_text('\n'.join(lines))

        elif mode == Mode.RECORDING:
            report_txt.set_text(
                f"RECORDING\n\n"
                f"Collecting point\n"
                f"{n_so_far} / {N_SAMPLES}\n\n"
                f"Move the ball now!"
            )
        elif mode == Mode.FITTING:
            report_txt.set_text("Fitting parabola\nto collected points...")
        else:
            report_txt.set_text(
                f"N_SAMPLES = {N_SAMPLES}\n\n"
                f"P  -  collect {N_SAMPLES} pts & fit\n"
                f"R  -  reset\n"
                f"Q  -  quit\n\n"
                f"Press P then move\nthe ball."
            )

    ani_ref[0] = animation.FuncAnimation(fig, update, interval=PLOT_INTERVAL_MS,
                                         blit=False, cache_frame_data=False)
    plt.show()

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 58)
    print("  OptiTrack Ball Tracker  —  Fixed-Count Sample Mode")
    print("=" * 58)
    print(f"  Motive server : {SERVER_IP}")
    print(f"  N_SAMPLES     : {N_SAMPLES}   (change in CONFIG)")
    print(f"  TRACKING_MODE : {TRACKING_MODE}")
    print(f"  GRAVITY_AXIS  : {GRAVITY_AXIS} ({'XYZ'[GRAVITY_AXIS]}-up)  "
          f"(auto-corrected if wrong)")
    print()
    print("  P  ->  Collect N_SAMPLES points then fit")
    print("  R  ->  Reset")
    print("  Q  ->  Quit")
    print()

    threading.Thread(target=udp_listener, daemon=True).start()
    time.sleep(0.8)
    print("[Viz] Opening plot...\n")
    start_live_plot()

if __name__ == "__main__":
    main()
