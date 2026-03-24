"""
OptiTrack Ball Tracker — Rolling Window Prediction Mode
========================================================
Continuously tracks the ball. Press P to toggle live prediction on/off.

While PREDICTING:
  - Uses the last N_FIT points to fit a parabola in real time
  - Draws the fitted curve forward in time (the prediction)
  - As new points arrive AFTER the fit window, plots them in a different
    colour so you can see how well the prediction matches reality
  - The "future error" panel shows 3D distance between predicted and
    actual positions for every point that arrives after the fit

Controls:
  P  ->  Toggle rolling prediction on / off
  R  ->  Reset / clear all prediction data
  Q  ->  Quit

Tune in CONFIG:
  N_FIT          = 20    # points used for each fit
  PREDICT_AHEAD_S = 1.0  # how many seconds ahead to draw the prediction arc
  GRAVITY_AXIS   = 1     # 1=Y-up, 2=Z-up (auto-detected)
  TRACKING_MODE  = "rigid_body" or "single_marker"

Requirements: pip install numpy scipy matplotlib
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
GRAVITY_AXIS     = 1          # 1=Y-up, 2=Z-up. Auto-detected if wrong.
TRACKING_MODE    = "rigid_body"  # "rigid_body" or "single_marker"

N_FIT            = 20         # ← points used to fit the parabola
PREDICT_AHEAD_S  = 1.0        # ← how far ahead (seconds) to draw the prediction
BUFFER_SIZE      = 2000       # total live trail length
TRAIL_LEN        = 60         # points shown in live trail
FUTURE_WINDOW    = 200        # max future points to store for error comparison
PLOT_INTERVAL_MS = 80         # screen refresh ms

# Capture volume limits (metres)
X_LIM = (-1.5, 1.5)
Y_LIM = (0.0,  2.5)
Z_LIM = (-1.5, 1.5)

# ── STATE MACHINE ─────────────────────────────────────────────────────────────
class Mode:
    IDLE        = "IDLE"        # live view, no prediction
    PREDICTING  = "PREDICTING"  # rolling fit + future comparison active

# ── SHARED STATE ──────────────────────────────────────────────────────────────
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()

        # live stream — ALL frames go here
        self.buffer    = deque(maxlen=BUFFER_SIZE)  # (t, xyz)
        self.connected = False
        self.rb_name   = "unknown"
        self.hw_fps    = 0.0
        self._fps_ts   = deque(maxlen=200)

        # mode
        self.mode      = Mode.IDLE

        # rolling fit — updated by fit thread
        self.fit_result    = None   # latest fit dict
        self.fit_window_t  = []     # timestamps of the N_FIT points used
        self.fit_window_p  = []     # positions  of the N_FIT points used
        self.fit_locked_at = None   # wall-clock when this fit was computed

        # future points: points that arrived AFTER the fit window ended
        # used to measure how well the prediction holds up
        self.future_t   = deque(maxlen=FUTURE_WINDOW)  # timestamps
        self.future_p   = deque(maxlen=FUTURE_WINDOW)  # positions
        self.future_err = deque(maxlen=FUTURE_WINDOW)  # 3D error vs prediction (mm)

        # fit thread control
        self._fit_running = False   # True while a fit is in progress
        self.plot_dirty   = True

state = SharedState()

# ── NATNET PARSERS ────────────────────────────────────────────────────────────
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

def parse_unlabeled_markers(data: bytes) -> List[np.ndarray]:
    try:
        offset = 8
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
            if TRACKING_MODE == "rigid_body" and rb_id_filter is None \
               and (time.time() - last_req) > 3.0:
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

        now = time.time()
        pos = None

        if TRACKING_MODE == "single_marker":
            markers = parse_unlabeled_markers(data)
            if markers:
                with state.lock:
                    last = state.buffer[-1][1] if state.buffer else None
                pos = (min(markers, key=lambda m: np.linalg.norm(m - last))
                       if last is not None else markers[0]) if len(markers) > 1 else markers[0]
        else:
            for rb_id, p in parse_rigid_bodies(data):
                if rb_id_filter is not None and rb_id != rb_id_filter:
                    continue
                pos = p; break

        if pos is None:
            continue

        with state.lock:
            state.buffer.append((now, pos))
            state.connected  = True
            state.plot_dirty = True
            state._fps_ts.append(now)
            if len(state._fps_ts) >= 2:
                span = state._fps_ts[-1] - state._fps_ts[0]
                if span > 0:
                    state.hw_fps = (len(state._fps_ts) - 1) / span

            cur_mode   = state.mode
            fit        = state.fit_result
            locked_at  = state.fit_locked_at

        # ── When PREDICTING: check if this point is AFTER the fit window ──────
        if cur_mode == Mode.PREDICTING and fit is not None and locked_at is not None:
            # Points that arrive after the fit window was locked are "future" points
            if now > locked_at:
                t_rel     = now - fit['t0_wall']   # time relative to fit t=0
                pred      = fit['predict'](t_rel)
                err_mm    = float(np.linalg.norm(pos - pred)) * 1000
                with state.lock:
                    state.future_t.append(now)
                    state.future_p.append(pos.copy())
                    state.future_err.append(err_mm)

        # ── Trigger a new rolling fit whenever we have N_FIT points ──────────
        if cur_mode == Mode.PREDICTING:
            with state.lock:
                n_buf        = len(state.buffer)
                fit_running  = state._fit_running
            if n_buf >= N_FIT and not fit_running:
                with state.lock:
                    # Grab the latest N_FIT points
                    recent = list(state.buffer)[-N_FIT:]
                    state._fit_running = True
                threading.Thread(
                    target=_run_rolling_fit,
                    args=(recent,),
                    daemon=True
                ).start()

# ── ROLLING FIT THREAD ────────────────────────────────────────────────────────
def _run_rolling_fit(recent: list):
    """Fit a parabola to the N_FIT most recent points, then unlock."""
    times  = np.array([r[0] for r in recent])
    points = np.array([r[1] for r in recent])

    fit = fit_parabola(times, points)

    with state.lock:
        if fit:
            state.fit_result    = fit
            state.fit_window_t  = list(times)
            state.fit_window_p  = list(points)
            state.fit_locked_at = times[-1]   # wall-clock of last fit point
            # Clear future buffer — new fit means old future points are stale
            state.future_t.clear()
            state.future_p.clear()
            state.future_err.clear()
        state._fit_running  = False
        state.plot_dirty    = True

# ── PARABOLA FIT ──────────────────────────────────────────────────────────────
def fit_parabola(times: np.ndarray, points: np.ndarray) -> Optional[dict]:
    if len(times) < 3:
        return None

    t_wall0 = float(times[0])
    t = times - t_wall0   # relative time starting at 0

    # Auto-detect gravity axis
    curvatures = [abs(np.polyfit(t, points[:, ax], 2)[0]) for ax in range(3)]
    detected   = int(np.argmax(curvatures))
    used_axis  = GRAVITY_AXIS
    if detected != GRAVITY_AXIS:
        ratio = curvatures[detected] / max(curvatures[GRAVITY_AXIS], 1e-9)
        if ratio > 2.0:
            used_axis = detected

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
            popt, _ = curve_fit(model, t, p_obs, p0=[p_obs[0], v_init], maxfev=2000)
        except Exception:
            c    = np.polyfit(t, p_obs - 0.5 * a_fixed * t**2, 1)
            popt = [c[1], c[0]]
        fitted.append(popt)
        residuals.append(np.sqrt(np.mean((p_obs - model(t, *popt))**2)))

    (x0, vx), (y0, vy), (z0, vz) = fitted
    _acc = acc

    def predict(t_rel):
        """t_rel = seconds since t_wall0 (first point in fit window)."""
        tq = np.atleast_1d(np.array(t_rel, dtype=float))
        return np.stack([
            x0 + vx*tq + 0.5*_acc[0]*tq**2,
            y0 + vy*tq + 0.5*_acc[1]*tq**2,
            z0 + vz*tq + 0.5*_acc[2]*tq**2,
        ], axis=-1).squeeze()

    v_up   = [vx, vy, vz][used_axis]
    t_apex = v_up / G if v_up > 0 else None

    pred_pts   = predict(t)
    per_pt_err = np.linalg.norm(points - pred_pts, axis=1) * 1000

    return dict(
        x0=x0, y0=y0, z0=z0, vx=vx, vy=vy, vz=vz,
        speed        = float(np.sqrt(vx**2 + vy**2 + vz**2)),
        residuals_mm = [r * 1000 for r in residuals],
        per_pt_err   = per_pt_err,
        rms_3d_mm    = float(np.sqrt(np.mean(per_pt_err**2))),
        mean_err_mm  = float(np.mean(per_pt_err)),
        n_points     = len(times),
        duration_s   = float(t[-1]),
        t_apex       = t_apex,
        predict      = predict,
        t0_wall      = t_wall0,      # absolute wall-clock of first fit point
        gravity_axis = used_axis,
    )

def predict_landing(fit: dict, floor_height=0.0) -> Optional[float]:
    g  = fit['gravity_axis']
    y0 = [fit['x0'], fit['y0'], fit['z0']][g]
    vy = [fit['vx'], fit['vy'], fit['vz']][g]
    a  = -0.5 * G; b = vy; c = y0 - floor_height
    disc = b**2 - 4*a*c
    if disc < 0: return None
    t1 = (-b + np.sqrt(disc)) / (2*a)
    t2 = (-b - np.sqrt(disc)) / (2*a)
    cands = [t for t in [t1, t2] if t > 0]
    return min(cands) if cands else None

# ── LIVE PLOT ─────────────────────────────────────────────────────────────────
MODE_COLORS = {
    Mode.IDLE:       ("#888888", "IDLE  --  Press P to start prediction"),
    Mode.PREDICTING: ("#00ffaa", "● PREDICTING  --  P to stop  |  R to reset"),
}

def start_live_plot():
    fig = plt.figure(figsize=(16, 8), facecolor='#0a0a0a')
    try:
        fig.canvas.manager.set_window_title("OptiTrack Ball Tracker — Rolling Prediction")
    except Exception:
        pass

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.05, right=0.97, top=0.90, bottom=0.08,
                           wspace=0.38, hspace=0.48)

    ax3d = fig.add_subplot(gs[:, 0], projection='3d', facecolor='#0a0a0a')
    ax_h = fig.add_subplot(gs[0, 1], facecolor='#0a0a0a')   # height vs time
    ax_e = fig.add_subplot(gs[1, 1], facecolor='#0a0a0a')   # future error
    ax_r = fig.add_subplot(gs[:, 2], facecolor='#0a0a0a')   # report

    # 3D static styling
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

    _style(ax_h, 'Height vs Time  (orange=fit window, green=future actual, blue=prediction)',
           'Time (s)', 'Height (m)')
    _style(ax_e, f'Prediction Error (future points vs parabola)',
           'Future point index', 'Error (mm)')
    ax_h.set_xlim(-0.1, PREDICT_AHEAD_S * 1.5)
    ax_h.set_ylim(*Y_LIM)
    ax_r.axis('off')
    ax_r.set_title('Live Stats', color='#ccc', fontsize=9, pad=4)

    # ── Persistent 3D artists ─────────────────────────────────────────────────
    trail_sc   = ax3d.scatter([], [], [], s=14, alpha=0.4, cmap='cool',   label='Trail')
    cur_sc     = ax3d.scatter([], [], [], c='#ff3333', s=80,  zorder=6,   label='Current')
    window_sc  = ax3d.scatter([], [], [], c='#ff8800', s=40,  zorder=5,   label=f'Fit window ({N_FIT} pts)')
    future_sc  = ax3d.scatter([], [], [], c='#00ff88', s=40,  zorder=5,   label='Future actual')
    pred_ln,   = ax3d.plot([], [], [], color='#00aaff', lw=2, alpha=0.9,  label='Prediction')
    apex_sc    = ax3d.scatter([], [], [], c='gold',    s=140, marker='^', zorder=7, label='Apex')
    land_sc    = ax3d.scatter([], [], [], c='#ff44ff', s=140, marker='*', zorder=7, label='Landing')
    ax3d.legend(fontsize=6, loc='upper left',
                facecolor='#111', edgecolor='#333', labelcolor='#bbb')

    # ── Persistent height plot artists ────────────────────────────────────────
    win_ln,  = ax_h.plot([], [], 'o', color='#ff8800', ms=5,  label=f'Fit window ({N_FIT} pts)', zorder=5)
    fut_ln,  = ax_h.plot([], [], 's', color='#00ff88', ms=5,  label='Future actual',             zorder=5)
    pred_h,  = ax_h.plot([], [], color='#00aaff', lw=2,       label='Prediction',                zorder=4)
    apex_vl  = ax_h.axvline(0, color='gold',    lw=1, ls='--', alpha=0, label='Apex')
    land_vl  = ax_h.axvline(0, color='#ff44ff', lw=1, ls='--', alpha=0, label='Landing')
    ax_h.legend(fontsize=7, facecolor='#111', edgecolor='#333', labelcolor='#bbb')

    report_txt = ax_r.text(0.04, 0.97, '', transform=ax_r.transAxes,
                           color='#cccccc', fontsize=9,
                           verticalalignment='top', fontfamily='monospace')
    status_bar = fig.text(0.5, 0.95, '', ha='center', va='top',
                          fontsize=12, fontweight='bold')

    # ── Key handler ───────────────────────────────────────────────────────────
    def on_key(event):
        key = (event.key or '').lower()
        if key == 'p':
            with state.lock:
                m = state.mode
            if m == Mode.IDLE:
                with state.lock:
                    state.mode = Mode.PREDICTING
                    state.fit_result    = None
                    state.future_t.clear()
                    state.future_p.clear()
                    state.future_err.clear()
                print(f"[Key] PREDICTING — rolling {N_FIT}-point fit active.")
            else:
                with state.lock:
                    state.mode = Mode.IDLE
                print("[Key] Prediction OFF.")
        elif key == 'r':
            with state.lock:
                state.mode = Mode.IDLE
                state.fit_result    = None
                state.fit_window_t  = []
                state.fit_window_p  = []
                state.future_t.clear()
                state.future_p.clear()
                state.future_err.clear()
            print("[Key] Reset.")
        elif key == 'q':
            plt.close('all')

    fig.canvas.mpl_connect('key_press_event', on_key)

    # ── Animation update ──────────────────────────────────────────────────────
    def update(_frame):
        with state.lock:
            dirty = state.plot_dirty
            if dirty:
                state.plot_dirty = False
        if not dirty:
            return

        with state.lock:
            buf       = list(state.buffer)
            mode      = state.mode
            fit       = state.fit_result
            win_t     = list(state.fit_window_t)
            win_p     = list(state.fit_window_p)
            fut_t     = list(state.future_t)
            fut_p     = list(state.future_p)
            fut_err   = list(state.future_err)
            rb        = state.rb_name
            conn      = state.connected
            hw_fps    = state.hw_fps

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
        else:
            trail_sc._offsets3d = cur_sc._offsets3d = ([], [], [])

        g_ax = fit['gravity_axis'] if fit else GRAVITY_AXIS

        # Fit window points (orange)
        if win_p:
            wp  = np.array(win_p)
            wt  = np.array(win_t) - win_t[0]
            window_sc._offsets3d = (wp[:, 0], wp[:, 2], wp[:, 1])
            win_ln.set_data(wt, wp[:, g_ax])
        else:
            window_sc._offsets3d = ([], [], [])
            win_ln.set_data([], [])

        # Future actual points (green)
        if fut_p:
            fp  = np.array(fut_p)
            # time relative to fit t=0
            ft  = np.array(fut_t) - fit['t0_wall'] if fit else np.zeros(len(fut_t))
            future_sc._offsets3d = (fp[:, 0], fp[:, 2], fp[:, 1])
            fut_ln.set_data(ft, fp[:, g_ax])
        else:
            future_sc._offsets3d = ([], [], [])
            fut_ln.set_data([], [])

        # Prediction arc (blue)
        if fit:
            t_fit_dur = fit['duration_s']
            # Draw from t=0 to t = fit_duration + PREDICT_AHEAD_S
            t_end   = t_fit_dur + PREDICT_AHEAD_S
            t_range = np.linspace(0, t_end, 400)
            pred    = fit['predict'](t_range)
            pred_ln.set_data_3d(pred[:, 0], pred[:, 2], pred[:, 1])
            pred_h.set_data(t_range, pred[:, g_ax])

            # Auto-scale height plot to show fit window + prediction
            total_t = max(t_end, ft[-1] if fut_t else 0) if fut_t else t_end
            ax_h.set_xlim(-0.05, total_t * 1.1)

            # Apex marker
            if fit['t_apex'] and 0 < fit['t_apex'] < t_end:
                ap = fit['predict'](fit['t_apex'])
                apex_sc._offsets3d = ([ap[0]], [ap[2]], [ap[1]])
                apex_vl.set_xdata([fit['t_apex']] * 2); apex_vl.set_alpha(0.8)
            else:
                apex_sc._offsets3d = ([], [], []); apex_vl.set_alpha(0)

            # Landing marker
            t_land = predict_landing(fit)
            if t_land and 0 < t_land < t_end + 1.0:
                lp = fit['predict'](t_land)
                land_sc._offsets3d = ([lp[0]], [lp[2]], [lp[1]])
                land_vl.set_xdata([t_land] * 2); land_vl.set_alpha(0.8)
            else:
                land_sc._offsets3d = ([], [], []); land_vl.set_alpha(0)

            # Future error bar chart
            ax_e.cla()
            _style(ax_e, f'Prediction Error — future pts vs parabola',
                   'Future point index', 'Error (mm)')
            if fut_err:
                errs = np.array(fut_err)
                norm = errs / max(errs.max(), 1e-9)
                ax_e.bar(range(len(errs)), errs,
                         color=plt.cm.RdYlGn_r(norm), alpha=0.85, width=0.8)
                mean_e = float(np.mean(errs))
                ax_e.axhline(mean_e, color='#ffdd00', lw=1.2, ls='--',
                             label=f'mean {mean_e:.1f} mm')
                ax_e.legend(fontsize=7, facecolor='#111',
                             edgecolor='#333', labelcolor='#bbb')

            # Report panel
            g_label = f"{'XYZ'[fit['gravity_axis']]} (axis {fit['gravity_axis']})"
            n_fut   = len(fut_err)
            mean_fut_err = float(np.mean(fut_err)) if fut_err else None
            max_fut_err  = float(np.max(fut_err))  if fut_err else None

            lines = [
                f"FIT WINDOW : {N_FIT} pts",
                f"PREDICT    : +{PREDICT_AHEAD_S} s ahead",
                f"Gravity ax : {g_label}",
                "",
                "FIT QUALITY (window)",
                f"  RMS err  : {fit['rms_3d_mm']:.2f} mm",
                f"  Mean err : {fit['mean_err_mm']:.2f} mm",
                f"  Speed    : {fit['speed']:.3f} m/s",
                f"           : {fit['speed']*3.6:.2f} km/h",
            ]
            if fit['t_apex']:
                ap = fit['predict'](fit['t_apex'])
                lines += [f"  Apex ht  : {ap[g_ax]:.3f} m @ {fit['t_apex']:.3f} s"]
            if t_land:
                lines += [f"  Landing  : t={t_land:.3f} s"]
            lines += [""]
            if n_fut > 0:
                lines += [
                    f"PREDICTION ACCURACY",
                    f"  Future pts     : {n_fut}",
                    f"  Mean fut. err  : {mean_fut_err:.2f} mm",
                    f"  Max fut. err   : {max_fut_err:.2f} mm",
                ]
                # Quality judgement
                if mean_fut_err < 10:
                    lines += ["  Quality : ✓ EXCELLENT (<10 mm)"]
                elif mean_fut_err < 30:
                    lines += ["  Quality : ~ GOOD (<30 mm)"]
                elif mean_fut_err < 100:
                    lines += ["  Quality : ! FAIR (<100 mm)"]
                else:
                    lines += ["  Quality : ✗ POOR (>100 mm)"]
            else:
                lines += ["PREDICTION ACCURACY",
                          "  Waiting for future",
                          "  points to arrive..."]

            report_txt.set_text('\n'.join(lines))

        else:
            pred_ln.set_data_3d([], [], [])
            pred_h.set_data([], [])
            apex_sc._offsets3d = land_sc._offsets3d = ([], [], [])
            apex_vl.set_alpha(0); land_vl.set_alpha(0)

            if mode == Mode.PREDICTING:
                report_txt.set_text(
                    f"PREDICTING\n\n"
                    f"Waiting for\n{N_FIT} points...\n\n"
                    f"Move the ball!"
                )
            else:
                report_txt.set_text(
                    f"N_FIT = {N_FIT} pts\n"
                    f"PREDICT = +{PREDICT_AHEAD_S} s\n\n"
                    f"P  -  start prediction\n"
                    f"R  -  reset\n"
                    f"Q  -  quit\n\n"
                    f"Press P then\nmove the ball."
                )

    ani = animation.FuncAnimation(fig, update, interval=PLOT_INTERVAL_MS,
                                  blit=False, cache_frame_data=False)
    plt.show()

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  OptiTrack Ball Tracker — Rolling Window Prediction")
    print("=" * 60)
    print(f"  Motive server  : {SERVER_IP}")
    print(f"  N_FIT          : {N_FIT}  (points used per fit)")
    print(f"  PREDICT_AHEAD  : {PREDICT_AHEAD_S} s")
    print(f"  TRACKING_MODE  : {TRACKING_MODE}")
    print(f"  GRAVITY_AXIS   : {GRAVITY_AXIS} ({'XYZ'[GRAVITY_AXIS]}-up, auto-corrected)")
    print()
    print("  P  ->  Toggle rolling prediction on / off")
    print("  R  ->  Reset")
    print("  Q  ->  Quit")
    print()

    threading.Thread(target=udp_listener, daemon=True).start()
    time.sleep(0.8)
    print("[Viz] Opening plot...\n")
    start_live_plot()

if __name__ == "__main__":
    main()
