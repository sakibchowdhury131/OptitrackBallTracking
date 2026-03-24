"""
OptiTrack Live Ball Tracker - Rigid Body Version (Raw UDP, No SDK)
===================================================================
Controls:
  P  ->  Arm / disarm recording. While ARMED the next throw is captured,
         trajectory fitted, and timing + accuracy analysis displayed.
  R  ->  Reset: clear last fit and flight data, ready for a new throw.
  Q  ->  Quit.

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

# ?? CONFIG ????????????????????????????????????????????????????????????????????
SERVER_IP      = "10.156.84.112"
CLIENT_IP      = "0.0.0.0"
MULTICAST_ADDR = "239.255.42.99"
DATA_PORT      = 1511

NAT_FRAMEOFDATA = 7

G            = 9.81
# GRAVITY_AXIS: which axis does OptiTrack report as "up"?
#   1 = Y-up  (Motive default, most common)
#   2 = Z-up  (some configurations)
# If your Y residual is huge but X/Z are fine, change this to 2.
GRAVITY_AXIS = 1

MIN_POINTS      = 5
BUFFER_SIZE     = 2000   # larger so fast UDP never drops frames into matplotlib
FLIGHT_GAP_S    = 0.30
MIN_SPEED       = 0.4
TRAIL_LEN       = 80

# Plot refresh rate. UDP listener runs in its own thread at FULL Motive speed
# regardless of this -- this only controls how often the screen redraws.
PLOT_INTERVAL_MS = 100

# Axis limits -- adjust to your capture volume
X_LIM = (-1.5, 1.5)
Y_LIM = (0.0,  2.5)
Z_LIM = (-1.5, 1.5)

# ?? RECORDER STATE MACHINE ????????????????????????????????????????????????????
# IDLE      -> user sees live position only, nothing is recorded
# ARMED     -> waiting for the ball to move (flight detector watching)
# RECORDING -> ball is in flight, points being collected
# FITTING   -> flight ended, fitting in progress
# DONE      -> fit complete, results displayed
class Mode:
    IDLE      = "IDLE"
    ARMED     = "ARMED"
    RECORDING = "RECORDING"
    FITTING   = "FITTING"
    DONE      = "DONE"

# ?? SHARED STATE ??????????????????????????????????????????????????????????????
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()

        # live stream
        self.buffer       = deque(maxlen=BUFFER_SIZE)   # (timestamp, xyz)
        self.connected    = False
        self.rb_name      = "unknown"

        # true hardware frame rate measured in UDP thread (NOT affected by matplotlib)
        self.hw_fps       = 0.0
        self._fps_times   = deque(maxlen=200)   # timestamps of last N UDP frames

        # recorder
        self.mode           = Mode.IDLE
        self.arm_time       = None   # wall-clock when P was pressed
        self.throw_time     = None   # wall-clock when motion first detected
        self.fit_done_time  = None   # wall-clock when fit finished

        self.flight_times  = []
        self.flight_points = []
        self.fit_result    = None    # dict from fit_parabola()
        self.timing        = None    # dict with timing analysis
        self.accuracy      = None    # dict with accuracy analysis

state = SharedState()

# ?? NATNET PARSER ?????????????????????????????????????????????????????????????

def unpack_string(data: bytes, offset: int) -> Tuple[str, int]:
    end = data.index(b'\x00', offset)
    return data[offset:end].decode('utf-8', errors='ignore'), end + 1

def parse_rigid_bodies(data: bytes) -> List[Tuple[int, np.ndarray]]:
    try:
        offset = 4 + 4   # header + frame_number

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
            rb_id          = struct.unpack_from('<i',    data, offset)[0]; offset += 4
            x, y, z        = struct.unpack_from('<fff',  data, offset);    offset += 12
            qx, qy, qz, qw = struct.unpack_from('<ffff', data, offset);   offset += 16
            _error         = struct.unpack_from('<f',    data, offset)[0]; offset += 4
            params         = struct.unpack_from('<h',    data, offset)[0]; offset += 2
            if bool(params & 0x01):
                results.append((rb_id, np.array([x, y, z], dtype=float)))
        return results
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

# ?? UDP LISTENER ??????????????????????????????????????????????????????????????
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
        print(f"[UDP] Multicast failed - trying unicast on :{DATA_PORT}")

    sock.settimeout(2.0)
    request_descriptions(sock)
    last_req = time.time()

    while True:
        try:
            data, _ = sock.recvfrom(65536)
        except socket.timeout:
            if rb_id_filter is None and (time.time() - last_req) > 3.0:
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
                        print(f"[UDP] Falling back to first RB: '{name}' (id={rb_id_filter})")
            continue

        if msg_id != NAT_FRAMEOFDATA:
            continue

        now    = time.time()
        bodies = parse_rigid_bodies(data)
        for rb_id, pos in bodies:
            if rb_id_filter is not None and rb_id != rb_id_filter:
                continue
            with state.lock:
                state.buffer.append((now, pos))
                state.connected = True
                state._fps_times.append(now)
                if len(state._fps_times) >= 2:
                    span = state._fps_times[-1] - state._fps_times[0]
                    if span > 0:
                        state.hw_fps = (len(state._fps_times) - 1) / span
            break

# ?? FLIGHT DETECTOR ???????????????????????????????????????????????????????????
class FlightDetector:
    def __init__(self, min_speed_mps=MIN_SPEED, window=6):
        self.min_speed = min_speed_mps
        self.window    = window

    def is_in_flight(self, recent: list) -> bool:
        if len(recent) < self.window:
            return False
        pts    = recent[-self.window:]
        times  = np.array([p[0] for p in pts])
        coords = np.array([p[1] for p in pts])
        dt     = times[-1] - times[0]
        if dt < 1e-6:
            return False
        return np.linalg.norm(coords[-1] - coords[0]) / dt > self.min_speed

# ?? PARABOLA FIT ??????????????????????????????????????????????????????????????
def fit_parabola(times: np.ndarray, points: np.ndarray) -> Optional[dict]:
    if len(times) < MIN_POINTS:
        return None

    t   = times - times[0]

    # ?? Auto-detect gravity axis ???????????????????????????????????????????????
    # For each axis, fit a parabola WITH gravity and WITHOUT gravity and pick
    # the axis where gravity makes the biggest difference (most curved).
    # This overrides GRAVITY_AXIS if the data clearly disagrees with it.
    curvatures = []
    for axis in range(3):
        p = points[:, axis]
        # Fit free parabola (a, b, c) -- no physics constraint
        coeffs = np.polyfit(t, p, 2)
        curvatures.append(abs(coeffs[0]))   # |coefficient of t^2| = |a/2|

    detected_axis = int(np.argmax(curvatures))
    used_axis     = GRAVITY_AXIS  # respect user config by default

    # If the detected axis strongly disagrees with config, warn and override
    if detected_axis != GRAVITY_AXIS:
        ratio = curvatures[detected_axis] / max(curvatures[GRAVITY_AXIS], 1e-9)
        if ratio > 3.0:   # detected axis has >3x more curvature -- override
            used_axis = detected_axis
            print(f"[Fit] WARNING: GRAVITY_AXIS={GRAVITY_AXIS} but data suggests "
                  f"axis {detected_axis} has gravity (curvatures: "
                  f"X={curvatures[0]:.3f} Y={curvatures[1]:.3f} Z={curvatures[2]:.3f}). "
                  f"Overriding. Set GRAVITY_AXIS={detected_axis} in CONFIG to silence.")
        else:
            print(f"[Fit] Curvatures: X={curvatures[0]:.3f} Y={curvatures[1]:.3f} "
                  f"Z={curvatures[2]:.3f}  -- using GRAVITY_AXIS={GRAVITY_AXIS}")

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

    def predict(t_query):
        t_q = np.atleast_1d(np.array(t_query, dtype=float))
        x   = x0 + vx * t_q
        y   = y0 + vy * t_q + 0.5 * acc[1] * t_q**2
        z   = z0 + vz * t_q
        return np.stack([x, y, z], axis=-1).squeeze()

    v_up   = [vx, vy, vz][used_axis]
    t_apex = v_up / G if v_up > 0 else None
    # store which axis was actually used
    _used_gravity_axis = used_axis

    # Per-point 3-D residuals
    pred_pts   = np.array([predict(ti) for ti in t])
    per_pt_err = np.linalg.norm(points - pred_pts, axis=1) * 1000   # mm

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
    a = -0.5 * G; b = fit['vy']; c = fit['y0'] - floor_height
    disc = b**2 - 4 * a * c
    if disc < 0:
        return None
    t1 = (-b + np.sqrt(disc)) / (2 * a)
    t2 = (-b - np.sqrt(disc)) / (2 * a)
    cands = [t for t in [t1, t2] if t > 0]
    return min(cands) if cands else None

# ?? PROCESSING THREAD ?????????????????????????????????????????????????????????
def processing_loop():
    detector  = FlightDetector()
    last_move = 0.0

    while True:
        time.sleep(0.02)

        with state.lock:
            mode   = state.mode
            recent = list(state.buffer)

        # Only run when ARMED or RECORDING
        if mode not in (Mode.ARMED, Mode.RECORDING):
            continue

        if not recent:
            continue

        in_flight = detector.is_in_flight(recent)
        now       = time.time()

        # ARMED -> RECORDING on first motion
        if mode == Mode.ARMED and in_flight:
            with state.lock:
                state.mode       = Mode.RECORDING
                state.throw_time = now
                state.flight_times.clear()
                state.flight_points.clear()
            print("[Processor] Throw detected! Recording...")
            last_move = now

        # Collect points while RECORDING
        if mode == Mode.RECORDING:
            if in_flight:
                t_last, p_last = recent[-1]
                with state.lock:
                    if not state.flight_times or t_last > state.flight_times[-1]:
                        state.flight_times.append(t_last)
                        state.flight_points.append(p_last.copy())
                last_move = now

            elif (now - last_move) > FLIGHT_GAP_S:
                # Flight ended -- grab data and fit
                with state.lock:
                    n     = len(state.flight_points)
                    f_ts  = list(state.flight_times)
                    f_pts = list(state.flight_points)
                    t_arm = state.arm_time
                    t_thr = state.throw_time
                    state.mode = Mode.FITTING

                print(f"[Processor] Flight ended ({n} pts). Fitting...")

                fit     = fit_parabola(np.array(f_ts), np.array(f_pts)) if n >= MIN_POINTS else None
                fit_end = time.time()

                if fit:
                    t_land = predict_landing(fit)
                    timing = {
                        "arm_to_throw_s"   : round(t_thr - t_arm,  3) if t_thr else None,
                        "throw_to_fit_s"   : round(fit_end - t_thr, 3) if t_thr else None,
                        "arm_to_fit_s"     : round(fit_end - t_arm, 3) if t_arm else None,
                        "flight_duration_s": round(fit['duration_s'], 3),
                        "n_points"         : fit['n_points'],
                        "sample_rate_hz"   : round(fit['sample_rate'], 1),
                        "hw_fps"           : round(state.hw_fps, 1),
                    }
                    accuracy = {
                        "rms_3d_mm"   : round(fit['rms_3d_mm'],   2),
                        "mean_err_mm" : round(fit['mean_err_mm'],  2),
                        "max_err_mm"  : round(fit['max_err_mm'],   2),
                        "res_x_mm"    : round(fit['residuals_mm'][0], 2),
                        "res_y_mm"    : round(fit['residuals_mm'][1], 2),
                        "res_z_mm"    : round(fit['residuals_mm'][2], 2),
                        "speed_mps"   : round(fit['speed'], 3),
                        "speed_kmh"   : round(fit['speed'] * 3.6, 2),
                    }
                    if fit['t_apex']:
                        ap = fit['predict'](fit['t_apex'])
                        accuracy['apex_height_m'] = round(float(ap[GRAVITY_AXIS]), 3)
                        accuracy['apex_time_s']   = round(fit['t_apex'], 3)
                    if t_land:
                        lp   = fit['predict'](t_land)
                        dist = float(np.sqrt((lp[0]-fit['x0'])**2 + (lp[2]-fit['z0'])**2))
                        accuracy['landing_pos'] = (round(float(lp[0]), 3),
                                                   round(float(lp[1]), 3),
                                                   round(float(lp[2]), 3))
                        accuracy['range_m'] = round(dist, 3)

                    with state.lock:
                        state.fit_result    = fit
                        state.timing        = timing
                        state.accuracy      = accuracy
                        state.fit_done_time = fit_end
                        state.mode          = Mode.DONE

                    _print_report(timing, accuracy)
                else:
                    print(f"[Processor] Not enough points ({n}/{MIN_POINTS}). Disarming.")
                    with state.lock:
                        state.mode = Mode.IDLE

def _print_report(timing: dict, accuracy: dict):
    print(f"\n{'='*56}")
    print("  TRAJECTORY REPORT")
    print(f"{'?'*56}")
    print(f"  {'Arm to throw lag':<28}: {timing['arm_to_throw_s']} s")
    print(f"  {'Throw to fit complete':<28}: {timing['throw_to_fit_s']} s")
    print(f"  {'Arm to fit complete':<28}: {timing['arm_to_fit_s']} s")
    print(f"  {'Flight duration':<28}: {timing['flight_duration_s']} s")
    print(f"  {'Points captured':<28}: {timing['n_points']}  @  {timing['sample_rate_hz']} Hz")
    print(f"{'?'*56}")
    print(f"  {'Speed':<28}: {accuracy['speed_mps']} m/s  ({accuracy['speed_kmh']} km/h)")
    print(f"  {'3D RMS error':<28}: {accuracy['rms_3d_mm']} mm")
    print(f"  {'Mean point error':<28}: {accuracy['mean_err_mm']} mm")
    print(f"  {'Max point error':<28}: {accuracy['max_err_mm']} mm")
    print(f"  {'Residuals X / Y / Z':<28}: "
          f"{accuracy['res_x_mm']} / {accuracy['res_y_mm']} / {accuracy['res_z_mm']} mm")
    if 'apex_height_m' in accuracy:
        print(f"  {'Apex height':<28}: {accuracy['apex_height_m']} m  "
              f"at t = {accuracy['apex_time_s']} s")
    if 'range_m' in accuracy:
        lp = accuracy['landing_pos']
        print(f"  {'Predicted landing':<28}: ({lp[0]}, {lp[1]}, {lp[2]}) m")
        print(f"  {'Horizontal range':<28}: {accuracy['range_m']} m")
    print(f"{'='*56}\n")

# ?? LIVE PLOT ?????????????????????????????????????????????????????????????????
MODE_COLORS = {
    Mode.IDLE:      ("#888888", "IDLE  --  Press P to arm"),
    Mode.ARMED:     ("#ffdd00", "ARMED  --  Waiting for throw..."),
    Mode.RECORDING: ("#ff4400", "RECORDING"),
    Mode.FITTING:   ("#00aaff", "Fitting..."),
    Mode.DONE:      ("#44ff88", "Done  --  Press R to reset  |  P to re-arm"),
}

def start_live_plot():
    fig = plt.figure(figsize=(16, 8), facecolor='#0a0a0a')
    fig.canvas.manager.set_window_title("OptiTrack Ball Tracker")

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.05, right=0.97, top=0.90, bottom=0.08,
                           wspace=0.38, hspace=0.48)

    ax3d = fig.add_subplot(gs[:, 0], projection='3d', facecolor='#0a0a0a')
    ax_h = fig.add_subplot(gs[0, 1], facecolor='#0a0a0a')
    ax_e = fig.add_subplot(gs[1, 1], facecolor='#0a0a0a')
    ax_r = fig.add_subplot(gs[:, 2], facecolor='#0a0a0a')

    # Static 3D styling
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor('#2a2a2a')
    ax3d.tick_params(colors='#666', labelsize=7)
    ax3d.set_xlabel('X (m)', color='#888', fontsize=8)
    ax3d.set_ylabel('Z (m)', color='#888', fontsize=8)
    ax3d.set_zlabel('Y/Height (m)', color='#888', fontsize=8)
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

    _style(ax_h, 'Height vs Time',       'Time (s)',     'Height (m)')
    _style(ax_e, 'Per-Point 3-D Error',  'Point index',  'Error (mm)')
    ax_h.set_xlim(0, 3); ax_h.set_ylim(*Y_LIM)

    ax_r.axis('off')
    ax_r.set_title('Analysis Report', color='#ccc', fontsize=9, pad=4)

    # Persistent 3D artists
    trail_sc  = ax3d.scatter([], [], [], s=16, alpha=0.55, cmap='cool',   label='Trail')
    cur_sc    = ax3d.scatter([], [], [], c='#ff3333', s=90, zorder=6,     label='Current')
    flight_sc = ax3d.scatter([], [], [], c='#ff8800', s=40, zorder=5,     label='Captured')
    fit_ln,   = ax3d.plot([], [], [], color='#00aaff', lw=2,              label='Fit')
    apex_sc   = ax3d.scatter([], [], [], c='gold',     s=160, marker='^', zorder=7, label='Apex')
    land_sc   = ax3d.scatter([], [], [], c='#44ff44',  s=160, marker='*', zorder=7, label='Landing')
    ax3d.legend(fontsize=7, loc='upper left',
                facecolor='#111', edgecolor='#333', labelcolor='#bbb')

    # Persistent height plot artists
    obs_ln, = ax_h.plot([], [], 'o', color='#ff8800', ms=4,  label='Observed')
    fit_h,  = ax_h.plot([], [], color='#00aaff', lw=2,       label='Fit')
    apex_vl = ax_h.axvline(0, color='gold',     lw=1, ls='--', alpha=0, label='Apex')
    land_vl = ax_h.axvline(0, color='#44ff44',  lw=1, ls='--', alpha=0, label='Landing')
    cur_hl  = ax_h.axhline(0, color='#ff3333',  lw=0.8, ls=':', alpha=0)
    ax_h.legend(fontsize=7, facecolor='#111', edgecolor='#333', labelcolor='#bbb')

    report_txt = ax_r.text(0.04, 0.97, '',
                           transform=ax_r.transAxes,
                           color='#cccccc', fontsize=8.5,
                           verticalalignment='top', fontfamily='monospace')

    status_bar = fig.text(0.5, 0.95, '', ha='center', va='top',
                          fontsize=12, fontweight='bold')

    # ?? Keyboard handler ??????????????????????????????????????????????????????
    def on_key(event):
        key = (event.key or '').lower()

        if key == 'p':
            with state.lock:
                m = state.mode
            if m in (Mode.IDLE, Mode.DONE):
                with state.lock:
                    state.mode         = Mode.ARMED
                    state.arm_time     = time.time()
                    state.throw_time   = None
                    state.fit_done_time = None
                print("[Key] ARMED -- throw the ball!")
            elif m == Mode.ARMED:
                with state.lock:
                    state.mode = Mode.IDLE
                print("[Key] Disarmed.")

        elif key == 'r':
            with state.lock:
                state.mode          = Mode.IDLE
                state.fit_result    = None
                state.timing        = None
                state.accuracy      = None
                state.flight_times  = []
                state.flight_points = []
                state.arm_time      = None
                state.throw_time    = None
                state.fit_done_time = None
            print("[Key] Reset.")

        elif key == 'q':
            plt.close('all')

    fig.canvas.mpl_connect('key_press_event', on_key)

    # ?? Animation update ??????????????????????????????????????????????????????
    def update(_frame):
        with state.lock:
            buf      = list(state.buffer)
            mode     = state.mode
            fit      = state.fit_result
            f_pts    = list(state.flight_points)
            f_ts     = list(state.flight_times)
            timing   = state.timing
            accuracy = state.accuracy
            rb       = state.rb_name
            conn     = state.connected
            hw_fps   = state.hw_fps

        # Status bar
        color, label = MODE_COLORS[mode]
        fps_str = f"  |  {hw_fps:.1f} Hz" if hw_fps > 0 else ""
        rb_str  = f"  |  RB: '{rb}'{fps_str}" if conn else "  -- waiting for stream..."
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
            trail_sc._offsets3d = ([], [], [])
            cur_sc._offsets3d   = ([], [], [])

        # Captured flight points
        if f_pts:
            fp = np.array(f_pts); ft = np.array(f_ts) - f_ts[0]
            flight_sc._offsets3d = (fp[:, 0], fp[:, 2], fp[:, 1])
            obs_ln.set_data(ft, fp[:, GRAVITY_AXIS])
        else:
            flight_sc._offsets3d = ([], [], [])
            obs_ln.set_data([], [])

        # Fitted trajectory
        if fit:
            t_end   = max((fit['t_apex'] or 0) * 2.2, 1.5)
            t_range = np.linspace(0, t_end, 300)
            pred    = fit['predict'](t_range)
            fit_ln.set_data_3d(pred[:, 0], pred[:, 2], pred[:, 1])
            fit_h.set_data(t_range, pred[:, GRAVITY_AXIS])

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

            # Per-point error bars (ax_e is cla'd each time -- it's fast)
            ax_e.cla()
            _style(ax_e, 'Per-Point 3-D Error', 'Point index', 'Error (mm)')
            errs = fit['per_pt_err']
            norm = errs / max(errs.max(), 1e-9)
            ax_e.bar(range(len(errs)), errs,
                     color=plt.cm.RdYlGn_r(norm), alpha=0.85, width=0.8)
            ax_e.axhline(fit['mean_err_mm'], color='#ffdd00', lw=1.2, ls='--',
                         label=f"mean {fit['mean_err_mm']:.1f} mm")
            ax_e.axhline(fit['rms_3d_mm'],   color='#00aaff', lw=1.0, ls=':',
                         label=f"RMS  {fit['rms_3d_mm']:.1f} mm")
            ax_e.legend(fontsize=7, facecolor='#111', edgecolor='#333', labelcolor='#bbb')
        else:
            fit_ln.set_data_3d([], [], [])
            fit_h.set_data([], [])
            apex_sc._offsets3d = ([], [], []); apex_vl.set_alpha(0)
            land_sc._offsets3d = ([], [], []); land_vl.set_alpha(0)

        # Text report panel
        if timing and accuracy:
            lines = [
                "TIMING",
                f"  Arm -> throw lag  : {timing['arm_to_throw_s']} s",
                f"  Throw -> fit done : {timing['throw_to_fit_s']} s",
                f"  Total (arm->fit)  : {timing['arm_to_fit_s']} s",
                f"  Flight duration   : {timing['flight_duration_s']} s",
                f"  Points captured   : {timing['n_points']}",
                f"  Sample rate (fit) : {timing['sample_rate_hz']} Hz",
                f"  Motive stream Hz  : {timing.get('hw_fps', '?')} Hz",
                "",
                "ACCURACY",
                f"  3D RMS error      : {accuracy['rms_3d_mm']} mm",
                f"  Mean point error  : {accuracy['mean_err_mm']} mm",
                f"  Max point error   : {accuracy['max_err_mm']} mm",
                f"  Residual X        : {accuracy['res_x_mm']} mm",
                f"  Residual Y        : {accuracy['res_y_mm']} mm",
                f"  Residual Z        : {accuracy['res_z_mm']} mm",
                "",
                "KINEMATICS",
                f"  Launch speed      : {accuracy['speed_mps']} m/s",
                f"                    : {accuracy['speed_kmh']} km/h",
            ]
            if 'apex_height_m' in accuracy:
                lines += [
                    f"  Apex height       : {accuracy['apex_height_m']} m",
                    f"  Apex time         : {accuracy['apex_time_s']} s",
                ]
            if 'range_m' in accuracy:
                lp = accuracy['landing_pos']
                lines += [
                    f"  Landing (x,y,z)   :",
                    f"   ({lp[0]}, {lp[1]}, {lp[2]}) m",
                    f"  Horizontal range  : {accuracy['range_m']} m",
                ]
            report_txt.set_text('\n'.join(lines))
        elif mode == Mode.ARMED:
            report_txt.set_text("ARMED\n\nWaiting for throw...\n\n"
                                "Throw the ball to\nstart recording.")
        elif mode == Mode.RECORDING:
            report_txt.set_text(f"RECORDING\n\n{len(f_pts)} points\ncaptured so far...")
        elif mode == Mode.FITTING:
            report_txt.set_text("Fitting trajectory...")
        else:
            report_txt.set_text("Controls\n\n"
                                "P  - arm recording\n"
                                "R  - reset\n"
                                "Q  - quit\n\n"
                                "Press P to arm,\nthen throw the ball.")

    ani = animation.FuncAnimation(fig, update, interval=PLOT_INTERVAL_MS,
                                  blit=False, cache_frame_data=False)
    plt.show()

# ?? MAIN ??????????????????????????????????????????????????????????????????????
def main():
    print("=" * 56)
    print("  OptiTrack Rigid Body Ball Tracker")
    print("=" * 56)
    print(f"  Motive server : {SERVER_IP}")
    print(f"  Data port     : {DATA_PORT}   multicast: {MULTICAST_ADDR}")
    print()
    print("  P  ->  Arm / disarm recording")
    print("  R  ->  Reset last trajectory")
    print("  Q  ->  Quit")
    print()

    threading.Thread(target=udp_listener,    daemon=True).start()
    time.sleep(0.8)
    threading.Thread(target=processing_loop, daemon=True).start()

    print("[Viz] Opening live plot...\n")
    start_live_plot()

if __name__ == "__main__":
    main()