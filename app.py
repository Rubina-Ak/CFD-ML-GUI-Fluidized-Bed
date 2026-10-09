"""
Fluidized Bed Reactor - ANN Surrogate Prediction Tool  (v2)
===========================================================
Loads a trained ANN + scalers and lets a user explore predicted reactor
behaviour (1D profiles and 2D axial-radial contours) without re-running CFD.

Run:  python app.py

Required folder layout:
    model/final_tuned_ann_3output.h5
    model/scaler_X.pkl
    model/scaler_y.pkl

What changed in v2 (see the review notes for the reasoning):
  * Air VOF (and any other bounded output) is clipped to its physical limits
    before plotting / saving; the raw ANN value is kept in the CSV (*_raw).
  * All horizontal axes are normalised to r/R = -1 ... +1 (R = column radius).
  * ONE colormap and ONE colour scale rule for velocity, VOF and pressure.
  * New "Nozzle line" sweep / plane built from the supplied nozzle coordinates.
  * Sweeps stay inside the column wall (no more predictions in empty space).
  * Figures use matplotlib.figure.Figure (no pyplot leak) + constrained layout
    + zoom/pan toolbar; the plot re-scales correctly when the window resizes.
  * Only the NON-swept inputs are validated / range-checked.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import numpy as np
import pandas as pd
import joblib

# ============================================================================
# 1. CENTRAL CONFIGURATION
#    Everything reactor/model-specific lives here. To reuse this GUI for a
#    different geometry or a different trained model, edit ONLY this block.
# ============================================================================

MODEL_PATH    = "model/final_tuned_ann_3output.h5"
SCALER_X_PATH = "model/scaler_X.pkl"
SCALER_Y_PATH = "model/scaler_y.pkl"

# --- Inputs: internal_name -> metadata ---
# 'scaler_col' MUST match the exact column name the scaler was fit on.
# 'range'      is the TRUE min/max from the training dataset.
# 'default'    is what the entry box is pre-filled with.
INPUT_VARIABLES = {
    "u_umf": {"label": "U / Umf",              "short": "U/Umf", "unit": "-",   "scaler_col": "U/Umf",
              "range": (2.0, 3.0),      "default": 2.5},
    "q":     {"label": "Side Injection (Q)",   "short": "Q",     "unit": "LPM", "scaler_col": "Q",
              "range": (20.0, 100.0),   "default": 60.0},
    "x":     {"label": "X coordinate",         "short": "x",     "unit": "m",   "scaler_col": "x-coordinate",
              "range": (-0.105, 0.137), "default": 0.0},
    "y":     {"label": "Y coordinate",         "short": "y",     "unit": "m",   "scaler_col": "y-coordinate",
              "range": (-0.209, 0.105), "default": 0.0},
    "z":     {"label": "Z coordinate (axial)", "short": "z",     "unit": "m",   "scaler_col": "z-coordinate",
              "range": (0.00, 1.96),    "default": 0.5},
    "an":    {"label": "Active Nozzle (AN)",   "short": "AN",    "unit": "-",   "scaler_col": "AN",
              "range": (1.0, 4.0),      "default": 1, "discrete": [1, 2, 3, 4]},
}
OPERATING_KEYS = ("u_umf", "q", "an")
POSITION_KEYS  = ("x", "y", "z")

# --- Outputs: internal_name -> metadata ---
# 'phys_bounds' : known physical validity range. Predictions outside are CLIPPED
#                 (the raw ANN value is still written to the CSV as <name>_raw).
# 'clim'        : fixed colour-scale limits for 2D contours (None = use data range).
# 'zero_line'   : draw a dashed zero reference line on 1D profiles.
OUTPUT_VARIABLES = {
    "z_velocity": {
        "label": "Solid mean z-velocity", "unit": "m/s", "index": 0,
        "phys_bounds": None, "clim": None, "zero_line": True,
    },
    "air_vof": {
        "label": "Air mean VOF", "unit": "-", "index": 1,
        "phys_bounds": (0.371, 1.0), "clim": (0.371, 1.0), "zero_line": False,
        "nozzle_ylim": (0.3, 0.8),    # 1D nozzle-line profile: y-axis window, widened only if data exceed it
    },
    "pressure": {
        "label": "Mean pressure", "unit": "Pa", "index": 2,
        "phys_bounds": None, "clim": None, "zero_line": False,
    },
}
OUTPUT_LABELS = {m["label"]: k for k, m in OUTPUT_VARIABLES.items()}
N_OUTPUTS = len(OUTPUT_VARIABLES)

# --- Nozzle line (points extracted along the nozzle plane) ---
NOZZLE_X = np.array([
     0.0414,  0.0405,  0.0349,  0.0330,  0.0324,  0.0318,
     0.0278,  0.0266,  0.0262,  0.0251,  0.0225,  0.0204,
     0.0173,  0.0170,  0.0166,  0.0163,  0.0129,  0.0129,
     0.00986, 0.00907, 0.00872, 0.00366, 0.00162,
    -0.00102,-0.00193,-0.00529,-0.00874,-0.0115,
    -0.0115,-0.0161,-0.0182,-0.0210,-0.0261,
    -0.0308,-0.0311,-0.0320,-0.0345,-0.0353,
    -0.0396,-0.0405,-0.0413,
])
NOZZLE_Y = np.array([
    -0.0680,-0.0665,-0.0573,-0.0543,-0.0533,-0.0522,
    -0.0456,-0.0438,-0.0432,-0.0413,-0.0370,-0.0336,
    -0.0284,-0.0279,-0.0273,-0.0268,-0.0213,-0.0212,
    -0.0162,-0.0149,-0.0143,-0.00602,-0.00267,
     0.00168, 0.00318, 0.00869, 0.0144, 0.0189,
     0.0190, 0.0265, 0.0299, 0.0345, 0.0429,
     0.0507, 0.0511, 0.0526, 0.0568, 0.0580,
     0.0651, 0.0666, 0.0679,
])


def _fit_nozzle_line(xs, ys):
    """Straight-line fit between the two end points of the nozzle line.
    Returns centre, unit direction, half-length, normalised position s of every
    supplied point (-1 ... +1) and the largest perpendicular deviation (m)."""
    pts = np.column_stack([xs, ys])
    p0, p1 = pts[0], pts[-1]
    centre = 0.5 * (p0 + p1)
    vec = p1 - p0
    length = float(np.hypot(*vec))
    direction = vec / length
    half = 0.5 * length
    rel = pts - centre
    s = (rel @ direction) / half
    off = np.abs(rel @ np.array([-direction[1], direction[0]])).max()
    return centre, direction, half, s, float(off)


NOZZLE_CENTRE, NOZZLE_DIR, NOZZLE_HALF_LEN, NOZZLE_S, NOZZLE_MAX_OFFSET = _fit_nozzle_line(NOZZLE_X, NOZZLE_Y)

# Column geometry used for the "-1 ... +1" normalisation and wall masking.
# The nozzle line runs wall-to-wall through the column axis, so its half-length is
# the column radius R.  >>> If you know the true wall radius, set it here. <<<
REACTOR_RADIUS = NOZZLE_HALF_LEN
REACTOR_CX, REACTOR_CY = float(NOZZLE_CENTRE[0]), float(NOZZLE_CENTRE[1])
MASK_OUTSIDE_WALL = True      # blank out contour points that lie outside the column

Z_RANGE = INPUT_VARIABLES["z"]["range"]

# --- Axis choices offered in the GUI:  label -> key ---
PROFILE_AXES = {
    "Nozzle line":                     "s",
    "Axial  (z)":                      "z",
    "Horizontal  x / R":               "x",
    "Horizontal  y / R":               "y",
}
CONTOUR_AXES = {
    "Nozzle line":                     "s",
    "x / R  (y fixed)":                "x",
    "y / R  (x fixed)":                "y",
}
# which INPUT variables are swept (i.e. not typed in) for each axis key
SWEPT = {"s": {"x", "y"}, "x": {"x"}, "y": {"y"}, "z": {"z"}}

# --- Resolution / plotting behaviour ---
N_PROFILE_POINTS  = 200
N_CONTOUR_R       = 81
N_CONTOUR_Z       = 121
N_CONTOUR_LEVELS  = 30
CONTOUR_CMAP      = "turbo"     # ONE colormap for every output ("jet" for a Fluent look)
PROFILE_Z_VERTICAL = False      # False: axial (z) on x-axis, variable on y-axis

# --- Visual theme ---
COLOR_BG        = "#F7F8FA"
COLOR_PANEL     = "#FFFFFF"
COLOR_ACCENT    = "#1B4F72"
COLOR_ACCENT_2  = "#2E86C1"
COLOR_TEXT      = "#1C2833"
COLOR_WARNING   = "#B03A2E"
COLOR_BORDER    = "#D5D8DC"

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#444444",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "lines.linewidth": 2.0,
    "axes.formatter.useoffset": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# ============================================================================
# 2. GEOMETRY  (axis construction, no GUI / plotting code)
# ============================================================================
def build_axis(key, n):
    """Return (display_values, coords, label) for a sweep axis.

    display_values : what is drawn on the plot axis (normalised for s, x, y)
    coords         : dict of physical model inputs that vary along the axis
    """
    R = REACTOR_RADIUS
    if key == "s":                                   # along the nozzle line
        s = np.linspace(-1.0, 1.0, n)
        coords = {"x": REACTOR_CX + s * NOZZLE_HALF_LEN * NOZZLE_DIR[0],
                  "y": REACTOR_CY + s * NOZZLE_HALF_LEN * NOZZLE_DIR[1]}
        return s, coords, "Nozzle-line position  s = r / R"
    if key in ("x", "y"):
        c = np.linspace(-1.0, 1.0, n)
        centre = REACTOR_CX if key == "x" else REACTOR_CY
        return c, {key: centre + c * R}, f"{key} / R   (R = {R * 1000:.1f} mm)"
    z = np.linspace(Z_RANGE[0], Z_RANGE[1], n)
    return z, {"z": z}, "Z coordinate \u2014 axial (m)"


# ============================================================================
# 3. PREDICTION LAYER  (no GUI code, no plotting code)
# ============================================================================
class ANNPredictor:
    def __init__(self):
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.status = "Not loaded"

    def load(self):
        try:
            import tensorflow as tf           # imported here so the window opens quickly
            self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            self.scaler_X = joblib.load(SCALER_X_PATH)
            self.scaler_y = joblib.load(SCALER_Y_PATH)
            n_out = int(self.model.output_shape[-1])
            if n_out != N_OUTPUTS:
                raise ValueError(f"model has {n_out} outputs, GUI expects {N_OUTPUTS}")
            self.status = "Loaded"
        except FileNotFoundError as e:
            self.status = f"Error: file not found ({e.filename})"
        except Exception as e:
            self.status = f"Error: {e}"

    def _feature_order(self):
        names = getattr(self.scaler_X, "feature_names_in_", None)
        if names is not None:
            return list(names)
        return [m["scaler_col"] for m in INPUT_VARIABLES.values()]

    def predict(self, inputs):
        """inputs: internal_name -> scalar or 1-D array (broadcast to a common length).
        Returns an (n, N_OUTPUTS) array in physical units."""
        n = max(np.size(v) for v in inputs.values())
        data = {}
        for key, meta in INPUT_VARIABLES.items():
            data[meta["scaler_col"]] = np.array(
                np.broadcast_to(np.asarray(inputs[key], dtype=float), (n,)))
        df = pd.DataFrame(data)
        order = self._feature_order()
        missing = [c for c in order if c not in df.columns]
        if missing:
            raise KeyError(f"Scaler expects column(s) not provided by the GUI: {missing}")
        df = df[order]
        has_names = getattr(self.scaler_X, "feature_names_in_", None) is not None
        scaled = self.scaler_X.transform(df if has_names else df.to_numpy())
        pred_scaled = self.model.predict(scaled, batch_size=4096, verbose=0)
        return self.scaler_y.inverse_transform(pred_scaled)


def apply_physical_limits(pred):
    """Clip every output that has 'phys_bounds'. NaNs (masked points) pass through.
    Returns (clipped_copy, {output_key: n_points_that_were_clipped})."""
    out = pred.copy()
    counts = {}
    for key, meta in OUTPUT_VARIABLES.items():
        bounds = meta["phys_bounds"]
        if bounds is None:
            continue
        lo, hi = bounds
        i = meta["index"]
        col = pred[:, i]
        counts[key] = int(np.count_nonzero((col < lo) | (col > hi)))
        out[:, i] = np.clip(col, lo, hi)
    return out, counts


# ============================================================================
# 4. VALIDATION HELPERS
# ============================================================================
def validate_number(value_str, var_name):
    if value_str.strip() == "":
        return None, f"Please enter a value for {var_name}."
    try:
        return float(value_str), None
    except ValueError:
        return None, f"Please enter a numerical value for {var_name}."


def range_warning(value, var_key):
    lo, hi = INPUT_VARIABLES[var_key]["range"]
    if value < lo or value > hi:
        return (f"'{INPUT_VARIABLES[var_key]['label']}' = {value:g} is outside the "
                f"training range [{lo}, {hi}] \u2014 extrapolated values may be unreliable.")
    return None


def collect_input_warnings(values, swept):
    """Range checks on the typed (non-swept) inputs + a column-wall check."""
    msgs = [w for w in (range_warning(v, k) for k, v in values.items()) if w]
    fixed_xy = [k for k in ("x", "y") if k not in swept]
    if len(fixed_xy) == 2:
        r = np.hypot(values["x"] - REACTOR_CX, values["y"] - REACTOR_CY)
        if r > REACTOR_RADIUS:
            msgs.append(f"Point (x, y) lies {r * 1000:.1f} mm from the axis, outside the column "
                        f"radius R = {REACTOR_RADIUS * 1000:.1f} mm.")
    elif len(fixed_xy) == 1:
        k = fixed_xy[0]
        c = REACTOR_CX if k == "x" else REACTOR_CY
        if abs(values[k] - c) > REACTOR_RADIUS:
            msgs.append(f"Fixed {k} = {values[k]:g} m lies outside the column "
                        f"(|{k} - {c:.3f}| > R = {REACTOR_RADIUS:.4f} m).")
    return msgs


def clip_messages(counts, n_valid):
    msgs = []
    for key, c in counts.items():
        if c:
            lo, hi = OUTPUT_VARIABLES[key]["phys_bounds"]
            msgs.append(f"{OUTPUT_VARIABLES[key]['label']}: {c} of {n_valid} predicted points were outside "
                        f"the physical range [{lo}, {hi}] and were clipped (raw values kept in CSV).")
    return msgs


def condition_text(values, swept):
    parts = []
    for k, meta in INPUT_VARIABLES.items():
        if k in swept or k not in values:
            continue
        unit = "" if meta["unit"] == "-" else f" {meta['unit']}"
        parts.append(f"{meta['short']} = {values[k]:g}{unit}")
    return "   \u00b7   ".join(parts)


def make_table(values, inputs, pred, raw, extra):
    """Flat DataFrame for CSV export: coordinates, operating conditions, outputs."""
    n = pred.shape[0]
    cols = dict(extra)
    for k in POSITION_KEYS:
        cols[k] = np.array(np.broadcast_to(np.asarray(inputs[k], float), (n,)))
    for k in OPERATING_KEYS:
        cols[k] = np.full(n, values[k])
    for name, meta in OUTPUT_VARIABLES.items():
        cols[name] = pred[:, meta["index"]]
        if meta["phys_bounds"] is not None:
            cols[name + "_raw"] = raw[:, meta["index"]]
    return pd.DataFrame(cols)


# ============================================================================
# 5. PLOTTING LAYER  (pure functions -> matplotlib Figure; no pyplot)
# ============================================================================
def _new_figure(size):
    fig = Figure(figsize=size, dpi=100, constrained_layout=True)
    return fig, fig.add_subplot(111)


def _safe_limits(lo, hi):
    """Avoid degenerate (zero-width) colour / axis ranges for flat fields."""
    lo, hi = float(lo), float(hi)
    if hi - lo < 1e-9 * max(1.0, abs(hi), abs(lo)):
        pad = max(abs(hi) * 0.01, 1e-6)
        return lo - pad, hi + pad
    return lo, hi


def _value_limits(meta, values, axis_key=None):
    b = meta["phys_bounds"]
    if b is not None:                                  # bounded quantity: fixed physical window
        pad = 0.02 * (b[1] - b[0])
        window = (b[0] - pad, b[1] + pad)
        pref = meta.get("nozzle_ylim")
        if axis_key == "s" and pref is not None:       # preferred tight window on the nozzle line
            v = values[np.isfinite(values)]
            if v.size:
                lo, hi = pref
                if v.min() < lo or v.max() > hi:       # data exceed it -> widen just enough
                    m = 0.03 * (v.max() - v.min() + 1e-9)
                    lo, hi = min(lo, v.min() - m), max(hi, v.max() + m)
                return lo, min(hi, window[1])
        return window
    vals = values[np.isfinite(values)]
    lo, hi = _safe_limits(vals.min(), vals.max())
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def plot_profile(disp, pred, raw, out_key, pos_label, axis_key, subtitle):
    meta = OUTPUT_VARIABLES[out_key]
    idx = meta["index"]
    y = pred[:, idx]
    vertical = PROFILE_Z_VERTICAL and axis_key == "z"

    def xy(pos, val):                                  # swap axes for vertical z-profiles
        return (val, pos) if vertical else (pos, val)

    fig, ax = _new_figure((6.8, 5.6))
    ax.plot(*xy(disp, y), color=COLOR_ACCENT)


    vlim = _value_limits(meta, y, axis_key)
    val_label = f"{meta['label']} ({meta['unit']})"
    if vertical:
        ax.set_xlim(*vlim); ax.set_ylim(*Z_RANGE)
        ax.set_xlabel(val_label); ax.set_ylabel(pos_label)
    else:
        ax.set_ylim(*vlim)
        ax.set_xlabel(pos_label); ax.set_ylabel(val_label)
        if axis_key in ("s", "x", "y"):
            ax.set_xlim(-1, 1)
            ax.set_xticks(np.linspace(-1, 1, 9))
        else:
            ax.set_xlim(*Z_RANGE)

    if meta["zero_line"] and vlim[0] < 0 < vlim[1]:
        (ax.axvline if vertical else ax.axhline)(0, color="#999999", lw=0.8, ls="--")
    if meta["phys_bounds"]:
        for b in meta["phys_bounds"]:
            (ax.axvline if vertical else ax.axhline)(b, color=COLOR_ACCENT_2, lw=0.8, ls=":", alpha=0.8)

    ax.set_title(f"{meta['label']} \u2014 predicted profile\n{subtitle}", fontsize=10.5,
                 color=COLOR_TEXT, loc="left")
    return fig


def plot_contour(h_disp, z_vals, grid_pred, out_key, h_label, subtitle):
    meta = OUTPUT_VARIABLES[out_key]
    idx = meta["index"]
    Z = np.ma.masked_invalid(grid_pred[:, :, idx])

    lo, hi = meta["clim"] if meta["clim"] else (Z.min(), Z.max())
    lo, hi = _safe_limits(lo, hi)
    eps = 1e-6 * (hi - lo)                             # keep values sitting exactly on a limit filled
    levels = np.linspace(lo - eps, hi + eps, N_CONTOUR_LEVELS)

    fig, ax = _new_figure((6.8, 6.0))
    cf = ax.contourf(h_disp, z_vals, Z, levels=levels, cmap=CONTOUR_CMAP)   # same cmap for ALL outputs

    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.locator = MaxNLocator(nbins=7)                # "nice" tick values
    cbar.update_ticks()
    cbar.set_label(f"{meta['label']} ({meta['unit']})")

    ax.set_xlim(-1, 1)
    ax.set_ylim(*Z_RANGE)
    ax.set_xticks(np.linspace(-1, 1, 5))
    for wall in (-1, 1):                               # column walls
        ax.axvline(wall, color="k", lw=1.6)
    ax.set_xlabel(h_label)
    ax.set_ylabel("Z coordinate \u2014 axial (m)")
    ax.set_title(f"{meta['label']} \u2014 axial-radial field\n{subtitle}", fontsize=10.5,
                 color=COLOR_TEXT, loc="left")
    ax.grid(False)
    return fig


# ============================================================================
# 6. GUI LAYER
# ============================================================================
class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fluidized Bed Reactor \u2014 ANN Prediction Tool")
        self.geometry("1300x820")
        self.minsize(1050, 700)
        self.configure(bg=COLOR_BG)

        self._setup_style()

        self.predictor = ANNPredictor()
        self.predictor.load()

        self.entries = {}
        self.plot_container = None
        self.canvas = None
        self.current_fig = None
        self.last_table = None

        self._build_layout()

    # ---------------- ttk styling ----------------
    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=COLOR_PANEL)
        style.configure("Header.TFrame", background=COLOR_ACCENT)
        style.configure("TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=COLOR_ACCENT, foreground="white",
                        font=("Segoe UI Semibold", 14))
        style.configure("Section.TLabel", background=COLOR_PANEL, foreground=COLOR_ACCENT,
                        font=("Segoe UI Semibold", 11))
        style.configure("Status.TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 10), padding=5)
        style.configure("Accent.TButton", font=("Segoe UI Semibold", 10), padding=7)
        style.map("Accent.TButton",
                  background=[("!disabled", COLOR_ACCENT)],
                  foreground=[("!disabled", "white")])
        style.configure("TEntry", padding=3)
        style.configure("TCombobox", padding=3)
        style.configure("TRadiobutton", background=COLOR_PANEL, font=("Segoe UI", 10))

    # ---------------- layout ----------------
    def _build_layout(self):
        header = ttk.Frame(self, style="Header.TFrame")
        header.pack(side="top", fill="x")
        ttk.Label(header, text="Fluidized Bed Reactor \u2014 ANN Surrogate Explorer",
                  style="Header.TLabel").pack(side="left", padx=16, pady=10)

        model_ok = self.predictor.status == "Loaded"
        status_text = "\u25cf Model loaded" if model_ok else f"\u25cf {self.predictor.status}"
        status_color = "#A9DFBF" if model_ok else "#F5B7B1"
        tk.Label(header, text=status_text, bg=COLOR_ACCENT, fg=status_color,
                 font=("Segoe UI", 10, "bold")).pack(side="right", padx=16)

        # status bar first, so it is never pushed off-screen
        self.bottom_status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.bottom_status, style="Status.TLabel", anchor="w").pack(
            side="bottom", fill="x", padx=14, pady=4)

        body = tk.Frame(self, bg=COLOR_BG)
        body.pack(fill="both", expand=True, padx=14, pady=(12, 4))

        left = tk.Frame(body, bg=COLOR_PANEL, highlightbackground=COLOR_BORDER, highlightthickness=1)
        left.pack(side="left", fill="y", padx=(0, 14))

        right = tk.Frame(body, bg=COLOR_PANEL, highlightbackground=COLOR_BORDER, highlightthickness=1)
        right.pack(side="left", fill="both", expand=True)

        self._build_input_panel(left)

        # persistent, non-modal warning area under the plot
        self.warn_var = tk.StringVar(value="")
        self.warn_label = tk.Label(right, textvariable=self.warn_var, bg=COLOR_PANEL, fg=COLOR_WARNING,
                                   font=("Segoe UI", 9), justify="left", anchor="w", wraplength=820)
        self.warn_label.pack(side="bottom", fill="x", padx=12, pady=(0, 8))
        self.plot_area = tk.Frame(right, bg=COLOR_PANEL)
        self.plot_area.pack(side="top", fill="both", expand=True)
        self._show_placeholder()

    def _show_placeholder(self):
        self.plot_container = tk.Frame(self.plot_area, bg=COLOR_PANEL)
        self.plot_container.pack(fill="both", expand=True)
        tk.Label(self.plot_container, text="Set the inputs and press  RUN PREDICTION",
                 bg=COLOR_PANEL, fg="#7B7D7D", font=("Segoe UI", 12)).pack(expand=True)

    def _add_input_row(self, parent, name):
        meta = INPUT_VARIABLES[name]
        row = tk.Frame(parent, bg=COLOR_PANEL)
        row.pack(fill="x", padx=16, pady=2)
        ttk.Label(row, text=meta["label"], width=20).pack(side="left")
        if "discrete" in meta:
            entry = ttk.Combobox(row, values=[str(v) for v in meta["discrete"]], width=8, state="readonly")
            entry.set(str(meta["default"]))
        else:
            entry = ttk.Entry(row, width=10)
            entry.insert(0, str(meta["default"]))
        entry.pack(side="left")
        ttk.Label(row, text=meta["unit"], foreground="#7B7D7D").pack(side="left", padx=(4, 0))
        self.entries[name] = entry

    def _build_input_panel(self, parent):
        pad = {"padx": 16, "pady": (6, 2)}

        ttk.Label(parent, text="OPERATING CONDITIONS", style="Section.TLabel").pack(anchor="w", **pad)
        for name in OPERATING_KEYS:
            self._add_input_row(parent, name)

        ttk.Label(parent, text="POSITION  (fixed coordinates)", style="Section.TLabel").pack(anchor="w", **pad)
        ttk.Label(parent, text="Greyed-out coordinates are swept by the chosen view.",
                  font=("Segoe UI", 8, "italic"), background=COLOR_PANEL, foreground="#7B7D7D").pack(
            anchor="w", padx=16, pady=(0, 2))
        for name in POSITION_KEYS:
            self._add_input_row(parent, name)

        ttk.Separator(parent).pack(fill="x", padx=16, pady=8)

        ttk.Label(parent, text="VIEW", style="Section.TLabel").pack(anchor="w", padx=16, pady=(0, 2))
        self.plot_mode = tk.StringVar(value="profile")
        ttk.Radiobutton(parent, text="1D profile", variable=self.plot_mode, value="profile",
                        command=self._toggle_mode_controls).pack(anchor="w", padx=20, pady=1)
        ttk.Radiobutton(parent, text="2D axial\u2013radial contour", variable=self.plot_mode, value="contour",
                        command=self._toggle_mode_controls).pack(anchor="w", padx=20, pady=1)

        # fixed-position holder => the axis selector never jumps to the bottom of the panel
        self.axis_holder = tk.Frame(parent, bg=COLOR_PANEL)
        self.axis_holder.pack(fill="x", padx=16, pady=4)

        self.profile_axis_frame = tk.Frame(self.axis_holder, bg=COLOR_PANEL)
        ttk.Label(self.profile_axis_frame, text="Sweep along:", width=13).pack(side="left")
        self.sweep_axis = tk.StringVar(value=list(PROFILE_AXES)[0])
        cb1 = ttk.Combobox(self.profile_axis_frame, textvariable=self.sweep_axis,
                           values=list(PROFILE_AXES), width=24, state="readonly")
        cb1.pack(side="left")
        cb1.bind("<<ComboboxSelected>>", lambda e: self._update_entry_states())

        self.contour_axis_frame = tk.Frame(self.axis_holder, bg=COLOR_PANEL)
        ttk.Label(self.contour_axis_frame, text="Horizontal axis:", width=13).pack(side="left")
        self.radial_axis = tk.StringVar(value=list(CONTOUR_AXES)[0])
        cb2 = ttk.Combobox(self.contour_axis_frame, textvariable=self.radial_axis,
                           values=list(CONTOUR_AXES), width=24, state="readonly")
        cb2.pack(side="left")
        cb2.bind("<<ComboboxSelected>>", lambda e: self._update_entry_states())

        ttk.Label(parent, text="OUTPUT VARIABLE", style="Section.TLabel").pack(anchor="w", **pad)
        self.output_choice = tk.StringVar(value=list(OUTPUT_LABELS)[0])
        ttk.Combobox(parent, textvariable=self.output_choice, values=list(OUTPUT_LABELS),
                     state="readonly", width=30).pack(anchor="w", padx=16, pady=2)

        ttk.Separator(parent).pack(fill="x", padx=16, pady=8)

        btn_frame = tk.Frame(parent, bg=COLOR_PANEL)
        btn_frame.pack(fill="x", padx=16, pady=(0, 12))
        ttk.Button(btn_frame, text="RUN PREDICTION", style="Accent.TButton",
                   command=self.run_prediction).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Export figure (300 DPI)", command=self.export_figure).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Save data (CSV)", command=self.save_results).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Reset inputs", command=self.reset_inputs).pack(fill="x", pady=2)

        self._toggle_mode_controls()

    # ---------------- mode / state helpers ----------------
    def _current_axis_key(self):
        if self.plot_mode.get() == "profile":
            return PROFILE_AXES[self.sweep_axis.get()]
        return CONTOUR_AXES[self.radial_axis.get()]

    def _current_swept(self):
        swept = set(SWEPT[self._current_axis_key()])
        if self.plot_mode.get() == "contour":
            swept.add("z")
        return swept

    def _update_entry_states(self):
        swept = self._current_swept()
        for k in POSITION_KEYS:
            self.entries[k].configure(state="disabled" if k in swept else "normal")

    def _toggle_mode_controls(self):
        self.profile_axis_frame.pack_forget()
        self.contour_axis_frame.pack_forget()
        if self.plot_mode.get() == "profile":
            self.profile_axis_frame.pack(fill="x")
        else:
            self.contour_axis_frame.pack(fill="x")
        self._update_entry_states()

    # ---------------- logic ----------------
    def _read_inputs(self, skip):
        values = {}
        for name, meta in INPUT_VARIABLES.items():
            if name in skip:
                continue
            val, err = validate_number(self.entries[name].get(), meta["label"])
            if err:
                return None, err
            values[name] = val
        return values, None

    def _embed_figure(self, fig):
        if self.plot_container is not None:
            self.plot_container.destroy()
        self.plot_container = tk.Frame(self.plot_area, bg=COLOR_PANEL)
        self.plot_container.pack(fill="both", expand=True)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        toolbar = NavigationToolbar2Tk(canvas, self.plot_container, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        canvas.draw()
        self.canvas = canvas
        self.current_fig = fig

    # -- 1D --
    def _run_profile(self, values, out_key):
        axis_key = PROFILE_AXES[self.sweep_axis.get()]
        disp, coords, label = build_axis(axis_key, N_PROFILE_POINTS)
        inputs = {**values, **coords}
        raw = self.predictor.predict(inputs)
        pred, counts = apply_physical_limits(raw)

        swept = SWEPT[axis_key]
        fig = plot_profile(disp, pred, raw, out_key, label, axis_key,
                           condition_text(values, swept))
        table = make_table(values, inputs, pred, raw, {"axis_value": disp})
        return fig, table, clip_messages(counts, len(disp)), f"Profile along {axis_key} completed."

    # -- 2D --
    def _run_contour(self, values, out_key):
        h_key = CONTOUR_AXES[self.radial_axis.get()]
        h_disp, h_coords, h_label = build_axis(h_key, N_CONTOUR_R)
        z_vals = np.linspace(Z_RANGE[0], Z_RANGE[1], N_CONTOUR_Z)
        nr, nz = len(h_disp), len(z_vals)

        ih = np.tile(np.arange(nr), nz)                   # horizontal index of every grid point
        iz = np.repeat(np.arange(nz), nr)                 # vertical index of every grid point
        coords = {k: arr[ih] for k, arr in h_coords.items()}
        coords["z"] = z_vals[iz]
        inputs = {**values, **coords}

        xs = np.broadcast_to(np.asarray(inputs["x"], float), ih.shape)
        ys = np.broadcast_to(np.asarray(inputs["y"], float), ih.shape)
        inside = (xs - REACTOR_CX) ** 2 + (ys - REACTOR_CY) ** 2 <= (REACTOR_RADIUS * (1 + 1e-6)) ** 2
        if not MASK_OUTSIDE_WALL:
            inside[:] = True
        if not inside.any():
            raise ValueError("The selected plane lies completely outside the column. "
                             "Choose a fixed coordinate closer to the axis.")

        raw = self.predictor.predict(inputs)
        raw[~inside] = np.nan                             # blank everything outside the wall
        pred, counts = apply_physical_limits(raw)

        fig = plot_contour(h_disp, z_vals, pred.reshape(nz, nr, N_OUTPUTS), out_key, h_label,
                           condition_text(values, SWEPT[h_key] | {"z"}))
        table = make_table(values, inputs, pred, raw, {"axis_value": h_disp[ih]})
        table = table[inside].reset_index(drop=True)
        return fig, table, clip_messages(counts, int(inside.sum())), f"Axial\u2013radial contour ({h_key}\u2013z) completed."

    def run_prediction(self):
        if self.predictor.status != "Loaded":
            messagebox.showerror("Model not loaded", self.predictor.status)
            return

        mode = self.plot_mode.get()
        out_key = OUTPUT_LABELS[self.output_choice.get()]
        swept = self._current_swept()

        values, err = self._read_inputs(skip=swept)       # swept coordinates are NOT validated
        if err:
            messagebox.showerror("Invalid input", err)
            return

        warnings = collect_input_warnings(values, swept)

        self.config(cursor="watch")
        self.bottom_status.set("Running prediction\u2026")
        self.update_idletasks()
        try:
            if mode == "profile":
                fig, table, clip_msgs, done_msg = self._run_profile(values, out_key)
            else:
                fig, table, clip_msgs, done_msg = self._run_contour(values, out_key)
        except Exception as e:
            self.bottom_status.set("Prediction failed.")
            messagebox.showerror("Prediction error", str(e))
            return
        finally:
            self.config(cursor="")

        warnings += clip_msgs
        self.warn_var.set("\n".join("\u26a0 " + w for w in warnings))
        self.bottom_status.set(done_msg + (f"   ({len(warnings)} warning(s) below the plot)" if warnings else ""))
        self.last_table = table
        self._embed_figure(fig)

    def reset_inputs(self):
        for name, entry in self.entries.items():
            default = INPUT_VARIABLES[name]["default"]
            if isinstance(entry, ttk.Combobox):
                entry.set(str(default))
            else:
                entry.configure(state="normal")           # a disabled Entry ignores delete/insert
                entry.delete(0, tk.END)
                entry.insert(0, str(default))
        self._update_entry_states()
        self.warn_var.set("")
        self.bottom_status.set("Inputs reset to defaults.")

    def save_results(self):
        if self.last_table is None:
            messagebox.showwarning("Nothing to save", "Run a prediction first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        self.last_table.to_csv(path, index=False)
        messagebox.showinfo("Saved", f"Data saved to {path}")

    def export_figure(self):
        if self.current_fig is None:
            messagebox.showwarning("Nothing to export", "Run a prediction first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png")])
        if not path:
            return
        self.current_fig.savefig(path, dpi=300, facecolor="white")
        messagebox.showinfo("Exported", f"Figure saved to {path} (300 DPI)")


if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()