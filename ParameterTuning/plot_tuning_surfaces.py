
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as mcolors

def parse_iterhistory(path):
    """
    Parse an iterhistory text file (e.g., iterhist.txt) containing lines like:
    [iter k] score=195.00 cfg={"pred_horizon": 8, "pred_step": 1, ...}
    Returns a list of dicts with numeric fields.
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if "score" not in line or "cfg=" not in line:
                continue
            m_score = re.search(r"score\s*=\s*([0-9]+(?:\.[0-9]+)?)", line)
            cfg_eq_idx = line.find("cfg=")
            brace_l = line.find("{", cfg_eq_idx)
            brace_r = line.rfind("}")
            if not m_score or brace_l == -1 or brace_r == -1 or brace_r <= brace_l:
                continue
            score = float(m_score.group(1))
            cfg_str = line[brace_l:brace_r+1]
            try:
                cfg = json.loads(cfg_str)
            except json.JSONDecodeError:
                cfg_fixed = cfg_str.replace("'", '"')
                cfg_fixed = re.sub(r",\s*}\s*$", "}", cfg_fixed)
                cfg = json.loads(cfg_fixed)
            row = {"score": score}
            row.update(cfg)
            rows.append(row)
    return rows

def make_scatter3d(ax, x, y, z, c, title, xlab, ylab, zlab, cmap, norm=None, marker_size=30):
    """
    Plot a 3D scatter where color encodes score (hotter = higher).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    c = np.asarray(c, dtype=float)

    sc = ax.scatter(x, y, z, c=c, cmap=cmap, norm=norm, s=marker_size)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)
    ax.set_title(title)
    return sc

def make_surface_color(ax, x, y, z, c, title, xlab, ylab, zlab, cmap, norm, below_rgba=None):
    """
    Interpolate a surface Z(X,Y) from scattered points and color it by interpolated score C(X,Y).
    - Geometry: uses LinearTriInterpolator on a triangulation of (x, y) for z
    - Color:    uses the same interpolator on score c to get smooth color variation
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    c = np.asarray(c, dtype=float)

    tri = mtri.Triangulation(x, y)

    z_interp = mtri.LinearTriInterpolator(tri, z)
    c_interp = mtri.LinearTriInterpolator(tri, c)

    xi = np.linspace(x.min(), x.max(), 80)
    yi = np.linspace(y.min(), y.max(), 80)
    XI, YI = np.meshgrid(xi, yi)
    ZI = z_interp(XI, YI)
    CI = c_interp(XI, YI)

    # Handle NaNs outside convex hull by nearest fill for continuity near edges
    nan_mask = np.isnan(ZI) | np.isnan(CI)
    if np.any(nan_mask):
        nn_interp_z = (mtri.CubicTriInterpolator(tri, z) if len(x) >= 16 else mtri.LinearTriInterpolator(tri, z))
        nn_interp_c = (mtri.CubicTriInterpolator(tri, c) if len(x) >= 16 else mtri.LinearTriInterpolator(tri, c))
        ZI[nan_mask] = nn_interp_z(XI, YI)[nan_mask]
        CI[nan_mask] = nn_interp_c(XI, YI)[nan_mask]

    # Map score to colors
    cmap_obj = plt.get_cmap(cmap)
    facecolors = cmap_obj(norm(CI))
    if below_rgba is not None:
        below_mask = CI < norm.vmin
        if np.any(below_mask):
            facecolors[below_mask] = below_rgba

    # Plot gridded surface with colored faces
    surf = ax.plot_surface(XI, YI, ZI, facecolors=facecolors, linewidth=0, antialiased=True)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)
    ax.set_title(title)
    return surf

def main():
    parser = argparse.ArgumentParser(description="Plot 3D score surfaces from iterhistory file.")
    parser.add_argument("--file", type=str, default="/Users/yousefradwan/Library/CloudStorage/GoogleDrive-radwanf2025@gmail.com/My Drive/Yousef/OttonomiAI/CarAccidentVideos/ParameterTuning/iterhist.txt", help="Path to iterhistory file (.rtf or .txt).")
    parser.add_argument("--min_speed", type=float, default=1.5, help="Keep rows with min_pair_speed > this value.")
    parser.add_argument("--x_param", type=str, default="pred_horizon", help="Parameter for X axis.")
    parser.add_argument("--y_param", type=str, default="pred_iou_thresh", help="Parameter for Y axis.")
    parser.add_argument("--z_param", type=str, default="intersect_margin", help="Parameter for Z axis (e.g., intersect_margin).")
    parser.add_argument("--out_prefix", type=str, default="/Users/yousefradwan/Library/CloudStorage/GoogleDrive-radwanf2025@gmail.com/My Drive/Yousef/OttonomiAI/CarAccidentVideos/ParameterTuning/score_surface", help="Output path prefix (no extension).")
    parser.add_argument("--no_save", action="store_true", help="Do not save PNG files; only display interactively.")
    parser.add_argument("--no_show", action="store_true", help="Do not display interactive windows; only save PNG files.")
    parser.add_argument("--cmap", type=str, default="hot", help="Matplotlib colormap to use for score coloring.")
    parser.add_argument("--marker_size", type=float, default=30, help="Marker size for scatter points.")
    parser.add_argument("--score_min", type=float, default=None, help="Minimum score for color scale; values below are colored gray (or --below_color).")
    parser.add_argument("--below_color", type=str, default="#808080", help="Color for values below --score_min (hex or name).")
    args = parser.parse_args()

    rows = parse_iterhistory(args.file)
    if not rows:
        raise SystemExit("No rows parsed. Check the file path and format.")

    # Filter and split
    rows = [r for r in rows if float(r.get("min_pair_speed", -1e9)) > args.min_speed]
    step1 = [r for r in rows if int(r.get("pred_step", -1)) == 1]
    step2 = [r for r in rows if int(r.get("pred_step", -1)) == 2]

    if not step1 and not step2:
        raise SystemExit("No data after filtering. Try lowering --min_speed.")

    def extract_xyzC(R):
        X = [float(r[args.x_param]) for r in R]
        Y = [float(r[args.y_param]) for r in R]
        Z = [float(r.get(args.z_param, np.nan)) for r in R]
        C = [float(r["score"]) for r in R]
        return X, Y, Z, C

    # Normalize colors consistently across plots (optionally clamp lower bound)
    all_scores = [float(r["score"]) for r in rows]
    data_vmin, data_vmax = (min(all_scores), max(all_scores))
    if args.score_min is not None:
        vmin = args.score_min if args.score_min < data_vmax else max(data_vmax - 1e-6, data_vmin)
    else:
        vmin = data_vmin
    vmax = data_vmax
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    below_rgba = mcolors.to_rgba(args.below_color) if args.score_min is not None else None

    # Plot pred_step = 1 (interpolated surface colored by score)
    if step1:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection="3d")
        X, Y, Z, C = extract_xyzC(step1)
        surf1 = make_surface_color(ax1, X, Y, Z, C, f"Score surface (pred_step=1)\nmin_pair_speed>{args.min_speed}", args.x_param, args.y_param, args.z_param, args.cmap, norm, below_rgba=below_rgba)
        mappable1 = plt.cm.ScalarMappable(norm=norm, cmap=args.cmap)
        mappable1.set_array([])
        cbar1 = fig1.colorbar(mappable1, ax=ax1)
        cbar1.set_label("score")
        fig1.tight_layout()
        if not args.no_save:
            fig1.savefig(f"{args.out_prefix}_step1.png", dpi=180)

    # Plot pred_step = 2 (interpolated surface colored by score)
    if step2:
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection="3d")
        X, Y, Z, C = extract_xyzC(step2)
        surf2 = make_surface_color(ax2, X, Y, Z, C, f"Score surface (pred_step=2)\nmin_pair_speed>{args.min_speed}", args.x_param, args.y_param, args.z_param, args.cmap, norm, below_rgba=below_rgba)
        mappable2 = plt.cm.ScalarMappable(norm=norm, cmap=args.cmap)
        mappable2.set_array([])
        cbar2 = fig2.colorbar(mappable2, ax=ax2)
        cbar2.set_label("score")
        fig2.tight_layout()
        if not args.no_save:
            fig2.savefig(f"{args.out_prefix}_step2.png", dpi=180)

    # Optional: also save a CSV of the parsed/filtered data for audit
    try:
        import csv
        with open(f"{args.out_prefix}_filtered.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(step1[0].keys() if step1 else step2[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    except Exception as e:
        print("CSV export skipped:", e)

    # Show interactive plots (allows rotate/pan) unless disabled
    if not args.no_show and (step1 or step2):
        plt.show()

if __name__ == "__main__":
    main()
