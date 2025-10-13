import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
try:
    from tqdm import tqdm
except Exception:
    class tqdm:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        def update(self, *args, **kwargs):
            pass
        def close(self):
            pass


SCRIPT_ARGS = None


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Generate motion vectors using dense optical flow (no detection).")
    parser.add_argument("video_path", type=str, help="Path to input video")

    # Visualization and processing
    parser.add_argument("--frame_limit", type=int, default=-1, help="Number of frames to process (-1 for full video)")
    parser.add_argument("--arrow_scale", type=float, default=0.02, help="Scale factor for arrow length (pixels/sec × scale)")
    parser.add_argument("--arrow_color", type=str, default="0,255,0", help="Arrow BGR color as 'B,G,R'")
    parser.add_argument("--thickness", type=int, default=2, help="Arrow thickness")
    parser.add_argument("--save_path", type=str, default=None, help="Output annotated video path (default next to input)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    # Flow and keypoint/track params
    parser.add_argument("--grid_step", type=int, default=16, help="Grid step for sampling flow vectors (pixels)")
    parser.add_argument("--min_flow_mag", type=float, default=1.0, help="Minimum flow magnitude (pixels/frame) to draw")
    parser.add_argument("--smoothing", type=int, default=1, help="Temporal smoothing length for per-point velocity (frames)")

    # Crash scoring and collision prediction (conservative defaults for low FP)
    parser.add_argument("--crash_threshold", type=float, default=0.9, help="Weighted score threshold to flag crash")
    parser.add_argument("--weight_abrupt", type=float, default=0.1, help="Weight for abrupt direction change factor")
    parser.add_argument("--weight_predict", type=float, default=1.0, help="Weight for predicted collision factor")
    parser.add_argument("--weight_intersect", type=float, default=0.0, help="Weight for bbox intersection factor")
    parser.add_argument("--abrupt_angle_deg", type=float, default=75.0, help="Minimum direction change angle (degrees)")
    parser.add_argument("--min_speed", type=float, default=5.0, help="Min speed (pixels/sec) for abrupt consideration")
    parser.add_argument("--pred_horizon", type=int, default=8, help="Frames ahead to predict for potential collision")
    parser.add_argument("--pred_step", type=int, default=1, help="Frame step for rolling prediction")
    parser.add_argument("--pred_iou_thresh", type=float, default=0.05, help="If >0, require IoU >= threshold for predicted collision")
    parser.add_argument("--intersect_margin", type=int, default=-8, help="Margin around boxes when checking intersection (negative shrinks)")
    parser.add_argument("--min_pair_speed", type=float, default=1.5, help="Min combined speed (px/frame) to consider a pair")
    parser.add_argument("--pred_viz_threshold", type=float, default=0.2, help="Min predicted collision factor to draw projections")
    parser.add_argument("--pair_pred_persist", type=int, default=3, help="Consecutive frames requiring predicted collision")
    parser.add_argument("--pair_intersect_persist", type=int, default=2, help="Consecutive frames requiring intersection")
    parser.add_argument("--track_max_age", type=int, default=10, help="Max frames to keep unmatched tracks")
    parser.add_argument("--pair_cooldown", type=int, default=45, help="Cooldown frames to avoid double-flagging same pair")
    parser.add_argument("--converge_angle_deg", type=float, default=90.0, help="Min angle between velocities to consider converging")

    # Farneback params (exposed for tuning)
    parser.add_argument("--fb_pyr_scale", type=float, default=0.5)
    parser.add_argument("--fb_levels", type=int, default=3)
    parser.add_argument("--fb_winsize", type=int, default=15)
    parser.add_argument("--fb_iterations", type=int, default=3)
    parser.add_argument("--fb_poly_n", type=int, default=5)
    parser.add_argument("--fb_poly_sigma", type=float, default=1.2)
    parser.add_argument("--fb_flags", type=int, default=0)

    # Motion cluster extraction parameters
    parser.add_argument("--cluster_mag_thresh", type=float, default=1.5, help="Flow magnitude threshold (px/frame) for cluster mask")
    parser.add_argument("--min_cluster_area", type=int, default=300, help="Minimum area (px) for a motion cluster")
    parser.add_argument("--match_iou_thresh", type=float, default=0.1, help="IoU threshold to match clusters across frames")

    if SCRIPT_ARGS is not None:
        defaults = {a.dest: a.default for a in parser._actions if a.dest != "help"}
        defaults.update(SCRIPT_ARGS)
        if not defaults.get("video_path"):
            raise ValueError("SCRIPT_ARGS must include 'video_path'")
        return argparse.Namespace(**defaults)
    return parser.parse_args()


def parse_bgr(color_str: str) -> Tuple[int, int, int]:

    try:
        b, g, r = [int(v) for v in color_str.split(",")]
        return b, g, r
    except Exception:
        return 0, 255, 0


def angle_between(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    x1, y1 = v1
    x2, y2 = v2
    n1 = float(np.hypot(x1, y1))
    n2 = float(np.hypot(x2, y2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosang = (x1 * x2 + y1 * y2) / (n1 * n2 + 1e-6)
    cosang = float(np.clip(cosang, -1.0, 1.0))
    ang = float(np.degrees(np.arccos(cosang)))
    return ang


def open_capture_preferring_ffmpeg(path: str) -> cv2.VideoCapture:
    cap_ff = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if cap_ff.isOpened():
        return cap_ff
    return cv2.VideoCapture(path)


def boxes_intersect(a: np.ndarray, b: np.ndarray, margin: int = 0) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ax1m = ax1 - margin
    ay1m = ay1 - margin
    ax2m = ax2 + margin
    ay2m = ay2 + margin
    bx1m = bx1 - margin
    by1m = by1 - margin
    bx2m = bx2 + margin
    by2m = by2 + margin
    inter_w = min(ax2m, bx2m) - max(ax1m, bx1m)
    inter_h = min(ay2m, by2m) - max(ay1m, by1m)
    return inter_w > 0 and inter_h > 0


def intersection_xyxy(a: np.ndarray, b: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = int(max(ax1, bx1))
    y1 = int(max(ay1, by1))
    x2 = int(min(ax2, bx2))
    y2 = int(min(ay2, by2))
    if x2 > x1 and y2 > y1:
        return x1, y1, x2, y2
    return None


def compute_center_xyxy(xyxy: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy.tolist()
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    inter = intersection_xyxy(a, b)
    if inter is None:
        return 0.0
    x1, y1, x2, y2 = inter
    inter_area = float(max(0, x2 - x1) * max(0, y2 - y1))
    area_a = float(max(0, a[2] - a[0]) * max(0, a[3] - a[1]))
    area_b = float(max(0, b[2] - b[0]) * max(0, b[3] - b[1]))
    union = area_a + area_b - inter_area + 1e-6
    return inter_area / union


def predict_collision_factor(
    a_box: np.ndarray,
    a_vel: Tuple[float, float],
    b_box: np.ndarray,
    b_vel: Tuple[float, float],
    horizon: int,
    margin: int,
    step: int = 1,
    iou_thresh: float = 0.0,
) -> float:
    max_factor = 0.0
    for t in range(step, max(1, horizon) + 1, max(1, step)):
        ax = a_box.copy()
        bx = b_box.copy()
        ax[[0, 2]] = ax[[0, 2]] + a_vel[0] * t
        ax[[1, 3]] = ax[[1, 3]] + a_vel[1] * t
        bx[[0, 2]] = bx[[0, 2]] + b_vel[0] * t
        bx[[1, 3]] = bx[[1, 3]] + b_vel[1] * t
        intersects = boxes_intersect(ax, bx, margin=margin)
        if iou_thresh > 0.0:
            inter = min(ax[2], bx[2]) - max(ax[0], bx[0])
            inter_h = min(ax[3], bx[3]) - max(ax[1], bx[1])
            inter_area = max(0.0, inter) * max(0.0, inter_h)
            if inter_area > 0:
                area_a = max(0.0, (ax[2] - ax[0])) * max(0.0, (ax[3] - ax[1]))
                area_b = max(0.0, (bx[2] - bx[0])) * max(0.0, (bx[3] - bx[1]))
                union = area_a + area_b - inter_area + 1e-6
                iou_val = inter_area / union
            else:
                iou_val = 0.0
            intersects = intersects and (iou_val >= iou_thresh)
        if intersects:
            factor = 1.0 - (t - 1) / max(1, horizon)
            if factor > max_factor:
                max_factor = factor
    return max_factor


def compute_dense_flow(prev_gray: np.ndarray, gray: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        args.fb_pyr_scale,
        args.fb_levels,
        args.fb_winsize,
        args.fb_iterations,
        args.fb_poly_n,
        args.fb_poly_sigma,
        args.fb_flags,
    )
    return flow  # float32 HxWx2 (dx, dy) in pixels/frame


def extract_motion_clusters(flow: np.ndarray, args: argparse.Namespace) -> List[Dict[str, object]]:
    h, w = flow.shape[:2]
    mag = np.hypot(flow[..., 0], flow[..., 1])
    mask = (mag >= float(args.cluster_mag_thresh)).astype(np.uint8) * 255
    # Morphological cleanup
    k = max(3, min(9, int(round(args.grid_step)) // 2 * 2 + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clusters: List[Dict[str, object]] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < float(args.min_cluster_area):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        x1, y1, x2, y2 = x, y, x + bw, y + bh
        # Mean flow inside contour region
        comp_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(comp_mask, [cnt], -1, 255, thickness=-1)
        comp_mask_bool = comp_mask.astype(bool)
        dx_vals = flow[..., 0][comp_mask_bool]
        dy_vals = flow[..., 1][comp_mask_bool]
        if dx_vals.size == 0:
            continue
        mean_dx = float(np.mean(dx_vals))
        mean_dy = float(np.mean(dy_vals))
        clusters.append({
            "box": np.array([x1, y1, x2, y2], dtype=np.float32),
            "vel": (mean_dx, mean_dy),
            "area": area,
        })
    return clusters


def main():

    args = parse_args()
    arrow_color = parse_bgr(args.arrow_color)

    if not os.path.isfile(args.video_path):
        print(f"Input video not found: {args.video_path}")
        sys.exit(1)

    cap = open_capture_preferring_ffmpeg(args.video_path)
    if not cap.isOpened():
        print("Failed to open video")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if args.save_path is None:
        base, ext = os.path.splitext(args.video_path)
        args.save_path = f"{base}_optflow_vectors{ext or '.mp4'}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.save_path, fourcc, fps, (width, height))

    if args.verbose:
        print(f"Starting optical-flow processing: {os.path.basename(args.video_path)} | {width}x{height} @ {fps:.2f} FPS | frames={total_frames or 'unknown'}")

    # For abruptness scoring and clustering/track state
    recent_avg_vels: List[Tuple[float, float]] = []
    next_track_id = 1
    tracks: Dict[int, Dict[str, object]] = {}
    # Pair persistence and cooldown states
    pair_pred_count: Dict[Tuple[int, int], int] = {}
    pair_inter_count: Dict[Tuple[int, int], int] = {}
    pair_last_flag: Dict[Tuple[int, int], int] = {}

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Empty video")
        cap.release()
        writer.release()
        sys.exit(1)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_idx = 1
    # Progress bar counts frames processed after the first already-read frame
    effective_total = None
    if args.frame_limit >= 0:
        if total_frames > 0:
            effective_total = min(args.frame_limit, total_frames) - 1
        else:
            effective_total = args.frame_limit - 1
    else:
        effective_total = (total_frames - 1) if total_frames > 0 else None
    pbar = tqdm(total=(effective_total if (effective_total is not None and effective_total > 0) else None), desc="Optical flow", unit="frame")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if args.frame_limit >= 0 and frame_idx >= args.frame_limit:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = compute_dense_flow(prev_gray, gray, args)

            # Compute grid-sampled arrows
            step = max(1, int(args.grid_step))
            ys, xs = np.mgrid[step//2:height:step, step//2:width:step]
            fx = flow[ys, xs, 0]
            fy = flow[ys, xs, 1]

            # Magnitude (pixels/frame) and convert to pixels/sec
            mag = np.hypot(fx, fy)
            mask = mag >= float(args.min_flow_mag)

            # Average velocity over all drawn points (in pixels/frame)
            if mask.any():
                avg_dx = float(np.mean(fx[mask]))
                avg_dy = float(np.mean(fy[mask]))
            else:
                avg_dx = 0.0
                avg_dy = 0.0

            # Temporal smoothing for average velocity if requested
            if args.smoothing > 1:
                recent_avg_vels.append((avg_dx, avg_dy))
                if len(recent_avg_vels) > args.smoothing:
                    recent_avg_vels.pop(0)
                sm_dx = float(np.mean([v[0] for v in recent_avg_vels]))
                sm_dy = float(np.mean([v[1] for v in recent_avg_vels]))
                avg_dx, avg_dy = sm_dx, sm_dy

            # Draw arrows for selected grid points
            arrow_len_scale = float(args.arrow_scale)
            pts_y = ys[mask]
            pts_x = xs[mask]
            sel_fx = fx[mask]
            sel_fy = fy[mask]
            for px, py, dx, dy in zip(pts_x, pts_y, sel_fx, sel_fy):
                # Convert per-frame displacement to per-second for length scaling
                speed_dx = dx * fps
                speed_dy = dy * fps
                end_x = int(px + speed_dx * arrow_len_scale)
                end_y = int(py + speed_dy * arrow_len_scale)
                cv2.arrowedLine(
                    frame,
                    (int(px), int(py)),
                    (end_x, end_y),
                    arrow_color,
                    thickness=max(1, args.thickness),
                    tipLength=0.3,
                )

            # Simple abruptness factor: change of average flow direction
            # Compare with previous average (stored as last of recent_avg_vels before append above)
            if len(recent_avg_vels) >= 1:
                prev_dx, prev_dy = recent_avg_vels[-1]
            else:
                prev_dx, prev_dy = 0.0, 0.0
            ang = angle_between((prev_dx, prev_dy), (avg_dx, avg_dy))
            speed_now = float(np.hypot(avg_dx * fps, avg_dy * fps))
            abrupt_factor = 1.0 if (speed_now >= args.min_speed and ang >= args.abrupt_angle_deg) else 0.0

            # Maintain history after computing abruptness
            if args.smoothing <= 1:
                recent_avg_vels.append((avg_dx, avg_dy))
                if len(recent_avg_vels) > 2:
                    recent_avg_vels.pop(0)

            # Extract motion clusters and track them via IoU
            clusters = extract_motion_clusters(flow, args)
            # Age existing tracks
            for tid in list(tracks.keys()):
                tracks[tid]["age"] = int(tracks[tid].get("age", 0)) + 1
                tracks[tid]["matched"] = False

            # Match clusters to tracks (greedy by IoU)
            unmatched_clusters = list(range(len(clusters)))
            for tid, tr in list(tracks.items()):
                best_ci = -1
                best_iou = 0.0
                tbox = tr["box"]  # type: ignore
                for ci in unmatched_clusters:
                    iou_val = iou_xyxy(np.array(tbox, dtype=np.float32), clusters[ci]["box"])  # type: ignore
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_ci = ci
                if best_ci >= 0 and best_iou >= float(args.match_iou_thresh):
                    # Update track
                    tracks[tid]["box"] = clusters[best_ci]["box"]
                    tracks[tid]["vel"] = clusters[best_ci]["vel"]
                    tracks[tid]["matched"] = True
                    tracks[tid]["age"] = 0
                    unmatched_clusters.remove(best_ci)

            # Create new tracks for unmatched clusters
            for ci in unmatched_clusters:
                tracks[next_track_id] = {
                    "box": clusters[ci]["box"],
                    "vel": clusters[ci]["vel"],
                    "matched": True,
                    "age": 0,
                }
                next_track_id += 1

            # Drop stale tracks
            for tid in list(tracks.keys()):
                if not tracks[tid].get("matched", False) and int(tracks[tid].get("age", 0)) > int(args.track_max_age):
                    del tracks[tid]

            # Pairwise predicted collision and current intersection factors across motion clusters
            predict_factor = 0.0
            intersect_factor = 0.0
            collision_highlight_rect: Optional[Tuple[int, int, int, int]] = None
            tids = sorted(tracks.keys())
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    ta = tracks[tids[i]]
                    tb = tracks[tids[j]]
                    va = ta["vel"]  # type: ignore
                    vb = tb["vel"]  # type: ignore
                    # Combined speed gate (pixels/frame)
                    pair_speed = float(np.hypot(va[0], va[1]) + np.hypot(vb[0], vb[1]))
                    if pair_speed < float(args.min_pair_speed):
                        continue
                    # Convergence gate: angle between velocities should be high (approaching)
                    if angle_between(va, (-vb[0], -vb[1])) < float(args.converge_angle_deg):
                        continue
                    pa = np.array(ta["box"], dtype=np.float32)  # type: ignore
                    pb = np.array(tb["box"], dtype=np.float32)  # type: ignore
                    pc = predict_collision_factor(
                        pa, va, pb, vb, int(args.pred_horizon), int(args.intersect_margin),
                        step=max(1, int(args.pred_step)), iou_thresh=max(0.0, float(args.pred_iou_thresh))
                    )
                    pair_key = (tids[i], tids[j])
                    # Persistence counters for low FPs
                    if pc > 0:
                        pair_pred_count[pair_key] = pair_pred_count.get(pair_key, 0) + 1
                    else:
                        pair_pred_count[pair_key] = 0
                    if boxes_intersect(pa, pb, margin=int(args.intersect_margin)):
                        pair_inter_count[pair_key] = pair_inter_count.get(pair_key, 0) + 1
                        inter_rect = intersection_xyxy(pa, pb)
                        if inter_rect is not None:
                            collision_highlight_rect = inter_rect
                    else:
                        pair_inter_count[pair_key] = 0

                    # Respect cooldown after a flagged collision
                    last_flag = pair_last_flag.get(pair_key, -10**9)
                    on_cooldown = (frame_idx - last_flag) < int(args.pair_cooldown)

                    if not on_cooldown:
                        if pair_pred_count.get(pair_key, 0) >= int(args.pair_pred_persist):
                            if pc > predict_factor:
                                predict_factor = pc
                        if pair_inter_count.get(pair_key, 0) >= int(args.pair_intersect_persist):
                            intersect_factor = 1.0

            crash_score = (
                args.weight_abrupt * abrupt_factor
                + args.weight_predict * predict_factor
                + args.weight_intersect * intersect_factor
            )

            # Diagnostics and CRASH banner
            diag_font = cv2.FONT_HERSHEY_SIMPLEX
            diag_scale = 0.6
            diag_th = 2
            t1 = f"Score={crash_score:.2f}"
            t2 = f"A={abrupt_factor:.2f} P={predict_factor:.2f} I={intersect_factor:.2f}"
            (t1w, t1h), t1b = cv2.getTextSize(t1, diag_font, diag_scale, diag_th)
            (t2w, t2h), t2b = cv2.getTextSize(t2, diag_font, diag_scale, diag_th)
            pad = 8
            box_w = max(t1w, t2w) + 2 * pad
            box_h = (t1h + t1b) + (t2h + t2b) + 3 * pad
            x1d, y1d = 10, 10
            x2d, y2d = x1d + box_w, y1d + box_h
            cv2.rectangle(frame, (x1d, y1d), (x2d, y2d), (0, 0, 0), thickness=-1)
            cv2.putText(frame, t1, (x1d + pad, y1d + pad + t1h), diag_font, diag_scale, (255, 255, 255), diag_th, cv2.LINE_AA)
            cv2.putText(frame, t2, (x1d + pad, y1d + 2 * pad + t1h + t2h), diag_font, diag_scale, (255, 255, 255), diag_th, cv2.LINE_AA)

            if crash_score >= args.crash_threshold:
                text = "CRASH"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                pad = 10
                rect_w = tw + 2 * pad
                rect_h = th + baseline + 2 * pad
                x1 = width - rect_w - 10
                y1 = 10
                x2 = x1 + rect_w
                y2 = y1 + rect_h
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=-1)
                cv2.putText(
                    frame,
                    text,
                    (x1 + pad, y1 + pad + th),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )
                # Remember last flag time for all pairs that met persistence this frame
                for pk, cnt in pair_pred_count.items():
                    if cnt >= int(args.pair_pred_persist):
                        pair_last_flag[pk] = frame_idx
                for pk, cnt in pair_inter_count.items():
                    if cnt >= int(args.pair_intersect_persist):
                        pair_last_flag[pk] = frame_idx

            writer.write(frame)

            prev_gray = gray
            frame_idx += 1
            try:
                pbar.update(1)
            except Exception:
                if args.verbose and (frame_idx % 100 == 0):
                    print(f"Processed {frame_idx} frames...")
    finally:
        cap.release()
        writer.release()
        try:
            pbar.close()
        except Exception:
            pass

    print(f"Saved annotated video to: {args.save_path}")


if __name__ == "__main__":

    main()


