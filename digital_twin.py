import argparse
import sys
from typing import List, Tuple, Optional

import cv2
import numpy as np


def draw_twin_canvas(canvas_size: int, grid_step: int = 100) -> np.ndarray:
    """Create a square digital-twin canvas with border, grid, and axis ticks."""
    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)

    # Border
    cv2.rectangle(canvas, (0, 0), (canvas_size - 1, canvas_size - 1), (0, 0, 0), 2)

    # Grid
    for x in range(grid_step, canvas_size, grid_step):
        cv2.line(canvas, (x, 0), (x, canvas_size), (230, 230, 230), 1)
    for y in range(grid_step, canvas_size, grid_step):
        cv2.line(canvas, (0, y), (canvas_size, y), (230, 230, 230), 1)

    # Ticks and labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x in range(0, canvas_size, grid_step):
        cv2.line(canvas, (x, 0), (x, 10), (0, 0, 0), 2)
        cv2.putText(canvas, str(x), (x + 4, 24), font, 0.4, (80, 80, 80), 1, cv2.LINE_AA)
    for y in range(0, canvas_size, grid_step):
        cv2.line(canvas, (0, y), (10, y), (0, 0, 0), 2)
        cv2.putText(canvas, str(y), (14, y + 16), font, 0.4, (80, 80, 80), 1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        "Digital Twin (top-left = (0,0))",
        (10, canvas_size - 10),
        font,
        0.6,
        (60, 60, 60),
        2,
        cv2.LINE_AA,
    )
    return canvas


def select_four_points(window_name: str, image: np.ndarray) -> List[Tuple[int, int]]:
    """Let the user click 4 points on the image, returning them in click order.

    Instruction displayed on window title and overlay text guides the user to click
    in the order: top-left, top-right, bottom-right, bottom-left.
    """
    points: List[Tuple[int, int]] = []

    clone = image.copy()
    overlay = image.copy()
    cv2.putText(
        overlay,
        "Click 4 pts: TL, TR, BR, BL. Press 'r' to reset, 'Enter' to accept.",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    image = overlay

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display = image.copy()
        for idx, (px, py) in enumerate(points):
            cv2.circle(display, (px, py), 6, (0, 0, 255), -1)
            cv2.putText(
                display,
                f"{idx+1}",
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == 13:  # Enter
            if len(points) == 4:
                break
        elif key == ord('r'):
            points.clear()
            image = clone.copy()
            overlay = image.copy()
            cv2.putText(
                overlay,
                "Click 4 pts: TL, TR, BR, BL. Press 'r' to reset, 'Enter' to accept.",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            image = overlay

        # Escape cancels
        if key == 27:
            points.clear()
            break

    cv2.destroyWindow(window_name)
    return points


def compute_homography(
    frame_points: List[Tuple[int, int]],
    twin_size: int,
) -> Optional[np.ndarray]:
    """Compute homography from 4 source points (frame) to twin square corners.

    Source order expected: TL, TR, BR, BL.
    Destination is square corners: (0,0), (W-1,0), (W-1,H-1), (0,H-1)
    """
    if len(frame_points) != 4:
        return None

    dst_points = np.array(
        [
            [0, 0],
            [twin_size - 1, 0],
            [twin_size - 1, twin_size - 1],
            [0, twin_size - 1],
        ],
        dtype=np.float32,
    )
    src_points = np.array(frame_points, dtype=np.float32)

    H, status = cv2.findHomography(src_points, dst_points, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return H


def load_yolo(model_preference: str):
    """Load a YOLO model, preferring provided model path/name.

    Falls back to yolo11x if yolo12x is unavailable.
    """
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - dependency import
        print("Error: ultralytics is not installed. Please install with 'pip install ultralytics'.", file=sys.stderr)
        raise

    model_candidates = [model_preference]
    # If user asked for yolo12x, prepare graceful fallback to yolo11x
    if model_preference.lower() in {"yolo12x", "yolo12x.pt", "yolo12x.yaml"}:
        model_candidates.append("yolo11x.pt")
        model_candidates.append("yolo11x")

    last_err = None
    for candidate in model_candidates:
        try:
            model = YOLO(candidate)
            print(f"Loaded model: {candidate}")
            return model
        except Exception as exc:
            last_err = exc
            print(f"Warning: failed to load model '{candidate}': {exc}")

    # If we got here, all candidates failed
    raise RuntimeError(f"Failed to load any YOLO model candidates. Last error: {last_err}")


def get_car_centers_from_results(results, target_class_names: List[str]) -> List[Tuple[float, float]]:
    """Extract center points of boxes for target class names from a single Results object."""
    centers: List[Tuple[float, float]] = []
    if results is None:
        return centers

    # results.boxes: Boxes object
    boxes = getattr(results, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return centers

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else None

    # Map class indices to names if available; otherwise fall back to COCO index assumption
    names = getattr(results, "names", None)
    if names is None and hasattr(results, "model") and hasattr(results.model, "names"):
        names = results.model.names

    for i, box in enumerate(xyxy):
        if cls is not None and names is not None:
            class_name = names.get(int(cls[i]), str(int(cls[i]))) if isinstance(names, dict) else (
                names[int(cls[i])] if isinstance(names, (list, tuple)) and int(cls[i]) < len(names) else str(int(cls[i]))
            )
            if class_name.lower() not in target_class_names:
                continue

        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        centers.append((cx, cy))
    return centers


def project_points(points: List[Tuple[float, float]], H: np.ndarray) -> np.ndarray:
    """Project points from frame coordinates to twin coordinates using homography."""
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(pts, H)
    proj = proj.reshape(-1, 2)
    return proj


def clamp_points(points: np.ndarray, w: int, h: int) -> np.ndarray:
    """Clamp projected points to be within the twin canvas bounds."""
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)
    return points


def run(
    input_video: str,
    output_video: str,
    model_name: str,
    twin_size: int,
    conf_threshold: float,
    device: str,
    show_preview: bool,
    draw_tracks: bool,
    class_names: List[str],
) -> None:
    # Open video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if np.isnan(fps) or fps <= 0:
        fps = 30.0

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        raise RuntimeError("Could not read the first frame from the video.")

    # Let the user select 4 points
    cv2.setWindowTitle if hasattr(cv2, 'setWindowTitle') else None
    points = select_four_points("Select 4 points (TL, TR, BR, BL)", first_frame)
    if len(points) != 4:
        raise RuntimeError("Point selection cancelled or insufficient points selected.")

    # Compute homography and prepare the twin canvas
    H = compute_homography(points, twin_size)
    if H is None:
        raise RuntimeError("Failed to compute homography from the selected points.")

    base_canvas = draw_twin_canvas(twin_size)

    # Prepare video writer for twin visualization
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (twin_size, twin_size))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video for writing: {output_video}")

    # Load YOLO model
    model = load_yolo(model_name)

    # Warm up (optional) to stabilize first-frame latency
    _ = model.predict(first_frame, conf=conf_threshold, verbose=False, device=device)

    # Reset capture to the first frame we already read
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    trail_points: List[np.ndarray] = []

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        results_list = model.predict(frame, conf=conf_threshold, verbose=False, device=device)
        results = results_list[0] if isinstance(results_list, (list, tuple)) else results_list

        centers = get_car_centers_from_results(results, class_names)
        proj_points = project_points(centers, H)
        proj_points = clamp_points(proj_points, twin_size, twin_size)

        canvas = base_canvas.copy()

        # Draw current frame detections
        for (px, py) in proj_points:
            cv2.circle(canvas, (int(round(px)), int(round(py))), 6, (0, 0, 255), -1)

        # Optionally maintain a simple trail of previous positions for visualization
        if draw_tracks and len(proj_points) > 0:
            trail_points.append(proj_points.copy())
            # Keep last N frames for the trail
            max_trail = 30
            if len(trail_points) > max_trail:
                trail_points = trail_points[-max_trail:]
            # Draw fading trail
            for idx, pts in enumerate(reversed(trail_points)):
                alpha = max(0.2, 1.0 - (idx / max_trail))
                color = (int(0), int(255 * alpha), int(0))
                for (tx, ty) in pts:
                    cv2.circle(canvas, (int(round(tx)), int(round(ty))), 3, color, -1)

        writer.write(canvas)

        if show_preview:
            cv2.imshow("Digital Twin", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to stop
                break

    cap.release()
    writer.release()
    if show_preview:
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project car detections onto a digital twin via homography.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input video file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output twin visualization video (e.g., output_twin.mp4).",
    )
    parser.add_argument(
        "--model",
        default="yolo12x.pt",
        help="YOLO model name or path (defaults to yolo12x.pt, falls back to yolo11x if unavailable).",
    )
    parser.add_argument(
        "--twin-size",
        type=int,
        default=800,
        help="Size (pixels) of the square digital twin canvas.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO detections.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference (e.g., 'cpu', 'cuda:0').",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable preview window during processing.",
    )
    parser.add_argument(
        "--tracks",
        action="store_true",
        help="Draw short trails of previous detections on the twin.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["car"],
        help="Target class names to project (default: car).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        input_video=args.input,
        output_video=args.output,
        model_name=args.model,
        twin_size=args.twin_size,
        conf_threshold=args.conf,
        device=args.device,
        show_preview=not args.no_preview,
        draw_tracks=args.tracks,
        class_names=[name.lower() for name in args.classes],
    )


if __name__ == "__main__":
    main()


