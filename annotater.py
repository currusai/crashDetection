import cv2
import csv
import os
import sys
import tkinter as tk
from tkinter import simpledialog

def annotate_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        return

    # Get total frame count up front
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    basename = os.path.basename(video_path)
    csv_path = os.path.splitext(video_path)[0] + "_annotations.csv"

    is_new = not os.path.exists(csv_path)
    csv_file = open(csv_path, 'a', newline='')
    writer = csv.writer(csv_file)
    if is_new:
        writer.writerow([
            "video_filename",
            "frame_start_number",
            "frame_end_number",
            "crash_flag",
            "crash_position_x",
            "crash_position_y",
            "crash_severity"
        ])

    # TK root for dialogs
    root = tk.Tk()
    root.withdraw()

    toggle = False           # in crash-annotation mode?
    crash_start = None
    crash_pos = None
    paused = False

    def click_event(event, x, y, flags, param):
        nonlocal crash_pos
        if toggle and event == cv2.EVENT_LBUTTONDOWN:
            crash_pos = (x, y)

    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", click_event)

    print(
        "Controls:\n"
        "  c = start/stop crash annotation interval\n"
        "  p = pause/resume playback\n"
        "  q = quit (annotations already saved)\n"
    )

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        disp = frame.copy()

        # draw crash-point if set
        if toggle and crash_pos is not None:
            cv2.circle(disp, crash_pos, 6, (0, 0, 255), -1)

        # draw playback bar
        h, w = disp.shape[:2]
        bar_h = 8
        # background bar
        cv2.rectangle(disp,
                      (0, h-bar_h),
                      (w, h),
                      (50, 50, 50),
                      thickness=-1)
        # filled progress
        fill_w = int(w * (frame_num / total_frames))
        cv2.rectangle(disp,
                      (0, h-bar_h),
                      (fill_w, h),
                      (0, 200, 0),
                      thickness=-1)

        # overlay frame numbers
        cv2.putText(disp,
                    f"{frame_num}/{total_frames}",
                    (10, h - bar_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

        cv2.imshow("Video", disp)
        key = cv2.waitKey(0 if paused else 30) & 0xFF

        if key == ord('p'):
            paused = not paused
            print("⏸ Paused" if paused else "▶ Resumed")

        elif key == ord('c'):
            if not toggle:
                toggle = True
                crash_start = frame_num
                crash_pos = None
                print(f"[Frame {frame_num}] ▶ Crash interval START – click to set position")
            else:
                toggle = False
                crash_end = frame_num
                if crash_pos is None:
                    print("No crash position set; skipping annotation.")
                else:
                    severity = None
                    while severity is None:
                        severity = simpledialog.askinteger(
                            "Crash Severity",
                            f"Frames {crash_start}–{crash_end}: Enter severity (1–5):",
                            minvalue=1, maxvalue=5
                        )
                    x, y = crash_pos
                    writer.writerow([
                        basename,
                        crash_start,
                        crash_end,
                        True,
                        x, y,
                        severity
                    ])
                    csv_file.flush()
                    print(
                        f"✔ Annotated frames {crash_start}–{crash_end}, "
                        f"pos=({x},{y}), sev={severity}"
                    )
                crash_pos = None

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"\nDone — annotations saved to:\n  {csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python annotate.py path/to/video.mp4")
    else:
        annotate_video(sys.argv[1])