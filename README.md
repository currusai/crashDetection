# crashDetection

A toolkit for detecting vehicle crashes in traffic-camera videos using YOLO-based detection/tracking, motion-vector analysis, and optional learning models. This folder contains:

- Data annotation utilities and generated `*_annotations.csv` files
- Motion-vector exporters and visualization tools
- End-to-end crash detection with interval mode and auto-tuning
- Training scripts for image/sequence models and for motion-vector features
- Parameter-tuning utilities and experiment logs/outputs

## Environment

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

- Weights: place YOLO weights (e.g., `yolo12x.pt`) in this directory or provide a path via `--weights`.

## Typical workflows

- Annotate crash intervals in videos → produce `*_annotations.csv`
- Run detection + tracking and visualize motion vectors
- Export per-frame motion vectors to JSONL
- Train a simple classifier on motion vectors or use image/video models
- Tune parameters and review experiment outputs

## Key scripts

- `annotater.py`: GUI tool to create `*_annotations.csv` for a video.
  - Controls: c = start/stop interval, p = pause, q = quit; click to set crash point; prompts for severity (1–5).
  - Usage:
    ```bash
    python annotater.py "Crashes caught on Seattle traffic cameras 8!.mp4"
    ```

- `vehicle_motion_vectors.py`: End-to-end detection, tracking, motion vectors, and crash scoring.
  - Supports interval-only processing using existing `*_annotations.csv`; auto-tuning of parameters; experiment logging and concatenated interval outputs.
  - Important args (defaults overridable in-code via `SCRIPT_ARGS`):
    - Detection: `--weights`, `--imgsz`, `--conf`, `--iou`, `--device`
    - Motion/visualization: `--history`, `--arrow_scale`, `--arrow_color`, `--thickness`, `--show_ids`
    - Crash scoring: `--crash_threshold`, `--weight_predict`, `--weight_intersect`, `--weight_abrupt`, `--pred_*`, `--intersect_margin`, `--min_pair_speed`
    - Interval/auto-tune: `--interval_mode`, `--glob`, `--buffer_seconds`, `--auto_tune`, `--tune_iters`, `--experiment_root`, `--data_dir`
  - Examples:
    ```bash
    # Process a single video
    python vehicle_motion_vectors.py "Crashes caught on Seattle traffic cameras 9!.mp4" --frame_limit -1 --show_ids --save_path output.mp4

    # Process only annotated intervals across matching videos and auto-tune
    python vehicle_motion_vectors.py any --data_dir . --interval_mode --glob "Crashes*.mp4" --buffer_seconds 1.0 --auto_tune --tune_iters 20 --experiment_root experiments
    ```

- `export_motion_vectors.py`: Export per-frame motion vectors to JSONL for training/analysis.
  - Writes one line per frame: `{video, frame_index, fps, vectors:[{track_id, class_id, start:{x,y}, end:{x,y}}]}`
  - Usage:
    ```bash
    python export_motion_vectors.py "Crashes caught on Seattle traffic cameras 9!.mp4" --out "Crashes caught on Seattle traffic cameras 9!_vectors.jsonl"
    ```

- `digital_twin.py`: Projects detections onto a square "digital twin" via homography and renders a twin video.
  - Interactive: select 4 points on the first frame (TL, TR, BR, BL), then detections are projected onto the twin canvas.
  - Usage:
    ```bash
    python digital_twin.py --input input.mp4 --output twin.mp4 --model yolo12x.pt --twin-size 800 --conf 0.25 --device cpu --tracks
    ```

- `frame_counter.py`: Summarize crash vs normal frames per video based on annotation CSVs.
  - Finds `*_annotations.csv`, reads corresponding video to compute total frames, prints per-video and overall stats.
  - Usage:
    ```bash
    python frame_counter.py
    ```

- `train_fcn_motion_vectors.py`: Train a simple fully-connected model on exported motion vectors.
  - Looks for matching pairs: `*_vectors.jsonl` and `*_annotations.csv` in a data directory (default: this folder).
  - Configure via in-file `CFG` class (data_dir, frames_per_sample, frame_interval, max_vectors_per_frame, batch size, epochs, etc.).
  - Saves `confusion_matrix.png` after training.
  - Usage:
    ```bash
    python train_fcn_motion_vectors.py
    ```

- `train_autoencoder_motion_vectors.py`: Train an encoder–decoder with reconstruction loss on 3-second motion-vector windows and probe embeddings for crash vs normal.
  - Builds 3s windows per video using FPS from a matching `.mp4` (falls back to 30 FPS), labels a window as crash if it overlaps any annotated crash interval.
  - Resamples each window to a fixed `CFG.target_frames` (default 48) to get consistent model input size.
  - Saves model weights (`sequence_autoencoder.pth`) and embeddings (`ae_embeddings.npz`), and prints linear-probe accuracy.
  - Usage:
  ```bash
  python train_autoencoder_motion_vectors.py
  ```

- `train_autoencoder_frames.py`: Train a per-frame convolutional autoencoder on 3-second video windows sampled directly from `.mp4` files.
  - Windows are labeled crash if overlapping any annotated interval from matching `*_annotations.csv`.
  - Each window is resampled to `CFG.target_frames` and frames are resized to `CFG.img_size`.
  - Saves `frame_autoencoder.pth` and `frame_ae_embeddings.npz`; prints linear-probe accuracy.
  - Usage:
  ```bash
  python train_autoencoder_frames.py
  ```

- `model_trainerr.py`: PyTorch classifier training on image sequences (3 frames sampled from 30-frame sequences) with optional severity signal.
  - Expects dataset laid out as `binary_format/{crashed,normal}/*.jpg` and tries to align frames with `*_annotations.csv` to inject severity.
  - Tunables at top of file (e.g., `DATA_DIR`, `ANNOTATIONS_DIR`, `MIN_CRASH_SEVERITY`, augmentation, sampler).

- `timesformer_trainer.py`: TimeSformer-based training on sampled frames from 30-frame sequences.
  - Mirrors logic of `model_trainerr.py` but uses `transformers` TimeSformer checkpoint and sequence sampling (`NUM_FRAMES`).
  - Configure constants at top of file (paths, sampling, augmentation).

- `aug_data.py`: Reserved for data augmentation utilities (currently empty placeholder).

## Data and artifacts

- Videos: multiple `*.mp4` files (Seattle traffic cams and related clips)
- Annotations: `*_annotations.csv` generated by `annotater.py`
- Motion vectors: `*_vectors.jsonl` and optional variants (`*_h264_vectors.jsonl`, motion-vector trial mp4s)
- Experiments: `experiments/exp_*/` contains auto-tuning logs and best outputs
- Parameter tuning: `ParameterTuning/` contains iterhistory files and plotting utility
- Pretrained models: `yolo12x.pt` (YOLO weights)
- Outputs: `confusion_matrix.png`, generated `*_motion_vectors*.mp4`, and experiment videos

## Parameter tuning utilities

- `ParameterTuning/plot_tuning_surfaces.py`: Visualize auto-tuning history (`iterhist.txt` or `.rtf`) as 3D surfaces and export filtered CSV.
  - Usage:
    ```bash
    python ParameterTuning/plot_tuning_surfaces.py --file ParameterTuning/iterhist.txt --x_param pred_horizon --y_param pred_iou_thresh --z_param intersect_margin --out_prefix ParameterTuning/score_surface
    ```

## Notes

- `vehicle_motion_vectors_old.py` is a prior version left for reference.
- `MotionVecs/`, `experiments/`, and similar directories can be large and are excluded in some listings.
- If `ultralytics` doesn’t auto-install `torch` on your platform, ensure `torch` and `torchvision` versions from `requirements.txt` are installed.


Videos Link
https://drive.google.com/drive/folders/1penLiKmmnOA5tkb30wTLzj1JFslUuIlw?usp=sharing