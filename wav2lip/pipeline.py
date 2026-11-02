import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Sequence, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm

from . import audio
from .hub import get_checkpoint_path, get_face_detector_path
from .models.wav2lip import Wav2Lip
from .face_detection import FaceAlignment, LandmarksType


def get_optimal_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_smoothened_boxes(boxes: np.ndarray, T: int = 5) -> np.ndarray:
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


class Wav2LipPipeline:
    """
    High-level, in-memory Wav2Lip lip-sync engine.
    Supports GPU (CUDA), Apple Silicon (MPS), and CPU execution.
    """

    def __init__(
        self,
        checkpoint: Union[str, Path] = "gan",
        device: str | None = None,
        face_detector: str = "sfd",
    ):
        self.device = device or get_optimal_device()

        if isinstance(checkpoint, (str, Path)) and (str(checkpoint) in ["gan", "base"] or not Path(checkpoint).exists()):
            self.checkpoint_path = get_checkpoint_path(str(checkpoint))
        else:
            self.checkpoint_path = Path(checkpoint)

        self.detector_path = get_face_detector_path()
        self.model = self._load_model()
        self.detector = FaceAlignment(
            LandmarksType._2D,
            device=self.device,
            path_to_detector=str(self.detector_path),
            face_detector=face_detector,
        )

    def _load_model(self) -> Wav2Lip:
        map_loc = torch.device(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=map_loc, weights_only=False)

        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model = Wav2Lip()
        model.load_state_dict(new_state_dict)
        return model.to(self.device).eval()

    def _face_detect(self, images: list[np.ndarray], pads: Sequence[int], nosmooth: bool, batch_size: int = 16):
        predictions = []
        cur_bs = batch_size
        while True:
            predictions = []
            try:
                for i in range(0, len(images), cur_bs):
                    batch = np.array(images[i : i + cur_bs])
                    predictions.extend(self.detector.get_detections_for_batch(batch))
            except RuntimeError as err:
                if cur_bs == 1:
                    raise RuntimeError("Image too large for face detection. Try increasing resize_factor.") from err
                cur_bs //= 2
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = pads
        start_idx = 0
        end_idx = len(predictions)

        # Handle boundary frames with missing detections (titles / fade-in / fade-out)
        for idx, rect in enumerate(predictions):
            if rect is not None:
                start_idx = idx
                break

        for idx in range(len(predictions) - 1, -1, -1):
            if predictions[idx] is not None:
                end_idx = idx + 1
                break

        trimmed_images = images[start_idx:end_idx]
        trimmed_predictions = predictions[start_idx:end_idx]

        for rect, image in zip(trimmed_predictions, trimmed_images):
            if rect is None:
                # If face temporarily drops in the middle, use nearest fallback box if available
                if results:
                    results.append(results[-1])
                    continue
                raise ValueError("Face not detected! Please ensure the video contains a clear frontal face.")

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not nosmooth and len(boxes) > 1:
            boxes = get_smoothened_boxes(boxes, T=min(5, len(boxes)))

        face_coords = [
            (img[y1:y2, x1:x2], (y1, y2, x1, x2))
            for img, (x1, y1, x2, y2) in zip(trimmed_images, boxes)
        ]
        return face_coords, trimmed_images

    def generate(
        self,
        face: Union[str, Path],
        audio_path: Union[str, Path],
        outfile: Union[str, Path],
        resize_factor: int = 1,
        pads: Sequence[int] = (0, 10, 0, 0),
        nosmooth: bool = False,
        wav2lip_batch_size: int = 128,
        face_det_batch_size: int = 16,
        fps: float = 25.0,
        box: Sequence[int] = (-1, -1, -1, -1),
        rotate: bool = False,
        crop: Sequence[int] = (0, -1, 0, -1),
    ) -> str:
        face_p = Path(face).resolve()
        audio_p = Path(audio_path).resolve()
        out_p = Path(outfile).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        temp_dir = out_p.parent / f"_temp_w2l_{out_p.stem}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        img_size = 96
        mel_step_size = 16

        # 1. Read input face image or video
        static_image = False
        if face_p.suffix.lower() in [".jpg", ".png", ".jpeg", ".webp", ".bmp"]:
            static_image = True
            frame = cv2.imread(str(face_p))
            if frame is None:
                raise ValueError(f"Could not read face image: {face_p}")
            full_frames = [frame]
            video_fps = fps
        else:
            cap = cv2.VideoCapture(str(face_p))
            if not cap.isOpened():
                raise ValueError(f"Could not open input video: {face_p}")
            video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
            full_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if resize_factor > 1:
                    frame = cv2.resize(
                        frame,
                        (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor),
                    )
                if rotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                y1, y2, x1, x2 = crop
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]
                frame = frame[y1:y2, x1:x2]
                full_frames.append(frame)
            cap.release()

        if not full_frames:
            raise ValueError("No video frames could be read from input video source.")

        # 2. Prepare audio mel spectrogram
        temp_wav = audio_p
        if audio_p.suffix.lower() != ".wav":
            temp_wav = temp_dir / "audio_16k.wav"
            try:
                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-i", str(audio_p),
                    "-ar", "16000", "-ac", "1",
                    str(temp_wav),
                ]
                subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            except FileNotFoundError:
                raise RuntimeError("FFmpeg executable not found. Please install FFmpeg on your system: sudo apt install ffmpeg")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"FFmpeg audio pre-conversion failed: {e.stderr.decode(errors='replace')}")

        wav = audio.load_wav(str(temp_wav), 16000)
        mel = audio.melspectrogram(wav)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError("Audio mel spectrogram contains NaN values.")

        mel_chunks = []
        mel_idx_multiplier = 80.0 / video_fps
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size :])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1

        # 3. Detect face bounding boxes
        total_chunks = len(mel_chunks)
        if box[0] == -1:
            if not static_image:
                # Only detect faces in frames needed for the audio (matching original Wav2Lip behavior)
                if len(full_frames) > total_chunks:
                    full_frames = full_frames[:total_chunks]
                face_det_results, full_frames = self._face_detect(
                    full_frames, pads=pads, nosmooth=nosmooth, batch_size=face_det_batch_size
                )
            else:
                face_det_results, full_frames = self._face_detect(
                    [full_frames[0]], pads=pads, nosmooth=nosmooth, batch_size=1
                )
        else:
            if not static_image and len(full_frames) > total_chunks:
                full_frames = full_frames[:total_chunks]
            y1, y2, x1, x2 = box
            face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in full_frames]

        # 4. Stream and generate lip-synced frames
        raw_avi = temp_dir / "result.avi"
        frame_h, frame_w = full_frames[0].shape[:-1]
        out_writer = cv2.VideoWriter(
            str(raw_avi),
            cv2.VideoWriter_fourcc(*"DIVX"),
            video_fps,
            (frame_w, frame_h),
        )

        try:
            img_batch, mel_batch, frames_batch, coords_batch = [], [], [], []
            total_chunks = len(mel_chunks)

            for i, m in enumerate(tqdm(mel_chunks, desc="Wav2Lip Lip-Sync")):
                idx = 0 if static_image else i % len(full_frames)
                frame_to_save = full_frames[idx].copy()
                face_crop, coords = face_det_results[idx]
                face_resized = cv2.resize(face_crop, (img_size, img_size))

                img_batch.append(face_resized)
                mel_batch.append(m)
                frames_batch.append(frame_to_save)
                coords_batch.append(coords)

                if len(img_batch) >= wav2lip_batch_size or i == total_chunks - 1:
                    img_np = np.asarray(img_batch)
                    mel_np = np.asarray(mel_batch)

                    img_masked = img_np.copy()
                    img_masked[:, img_size // 2 :] = 0

                    img_np = np.concatenate((img_masked, img_np), axis=3) / 255.0
                    mel_np = np.reshape(mel_np, [len(mel_np), mel_np.shape[1], mel_np.shape[2], 1])

                    img_tensor = torch.FloatTensor(np.transpose(img_np, (0, 3, 1, 2))).to(self.device)
                    mel_tensor = torch.FloatTensor(np.transpose(mel_np, (0, 3, 1, 2))).to(self.device)

                    with torch.no_grad():
                        pred = self.model(mel_tensor, img_tensor)

                    pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

                    for p, f, c in zip(pred, frames_batch, coords_batch):
                        cy1, cy2, cx1, cx2 = c
                        p_resized = cv2.resize(p.astype(np.uint8), (cx2 - cx1, cy2 - cy1))
                        f[cy1:cy2, cx1:cx2] = p_resized
                        out_writer.write(f)

                    img_batch, mel_batch, frames_batch, coords_batch = [], [], [], []

            out_writer.release()

            # 5. Mux generated video with the audio using FFmpeg
            mux_cmd = [
                "ffmpeg", "-y",
                "-i", str(temp_wav),
                "-i", str(raw_avi),
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                str(out_p),
            ]
            try:
                subprocess.run(mux_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            except FileNotFoundError:
                import shutil
                shutil.copy(str(raw_avi), str(out_p))
                raise RuntimeError("FFmpeg is required to mux audio into the final MP4. Please install FFmpeg: sudo apt install ffmpeg")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"FFmpeg muxing failed: {e.stderr.decode(errors='replace')}")

        finally:
            out_writer.release()
            # Clean up temporary scratch directory
            for tmp in temp_dir.glob("*"):
                try:
                    tmp.unlink()
                except OSError:
                    pass
            try:
                temp_dir.rmdir()
            except OSError:
                pass

        return str(out_p)
