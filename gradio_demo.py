import os
from os import listdir, path
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform
import gradio as gr

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

model_paths = {
    "wav2lip": "./checkpoints/wav2lip.pth",
    "wav2lip_gan": "./checkpoints/wav2lip_gan.pth"
}
# Dictionary to store loaded models
loaded_models = {}
current_model_name = None

detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

# Default parameters
mel_step_size = 16
img_size = 96
wav2lip_batch_size = 128
face_det_batch_size = 16
pads = [0, 10, 0, 0]
nosmooth = True # or False
static = False
resize_factor = 1
crop = [0, -1, 0, -1]
box = [-1, -1, -1, -1]
rotate = False

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images, nosmooth=True):
    batch_size = face_det_batch_size

    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size), desc="face_dection"):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise gr.Error('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = pads
    last_valid_rect = None
    for i, (rect, image) in enumerate(zip(predictions, images)):
        if rect is None:
            if last_valid_rect is None:
                cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
                raise gr.Error('Face not detected! Ensure the video contains a face in all the frames.')
            else:
                rect = last_valid_rect
                print(f'Face not detected in frame {i}. Using previous detection.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    return results 

def datagen(frames, mels, nosmooth=True):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if box[0] == -1:
		if not static:
			face_det_results = face_detect(frames, nosmooth) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect([frames[0]], nosmooth)
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

	for i, m in enumerate(mels):
		idx = 0 if static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		face = cv2.resize(face, (img_size, img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

def cut_and_rotate_video(input_path, output_path, start_time, end_time, angle=0):
    rotation_filter = ""
    if angle == 90:
        rotation_filter = ",transpose=1"  # 90도 시계 방향
    elif angle == 180:
        rotation_filter = ",transpose=2,transpose=2"  # 180도
    elif angle == 270:
        rotation_filter = ",transpose=2"  # 90도 반시계 방향
    
    cmd = [
        'ffmpeg', '-i', input_path, 
        '-ss', str(start_time), '-to', str(end_time),
        '-vf', f'format=yuv420p{rotation_filter}',
        '-c:v', 'libx264', '-c:a', 'aac', 
        '-movflags', '+faststart', output_path
    ]
    
    subprocess.run(cmd)
    print(f"Trimmed and rotated video saved to {output_path}")
    return output_path


def lip_sync(face_file, audio_file, model_choice, nosmooth=True):
    global current_model_name, loaded_models

    # 임시 디렉토리 생성
    os.makedirs('temp', exist_ok=True)
    
    # 결과 저장 경로
    temp_output_path = 'temp/result_output.mp4'
    
    # 입력 파일 확인
    if face_file is None or audio_file is None:
        return None
    
    # 모델 로드 (필요한 경우에만)
    if model_choice != current_model_name or model_choice not in loaded_models:
        print(f"Loading {model_choice} model...")
        model_path = model_paths[model_choice]
        loaded_models[model_choice] = load_model(model_path)
        current_model_name = model_choice
    
    model = loaded_models[model_choice]
    
    # 이미지 또는 비디오 파일 처리
    is_image = face_file.name.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']
    
    if is_image:
        full_frames = [cv2.imread(face_file.name)]
        fps = 25.0  # 기본 FPS
    else:
        video_stream = cv2.VideoCapture(face_file.name)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//resize_factor, frame.shape[0]//resize_factor))

            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            y1, y2, x1, x2 = crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)
    
    print("Number of frames available for inference: "+str(len(full_frames)))
    
    # 오디오 추출
    temp_audio_path = 'temp/temp.wav'
    if not audio_file.endswith('.wav'):
        print('Extracting raw audio...')
        command = 'ffmpeg -y -i "{}" -strict -2 "{}"'.format(audio_file, temp_audio_path)
        subprocess.call(command, shell=True)
        audio_path = temp_audio_path
    else:
        audio_path = audio_file
    
    # 오디오 처리
    wav = audio.load_wav(audio_path, 16000)
    audio_sec = wav.shape[-1] / 16000
    mel = audio.melspectrogram(wav)
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise gr.Error('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80./fps 
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[:len(mel_chunks)]

    batch_size = wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks, nosmooth)

    temp_video = 'temp/result.avi'
    
    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                        total=int(np.ceil(float(len(mel_chunks))/batch_size)), desc="wav2lip")):
        if i == 0:
            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter(temp_video, 
                                  cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()

    # 최종 결과 생성
    command = 'ffmpeg -y -i "{}" -i "{}" -strict -2 -q:v 1 "{}"'.format(audio_path, temp_video, temp_output_path)
    subprocess.call(command, shell=platform.system() != 'Windows')
    
    return temp_output_path

iface = gr.Interface(
    fn=lip_sync,
    inputs=[
        gr.File(label="Face Video/Image", type="filepath"),
        gr.Audio(label="Audio File", type="filepath"),
        gr.Radio(
            ["wav2lip", "wav2lip_gan"], 
            label="Model Selection", 
            value="wav2lip",
            info="wav2lip: Better lip sync, wav2lip_gan: Better image quality"
        ),
        gr.Checkbox(label="No Smooth (Enable for sharper face detection, disable for smoother transitions)", value=True)
    ],
    outputs=gr.Video(label="Lip-synced Video"),  # You can customize the output as needed
	title="Wav2Lip - Lip Sync Demo",
    description="Upload a face video/image and an audio file to generate a lip-synced video.",
	allow_flagging="never",
    examples=[
          ["/data/donggukang/Wav2Lip/영어/inside_farewell_positive_02.mp4", "/data/donggukang/Wav2Lip/gradio_audio_sample.wav"],
	]
)


if __name__ == '__main__':
	iface.launch(share=True)
