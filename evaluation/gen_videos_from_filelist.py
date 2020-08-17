from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse
import dlib, json, subprocess
from tqdm import tqdm
from glob import glob
import torch

sys.path.append('../')
import audio
import face_detection
from models import Wav2Lip

parser = argparse.ArgumentParser(description='Code to generate results for test filelists')

parser.add_argument('--filelist', type=str, 
					help='Filepath of filelist file to read', required=True)
parser.add_argument('--results_dir', type=str, help='Folder to save all results into', 
									required=True)
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--checkpoint_path', type=str, 
					help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 0, 0, 0], 
					help='Padding (top, bottom, left, right)')
parser.add_argument('--face_det_batch_size', type=int, 
					help='Single GPU batch size for face detection', default=64)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip', default=128)

# parser.add_argument('--resize_factor', default=1, type=int)

args = parser.parse_args()
args.img_size = 96

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images):
	batch_size = args.face_det_batch_size
	
	while 1:
		predictions = []
		try:
			for i in range(0, len(images), batch_size):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1:
				raise RuntimeError('Image too big to run face detection on GPU')
			batch_size //= 2
			args.face_det_batch_size = batch_size
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			raise ValueError('Face not detected!')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = get_smoothened_boxes(np.array(results), T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2), True] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	return results 

def datagen(frames, face_det_results, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	for i, m in enumerate(mels):
		if i >= len(frames): raise ValueError('Equal or less lengths only')

		frame_to_save = frames[i].copy()
		face, coords, valid_frame = face_det_results[i].copy()
		if not valid_frame:
			continue

		face = cv2.resize(face, (args.img_size, args.img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

fps = 25
mel_step_size = 16
mel_idx_multiplier = 80./fps
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

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

model = load_model(args.checkpoint_path)

def main():
	assert args.data_root is not None
	data_root = args.data_root

	if not os.path.isdir(args.results_dir): os.makedirs(args.results_dir)

	with open(args.filelist, 'r') as filelist:
		lines = filelist.readlines()

	for idx, line in enumerate(tqdm(lines)):
		audio_src, video = line.strip().split()

		audio_src = os.path.join(data_root, audio_src) + '.mp4'
		video = os.path.join(data_root, video) + '.mp4'

		command = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'.format(audio_src, '../temp/temp.wav')
		subprocess.call(command, shell=True)
		temp_audio = '../temp/temp.wav'

		wav = audio.load_wav(temp_audio, 16000)
		mel = audio.melspectrogram(wav)
		if np.isnan(mel.reshape(-1)).sum() > 0:
			continue

		mel_chunks = []
		i = 0
		while 1:
			start_idx = int(i * mel_idx_multiplier)
			if start_idx + mel_step_size > len(mel[0]):
				break
			mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
			i += 1

		video_stream = cv2.VideoCapture(video)
			
		full_frames = []
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading or len(full_frames) > len(mel_chunks):
				video_stream.release()
				break
			full_frames.append(frame)

		if len(full_frames) < len(mel_chunks):
			continue

		full_frames = full_frames[:len(mel_chunks)]

		try:
			face_det_results = face_detect(full_frames.copy())
		except ValueError as e:
			continue

		batch_size = args.wav2lip_batch_size
		gen = datagen(full_frames.copy(), face_det_results, mel_chunks)

		for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
			if i == 0:
				frame_h, frame_w = full_frames[0].shape[:-1]
				out = cv2.VideoWriter('../temp/result.avi', 
								cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

			img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
			mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

			with torch.no_grad():
				pred = model(mel_batch, img_batch)
					

			pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
			
			for pl, f, c in zip(pred, frames, coords):
				y1, y2, x1, x2 = c
				pl = cv2.resize(pl.astype(np.uint8), (x2 - x1, y2 - y1))
				f[y1:y2, x1:x2] = pl
				out.write(f)

		out.release()

		vid = os.path.join(args.results_dir, '{}.mp4'.format(idx))

		command = 'ffmpeg -loglevel panic -y -i {} -i {} -strict -2 -q:v 1 {}'.format(temp_audio, 
								'../temp/result.avi', vid)
		subprocess.call(command, shell=True)

if __name__ == '__main__':
	main()
