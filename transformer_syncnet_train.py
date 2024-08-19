from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import TransformerSyncnet as TransformerSyncnet
import audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list

from models.conv import Conv2d, Conv2dTranspose

# import module 
import traceback
import wandb
import time
import multiprocessing
import logging

from PIL import Image

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)
parser.add_argument('--use_cosine_loss', help='Whether to use cosine loss', default=True, type=str2bool)
parser.add_argument('--sample_mode', help='easy or random', default=True, type=str)

args = parser.parse_args()

# Define lip landmark indices according to MediaPipe documentation
LIP_LANDMARKS = list(range(61, 79)) + list(range(191, 209))

global_step = 1
global_epoch = 1
use_cuda = torch.cuda.is_available()
use_cosine_loss=True
sample_mode='random'
face_image_cache = multiprocessing.Manager().dict()
file_exist_cache = multiprocessing.Manager().dict()
orig_mel_cache = multiprocessing.Manager().dict()

current_training_loss = 0.6
learning_step_loss_threshhold = 0.3
consecutive_threshold_count = 0
samples = [True, True,True, True,True, True,True, False, False, False]

print('use_cuda: {}'.format(use_cuda))

"""
The FPS is set to 25 for video, 5/25 is 0.2, we need to have 0.2 seconds for the audio,
because the audio mel spectrogram ususlly has 80 frame per seconds, so 16/80 is 0.2 seconds
"""
syncnet_T = 5
syncnet_mel_step_size = 16

# Initialize the MediaPipe FaceMesh model



class Dataset(object):
    
    def __init__(self, split, use_image_cache):
        self.all_videos = get_image_list(args.data_root, split)  
        

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            
            if not frame in file_exist_cache:
              if not isfile(frame):
                return None    
            
            
            file_exist_cache[frame] = True
            window_fnames.append(frame)
        return window_fnames
    

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)

        """
        80. is a scaling factor used to convert the time in seconds to the index in the audio spectrogram.
        This scaling factor is related to how the audio spectrogram is calculated and the time resolution of the spectrogram.
        For instance, if the spectrogram has a time resolution of 12.5 ms per frame (which is typical for many audio processing tasks), 
        80 frames per second would correspond to 1.25 seconds. This means the spectrogram has a higher temporal resolution than the video.
        """
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]


    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        """
        Randomly select a video and corresponding images.
        Randomly choose a correct or incorrect image pair.
        Read and preprocess the images and audio data.
        Handle exceptions and retries in case of read errors.
        Return the processed image data, audio features, and label.
        """
        should_load_diff_video = False
        while 1:
                if should_load_diff_video:
                    idx = random.randint(0, len(self.all_videos) - 1)
                    should_load_diff_video = False

                vidname = self.all_videos[idx]
                img_names = list(glob(join(vidname, '*.jpg')))
                
                if len(img_names) <= 3 * syncnet_T:
                    should_load_diff_video = True
                    print('The video has not enough frames, it only has {0}, will retry with a differnt video'.format(len(img_names)))
                    continue
                
                img_name = random.choice(img_names)
                correct_window_images = self.get_window(img_name)
                while correct_window_images is None:
                  img_name = random.choice(img_names)
                  correct_window_images = self.get_window(img_name)

                chosen_id = self.get_frame_id(img_name)

                wrong_img_name = random.choice(img_names)          
                wrong_img_id = self.get_frame_id(wrong_img_name)
                wrong_window_images = self.get_window(wrong_img_name)
                
                """
                Changed by eddy, the following are the original codes, it uses random to get the wrong_img_name, 
                this might get an image that very close to the correct image(the next frame) which is a bit hard to learn.
                Eddy introduced a new algorithm that to get a image a bit futher from the img_name to have enough difference,
                this might help the model to converge.
                """
                attempt = 0
                while wrong_img_name == img_name or abs(wrong_img_id - chosen_id) < 5 or wrong_window_images is None:
                      #print('The selected wrong image {0} is not far engough from {1}, diff {2}, window is None {3}'.format(wrong_img_id, chosen_id, abs(wrong_img_id - chosen_id), wrong_window_images is None))
                      wrong_img_name = random.choice(img_names)
                      wrong_img_id = self.get_frame_id(wrong_img_name)
                      wrong_window_images = self.get_window(wrong_img_name)
                      attempt += 1
                      if attempt > 5:
                          should_load_diff_video = True
                          break
                
                if should_load_diff_video:
                    continue

                
                # We firstly to learn all the positive, once it reach the loss of less than 0.2, we incrementally add some negative samples 10% per step
                good_or_bad = True
                good_or_bad = random.choice(samples)

                if good_or_bad:
                    y = 1
                    window_fnames = correct_window_images
                else:
                    y = 0
                    window_fnames = wrong_window_images
                
                
                face_window = []

                all_read = True
                for fname in window_fnames:
                    if fname in face_image_cache:
                        img = face_image_cache[fname]
                    else:
                        img = cv2.imread(fname)
                        if img is None:
                            all_read = False
                            break
                        try:
                            img = cv2.resize(img, (hparams.img_size, hparams.img_size))                            
                            
                            if len(face_image_cache) < 350000:
                              face_image_cache[fname] = img  # Cache the resized image
                            
                        except Exception as e:
                            all_read = False
                            break
                    
                    '''
                    Data augmentation
                    0 means unchange
                    1 for grayscale
                    2 for brightness
                    3 for contrast
                    '''
                    option = random.choices([0, 1, 2, 3]) 
                    if option == 1:
                        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = cv2.merge([img_gray, img_gray, img_gray])
                    elif option == 2:
                        brightness_factor = np.random.uniform(0.7, 1.3)
                        img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
                    elif option == 3:
                        contrast_factor = np.random.uniform(0.7, 1.3)
                        img = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=0)

                    face_window.append(img)

                if not all_read: continue

                try:
                    wavpath = join(vidname, "audio.wav")

                    if wavpath in orig_mel_cache:
                        orig_mel = orig_mel_cache[wavpath]
                        #print('The audio cache hit ', wavpath)
                    else:
                        wav = audio.load_wav(wavpath, hparams.sample_rate)
                        orig_mel = audio.melspectrogram(wav).T
                        orig_mel_cache[wavpath] = orig_mel
                    
                except Exception as e:
                    should_load_diff_video = True
                    print('The audio is invalid, file name {0}, will retry with a differnt video'.format(join(vidname, "audio.wav")))
                    continue

                mel = self.crop_audio_window(orig_mel.copy(), img_name)

                if (mel.shape[0] != syncnet_mel_step_size):
                    should_load_diff_video = True
                    print("This specific audio is invalid {0}".format(join(vidname, "audio.wav")))
                    continue
                
                # Save the sample images
                # if idx % 100 == 0:
                #   print('The video is ', vidname)
                #   for i, img in enumerate(window):
                #         img_to_save = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                #         img_to_save.save(f'temp1/saved_image_{idx}_{i}.png')

                # H x W x 3 * T
                x = np.concatenate(face_window, axis=2) / 255.
                x = x.transpose(2, 0, 1)
                x = x[:, x.shape[1]//2:]

                x = torch.FloatTensor(x)
                mel = torch.FloatTensor(mel.T).unsqueeze(0)

                return x, mel, y


cross_entropy_loss = nn.CrossEntropyLoss()

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)

    target_tensor = y.float()

    loss = logloss(d, target_tensor)

    return loss

def get_lip_landmark(image, face_mesh):
  try:
    
    # Load an image (make sure the image path is correct)
    height, width, _ = image.shape

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to extract face landmarks
    results = face_mesh.process(image_rgb)

        
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Create an empty list to store the lip landmarks
            lip_embeddings = []
            
            # Iterate through the specific lip landmarks
            for idx in LIP_LANDMARKS:
                landmark = face_landmarks.landmark[idx]
                # Extract the x, y, z coordinates
                x = landmark.x
                y = landmark.y
                z = landmark.z
                # Append the coordinates to the embeddings list
                lip_embeddings.append([x, y, z])
            
            # Convert the list to a numpy array (optional, for easier manipulation)
            lip_embeddings = np.array(lip_embeddings)
            break

    # Normalize the embedding (optional but recommended)
    # This normalization ensures that all coordinates are on the same scale (between 0 and 1)
    lip_embeddings = lip_embeddings / np.linalg.norm(lip_embeddings)

    # Print the embedding
    #print("Lip Embedding:", lip_embedding)

    
    return lip_embeddings
  except Exception as e:
    print('error', e)
    traceback.print_exc() 
    return None

# added by eddy
# Register hooks to print gradient norms
def print_grad_norm(module, grad_input, grad_output):
    for i, grad in enumerate(grad_output):
        if grad is not None and global_step % hparams.syncnet_eval_interval == 0:
            print(f'{module.__class__.__name__} - grad_output[{i}] norm: {grad.norm().item()}')

# end added by eddy


def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None, should_print_grad_norm=False):

    
    global global_step, global_epoch, consecutive_threshold_count, current_training_loss
    resumed_step = global_step
    print('start training data folder', train_data_loader)
    patience = 100

    current_lr = get_current_lr(optimizer)
    print('The learning rate is: {0}'.format(current_lr))

    # Added by eddy
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=patience, verbose=True)
    
    if should_print_grad_norm:
      for name, module in model.named_modules():
        if isinstance(module, (Conv2d, Conv2dTranspose, nn.Linear)):
            module.register_backward_hook(print_grad_norm)
    
    # end

    

    while global_epoch < nepochs:
        running_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        current_lr = get_current_lr(optimizer)
        for step, (x, mel, y) in prog_bar:
            
            model.train()
            optimizer.zero_grad()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            output, audio_embedding, face_embedding = model(x, mel)
            
            y = y.to(device)                        
            
            ce_loss = cross_entropy_loss(output, y) #if (global_epoch // 50) % 2 == 0 else contrastive_loss2(a, v, y)
            cos_loss = cosine_loss(audio_embedding, face_embedding, y)
            loss = 0.5 * ce_loss + 0.5 * cos_loss
            loss.backward()
            optimizer.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, global_step, device, model, checkpoint_dir, scheduler)
                
            current_training_loss = running_loss / (step + 1)
            prog_bar.set_description('Global Step: {0}, Epoch: {1}, Loss: {2}, current learning rate: {3}'.format(global_step, global_epoch, current_training_loss, current_lr))
            metrics = {"train/train_loss": current_training_loss, 
                       "train/step": global_step, 
                       "train/epoch": global_epoch,
                       "train/learning_rate": current_lr}
            
            wandb.log({**metrics})

        if current_training_loss < 0.25:
          consecutive_threshold_count += 1
        else:
          consecutive_threshold_count = 0
            
        if consecutive_threshold_count >= 10:
          false_count = samples.count(False)
          if false_count < 5:
            # Find the index of the first occurrence of True
            first_true_index = samples.index(True)
            # Change the element at that index to False
            samples[first_true_index] = False

            print('Adding negative samples, the current samples are', samples)
                
            
        global_epoch += 1
        # if should_print_grad_norm or global_step % 20==0:
        #   for param in model.parameters():
        #         if param.grad is not None:
        #             print('The gradient is ', param.grad.norm())
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        

# Added by eddy
def get_current_lr(optimizer):
    # Assuming there is only one parameter group
    for param_group in optimizer.param_groups:
        return param_group['lr']


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir, scheduler):
    #eval_steps = 1400
    eval_steps = 20
    eval_loop = 10
    current_step = 1


    print('Evaluating for {0} steps of total steps {1}'.format(eval_steps, len(test_data_loader)))
    prog_bar = tqdm(enumerate(test_data_loader))
    losses = []
    while 1:
        for step, (x, mel, y) in enumerate(test_data_loader):

            model.eval()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            output, audio_embedding, face_embedding = model(x, mel)
            y = y.to(device)                

            ce_loss = cross_entropy_loss(output, y) #if (global_epoch // 50) % 2 == 0 else contrastive_loss2(a, v, y)
            cos_loss = cosine_loss(audio_embedding, face_embedding, y)
            loss = 0.5 * ce_loss + 0.5 * cos_loss

            losses.append(loss.item())

            if step > eval_steps: break

        averaged_loss = sum(losses) / len(losses)
        
        prog_bar.set_description('Step: {0}/{1}, Loss: {2}'.format(current_step, eval_loop, averaged_loss))

        metrics = {"val/loss": averaged_loss, 
                    "val/step": global_step, 
                    "val/epoch": global_epoch}
            
        wandb.log({**metrics})
        # Scheduler step with average training loss
        scheduler.step(averaged_loss)
        current_step += 1
        if current_step > eval_loop: 
            return

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):

    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    # Reset the new learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.0001

    return model

if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path
    sample_mode = args.sample_mode

    wandb.init(
      # set the wandb project where this run will be logged
      project="my-wav2lip-syncnet",

      # track hyperparameters and run metadata
      config={
      "learning_rate": hparams.syncnet_lr,
      "architecture": "TransformerSyncnet",
      "dataset": "MyOwn",
      "epochs": 200000,
      }
    )

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    # Dataset and Dataloader setup
    train_dataset = Dataset('train', False)
    test_dataset = Dataset('val', False)
    #print(train_dataset.all_videos)

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=False,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=8)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = TransformerSyncnet(embed_size=256, num_heads=8, num_encoder_layers=6).to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr,betas=(0.5, 0.999), weight_decay=1e-5)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs, should_print_grad_norm=False)
