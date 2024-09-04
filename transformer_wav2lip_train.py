from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import TransformerSyncnet as SyncNet
from models import Wav2Lip as Wav2Lip
import audio

import torch

import wandb

from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
import torchvision.models as models


from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list

from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.conv import Conv2d, Conv2dTranspose
import time
import multiprocessing
from torch.nn import functional as F


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model without the visual quality discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True, type=str)

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--syncnet_checkpoint_path', help='Load the pre-trained Expert discriminator', required=True, type=str)

parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default=None, type=str)
parser.add_argument('--use_cosine_loss', help='Whether to use cosine loss', default=True, type=str2bool)

args = parser.parse_args()


global_step = 0
global_epoch = 0
use_cosine_loss=True
use_cuda = torch.cuda.is_available()
image_cache = multiprocessing.Manager().dict()
orig_mel_cache = multiprocessing.Manager().dict()

print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16

class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
                #print('The image name ', fname)
                if fname in image_cache:
                    img = image_cache[fname]
                    #print('The image cache hit ', fname)
                else:
                    img = cv2.imread(fname)
                    if img is None:
                        break
                    try:
                        img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                        if len(image_cache) < hparams.image_cache_size:
                          image_cache[fname] = img  # Cache the resized image and prevent OOM
                    
                        
                    except Exception as e:
                        break
                    
                    '''
                    Data augmentation
                    0 means unchange
                    1 for grayscale
                    2 for brightness
                    3 for contrast
                    '''
                    option = random.choices([0, 1, 2, 3, 4])[0] 
                    
                    if option == 1:
                        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = cv2.merge([img_gray, img_gray, img_gray])
                    elif option == 2:
                        brightness_factor = np.random.uniform(0.7, 1.3)
                        img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
                    elif option == 3:
                        contrast_factor = np.random.uniform(0.7, 1.3)
                        img = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=0)
                    elif option == 4:
                        angle = np.random.uniform(-15, 15)  # Random angle between -15 and 15 degrees

                        # Get the image dimensions
                        (h, w) = img.shape[:2]

                        # Calculate the center of the image
                        center = (w // 2, h // 2)

                        # Get the rotation matrix
                        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

                        # Perform the rotation
                        img = cv2.warpAffine(img, rotation_matrix, (w, h))

                window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        """
        3 x T x H x W
        Normalization: The pixel values of the images are divided by 255 to normalize them from a range of [0, 255] to [0, 1]. 
        This is a common preprocessing step for image data in machine learning to help the model converge faster during training.
        """
        x = np.asarray(window) / 255.

        """
        Transposition: The method transposes the dimensions of the array using np.transpose(x, (3, 0, 1, 2)).
        The original shape of x is assumed to be (T, H, W, C) where:
        T is the number of images (time steps if treating images as a sequence).
        H is the height of the images.
        W is the width of the images.
        C is the number of color channels (typically 3 for RGB images).
        The transposition changes the shape to (C, T, H, W) which means:
        C (number of channels) comes first.
        T (number of images) comes second.
        H (height of images) comes third.
        W (width of images) comes fourth.
        """
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        #start_time = time.perf_counter()
        
        should_load_diff_video = False

        while 1:
            if should_load_diff_video:
                idx = random.randint(0, len(self.all_videos) - 1)
                should_load_diff_video = False

            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, '*.jpg')))
            
            if len(img_names) <= 3 * syncnet_T:
                print('The length', len(img_names))
                should_load_diff_video = True
            

            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                should_load_diff_video = True
                continue

            window = self.read_window(window_fnames)
            if window is None:
                should_load_diff_video = True
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                should_load_diff_video = True
                continue

            try:
                wavpath = join(vidname, "audio.wav")

                if wavpath in orig_mel_cache:
                    orig_mel = orig_mel_cache[wavpath]
                    #print('The audio cache hit ', wavpath)
                else:
                    wav = audio.load_wav(wavpath, hparams.sample_rate)
                    orig_mel = audio.melspectrogram(wav).T
                    orig_mel_cache[wavpath] = orig_mel

                mel = self.crop_audio_window(orig_mel.copy(), img_name)
                
                if (mel.shape[0] != syncnet_mel_step_size):
                    continue

                indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
                if indiv_mels is None: continue

                window = self.prepare_window(window)
                y = window.copy()


                '''
                Set the second half of the images to be black, the window has 5 images
                The wrong_window contains images that do not align with the audio
                x contains 10 images, the first 5 are the correct iamges with second half black out, the last 5 are the incorrect images to the audio
                indiv_mels contains the corresponding audio for the given window
                y is the window that without the second half black out
                '''

                
                window[:, :, window.shape[2]//2:] = 0.

                wrong_window = self.prepare_window(wrong_window)

                x = np.concatenate([window, wrong_window], axis=0)

                x = torch.FloatTensor(x)
                mel = torch.FloatTensor(mel.T).unsqueeze(0)

                indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)

                y = torch.FloatTensor(y)

                return x, indiv_mels, mel, y

            except Exception as e:
                print('An error has occured', vidname, img_name, wrong_img_name)
                print(e)
                continue

def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    '''
    refs: Reference images (extracted from the input x with channels 3 onward).
    inps: Input images (extracted from the input x with the first 3 channels).
    g: Generated images by the model.
    gt: Ground truth images.
    '''
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    
    # Scale cosine similarity to range [0, 1]
    cos_sim_scaled = (1 + d) / 2.0
    
    # Calculate the loss: the target is 1 for similar pairs and 0 for dissimilar pairs
    loss = nn.functional.mse_loss(cos_sim_scaled, y.float())
    
    return loss

def contrastive_loss(a, v, y, margin=0.5):
    """
    Contrastive loss tries to minimize the distance between similar pairs and maximize the distance between dissimilar pairs up to a margin.
    """
    d = nn.functional.pairwise_distance(a, v)
    loss = torch.mean((1 - y) * torch.pow(d, 2) + y * torch.pow(torch.clamp(margin - d, min=0.0), 2))
    return loss

device = torch.device("cuda" if use_cuda else "cpu")
syncnet = SyncNet(num_heads=8, num_encoder_layers=6).to(device)
for p in syncnet.parameters():
    p.requires_grad = False


cross_entropy_loss = nn.CrossEntropyLoss()
recon_loss = nn.L1Loss()
def get_sync_loss(mel, g):
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    output, audio_embedding, face_embedding = syncnet(g, mel)

    y = torch.ones(g.size(0), dtype=torch.long).squeeze().to(device)
    
    return cross_entropy_loss(output, y)

def perceptual_loss(gen_features, gt_features):
    return nn.functional.mse_loss(gen_features, gt_features)

def print_grad_norm(module, grad_input, grad_output):
    for i, grad in enumerate(grad_output):
        if grad is not None and global_step % 1000 == 0:
            print(f'{module.__class__.__name__} - grad_output[{i}] norm: {grad.norm().item()}')

# Added by eddy
def get_current_lr(optimizer):
    # Assuming there is only one parameter group
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None, should_print_grad_norm=False):

    global global_step, global_epoch
    resumed_step = global_step

    patience = 50

    current_lr = get_current_lr(optimizer)
    print('The learning rate is: {0}'.format(current_lr))

    # Added by eddy
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=patience, verbose=True)

    if should_print_grad_norm:
      for name, module in model.named_modules():
        if isinstance(module, (Conv2d, Conv2dTranspose, nn.Linear)):
            module.register_backward_hook(print_grad_norm)

    
    eval_loss = 0.0

    while global_epoch < nepochs:
        current_lr = get_current_lr(optimizer)
        #print('Starting Epoch: {}'.format(global_epoch))
        running_sync_loss, running_l1_loss, running_l2_loss = 0., 0., 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        running_img_loss = 0.0
        for step, (x, indiv_mels, mel, gt) in prog_bar:
            #print("The batch size", x.shape)
            if x.shape[0] == hparams.batch_size:
              model.train()
              optimizer.zero_grad()

              # Move data to CUDA device
              x = x.to(device)
              mel = mel.to(device)
              indiv_mels = indiv_mels.to(device)
              gt = gt.to(device)

              g = model(indiv_mels, x)

              # Get the second half of the images and calculate the loss
              lower_half1 = g[:, :, :, 96:, :]
              lower_half2 = gt[:, :, :, 96:, :]
              lower_half_l1_loss = F.l1_loss(lower_half1, lower_half2)

              if hparams.syncnet_wt > 0.:
                  sync_loss = get_sync_loss(mel, g)
              else:
                  sync_loss = 0.

              l1loss = recon_loss(g, gt)

              l2loss = nn.functional.mse_loss(g, gt)

              running_l1_loss += l1loss.item()
              running_l2_loss += l2loss.item()

              '''
              If the syncnet_wt is 0.03, it means the sync_loss has 3% of the loss wheras the rest occupy 97% of the loss
              '''

              #l1l2_loss = 0.8 * l1loss + 0.2 * l2loss
              loss = hparams.syncnet_wt * sync_loss + (1 - hparams.syncnet_wt) * l1loss
              
              l1loss.backward()
              optimizer.step()

              if global_step % checkpoint_interval == 0:
                  save_sample_images(x, g, gt, global_step, checkpoint_dir)

              global_step += 1

              running_img_loss += loss.item()

              if hparams.syncnet_wt > 0.:
                  running_sync_loss += sync_loss.item()
              else:
                  running_sync_loss += 0.

              if global_step == 1 or global_step % checkpoint_interval == 0:
                  save_checkpoint(
                      model, optimizer, global_step, checkpoint_dir, global_epoch)

              avg_img_loss = (running_img_loss) / (step + 1)

              avg_l1_loss = running_l1_loss / (step + 1)

              avg_l2_loss = running_l2_loss / (step + 1)
              
              if global_step % hparams.eval_interval == 0:
                with torch.no_grad():
                  eval_loss = eval_model(test_data_loader, global_step, device, model, checkpoint_dir, scheduler, 20)

                  #if average_sync_loss < .75:
                  if avg_img_loss < .01: # change 
                          hparams.set_hparam('syncnet_wt', 0.01) # without image GAN a lesser weight is sufficient

              prog_bar.set_description('Step: {}, Img Loss: {}, Sync Loss: {}, Lower Half Loss: {}, L1: {}, L2: {}, LR: {}'.format(global_step, avg_img_loss,
                                                                      running_sync_loss / (step + 1), lower_half_l1_loss.item(), avg_l1_loss, avg_l2_loss, current_lr))
              
              metrics = {
                  "train/overall_loss": avg_img_loss, 
                  "train/avg_l1": avg_l1_loss, 
                  "train/avg_l2": avg_l2_loss, 
                  "train/sync_loss": running_sync_loss / (step + 1), 
                  "train/step": global_step,
                  "train/lower_half_loss": lower_half_l1_loss.item(),
                  "train/learning_rate": current_lr
                  }
            
              wandb.log({**metrics})

        global_epoch += 1


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir, scheduler, eval_steps = 100):
    print('Evaluating for {} steps'.format(eval_steps))
    sync_losses, recon_losses = [], []
    step = 0
    while 1:
        for x, indiv_mels, mel, gt in test_data_loader:
            if x.shape[0] == hparams.batch_size:
              step += 1
              model.eval()

              # Move data to CUDA device
              x = x.to(device)
              gt = gt.to(device)
              indiv_mels = indiv_mels.to(device)
              mel = mel.to(device)

              g = model(indiv_mels, x)

              sync_loss = get_sync_loss(mel, g)
              
              l1loss = recon_loss(g, gt)

              sync_losses.append(sync_loss.item())
              recon_losses.append(l1loss.item())

              averaged_sync_loss = sum(sync_losses) / len(sync_losses)
              averaged_recon_loss = sum(recon_losses) / len(recon_losses)

              print('Eval Loss, L1: {}, Sync loss: {}'.format(averaged_recon_loss, averaged_sync_loss))

              metrics = {"val/l1_loss": averaged_recon_loss, 
                       "val/sync_loss": averaged_sync_loss, 
                       "val/epoch": global_epoch,
                       }
            
              wandb.log({**metrics})

              scheduler.step(averaged_sync_loss + averaged_recon_loss)

              if step > eval_steps: 
                return averaged_sync_loss
            

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

def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s, strict=False)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    if optimizer != None:
      for param_group in optimizer.param_groups:
        param_group['lr'] = 0.00002

    return model

if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir
    use_cosine_loss = args.use_cosine_loss

    # Dataset and Dataloader setup
    train_dataset = Dataset('train')
    test_dataset = Dataset('val')

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.batch_size,
        num_workers=4)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    #model = Wav2Lip(embed_size=256, num_heads=8, num_encoder_layers=6).to(device)
    model = Wav2Lip().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate)

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)
        
    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True, overwrite_global_states=False)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    wandb.init(
      # set the wandb project where this run will be logged
      project="my-wav2lip",

      # track hyperparameters and run metadata
      config={
      "learning_rate": hparams.initial_learning_rate,
      "architecture": "Wav2lip",
      "dataset": "MyOwn",
      "epochs": 200000,
      }
    )

    # Train!
    train(device, model, train_data_loader, test_data_loader, optimizer,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=hparams.nepochs)
