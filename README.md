# **Wav2Lip**: *Accurately Lip-syncing Videos In The Wild*

This code is part of the paper: _A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild_ published at ACM Multimedia 2020.

[[Paper]](http://arxiv.org/abs/2008.10010) | [[Project Page]](http://cvit.iiit.ac.in/research/projects/cvit-projects/a-lip-sync-expert-is-all-you-need-for-speech-to-lip-generation-in-the-wild/) | [[Demo Video] (coming soon)](#) | [[Interactive Demo]](https://bhaasha.iiit.ac.in/lipsync) | [[Collab Notebook]](https://colab.research.google.com/drive/1tZpDWXz49W6wDcTprANRGLo2D_EbD5J8?usp=sharing) | [[ReSyncED] (coming soon)](#)

 <img src="https://drive.google.com/uc?export=view&id=1Wn0hPmpo4GRbCIJR8Tf20Akzdi1qjjG9"/>

----------
**Highlights**
----------
 - Lip-sync videos to any target speech with high accuracy. Try our [interactive demo](https://bhaasha.iiit.ac.in/lipsync).
 - Works for any identity, voice, and language. Also works for CGI faces and synthetic voices.
 - Complete training code, inference code, and pretrained models are available.
 - Or, quick-start with the Google Colab Notebook: [Link](https://colab.research.google.com/drive/1tZpDWXz49W6wDcTprANRGLo2D_EbD5J8?usp=sharing)
 - Several new, reliable evaluation benchmarks and metrics [[`evaluation/` folder of this repo]](https://github.com/Rudrabha/Wav2Lip/tree/master/evaluation) released.
 - Code to calculate metrics reported in the paper is also made available.

Prerequisites
-------------
- `Python 3.5.2` (code has been tested with this version)
- ffmpeg: `sudo apt-get install ffmpeg`
- Install necessary packages using `pip install -r requirements.txt`
- Face detection [pre-trained model](https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth) should be downloaded to `face_detection/detection/sfd/s3fd.pth`

Getting the weights
----------
| Model  | Description |  Link to the model | 
| :-------------: | :---------------: | :---------------: |
| Wav2Lip  | Highly accurate lip-sync | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?e=TBFBVW)  |
| Wav2Lip + GAN  | Slightly inferior lip-sync, but better visual quality | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW) |

Lip-syncing videos using the pre-trained models (Inference)
-------
You can lip-sync any video to any audio:
```bash
python inference.py --checkpoint_path <ckpt> --face <video.mp4> --audio <an-audio-source> 
```
The result is saved (by default) in `results/result_voice.mp4`. You can specify it as an argument,  similar to several other available options. The audio source can be any file supported by `FFMPEG` containing audio data: `*.wav`, `*.mp3` or even a video file, from which the code will automatically extract the audio.

##### Tips for better results:
- Experiment with the `--pads` argument to adjust the detected face bounding box. Often leads to improved results. You might need to increase the bottom padding to include the chin region. E.g. `--pads 0 20 0 0`.
- Experiment with the `--resize_factor` argument, to get a lower resolution video. Why? The models are trained on faces which were at a lower resolution. You might get better, visually pleasing results for 720p videos than for 1080p videos (in many cases, the latter works well too). 
- The Wav2Lip model without GAN usually needs more experimenting with the above two to get the most ideal results, and sometimes, can give you a better result as well.

Preparing LRS2 for training
----------
Our models are trained on LRS2. Training on other datasets might require small modifications to the code.
##### LRS2 dataset folder structure

```
data_root (mvlrs_v1)
├── main, pretrain (we use only main folder in this work)
|	├── list of folders
|	│   ├── five-digit numbered video IDs ending with (.mp4)
```

Place the LRS2 filelists (train, val, test) `.txt` files in the `filelists/` folder.

##### Preprocess the dataset for fast training

```bash
python preprocess.py --data_root data_root/main --preprocessed_root lrs2_preprocessed/
```
Additional options like `batch_size` and number of GPUs to use in parallel to use can also be set.

##### Preprocessed LRS2 folder structure
```
preprocessed_root (lrs2_preprocessed)
├── list of folders
|	├── Folders with five-digit numbered video IDs
|	│   ├── *.jpg
|	│   ├── audio.wav
```

Train!
----------
There are two major steps: (i) Train the expert lip-sync discriminator, (ii) Train the Wav2Lip model(s).

##### Training the expert discriminator
You can download [the pre-trained weights]() if you want to skip this step. To train it:
```bash
python color_syncnet_train.py --data_root lrs2_preprocessed/ --checkpoint_dir <folder_to_save_checkpoints>
```
##### Training the Wav2Lip models
You can either train the model without the additional visual quality disriminator (< 1 day of training) or use the discriminator (~2 days). For the former, run: 
```bash
python wav2lip_train.py --data_root lrs2_preprocessed/ --checkpoint_dir <folder_to_save_checkpoints> --syncnet_checkpoint_path <path_to_expert_disc_checkpoint>
```

To train with the visual quality discriminator, you should run `hq_wav2lip_train.py` instead. The arguments for both the files are similar. In both the cases, you can resume training as well. Look at `python wav2lip_train.py --help` for more details. You can also set additional less commonly-used hyper-parameters at the bottom of the `hparams.py` file.

Evaluation
----------
Will be updated.

License and Citation
----------
The software is licensed under the MIT License. Please cite the following paper if you have use this code:
```
@misc{prajwal2020lip,
    title={A Lip Sync Expert Is All You Need for Speech to Lip Generation In The Wild},
    author={K R Prajwal and Rudrabha Mukhopadhyay and Vinay Namboodiri and C V Jawahar},
    year={2020},
    eprint={2008.10010},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```


Acknowledgements
----------
Parts of the code structure is inspired by this [TTS repository](https://github.com/r9y9/deepvoice3_pytorch). We thank the author for this wonderful code. The code for Face Detection has been taken from the [face_alignment](https://github.com/1adrianb/face-alignment) repository. We thank the authors for releasing their code and models.
