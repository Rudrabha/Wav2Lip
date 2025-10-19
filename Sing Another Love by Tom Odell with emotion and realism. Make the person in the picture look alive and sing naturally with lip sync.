# **Wav2Lip**: *Accurately Lip-syncing Videos In The Wild* 

# Commercial Version

Create your first lipsync generation in minutes. Please note, the commercial version is of a much higher quality than the old open source model!

## Create your API Key

Create your API key from the [Dashboard](https://sync.so/keys). You will use this key to securely access the Sync API.

## Make your first generation

The following example shows how to make a lipsync generation using the Sync API.

### Python

#### Step 1: Install Sync SDK

```bash
pip install syncsdk
```

#### Step 2: Make your first generation

Copy the following code into a file `quickstart.py` and replace `YOUR_API_KEY_HERE` with your generated API key.

```python
# quickstart.py
import time
from sync import Sync
from sync.common import Audio, GenerationOptions, Video
from sync.core.api_error import ApiError

# ---------- UPDATE API KEY ----------
# Replace with your Sync.so API key
api_key = "YOUR_API_KEY_HERE" 

# ----------[OPTIONAL] UPDATE INPUT VIDEO AND AUDIO URL ----------
# URL to your source video
video_url = "https://assets.sync.so/docs/example-video.mp4"
# URL to your audio file
audio_url = "https://assets.sync.so/docs/example-audio.wav"
# ----------------------------------------

client = Sync(
    base_url="https://api.sync.so", 
    api_key=api_key
).generations

print("Starting lip sync generation job...")

try:
    response = client.create(
        input=[Video(url=video_url),Audio(url=audio_url)],
        model="lipsync-2",
        options=GenerationOptions(sync_mode="cut_off"),
        outputFileName="quickstart"
    )
except ApiError as e:
    print(f'create generation request failed with status code {e.status_code} and error {e.body}')
    exit()

job_id = response.id
print(f"Generation submitted successfully, job id: {job_id}")

generation = client.get(job_id)
status = generation.status
while status not in ['COMPLETED', 'FAILED']:
    print('polling status for generation', job_id)
    time.sleep(10)
    generation = client.get(job_id)
    status = generation.status

if status == 'COMPLETED':
    print('generation', job_id, 'completed successfully, output url:', generation.output_url)
else:
    print('generation', job_id, 'failed')
```

Run the script:

```bash
python quickstart.py
```

#### Step 3: Done!

It may take a few minutes for the generation to complete. You should see the generated video URL in the terminal post completion.

---

### TypeScript

#### Step 1: Install dependencies

```bash
npm i @sync.so/sdk
```

#### Step 2: Make your first generation

Copy the following code into a file `quickstart.ts` and replace `YOUR_API_KEY_HERE` with your generated API key.

```typescript
// quickstart.ts
import { SyncClient, SyncError } from "@sync.so/sdk";

// ---------- UPDATE API KEY ----------
// Replace with your Sync.so API key
const apiKey = "YOUR_API_KEY_HERE";

// ----------[OPTIONAL] UPDATE INPUT VIDEO AND AUDIO URL ----------
// URL to your source video
const videoUrl = "https://assets.sync.so/docs/example-video.mp4";
// URL to your audio file
const audioUrl = "https://assets.sync.so/docs/example-audio.wav";
// ----------------------------------------

const client = new SyncClient({ apiKey });

async function main() {
    console.log("Starting lip sync generation job...");

    let jobId: string;
    try {
        const response = await client.generations.create({
            input: [
                {
                    type: "video",
                    url: videoUrl,
                },
                {
                    type: "audio",
                    url: audioUrl,
                },
            ],
            model: "lipsync-2",
            options: {
                sync_mode: "cut_off",
            },
            outputFileName: "quickstart"
        });
        jobId = response.id;
        console.log(`Generation submitted successfully, job id: ${jobId}`);
    } catch (err) {
        if (err instanceof SyncError) {
            console.error(`create generation request failed with status code ${err.statusCode} and error ${JSON.stringify(err.body)}`);
        } else {
            console.error('An unexpected error occurred:', err);
        }
        return;
    }

    let generation;
    let status;
    while (status !== 'COMPLETED' && status !== 'FAILED') {
        console.log(`polling status for generation ${jobId}...`);
        try {
            await new Promise(resolve => setTimeout(resolve, 10000));
            generation = await client.generations.get(jobId);
            status = generation.status;
        } catch (err) {
            if (err instanceof SyncError) {
                console.error(`polling failed with status code ${err.statusCode} and error ${JSON.stringify(err.body)}`);
            } else {
                console.error('An unexpected error occurred during polling:', err);
            }
            status = 'FAILED';
        }
    }

    if (status === 'COMPLETED') {
        console.log(`generation ${jobId} completed successfully, output url: ${generation?.outputUrl}`);
    } else {
        console.log(`generation ${jobId} failed`);
    }
}

main();
```

Run the script:

```bash
npx tsx quickstart.ts -y
```

#### Step 3: Done!

You should see the generated video URL in the terminal.

---

## Next Steps

Well done! You've just made your first lipsync generation with sync.so!

Ready to unlock the full potential of lipsync? Dive into our interactive [Studio](https://sync.so/login) to experiment with all available models, or explore our [API Documentation](/api-reference) to take your lip-sync generations to the next level!

## Contact
- prady@sync.so
- pavan@sync.so
- sanjit@sync.so



# Non Commercial Open-source Version

This code is part of the paper: _A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild_ published at ACM Multimedia 2020. 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-lip-sync-expert-is-all-you-need-for-speech/lip-sync-on-lrs2)](https://paperswithcode.com/sota/lip-sync-on-lrs2?p=a-lip-sync-expert-is-all-you-need-for-speech)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-lip-sync-expert-is-all-you-need-for-speech/lip-sync-on-lrs3)](https://paperswithcode.com/sota/lip-sync-on-lrs3?p=a-lip-sync-expert-is-all-you-need-for-speech)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-lip-sync-expert-is-all-you-need-for-speech/lip-sync-on-lrw)](https://paperswithcode.com/sota/lip-sync-on-lrw?p=a-lip-sync-expert-is-all-you-need-for-speech)
|📑 Original Paper|📰 Project Page|🌀 Demo|⚡ Live Testing|📔 Colab Notebook
|:-:|:-:|:-:|:-:|:-:|
[Paper](http://arxiv.org/abs/2008.10010) | [Project Page](http://cvit.iiit.ac.in/research/projects/cvit-projects/a-lip-sync-expert-is-all-you-need-for-speech-to-lip-generation-in-the-wild/) | [Demo Video](https://youtu.be/0fXaDCZNOJc) | [Interactive Demo](https://synclabs.so/) | [Colab Notebook](https://colab.research.google.com/drive/1tZpDWXz49W6wDcTprANRGLo2D_EbD5J8?usp=sharing) /[Updated Collab Notebook](https://colab.research.google.com/drive/1IjFW1cLevs6Ouyu4Yht4mnR4yeuMqO7Y#scrollTo=MH1m608OymLH)
 
![Logo](https://drive.google.com/uc?export=view&id=1Wn0hPmpo4GRbCIJR8Tf20Akzdi1qjjG9)
----------
**Highlights**
----------
 - Weights of the visual quality disc has been updated in readme!
 - Lip-sync videos to any target speech with high accuracy :100:. Try our [interactive demo](https://sync.so/).
 - :sparkles: Works for any identity, voice, and language. Also works for CGI faces and synthetic voices.
 - Complete training code, inference code, and pretrained models are available :boom:
 - Or, quick-start with the Google Colab Notebook: [Link](https://colab.research.google.com/drive/1tZpDWXz49W6wDcTprANRGLo2D_EbD5J8?usp=sharing). Checkpoints and samples are available in a Google Drive [folder](https://drive.google.com/drive/folders/1I-0dNLfFOSFwrfqjNa-SXuwaURHE5K4k?usp=sharing) as well. There is also a [tutorial video](https://www.youtube.com/watch?v=Ic0TBhfuOrA) on this, courtesy of [What Make Art](https://www.youtube.com/channel/UCmGXH-jy0o2CuhqtpxbaQgA). Also, thanks to [Eyal Gruss](https://eyalgruss.com), there is a more accessible [Google Colab notebook](https://j.mp/wav2lip) with more useful features. A tutorial collab notebook is present at this [link](https://colab.research.google.com/drive/1IjFW1cLevs6Ouyu4Yht4mnR4yeuMqO7Y#scrollTo=MH1m608OymLH).  
 - :fire: :fire: Several new, reliable evaluation benchmarks and metrics [[`evaluation/` folder of this repo]](https://github.com/Rudrabha/Wav2Lip/tree/master/evaluation) released. Instructions to calculate the metrics reported in the paper are also present.
--------
**Disclaimer**
--------
All results from this open-source code or our [demo website](https://bhaasha.iiit.ac.in/lipsync) should only be used for research/academic/personal purposes only. As the models are trained on the <a href="http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html">LRS2 dataset</a>, any form of commercial use is strictly prohibited. For commercial requests please contact us directly!
Prerequisites
-------------
- `Python 3.6` 
- ffmpeg: `sudo apt-get install ffmpeg`
- Install necessary packages using `pip install -r requirements.txt`. Alternatively, instructions for using a docker image is provided [here](https://gist.github.com/xenogenesi/e62d3d13dadbc164124c830e9c453668). Have a look at [this comment](https://github.com/Rudrabha/Wav2Lip/issues/131#issuecomment-725478562) and comment on [the gist](https://gist.github.com/xenogenesi/e62d3d13dadbc164124c830e9c453668) if you encounter any issues. 
- Face detection [pre-trained model](https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth) should be downloaded to `face_detection/detection/sfd/s3fd.pth`. Alternative [link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/prajwal_k_research_iiit_ac_in/EZsy6qWuivtDnANIG73iHjIBjMSoojcIV0NULXV-yiuiIg?e=qTasa8) if the above does not work.
Getting the weights
----------
| Model  | Description |  Link to the model | 
| :-------------: | :---------------: | :---------------: |
| Wav2Lip  | Highly accurate lip-sync | [Link](https://drive.google.com/drive/folders/153HLrqlBNxzZcHi17PEvP09kkAfzRshM?usp=share_link)  |
| Wav2Lip + GAN  | Slightly inferior lip-sync, but better visual quality | [Link](https://drive.google.com/file/d/15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk/view?usp=share_link) |


Lip-syncing videos using the pre-trained models (Inference)
-------
You can lip-sync any video to any audio:
```bash
python inference.py --checkpoint_path <ckpt> --face <video.mp4> --audio <an-audio-source> 
```
The result is saved (by default) in `results/result_voice.mp4`. You can specify it as an argument,  similar to several other available options. The audio source can be any file supported by `FFMPEG` containing audio data: `*.wav`, `*.mp3` or even a video file, from which the code will automatically extract the audio.
##### Tips for better results:
- Experiment with the `--pads` argument to adjust the detected face bounding box. Often leads to improved results. You might need to increase the bottom padding to include the chin region. E.g. `--pads 0 20 0 0`.
- If you see the mouth position dislocated or some weird artifacts such as two mouths, then it can be because of over-smoothing the face detections. Use the `--nosmooth` argument and give it another try. 
- Experiment with the `--resize_factor` argument, to get a lower-resolution video. Why? The models are trained on faces that were at a lower resolution. You might get better, visually pleasing results for 720p videos than for 1080p videos (in many cases, the latter works well too). 
- The Wav2Lip model without GAN usually needs more experimenting with the above two to get the most ideal results, and sometimes, can give you a better result as well.
Preparing LRS2 for training
----------
Our models are trained on LRS2. See [here](#training-on-datasets-other-than-lrs2) for a few suggestions regarding training on other datasets.
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
Additional options like `batch_size` and the number of GPUs to use in parallel to use can also be set.
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
You can download [the pre-trained weights](#getting-the-weights) if you want to skip this step. To train it:
```bash
python color_syncnet_train.py --data_root lrs2_preprocessed/ --checkpoint_dir <folder_to_save_checkpoints>
```
##### Training the Wav2Lip models
You can either train the model without the additional visual quality discriminator (< 1 day of training) or use the discriminator (~2 days). For the former, run: 
```bash
python wav2lip_train.py --data_root lrs2_preprocessed/ --checkpoint_dir <folder_to_save_checkpoints> --syncnet_checkpoint_path <path_to_expert_disc_checkpoint>
```
To train with the visual quality discriminator, you should run `hq_wav2lip_train.py` instead. The arguments for both files are similar. In both cases, you can resume training as well. Look at `python wav2lip_train.py --help` for more details. You can also set additional less commonly-used hyper-parameters at the bottom of the `hparams.py` file.
Training on datasets other than LRS2
------------------------------------
Training on other datasets might require modifications to the code. Please read the following before you raise an issue:
- You might not get good results by training/fine-tuning on a few minutes of a single speaker. This is a separate research problem, to which we do not have a solution yet. Thus, we would most likely not be able to resolve your issue. 
- You must train the expert discriminator for your own dataset before training Wav2Lip.
- If it is your own dataset downloaded from the web, in most cases, needs to be sync-corrected.
- Be mindful of the FPS of the videos of your dataset. Changes to FPS would need significant code changes. 
- The expert discriminator's eval loss should go down to ~0.25 and the Wav2Lip eval sync loss should go down to ~0.2 to get good results. 
When raising an issue on this topic, please let us know that you are aware of all these points.
We have an HD model trained on a dataset allowing commercial usage. The size of the generated face will be 192 x 288 in our new model.
Evaluation
----------
Please check the `evaluation/` folder for the instructions.
License and Citation
----------
This repository can only be used for personal/research/non-commercial purposes. However, for commercial requests, please contact us directly at rudrabha@synclabs.so or prajwal@synclabs.so. We have a turn-key hosted API with new and improved lip-syncing models here: https://synclabs.so/
The size of the generated face will be 192 x 288 in our new models. Please cite the following paper if you use this repository:
```
@inproceedings{10.1145/3394171.3413532,
author = {Prajwal, K R and Mukhopadhyay, Rudrabha and Namboodiri, Vinay P. and Jawahar, C.V.},
title = {A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild},
year = {2020},
isbn = {9781450379885},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3394171.3413532},
doi = {10.1145/3394171.3413532},
booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
pages = {484–492},
numpages = {9},
keywords = {lip sync, talking face generation, video generation},
location = {Seattle, WA, USA},
series = {MM '20}
}
```
Acknowledgments
----------
Parts of the code structure are inspired by this [TTS repository](https://github.com/r9y9/deepvoice3_pytorch). We thank the author for this wonderful code. The code for Face Detection has been taken from the [face_alignment](https://github.com/1adrianb/face-alignment) repository. We thank the authors for releasing their code and models. We thank [zabique](https://github.com/zabique) for the tutorial collab notebook.
## Acknowledgements
 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)
