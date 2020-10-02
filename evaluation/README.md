# Evaluation of Lip-sync using LSE-D and LSE-C metric.

We use the pre-trained syncnet model available in this [repository](https://github.com/joonson/syncnet_python). 

### Steps to set-up the evaluation repository:
* Clone the SyncNet repository.
``` 
git clone https://github.com/joonson/syncnet_python.git 
```
* Follow the procedure given in the above linked [repository](https://github.com/joonson/syncnet_python) to download the pretrained models and set up the dependencies. 
    * **Note: Please install a separate virtual environment for the evaluation scripts. The versions used by Wav2Lip and the publicly released code of SyncNet is different and can cause version mis-match issues. To avoid this, we suggest the users to install a separate virtual environment for the evaluation scripts**
```
cd syncnet_python
pip install -r requirements.txt
sh download_model.sh
```
* The above step should ensure that all the dependencies required by the repository is installed and the pre-trained models are downloaded.

### Running the evaluation scripts:
* Copy our evaluation scripts given in this folder to the cloned repository.
```  
    cd WaveLip/evaluation/scores/
    cp *.py syncnet_python/
    cp *.sh syncnet_python/ 
```
**Note: We will release the files test filelists for LRW, LRS2 and LRS3 shortly once we receive permission from the dataset creators. We will also release the Real World Dataset we have collected shortly.**

* Our evaluation technique does not require ground-truth of any sorts. Given lip-synced videos we can directly calculate the scores from only the generated videos. Please store the generated videos (from our test sets or your own generated videos) in the following folder structure.
```
video data root (Folder containing all videos)
├── All .mp4 files
```

* To run evaluation on the LRW, LRS2 and LRS3 test files, please run the following command:
```
python calc_scores_syncnet.py --data_root /path/to/video/data/root --tmp_dir tmp_dir/
```



