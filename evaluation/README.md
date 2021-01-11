# Novel Evaluation Framework, new filelists, and using the LSE-D and LSE-C metric.

Our paper also proposes a novel evaluation framework (Section 4). To evaluate on LRS2, LRS3, and LRW, the filelists are present in the `test_filelists` folder. Please use `gen_videos_from_filelist.py` script to generate the videos. After that, you can calculate the LSE-D and LSE-C scores using the instructions below. Please see [this thread](https://github.com/Rudrabha/Wav2Lip/issues/22#issuecomment-712825380) on how to calculate the FID scores. 

The videos of the ReSyncED benchmark for real-world evaluation will be released soon. 

### Steps to set-up the evaluation repository for LSE-D and LSE-C metric:
We use the pre-trained syncnet model available in this [repository](https://github.com/joonson/syncnet_python). 

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
    cd Wav2Lip/evaluation/scores_LSE/
    cp *.py syncnet_python/
    cp *.sh syncnet_python/ 
```
**Note: We will release the test filelists for LRW, LRS2 and LRS3 shortly once we receive permission from the dataset creators. We will also release the Real World Dataset we have collected shortly.**

* Our evaluation technique does not require ground-truth of any sorts. Given lip-synced videos we can directly calculate the scores from only the generated videos. Please store the generated videos (from our test sets or your own generated videos) in the following folder structure.
```
video data root (Folder containing all videos)
├── All .mp4 files
```
* Change the folder back to the cloned repository. 
```
cd syncnet_python
```
* To run evaluation on the LRW, LRS2 and LRS3 test files, please run the following command:
```
python calculate_scores_LRS.py --data_root /path/to/video/data/root --tmp_dir tmp_dir/
```

* To run evaluation on the ReSynced dataset or your own generated videos, please run the following command:
```
sh calculate_scores_real_videos.sh /path/to/video/data/root
```
* The generated scores will be present in the all_scores.txt generated in the ```syncnet_python/``` folder

# Evaluation of image quality using FID metric.
We will update the scripts for the FID metric shortly


# Opening issues related to evaluation scripts
* Please open the issues with the "Evaluation" label if you face any issues in the evaluation scripts. 

# Acknowledgements
Our evaluation pipeline in based on two existing repositories. LSE metrics are based on the [syncnet_python](https://github.com/joonson/syncnet_python) repository and the FID score is based on [pytorch-fid](https://github.com/mseitzer/pytorch-fid) repository. We thank the authors of both the repositories for releasing their wonderful code.



