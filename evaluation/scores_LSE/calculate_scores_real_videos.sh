rm all_scores.txt
yourfilenames=`ls $1`
for eachfile in $yourfilenames
do
   python run_pipeline.py --videofile $eachfile --reference wav2lip --data_dir tmp_dir
   python calculate_scores_real_videos.py --videofile $eachfile --reference wav2lip --data_dir /ssd_scratch/cvit/prajwalkr_rudra/tmp_dir >> all_scores.txt
done
