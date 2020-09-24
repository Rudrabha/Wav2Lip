yourfilenames=`ls $1`
for eachfile in $yourfilenames
do
   python run_pipeline.py --videofile $eachfile --reference lipsyncnet --data_dir /ssd_scratch/cvit/prajwalkr_rudra/tmp_dir
   python calc_real_video.py --videofile $eachfile --reference lipsyncnet --data_dir /ssd_scratch/cvit/prajwalkr_rudra/tmp_dir >> test_score.txt
done
