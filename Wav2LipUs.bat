@echo off
call conda activate uc
python inference.py --checkpoint_path "checkpoints/wav2lip.pth" --face "fatos.jpg" --audio "ses.wav" --outfile "out_video.mp4"
timeout /t 3 /nobreak > nul
pause


