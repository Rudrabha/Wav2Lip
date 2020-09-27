
# Helper utils for wav2lip
# Downloads AV-Datasets and cuts and split it
# Written by G. Hilgemann for community and rebotnix.com
# Version 0.1 - September
# this is experimentel code and for research only

import youtube_dl
import csv
import os

# import python-ffmpeg bindings
import ffmpeg

with open("avspeech_train.csv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
 
    for i, line in enumerate(reader):  
        name = "movie_" + line[0].split(',')[0] # youtube ID
        starttime = line[0].split(',')[1] # starttime for speech segment
        endtime = line[0].split(',')[2] # endtime for speech segment
        output = line[0].split(',')[0] # ffmpeg output filename
        
        # set filename
        ydl_opts = {'outtmpl':name}

        try:
        	# youtube downloads, outputs .mkv as extension!
        	with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        		ydl.download(['https://www.youtube.com/watch?v=' + line[0] ])

        	# we split and cut out the downloaded movie to the right segment
        	stream = ffmpeg.input(name + ".mkv",ss=starttime,t=endtime).output(output + '.mp4',ss=starttime,t=endtime,acodec="aac",vcodec="libx264",g="25",video_bitrate="6500k",audio_bitrate="192",preset="fast").overwrite_output().run()
        	os.remove(name+".mkv")
        except:
        	print("ID:" + name + " is not downlable")

        # remove original downloded youtube video .mkv
        #os.remove(name + ".mkv")
