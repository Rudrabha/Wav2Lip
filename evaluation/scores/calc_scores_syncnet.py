#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess
import glob
import os
from tqdm import tqdm

from SyncNetInstance_calc_scores import *

# ==================== LOAD PARAMS ====================


parser = argparse.ArgumentParser(description = "SyncNet");

parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--data_root', type=str, required=True, help='');
parser.add_argument('--tmp_dir', type=str, default="data/work/pytmp", help='');
parser.add_argument('--reference', type=str, default="demo", help='');

opt = parser.parse_args();


# ==================== RUN EVALUATION ====================

s = SyncNetInstance();

s.loadParameters(opt.initial_model);
#print("Model %s loaded."%opt.initial_model);
path = os.path.join(opt.data_root, "*.mp4")

all_videos = glob.glob(path)

prog_bar = tqdm(range(len(all_videos)))
avg_confidence = 0.
avg_min_distance = 0.


for videofile_idx in prog_bar:
	videofile = all_videos[videofile_idx]
	offset, confidence, min_distance = s.evaluate(opt, videofile=videofile)
	avg_confidence += confidence
	avg_min_distance += min_distance
	prog_bar.set_description('Avg Confidence: {}, Avg Minimum Dist: {}'.format(round(avg_confidence / (videofile_idx + 1), 3), round(avg_min_distance / (videofile_idx + 1), 3)))
	prog_bar.refresh()

print ('Average Confidence: {}'.format(avg_confidence/len(all_videos)))
print ('Average Minimum Distance: {}'.format(avg_min_distance/len(all_videos)))



