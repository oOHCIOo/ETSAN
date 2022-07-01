"""
This file is used to get each segment attention weight.
"""
import os
import time
import cv2
import pickle
import numpy as np
# import Sampling
import sys
sys.path.append("/home/xingmeng/qc/Attention/AttentionModule/pytorch-video-recognition/network")
sys.path.append("/home/xingmeng/qc/Attention/AttentionModule/c3d-pytorch")
import torch
from predict import *

video_frames_path = "/home/xingmeng/qc/Attention/Videodata/frames"
all_class_name = sorted([f for f in os.listdir(video_frames_path) if not f.startswith('.')])
segments_path = "/home/xingmeng/qc/Attention/Videodata/segments"
class_segments_path = [segments_path+"/"+f for f in all_class_name]
for each_class_segment_path in class_segments_path:
    if not os.path.exists(each_class_segment_path):
        os.makedirs(each_class_segment_path)



"""
Input Segments to pre-trained CNN architecture to extract features. Here Use the C3D Model.
"""
df_load_frames = open("/home/xingmeng/qc/Attention/Videodata/all_videos_choose_frames_path_3_6.21.pickle",'rb')
all_videos_choose_frames_path = pickle.load(df_load_frames)
T = 24
K = 16
print("Starting...")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Using gpu or cpu:", device)
all_class_videos_feature_list = []
all_class_Videos_feature_concat = []
video_num = 0
for i in range(len(all_videos_choose_frames_path)):
    each_class_videos_choose_frames_path = all_videos_choose_frames_path[i]
    each_class_video_segment_path = class_segments_path[i]
    each_class_video_feature_list = []

    for each_video_choose_frames_path in each_class_videos_choose_frames_path:
        segment_count = 0
        img = cv2.imread(each_video_choose_frames_path[0][0])
        imgSize = img.shape
        each_video_name = each_video_choose_frames_path[0][0].split("/")[-2]
        each_video_feature_list = []
        for j in range(len(each_video_choose_frames_path)):
            each_video_segment_choose_frames_path = each_video_choose_frames_path[j]
            print(each_video_segment_choose_frames_path)
            each_video_path = each_class_video_segment_path + "/" +each_video_name
            if not os.path.exists(each_video_path):
                os.makedirs(each_video_path)
            segment_count += 1
            print(len(each_video_segment_choose_frames_path))
            """
            Get C3D feature
            """
            # print("Start to extract features....")
            with torch.no_grad():
                each_segment_features = main(each_video_segment_choose_frames_path).to(device)
            print("each segment_features is:", each_segment_features)
            print("Current Segment index:",j, "      The size is:",each_segment_features.size())
            each_video_feature_list.append(each_segment_features)
        print("each_video_feature_list is:", each_video_feature_list)
        print("each_video_feature_list length is:", len(each_video_feature_list))
        print("Finish extracting", video_num,  " video C3D features!")
        video_num += 1
        each_class_video_feature_list.append(each_video_feature_list)
    print("class :", i, "Finish Extracting features")
    all_class_videos_feature_list.append(each_class_video_feature_list)

print("Save size is:", sys.getsizeof(all_class_videos_feature_list))
df_save_feature_list = open("all_class_videos_feature_list_3_6.21.pickle", 'wb')
pickle.dump(all_class_videos_feature_list, df_save_feature_list)
print("Finished!")