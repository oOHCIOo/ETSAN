import json
from pathlib import Path
import os
import torch
import numpy as np

def choose_valid_adaptive_frame(score, video_name):
    """
    将分值选择可变的，而不是按照固定数量去实现，阈值分值为所有分值的平均

    :param begin:
    :param end:
    :param score:
    :param video_name:
    :return:
    """
    # todo : frame_path needs to change
    frame_path = os.path.join('/home/qc/qc/Data/hmdb_frames', video_name)
    frames_list = os.listdir(frame_path)

    all_num = len(frames_list)
    all_score = score[:all_num]

    score_thre = sum(all_score) / len(all_score)

    valid_index = []
    for i in range(len(all_score)):
        if all_score[i] >= score_thre:
            valid_index.append(i)
    return valid_index


def get_valid_frame():
    scores = json.load(open('/home/qc/qc/Data/sumGan/scores/13.json','r'))

    video_names = list(Path('/home/qc/qc/Data/HMDB/frames').iterdir())
    all_score_index = {}
    for video_name in video_names:
        video_name_per = video_name.name[4:]
        score = scores[video_name_per]
        valid_index_per = choose_valid_adaptive_frame(score, video_name.name)
        print("Choosing frames length: ", len(valid_index_per))
        all_score_index[video_name.name] = valid_index_per

    np.save('/home/qc/qc/Data/attention_data/score_index_whole_sumGan_5_10.py', all_score_index)


if __name__ == "__main__":
    get_valid_frame()