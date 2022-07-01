import numpy as np
import torch


def padding_feature():
    maxT = 34
    feature_path = '/home/qc/qc/Data/avs_res/all_class_videos_feature_list_51_4_10.npy'
    all_class_videos_feature_list = np.load(feature_path, allow_pickle=True)
    padding_segment = torch.zeros(1, 4096)
    num = 0
    for i in range(len(all_class_videos_feature_list)):
        per_class = all_class_videos_feature_list[i]
        for j in range(len(per_class)):
            per_segment = per_class[j]
            if (len(per_segment)) < maxT:
                print("padding feature")
                per_segment += [padding_segment] * (maxT - len(per_segment))

                all_class_videos_feature_list[i][j] = per_segment
    print("Saving data!")
    np.save('/home/qc/qc/Data/avs_res/padding_feature_4_10.npy', all_class_videos_feature_list)

def getMaxT():
    maxT = 0
    feature_path = '/home/qc/qc/Data/avs_res/all_class_videos_feature_list_51_4_26.npy'
    all_class_videos_feature_list = np.load(feature_path, allow_pickle=True)
    for i in range(len(all_class_videos_feature_list)):
        per_class = all_class_videos_feature_list[i]
        for j in range(len(per_class)):
            per_segment = per_class[j]
            maxT = max(len(per_segment), maxT)
    print("maxT is --> ", maxT)


def checkMaxT():
    maxT = 34
    feature_path = '/home/qc/qc/Data/avs_res/padding_feature_4_10.npy'
    all_class_videos_feature_list = np.load(feature_path, allow_pickle=True)
    for i in range(len(all_class_videos_feature_list)):
        per_class = all_class_videos_feature_list[i]
        for j in range(len(per_class)):
            per_segment = per_class[j]
            if (len(per_segment) != maxT) :
                print("Error! ")
                # break
            # maxT = max(maxT, len(per_segment))
    print(maxT)

if __name__ == "__main__":
    getMaxT()
    # padding_feature()
    # checkMaxT()
    # emb_path = '/home/qc/qc/Data/avs_res/hmdb_class_embedding.npy'
    # emb = np.load(emb_path, allow_pickle=True)
    # emb_re = []
    # for i in range(len(emb)):
    #     emb_re.append(emb[i][0])
    # print(len(emb_re))