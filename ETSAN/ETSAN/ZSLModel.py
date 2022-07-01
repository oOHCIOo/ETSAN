import torch
import torch.nn as nn
import gensim
import os
import numpy as np
import torch.nn.functional as F


def choose_valid_segments(attention_weights, segment_features):
    valid_segment = []
    for j in range(len(attention_weights)):
        attention_weights_per = attention_weights[j]
        valid_segment_per = []
        segment_features_per = segment_features[j]
        attention_threshold_per = sum(attention_weights_per) / len(attention_weights_per)
        for i in range(len(attention_weights_per)):
            if attention_weights_per[i] >= attention_threshold_per:
                valid_segment_per.append(segment_features_per[i])
        # print("valid_segmnet_per length before padding:", len(valid_segment_per))
        valid_segment_per += [torch.zeros([4096])] * (32 - len(valid_segment_per))
        # print("valid segment length is:", len(valid_segment_per))
        valid_segment.append(torch.stack(valid_segment_per))
    valid_segment = torch.stack(valid_segment)
    return valid_segment


# -----------------------Semantic Embedding Model-----------------------
def class2vec(classes):
    """
    Two way.One way is no use of pretrained model, Another way is to use a pretrained model
    :param path: All classes
    :return:classes name vector
    """
    classes_name_split = [i.split("_") for i in classes]

    # model_path = "/Users/chengqi/GoogleNews-vectors-negative300.bin.gz"
    model_path = r"D:\attention_stpn\ServerProgram\GoogleNews-vectors-negative300.bin.gz"
    Model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

    classes_name_embedding = {}
    classes_name_arr_embedding =[]
    for i in classes_name_split:
        per_embed = Model[i]
        # print(per_embed)
        classes_name_embedding["_".join(i)] = np.sum(per_embed, axis=0) / len(per_embed)
        classes_name_arr_embedding.append(np.sum(per_embed, axis=0)/ len(per_embed))
    return classes_name_embedding, classes_name_arr_embedding


class SemanticEmbeddingModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SemanticEmbeddingModule, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        # print("After First FC, the grad of x is:",x.grad)
        x = self.relu(self.fc2(x))
        x = self.bn2(x)
        # print("After Second FC, the grad of x is:", x.grad)
        return x


# --------------------Reshaping C3D feature to 512-d -----------------------
# Binary GRU with hidden size of 512
class BiGRU(nn.Module):
    """
    bidirectional GRU output is double hidden_size
    """

    def __init__(self, input_size, hidden_size):
        super(BiGRU,self).__init__()
        self.bi_gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, bidirectional=True)
        self.bn3 = nn.BatchNorm1d(32)
        # self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.bi_gru(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x


# class BiGRU(nn.Module):
#     """
#     bidirectional GRU output is double hidden_size
#     """
#
#     def __init__(self, input_size, hidden_size):
#         super(BiGRU,self).__init__()
#         self.fc1 = nn.Linear(input_size, 2048)
#         self.fc2 = nn.Linear(2048, hidden_size * 2)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         return x


# ----Two FC layer and An Average Pooling layer to get final Relation Score---
class Aggregate(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Aggregate,self).__init__()
        self.fc3 = nn.Linear(input_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.bn4 = nn.BatchNorm1d(32)


    def forward(self, x):
        x = self.relu(self.fc3(x))
        x = self.bn4(x)
        # x = self.fc4(x)
        relation_score = self.softmax(self.fc4(x))
        relation_score = self.pool(relation_score)
        # print(relation_score)
        return relation_score


# ---------------------------Attention Module------------------------------------
class AttentionModule(nn.Module):
    def __init__(self, num_classes=51):
        super(AttentionModule, self).__init__()
        self.num_classes = num_classes
        self.model_name = "Module"
        self.fc1 = nn.Linear(4096, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, inp):
        # print("Input feature size is:", inp.size())
        inp = self.fc1(inp)
        inp = self.relu(inp)
        inp = self.fc2(inp)
        inp = self.sigmoid(inp)
        return inp

attention_module = AttentionModule()
# attention_module.load_state_dict(torch.load(r'D:\attention_stpn\08-01_21-59-13\attention_model_params.pkl', map_location="cpu"))
attention_module_state_dict = attention_module.state_dict()
predict_attention_module_state_dict = torch.load(r'D:\attention_stpn\08-01_21-59-13\attention_model_params.pkl', map_location="cpu")
pretrained_dict_1 = {k: v for k, v in predict_attention_module_state_dict.items() if k in attention_module_state_dict}
attention_module_state_dict.update(pretrained_dict_1)
attention_module.eval()


class wholeModel(nn.Module):
    def __init__(self):
        super(wholeModel, self).__init__()
        self.fc1 = nn.Linear(300,4096)
        self.fc2 = nn.Linear(4096, 512)
        self.relu = nn.ReLU()
        self.bi_gru = nn.GRU(4096,256, bidirectional=True)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.softmax = nn.Softmax()
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(32)
        self.semantic = nn.Sequential(self.fc1, self.relu, self.bn1, self.fc2, self.relu, self.bn2)
        self.relation = nn.Sequential(self.fc3, self.relu, self.bn4, self.fc4, self.softmax, self.pool)


    def forward(self, batch_features,batch_attributes):
        batch_att = attention_module(batch_features).detach_()
        batch_features = choose_valid_segments(batch_att, batch_features)
        batch_features, _ = self.bi_gru(batch_features)
        sample_attributes = self.semantic(batch_attributes)
        batch_size = batch_features.shape[0]
        class_num = sample_attributes.shape[0]
        batch_features_ext = batch_features.unsqueeze(1).repeat(1, class_num, 1, 1)
        sample_features_ext = sample_attributes.unsqueeze(1).repeat(1, 32, 1)
        sample_features_ext = sample_features_ext.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        relation_score = self.relation(
            torch.mul(torch.sub(batch_features_ext, sample_features_ext), torch.sub(batch_features_ext, sample_features_ext)).view(-1,32,512)
        )
        return relation_score


