import torch
import torch.nn as nn
import random
import pickle
import pdb
from sklearn.model_selection import train_test_split
from torch import optim
import torch.utils.data as Data
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import numpy as np
from utils.utils import validate, show_confMat
import sys
sys.path.append("/home/xingmeng/qc/Attention/AttentionModule")
from ZSLModel1003_ucf import *
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

#------------------------Preparing Data----------------------------------
batch_size = 8
episode = 3000
test_episode = 100
learning_rate = 1e-3
mms = MinMaxScaler()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# log
result_dir = os.path.join('/home/qc/qc/Data/avs_res', "ZS_Result_hmdb")
print("result_dir: ", result_dir)
now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
log_dir = os.path.join(result_dir, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)


# df_dump_path = open(r'D:\data\train_features_cpu', 'rb')
# train_features_cpu = pickle.load(df_dump_path)
#
# df_dump_path_test = open(r'D:\data\test_features_cpu','rb')
# test_features_cpu = pickle.load(df_dump_path_test)
all_class_videos_feature_list = np.load('/home/qc/qc/Data/avs_res/padding_feature_4_10.npy', allow_pickle=True)

all_data = []
all_labels = []
train_features = []
train_labels = []
test_features = []
test_labels = []
for i in range(51):
    per_class_videos_features = all_class_videos_feature_list[i]
    if i < 26:
        for j in range(len(per_class_videos_features)):
            per_video_segment = torch.stack(per_class_videos_features[j])
            per_video_segment = per_video_segment.squeeze(dim=1)
            per_video_segment = torch.from_numpy(mms.fit_transform(per_video_segment.numpy())
                                                 )
            train_features.append(per_video_segment)
            train_labels.append(i)  # label is starting from 0
    else:
        for j in range(len(per_class_videos_features)):
            per_video_segment = torch.stack(per_class_videos_features[j])
            per_video_segment = per_video_segment.squeeze(dim=1)
            # per_video_segment = torch.from_numpy(mms.fit_transform(per_video_segment.numpy())
            #                                      )
            test_features.append(per_video_segment)
            test_labels.append(i)
print(max(train_labels))
print(min(test_labels))
# pdb.set_trace()

# df_load_attributes = open(r"C:\Users\qicheng\Desktop\classes_name_arr_embedding.pickle", 'rb')
attributes = np.load('/home/qc/qc/Data/avs_res/hmdb_class_embedding.npy',allow_pickle=True)
attributes = np.array(attributes)

df_load_attributes = open("/home/qc/qc/Data/avs_res/attributes.pickle", 'rb')
attributes = pickle.load(df_load_attributes)
attributes = np.array(attributes)
# attributes = mms.fit_transform(attributes)

train_features = torch.stack(train_features)
train_labels = np.array(train_labels) #
train_ids = np.unique(train_labels)
att = attributes[train_labels]
test_features = torch.stack(test_features)

test_labels = np.array(test_labels)
test_id = np.unique(test_labels)
att_pro = attributes[test_id]
# pdb.set_trace()
# train set
print(train_features.shape)

train_labels = torch.from_numpy(train_labels)

train_labels = train_labels.unsqueeze(1)
print(train_labels.shape)

all_attributes = np.array(attributes)
print(all_attributes.shape)

attributes = torch.from_numpy(attributes)

# test set
# test_features = torch.from_numpy(x_test)
print(test_features.shape)

test_labels = torch.from_numpy(test_labels)
test_labels = test_labels.unsqueeze(1)
print(test_labels.shape)

testclasses_id = np.array(test_id)
print(testclasses_id.shape)


test_attributes = torch.from_numpy(att_pro).float()
print(test_attributes.shape)

train_data = Data.TensorDataset(train_features, train_labels)


# -----------------------------Choose valid segments according to attention weights------------------
# def choose_valid_segments(attention_weights, segment_features):
#     maxT = 0
#     valid_segment = []
#     for j in range(len(attention_weights)):
#         attention_weights_per = attention_weights[j]
#         valid_segment_per = []
#         segment_features_per = segment_features[j]
#         attention_threshold_per = sum(attention_weights_per) / len(attention_weights_per)
#         for i in range(len(attention_weights_per)):
#             if attention_weights_per[i] >= attention_threshold_per:
#                 valid_segment_per.append(segment_features_per[i])
#         # print("valid_segmnet_per length before padding:", len(valid_segment_per))
#         # valid_segment_per += [torch.zeros([4096])] * (34 - len(valid_segment_per))
#         # print("valid segment length is:", len(valid_segment_per))
#         maxT = max(maxT, len(valid_segment_per))
#         valid_segment.append(valid_segment_per)
#     # for i in range(len(valid_segment)):
#     #     if len(valid_segment[i]) < maxT:
#     #         valid_segment[i] += [torch.zeros([4096])] * (maxT - len(valid_segment_per))
#     #         valid_segment[i] = torch.stack(valid_segment[i])
#
#     valid_segment = torch.stack(valid_segment)
#     return valid_segment, maxT

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
        valid_segment_per += [torch.zeros([4096])] * (34 - len(valid_segment_per))
        # print("valid segment length is:", len(valid_segment_per))
        valid_segment.append(torch.stack(valid_segment_per))
    valid_segment = torch.stack(valid_segment)

    return valid_segment


# Here Use attention weights multiply feature
def weighted_segments(attention_weights, segment_features):
    weighted_segment_features = attention_weights * segment_features
    return weighted_segment_features


#-------------------------Construct whole Model -------------------------------
print("init networks")
attention_module = AttentionModule()
semantic_embedding_model = SemanticEmbeddingModule(300, 4096, 512)
bi_gru_model = BiGRU(4096, 256)
# relation_score_aggregate_model = Aggregate(512, 256, 1)
relation_score_aggregate_model = Aggregate(1024, 256, 1024)
attention_module_state_dict = attention_module.state_dict()
# todo pretrained attention weights path/
predict_attention_module_state_dict = torch.load('/home/qc/qc/Data/attention_result/pretrain_attention_module_Result/04-12_08-33-06/attention_model_params.pkl')
pretrained_dict_1 = {k: v for k, v in predict_attention_module_state_dict.items() if k in attention_module_state_dict}
attention_module_state_dict.update(pretrained_dict_1)
# attention_module.load_state_dict(attention_module_state_dict)
attention_module.eval()
mse = nn.BCELoss()
semantic_embedding_model_optim = torch.optim.Adam(semantic_embedding_model.parameters(), lr=learning_rate,
                                                  weight_decay=1e-5)
semantic_embedding_model_scheduler = torch.optim.lr_scheduler.StepLR(semantic_embedding_model_optim, step_size=500,
                                                                     gamma=0.5)
bi_gru_model_optim = torch.optim.Adam(bi_gru_model.parameters(), lr=learning_rate, weight_decay=1e-5)
bi_gru_model_scheduler = torch.optim.lr_scheduler.StepLR(bi_gru_model_optim, step_size=500, gamma=0.5)
relation_score_aggregate_model_optim = torch.optim.Adam(relation_score_aggregate_model.parameters(),lr=learning_rate,
                                                        weight_decay=1e-5)
relation_score_aggregate_model_scheduler = torch.optim.lr_scheduler.StepLR(relation_score_aggregate_model_optim,
                                                                           step_size=500,gamma=0.5)

# ---------------------------Episode training------------------------------------
print("training...")
last_accuracy = 0.0
best_epoch = 0


for ep in range(episode):
    semantic_embedding_model_scheduler.step(ep)
    relation_score_aggregate_model_scheduler.step(ep)
    bi_gru_model_scheduler.step(ep)

    train_loader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    batch_features, batch_labels = train_loader.__iter__().next()

    sample_labels = []
    for label in batch_labels.numpy():
        if label not in sample_labels:
            sample_labels.append(label)

    sample_attributes = torch.Tensor([all_attributes[i] for i in sample_labels]).squeeze(1)
    class_num = sample_attributes.shape[0]


    out, batch_att = attention_module(batch_features)
    batch_att = batch_att.detach()
    # print("batch_att is--> ", batch_att)
    batch_features = choose_valid_segments(batch_att, batch_features)
    # batch_features = torch.randn([batch_size, 32, 4096])
    batch_features = torch.zeros(batch_size, 34, 4096)
    batch_features = Variable(batch_features)
    sample_attributes = Variable(sample_attributes)

    sample_features = semantic_embedding_model(sample_attributes)
    sample_features_ext = sample_features.unsqueeze(1).repeat(batch_size,1,1)
    # pdb.set_trace()
    # sample_features_ext = sample_features_ext.unsqueeze(0).repeat(batch_size, 1, 1,1)
    batch_features_ext = bi_gru_model(batch_features)
    batch_features_ext = torch.mean(batch_features_ext, dim=1)
    batch_features_ext = batch_features_ext.unsqueeze(1).repeat(1, class_num, 1)
    relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2)

    em_s, em_v = relation_score_aggregate_model(relation_pairs)

    cos_dis = em_s - em_v

    score = np.ones((cos_dis.shape[0], cos_dis.shape[1]))
    for row in range(cos_dis.shape[0]):
        for cul in range(cos_dis.shape[1]):
            a = cos_dis[row, cul, :]

            score[row, cul] = torch.mm(a.unsqueeze(0), a.unsqueeze(0).T)

    score = torch.tensor(score).cuda()
    score = F.softmax(score, 1)
    min_energy = torch.mul(score, one_hot_labels)

    loss_en = 0
    for bs in range(batch_size):
        min = torch.min(score[bs, :])
        max = torch.max(min_energy[bs, :])
        loss_en = loss_en + F.sigmoid(max - min)
    loss_en = loss_en / batch_size

    mse = nn.BCELoss()
    loss_clf = mse(score.float(), one_hot_labels.float())
    loss_En = loss_en + loss_clf
    loss_En.requires_grad_(True)

    loss_cor = mse(out, one_hot_labels)
    loss_cor += 0.0001 * torch.abs(batch_att).sum()

    loss = loss_cor + loss_En

    # relation_scores = relation_scores.view(-1,class_num)
    # print("relation_scores--> ", relation_scores)
    sample_labels = np.array(sample_labels)
    re_batch_labels = []
    for label in batch_labels.numpy():
        index = np.argwhere(sample_labels == label)
        re_batch_labels.append(index[0][0])
    re_batch_labels = torch.LongTensor(re_batch_labels)
    # pdb.set_trace()
    relation_scores = score
    _, predict_labels = torch.max(relation_scores, 1)
    print("re_batch_labels: ", re_batch_labels)
    print("predict_labels: ", predict_labels)
    train_acc = accuracy_score(re_batch_labels, predict_labels)
    mse = nn.BCELoss()
    one_hot_labels = Variable(torch.zeros(batch_size, class_num).scatter_(1,re_batch_labels.view(-1,1),1))

    writer.add_scalars('Loss_group', {'train_loss': loss.data}, ep)
    writer.add_scalars('Acc_group', {"train_acc": train_acc}, ep)
    # pdb.set_trace()

    semantic_embedding_model.zero_grad()
    bi_gru_model.zero_grad()
    relation_score_aggregate_model.zero_grad()

    loss.backward()
    for name, parms in semantic_embedding_model.named_parameters():
        writer.add_histogram(name, parms.clone().cpu().data.numpy(), ep)

    for name, parms in bi_gru_model.named_parameters():
        writer.add_histogram(name, parms.clone().cpu().data.numpy(), ep)

    for name, parms in relation_score_aggregate_model.named_parameters():
        writer.add_histogram(name, parms.clone().cpu().data.numpy(), ep)
    nn.utils.clip_grad_value_(semantic_embedding_model.parameters(), clip_value=0.5)
    nn.utils.clip_grad_value_(bi_gru_model.parameters(), clip_value= 0.5)
    nn.utils.clip_grad_value_(relation_score_aggregate_model.parameters(), clip_value=0.5)

    semantic_embedding_model_optim.step()
    bi_gru_model_optim.step()
    relation_score_aggregate_model_optim.step()

    writer.add_scalars('learning_rate_group', {'semantic_model': semantic_embedding_model_scheduler.get_lr()[0]},
                       ep)
    writer.add_scalars('learning_rate_group', {"relation_module": relation_score_aggregate_model_scheduler.get_lr()[0]},
                       ep)
    writer.add_scalars('learning_rate_group', {'bigru_model': bi_gru_model_scheduler.get_lr()[0]}, ep)

    print("episode [{}/{}]".format(ep + 1, episode), "loss:", loss.data, "acc:", train_acc, "class_num:", class_num)

    if(ep + 1) % 5 == 0:
        print("Testing...")

        def compute_accuracy(test_features, test_labels, test_id, test_attributes):
            # pdb.set_trace()
            # test features shape [3671,32,1,4096]
            test_data = Data.TensorDataset(test_features, test_labels)
            test_batch = 8
            test_loader = Data.DataLoader(test_data, batch_size= test_batch, shuffle=False)
            test_loss = 0
            sample_labels = test_id
            sample_attributes = test_attributes
            class_num = sample_attributes.shape[0]
            test_size = test_features.shape[0]
            print("class_num is:", class_num)
            predict_labels_total = []
            re_batch_labels_total = []
            relation_scores_total = []

            for i, data in enumerate(test_loader):
                batch_features, batch_labels = data
                batch_size = batch_labels.shape[0]

                # batch_features = Variable(batch_features).float()
                batch_att = attention_module(batch_features)
                batch_features = choose_valid_segments(batch_att, batch_features)
                # batch_features = torch.nn.functional.normalize(batch_features)
                batch_features = Variable(batch_features)
                sample_attributes = Variable(sample_attributes)
                sample_features = semantic_embedding_model(sample_attributes)
                sample_features = torch.nn.functional.normalize(sample_features)
                sample_features_ext = sample_features.unsqueeze(1).repeat(batch_size, 1, 1)
                # pdb.set_trace()
                # sample_features_ext = sample_features_ext.unsqueeze(0).repeat(batch_size, 1, 1,
                #                                                               1)
                # pdb.set_trace()
                batch_features_ext = bi_gru_model(batch_features.cpu())
                batch_features_ext = torch.mean(batch_features_ext, dim=1)
                batch_features_ext = batch_features_ext.unsqueeze(1).repeat(1, class_num, 1)
                relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2)

                relation_test = relation_pairs

                em_s, em_v = relation_score_aggregate_model(relation_test)

                cos_dis = em_s - em_v

                score = np.ones((cos_dis.shape[0], cos_dis.shape[1]))
                for row in range(cos_dis.shape[0]):
                    for cul in range(cos_dis.shape[1]):
                        a = cos_dis[row, cul, :]

                        score[row, cul] = torch.mm(a.unsqueeze(0), a.unsqueeze(0).T)

                score = torch.tensor(score).cuda()
                score = F.softmax(score, 1)

                relation_scores = score
                # relation_scores = relation_scores.view(-1, class_num)
                # print("Relation Scores in one batch is:")
                # print(relation_scores)
                sample_labels = np.array(sample_labels)
                re_batch_labels = []
                for label in batch_labels.numpy():
                    index = np.argwhere(sample_labels == label)
                    re_batch_labels.append(index[0][0])
                re_batch_labels = torch.LongTensor(re_batch_labels)
                # pdb.set_trace()

                one_hot_labels = Variable(
                    torch.zeros(batch_size, class_num).scatter_(1, re_batch_labels.view(-1, 1), 1))

                loss_test = mse(relation_scores, one_hot_labels)
                test_loss += loss_test.data
                # print("Testing batch ", i, " loss is: ", loss_test.data)
                # print("relation scores is:", relation_scores)

                _, predict_labels = torch.max(relation_scores.data, 1)
                predict_labels = predict_labels.cpu().numpy()
                re_batch_labels = re_batch_labels.cpu().numpy()

                predict_labels_total = np.append(predict_labels_total, predict_labels)
                re_batch_labels_total = np.append(re_batch_labels_total, re_batch_labels)

                # pdb.set_trace()
            print("Test Episode ", ep+1, "is:", test_loss/i)
            writer.add_scalars("Loss_group",{"test_loss":test_loss/i},ep)

            # Compute per class Acc:
            predict_labels_total = np.array(predict_labels_total, dtype= 'int')
            re_batch_labels_total = np.array(re_batch_labels_total, dtype='int')
            unique_labels = np.unique(re_batch_labels_total)

            acc = 0
            for l in unique_labels:
                idx = np.nonzero(re_batch_labels_total == l)[0]
                acc_per = accuracy_score(re_batch_labels_total[idx], predict_labels_total[idx])
                acc += accuracy_score(re_batch_labels_total[idx], predict_labels_total[idx])
                print("acc for class ",l,"is:", acc_per)
                # pdb.set_trace()
            acc = acc / unique_labels.shape[0]
            return acc


        zsl_accuracy = compute_accuracy(test_features, test_labels, test_id, test_attributes)
        print('zsl:', zsl_accuracy)

        writer.add_scalars('Acc_group', {'episode_test_acc': zsl_accuracy}, ep)
        if zsl_accuracy > last_accuracy:
            print("Get better performance network, Saving model...")
            print("better zsl_accuracy is:", zsl_accuracy,"   episode:",ep)
            # save networks
            torch.save(semantic_embedding_model.state_dict(),
                       "semantic_embedding_model.pkl")
            torch.save(bi_gru_model.state_dict(), "bi_gru_model.pkl")
            torch.save(relation_score_aggregate_model.state_dict(),
                       "relation_score_aggregate_model.pkl")

            last_accuracy = zsl_accuracy
            best_epoch = ep
        print("Current Best Acc is: ", last_accuracy,"--->Epoch ", best_epoch)
print("Best Acc on test set is:", last_accuracy)