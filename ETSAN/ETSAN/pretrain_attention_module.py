import torch
import torch.nn as nn
import random
import pickle

from sklearn.model_selection import train_test_split
from torch import optim
import torch.utils.data as Data
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import numpy as np
# np.set_printoptions(threshold=np.inf)
from utils.utils import validate, show_confMat
import sys
sys.path.append("/home/xingmeng/qc/Attention/AttentionModule")
from ZSLModel1003_ucf import *
from sklearn.metrics import accuracy_score
# torch.set_printoptions(threshold=np.inf)

num_classes = 51
num_segments = 32
# log
result_dir = os.path.join("..", "..", "attention_module_Result")
print("result_dir: ",result_dir)
now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')

log_dir = os.path.join(result_dir, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

writer = SummaryWriter(log_dir=log_dir)

classes_name = ['brush_hair', 'cartwheel', 'catch', 'chew', 'clap', 'climb', 'climb_stairs', 'dive', 'draw_sword',
                'dribble', 'drink', 'eat', 'fall_floor', 'fencing', 'flic_flac', 'golf', 'handstand', 'hit', 'hug',
                'jump', 'kick', 'kick_ball', 'kiss', 'laugh', 'pick', 'pour', 'pullup', 'punch', 'push', 'pushup',
                'ride_bike', 'ride_horse', 'run', 'shake_hands', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit',
                'situp', 'smile', 'smoke', 'somersault', 'stand', 'swing_baseball', 'sword', 'sword_exercise',
                'talk', 'throw', 'turn', 'walk', 'wave']
print(classes_name)


def adjust_learning_rate(optimizer, epoch):
    lr = 0.0001 * (0.1 ** (epoch // 80))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Module(nn.Module):
    def __init__(self, num_classes=51):
        super(Module, self).__init__()
        self.num_classes = num_classes
        self.model_name = "Module"
        self.fc1 = nn.Linear(4096, 256)
        # nn.init.normal_(self.fc1.weight, std=0.001)
        # nn.init.constant_(self.fc1.bias, 0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        # nn.init.normal_(self.fc2.weight, std=0.001)
        # nn.init.constant_(self.fc2.bias, 0)
        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Linear(4096, num_classes)
        # nn.init.normal_(self.fc3.weight, std=0.001)
        # nn.init.constant_(self.fc3.bias, 0)
        self.softmax = nn.Softmax()

    def forward(self, inp):
        #print("Input feature size is:", inp.size())
        # inp = inp.squeeze(dim=2)
        # print("Squeezing:",inp.size())
        x = inp
        inp = self.fc1(inp)
        inp = self.relu(inp)
        inp = self.fc2(inp)
        #print("Before activation, inp is:",inp)
        #print("inp size is:",inp.size())
        inp = self.softmax(inp)
        x = inp * x
        #print("attention_weights size is:", inp.size())
        #print("attention_weights is:",inp)
        #print("Before sum attention, x size is:",x.size())
        x = torch.sum(x, dim=1)
        #print("After attention module, x size is:", x.size())
        x = self.fc3(x)
        x = self.sigmoid(x)
        #print("Output size is:", x.size())
        return x, inp

df_load_feature_list = open("all_class_videos_feature_list_51_6.22.pickle",'rb')
all_class_videos_feature_list = pickle.load(df_load_feature_list)


# ------------------------------Loading dataset---------------------------------
"""Loading dataSet..."""
"""
for segments, a tuple to show the segment and its label
"""
all_segments = []
label_segments = []
for i in range(len(all_class_videos_feature_list)):
    each_class_video_segments = all_class_videos_feature_list[i]
    for j in each_class_video_segments:
        # each_video segments, Use only two segments
        j = torch.stack(j)
        # print(type(j))
        all_segments.append(j)
        one_hot = torch.zeros(51)
        one_hot[i] = 1
        label_segments.append(one_hot)
print("Length of label_segments:", len(label_segments))
print("Length of all_segments:", len(all_segments))
X_video_data = all_segments
Y_video_data = label_segments
X_video_train, X_video_test, Y_video_train, Y_video_test = train_test_split(X_video_data, Y_video_data, random_state=0,
                                                                            test_size=0.2)

# X_video_train, X_video_val,Y_video_train, Y_video_val = train_test_split(X_video_train, Y_video_train,random_state=0,
#                                                                          test_size=0.25)
X_video_val,X_video_test,Y_video_val, Y_video_test = train_test_split(X_video_test,Y_video_test, random_state=0,
                                                                      test_size=0.5)
print(len(X_video_train))
print(type(X_video_train[0]))
print(type(X_video_train))
print(len(X_video_val))

"""
Use DataLoader to pack own data
"""
# Train data
batch_size = 8
X_video_train = X_video_data
Y_video_train = Y_video_data
X_video_train = torch.stack(X_video_train)
X_video_train.detach_()
Y_video_train = torch.stack(Y_video_train)
Y_video_train.detach_()
torch_dataset_train = Data.TensorDataset(X_video_train, Y_video_train)
# Put data into DataLoader
train_loader = Data.DataLoader(
    dataset=torch_dataset_train,
    batch_size=batch_size,
    shuffle=True
)
X_video_val = torch.stack(X_video_val)
Y_video_val = torch.stack(Y_video_val)
torch_dataset_val = Data.TensorDataset(X_video_val, Y_video_val)
val_loader = Data.DataLoader(
    dataset=torch_dataset_val,
    batch_size=batch_size,
    shuffle=True
)

X_video_test = torch.stack(X_video_test)
Y_video_test = torch.stack(Y_video_test)
torch_dataset_test = Data.TensorDataset(X_video_test, Y_video_test)
test_loader = Data.DataLoader(
    dataset=torch_dataset_test,
    batch_size = batch_size,
    shuffle= True
)


# ------------------------------Define loss function,optimizer and other parameters----------------
num_classes = 51
epochs = 50
learning_rate = 10 ** -4

# ------------------------------Training the module-------------------------------

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Using gpu or cpu:", device)
model = Module()  # Create a module
criterion = torch.nn.BCELoss()
# model.initialize_weights()  # Initialize weights
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 设置学习率下降策略
schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
print("model is:", model)  # print model
print("Start training Attention Model...")
acc_test = 0.0
for epoch in range(epochs):
    print("Current epoch is:",epoch)
    print('Epoch {}/{}'.format(epoch, epochs-1))
    print('-' * 10)
    tot_loss = 0.0  # Record loss in one epoch
    correct = 0.0
    total = 0.0
    # schedule.step(epoch)  # Update learning rate
    # learning_rate = adjust_learning_rate(optimizer, epoch)
    #print("learning_rate is:",learning_rate)
    for i, data in enumerate(train_loader):
        # Get input data and label
        inputs, labels = data
        #print("Inputs size is:",inputs.size())
        #print("Inputs is:",inputs)
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
        optimizer.zero_grad()
        out_puts,attention_weights = model(inputs)
        out_puts = out_puts.squeeze(dim=1)
        #print("attention_weights type:", type(attention_weights))
        #print("attention_weights size:", attention_weights.size())
        print("attention_weights is:",attention_weights)
        #print("outputs is:",out_puts)
        #print("outputs size:",out_puts.size())
        #print("labels is:",labels)
        #print("label size:",labels.size())
        # out_puts =torch.max(out_puts,-1)[1]
        loss = criterion(out_puts, labels)
        #print("classification loss is:",loss)
        loss += 0.0001 * torch.abs(attention_weights).sum()
        #print("Add Sparsity loss:",loss)
        tot_loss += loss.data[0]
        #print("toto_loss is:",tot_loss)
        loss.backward()
        optimizer.step()

        # Gather predicted info
        predicted = torch.max(out_puts.data, 1)[1]
        labels = torch.max(labels.data,1)[1]
        total += labels.size(0)
        #print("predicted is:",predicted)
        #print("labels is :", labels)
        #print("labels size is:",labels.size())
        #print("total is:", total)
        correct += (predicted.cpu() == labels.cpu()).squeeze().sum().numpy()
        #print("correct is:", correct)

    tot_loss = tot_loss / i  # get each epoch loss
    print("Training: Epoch[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
        epoch + 1, epochs, tot_loss, correct / total))

    # Record training loss
    writer.add_scalars('Loss_group', {'train_loss': tot_loss}, epoch)
    # Record learning rate
    writer.add_scalar('learning_rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
    # Record Accuracy
    writer.add_scalars("Accuracy_group", {'train_acc': correct / total}, epoch)
        

    # Each epoch, record gradient and weights
    for name, layer in model.named_parameters():
        writer.add_histogram(name + "_grad", layer.grad.cpu().data.numpy(), epoch)
        writer.add_histogram(name + "_data", layer.cpu().data.numpy(), epoch)
    writer.add_graph(model)

    # --------------------Model performance on validate dataset Every Epoch-------------------
    tot_loss_val = 0.0
    cls_num = len(classes_name)
    conf_mat = np.zeros([cls_num, cls_num])
    model.eval()
    total_val = 0
    correct_val = 0

    for i, data in enumerate(val_loader):
        inputs, labels = data
        #print("val Inputs size is:",inputs.size())
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

        # forward
        out_puts, attention_weights_val = model(inputs)
        # out_puts.detach_()
        out_puts = out_puts.squeeze(dim=1)
        print("attention_weights in val is:", attention_weights_val)

        # Calculate loss
        #print("out_puts in val is :", out_puts)
        #print("out_puts size in val is:", out_puts.size())
        #print("labels in val is:", labels)
        #print("labels size in val is:", labels.size())
        loss = criterion(out_puts, labels)
        #print("loss is:",loss)
        loss += 0.0001 * torch.abs(attention_weights_val).sum()
        #print("loss is:",loss)
        tot_loss_val += loss.data[0]

        # Statistic
        predicted = torch.max(out_puts.data, 1)[1]
        labels = torch.max(labels.data, 1)[1]
        total_val += labels.size(0)
        #print("predicted in val is:", predicted)
        #print("labels in val is :", labels)
        #print("labels in val size is:", labels.size())
        #print("total in val is:", total)
        correct_val += (predicted.cpu() == labels.cpu()).squeeze().sum().numpy()
    schedule.step(tot_loss_val)
    print("{} set Accuracy:{:.2%}".format('val', correct_val / total_val))
    writer.add_scalars('Loss_group', {'val_loss': tot_loss_val / len(val_loader)}, epoch)
    writer.add_scalars('Accuracy_group', {'val_acc': correct_val / total_val}, epoch)

    if (epoch+1) % 2 == 0:
        tot_loss_test = 0.0
        cls_num = len(classes_name)
        conf_mat = np.zeros([cls_num, cls_num])
        model.eval()
        total_test = 0
        correct_test = 0
        for i, data in enumerate(test_loader):
            inputs, labels = data
            #print("test Inputs size is:", inputs.size())
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

            # forward
            out_puts,attention_weights_test = model(inputs)
            # out_puts.detach_()
            out_puts = out_puts.squeeze(dim=1)
            print("attention_weights in test is:", attention_weights_test)

            # Calculate loss
            #print("out_puts in test is :",out_puts)
            #print("out_puts size in test is:", out_puts.size())
            #print("labels in test is:",labels)
            #print("labels size in test is:",labels.size())
            loss = criterion(out_puts, labels)
            loss += 0.0001 * torch.abs(attention_weights_test).sum()
            tot_loss_test += loss.data[0]

            # Statistic
            predicted = torch.max(out_puts.data, 1)[1]
            labels = torch.max(labels.data, 1)[1]
            total_test += labels.size(0)
            #print("predicted in test is:", predicted)
            #print("labels in test is :", labels)
            #print("labels in test size is:", labels.size())
            #print("total in test is:", total)
            correct_test += (predicted.cpu() == labels.cpu()).squeeze().sum().numpy()
        
        print("{} set Accuracy:{:.2%}".format('test', correct_test / total_test))
        if correct_test / total_test > acc_test:
            # Save Model
            writer.add_graph(model,(inputs, ))
            net_save_path = os.path.join(log_dir, "attention_model_params.pkl")
            torch.save(model.state_dict(), net_save_path)
            acc_test = correct_test / total_test
            print("Saving model for epoch:", epoch)
print("Best Current Acc on Test is:", acc_test)
print("Finished Training!")

# writer.add_graph(model,(inputs, ))
#net_save_path = os.path.join(log_dir, "attention_model_params.pkl")
#torch.save(model.state_dict(), net_save_path)
def adjust_learning_rate(optimizer, epoch):
    lr = 0.0001 * (0.1 ** (epoch // 80))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
