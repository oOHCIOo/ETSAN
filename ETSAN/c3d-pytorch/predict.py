""" How to use C3D network. """
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import transforms
from os.path import join
from glob import glob

from skimage.transform import resize

import skimage.io as io
from C3D_model2 import C3D
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def get_sport_clip(clip_path, verbose=True):
    """
    Loads a clip to be fed to C3D for classification.
    TODO: should I remove mean here?
    
    Parameters
    ----------
    clip_path: list
        the list of frames in  clip
    verbose: bool
        if True, shows the unrolled clip (default is True).
    Returns
    -------
    Tensor
        a pytorch batch (n, ch, fr, h, w).
    """
    clip = clip_path
    # 图片归一化问题
    # clip = np.array([np.load(frame) for frame in clip])
    # std_image = (np.load(frame) - np.min(np.load(frame))) / (np.max(np.load(frame)) - np.min(np.load(frame)))
    # normalize = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # for frame in clip:
    #     img = np.load(frame)
    #     img_norm = normalize(img)
    #     img_norm = img_norm.transpose(1,2,0)
    #     pass
    # clip = np.array([resize(normalize(np.load(frame)), output_shape=(112, 200), preserve_range=True) for frame in clip])
    clip = np.array([resize(np.load(frame), output_shape=(112, 200), preserve_range=True) for frame in clip])
    # clip = np.array([resize(io.imread(frame), output_shape=(112, 200), preserve_range=True) for frame in clip])
    # clip_std = np.array([(np.load(frame) - np.min(np.load(frame))) / (np.max(np.load(frame)) - np.min(np.load(frame))) for frame in clip])
    clip = clip[:, :, 44:44+112, :]  # crop centrally


    # clip = np.array(normalize(clip))


    # if verbose:
    #     clip_img = np.reshape(clip.transpose(1, 0, 2, 3), (112, 16 * 112, 3))
    #     io.imshow(clip_img.astype(np.uint8))
    #     io.show()


    clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
    clip = np.expand_dims(clip, axis=0)  # batch axis
    clip = np.float32(clip)

    return torch.from_numpy(clip)


def read_labels_from_file(filepath):
    """
    Reads Sport1M labels from file
    
    Parameters
    ----------
    filepath: str
        the file.
        
    Returns
    -------
    list
        list of sport names.
    """
    with open(filepath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def main(clip_path):
    """

    :param clip_path: list, the frames' path of one segment
    :return:
    """

    # load a clip to be predicted
    X = get_sport_clip(clip_path)
    X = Variable(X)
    X = X.to(device)

    # get network pretrained model
    net = C3D()
    net.load_state_dict(torch.load('/home/qc/qc/All_codes/attention_module_gcn/AttentionModule/c3d-pytorch/c3d.pickle'))
    # net.load_state_dict(torch.load('/home/qc/qc/All_codes/attention_module_gcn/AttentionModule/pretrainedModels/ucf101-caffe.pth'))
    net.to(device)
    net.eval()

    # perform prediction
    feature = net(X)
    # feature = feature.data.cpu().numpy()
    return feature

# def main():
#     X = get_sport_clip('roger')
#     X = Variable(X)
#     # X = X.cuda()
#
#     # get network pretrained model
#     net = C3D()
#     net.load_state_dict(torch.load('c3d.pickle'))
#     # net.cuda()
#     net.eval()
#
#     # perform prediction
#     prediction = net(X)
#     print(prediction)
#
#     # read labels
#     labels = read_labels_from_file('labels.txt')
#
#     # print top predictions
#     top_inds = prediction[0].argsort()[::-1][:5]  # reverse sort and take five largest items
#     print('\nTop 5:')
#     for i in top_inds:
#         print('{:.5f} {}'.format(prediction[0][i], labels[i]))

# entry point
if __name__ == '__main__':
    main()
