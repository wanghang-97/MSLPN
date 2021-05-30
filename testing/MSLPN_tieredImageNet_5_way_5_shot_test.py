"""
Author: WangHang

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pickle as pkl
import os
import glob
from tqdm import tqdm
import pandas as pd
from scipy.ndimage import imread
from scipy.misc import imresize
from torch.optim.lr_scheduler import StepLR
import argparse
import scipy as sp
import scipy.stats
import scipy
from torchvision import transforms
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

trans = transforms.Compose(
    [
        transforms.ToTensor(),
    ])
class dataset_tiered(object):
    def __init__(self, n_examples, n_episodes, split, args):
        self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
        self.n_examples = n_examples
        self.n_episodes = n_episodes
        self.split      = split
        self.ratio      = args['ratio']
        self.seed       = args['seed']
        self.root_dir   = './tieredImagenet'

        self.iamge_data = []
        self.dict_index_label = []
        self.dict_index_unlabel = []
    


    def load_data_pkl(self):
        """
            load the pkl processed tieredImagenet into label,unlabel
            maintain label,unlabel data dictionary for indexes
        """
        labels_name = '{}/{}_labels.pkl'.format(self.root_dir, self.split)
        images_name = '{}/{}_images.npz'.format(self.root_dir, self.split)
        print('labels:',labels_name)
        print('images:',images_name)
        
        # decompress images if npz not exits
        if not os.path.exists(images_name):
            png_pkl = images_name[:-4] + '_png.pkl'
            if os.path.exists(png_pkl):
                decompress(images_name, png_pkl)
            else:
                raise ValueError('path png_pkl not exits')
        
        if os.path.exists(images_name) and os.path.exists(labels_name):
            try:
                with open(labels_name) as f:
                    data            = pkl.load(f)
                    label_specific  = data["label_specific"]
            except:
                with open(labels_name, 'rb') as f:
                    data            = pkl.load(f, encoding='bytes')
                    label_specific  = data['label_specific']
            print('read label data:{}'.format(len(label_specific)))
        labels      = label_specific

        with np.load(images_name, mmap_mode="r", encoding='latin1') as data:
            image_data = data["images"]    
            print('read image data:{}'.format(image_data.shape))
        
        n_classes   = np.max(labels)+1

        print('n_classes:{}, n_label:{}%, n_unlabel:{}%'.format(n_classes,self.ratio*100,(1-self.ratio)*100))
        dict_index_label    = {} 
        dict_index_unlabel  = {}
        
        for cls in range(n_classes):
            idxs = np.where(labels==cls)[0]
            nums = idxs.shape[0]
            np.random.RandomState(self.seed).shuffle(idxs) 

            n_label     = int(self.ratio*nums)
            n_unlabel   = nums-n_label

            dict_index_label[cls]   = idxs[0:n_label]
            dict_index_unlabel[cls] = idxs[n_label:]
    
        self.image_data         = image_data
        self.dict_index_label   = dict_index_label
        self.dict_index_unlabel = dict_index_unlabel
        self.n_classes          = n_classes
        print(dict_index_label[0])
        print(dict_index_unlabel[0])


    def next_data(self, n_way, n_shot, n_query, num_unlabel=0, n_distractor=0, train=True):
        """
            get support,query,unlabel data from n_way
            get unlabel data from n_distractor
        """
        support     = np.zeros([n_way, n_shot, self.channels, self.im_height, self.im_width], dtype=np.float32)
        query       = np.zeros([n_way, n_query, self.channels, self.im_height, self.im_width], dtype=np.float32)
        if num_unlabel>0:
            unlabel = np.zeros([n_way+n_distractor, num_unlabel, self.channels, self.im_height, self.im_width], dtype=np.float32)
        else:
            unlabel = []
            n_distractor = 0

        selected_classes = np.random.permutation(self.n_classes)[:n_way+n_distractor]
        for i, cls in enumerate(selected_classes[0:n_way]): # train way
            # labled data
            idx             = self.dict_index_label[cls]
            np.random.RandomState().shuffle(idx)
            idx1            = idx[0:n_shot + n_query]
            support[i]      = self.image_data[idx1[:n_shot]]
            query[i]        = self.image_data[idx1[n_shot:]]

            # unlabel
            if num_unlabel>0:
                idx         = self.dict_index_unlabel[cls]
                np.random.RandomState().shuffle(idx)
                idx2        = idx[0:num_unlabel]
                unlabel[i]  = self.image_data[idx2]
        
        for j,cls in enumerate(selected_classes[self.n_classes:]): 
            idx             = self.dict_index_unlabel[cls]
            np.random.RandomState().shuffle(idx)
            idx3            = idx[0:num_unlabel]
            unlabel[i+j]    = self.image_data[idx3]
        
        support_labels      = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_shot)).astype(np.uint8)
        query_labels        = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)
        return support, support_labels, query, query_labels, unlabel
def compress(path, output):
    with np.load(path, mmap_mode="r") as data:
        images = data["images"]
        array = []
        for ii in tqdm(six.moves.xrange(images.shape[0]), desc='compress'):
            im = images[ii]
            im_str = cv2.imencode('.png', im)[1]
            array.append(im_str)
    with open(output, 'wb') as f:
        pkl.dump(array, f, protocol=pkl.HIGHEST_PROTOCOL)


def decompress(path, output):
    with open(output, 'rb') as f:
        array = pkl.load(f, encoding='bytes')
    images = np.zeros([len(array), 3, 84, 84], dtype=np.float32)
    for ii, item in tqdm(enumerate(array), desc='decompress'):
        im = cv2.imdecode(item, 1)
        im = trans(im)
        images[ii] = im
    np.savez(path, images=images)
# Define Model
class CNNEncoder(nn.Module):
    def __init__(self,):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
#             nn.MaxPool2d(2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
#             nn.MaxPool2d(2),
        )
        self.layer_scale2 = nn.MaxPool2d(2)
        self.layer_scale3 = nn.Sequential(
            nn.Conv2d(64,64,3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.layer_scale4 = nn.Sequential(
            nn.Conv2d(64,64,5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.layer_scale5 = nn.Sequential(
            nn.Conv2d(64,64,(1,7)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64,64,(7,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

    def forward(self, x):
        layer_out1 = self.layer1(x)
        layer_out2 = self.layer2(layer_out1)
        layer_out3 = self.layer3(layer_out2)
        out1 = self.layer4(layer_out3)
        out2 = self.layer_scale2(out1)
        out3 = self.layer_scale3(out1)
        out4 = self.layer_scale4(out1)
        out5 = self.layer_scale5(out1)
        return layer_out3, out1, out2, out3, out4, out5


    
    
class GraphConstruction_scale(nn.Module):
    def __init__(self):
        super(GraphConstruction_scale, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
        )
        self.fc1 = nn.Linear(6 * 6, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        layer_out1 = self.layer1(x)
        layer_out2 = self.layer2(layer_out1)
        # flatten
        layer_out2 = layer_out2.view(layer_out2.size(0), -1)
        fc_out1 = F.relu(self.fc1(layer_out2))
        fc_out2 = self.fc2(fc_out1)
        out = fc_out2.view(fc_out2.size(0), -1)
        return out


class GraphConstruction_scale1(nn.Module):
    def __init__(self):
        super(GraphConstruction_scale1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
        )
        self.fc1 = nn.Linear(6 * 6, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        layer_out1 = self.layer1(x)
        layer_out2 = self.layer2(layer_out1)
        # flatten
        layer_out2 = layer_out2.view(layer_out2.size(0), -1)
        fc_out1 = F.relu(self.fc1(layer_out2))
        fc_out2 = self.fc2(fc_out1)
        out = fc_out2.view(fc_out2.size(0), -1)
        return out
class GraphConstruction_scale2(nn.Module):
    def __init__(self):
        super(GraphConstruction_scale2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
        )
        self.fc1 = nn.Linear(4 * 4, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        layer_out1 = self.layer1(x)
        layer_out2 = self.layer2(layer_out1)
        # flatten
        layer_out2 = layer_out2.view(layer_out2.size(0), -1)
        fc_out1 = F.relu(self.fc1(layer_out2))
        fc_out2 = self.fc2(fc_out1)
        out = fc_out2.view(fc_out2.size(0), -1)
        return out
class GraphConstruction_scale3(nn.Module):
    def __init__(self):
        super(GraphConstruction_scale3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
        )
        self.fc1 = nn.Linear(6 * 6, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        layer_out1 = self.layer1(x)
        layer_out2 = self.layer2(layer_out1)
        # flatten
        layer_out2 = layer_out2.view(layer_out2.size(0), -1)
        fc_out1 = F.relu(self.fc1(layer_out2))
        fc_out2 = self.fc2(fc_out1)
        out = fc_out2.view(fc_out2.size(0), -1)
        return out
class GraphConstruction_scale4(nn.Module):
    def __init__(self):
        super(GraphConstruction_scale4, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
        )
        self.fc1 = nn.Linear(5 * 5, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        layer_out1 = self.layer1(x)
        layer_out2 = self.layer2(layer_out1)
        # flatten
        layer_out2 = layer_out2.view(layer_out2.size(0), -1)
        fc_out1 = F.relu(self.fc1(layer_out2))
        fc_out2 = self.fc2(fc_out1)
        out = fc_out2.view(fc_out2.size(0), -1)
        return out
class GraphConstruction_scale5(nn.Module):
    def __init__(self):
        super(GraphConstruction_scale5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
        )
        self.fc1 = nn.Linear(5 * 5, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        layer_out1 = self.layer1(x)
        layer_out2 = self.layer2(layer_out1)
        # flatten
        layer_out2 = layer_out2.view(layer_out2.size(0), -1)
        fc_out1 = F.relu(self.fc1(layer_out2))
        fc_out2 = self.fc2(fc_out1)
        out = fc_out2.view(fc_out2.size(0), -1)
        return out
class LabelPropagation(nn.Module):
    def __init__(self, args):
        super(LabelPropagation, self).__init__()
        self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
        self.h_dim, self.z_dim = args['h_dim'], args['z_dim']
        self.args = args
        self.encoder = CNNEncoder()
        self.graph = GraphConstruction_scale()
        self.graph1 = GraphConstruction_scale1()
        self.graph2 = GraphConstruction_scale2()
        self.graph3 = GraphConstruction_scale3()
        self.graph4 = GraphConstruction_scale4()
        self.graph5 = GraphConstruction_scale5()

        # learn sigma
        if args['rn'] == 300:
            self.alpha = torch.tensor([args['alpha']], requires_grad=False).cuda()
        # learning sigma and alpha
        elif args['rn'] == 30:
            self.alpha = nn.Parameter(torch.tensor([args['alpha']]).cuda(), requires_grad=True)
    
    def propagation(self,emb_all, N, graph, eps, num_classes, num_support, num_queries, s_labels, q_labels):
        if self.args['rn'] in [30, 300]:
            N = emb_all.shape[0]
            emb_all_f = emb_all.view(N,-1)
            self.sigma = graph(emb_all)
            emb_all = emb_all_f / (self.sigma + eps)
            emb1 = torch.unsqueeze(emb_all, 1)  
            emb2 = torch.unsqueeze(emb_all, 0) 

            W = ((emb1 - emb2) ** 2).mean(2)  
            W = torch.exp(-W / 2)
        # keep top-K
        if self.args['k'] > 0:
            topk, indicies = torch.topk(W, self.args['k'])
            mask = torch.zeros_like(W)
            mask = mask.scatter(1, indicies, 1)
            mask = ((mask + torch.t(mask)) > 0).type(torch.float32)  
            mask = mask.type(torch.float32)
            W = W * mask
        # Normalize
        D = W.sum(1)
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N)
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(N, 1)
        S = D1 * W * D2
        # Label propagation
        ys = s_labels
        yu = torch.zeros(num_classes * num_queries, num_classes).cuda()
        y = torch.cat((ys, yu), 0) 
        F = torch.matmul(torch.inverse(torch.eye(N).cuda() - self.alpha * S + eps), y)
        Fq = F[num_classes * num_support:, :]  

        crossentropy loss
        crossentropy = nn.CrossEntropyLoss().cuda()
        gt = torch.argmax(torch.cat((s_labels, q_labels), 0), 1) 
        loss = crossentropy(F, gt)
        return loss,Fq
    
    def forward(self, inputs):
        # support:  n_way * n_shot *3*84*84
        # query:    n_way * n_query *3*84*84
        # s_labels: n_way * n_shot * n_way,one-hot
        # q_labels: n-way * n_query * n_way,one-hot
        eps = np.finfo(float).eps  
        [support, s_labels, query, q_labels] = inputs
        num_classes = s_labels.shape[1]
        num_support = int(s_labels.shape[0] / num_classes)
        num_queries = int(query.shape[0] / num_classes)

        input = torch.cat((support, query), 0)
        emb_all0, emb_all1,emb_all2,emb_all3,emb_all4,emb_all5 = self.encoder(input)
        N = 100
        
        loss1,Fq1 = self.propagation(emb_all1,N,self.graph1,eps, num_classes, num_support, num_queries, s_labels, q_labels)
        loss2,Fq2 = self.propagation(emb_all2,N,self.graph2,eps, num_classes, num_support, num_queries, s_labels, q_labels)
        loss3,Fq3 = self.propagation(emb_all3,N,self.graph3,eps, num_classes, num_support, num_queries, s_labels, q_labels)
        loss4,Fq4 = self.propagation(emb_all4,N,self.graph4,eps, num_classes, num_support, num_queries, s_labels, q_labels)
        loss5,Fq5 = self.propagation(emb_all5,N,self.graph5,eps, num_classes, num_support, num_queries, s_labels, q_labels)
        loss = loss1 + loss2 + loss3 + loss4 + loss5
        Fq = Fq1 + Fq2 + Fq3 + Fq4 + Fq5
        # acc
        predq = torch.argmax(Fq, 1)
        gtq = torch.argmax(q_labels, 1)
        correct = (predq == gtq).sum()
        total = num_queries * num_classes
        acc = 1.0 * correct.float() / float(total)
        return loss, acc

# Define parameters
parser = argparse.ArgumentParser(description='Train multi-scale label propagation network')
# model params
n_examples = 600
parser.add_argument('--x_dim', type=str, default="84,84,3", metavar='XDIM',
                    help='input image dims')
parser.add_argument('--h_dim', type=int, default=64, metavar='HDIM',
                    help="dimensionality of hidden layers (default: 64)")
parser.add_argument('--z_dim', type=int, default=64, metavar='ZDIM',
                    help="dimensionality of output channels (default: 64)")

# training hyper-parameters
n_episodes = 100  # test interval
parser.add_argument('--n_way', type=int, default=5, metavar='NWAY',
                    help="nway")
parser.add_argument('--n_shot', type=int, default=5, metavar='NSHOT',
                    help="nshot")
parser.add_argument('--n_query', type=int, default=15, metavar='NQUERY',
                    help="nquery")
parser.add_argument('--n_epochs', type=int, default=2100, metavar='NEPOCHS',
                    help="nepochs")
# test hyper-parameters
parser.add_argument('--n_test_way', type=int, default=5, metavar='NTESTWAY',
                    help="ntestway")
parser.add_argument('--n_test_shot', type=int, default=5, metavar='NTESTSHOT',
                    help="ntestshot")
parser.add_argument('--n_test_query', type=int, default=15, metavar='NTESTQUERY',
                    help="ntestquery")

# optimization params
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help="base learning rate")
parser.add_argument('--step_size', type=int, default=10000, metavar='STEPSIZE',
                    help="lr decay step size")
parser.add_argument('--gamma', type=float, default=0.5, metavar='GAMMA',
                    help="decay rate")
parser.add_argument('--patience', type=int, default=200, metavar='PATIENCE',
                    help="train patience until stop")

# dataset params
parser.add_argument('--dataset', type=str, default='mini', metavar='DATASET',
                    help="mini or tiered")
parser.add_argument('--ratio', type=float, default=1.0, metavar='RATIO',
                    help="ratio of labeled data each class")
# label propagation params
parser.add_argument('--alg', type=str, default='MSLPN', metavar='ALG',
                    help="algorithm used, MSLPN")
parser.add_argument('--k', type=int, default=20, metavar='K',
                    help="top k in constructing the graph W")
parser.add_argument('--sigma', type=float, default=0.25, metavar='SIGMA',
                    help="Initial sigma in label propagation")
parser.add_argument('--alpha', type=float, default=0.99, metavar='ALPHA',
                    help="Initial alpha in label propagation")
parser.add_argument('--rn', type=int, default=300, metavar='RN',
                    help="graph construction types: "
                         "300: sigma is learned, alpha is fixed" +
                         "30:  both sigma and alpha learned")

# save and restore params
parser.add_argument('--seed', type=int, default=1000, metavar='SEED',
                    help="random seed for code and data sample")
parser.add_argument('--exp_name', type=str, default='exp', metavar='EXPNAME',
                    help="experiment name")
parser.add_argument('--iters', type=int, default=0, metavar='ITERS',
                    help="iteration to restore params")

args = vars(parser.parse_args(args=[]))
im_width, im_height, channels = list(map(int, args['x_dim'].split(',')))
torch.backends.cudnn.benchmark = True

iters = args['iters']
repeat = 10
n_test_episodes = 600
n_test_way = args['n_test_way']
n_test_shot = args['n_test_shot']
n_test_query = args['n_test_query']
alg = args['alg']

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    # 求均值
    m = np.mean(a)
    # 求标准误差（均方根误差）
    se = scipy.stats.sem(a)
    # t分布
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m,h
# step1: initial dataloader
print('init data loader')
args_data = {}
args_data['x_dim'] = args['x_dim']
args_data['ratio'] = args['ratio']
args_data['seed'] = args['seed']
loader_test = dataset_tiered(n_examples, n_episodes, 'test', args_data)
loader_test.load_data_pkl()
# step2: initial MSLPN
print("init neural networks")
model = LabelPropagation(args)
model.load_state_dict(torch.load('tieredImageNet_checkpoint_five_shot/%s/models/%s_%d_model.pkl' %(args['exp_name'],alg,135900)))
model.cuda()
# step3: testing
print("Testing...")
all_acc = []
all_std = []
all_ci95 = []

ce_list = []

for rep in range(repeat):
    list_acc = []

    for epi in tqdm(range(n_test_episodes), desc='test:{}'.format(rep)):

        model.eval()

        # sample data for next batch
        support, s_labels, query, q_labels, unlabel = loader_test.next_data(n_test_way, n_test_shot, n_test_query)
        support = np.reshape(support, (support.shape[0] * support.shape[1],) + support.shape[2:])
        support = torch.from_numpy(support)
        query = np.reshape(query, (query.shape[0] * query.shape[1],) + query.shape[2:])
        query = torch.from_numpy(query)
        s_labels = torch.from_numpy(np.reshape(s_labels, (-1,)))
        q_labels = torch.from_numpy(np.reshape(q_labels, (-1,)))
        s_labels = s_labels.type(torch.LongTensor)
        q_labels = q_labels.type(torch.LongTensor)
        s_onehot = torch.zeros(n_test_way * n_test_shot, n_test_way).scatter_(1, s_labels.view(-1, 1), 1)
        q_onehot = torch.zeros(n_test_way * n_test_query, n_test_way).scatter_(1, q_labels.view(-1, 1), 1)
        with torch.no_grad():
            inputs = [support.cuda(), s_onehot.cuda(), query.cuda(), q_onehot.cuda()]
            loss, acc = model(inputs)

        list_acc.append(acc.item())

    mean_acc = np.mean(list_acc)
    std_acc  = np.std(list_acc)
    ci95 = 1.96*std_acc/np.sqrt(n_test_episodes)
    m,ci = mean_confidence_interval(list_acc)

    print('label, acc:{:.4f},std:{:.4f},ci95:{:.4f},ci:{:.4f}'.format(mean_acc, std_acc, ci95, ci))
    all_acc.append(mean_acc)
    all_std.append(std_acc)
    all_ci95.append(ci95)

ind = np.argmax(all_acc)
print('Max acc:{:.5f}, std:{:.5f}, ci95: {:.5f}'.format(all_acc[ind], all_std[ind], all_ci95[ind]))
print('Avg over {} runs: mean:{:.5f}, std:{:.5f}, ci95: {:.5f}'.format(repeat,np.mean(all_acc),np.mean(all_std),np.mean(all_ci95)))