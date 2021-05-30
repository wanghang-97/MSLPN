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
# Loading Dataset
class dataset_mini(object):
    def __init__(self, n_examples, n_episodes, split, args):
        self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
        self.n_examples = n_examples
        self.n_episodes = n_episodes
        self.split = split
        self.ratio = args['ratio']  
        self.seed = args['seed']  
        self.root_dir = './mini-imagenet/'
        self.n_label = int(self.ratio * self.n_examples)
        self.n_unlabel = self.n_examples - self.n_label
        self.dataset_l = []
        self.dataset_u = []
        self.args = args

    def load_data(self):
        print('Loading {} dataset'.format(self.split))
        data_split_path = os.path.join(self.root_dir, '{}.csv'.format(self.split))
        data = pd.read_csv(data_split_path)
        data_classes = np.unique(data['label'])

        print(data_classes)

        n_classes = len(data_classes)
        print('n_classes:{},n_label:{},n_unlabel:{}'.format(n_classes, self.n_label, self.n_unlabel))
        dataset_l = np.zeros([n_classes, self.n_label, self.channels, self.im_height, self.im_width], dtype=np.float32)
        if self.n_unlabel > 0:
            dataset_u = np.zeros([n_classes, self.n_unlabel, self.channels, self.im_height, self.im_width],
                                 dtype=np.float32)
        else:
            dataset_u = []

        for i, cls in enumerate(tqdm(data_classes)):
            im_dir = os.path.join(self.root_dir, '{}/'.format(self.split), cls)
            im_files = sorted(glob.glob(os.path.join(im_dir, '*.jpg')))
            np.random.RandomState(self.seed).shuffle(im_files)
            for j, im_file in enumerate(im_files):
                im = cv2.resize(cv2.imread(im_file),(self.im_width,self.im_height))
                img = trans(im)
                if j < self.n_label:
                    dataset_l[i, j] = img
                else:
                    dataset_u[i, j - self.n_label] = img
        print('labeled data:', np.shape(dataset_l))
        print('unlabeled data:', np.shape(dataset_u))
        self.dataset_l = dataset_l
        self.dataset_u = dataset_u
        self.n_classes = n_classes

    def next_data(self, n_way, n_shot, n_query, num_unlabel=0, n_distractor=0, train=True):
        support = np.zeros([n_way, n_shot, self.channels, self.im_height, self.im_width], dtype=np.float32)
        query = np.zeros([n_way, n_query, self.channels, self.im_height, self.im_width], dtype=np.float32)
        if num_unlabel > 0:
            unlabel = np.zeros([n_way + n_distractor, num_unlabel, self.channels, self.im_height, self.im_width],
                               dtype=np.float32)
        else:
            unlabel = []
            n_distractor = 0
        selected_classes = np.random.permutation(self.n_classes)[:n_way + n_distractor]
        for i, cls in enumerate(selected_classes[0:n_way]):
            idx1 = np.random.permutation(self.n_label)[:n_shot + n_query]
            support[i] = self.dataset_l[cls, idx1[:n_shot]]
            query[i] = self.dataset_l[cls, idx1[n_shot:]]
            if num_unlabel > 0:
                idx2 = np.random.permutation(self.n_unlabel)[:num_unlabel]
                unlabel[i] = self.dataset_u[cls, idx2]
        for j, cls in enumerate(selected_classes[self.n_classes:]):
            idx3 = np.random.permutation(self.n_unlabel)[:num_unlabel]
            unlabel[i + j] = self.dataset_u[cls, idx3]
        support_labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_shot)).astype(np.uint8)
        query_labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)

        return support, support_labels, query, query_labels, unlabel
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
parser.add_argument('--n_shot', type=int, default=1, metavar='NSHOT',
                    help="nshot")
parser.add_argument('--n_query', type=int, default=15, metavar='NQUERY',
                    help="nquery")
parser.add_argument('--n_epochs', type=int, default=2100, metavar='NEPOCHS',
                    help="nepochs")
# test hyper-parameters
parser.add_argument('--n_test_way', type=int, default=5, metavar='NTESTWAY',
                    help="ntestway")
parser.add_argument('--n_test_shot', type=int, default=1, metavar='NTESTSHOT',
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
n_epochs = args['n_epochs']
n_way = args['n_way']
n_shot = args['n_shot']
n_query = args['n_query']
n_test_way = args['n_test_way']
n_test_shot = args['n_test_shot']
n_test_query = args['n_test_query']
patience = args['patience']

def _init_():
    if not os.path.exists('miniImageNet_checkpoint_one_shot'):
        os.makedirs('miniImageNet_checkpoint_one_shot')
    if not os.path.exists('miniImageNet_checkpoint_one_shot/' + args['exp_name']):
        os.makedirs('miniImageNet_checkpoint_one_shot/' + args['exp_name'])
    if not os.path.exists('miniImageNet_checkpoint_one_shot/' + args['exp_name'] + '/' + 'models'):
        os.makedirs('miniImageNet_checkpoint_one_shot/' + args['exp_name'] + '/' + 'models')
    f = open('miniImageNet_checkpoint_one_shot/' + args['exp_name'] + '/log.txt', 'a')
    print(args, file=f)
    print(u'\r')
    f.close()
_init_()
# step1: initial dataloader
print('init data loader')
args_data = {}
args_data['x_dim'] = args['x_dim']
args_data['ratio'] = args['ratio']
args_data['seed'] = args['seed']
loader_train = dataset_mini(n_examples, n_episodes, 'train', args_data)
loader_val = dataset_mini(n_examples, n_episodes, 'val', args_data)
loader_train.load_data()
loader_val.load_data()
# step2: initial MSLPN
print('init MSLPN neural network')
model = LabelPropagation(args)
model.cuda()
model_optim = torch.optim.Adam(model.parameters(), lr=args['lr'])
model_schedular = StepLR(model_optim, step_size=args['step_size'], gamma=args['gamma'])
result = pd.DataFrame(columns=['episodes','loss','acc'])
print('Training')
best_acc = 0.0
best_loss = np.inf
wait = 0
# step3: training
for ep in range(iters, n_epochs):
    loss_tr = []
    ce_list = []

    acc_tr = []
    loss_val = []
    acc_val = []
    for epi in tqdm(range(n_episodes), desc='train_epoc:{}'.format(ep)):
        model_schedular.step(ep * n_episodes + epi)
        model.train()
        support, s_labels, query, q_labels, unlabel = loader_train.next_data(n_way, n_shot, n_query)
        support = np.reshape(support, (support.shape[0] * support.shape[1],) + support.shape[2:])
        support = torch.from_numpy(support)
        query = np.reshape(query, (query.shape[0] * query.shape[1],) + query.shape[2:])
        query = torch.from_numpy(query)
        s_labels = torch.from_numpy(np.reshape(s_labels, (-1,)))
        q_labels = torch.from_numpy(np.reshape(q_labels, (-1,)))
        s_labels = s_labels.type(torch.LongTensor)
        q_labels = q_labels.type(torch.LongTensor)
        s_oneshot = torch.zeros(n_way * n_shot, n_way).scatter_(1, s_labels.view(-1, 1), 1)
        q_oneshot = torch.zeros(n_way * n_query, n_way).scatter_(1, q_labels.view(-1, 1), 1)

        inputs = [support.cuda(), s_oneshot.cuda(), query.cuda(), q_oneshot.cuda()]
        loss, acc = model(inputs)
        loss_tr.append(loss.item())
        acc_tr.append(acc.item())

        model.zero_grad()
        loss.backward()
        model_optim.step()
    for epi in tqdm(range(n_episodes), desc='val epoc:{}'.format(ep)):
        model.eval()

        # sample data for next batch
        support, s_labels, query, q_labels, unlabel = loader_val.next_data(n_test_way, n_test_shot, n_test_query)
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
            inputs = [support.cuda(0), s_onehot.cuda(0), query.cuda(0), q_onehot.cuda(0)]
            loss, acc = model(inputs)

        loss_val.append(loss.item())
        acc_val.append(acc.item())
    print('epoch:{}, loss_tr:{:.5f}, acc_tr:{:.5f}, loss_val:{:.5f}, acc_val:{:.5f}'.format(ep, np.mean(loss_tr),
                                                                                            np.mean(acc_tr),
                                                                                            np.mean(loss_val),
                                                                                            np.mean(acc_val)))

    # model save and stop criterion
    cond1 = (np.mean(acc_val) > best_acc)
    cond2 = (np.mean(loss_val) < best_loss)

    if cond1 or cond2:
        best_acc = np.mean(acc_val)
        best_loss = np.mean(loss_val)
        print('best val loss:{:.5f},acc:{:.5f}'.format(best_loss, best_acc))
        torch.save(model.state_dict(),
                   'miniImageNet_checkpoint_one_shot/%s/models/%s_%d_model.pkl' % (args['exp_name'], args['alg'], (ep + 1) * n_episodes))
        result = result.append({'episodes':(ep + 1) * n_episodes,'loss':best_loss,'acc':best_acc},ignore_index=True)
        wait = 0
    else:
        wait += 1
        if ep % 100 == 0:
            torch.save(model.state_dict(),
                       'miniImageNet_checkpoint_one_shot/%s/models/%s_%d_model.pkl' % (
                           args['exp_name'], args['alg'], (ep + 1) * n_episodes))
            result = result.append({'episodes':(ep + 1) * n_episodes,'loss':np.mean(loss_val),'acc':np.mean(acc_val)},ignore_index=True)
    if wait > patience and ep > n_epochs:
        break
    result.to_csv('./miniImageNet_checkpoint_one_shot/val_acc_loss1.csv')