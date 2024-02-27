
import copy
import os.path
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import itertools
from utils.data_utils import load_data
from utils.shot_utils import *
from utils.plt_utils import *

from models.MLPtoMLP import mlptomlp
from models.GCN_model_constractive import GCN
from torch.distributions import Bernoulli
from utils.GNN_utils import Generate_key_Adj
from utils.reward_utils import compute_reward
import random
import tqdm
# from sklearn.metrics import f1_score
from config import Get_args
import matplotlib

#将图对比学习的encoder更改后的GCN
from utils.loss_utils1 import *
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GCNConv
from utils.transition_utils import *
from utils.pool_utils import *
# matplotlib.use('TkAgg')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Object():
    def __init__(self, args, model=None, k=None):
        self.args = args
        self.k = k
        self.device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

        self.split_key, self.labels_score_dic, self.labels_classes_dic, \
        self.cps_dic, self.num_frames_dic, self.nfps_dic, self.positions_dic, \
        self.user_summary_dic, self.shot_score_dic, self.shot_classes_dic = load_data(self.args)

        #k从1取到5，将k-1的数据集作为验证集
        self.test_keys = self.split_key[k - 1]
        num = [1, 2, 3, 4, 0]
        num.remove(k-1)
        self.train_keys = []
        for i in num:
            self.train_keys += self.split_key[i]

        #定义模型
        if model == None:
            # self.model = mlptomlp().to(self.device)
            self.model = GCN(args=args).to(self.device)
            encoder = Encoder(1024, args.num_hidden).to(self.device)
            encoder1 = Encoder1(1024, args.num_hidden).to(self.device)
            self.grace_model = Model(encoder, args.num_hidden, args.num_proj_hidden, args.tau).to(self.device)
            self.grace_model1 = Model1(encoder1, args.num_hidden, args.num_proj_hidden, args.tau).to(self.device)
        else:
            pass

        #定义优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.grace_optimizer1 = optim.Adam(self.grace_model1.parameters(), lr=args.grace_lr)
        #加l2正则化
        # self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0.001)
        if args.lr_step > 0:
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                              step_size=args.lr_step,
                                                              gamma=float(args.lr_gamma))
        if args.lr_step > 0:
            self.grace_lr_scheduler = optim.lr_scheduler.StepLR(self.grace_optimizer,
                                                              step_size=args.lr_step,
                                                              gamma=float(args.lr_gamma))
        #定义损失函数
        if args.logist_classes == 'logist':
            self.criterion = nn.MSELoss().to(self.device)
        else:
            self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.train_loss = []
        self.max_f1 = -1
        self.val_loss = []
        self.pre_f1 = []
        self.real_f1 = []
        self.great_model = None
        self.video_all_logists = dict([(k, []) for k in self.test_keys])
        self.grace_loss = []
        self.key_grace_loss = []
        self.reward_loss = []


    def train(self, args):
        print("----------------开始训练---------------")
        self.model.train()
        self.grace_model.train()
        self.grace_model1.train()
        tot_params = sum([np.prod(p.size()) for p in self.model.parameters()])
        print("\nTotal number of parameters: {}".format(tot_params))
        if args.logist_classes == 'logist':
            feature_dic = self.feature_dic
            cps_dic = self.cps_dic
            positions_dic = self.positions_dic
            colors = self.shot_classes_dic
            if args.shot_mode == False:
                labels_dic = self.labels_score_dic
            else:
                feature_dic = torch.load('data/{}_represent_frame.pt'.format(args.dataset))
                labels_dic = self.shot_score_dic
        else:
            feature_dic = self.feature_dic
            cps_dic = self.cps_dic
            positions_dic = self.positions_dic
            colors = self.shot_classes_dic
            if args.shot_mode == False:
                labels_dic = self.labels_classes_dic
            else:
                feature_dic = torch.load('data/{}_represent_frame.pt'.format(args.dataset))
                labels_dic = self.shot_score_dic
        if args.shot_mode == True:
            name = 'shot'
        else:
            name = 'frame'

        # 是否设置生成邻接矩阵
        if args.Generate_adj == True:
            from utils.GNN_utils import Generate_Adjacency
            Adj_dic = Generate_Adjacency(feature_dic=feature_dic, args=args)
            torch.save(Adj_dic, 'data/{}_Adjacency_{}_{}'.format(args.dataset, name, args.Gene_ways))
        else:
            Adj_dic = torch.load('data/{}_Adjacency_{}_{}'.format(args.dataset, name, args.Gene_ways))
        if args.lim_adj == True:
            from utils.GNN_utils import lim_Adj
            Adj_dic = lim_Adj(Adj_dic)

        baselines = {key: 0. for key in self.train_keys}  # baseline rewards for videos 视频的基线奖励   刚开始每个视频的baseline均为0
        reward_writers = {key: [] for key in self.train_keys}  # record reward changes for each video  #记录每个视频的奖励变化

        for epoch in range(args.max_epoch):
            sum_loss = []
            sum_grace_loss1 = []
            sum_key_grace_loss1 = []
            for video_index in self.train_keys:
                feature = torch.Tensor(feature_dic[video_index]).to(self.device)
                labels = torch.Tensor(labels_dic[video_index]).to(self.device)
                #此时的feature是si
                #将si输入GCN模型中，得到ri
                adj = torch.Tensor(Adj_dic[video_index]).to(self.device)

                # -------------------------对原图进行一次图对比学习-------------------------
                #对特征进行drop_feature，对adj进行池化得到pool_adj
                pool_adj = adj_pool(adj, 0.4)
                x_1 = drop_feature(feature, args.drop_feature_rate_1)  # feature [34, 1024]   x_1 [34, 1024]
                z1 = self.grace_model(feature, adj)
                z2 = self.grace_model(x_1, pool_adj)
                grace_loss1 = self.grace_model.loss(z1, z2, mean=True, batch_size=0)  # tensor(4.2056)
                grace_loss1.backward()
                self.grace_optimizer.step()
                #每个feature会有10个grace_loss1
                sum_grace_loss1.append(grace_loss1.item())
                z = self.grace_model(feature, adj)
                new_feature = z.detach()
                #r_output = ri [34, 256]    h [34, 2]   output  [34, 1]
                probs, r_ouput, h, output = self.model(feature, new_feature, adj)
                cost = args.beta * (probs.mean() - 0.5)**2 # minimize summary length penalty term [Eq.11]
                m = Bernoulli(probs)
                epis_rewards = []
                #进行num_episode次， 每次都重新采样，那么每次重新采样都会有新的摘要，所以每次采样都要进行摘要的更新
                for _ in range(args.num_episode):
                    actions = m.sample()
                    log_probs = m.log_prob(actions)
                    _actions = actions.detach()
                    pick_idxs = _actions.squeeze().nonzero().squeeze()  # 选中的关键镜头的下标
                    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1  # 选中的关键镜头的个数
                    # feature[34, 1024] -> key_feature[21, 1024]
                    key_feature = torch.zeros((num_picks, 1024)).to(self.device)

                    # -------------------------对内循环的原图进行次图对比学习-------------------------
                    pool_adj = adj_pool(adj, 0.4)
                    x_1 = drop_feature(feature, args.drop_feature_rate_1)  # feature [34, 1024]   x_1 [34, 1024]
                    z1 = self.grace_model(feature, adj)
                    z2 = self.grace_model(x_1, pool_adj)
                    grace_loss1 = self.grace_model.loss(z1, z2, mean=True, batch_size=0)  # tensor(4.2056)
                    grace_loss1.backward()
                    self.grace_optimizer.step()
                    # 每个feature会有10个grace_loss1
                    sum_grace_loss1.append(grace_loss1.item())
                    z = self.grace_model(feature, adj)
                    new_feature = z.detach()

                    j = 0
                    for i in range(feature.shape[0]):
                        if actions[i] == 1:
                            key_feature[j] = feature[i]
                            j = j + 1
                    key_adj = Generate_key_Adj(feature=key_feature, args=args).to(self.device)

                    # ---------------------------------------------对摘要图进行图对比学习-------------------------------------------
                    pool_key_adj = adj_pool(key_adj, 0.4)
                    key_x_1 = drop_feature(key_feature, args.drop_feature_rate_1)
                    key_z1 = self.grace_model1(key_feature, key_adj)
                    key_z2 = self.grace_model1(key_x_1, pool_key_adj)
                    key_grace_loss1 = self.grace_model1.loss(key_z1, key_z2, mean=True, batch_size=0)
                    key_grace_loss1.backward()
                    self.grace_optimizer.step()
                    sum_key_grace_loss1.append(key_grace_loss1.item())
                    key_z = self.grace_model1(key_feature, key_adj)
                    new_key_feature = key_z.detach()

                    key_probs, key_r_output, key_h, key_output = self.model(key_feature, new_key_feature, key_adj)

                    reward = compute_reward(actions, feature, key_feature, probs, key_probs, r_ouput, key_r_output)
                    expected_reward = log_probs.mean() * (reward - baselines[video_index])
                    cost -= expected_reward  # minimize negative expected reward
                    #cost是所有奖励的总和，希望奖励的总和最大，那么奖励的负最小，就把负的奖励当做loss就和机器学习一样了
                    epis_rewards.append(reward.item())
                #加l1正则化
                # lamda = 0.01
                # regularization_loss = 0
                # for param in self.model.parameters():
                #     regularization_loss += torch.sum(abs(param))
                # cost = cost + lamda * regularization_loss
                self.optimizer.zero_grad()
                cost.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                baselines[video_index] = 0.9 * baselines[video_index] + 0.1 * np.mean(
                    epis_rewards)  # update baseline reward via moving average
                reward_writers[video_index].append(np.mean(epis_rewards))
                sum_loss.append(cost.item())
                # print("\n---The K{}_epoch {} train_loss is {}".format(self.k, epoch, np.mean(sum_loss)))
                # print("key {}/ epoch {}\t reward of each video{}\t".format(video_index, epoch + 1, np.mean(epis_rewards)))
            epoch_reward = np.mean([reward_writers[key][epoch] for key in self.train_keys])
            print("epoch {}/{}\t reward {}\t".format(epoch+1, args.max_epoch, epoch_reward))
            self.val(args, epoch=epoch)
            self.train_loss.append(np.mean(sum_loss))

    def val(self, args, epoch):
        print("============> Test")
        with torch.no_grad():
            self.model.eval()
            self.grace_model.eval()
            self.grace_model1.eval()
            if args.logist_classes == 'logist':
                feature_dic = self.feature_dic
                cps_dic = self.cps_dic
                n_frames_dic = self.num_frames_dic
                nfps_dic = self.nfps_dic
                user_summary_dic = self.user_summary_dic
                positions_dic = self.positions_dic
                if args.shot_mode == False:
                    labels_dic = self.labels_score_dic
                else:
                    feature_dic = torch.load('data/{}_represent_frame.pt'.format(args.dataset))
                    labels_dic = self.shot_score_dic

                    colors = self.shot_classes_dic
            else:
                feature_dic = self.feature_dic
                cps_dic = self.cps_dic
                n_frames_dic = self.num_frames_dic
                nfps_dic = self.nfps_dic
                user_summary_dic = self.user_summary_dic
                positions_dic = self.positions_dic
                if args.shot_mode == False:
                    labels_dic = self.labels_classes_dic
                else:
                    feature_dic = torch.load('data/{}_represent_frame.pt'.format(args.dataset))
                    labels_dic = self.shot_score_dic
                    colors = self.shot_classes_dic
            sum_loss = []
            real_f1 = []
            pre_f1 = []
            pre_score = {}
            real_score = {}

            if args.shot_mode == True:
                name = 'shot'
            else:
                name = 'frame'

            # 是否设置生成邻接矩阵
            if args.Generate_adj == True:
                from utils.GNN_utils import Generate_Adjacency
                Adj_dic = Generate_Adjacency(feature_dic=feature_dic, args=args)
                torch.save(Adj_dic, 'data/{}_Adjacency_{}_{}'.format(args.dataset, name, args.Gene_ways))
            else:
                Adj_dic = torch.load('data/{}_Adjacency_{}_{}'.format(args.dataset, name, args.Gene_ways))

            for video_index in self.test_keys:
                feature = torch.Tensor(feature_dic[video_index]).to(self.device)# h = self.fc2(r_output)
                labels = torch.Tensor(labels_dic[video_index]).to(self.device)
                adj = torch.Tensor(Adj_dic[video_index]).to(self.device)
                #第一步：得到probs
                #第二步：得到machine_summary
                z = self.grace_model(feature, adj)
                probs, r_ouput, h, output = self.model(feature, z, adj)
                # probs = probs.data.cpu().squeeze().numpy()
                # pre_spot_score = output.view(-1)
                pre_spot_score = probs.squeeze(1)
                # pre_spot_score = probs.view(-1)

                if args.logist_classes == 'logist':
                    loss = self.criterion(pre_spot_score, labels)
                else:
                    loss = self.criterion(h, labels.long())

                # visualize_embedding(k=k, h=h, color=colors[video_index], epoch=epoch, loss=loss, args=args, video_num=video_index, model_name=self.model.__class__.__name__)
                self.video_all_logists[video_index].append(h.detach())

                sum_loss.append(loss.item())
                if args.shot_mode == False:
                    machine_summary = generate_summary(pre_spot_score.cpu().detach().numpy(), cps_dic[video_index],
                                                       n_frames_dic[video_index], nfps_dic[video_index],
                                                       positions_dic[video_index])
                    pre_fm, pre_prec, pre_rec = evaluate_summary(machine_summary, user_summary_dic[video_index],
                                                                 eval_metric=args.generate_mode)
                    pre_f1.append(pre_fm)
                    pre_score[video_index] = pre_spot_score.cpu().detach().numpy().tolist()

                    real_user_score = generate_summary(labels_dic[video_index], cps_dic[video_index],
                                                       n_frames_dic[video_index], nfps_dic[video_index],
                                                       positions_dic[video_index])
                    real_fm, real_prec, real_rec = evaluate_summary(real_user_score, user_summary_dic[video_index],
                                                                    eval_metric=args.generate_mode)
                    real_f1.append(real_fm)
                    real_score[video_index] = labels_dic[video_index]
                else:
                    pre_user_score1 = SpotScore_FrameScore(pre_spot_score.cpu().detach().numpy().tolist(),
                                                           cps_dic[video_index], n_frames_dic[video_index],
                                                           nfps_dic[video_index])
                    pre_fm, pre_prec, pre_rec = evaluate_summary(pre_user_score1, user_summary_dic[video_index],
                                                                 eval_metric=args.generate_mode)
                    pre_f1.append(pre_fm)
                    pre_score[video_index] = pre_spot_score.cpu().detach().numpy().tolist()

                    real_user_score = SpotScore_FrameScore(labels_dic[video_index], cps_dic[video_index],
                                                           n_frames_dic[video_index], nfps_dic[video_index])
                    real_fm, real_prec, real_rec = evaluate_summary(real_user_score, user_summary_dic[video_index],
                                                                    eval_metric=args.generate_mode)
                    real_f1.append(real_fm)
                    real_score[video_index] = labels_dic[video_index]
            if np.mean(pre_f1) >= self.max_f1:
                self.max_f1 = np.mean(pre_f1)
                self.real_spot_score = real_score
                self.pre_spot_score = pre_score
                self.great_model = copy.deepcopy(self.model)

            self.val_loss.append(np.mean(sum_loss))
            print("the real_f1 is {}, the pre_f1 is {}".format(np.mean(real_f1), np.mean(pre_f1)))
            print("最好的f1为{}".format(self.max_f1))

            self.pre_f1.append(np.mean(pre_f1))
            self.real_f1.append(np.mean(real_f1))
        return

    def draw_loss(self, k):
        # for keys in self.real_spot_score.keys():
        #     print(2)
        if args.shot_mode == True:
            name = 'shot'
        else:
            name = 'frame'

        Dir = 'save_result/the compare of predict and real' + '/{}{}_{}'.format(self.model.__class__.__name__, name, args.dataset)
        if not os.path.exists(Dir):
            os.mkdir(Dir)

        #此处的train_loss就是reward_loss
        plt.figure(1)
        plt.subplot(121)
        plt.plot(range(len(self.train_loss)), self.train_loss, '#c0504d', label='train_loss')
        plt.plot(range(len(self.val_loss)), self.val_loss, '#483D8B', label='val_loss')
        plt.title('Train_Val_loss')
        plt.legend(loc='upper right')

        plt.subplot(122)
        plt.plot(range(len(self.real_f1)), self.real_f1, color='#E6E6FA', label='real_f1')
        plt.plot(range(len(self.pre_f1)), self.pre_f1, color='#FFDEAD', label='pre_f1')
        plt.title('real_pre_f1')
        plt.legend(loc='upper right')

        plt.savefig(Dir + '/the k{} reinforcement_loss and F1'.format(k))
        plt.close()

if __name__ == "__main__":
    print("————————————————————————————————————————————————————————————————————————————")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    args = Get_args()
    activation = 'relu'
    if args.dataset == 'tvsum':
        args.generate_mode = 'avg'
    setup_seed(args.model_seed)

    pre_f1 = [0, 0, 0, 0, 0]
    real_f1 = []
    spearman_rou = []
    kendall_tao = []
    if args.logist_classes != 'logist':
        label_f1 = []
        real_label_f1 = []


    for k in range(1, 6):
        print('\n')
        print("===========================This is K{}===========================".format(k))
        #实例化模型
        object = Object(args=args, k=k)

        # 训练
        object.train(args=args)

        # 验证
        object.val(args=args, epoch=args.epochs)

        # #画损失曲线
        object.draw_loss(k=k)

        if object.max_f1 >= pre_f1[k - 1]:
            pre_f1[k - 1] = object.max_f1
        real_f1.append(object.real_f1[0])
        if args.shot_mode == True:
            name = 'shot'
        else:
            name = 'frame'

    print("——————————————————————————————————————————")
    for i in range(len(real_f1)):
        print("This is K{}:".format(i + 1))
        print("The real_f1 is {}, the pre_f1 is {}".format(real_f1[i], pre_f1[i]))
    print("——————————————————————————————————————————")
    print("The avg real_f1 is{}, the avg pre_f1 is {}".format(np.mean(real_f1), np.mean(pre_f1)))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))