import torch
import torch.nn as nn
import torch.nn.functional as F

def Generate_Adjacency(feature_dic, args):
    fc = nn.Linear(1024, 1024)

    Adj_dic = {}

    for video_index in feature_dic.keys():
        #点积
        adjacency = torch.mm(data, data.t())
        # adjacency = torch.mm(fc(feature), fc(feature).t())
        adjacency += torch.eye(adjacency.shape[0])
        # degree = adjacency.sum(1, dim=1)
        degree = torch.diag(adjacency.sum(dim=1))
        # d_hat = degree.pow(-0.5)
        # adjacency = torch.mm(torch.mm(d_hat, adjacency), d_hat)
        # Adj_dic[video_index] = adjacency

        #连接
        # N = feature.shape[0]
        # adjacency = torch.zeros((N, N))
        # for i in range(N):
        #     for j in range(N):
        #         temp = torch.cat((feature[i], feature[j]), dim=0)
                # adjacency[i, j] = self.cat_adj(temp)
                adjacency[i, j] = fc1(temp)
        # Adj_dic[video_index] = adjacency

        # cosine余弦相似
        N = feature.shape[0]
        adjacency = torch.zeros((N, N))

        if args.Gene_ways == 'cosine_similarity':

            for i in range(N):
                for j in range(N):
                    # smi = nn.functional.cosine_similarity(feature[i], feature[j], dim=0)
                    smi = nn.functional.cosine_similarity(feature[i], feature[j], dim=0)
                    adjacency[i, j] = smi

        Adj_dic[video_index] = adjacency

    return Adj_dic

def lim_Adj(adj_dic):
    New_adj_dic = {}
    Epsilon = 0.1
    for key in adj_dic.keys():
        adj = adj_dic[key]
        for i in range(0, 50):
            # new_adj = torch.matmul(adj, adj)
            if torch.norm((new_adj - adj), p=2, dim=1)[0] < Epsilon:
                break
            else:
                adj = new_adj

        New_adj_dic[key] = new_adj

    return New_adj_dic

def Generate_key_Adj(feature, args):
    # 计算节点个数
    N = feature.shape[0]
    adjacency = torch.zeros((N, N))

    if args.Gene_ways == 'cosine_similarity':

        for i in range(N):
            for j in range(N):
                smi = nn.functional.cosine_similarity(feature[i], feature[j], dim=0)
                adjacency[i, j] = smi

    return adjacency

#动态图相关函数
class GraphLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.a = a
        # self.b = b
        # self.c = c
    def forward(self, adj, X):
        # temp = torch.matmul(X.transpose(2,1), L) #(512, 64) (64, 512)->(512, 512)
        # temp = torch.matmul(temp, X) # (512, 512) (64, 512)
        # loss1 = self.a * temp.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)/(adj.size(1)**2)
        # for i in range(adj.size(0)):
        # print("样本大小为{},其中非零的个数为{}".format(adj.size(0) * adj.size(1), torch.count_nonzero(adj)))
        A_idx = adj > 0
        adj = adj.masked_fill(A_idx, 1)
        adj = torch.zeros_like(adj) + torch.triu(adj, diagonal=1) + torch.tril(adj, diagonal=-1)
        # for i in range(adj.size(0)):此处遍历batch，但视频摘要没有设置batch
        # print("样本大小为{},其中去掉自环后非零的个数为{}".format(adj.size(0) * adj.size(1), torch.count_nonzero(adj)))
        Y = X
        # loss1 = self.a * torch.sum((torch.cdist(X, Y, p=2) * adj).view(X.size(0), -1), dim=-1) / (2 * adj.size(1) ** 2)
        temp_loss1 = torch.sum((torch.cdist(X, Y, p=2) * adj).view(X.size(0), -1), dim=-1) / (2 * adj.size(1) ** 2)
        ones_vec = torch.ones(adj.shape[:-1]).cuda()

        # loss2 = -self.b * torch.matmul(ones_vec.unsqueeze(1), torch.log(torch.matmul(adj, ones_vec.unsqueeze(-1)) + (1e-12))).squeeze(-1).squeeze(-1) / adj.shape[-1]

        # temp_loss2 = -torch.matmul(ones_vec.unsqueeze(1), torch.log(torch.matmul(adj, ones_vec.unsqueeze(-1)) + (1e-12))).squeeze(-1).squeeze(-1) / (adj.shape[-1])
        temp_loss2 = -torch.matmul(ones_vec, torch.log(torch.matmul(adj, ones_vec) + (1e-12))) / (adj.shape[-1])
        # loss3 = self.c * torch.sum(torch.pow(adj, 2), (1, 2)) / (adj.size(1) ** 2)
        temp_loss3 = torch.sum(torch.pow(adj, 2), (0, 1)) / (adj.size(0) ** 2)
        print("loss1: {}、loss2: {}、loss3: {}".format(temp_loss1.mean(), temp_loss2.mean(), temp_loss3.mean()))
        return (temp_loss1 + temp_loss2 + temp_loss3).mean(0)

