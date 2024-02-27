
import torch
import numpy as np
from sklearn.metrics import mutual_info_score

def compute_reward(actions, feature, labels, key_labels, key_feature,  r_output, key_r_output):
    n_originalVideo = feature.shape[0]
    n_summaryVideo = key_feature.shape[0]   #可能每次个数不一样，12或13

    if n_summaryVideo == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.)
        return reward
    #此处的as用每个镜头的分数
    labels = torch.sigmoid(labels)
    # print(labels.shape)
    #计算Cv
    sum_originalVideo = 0.0
    for i in range(n_originalVideo):
        sum_originalVideo = sum_originalVideo + labels[i] * feature[i]
    Cv = sum_originalVideo / n_originalVideo

    #计算Cs
    sum_summaryVideo = 0.0
    for i in range(n_summaryVideo):
        sum_summaryVideo = sum_summaryVideo + key_labels[i] * key_feature[i]
    Cs = sum_summaryVideo / n_summaryVideo
    # print(Cv.shape)  [1024]
    # print(Cs.shape)  [1024]
    norm2_lc = torch.norm(Cv-Cs, p=2, dim=0)
    lc = torch.exp(-torch.pow(norm2_lc, 2))

    # print(r_output.shape)  [34, 128]
    # print(key_r_output.shape)  [21, 128]
    def calc_MI(x, y, bins):
        #得到共现矩阵
        c_xy = np.histogram2d(x, y, bins)[0]
        #计算互信息
        mi = mutual_info_score(None, None, contingency=c_xy)
        return mi
    #x:1024
    x = Cv.cpu().detach().numpy().tolist()
    y = Cs.cpu().detach().numpy().tolist()
    lc_m = calc_MI(x, y, 4)

    #计算ld
    sum_k = 0.0
    j = 0
    sum_ld_m = 0.0
    for i in range(n_originalVideo):
        if actions[i] == 1:
            #x1:256
            x1 = r_output[i].cpu().detach().numpy().tolist()
            y1 = key_r_output[j].cpu().detach().numpy().tolist()
            sum_ld_m = sum_ld_m + calc_MI(x1, y1, 4)
            sum_k = sum_k + torch.exp(-torch.pow(torch.norm(r_output[i] - key_r_output[j]), 2))
            j = j + 1
    ld = sum_k / n_summaryVideo
    ld_m = sum_ld_m / n_summaryVideo


    #计算l^i
    #给出feature原视频的特征，key_feature摘要的特征；adj原视频的邻接矩阵，key_adj摘要的邻接矩阵
    #r_output卷积后的原视频的特征，key_r_output卷积后的摘要视频的特征
    # from sklearn.metrics import mutual_info_score
    # X = [1, 1, 2]
    # Y = [2, 3, 1]
    # # 计算X和Y之间的互信息
    # print(mutual_info_score(X, Y))
    # from sklearn.metrics import mutual_info_score
    # def calc_MI(x, y, bins):
    #     c_xy = np.histogram2d(x, y, bins)[0]
    #     mi = mutual_info_score(None, None, contingency=c_xy)
    #     return mi
    # x = [1,0,1,1,2,2,2,2,3,6,5,6,8,7,8,9]
    # y = [3,0,4,4,4,5,4,6,7,7,8,6,8,7,9,9]
    # mi = calc_MI(x,y,4)




    # reward = (lc + ld) * 0.5
    reward = (lc_m + ld_m) * 0.5
    # reward = (lc + ld + lm)
    # reward = (lm + ld)
    # reward = (lc_m + ld_m) * 2
    #reward的值对f1值有一定的影响

    return reward