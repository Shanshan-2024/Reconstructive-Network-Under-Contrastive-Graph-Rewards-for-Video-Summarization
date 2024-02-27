'''模型参数设置'''

import argparse

def Get_args():
    parser = argparse.ArgumentParser()

    #数据集设置
    parser.add_argument('--Train_dataset', default='canonical', help='有canonical， augment， transfer三种模式，选择以哪种模式读取数据集')
    parser.add_argument('--Dataset_root', default='data/', help='存放数据集的路径')
    parser.add_argument('--dataset', default='tvsum', help='选用的数据集{tvsum, summe}')
    parser.add_argument('--generate_mode', default='arg', help='在生成f1的时候选择评价指标，tvsum->arg, summe->max')
    parser.add_argument('--split_seed', default='1226', help='随机划分的随机数种子')
    parser.add_argument('--shot_mode', type=bool, default=True, help='是否使用镜头模式')

    #修改部分
    parser.add_argument('--summe_dataset', default='summe', help='选用的数据集{summe}')
    parser.add_argument('--tvsum_dataset', default='tvsum', help='选用的数据集{tvsum}')
    parser.add_argument('--ovp_dataset', default='ovp', help='选用的数据集{ovp}')
    parser.add_argument('--youtube_dataset', default='youtube', help='选用的数据集{youtube}')

    #训练设置
    parser.add_argument('--lr', default=1e-5, help='学习率')
    parser.add_argument('--lr_step', type=int, default=30, help='调整学习率的轮数')
    parser.add_argument('--lr_gamma', default=0.1, help='学习率衰减频率')
    parser.add_argument('--cuda', default='cuda:0', help='使用哪个GPU学习')
    parser.add_argument('--epochs', type=int, default=60, help='训练轮数')
    parser.add_argument('--weight_decay', default=0.00001, help='l2正则化权重')
    parser.add_argument('--model_seed', default=1226, help='模型初始化的随机数种子')
    parser.add_argument('--save_dir', default='save_result/save_model', help='模型存储位置')
    parser.add_argument('--logist_classes', default='logist', help='回归问题还是分类问题,classes,logist')
    parser.add_argument('--act_fun', default=False, help='最后一层是否用激活函数')
    parser.add_argument('--dropout', default=0.0, help='')

    #强化学习参数
    parser.add_argument('--num-episode', type=int, default=10, help="number of episodes (default: 10)")
    parser.add_argument('--beta', type=float, default=0.01,
                        help="weight for summary length penalty term (default: 0.01)")  # 摘要长度权重的
    parser.add_argument('--max-epoch', type=int, default=1, help="maximum epoch for training (default: 60)")  # 训练的最大轮数

    # GCN设置
    parser.add_argument('--GNN_Use', type=bool, default=True, help='使用图神经网络')
    parser.add_argument('--Generate_adj', type=bool, default=True, help='计算邻接矩阵')
    parser.add_argument('--Gene_ways', default='cosine_similarity', help='生成邻接矩阵的计算方法：'
                                                                         '余弦距离：cosine_similarity， ')
    parser.add_argument('--lim_adj', default=False, help='计算邻接矩阵的极限稳态值')

    #存储设置
    parser.add_argument('--save_embedding', default='save_result', help='二维可视化存储位置')

    #图对比学习设置
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--grace_lr', default=0.0005, help='对比学习学习率')
    parser.add_argument('--num_hidden', default=256, help='')
    parser.add_argument('--num_proj_hidden', default=256, help='')
    parser.add_argument('--drop_edge_rate_1', default=0.2, help='随机去除的边的占所有边的比率')
    parser.add_argument('--drop_feature_rate_1', default=0.3, help='对特征做掩码的比率')
    parser.add_argument('--tau', default=0.8, help='系数')

    return parser.parse_args()

# print(Get_args())