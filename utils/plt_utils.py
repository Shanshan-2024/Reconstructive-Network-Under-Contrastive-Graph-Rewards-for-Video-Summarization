import torch
import networkx as nx
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import os
import numpy as np
from typing import List, Optional, Tuple, Union

def visualize_embedding(k, h, color, epoch=None, loss=None, args=None, video_num=None, model_name='NoName'):
    '''
    画二维节点的表示的
    :param k:
    :param h:
    :param color:
    :param epoch:
    :param loss:
    :param args:
    :param video_num:
    :param model_name:
    :return:
    '''
    if args.shot_mode == True:
        name = 'shot'
    else:
        name = 'frame'

    # imgDir = args.save_embedding + '/{}_{}_{}_{}'.format(model_name, args.logist_classes, name, args.dataset)
    imgDir = args.save_embedding + '/{}_{}_{}_{}'.format(model_name, args.logist_classes, name, args.dataset)
    if not os.path.exists(imgDir):
        os.mkdir(imgDir)

    videoDir = imgDir + '/{}'.format(video_num)
    if not os.path.exists(videoDir):
        os.mkdir(videoDir)

    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    color = np.array(color)
    plt.scatter(h[:, 0], h[:, 1], s=10, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=20)
    # plt.show()
    plt.savefig(videoDir + '/K{}_epoch_{}.jpg'.format(k, epoch))
    plt.close()

def visualize_embedding_edge(adj, all_logits, color, args, k, model_name='NoName', video_num=None):
    def draw(i):

        pos = {}
        # colors = []
        for v in range(num_nodes):
            pos[v] = all_logits[i][v].cpu().numpy()
            # cls = pos[v].argmax()
            # colors.append(cls1color if cls else cls2color)

        # pos = nx.spring_layout(nx_G, iterations=20)
        # edgewidth = [i * 2 for i in edge_index_Weight]
        edge_color = []
        for j in edge_index_Weight:
            if j >= 0.8:
                edge_color.append('#1764aa')
            elif j >= 0.6:
                edge_color.append('#4b94c7')
            elif j >= 0.4:
                edge_color.append('#93c2de')
            else:
                edge_color.append('#cedfef')

        node_color = []
        for k in color:
            if k == 1:
                node_color.append('#00315d')
            else:
                node_color.append('#b91b1c')
        # for (u, v, d) in nx_G.edges(data=True):
        #     edgewidth.append(round(nx_G.get_edge_data(u, v).values()[0] * 20, 2))
        ax.cla()
        ax.axis('off')
        ax.set_title('Epoch: %d' % i)
        # nx.draw_networkx(nx_G.to_undirected(), pos, node_color=colors,with_labels=True, node_size=300, ax=ax)
        nx.draw_networkx(nx_G.to_undirected(), pos, node_color=node_color, with_labels=False,
                         node_size=5, ax=ax, width=0.3, edge_color=edge_color, alpha=0.5)
                         # node_size=10, ax=ax, width=0.5, edge_color='#c0c0c0')

    #存储位置
    if args.shot_mode == True:
        name = 'shot'
    else:
        name = 'frame'

    imgDir = args.save_embedding + '/{}_{}_{}_{}'.format(model_name, args.logist_classes, name, args.dataset)
    if not os.path.exists(imgDir):
        os.mkdir(imgDir)

    # videoDir = imgDir + '/{}'.format(video_num)
    # if not os.path.exists(videoDir):
    #     os.mkdir(videoDir)

    nx_G, edge_index_Weight = to_networkx(adj.numpy(), to_undirected=True)
    num_nodes = all_logits[0].size(0)
    fig = plt.figure(dpi=150)
    fig.clf()
    ax = fig.subplots()
    for i in range(args.epochs):
        draw(i)
        plt.pause(0.2)
    ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=200)
    # ani.save('change1.gif', writer='imagemagick', fps=10)
    ani.save(imgDir + '/K{}_{}.gif'.format(k, video_num), writer='imagemagick', fps=2)
    # plt.show()
    plt.close()

def to_networkx(adj, node_attrs=None, edge_attrs=None,
                to_undirected: Union[bool, str] = False,
                remove_self_loops: bool = True):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
    a directed :obj:`networkx.DiGraph` otherwise.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool or str, optional): If set to :obj:`True` or
            "upper", will return a :obj:`networkx.Graph` instead of a
            :obj:`networkx.DiGraph`. The undirected graph will correspond to
            the upper triangle of the corresponding adjacency matrix.
            Similarly, if set to "lower", the undirected graph will correspond
            to the lower triangle of the adjacency matrix. (default:
            :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)
    """
    # import networkx as nx

    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    G.add_nodes_from(range(len(adj))) #设置节点数量
    # G.add_nodes_from(range(data.num_nodes)) #设置节点数量

    node_attrs, edge_attrs = node_attrs or [], edge_attrs or [] #设置节点集和边集

    to_undirected = "upper" if to_undirected is True else to_undirected
    to_undirected_upper = True if to_undirected == "upper" else False
    to_undirected_lower = True if to_undirected == "lower" else False

    edge_index_X, edge_index_Y, edge_index_Weight = AdjtoEdge_Weight(adj)

    for i in range(len(edge_index_X)):

        u = edge_index_X[i]
        v = edge_index_Y[i]
        Weight = edge_index_Weight[i]

        if to_undirected_upper and u > v:
            continue
        elif to_undirected_lower and u < v:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v, weight=Weight)

    return G, edge_index_Weight

def AdjtoEdge_Weight(adj):
    edge_index_X = []
    edge_index_Y = []
    edge_index_Weight = []
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i, j] >= 0:
                edge_index_X.append(i)
                edge_index_Y.append(j)
                edge_index_Weight.append(adj[i, j])

    return edge_index_X, edge_index_Y, edge_index_Weight

def visualize_embedding_edge1(adj, all_logits, color, args, k, model_name='NoName', video_num=None, loss=None):
    def draw(epoch, k):

        pos = {}
        # colors = []
        for v in range(num_nodes):
            pos[v] = all_logits[epoch][v].cpu().numpy()
            # cls = pos[v].argmax()
            # colors.append(cls1color if cls else cls2color)

        # pos = nx.spring_layout(nx_G, iterations=20)
        # edgewidth = [i * 2 for i in edge_index_Weight]
        edge_color = []
        for j in edge_index_Weight:
            if j >= 0.8:
                edge_color.append('#1764aa')
            elif j >= 0.6:
                edge_color.append('#4b94c7')
            elif j >= 0.4:
                edge_color.append('#93c2de')
            else:
                edge_color.append('#cedfef')

        node_color = []
        for k in color:
            if k == 1:
                node_color.append('#00315d')
            else:
                node_color.append('#b91b1c')
        # for (u, v, d) in nx_G.edges(data=True):
        #     edgewidth.append(round(nx_G.get_edge_data(u, v).values()[0] * 20, 2))
        plt.figure(figsize=(7, 7))
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(f'Epoch: {epoch}', fontsize=16)
        # nx.draw_networkx(nx_G.to_undirected(), pos, node_color=colors,with_labels=True, node_size=300, ax=ax)
        nx.draw_networkx(nx_G.to_undirected(), pos, node_color=node_color, with_labels=False,
                         node_size=15, width=0.2, edge_color=edge_color, alpha=0.7)
                         # node_size=10, ax=ax, width=0.5, edge_color='#c0c0c0')

        plt.savefig(videoDir + '/K{}_epoch_{}.jpg'.format(k, epoch))
        plt.close()

    #存储位置
    if args.shot_mode == True:
        name = 'shot'
    else:
        name = 'frame'

    imgDir = args.save_embedding + '/{}_{}_{}'.format(model_name, name, args.dataset)
    if not os.path.exists(imgDir):
        os.mkdir(imgDir)

    videoDir = imgDir + '/{}'.format(video_num)
    if not os.path.exists(videoDir):
        os.mkdir(videoDir)

    # videoDir = imgDir + '/{}'.format(video_num)
    # if not os.path.exists(videoDir):
    #     os.mkdir(videoDir)

    nx_G, edge_index_Weight = to_networkx(adj.numpy(), to_undirected=True)
    num_nodes = all_logits[0].size(0)
    # fig = plt.figure(dpi=150)
    # fig.clf()
    # ax = fig.subplots()
    for i in range(args.epochs):
        draw(i, k)
