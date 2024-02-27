import torch


def AdjToEdgeindex(adj):
    # print(adj.shape) [34, 34]
    N = adj.shape[0]
    edge_index = torch.IntTensor(2, N*N)

    # print(len(edge_index[0]))

    count = 0
    for i in range(N):
        for j in range(N):
            edge_index[0][count] = i
            # print(edge_index[0][count])
            edge_index[1][count] = j
            # print(edge_index[1][count])
            count = count + 1
    # print(edge_index[0][0].item())

    return edge_index

