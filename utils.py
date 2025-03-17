import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
import random
from munkres import Munkres
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
import os
import scipy.io as sio




def dir_exist(p):
    if not os.path.exists(p):
        os.makedirs(p)

def add_noise(data, noise_level):
    noise = np.random.normal(0, noise_level, data.shape)
    data = data + noise

    return data

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_graph(dataset):

    graph = 'E:/data/{}/{}_graph.txt'.format(dataset, dataset)
    print("Loading path:", graph)

    data = 'E:/data/{}/{}.txt'.format(dataset, dataset)
    dataset = np.loadtxt(data, dtype=float)
    # dataset = add_noise(dataset, noise_level=1)
    n, _ = dataset.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(graph, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj


def load_cpw(dataset):
    graph = 'E:/dk/MBN-main/data/{}/{}_graph.npz'.format(dataset,dataset)
    adj = sp.load_npz(graph)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def normalize_adj(adj, self_loop=True):
    """Symmetrically normalize adjacency matrix."""
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


class LoadDataset(Dataset):

    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def cluster_acc(y_true, y_pred):
    """

    """
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):#enumerate列出数据和数据下标
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):#得到新的pred值
        c2 = l2[indexes[i][1]]

        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro

def eva(y_true, y_pred, epoch=0):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    # print('Epoch_{}'.format(epoch), ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
    #       ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1


def get_data(dataset):
    if dataset in ['cora', 'amazon', 'pubmed', 'cs', 'phy', 'amazon-photo']:
        x_p = 'E:/data/{}/{}.npy'.format(dataset, dataset)
        y_p = 'E:/data/{}/{}_label.npy'.format(dataset, dataset)
        adj = load_cpw(dataset)
        x = np.load(x_p)
        y = np.load(y_p)
    else:
        x_p = 'E:/data/{}/{}.txt'.format(dataset, dataset)
        y_p = 'E:/data/{}/{}_label.txt'.format(dataset, dataset)
        adj = load_graph(dataset)
        x = np.loadtxt(x_p)
        y = np.loadtxt(y_p)
    return x, y, adj
def load_multigraph(dataset):

    data = sio.loadmat('data/acm/acm.mat')
    # feature
    feature = data['feature']
    features = sp.csr_matrix(feature, dtype=np.float32)

    labels = data['label']
    num_nodes = data['label'].shape[0]

    data['PAP'] = sp.coo_matrix(data['PAP'] + np.eye(num_nodes))
    data['PAP'] = data['PAP'].todense()
    data['PAP'][data['PAP'] > 0] = 1.0
    adj1 = sp.coo_matrix(data['PAP'] - np.eye(num_nodes))
    data['PLP'] = sp.coo_matrix(data['PLP'] + np.eye(num_nodes))
    data['PLP'] = data['PLP'].todense()
    data['PLP'][data['PLP'] > 0] = 1.0
    adj2 = sp.coo_matrix(data['PLP'] - np.eye(num_nodes))

    PAP = np.stack((np.array(adj1.row), np.array(adj1.col)), axis=1)
    PLP = np.stack((np.array(adj2.row), np.array(adj2.col)), axis=1)
    # print(PAP)
    # print(PLP)

    #
    PAPedges = np.array(list(PAP), dtype=np.int32).reshape(PAP.shape)
    PAP_adj = sp.coo_matrix((np.ones(PAPedges.shape[0]), (PAPedges[:, 0], PAPedges[:, 1])), shape=(num_nodes, num_nodes), dtype=np.float32)
    PAP_adj = PAP_adj + PAP_adj.T.multiply(PAP_adj.T > PAP_adj) - PAP_adj.multiply(PAP_adj.T > PAP_adj)
    PAP_normalize_adj = normalize(PAP_adj)
    # print(PAP_normalize_adj)

    PLPedges = np.array(list(PLP), dtype=np.int32).reshape(PLP.shape)
    PLP_adj = sp.coo_matrix((np.ones(PLPedges.shape[0]), (PLPedges[:, 0], PLPedges[:, 1])), shape=(num_nodes, num_nodes), dtype=np.float32)
    PLP_adj = PLP_adj + PLP_adj.T.multiply(PLP_adj.T > PLP_adj) - PLP_adj.multiply(PLP_adj.T > PLP_adj)
    PLP_normalize_adj = normalize(PLP_adj)
    # print(PLP_normalize_adj)


    adj_list = [PAP_normalize_adj, PLP_normalize_adj]

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, features.todense(), labels, idx_train, idx_val, idx_test