from matplotlib import patheffects
from torch.optim import Adam, AdamW

from sklearn.cluster import KMeans
import torch.nn.functional as F
from SCCGC import SCCGC
from utils import setup_seed, target_distribution, eva, LoadDataset, get_data, cluster_acc, \
    sparse_mx_to_torch_sparse_tensor, dir_exist
from opt import args
import datetime
from logger import Logger, metrics_info, record_info

import gc
import torch
import numpy as np
import scipy.sparse as sp


nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def Ncontrast(x_dis, adj_label, tau=1):
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis * adj_label, 1)
    loss = -torch.log((x_dis_sum_pos + 1e-8) * (x_dis_sum + 1e-8) ** (-1)).mean()
    return loss.requires_grad_(True) - 1 + 1


def get_feature_dis(x, kt):
    x_dis = x @ x.T
    mask = torch.eye(x_dis.shape[0]).to(x.device)
    x_sum = torch.sum(x ** 2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis * (x_sum ** (-1))
    x_dis = (1 - mask) * x_dis
    return x_dis


def get_A_r_flex(adj, r, cumulative=True):
    adj_d = adj.to_dense()
    adj_c = adj_d  # A1, A2, A3 .....
    adj_label = adj_d
    for i in range(r):
        adj_c = adj_c @ adj_d

        adj_X = adj_c ** (2 / (i + 2))
        adj_label = adj_label + adj_X if cumulative else adj_X

    return adj_label


def train(model, x, y, adj):


    adj=adj.to(device)
    x=x.to(device)




    acc_reuslt = []
    nmi_result = []
    ari_result = []
    f1_result = []
    original_acc = -1
    metrics = [' acc', ' nmi', ' ari', ' f1']
    logger = Logger(args.name + '==' + nowtime)
    logger.info(model)
    logger.info(args)
    logger.info(metrics_info(metrics))

    n_clusters = args.n_clusters

    optimizer = Adam(model.parameters(), lr=args.lr)
    with torch.no_grad():
        z, _, _, _, _ = model.ae(x)

    adj1 = get_A_r_flex(adj, args.order, cumulative=args.influence)
    adj1 = adj1.to(device)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    cluster_id = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)


    for epoch in range(300):

        x_bar, z_hat, adj_hat, z_ae, q, q1, z_l, enc_h3 = model(x, adj)


        if epoch % 1 == 0:
            tmp_q = q.data
            p = target_distribution(tmp_q)

        ae_loss = F.mse_loss(x_bar, x)
        loss_w = F.mse_loss(z_hat, torch.spmm(adj, x))
        loss_a = F.mse_loss(adj_hat, adj.to_dense())
        loss_gae = loss_w + 0.1 * loss_a
        re_loss = 1 * ae_loss + 1 * loss_gae
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        q1q_loss = F.kl_div(q1.log(), q, reduction='batchmean')
        x_dis = get_feature_dis(enc_h3, x)

        loss_con = Ncontrast(x_dis, adj1, 1)

        loss = re_loss + args.gama * loss_con + args.alpha * (kl_loss + q1q_loss)



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        res1 = p.data.cpu().numpy().argmax(1)  # P
        res2 = q.data.cpu().numpy().argmax(1)  # Q
        res3 = q1.data.cpu().numpy().argmax(1)  # Q1

        plist = eva(y, res1, str(epoch) + 'P')
        qlist = eva(y, res2, str(epoch) + 'Q')
        acc, nmi, ari, f1 = eva(y, res3, str(epoch) + 'Q1')
        #
        logger.info("epoch%d%s:\t%s" % (epoch, ' Q1', record_info([acc, nmi, ari, f1])))


        acc_reuslt.append(acc)
        nmi_result.append(nmi)
        ari_result.append(ari)
        f1_result.append(f1)


    best_acc = max(acc_reuslt)
    t_nmi = nmi_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]
    t_ari = ari_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]
    t_f1 = f1_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]
    t_epoch = np.where(acc_reuslt == np.max(acc_reuslt))[0][0]
    logger.info("%sepoch%d:\t%s" % ('Best Acc is at ', t_epoch, record_info([int(best_acc*10000)/10000, int(t_nmi*10000)/10000, int(t_ari*10000)/10000, int(t_f1*10000)/10000])))



if __name__ == "__main__":


    setup_seed(2018)
    device = torch.device("cuda" if args.cuda else "cpu")

    x, y, adj = get_data(args.name)


    dataset = LoadDataset(x)
    x = torch.Tensor(dataset.x).to(device)



    print(adj.shape)
    model = SCCGC(
        ae_n_enc_1=500,
        ae_n_enc_2=500,
        ae_n_enc_3=2000,
        ae_n_dec_1=2000,
        ae_n_dec_2=500,
        ae_n_dec_3=500,
        gae_n_enc_1=500,
        gae_n_enc_2=500,
        gae_n_enc_3=2000,
        gae_n_dec_1=2000,
        gae_n_dec_2=500,
        gae_n_dec_3=500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        v=1.0,

        device=device).to(device)

    train(model, x, y, adj)
