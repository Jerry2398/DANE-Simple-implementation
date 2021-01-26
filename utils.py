import scipy.sparse as sp
import torch
import json
import numpy as np
from sklearn import metrics
import math


# each column represents a class
def encode_onehot(labels):
    #classes = set(['AI&DM','SE','CA','CN','CG','CT','HC','other'])
    classes = set(['AI&DM','CA','CN','HC'])
    #classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),dtype=np.int32)
    return labels_onehot


def load_data(path="data/transfer/", dataset="chn", preserve_order=1, multilabel=False):
    """Load citation network dataset (cora only for now)"""

    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    multilabels = None
    if multilabel:
        f = open("{}{}.multilabel".format(path, dataset))
        multilabels =np.genfromtxt("{}{}.multilabel".format(path, dataset), dtype=np.dtype(int))

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = sp.coo_matrix(adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj))
    for item in adj.__dict__.items():
        print(item)
    print(adj.col)

    edge_ret = []

    edge_weight = []

    node_weight = [0.0 for i in range(0, len(idx))]

    if preserve_order == 1:
        adj_pres = adj
    else:
        adj_pres = sp.coo_matrix(adj**2)

    # sampling weight
    for i in range(0, len(adj.data)):
        edge_ret.append((adj_pres.row[i], adj_pres.col[i]))
        edge_weight.append(float(adj_pres.data[i]))
        node_weight[adj.row[i]] += adj.data[i]


    features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])
    D = sp.coo_matrix([[1.0/math.sqrt(node_weight[j]) if j== i else 0 for j in range(len(idx))] for i in range(len(idx))])
    adj = D*adj*D

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    for i in range(0, len(node_weight)):
        node_weight[i] = math.pow(node_weight[i],0.75)

    if multilabel:
        multilabels = torch.LongTensor(multilabels)

    return adj, features, labels, idx_train, idx_val, idx_test, edge_ret, torch.tensor(edge_weight), torch.tensor(node_weight), multilabels


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
