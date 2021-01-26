import argparse
import random
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
from sklearn import manifold
import matplotlib.pyplot as plt

from utils import *
from models import *

parser = argparse.ArgumentParser(description="simple implementation of DANE")
parser.add_argument("--seed", action="store", default=13, type=int, help="random seed")
parser.add_argument("--multilabel", action="store", default=False, type=bool, help="if use multilabel dataset")
parser.add_argument("--path", action="store", default="data/transfer/", type=str, help="data path")
parser.add_argument("--datasetA", action="store", default="usa", type=str, help="dataset name")
parser.add_argument("--datasetB", action="store", default="chn", type=str, help="dataset name")
parser.add_argument("--cuda", action="store", default=False, help="use cuda or not, default: False")
parser.add_argument("--model", action="store", default="DANE", help="choose model")
parser.add_argument("--G_lr", action="store", default=0.001, type=float, help="generator learning rate")
parser.add_argument("--G_weight_decay", action="store", default=1e-5, type=float, help="generator weight decay")
parser.add_argument("--D_lr", action="store", default=0.001, type=float, help="discriminator learning rate")
parser.add_argument("--D_weight_decay", action="store", default=1e-5, type=float, help="discriminator weight decay")
parser.add_argument("--epochs", action="store", default=500, type=int, help="training epochs")
parser.add_argument("--dis_epochs", action="store", default=5, type=int, help="discriminator update number per epoch")
parser.add_argument("--batch_size", action="store", default=64, type=int, help="batch size")
parser.add_argument("--hidden_dim", action="store", default=128, type=int, help="hidden layer dimension")
parser.add_argument("--G_dropout", action="store", default=0.5, type=float, help="generator dropout rate")
parser.add_argument("--src_rate", action="store", default=0.2, type=float, help="source graph rate of labeled nodes")
parser.add_argument("--tgt_rate", action="store", default=0.05, type=float, help="target graph rate of labeled nodes")
parser.add_argument("--k", action="store", default=5, type=int, help="number of negative samples")
parser.add_argument("--k1", action="store", default=0.1, type=float, help="lambda")
parser.add_argument("--k2", action="store", default=1, type=float, help="theta")
parser.add_argument("--k3", action="store", default=1, type=float, help="gamma")

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.random.manual_seed(args.seed)

if args.multilabel:
    assert args.path == "data/transfer3/", "path error"
else:
    assert args.path == "data/transfer/", "path error"

adjA, featuresA, labelsA, idx_trainA, idx_valA, idx_testA, edgesA, edges_weightA, nodes_weightA, multilabels_A = load_data(
    path=args.path, dataset="usa", preserve_order=1, multilabel=args.multilabel)
adjB, featuresB, labelsB, idx_trainB, idx_valB, idx_testB, edgesB, edges_weightB, nodes_weightB, multilabels_B = load_data(
    path=args.path, dataset="chn", preserve_order=1, multilabel=args.multilabel)

if args.multilabel:
    args.n_label = multilabels_A.shape[1]
    args.nclass = args.n_label
else:
    args.n_label = max(labelsA) + 1
    args.nclass = args.n_label

nodeA = torch.tensor([1.0 for i in range(0, len(labelsA))])
nodeB = torch.tensor([1.0 for i in range(0, len(labelsB))])

GCN_model = GCN(featuresA.shape[1], args.hidden_dim, args.multilabel, args.nclass, args.G_dropout)
dis_model = NetD(args.hidden_dim)

g_opt = optim.Adam(GCN_model.parameters(), lr=args.G_lr, weight_decay=args.G_weight_decay)
d_opt = optim.Adam(dis_model.parameters(), lr=args.D_lr, weight_decay=args.D_weight_decay)

if args.cuda:
    GCN_model.cuda()
    dis_model.cuda()
    featuresA = featuresA.cuda()
    featuresB = featuresB.cuda()
    adjA = adjA.cuda()
    adjB = adjB.cuda()
    labelsA = labelsA.cuda()
    labelsB = labelsB.cuda()
    multilabels_A = multilabels_A.cuda()
    multilabels_B = multilabels_B.cuda()
    edgesA = edgesA.cuda()
    edgesB = edgesB.cuda()
    edges_weightA = edges_weightA.cuda()
    edges_weightB = edges_weightB.cuda()
    nodes_weightA = nodes_weightA.cuda()
    nodes_weightB = nodes_weightB.cuda()
    nodeA = nodeA.cuda()
    nodeB = nodeB.cuda()


def visualize(embeddings_A, labels_A, embeddings_B, labels_B, title_A=None, title_B=None):
    embeds_A = np.array(embeddings_A.detach())
    embeds_B = np.array(embeddings_B.detach())
    np_labels_A = np.array(labels_A.detach())
    np_labels_B = np.array(labels_B.detach())
    TSNE = manifold.TSNE(n_components=2, init='pca', random_state=args.seed)
    coordinate_A = TSNE.fit_transform(embeds_A)
    coordinate_B = TSNE.fit_transform(embeds_B)
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.scatter(coordinate_A[:, 0], coordinate_A[:, 1], s=1, c=np_labels_A, cmap='coolwarm')
    plt.title(title_A)
    plt.subplot(3, 1, 2)
    plt.scatter(coordinate_B[:, 0], coordinate_B[:, 1], s=1, c=np_labels_B, cmap='coolwarm')
    plt.title(title_B)
    plt.subplot(3, 1, 3)
    plt.scatter(coordinate_A[:, 0], coordinate_A[:, 1], s=1, c=np_labels_A, cmap='coolwarm')
    plt.scatter(coordinate_B[:, 0], coordinate_B[:, 1], s=1, c=np_labels_B, cmap='coolwarm')
    plt.title("{} + {}".format(title_A, title_B))
    plt.show()


def L_GCN(embedding, nodes_weight, idx_u, idx_v, k):
    embedding_u = embedding[idx_u]
    embedding_v = embedding[idx_v]

    embedding_neg = [embedding[torch.multinomial(nodes_weight, args.batch_size, replacement=False)] for i in
                     range(0, k)]

    pos = torch.sum(torch.mul(embedding_u, embedding_v), dim=1)
    neg = [torch.sum(torch.mul(embedding_u, embedding_neg[i]) * (-1), dim=1) for i in range(0, k)]
    loss = - torch.sum(F.logsigmoid(pos))
    for i in range(0, k):
        loss = loss - torch.sum(F.logsigmoid(neg[i]))
    return loss


def CE_loss(output, labels, multilabel):
    if multilabel:
        tmp_output = 1 - output
        return (-1) * torch.sum(
            torch.mul(output, torch.log(output + 1e-8)) + torch.mul(tmp_output, torch.log(tmp_output + 1e-8)))
    else:
        return F.cross_entropy(output, labels)


def L_cluster(labelsA, embA, labelsB, embB, multilabel):
    if multilabel:
        loss = 0.0
        for i in range(args.n_label):
            idxA = np.where(labelsA[:, i] == 0)
            idxB = np.where(labelsB[:, i] == 0)
            if (idxA[0].size > 0 and idxB[0].size > 0):
                loss += torch.sum((torch.mean(embA[idxA], dim=0) - torch.mean(embB[idxB], dim=0)) ** 2)
            idxA = np.where(labelsA[:, i] == 1)
            idxB = np.where(labelsB[:, i] == 1)
            if (idxA[0].size > 0 and idxB[0].size > 0):
                loss += torch.sum((torch.mean(embA[idxA], dim=0) - torch.mean(embB[idxB], dim=0)) ** 2)
        answer = loss / (2 * args.n_label)
    else:
        loss = 0.0
        for i in range(args.n_label):
            idxA = np.where(labelsA == i)
            idxB = np.where(labelsB == i)
            if (idxA[0].size > 0 and idxB[0].size > 0):
                loss += torch.sum((torch.mean(embA[idxA], dim=0) - torch.mean(embB[idxB], dim=0)) ** 2)
        answer = loss / args.n_label
    return answer


def train_d():
    GCN_model.eval()
    dis_model.train()

    outputA, embeddingA = GCN_model(featuresA, adjA)
    outputB, embeddingB = GCN_model(featuresB, adjB)

    train_idxA = torch.multinomial(nodeA, 8 * args.batch_size, replacement=True)
    train_idxB = torch.multinomial(nodeB, 8 * args.batch_size, replacement=True)
    preA = dis_model(embeddingA[train_idxA])
    preB = dis_model(embeddingB[train_idxB])
    d_opt.zero_grad()

    loss = (preA ** 2).mean() + ((preB - 1) ** 2).mean()
    loss.backward()
    d_opt.step()
    return loss.item()


def train_g():
    GCN_model.train()
    dis_model.eval()

    outputA, embeddingA = GCN_model(featuresA, adjA)
    outputB, embeddingB = GCN_model(featuresB, adjB)

    sample_edgeA = torch.multinomial(edges_weightA, args.batch_size, replacement=False)
    idx_uA = [edgesA[i][0] for i in sample_edgeA]
    idx_vA = [edgesA[i][1] for i in sample_edgeA]

    sample_edgeB = torch.multinomial(edges_weightB, args.batch_size, replacement=False)
    idx_uB = [edgesB[i][0] for i in sample_edgeB]
    idx_vB = [edgesB[i][1] for i in sample_edgeB]

    train_idxA = torch.multinomial(nodeA, 8 * args.batch_size, replacement=True)
    train_idxB = torch.multinomial(nodeB, 8 * args.batch_size, replacement=True)
    preA = dis_model(embeddingA[train_idxA])
    preB = dis_model(embeddingB[train_idxB])

    L_adv = (preB ** 2).mean() + ((preA - 1) ** 2).mean()

    L_gcn1 = L_GCN(embeddingA, nodes_weightA, idx_uA, idx_vA, args.k)
    L_gcn2 = L_GCN(embeddingB, nodes_weightB, idx_uB, idx_vB, args.k)

    L_gcn = L_gcn1 + L_gcn2

    label_idxA = torch.multinomial(nodeA, int(args.src_rate * len(nodeA)), replacement=False)
    label_idxB = torch.multinomial(nodeB, int(args.tgt_rate * len(nodeB)), replacement=False)

    if args.multilabel:
        L_ce1 = CE_loss(outputA[label_idxA], multilabels_A[label_idxA], args.multilabel)
        L_ce2 = CE_loss(outputB[label_idxB], multilabels_B[label_idxB], args.multilabel)
        L_ce = L_ce1 + L_ce2
        L_c = L_cluster(multilabels_A[label_idxA], embeddingA[label_idxA], multilabels_B[label_idxB],
                        embeddingB[label_idxB],
                        args.multilabel)
    else:
        L_ce1 = CE_loss(outputA[label_idxA], labelsA[label_idxA], args.multilabel)
        L_ce2 = CE_loss(outputB[label_idxB], labelsB[label_idxB], args.multilabel)
        L_ce = L_ce1 + L_ce2
        L_c = L_cluster(labelsA[label_idxA], embeddingA[label_idxA], labelsB[label_idxB],
                        embeddingB[label_idxB],
                        args.multilabel)

    loss = L_gcn + args.k1 * L_adv + args.k2 * L_ce + args.k3 * L_c
    g_opt.zero_grad()
    loss.backward()
    g_opt.step()
    return loss.item()


def evaluate_model(embedding_src, labels_src, embedding_tgt, labels_tgt, multilabel=False):
    if multilabel:
        classifier = MultiOutputClassifier(SGDClassifier())
        source = np.array(embedding_src.detach())
        target = np.array(embedding_tgt.detach())
        src_label = np.array(labels_src.detach())
        tgt_label = np.array(labels_tgt.detach())
        classifier.fit(source, src_label)
        label_pred = classifier.predict(target)
        macro_f1 = 0
        for i in range(len(labels_src[0])):
            macro_f1 = macro_f1 + f1_score(tgt_label[:, i], label_pred[:, i], average='macro')
        macro_f1 = macro_f1 / len(labels_src[0])
    else:
        classifier = SGDClassifier()
        source = np.array(embedding_src.detach())
        target = np.array(embedding_tgt.detach())
        src_label = np.array(labels_src.detach())
        tgt_label = np.array(labels_tgt.detach())
        classifier.fit(source, src_label)
        label_pred = classifier.predict(target)
        macro_f1 = f1_score(tgt_label, label_pred, average='macro')
    return macro_f1


print("training process begins")
for epoch in range(args.epochs):
    for dis_epoch in range(args.dis_epochs):
        discriminator_loss = train_d()
        print("discriminator_loss is ", discriminator_loss)
    generator_loss = train_g()
    print("generator_loss is ", generator_loss)
    if epoch % 50 == 0:
        GCN_model.eval()
        outputA, embeddingA = GCN_model(featuresA, adjA)
        outputB, embeddingB = GCN_model(featuresB, adjB)
        if args.multilabel:
            print("@@@")
            macro_f1 = evaluate_model(embeddingA, multilabels_A, embeddingA, multilabels_A, multilabel=args.multilabel)
            print("{}->{} macro f1 is ".format(args.datasetA, args.datasetA), macro_f1)
            macro_f1 = evaluate_model(embeddingA, multilabels_A, embeddingB, multilabels_B, multilabel=args.multilabel)
            print("{}->{} macro f1 is ".format(args.datasetA, args.datasetB), macro_f1)
            print("@@@")
        else:
            print("@@@")
            macro_f1 = evaluate_model(embeddingA, labelsA, embeddingA, labelsA, multilabel=args.multilabel)
            print("{}->{} macro f1 is ".format(args.datasetA, args.datasetA), macro_f1)
            macro_f1 = evaluate_model(embeddingA, labelsA, embeddingB, labelsB, multilabel=args.multilabel)
            print("{}->{} macro f1 is ".format(args.datasetA, args.datasetB), macro_f1)
            print("@@@")
    print("#####################################")

print("test process begins")
GCN_model.eval()
outputA, embeddingA = GCN_model(featuresA, adjA)
outputB, embeddingB = GCN_model(featuresB, adjB)
if args.multilabel:
    macro_f1 = evaluate_model(embeddingA, multilabels_A, embeddingA, multilabels_A, multilabel=args.multilabel)
    print("{}->{} macro f1 is ".format(args.datasetA, args.datasetA), macro_f1)
    macro_f1 = evaluate_model(embeddingA, multilabels_A, embeddingB, multilabels_B, multilabel=args.multilabel)
    print("{}->{} macro f1 is ".format(args.datasetA, args.datasetB), macro_f1)
else:
    macro_f1 = evaluate_model(embeddingA, labelsA, embeddingA, labelsA, multilabel=args.multilabel)
    print("{}->{} macro f1 is ".format(args.datasetA, args.datasetA), macro_f1)
    macro_f1 = evaluate_model(embeddingA, labelsA, embeddingB, labelsB, multilabel=args.multilabel)
    print("{}->{} macro f1 is ".format(args.datasetA, args.datasetB), macro_f1)
    visualize(embeddingA, labelsA, embeddingB, labelsB, title_A=args.datasetA, title_B=args.datasetB)

'''
For DANE:
citation/A->B: python train.py --datasetA "usa" --datasetB "chn"
citation/B->A: python train.py --datasetA "chn" --datasetB "usa"
coauthor/A->B: python train.py --path "data/transfer3/" --multilabel True --datasetA "usa" --datasetB "chn"
coauthor/B->A: python train.py --path "data/transfer3/" --multilabel True --datasetA "chn" --datasetB "usa"
'''
