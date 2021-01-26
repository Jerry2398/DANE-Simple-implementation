import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layers import GraphConvolution
import torch
import math
from torch import autograd
from torch import optim

LAMBDA = 10

class NetD(nn.Module):
    def __init__(self, nhid, opt=None):
        super(NetD, self).__init__()
        self.emb_dim = nhid
        if opt is not None:
            self.dis_layers = opt['dis_layers']
            self.dis_hid_dim = opt['dis_hid_dim']
            self.dis_dropout = opt['dis_dropout']
            self.dis_input_dropout = opt['dis_input_dropout']
        else:
            self.dis_layers = 2
            self.dis_hid_dim = 4*nhid
            self.dis_dropout = 0.1
            self.dis_input_dropout = 0.1

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dis_dropout))
        #layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)



    def forward(self, x):
        return self.layers(x).view(-1)

    def calc_gradient_penalty(self, real_data, fake_data, BATCH_SIZE, use_cuda):
        # print "real_data: ", real_data.size(), fake_data.size()
        alpha = torch.rand(BATCH_SIZE, 1)
        alpha = alpha.expand(BATCH_SIZE, real_data.nelement() / BATCH_SIZE).contiguous().view(BATCH_SIZE, self.emb_dim)
        alpha = alpha.cuda() if use_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if use_cuda:
            interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.forward(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(
                                      ) if use_cuda else torch.ones(
                                      disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, multilabel, nclass, dropout):
        super(GCN, self).__init__()

        self.multilabel = multilabel
        self.gc1 = GraphConvolution(nfeat, nhid, bias = False)
        torch.nn.init.xavier_normal(self.gc1.weight, gain=1)
        self.gc2 = GraphConvolution(nhid, nhid,bias = False)
        torch.nn.init.xavier_normal(self.gc2.weight, gain=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.weight = Parameter(torch.FloatTensor(nhid, nclass))
        self.dropout = dropout
        self.reset_parameters()

    def forward(self, x, adj):
        x1 = F.leaky_relu(self.gc1(x, adj))
        x3 = self.gc2(x1, adj)
        if self.multilabel:
            Y = self.sigmoid(torch.mm(x3,self.weight))
        else:
            Y = torch.mm(x3,self.weight)
        return Y, x3


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)


class MLP(nn.Module):
    def __init__(self, in_features, nclass, bias=True):
        super(MLP, self).__init__()
        self.weight1 = Parameter(torch.FloatTensor(in_features, in_features))
        self.weight2 = Parameter(torch.FloatTensor(in_features, nclass))
        if bias:
            self.bias1 = Parameter(torch.FloatTensor(in_features))
            self.bias2 = Parameter(torch.FloatTensor(nclass))
        else:
            self.register_parameter('bias1', None)
            self.register_parameter('bias2', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv, stdv)
        if self.bias1 is not None:
            self.bias1.data.uniform_(-stdv, stdv)
        if self.bias2 is not None:
            self.bias2.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight1)
        if self.bias1 is not None:
            output += self.bias1
        output = torch.mm(output, self.weight2)
        if self.bias2 is not None:
            return F.log_softmax(output + self.bias2)
        else:
            return F.log_softmax(output)

class Center(nn.Module):
    def __init__(self, in_features):
        super(Center, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = -torch.mm(input, self.weight)
        output = F.logsigmoid(output)
        return -output
