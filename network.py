import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from torch.autograd import Variable
from sequence import EventSeq, ControlSeq

class EventSequenceEncoder(nn.Module):
    def __init__(self, event_dim=EventSeq.dim(), hidden_dim=512,
                 gru_layers=3, gru_dropout=0.3):
        super().__init__()
        self.event_embedding = nn.Embedding(event_dim, event_dim)
        self.gru = nn.GRU(event_dim, hidden_dim,
                          num_layers=gru_layers, dropout=gru_dropout)
        self.attn = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)
        self.output_fc = nn.Linear(hidden_dim, 128)
        self.output_fc_activation = nn.Sigmoid()

    def forward(self, events, hidden=None, output_logits=False):
        # events: [steps, batch_size]
        events = self.event_embedding(events)
        outputs, _ = self.gru(events, hidden) # [t, b, h]
        weights = (outputs * self.attn).sum(-1, keepdim=True)
        output = (outputs * weights).mean(0) # [b, h]
        output = self.output_fc(output).squeeze(-1) # [b]
        if output_logits:
            return output
        output = self.output_fc_activation(output)
        return output

class OSRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(OSRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.05)
        self.fc = nn.Linear(hidden_size, 512)
        self.event_embedding = nn.Embedding(EventSeq.dim(), EventSeq.dim())

        self.num_layers = num_layers

    def forward(self, input_layer):
        h0 = torch.zeros(self.num_layers, input_layer.size[1:2], self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input_layer.size[1:2], self.hidden_size).to(device)
        
        input_layer = self.event_embedding(input_layer)
        out, hidden = self.lstm(input_layer, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def _class_name(self):
        return "TripletNet"

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()