import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data


class ContentBaseModel(nn.Module):

    def __init__(self,  embed_dim,enc_post_title, device="cpu"):
        super(ContentBaseModel, self).__init__()
        self.device = device
        self. embed_dim =  embed_dim

        self.p2e = enc_post_title
        self.u2e = nn.Embedding(20000, embed_dim).to(device)

        self.w_u1 = nn.Linear(self.embed_dim, self.embed_dim).to(self.device)
        self.w_u2 = nn.Linear(self.embed_dim, self.embed_dim).to(self.device)

        self.w_p1 = nn.Linear(768, self.embed_dim).to(self.device)
        self.w_p2 = nn.Linear(self.embed_dim, self.embed_dim).to(self.device)

        self.w_up1 = nn.Linear(self.embed_dim * 2, self.embed_dim).to(self.device)
        self.w_up2 = nn.Linear(self.embed_dim, 16).to(self.device)
        self.w_up3 = nn.Linear(16, 1).to(self.device)
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5).to(self.device)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5).to(self.device)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5).to(self.device)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5).to(self.device)
        self.criterion = nn.MSELoss().to(self.device)

    def forward(self, nodes_u, nodes_v):
        embeds_p = []
        for i in nodes_u.numpy():
            embeds_p.append(self.p2e[i])
        if self.device == "cpu":
            embeds_p = torch.FloatTensor(embeds_p)
        else:
            embeds_p = torch.cuda.FloatTensor(embeds_p)
        embeds_u = self.u2e.weight[nodes_u.numpy()].to(self.device)

        x_u = F.relu(self.bn1(self.w_u1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_u2(x_u)
        x_p = F.relu(self.bn2(self.w_p1(embeds_p)))
        x_p = F.dropout(x_p, training=self.training)
        x_p = self.w_p2(x_p)

        x_uv = torch.cat((x_u, x_p), 1)
        x = F.relu(self.bn3(self.w_up1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_up2(x)))
        x = F.dropout(x, training=self.training)
        scores = F.sigmoid(self.w_up3(x))
        return scores.squeeze()
    
    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)