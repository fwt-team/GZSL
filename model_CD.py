import torch.nn as nn
import torch
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def reparameter(mu,sigma):
    return (torch.randn_like(mu) *sigma) + mu

class Embedding_Net(nn.Module):
    def __init__(self, opt):
        super(Embedding_Net, self).__init__()

        self.fc1 = nn.Linear(opt.resSize, opt.embedSize)
        self.fc2 = nn.Linear(opt.embedSize, opt.outzSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, features):
        embedding= self.relu(self.fc1(features))
        out_z = F.normalize(self.fc2(embedding), dim=1)
        return embedding,out_z

class MLP_CRITIC(nn.Module):
    def __init__(self, opt):
        super(MLP_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h
class MLP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h


    # hs和hu的语义匹配模块
class Dis_Embed_Att(nn.Module):
    def __init__(self, opt):
        super(Dis_Embed_Att, self).__init__()
        self.fc1 = nn.Linear(opt.hs_dim+opt.attSize, opt.nhF)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.nhF, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #feedback layer
        self.h = None
        self.apply(weights_init)

    def forward(self, input):
        self.h = self.lrelu(self.fc1(input))
        h = self.fc2(self.h)
        return h

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()

        self.hs_dim = opt.hs_dim
        self.hu_dim = opt.hu_dim
        self.fc1 = nn.Linear(opt.resSize, 2 * opt.resSize)
        self.fc2 = nn.Linear(2 * opt.resSize, opt.hidden_dim)
        self.fc3 = nn.Linear(opt.hs_dim, opt.outzSize)

        self.mu = nn.Linear(opt.hidden_dim,opt.hs_dim + opt.hu_dim)
        self.logvar = nn.Linear(opt.hidden_dim,opt.hs_dim + opt.hu_dim)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)
        self.apply(weights_init)


    def forward(self, img):
        out = self.fc1(img)
        # out = self.dropout(out)
        out = self.lrelu(out)
        out = self.fc2(out)
        # out = self.dropout(out)
        out = self.lrelu(out)
        mu = self.mu(out)
        # logvar = self.logvar(out)
        # sigma = torch.exp(logvar)
        # z = reparameter(mu,sigma)
        #直接是一个autoencoder
        z = mu
        hs = z[:, :self.hs_dim]
        hu = z[:, self.hs_dim:]
        # hs_l2_real = F.normalize(self.fc3(hs), dim=1)
        hs_l2_real = F.normalize(self.fc3(hs))
        # z[:, self.hs_dim:] = 0
        return mu, hs, hu, z, hs_l2_real


class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.hs_dim = opt.hs_dim
        self.hu_dim = opt.hu_dim
        self.fc1 = nn.Linear(opt.hs_dim + opt.hu_dim, opt.hidden_dim)
        self.fc2 = nn.Linear(opt.hidden_dim, 2 * opt.resSize)
        self.fc3 = nn.Linear(2 * opt.resSize, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)
        self.apply(weights_init)

    def forward(self, z):
        out = self.fc1(z)
        # out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.dropout(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

    # total correlation model 分辨hs和hu的无关性 通过重组排序的方法
class Dis_TC(nn.Module):
    def __init__(self, opt):
        super(Dis_TC, self).__init__()
        self.fc1 = nn.Linear(opt.hs_dim+opt.hu_dim, opt.nhF)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.nhF, 1)
        self.lrelu = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, hs,hu):
        h = torch.cat((hs,hu),dim=1)
        # print(h.shape)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        h = self.lrelu(h)
        return h