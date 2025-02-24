import torch
import torch.nn as nn
from data.data import get_iter
import torch.nn.functional as F

class Truncated_power():
    def __init__(self, degree, knots):

        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):

        x = x.squeeze()
        out = torch.zeros(x.shape[0], self.num_of_basis)
        for _ in range(self.num_of_basis):
            if _ <= self.degree:
                if _ == 0:
                    out[:, _] = 1.
                else:
                    out[:, _] = x**_
            else:
                if self.degree == 1:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree]))
                else:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree - 1])) ** self.degree

        return out

class Dynamic_FC(nn.Module):
    def __init__(self, ind, outd, degree, knots, act='relu', isbias=1, islastlayer=0):
        super(Dynamic_FC, self).__init__()
        self.ind = ind
        self.outd = outd
        self.degree = degree
        self.knots = knots

        self.islastlayer = islastlayer

        self.isbias = isbias

        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis # num of basis

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd, self.d), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd, self.d), requires_grad=True)
        else:
            self.bias = None

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):

        x_feature = x[:, 1:]
        x_treat = x[:, 0]
        x_feature_weight = torch.matmul(self.weight.T, x_feature.T).T # bs, outd, d
        x_treat_basis = self.spb.forward(x_treat) # bs, d
        x_treat_basis_ = torch.unsqueeze(x_treat_basis, 1)

        out = torch.sum(x_feature_weight * x_treat_basis_, dim=2) # bs, outd

        if self.isbias:
            out_bias = torch.matmul(self.bias, x_treat_basis.T).T
            out = out + out_bias

        if self.act is not None:
            out = self.act(out)


        if not self.islastlayer:
            out = torch.cat((torch.unsqueeze(x_treat, 1), out), 1)

        return out


def comp_grid(y, num_grid):



    U = torch.ceil(y * num_grid)
    inter = 1 - (U - y * num_grid)
    L = U - 1
    L += (L < 0).int()

    return L.int().tolist(), U.int().tolist(), inter


class Density_Block(nn.Module):
    def __init__(self, num_grid, ind, isbias=1):
        super(Density_Block, self).__init__()
        """
        Assume the variable is bounded by [0,1]
        the output grid: 0, 1/B, 2/B, ..., B/B; output dim = B + 1; num_grid = B
        """
        self.ind = ind
        self.num_grid = num_grid
        self.outd = num_grid + 1

        self.isbias = isbias

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd), requires_grad=True)
        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd), requires_grad=True)
        else:
            self.bias = None

        self.softmax = nn.Softmax(dim=1)

    def forward(self, t, x):
        out = torch.matmul(x, self.weight)
        if self.isbias:
            out += self.bias
        out = self.softmax(out)

        x1 = list(torch.arange(0, x.shape[0]))
        L, U, inter = comp_grid(t, self.num_grid)

        L_out = out[x1, L]
        U_out = out[x1, U]

        out = L_out + (U_out - L_out) * inter

        return out

class CGNet(nn.Module):
    def __init__(self, cfg_density, num_grid, cfg, degree, knots):
        super(CGNet, self).__init__()



        self.cfg_density = cfg_density
        self.num_grid = num_grid

        self.cfg = cfg
        self.degree = degree
        self.knots = knots

        # construct the density estimator
        density_blocks = []
        density_hidden_dim = -1
        for layer_idx, layer_cfg in enumerate(cfg_density):
            # fc layer
            if layer_idx == 0:
                # weight connected to feature
                self.feature_weight = nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1],
                                                bias=layer_cfg[2])
                density_blocks.append(self.feature_weight)
            else:
                density_blocks.append(
                    nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2]))
            density_hidden_dim = layer_cfg[1]
            if layer_cfg[3] == 'relu':
                density_blocks.append(nn.ReLU(inplace=True))
            elif layer_cfg[3] == 'tanh':
                density_blocks.append(nn.Tanh())
            elif layer_cfg[3] == 'sigmoid':
                density_blocks.append(nn.Sigmoid())
            else:
                print('No activation')

        self.hidden_features = nn.Sequential(*density_blocks)

        self.density_hidden_dim = density_hidden_dim
        self.density_estimator_head = Density_Block(self.num_grid, density_hidden_dim, isbias=1)

        # construct the dynamics network
        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                last_layer = Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3],
                                        isbias=layer_cfg[2], islastlayer=1)
            else:
                blocks.append(
                    Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3],
                               isbias=layer_cfg[2], islastlayer=0))
        blocks.append(last_layer)
        self.Q = nn.Sequential(*[nn.Linear(1, 50)])
        self.Q2 = nn.Sequential(*[nn.Linear(100, 100), nn.ReLU(), nn.ReLU(), nn.Linear(100, 1)])

        self.fc_mu_l = nn.Linear(20 * 2, 20 )
        self.fc_logvar_l = nn.Linear(20 * 2, 20 )

        self.fc1 = nn.Linear(50, 50)
        self.fc1_add = nn.Linear(50, 20)
        self.fc_mu = nn.Linear(20, 20)
        self.fc_logvar = nn.Linear(20, 20)

        self.fc2 = nn.Linear(20, 50)
        self.fc3 = nn.Linear(50, 8)

        self.fc22 = nn.Linear(20, 100)
        self.fc22_add = nn.Linear(100, 100)
        self.fc22_add2 = nn.Linear(100, 100)
        self.fc22_add3 = nn.Linear(100, 100)
        self.fc33 = nn.Linear(100, 6)
        self.maplabel = nn.Linear(1, 20)
        self.recons_loss = nn.BCELoss(reduction='sum')
        self.densel_z = nn.Linear(20 , 20 * 2)
        self.densel_z_2 = nn.Linear(20, 20 * 10)
        self.densel_z_3 = nn.Linear(20, 20 * 10)

        self.maplabel2=nn.Linear(40, 20)
        self.maplabel3 = nn.Linear(20, 20)
        self.maplabel4 = nn.Linear(20, 20)
        self.maplabel5 = nn.Linear(20, 20)
        # self.Q = nn.Sequential(*blocks)

    def pixel_norm(self, x, epsilon=1e-8):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + epsilon)

    def map_label_z(self, l, z):
        l = self.maplabel(l)
        x = torch.concat((z, l), axis=1)
        x = self.pixel_norm(x)
        return x

    def StyleMod(self, x, l_z):
        style_l_z = self.densel_z(l_z)
        style_l_z = style_l_z.view(-1, 2, x.size(1))

        return x * (style_l_z[:, 0] + 1) + style_l_z[:, 1]

    def StyleMod_2(self, x, l_z):
        style_l_z = self.densel_z_2(l_z)
        style_l_z = style_l_z.view(-1, 2, x.size(1))

        return x * (style_l_z[:, 0] + 1) + style_l_z[:, 1]

    def StyleMod_3(self, x, l_z):
        style_l_z = self.densel_z_3(l_z)
        style_l_z = style_l_z.view(-1, 2, x.size(1))

        return x * (style_l_z[:, 0] + 1) + style_l_z[:, 1]


    def encode_l(self,  labels):
        z_l = torch.randn(len(labels), 20)

        labels = labels.unsqueeze(1)
        latent_z_l = self.map_label_z(labels, z_l)
        latent_z_l = torch.relu(latent_z_l)

        latent_z_l = self.maplabel2(latent_z_l)
        latent_z_l = torch.relu(latent_z_l)
        latent_z_l = self.maplabel3(latent_z_l)
        #latent_z_l = torch.relu(latent_z_l)
        #latent_z_l = self.maplabel4(latent_z_l)
        #

        return latent_z_l
    def reparametrization(self, mu, logvar):
        # sigma = 0.5*exp(log(sigma^2))= 0.5*exp(log(var))
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        # N(mu, std^2) = N(0, 1) * std + mu
        z = eps * std + mu
        return z
    def encode(self, x):

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc1_add(x))
        # x = torch.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z_x):  # z_x,z_l, labels
        # z = torch.cat((z, labels), dim=1)

        z_x = torch.relu(self.fc2(z_x))
        z_x = self.fc3(z_x)
        return z_x

    def decode_pred(self, z,mu_l):  # z_x,z_l, labels

        x_styled = self.StyleMod(z, mu_l)

        pred = torch.relu(self.fc22(x_styled))
        pred = self.StyleMod_2(pred, mu_l)
        pred = torch.relu(self.fc22_add(pred))
        pred = self.StyleMod_3(pred, mu_l)
        pred = torch.relu(self.fc22_add2(pred))
        pred = self.fc22_add3(pred)
        pred = self.fc33(pred)
        return pred



    def forward(self, t, x):
        hidden = self.hidden_features(x)
        t_hidden = torch.cat((torch.unsqueeze(t, 1), hidden), 1)
        g = self.density_estimator_head(t, hidden)
        mu_l = self.encode_l(t)
        mu, logvar = self.encode(hidden)
        z = self.reparametrization(mu, logvar)
        x_out = self.decode(z)
        Q=self.decode_pred(z,mu_l)

        return x_out,g, Q,mu,logvar,mu,logvar

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, Dynamic_FC):
                m.weight.data.normal_(0, 1.)
                if m.isbias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Density_Block):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()
