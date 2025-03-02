import torch
import torch.nn as nn
from math import sqrt

class AttrProxy(object):

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class Propogator(nn.Module):

    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()
        self.state_dim = state_dim

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        A_in = A[:, :, 0]
        A_out = A[:, :, 1]

        a_in = torch.matmul(A_in, state_in)
        a_out = torch.matmul(A_out, state_in)
        a = torch.cat((a_in, a_out, state_in), 1)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in.squeeze(0), a_out.squeeze(0), r * state_cur), 1)
        h_hat = self.transform(joined_input)

        output = (1 - z) * state_cur + z * h_hat
        return output
#多头注意力机制
class MultiHeadSelfAttention(nn.Module):
    dim_in: int
    dim_k: int
    dim_v: int
    num_heads: int

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0  #判断语句，条件为真程序继续执行，条件为否就会抛出异常
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self.fc_out = nn.Linear(dim_v, dim_in)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

        self.feed_forward = nn.Sequential(
            nn.Linear(dim_in, 2 * dim_in),
            nn.ReLU(),
            nn.Linear(2 * dim_in, dim_in),
        )
        self.layer_norm1 = nn.LayerNorm(dim_in)
        self.layer_norm2 = nn.LayerNorm(dim_in)

    def forward(self, x):

        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh
        dv = self.dim_v // nh

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)

        dist = torch.matmul(q, k.transpose(-2, -1)) * self._norm_fact
        dist = torch.softmax(dist, dim=-1)

        att = torch.matmul(dist, v)
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)

        att = self.fc_out(att)

        x = self.layer_norm1(x + att)

        x = self.feed_forward(x)

        x = self.layer_norm2(x + att)

        return x


class GSIGNN(nn.Module):
    def __init__(self, opt):
        super(GSIGNN, self).__init__()
        self.state_dim = opt['state_dim']
        self.n_edge_types = opt['n_edge_types']
        self.n_node = opt['n_node']
        self.n_steps = opt['n_steps']
        self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)

        self.attention = MultiHeadSelfAttention(
            dim_in=self.state_dim,
            dim_k=self.state_dim,
            dim_v=self.state_dim,
            num_heads=8
        )

        # Initialize incoming and outgoing edge embeddings
        for i in range(self.n_edge_types):
            setattr(self, "in_{}".format(i), nn.Linear(self.state_dim, self.state_dim))
            setattr(self, "out_{}".format(i), nn.Linear(self.state_dim, self.state_dim))

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, state_cur, A):
        for i_step in range(self.n_steps):

            if state_cur.dim() == 2:
                state_cur = state_cur.unsqueeze(0)

            state_cur = self.attention(state_cur)

            if state_cur.dim() == 3 and state_cur.shape[0] == 1:
                state_cur = state_cur.squeeze(0)

            state_in = state_out = state_cur
            for i in range(self.n_edge_types):
                in_transform = getattr(self, "in_{}".format(i))
                out_transform = getattr(self, "out_{}".format(i))
                state_in = in_transform(state_in)
                state_out = out_transform(state_out)

            state_cur = self.propogator(state_in, state_out, state_cur, A)

        return state_cur
