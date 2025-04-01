import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn.init as init
import yaml

class SparseMSSA(nn.Module):
    def __init__(self,
                 dim,
                 memory_token_num,
                 memory_hidden_token_num,
                 heads=8,
                 dim_head=64,
                 dropout=0.0,
                 memory_decay=False):
        super().__init__()

        self.memory_decay = memory_decay
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim),
                                    nn.Dropout(dropout)) if project_out else nn.Identity()

        self.memory_token_num = memory_token_num
        self.memory_hidden_token_num = memory_hidden_token_num
        self.mixer = nn.Sequential(
            nn.LayerNorm(memory_token_num),
            nn.Linear(memory_token_num, memory_hidden_token_num),
            nn.GELU(),
            nn.LayerNorm(memory_hidden_token_num),
            nn.Linear(memory_hidden_token_num, memory_token_num)
        )

    def forward(self, x, data_token_num):
        if(data_token_num + self.memory_token_num > x.size(1)):
            data_token_num = x.size(1) - self.memory_token_num
        print("data_token_num = ", data_token_num)
        qkv = self.qkv(x[:,:(self.memory_token_num + data_token_num), :])
        tokens = rearrange(qkv, 'b n (h d) -> b h n d', h=self.heads)

        #sparse attention
        memory_tokens = tokens[:, :, :self.memory_token_num, :]
        data_tokens = tokens[:, :, self.memory_token_num:, :]
        mutual_retrieval = torch.matmul(memory_tokens, data_tokens.transpose(-1, -2))  #b h memory_token_num data_token_num

        attn_memory_to_data = torch.nn.functional.normalize(mutual_retrieval, dim=-1)
        attn_memory_to_data = attn_memory_to_data ** 2
        attn_data_to_memory = torch.nn.functional.normalize(mutual_retrieval.transpose(-1, -2), dim=-1)
        attn_data_to_memory = attn_data_to_memory ** 2

        out_memory_tokens = torch.matmul(attn_memory_to_data, data_tokens)
        if(self.memory_decay):
            out_memory_tokens += memory_tokens / (data_token_num + 1)

        out_memory_tokens = rearrange(out_memory_tokens, 'b h n d -> b n (h d)')
        out_data_tokens = torch.matmul(attn_data_to_memory, memory_tokens) + data_tokens
        out_data_tokens = rearrange(out_data_tokens, 'b h n d -> b n (h d)')

        #memory tokens mixer
        out_memory_tokens_ = out_memory_tokens.transpose(-1, -2)
        out_memory_tokens_ = self.mixer(out_memory_tokens_)
        out_memory_tokens = out_memory_tokens_.transpose(-1, -2)

        out = torch.concatenate((out_memory_tokens, out_data_tokens), dim=1)
        out = self.to_out(out)

        x_ = torch.concatenate((out, x[:,(self.memory_token_num + data_token_num):, :]), dim=1)
        return x_

class OverCompleteISTABlock(nn.Module):
    def __init__(self,
                 dim,
                 C=4,
                 eta=0.1,
                 lmbda=0.1):
        super().__init__()
        self.eta = eta
        self.lmbda = lmbda
        self.C = C
        self.token_dim = dim

        self.D = nn.Parameter(torch.Tensor(dim, C * dim))
        with torch.no_grad():
            init.kaiming_uniform_(self.D)

        self.D1 = nn.Parameter(torch.Tensor(dim, C * dim))
        with torch.no_grad():
            init.kaiming_uniform_(self.D1)

        self.relu = nn.ReLU()

    def forward(self, token_tensor):
        negative_lasso_grad = F.linear(token_tensor, self.D.t(), bias=None)
        z1 = self.eta * negative_lasso_grad - self.eta * self.lmbda
        z1_relu = self.relu(z1)
        Dz1 = F.linear(z1_relu, self.D, bias=None)
        lasso_grad = F.linear(Dz1 - token_tensor, self.D.t(), bias=None)
        z2 = z1_relu - self.eta * lasso_grad - self.eta * self.lmbda
        z2_relu = self.relu(z2)
        xhat = F.linear(z2_relu, self.D1, bias=None)
        return xhat


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(x)


class Linear(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        return self.lin(x)


class CRATEEncoder(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 memory_token_num,
                 memory_hidden_token_num,
                 C,
                 eta=0.1,
                 lmbda=0.1,
                 dropout=0.0,
                 memory_decay=False):
        super().__init__()

        self.layers = nn.ModuleList()
        self.depth = depth
        for _ in range(depth):
            self.layers.extend([LayerNorm(dim),SparseMSSA(dim, memory_token_num, memory_hidden_token_num, heads, dim_head, dropout, memory_decay)])
            self.layers.extend([LayerNorm(dim), OverCompleteISTABlock(dim, C, eta, lmbda)])

    def forward(self, x, data_token_num):
        for index, op in enumerate(self.layers):
            if (index % 4 == 1):
                x = op(x, data_token_num) + x
            else:
                x = op(x)

        return x


class CRATEDecoder(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 memory_token_num,
                 memory_hidden_token_num,
                 dropout=0.0,
                 memory_decay=False):
        super().__init__()

        self.layers = nn.ModuleList()
        self.depth = depth
        for _ in range(depth):
            self.layers.extend([LayerNorm(dim), Linear(dim)])
            self.layers.extend([LayerNorm(dim), SparseMSSA(dim, memory_token_num, memory_hidden_token_num, heads, dim_head, dropout, memory_decay)])

    def forward(self, x, data_token_num):
        for index, op in enumerate(self.layers):
            if (index % 4 == 3):
                x = x - op(x, data_token_num)
            else:
                x = op(x)

        return x


class CRATEImgClassifier(nn.Module):
    def __init__(self,
                 image_height,
                 image_width,
                 patch_height,
                 patch_width,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 memory_token_num,
                 memory_hidden_token_num,
                 with_decoder=False,
                 channels=3,
                 C=4,
                 eta=0.1,
                 lmbda=0.1,
                 dropout=0.0,
                 emb_dropout=0.0,
                 memory_decay=False):
        super().__init__()

        self.memory_token_num = memory_token_num
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + memory_token_num, dim))

        self.memory_token_num = memory_token_num
        self.memory_token = nn.Parameter(torch.randn(1, memory_token_num, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.token_encoder = CRATEEncoder(dim, depth, heads, dim_head, memory_token_num, memory_hidden_token_num, C, eta, lmbda, dropout, memory_decay)
        self.with_decoder = with_decoder
        if (with_decoder):
            self.token_decoder = CRATEDecoder(dim, depth, heads, dim_head, memory_token_num, memory_hidden_token_num, dropout, memory_decay)

        self.score_mat = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(memory_token_num),
            nn.Linear(memory_token_num, 1)
        )

    def forward(self, img, data_token_num):
        x = self.to_patch_embedding(img)
        batchsize, n, _ = x.shape

        memory_tokens = repeat(self.memory_token, '1 g d -> b g d', b=batchsize)
        x = torch.concatenate((memory_tokens, x), dim=1)

        x += self.pos_embedding[:, :(n + self.memory_token_num)]
        x = self.dropout(x)
        x = self.token_encoder(x, data_token_num)

        if (self.with_decoder):
            x_reconstruct = self.token_decoder(x, data_token_num)

        logits = self.score_mat(x[:, :self.memory_token_num, :])
        logits = self.mlp_head(logits.transpose(-1, -2))
        logits = torch.squeeze(logits, dim=-1)

        if (self.with_decoder):
            return logits, x_reconstruct
        else:
            return logits

class FractalAtom(nn.Module):
    def __init__(self,
                 Global_token_num,
                 G_hidden,
                 Data_token_num,
                 dim_x,
                 dim_qkv,
                 head_num,
                 Layernum,
                 dim_out,
                 C=4,
                 eta=0.1,
                 lmbda=0.1,
                 dropout=0.0,
                 memory_decay=True):
        super().__init__()

        self.global_token_num = Global_token_num
        self.data_token_num = Data_token_num
        self.pos_embedding = nn.Parameter(torch.randn(1, Data_token_num + Global_token_num, dim_x))
        self.global_token = nn.Parameter(torch.randn(1, Global_token_num, dim_x))
        self.crate_beta = CRATEEncoder(dim_x,
                                       Layernum,
                                       head_num,
                                       dim_qkv // head_num,
                                       Global_token_num,
                                       G_hidden,
                                       C,
                                       eta,
                                       lmbda,
                                       dropout,
                                       memory_decay)
        self.ln1 = nn.LayerNorm(Global_token_num)
        self.linear1 = nn.Linear(Global_token_num, 1)
        self.ln2 = nn.LayerNorm(dim_x)
        self.linear2 = nn.Linear(dim_x, dim_out)

    def forward(self, x):
        batchsize, n, _ = x.shape
        if(n != self.data_token_num):
            print("in FractalAtom, x.shape is wrong")
            return None

        global_tokens = repeat(self.global_token, '1 g d -> b g d', b=batchsize)
        x = torch.concatenate((global_tokens, x), dim=1)
        x += self.pos_embedding[:, :(self.data_token_num + self.global_token_num)]
        x = self.crate_beta(x, self.data_token_num)
        z = x[:, :self.global_token_num]
        z = self.ln1(z.transpose(-1, -2))
        z = self.linear1(z)
        z = self.ln2(z.transpose(-1, -2))
        z = self.linear2(z)

        return z

class FractalImageEncoder(nn.Module):
    def __init__(self,
                 batchsize,
                 channels,
                 arch_file):
        super().__init__()

        with open(arch_file, 'r') as file:
            arch = yaml.safe_load(file)

        self.input_size = arch['input_size']
        self.K = arch['K']
        self.M = arch['M']
        if(self.K ** self.M != self.input_size):
            print('in FractalImageEncoder, arch is wrong, K ** M != input_size')
            return

        self.pixel_embedding_dim = arch['pixel_embedding_dim']
        self.pixel_embedding_kernel_size = arch['pixel_embedding_kernel_size']
        self.pixel_embedding = nn.Sequential(
            nn.Conv2d(channels, self.pixel_embedding_dim, self.pixel_embedding_kernel_size),
            nn.BatchNorm2d(self.pixel_embedding_dim),
            nn.GELU()
        )

        self.data_token_num = self.K ** 2
        self.Global_token_num_sets = arch['Global_token_num']
        self.G_hidden = arch['G_hidden']
        self.Data_token_num = arch['Data_token_num']
        for i in range(self.M):
            if(self.data_token_num != self.Data_token_num[i]):
                print('in FractalImageEncoder, arch is wrong, data_token_num != Data_token_num[i]， i = ', i)

        self.dim_x = arch['dim_x']
        self.dim_qkv = arch['dim_qkv']
        self.head_num = arch['head_num']
        self.Layernum = arch['Layernum']
        self.dim_out = arch['dim_out']
        self.C = arch['C']
        self.eta = arch['eta']
        self.lmbda = arch['lmbda']
        self.dropout = arch['dropout']
        self.memory_decay = arch['memory_decay']

        self.atoms = nn.ModuleList()
        for level in range(self.M):
            level = self.M - level - 1
            if(level == self.M - 1):
                self.atoms.extend([Rearrange('b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=self.K, p2=self.K),
                                   FractalAtom(Global_token_num=self.Global_token_num_sets[level],
                                         G_hidden=self.G_hidden[level],
                                         Data_token_num=self.Data_token_num[level],
                                         dim_x=self.dim_x[level],
                                         dim_qkv=self.dim_qkv[level],
                                         head_num=self.head_num[level],
                                         Layernum=self.Layernum[level],
                                         dim_out=self.dim_out[level],
                                         C=self.C[level],
                                         eta=self.eta[level],
                                         lmbda=self.lmbda[level],
                                         dropout=self.dropout[level],
                                         memory_decay=self.memory_decay[level])])
            else:
                self.atoms.extend([Rearrange('(b t p) n d -> (b t) (n p) d', b=batchsize, p=self.K * self.K),
                                   FractalAtom(Global_token_num=self.Global_token_num_sets[level],
                                               G_hidden=self.G_hidden[level],
                                               Data_token_num=self.Data_token_num[level],
                                               dim_x=self.dim_x[level],
                                               dim_qkv=self.dim_qkv[level],
                                               head_num=self.head_num[level],
                                               Layernum=self.Layernum[level],
                                               dim_out=self.dim_out[level],
                                               C=self.C[level],
                                               eta=self.eta[level],
                                               lmbda=self.lmbda[level],
                                               dropout=self.dropout[level],
                                               memory_decay=self.memory_decay[level])])




    def forward(self, x):
        x_ = self.pixel_embedding(x)
        for index, op in enumerate(self.atoms):
            print("计算前维度: ", x_.shape)
            x_ = op(x_)
            print("计算后维度: ", x_.shape)
        return x_





