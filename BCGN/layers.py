import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(out_dim), nn.ReLU(True))


class CoordConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size,
                                padding, stride)

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 nhead,
                 dim_ffn,
                 dropout,
                 return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=d_model,
                                    nhead=nhead,
                                    dim_feedforward=dim_ffn,
                                    dropout=dropout) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    @staticmethod
    def pos1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512

    @staticmethod
    def pos2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512

    def forward(self, vis, txt, pad_mask):
       # vis: b, 512, h, w
       # txt: b, L, 512
       # pad_mask: b, L
        B, C, H, W = vis.size()
        _, L, D = txt.size()
        # position encoding
        vis_pos = self.pos2d(C, H, W)
        txt_pos = self.pos1d(D, L)
        # reshape & permute
        vis = vis.reshape(B, C, -1).permute(2, 0, 1)
        txt = txt.permute(1, 0, 2)
        # forward
        output = vis
        intermediate = []
        for layer in self.layers:
            output = layer(output, txt, vis_pos, txt_pos, pad_mask)
            if self.return_intermediate:
                # HW, b, 512 -> b, 512, HW
                intermediate.append(self.norm(output).permute(1, 2, 0))

        if self.norm is not None:
            # HW, b, 512 -> b, 512, HW
            output = self.norm(output).permute(1, 2, 0)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                # [output1, output2, ..., output_n]
                return intermediate
            else:
                # b, 512, HW
                return output
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=512,
                 nhead=9,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        # Normalization Layer
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        # Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout,
                                                    kdim=d_model,
                                                    vdim=d_model)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                 nn.ReLU(True), nn.Dropout(dropout),
                                 nn.LayerNorm(dim_feedforward),
                                 nn.Linear(dim_feedforward, d_model))
        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)

    def forward(self, vis, txt, vis_pos, txt_pos, pad_mask):
        '''
            vis: 26*26, b, 512
            txt: L, b, 512
            vis_pos: 26*26, 1, 512
            txt_pos: L, 1, 512
            pad_mask: b, L
        '''
        # Self-Attention
        vis2 = self.norm1(vis)
        q = k = self.with_pos_embed(vis2, vis_pos)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = vis + self.dropout1(vis2)
        # Cross-Attention
        vis2 = self.norm2(vis)
        vis2 = self.multihead_attn(query=self.with_pos_embed(vis2, vis_pos),
                                   key=self.with_pos_embed(txt, txt_pos),
                                   value=txt,
                                   key_padding_mask=pad_mask)[0]
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.dropout2(vis2)
        # FFN
        vis2 = self.norm3(vis)
        vis2 = self.ffn(vis2)
        vis = vis + self.dropout3(vis2)
        return vis
