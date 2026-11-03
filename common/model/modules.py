import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        output = torch.bmm(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.fc = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        output, _ = self.attention(q, k, v, mask=mask)
        output = output.view(n_head, sz_b, len_q, d_v).permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, fft_conv1d_kernel_size, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, kernel_size=fft_conv1d_kernel_size[0], padding=(fft_conv1d_kernel_size[0] - 1) // 2)
        self.w_2 = nn.Conv1d(d_hid, d_in, kernel_size=fft_conv1d_kernel_size[1], padding=(fft_conv1d_kernel_size[1] - 1) // 2)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(torch.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class FFTBlock(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, d_inner, fft_conv1d_kernel_size, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, fft_conv1d_kernel_size, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask
        enc_output = self.pos_ffn(enc_output)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask
        return enc_output

