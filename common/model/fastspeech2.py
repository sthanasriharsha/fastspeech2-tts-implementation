import torch
import torch.nn as nn
from .modules import FFTBlock
from .variance_adaptor import VarianceAdaptor

class FastSpeech2(nn.Module):
    """
    FastSpeech 2 End-to-End Architecture
    """
    def __init__(self, vocab_size=300, d_model=256, n_mel_channels=80):
        super().__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 4 FFT Blocks for Encoder
        self.encoder = nn.ModuleList([
            FFTBlock(d_model=d_model, n_head=2, d_k=128, d_v=128, d_inner=1024, fft_conv1d_kernel_size=[9, 1])
            for _ in range(4)
        ])
        
        # Variance Adaptor
        self.variance_adaptor = VarianceAdaptor(d_model=d_model)
        
        # 4 FFT Blocks for Mel Decoder
        self.decoder = nn.ModuleList([
            FFTBlock(d_model=d_model, n_head=2, d_k=128, d_v=128, d_inner=1024, fft_conv1d_kernel_size=[9, 1])
            for _ in range(4)
        ])
        
        self.mel_linear = nn.Linear(d_model, n_mel_channels)

    def forward(
        self,
        src_seq,
        src_mask=None,
        mel_mask=None,
        d_target=None,
        p_target=None,
        e_target=None,
        max_len=None
    ):
        out = self.encoder_embedding(src_seq)
        for enc_layer in self.encoder:
            out = enc_layer(out)

        (
            adapted,
            log_d_pred,
            p_pred,
            e_pred,
            duration_rounded
        ) = self.variance_adaptor(
            out,
            src_mask=src_mask,
            mel_mask=mel_mask,
            d_target=d_target,
            p_target=p_target,
            e_target=e_target,
            max_len=max_len
        )

        dec_out = adapted
        for dec_layer in self.decoder:
            dec_out = dec_layer(dec_out)

        mel_output = self.mel_linear(dec_out)

        return mel_output, log_d_pred, p_pred, e_pred, duration_rounded

