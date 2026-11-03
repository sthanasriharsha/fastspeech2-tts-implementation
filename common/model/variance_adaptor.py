import torch
import torch.nn as nn
import torch.nn.functional as F

class VariancePredictor(nn.Module):
    """
    Duration, Pitch, or Energy Predictor composed of 2-layer Conv1D with ReLU + LayerNorm.
    """
    def __init__(self, d_model=256, filter_size=256, kernel_size=3, dropout=0.5):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(d_model, filter_size, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.LayerNorm(filter_size),
            nn.Dropout(dropout)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(filter_size, filter_size, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.LayerNorm(filter_size),
            nn.Dropout(dropout)
        )
        self.linear_layer = nn.Linear(filter_size, 1)

    def forward(self, encoder_output, mask=None):
        out = encoder_output.contiguous().transpose(1, 2)
        out = self.conv_1[0](out).transpose(1, 2)
        out = self.conv_1[1:](out).transpose(1, 2)
        out = self.conv_2[0](out).transpose(1, 2)
        out = self.conv_2[1:](out)
        out = self.linear_layer(out).squeeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out


class LengthRegulator(nn.Module):
    """
    Expands hidden states according to predicted or ground truth durations.
    """
    def __init__(self):
        super().__init__()

    def LR(self, x, duration, max_len):
        output = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
        if max_len is not None:
            output = self.pad(output, max_len)
        else:
            output = self.pad(output)
        return output

    def expand(self, batch, predicted):
        out = list()
        for i, vec in enumerate(batch):
            expand_size = int(predicted[i].item())
            if expand_size > 0:
                out.append(vec.expand(expand_size, -1))
        if len(out) == 0:
            return batch[:1]
        out = torch.cat(out, 0)
        return out

    def pad(self, input_ele, mel_max_len=None):
        if mel_max_len:
            max_len = mel_max_len
        else:
            max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])
        out_list = list()
        for i, batch in enumerate(input_ele):
            if batch.size(0) < max_len:
                one_batch_padded = F.pad(batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0)
                out_list.append(one_batch_padded)
            else:
                out_list.append(batch[:max_len])
        return torch.stack(out_list)

    def forward(self, x, duration, max_len=None):
        output = self.LR(x, duration, max_len)
        return output


class VarianceAdaptor(nn.Module):
    """
    Variance Adaptor module incorporating Duration, Pitch, and Energy predictors.
    """
    def __init__(self, d_model=256, n_bins=256):
        super().__init__()
        self.duration_predictor = VariancePredictor(d_model)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(d_model)
        self.energy_predictor = VariancePredictor(d_model)

        self.pitch_bins = nn.Parameter(torch.linspace(0.0, 500.0, n_bins - 1), requires_grad=False)
        self.energy_bins = nn.Parameter(torch.linspace(0.0, 100.0, n_bins - 1), requires_grad=False)

        self.pitch_embedding = nn.Embedding(n_bins, d_model)
        self.energy_embedding = nn.Embedding(n_bins, d_model)

    def forward(
        self,
        x,
        src_mask=None,
        mel_mask=None,
        d_target=None,
        p_target=None,
        e_target=None,
        max_len=None
    ):
        # Predict duration
        log_d_prediction = self.duration_predictor(x, src_mask)
        
        if d_target is not None:
            lr_output = self.length_regulator(x, d_target, max_len)
            duration_rounded = d_target
        else:
            duration_rounded = torch.clamp((torch.exp(log_d_prediction) - 1.0).round(), min=0)
            lr_output = self.length_regulator(x, duration_rounded, max_len)

        # Predict pitch
        p_prediction = self.pitch_predictor(lr_output, mel_mask)
        if p_target is not None:
            p_embedding = self.pitch_embedding(torch.bucketize(p_target, self.pitch_bins))
        else:
            p_embedding = self.pitch_embedding(torch.bucketize(p_prediction, self.pitch_bins))

        # Predict energy
        e_prediction = self.energy_predictor(lr_output, mel_mask)
        if e_target is not None:
            e_embedding = self.energy_embedding(torch.bucketize(e_target, self.energy_bins))
        else:
            e_embedding = self.energy_embedding(torch.bucketize(e_prediction, self.energy_bins))

        adapted_output = lr_output + p_embedding + e_embedding
        return adapted_output, log_d_prediction, p_prediction, e_prediction, duration_rounded

