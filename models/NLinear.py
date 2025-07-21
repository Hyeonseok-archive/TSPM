import torch
import torch.nn as nn


class NLinear(nn.Module):
    """
    NLinear
    """

    def __init__(self, seq_len, pred_len, c_in, mode='multi'):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.channels = c_in
        self.individual = mode
        if self.individual == 'multi':
            self.Linear = nn.ModuleList()
            for _ in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [B, T, C]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual == 'multi':
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [B, T, C]