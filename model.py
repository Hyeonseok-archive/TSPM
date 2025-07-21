import math
import torch
from torch import nn
from timm.models.layers import DropPath, trunc_normal_


class S_layer(nn.Module):
    def __init__(self, in_shape, hid, drop_path=0.0, init_value=1e-2):
        super(S_layer, self).__init__()
        T, H, W = in_shape
        self.norm = nn.LayerNorm((1, H, W), eps=1e-6)
        self.layer = nn.Sequential(
            nn.Conv2d(1, hid, 1),
            nn.Conv2d(hid, hid, kernel_size=3, groups=hid, padding='same'),
        )
        self.att = nn.Sequential(
            nn.Conv2d(1, hid, 1),
            # act
            nn.SiLU(inplace=True)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((hid)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((hid)), requires_grad=True)
        self.final = nn.Conv2d(hid, 1, 1)
        self.apply(self._init_weights)

    def forward(self, x):
        B, T, H, W = x.shape
        x = x.view(B*T, 1, H, W)
        a = self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.att(self.norm(x)))
        x = self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.layer(self.norm(x)))
        x = a * x
        x = self.final(x)
        x = x.view(B, T, H, W)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):            
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            trunc_normal_(m.weight, std=math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @ torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}


class T_layer(nn.Module):
    def __init__(self, in_shape, hid, drop_path=0.0, init_value=1e-2):
        super(T_layer, self).__init__()
        T, H, W = in_shape
        self.norm = nn.LayerNorm((T, H, W), eps=1e-6)
        self.initial = nn.Conv2d(T, hid, 1)
        self.statical = nn.Sequential(
            nn.Conv2d(hid, hid, 3, padding='same', groups=hid),
            nn.Conv2d(hid, hid, 3, dilation=2, padding='same', groups=hid),
            nn.Conv2d(hid, hid, 1)
        )
        self.dynamical = nn.Sequential(
            nn.AdaptiveAvgPool2d((H, W)),
            nn.Conv2d(hid, hid, 1)
        )
        self.final = nn.Conv2d(hid, T, 1)
        # act
        self.act = nn.SiLU(inplace=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((hid)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((hid)), requires_grad=True)

    def forward(self, x):
        x = self.act(self.initial(self.norm(x)))
        # statical 의 출력을 바로 T가 아닌 hid? or T
        t = self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.statical(x))
        x = self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.dynamical(x))
        x = t * x
        x = self.act(x)
        x = self.final(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):            
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            trunc_normal_(m.weight, std=math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}
    

class STModel(nn.Module):
    def __init__(self, in_shape, s_hid, t_hid, n_section, drop_path=0.0, init_value=1e-2):
        super(STModel, self).__init__()
        self.S_layer = S_layer(
            in_shape=in_shape, hid=s_hid, drop_path=drop_path, init_value=init_value
        )
        self.T_layer = T_layer(
            in_shape=in_shape, hid=t_hid, drop_path=drop_path, init_value=init_value
        )
        self.final = nn.AdaptiveAvgPool2d(output_size=(1, n_section))

    def forward(self, x):
        x = self.S_layer(x)
        # print(f'S:{x.shape}')
        x = self.T_layer(x)
        # print(f'T:{x.shape}')
        x = self.final(x)
        # print(f'Final:{x.shape}')
        x = x.squeeze(2)
        # print(f'Squeeze:{x.shape}')
        
        return x