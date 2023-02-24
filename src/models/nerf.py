import torch.nn as nn
import torch
import torch.nn.functional as F


class NeRF(nn.Module):
    def __init__(self, depth=8,
                 intermediate_feature=256,
                 in_feature_coord=3,
                 in_feature_dir=3,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super().__init__()
        self._depth = depth
        self._intermediate_feature = intermediate_feature
        self._in_feature_coord = in_feature_coord
        self._in_feature_dir = in_feature_dir
        self._out_ch = output_ch
        self._skips = skips
        self._use_viewdir = use_viewdirs

        self._layers = nn.ModuleList([nn.Linear(self._in_feature_coord,
                                                self._intermediate_feature)] +
                                     [nn.Linear(self._intermediate_feature, self._intermediate_feature)
                                     if i not in self._skips else
                                     nn.Linear(self._intermediate_feature + self._in_feature_coord,
                                               self._intermediate_feature) for i in range(self._depth-1)])

        if self._use_viewdir:
            self._rgb_temp_layer = nn.Linear(
                self._intermediate_feature + self._in_feature_dir,
                self._intermediate_feature//2)
            self._bt_neck = nn.Linear(
                self._intermediate_feature, self._intermediate_feature)
            self._alpha_head = nn.Linear(self._intermediate_feature, 1)
            self._rgb_head = nn.Linear(self._intermediate_feature//2, 3)
        else:
            self._out_layer = nn.Linear(self._intermediate_feature, output_ch)
        # self.initialize()

    def forward(self, x):
        # x: N x (in_feature_coor + in_feature_dir)
        coor_ft, dir_ft = x[:, :self._in_feature_coord], x[:,
                                                           self._in_feature_coord:]
        _x = coor_ft
        for i, layer in enumerate(self._layers):
            _x = F.relu(layer(_x))
            if i in self._skips:
                _x = torch.cat([_x, coor_ft], dim=-1)

        if self._use_viewdir:
            alphas = self._alpha_head(_x)
            intermediate = torch.concat([self._bt_neck(_x), dir_ft], -1)
            intermediate = F.relu(self._rgb_temp_layer(intermediate))
            rgb = self._rgb_head(intermediate)
            return torch.concat([rgb, alphas], -1)

        else:
            return self._out_layer(_x)

if __name__ == "__main__":
    a = torch.randn(5,6)
    md = NeRF()
    print(md(a))