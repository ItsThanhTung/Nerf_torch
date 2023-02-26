import torch.nn as nn
import torch
from src.models.nerf import NeRF

class EncoderNeRF(nn.Module):
    def __init__(self, coord_encoder, view_encoder, device, N_importance=64, netchunk=1024*64):
        """ 
        """
        super(EncoderNeRF, self).__init__()

        self.coord_encoder_fn = coord_encoder[0]
        self.input_coord_ch = coord_encoder[1]

        self.view_encoder_fn = view_encoder[0]
        self.input_viewdir_ch = view_encoder[1]

        self.nerf_model = NeRF(D=8, W=256, input_ch=self.input_coord_ch,\
                            input_ch_views=self.input_viewdir_ch , output_ch=5, \
                            skips=[4], use_viewdirs=True).to(device)
 
        self.netchunk = netchunk


    def forward(self, inputs, viewdirs):

        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = self.coord_encoder_fn(inputs_flat)

        if viewdirs is not None:
            input_dirs = viewdirs[:,None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = self.view_encoder_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        outputs_flat = batchify(self.nerf_model, self.netchunk)(embedded)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs

    def get_grad_vars(self):
        return list(self.nerf_model.parameters()) 

    def get_state_dict(self):
        return self.nerf_model.state_dict()

    def load_state_dict(self, state_dict):
        self.nerf_model.load_state_dict(state_dict, strict=True)

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


  