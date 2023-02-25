import torch
import torch.nn as nn
import torch.nn.functional as F



class PositionalEncoding:
    def __init__(self, multires):
        self.max_freq_log2 = multires-1
        self.num_freqs = multires
        self.input_dims = 3
        self.include_input = True

        self.periodic_fns = [torch.sin, torch.cos]
        self.log_sampling = True


        self.encode_fns, self.out_dim = self.create_embedding_fn()
        
    def create_embedding_fn(self):
        encode_fns = []
        d = self.input_dims
        out_dim = 0

        if self.include_input:
            encode_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs
        
        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                encode_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
        
        return encode_fns, out_dim
        
        
    def encode(self, inputs):
        return torch.cat([fn(inputs) for fn in self.encode_fns], -1)


def get_position_encoder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    positional_encoder = PositionalEncoding(multires)
    embed = lambda x, eo=positional_encoder : eo.encode(x)
    return embed, positional_encoder.out_dim