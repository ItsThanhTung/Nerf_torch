from src.models.PositionalEncoding import get_coord_view_encoder
from src.models.EncoderNeRF import EncoderNeRF
import torch

import os

def create_model():
    N_importance = 64  # num coarsed sampling points
    lrate = 0.0005
    basedir = r"C:\Users\Asus\Downloads\VINAI\ComputerGraphic\Nerf_torch\src\logs'"
    expname = r"trex_test"

    coord_encoder, view_encoder =  get_coord_view_encoder()

    coarsed_net = EncoderNeRF(coord_encoder, view_encoder, N_importance=N_importance)
    grad_vars = coarsed_net.get_grad_vars()

    if N_importance > 0:
        fined_net = EncoderNeRF(coord_encoder, view_encoder, N_importance=N_importance)

        grad_vars += fined_net.get_grad_vars()
    
    optimizer = torch.optim.Adam(params=grad_vars, lr=lrate, betas=(0.9, 0.999))

    start = 0
    basedir = basedir
    expname = expname

    return  coarsed_net, fined_net, start, grad_vars, optimizer