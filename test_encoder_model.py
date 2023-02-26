from src.models.EncoderNeRF import EncoderNeRF
from src.models.PositionalEncoding import get_coord_view_encoder
from src.utils.utils import raw2outputs

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_importance = 64  # num coarsed sampling points
coord_encoder, view_encoder =  get_coord_view_encoder()

coarsed_net = EncoderNeRF(coord_encoder, view_encoder, device, N_importance=N_importance)

