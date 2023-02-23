from src.models.PositionalEncoding import get_position_encoder
import torch

multires = 10

input_data = torch.load(r'C:\Users\Asus\Downloads\VINAI\ComputerGraphic\Nerf_torch\input.pt')

embed, out_dim = get_position_encoder(multires, i=0)

out_data = embed(input_data)

test_out_data = torch.load(r'C:\Users\Asus\Downloads\VINAI\ComputerGraphic\Nerf_torch\output.pt')


print(torch.equal(out_data, test_out_data))


multires = 4

input_data = torch.load(r'C:\Users\Asus\Downloads\VINAI\ComputerGraphic\Nerf_torch\input_dir.pt')

embed, out_dim = get_position_encoder(multires, i=0)

out_data = embed(input_data)

test_out_data = torch.load(r'C:\Users\Asus\Downloads\VINAI\ComputerGraphic\Nerf_torch\output_dir.pt')
print(torch.equal(out_data, test_out_data))
