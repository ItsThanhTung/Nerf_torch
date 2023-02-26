import torch
from src.models.EncoderNeRF import EncoderNeRF
from src.models.PositionalEncoding import get_coord_view_encoder

from src.utils.render import Rendering

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    renderer = Rendering(n_samples=64, n_importance=64, perturb=1.0, \
                        lindisp=False, raw_noise_std=1.0, white_bkgd=False)


    N_importance = 64  # num coarsed sampling points
    coord_encoder, view_encoder =  get_coord_view_encoder()

    model_state_dict = torch.load(r'C:\Users\Asus\Downloads\VINAI\ComputerGraphic\Nerf_torch\model.pt')
    coarsed_net = EncoderNeRF(coord_encoder, view_encoder, device, N_importance=N_importance)
    coarsed_net.load_state_dict(model_state_dict)

    model_state_dict = torch.load(r'C:\Users\Asus\Downloads\VINAI\ComputerGraphic\Nerf_torch\fined_model.pt')
    fined_net = EncoderNeRF(coord_encoder, view_encoder, device, N_importance=N_importance)
    fined_net.load_state_dict(model_state_dict)

    input = torch.load(r'C:\Users\Asus\Downloads\VINAI\ComputerGraphic\Nerf_torch\ray_batch.pt')
    test_pts = torch.load(r'C:\Users\Asus\Downloads\VINAI\ComputerGraphic\Nerf_torch\pts.pt')

    torch.manual_seed(0)
    renderer.set_rays(input)
    pts = renderer.sample_coarsed_pts(input)
    raw = coarsed_net.forward(pts, renderer.viewdirs)

    test_raw = torch.load(r'C:\Users\Asus\Downloads\VINAI\ComputerGraphic\Nerf_torch\raw.pt')
    print("pts: ", torch.equal(test_pts, pts))
    print("raw: ", torch.equal(test_raw, raw))

    rgb_map, disp_map, acc_map, weights, depth_map = renderer.rendering(raw, is_fined=False)

    test_rgb_map = torch.load(r'C:\Users\Asus\Downloads\VINAI\ComputerGraphic\Nerf_torch\rgb_map.pt')
    test_disp_map = torch.load(r'C:\Users\Asus\Downloads\VINAI\ComputerGraphic\Nerf_torch\disp_map.pt')
    test_acc_map = torch.load(r'C:\Users\Asus\Downloads\VINAI\ComputerGraphic\Nerf_torch\acc_map.pt')
    test_weights = torch.load(r'C:\Users\Asus\Downloads\VINAI\ComputerGraphic\Nerf_torch\weights.pt')
    test_depth_map = torch.load(r'C:\Users\Asus\Downloads\VINAI\ComputerGraphic\Nerf_torch\depth_map.pt')

    print("rgb_map: ", torch.equal(rgb_map, test_rgb_map))
    print("disp_map: ", torch.equal(disp_map, test_disp_map))
    print("acc_map: ", torch.equal(acc_map, test_acc_map))
    print("weights: ", torch.equal(weights, test_weights))
    print("depth_map: ", torch.equal(depth_map, test_depth_map))
    

    fined_pts = renderer.sample_fined_pts()
    test_fined_pts = torch.load(r'C:\Users\Asus\Downloads\VINAI\ComputerGraphic\Nerf_torch\fined_pts.pt')
    print("fined_pts: ", torch.equal(fined_pts, test_fined_pts))

    fined_raw = fined_net.forward(fined_pts, renderer.viewdirs)
    test_fined_raw = torch.load(r'C:\Users\Asus\Downloads\VINAI\ComputerGraphic\Nerf_torch\fined_raw.pt')
    print("fined_raw: ", torch.equal(fined_raw, test_fined_raw))

    fined_rgb_map, fined_disp_map, fined_acc_map, fined_weights, fined_depth_map = renderer.rendering(fined_raw, is_fined=True)
    
    test_fined_rgb_map = torch.load(r'C:\Users\Asus\Downloads\VINAI\ComputerGraphic\Nerf_torch\fined_rgb_map.pt')
    test_fined_disp_map = torch.load(r'C:\Users\Asus\Downloads\VINAI\ComputerGraphic\Nerf_torch\fined_disp_map.pt')
    test_fined_acc_map = torch.load(r'C:\Users\Asus\Downloads\VINAI\ComputerGraphic\Nerf_torch\fined_acc_map.pt')
    test_fined_weights = torch.load(r'C:\Users\Asus\Downloads\VINAI\ComputerGraphic\Nerf_torch\fined_weights.pt')
    test_fined_depth_map = torch.load(r'C:\Users\Asus\Downloads\VINAI\ComputerGraphic\Nerf_torch\fined_depth_map.pt')

    print("fined_rgb_map: ", torch.equal(fined_rgb_map, test_fined_rgb_map))
    print("fined_disp_map: ", torch.equal(fined_disp_map, test_fined_disp_map))
    print("fined_acc_map: ", torch.equal(fined_acc_map, fined_acc_map))
    print("fined_weights: ", torch.equal(fined_weights, test_fined_weights))
    print("fined_depth_map: ", torch.equal(fined_depth_map, test_fined_depth_map))