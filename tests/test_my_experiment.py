import os
import pickle
import sys
import unittest
import torch
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
package_root = os.path.join(project_root, "packages")
sys.path.append(package_root)

from motion_inbetween import benchmark
from motion_inbetween.train import context_model, rmi
from motion_inbetween.config import load_config_by_name
from motion_inbetween.model import ContextTransformer
from motion_inbetween.data import utils_torch as data_utils

    
class TestMyExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Config, device, and model
        lafan1_context_model = "lafan1_context_model"
        self.config = load_config_by_name(lafan1_context_model)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = ContextTransformer(self.config["model"]).to(self.device)

        # Shared constants
        self.context_len = 10
        self.interpolation_window_offset = self.context_len
        self.dtype = torch.float32

        # Dummy test tensors
        self.positions = torch.rand(32, 65, 22, 3, dtype=self.dtype)
        self.rotations = torch.rand(32, 65, 22, 3, 3, dtype=self.dtype)
        self.parents = torch.tensor(
            [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16,
             11, 18, 19, 20]
        )
        trans_len = 15
        
        self.mean_rmi = torch.tensor([[-1.3550e-16,  8.2261e+01,  2.9567e-17],
        [ 1.3790e+00,  8.2255e+01, -8.7167e+00],
        [ 1.2952e+01,  4.8039e+01, -1.3106e+01],
        [-1.9123e+00,  1.7409e+01, -1.1137e+01],
        [ 7.7733e+00,  9.0416e+00, -1.3457e+01],
        [ 1.5946e+00,  8.2281e+01,  8.6834e+00],
        [ 1.3283e+01,  4.7551e+01,  1.2050e+01],
        [-2.2272e+00,  1.7239e+01,  1.1537e+01],
        [ 7.5058e+00,  9.2053e+00,  1.4457e+01],
        [-1.7892e+00,  8.8648e+01, -3.6445e-03],
        [ 4.7643e-01,  9.9766e+01, -7.2591e-02],
        [ 4.2435e+00,  1.0988e+02, -1.7152e-01],
        [ 1.4634e+01,  1.2862e+02, -4.5334e-01],
        [ 2.0094e+01,  1.3648e+02, -1.6513e-01],
        [ 1.1147e+01,  1.2492e+02, -5.2614e+00],
        [ 1.0541e+01,  1.2366e+02, -1.4205e+01],
        [ 6.3485e+00,  9.9732e+01, -2.4182e+01],
        [ 1.6915e+01,  8.7572e+01, -2.2820e+01],
        [ 1.1370e+01,  1.2486e+02,  4.5336e+00],
        [ 1.1296e+01,  1.2274e+02,  1.3351e+01],
        [ 6.6718e+00,  9.9588e+01,  2.3196e+01],
        [ 1.6639e+01,  8.7020e+01,  2.1650e+01]], dtype=self.dtype)
        self.std_rmi = torch.tensor([[47.0959, 28.0501, 23.8774],
        [47.0970, 28.2485, 24.1586],
        [49.8518, 23.5715, 28.0785],
        [54.2343, 21.4503, 30.8202],
        [56.3064, 20.8736, 32.9498],
        [47.3081, 28.1009, 24.3212],
        [49.9828, 23.7494, 28.1126],
        [54.3852, 21.2752, 30.6427],
        [56.3387, 20.8057, 32.7585],
        [47.0860, 28.9321, 23.7596],
        [46.9042, 30.7817, 23.9746],
        [46.9091, 32.8452, 24.6356],
        [47.7384, 38.6454, 27.5110],
        [48.4736, 41.0710, 29.4982],
        [47.5560, 37.0727, 26.6091],
        [48.0217, 37.0878, 26.9706],
        [50.4696, 37.1883, 28.2626],
        [51.3727, 39.6318, 31.9564],
        [47.6694, 37.0179, 26.7824],
        [48.2721, 36.9326, 27.3811],
        [51.1598, 37.6890, 28.8818],
        [51.8224, 40.3614, 32.6812]], dtype=self.dtype)

        self.indices = {'r_start_idx': 0, 'r_end_idx': 132, 'p_start_idx': 132, 'p_end_idx': 135, 'c_start_idx': 135, 'c_end_idx': 139}

        
    def test_model_masks_evaluation(self):
            
        for trans_len in [15,30,45]:
            for past_context_len in range(0,10):
                
                self.config["train"]["past_context"] = past_context_len #0
                beg_context_len = self.context_len - past_context_len # 10
                
                target_idx = self.interpolation_window_offset + trans_len #  25
                interpolation_window_slice = slice(self.interpolation_window_offset, target_idx)  # 10:25
                processing_window_len = self.interpolation_window_offset + trans_len + self.interpolation_window_offset

                beg_context_slice = slice(self.interpolation_window_offset- beg_context_len, self.interpolation_window_offset)
                past_context_slice = slice(target_idx, target_idx+past_context_len)

                mean, std = context_model.get_train_stats_torch(self.config, self.dtype, self.device)
                mean_rmi, std_rmi = rmi.get_rmi_benchmark_stats_torch(
                    self.config, self.dtype, self.device)

                atten_mask = context_model.get_attention_mask(
                    window_len=processing_window_len,
                    interpolation_window_slice=interpolation_window_slice,
                    device= self.device)
                
                positions = self.positions[..., :processing_window_len, :, :].clone() #[batch, window, :,:]
                rotations = self.rotations[..., :processing_window_len, :, :, :].clone()

                positions, rotations = data_utils.to_start_centered_data(
                    positions, rotations, self.interpolation_window_offset)

                pos_new_normal, rot_new_normal = context_model.evaluate(
                    self.model, positions, rotations, interpolation_window_slice,
                    self.indices, mean, std, atten_mask, past_context_len, False)  #[32, 35, 22, 3]), torch.Size([32, 35, 22, 3, 3]) 
                
                (gpos_batch_loss_normal, gquat_batch_loss_normal,
                npss_batch_loss_normal, npss_batch_weights_normal) = \
                    benchmark.get_rmi_style_batch_loss(
                        positions, rotations, pos_new_normal, rot_new_normal, self.parents,
                        beg_context_len, target_idx, mean_rmi, std_rmi
                    )
                
                #zeroing whats not inside context               
                positions_zeroed = positions.clone()
                positions_zeroed[..., interpolation_window_slice, :, :] = 0
                
                rotations_zeroed = rotations.clone()
                rotations_zeroed[..., interpolation_window_slice, :, :] = 0
                pos_new_zeroed, rot_new_zeroed = context_model.evaluate(
                    self.model, positions, rotations, interpolation_window_slice,
                    self.indices, mean, std, atten_mask, past_context_len, False)  #[32, 35, 22, 3]), torch.Size([32, 35, 22, 3, 3]) 

                (gpos_batch_loss_zeroed, gquat_batch_loss_zeroed,
                npss_batch_loss_zeroed, npss_batch_weights_zeroed) = \
                    benchmark.get_rmi_style_batch_loss(
                        positions, rotations, pos_new_zeroed, rot_new_zeroed, self.parents,
                        beg_context_len, target_idx, mean_rmi, std_rmi
                    )
                    
                self.assertTrue(
                    torch.allclose(
                        torch.as_tensor(gpos_batch_loss_normal),
                        torch.as_tensor(gpos_batch_loss_zeroed),
                        atol=1e-5
                    ),
                    msg=f"T{trans_len} PC{past_context_len} gpos_batch_loss mismatch. {gpos_batch_loss_normal} vs {gpos_batch_loss_zeroed}"
                )

                self.assertTrue(
                    torch.allclose(
                        torch.as_tensor(gquat_batch_loss_normal),
                        torch.as_tensor(gquat_batch_loss_zeroed),
                        atol=1e-5
                    ),
                    msg=f"T{trans_len} PC{past_context_len} gquat_batch_loss mismatch. {gquat_batch_loss_normal} vs {gquat_batch_loss_zeroed}"
                )

                self.assertTrue(
                    torch.allclose(
                        torch.as_tensor(npss_batch_loss_normal),
                        torch.as_tensor(npss_batch_loss_zeroed),
                        atol=1e-5
                    ),
                    msg=f"T{trans_len} PC{past_context_len} npss_batch_loss mismatch. {npss_batch_loss_normal} vs {npss_batch_loss_zeroed}"
                )

                self.assertTrue(
                    torch.allclose(
                        torch.as_tensor(npss_batch_weights_normal),
                        torch.as_tensor(npss_batch_weights_zeroed),
                        atol=1e-5
                    ),
                    msg=f"T{trans_len} PC{past_context_len} npss_batch_weights mismatch. {npss_batch_weights_normal} vs {npss_batch_weights_zeroed}"
                )
                
                self.assertTrue(
                    torch.allclose(pos_new_normal, pos_new_zeroed, atol=1e-5),
                    msg=f"T{trans_len} PC{past_context_len} positions mismatch. {pos_new_normal} vs {pos_new_zeroed}"
                )

                self.assertTrue(
                    torch.allclose(rot_new_normal, rot_new_zeroed, atol=1e-5),
                    msg=f"T{trans_len} PC{past_context_len} rotations mismatch. {rot_new_normal} vs {rot_new_zeroed}"
                )

if __name__ == "__main__":
    unittest.main()