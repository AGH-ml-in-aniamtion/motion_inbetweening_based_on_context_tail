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
from motion_inbetween.train import utils as train_utils

import matplotlib.pyplot as plt

    
class TestMyExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Config, device, and model
        torch.use_deterministic_algorithms(True)
        lafan1_context_model = "lafan1_context_model"
        self.config = load_config_by_name(lafan1_context_model)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = ContextTransformer(self.config["model"]).to(self.device)

        # Shared constants
        self.context_len = 10
        self.interpolation_window_offset = self.context_len
        self.dtype = torch.float32
        

        dataset, data_loader, sequence_filenames = train_utils.init_bvh_dataset_w_file_info(
            self.config, "benchmark", device=self.device)

        self.dataset = dataset
        self.data_loader = data_loader
        self.sequence_filenames = sequence_filenames

        data = next(iter(data_loader))
        (self.positions, self.rotations, self.global_positions, self.global_rotations,
         self.foot_contact, self.parents, _) = data
        
        
        self.mean, self.std = context_model.get_train_stats_torch(self.config, self.dtype, self.device)
        self.mean_rmi, self.std_rmi = rmi.get_rmi_benchmark_stats_torch(
            self.config, self.dtype, self.device)

        self.indices = self.config["indices"]

    def test_attention_mask(self):
        window_len = 65
        context_len = 10
        for interpolation_length in [5, 15, 30, 45]:
            target_idx = context_len + interpolation_length
            
            get_attention_mask_slice = context_model.get_attention_mask_slice(
                window_len=window_len,
                pred_context_slice=slice(0, context_len),
                past_context_slice=slice(target_idx, target_idx+1),
                device=self.device
            )
            #plt.imshow(get_attention_mask_slice[0])
            #plt.show()
            get_attention_mask = context_model.get_attention_mask(
                window_len=window_len,
                context_len=context_len,
                target_idx=target_idx,
                device=self.device
            )
            #plt.imshow(get_attention_mask[0])
            #plt.show()
            self.assertEqual(
                (get_attention_mask_slice != get_attention_mask).sum().item(), 0,
                msg=f"Attention masks do not match for interpolation length {interpolation_length}"
            )
            
    def test_data_masks(self):
        window_len = 65
        d_mask = self.model.d_mask
        constrained_slices = self.model.constrained_slices
        device = self.device
        dtype = self.dtype
        
        beg_context_len = 10
        trans_len = 15
        target_idx = beg_context_len + trans_len
        
        baseline_d_mask = context_model.get_data_mask(
            window_len=window_len,
            d_mask=d_mask,
            constrained_slices=constrained_slices,
            context_len=beg_context_len,
            target_idx=target_idx,
            device=device,
            dtype=dtype
        )
        
        my_d_mask = context_model.get_data_mask_slice(
            window_len=window_len,
            d_mask=d_mask,
            constrained_slices=constrained_slices,
            beg_context_slice=slice(0, beg_context_len),
            past_context_slice=slice(target_idx, target_idx+1),
            device=device,
            dtype=dtype
        )
        self.assertEqual(
                (baseline_d_mask != my_d_mask).sum().item(), 0,
                msg=f"Data masks do not match"
            )

    def test_evaluation_functions(self):
        model = self.model
        positions = self.positions
        rotations = self.rotations
        indices = self.indices
        device = self.device
        post_process = False
        
        beg_context_len = self.context_len
        beg_context_slice = slice(0, beg_context_len)
        trans_len = 15
        target_idx = beg_context_len + trans_len 
        past_context_len = 1
        past_context_slice = slice(target_idx, target_idx + past_context_len)
        seq_slice = slice(self.context_len, target_idx)
        
        
        window_len = beg_context_len + trans_len + past_context_len + 2
                
        atten_mask = context_model.get_attention_mask(
            window_len=window_len,
            context_len=beg_context_len,
            target_idx=target_idx,
            device=device
        )
        atten_mask_slice = context_model.get_attention_mask_slice(
            window_len=window_len,
            pred_context_slice=beg_context_slice,
            past_context_slice=past_context_slice,
            device=device
        )
        self.assertEqual(
                (atten_mask != atten_mask_slice).sum().item(), 0,
                msg=f"Attention masks do not match"
            )

        positions = positions[..., :window_len, :, :]
        rotations = rotations[..., :window_len, :, :, :]
        positions, rotations = data_utils.to_start_centered_data(
            positions, rotations, beg_context_len)

        pos_new_baseline, rot_new_baseline = context_model.evaluate(
            model=model, 
            positions=positions,
            rotations=rotations, 
            seq_slice=seq_slice,
            indices=indices,
            mean=self.mean,
            std=self.std,
            atten_mask=atten_mask,
            post_process=post_process)

        pos_new, rot_new = context_model.evaluate_my_experiment(
            model=model,
            positions=positions,
            rotations=rotations,
            seq_slice=seq_slice,
            indices=indices,
            mean=self.mean,
            std=self.std,
            atten_mask=atten_mask,
            beg_context_slice=beg_context_slice,
            past_context_slice=past_context_slice,
            post_process=post_process)  #[32, 35, 22, 3]), torch.Size([32, 35, 22, 3, 3]) 
        
        self.assertTrue( (pos_new_baseline - pos_new).abs().sum() == 0 )
        self.assertTrue( (rot_new_baseline - rot_new).abs().sum() == 0 )

    def test_baseline_eval_against_my_experiment(self):
        #eval on dataset
        config = self.config
        model = self.model
        indices = config["indices"]
        
        beg_context_len = config["train"]["context_len"]
        trans_len = self.trans_len
        device = self.device
        post_process = False
        
        target_idx = beg_context_len + trans_len
        seq_slice = slice(beg_context_len, target_idx)
        window_len_baseline = beg_context_len + trans_len + 2

        # attention mask
        atten_mask_baseline = context_model.get_attention_mask(
            window_len_baseline, beg_context_len, target_idx, device)

        (positions, rotations, parents) = (self.positions, self.rotations, self.parents)
        parents = parents[0]

        positions_baseline = positions[..., :window_len_baseline, :, :]
        rotations_baseline = rotations[..., :window_len_baseline, :, :, :]

        positions_baseline, rotations_baseline = data_utils.to_start_centered_data(
            positions_baseline, rotations_baseline, beg_context_len)

        pos_new_baseline, rot_new_baseline = context_model.evaluate(
            model, positions_baseline, rotations_baseline, seq_slice,
            indices, self.mean, self.std, atten_mask_baseline, post_process)

        (gpos_batch_loss, gquat_batch_loss,
        npss_batch_loss, npss_batch_weights) = \
            benchmark.get_rmi_style_batch_loss(
                positions_baseline, rotations_baseline, pos_new_baseline, rot_new_baseline, parents,
                beg_context_len, target_idx, self.mean_rmi, self.std_rmi
        )
        gpos_loss_baseline = np.mean(gpos_batch_loss)
        gquat_loss_baseline = np.mean(gquat_batch_loss)
        npss_weights = npss_batch_weights / np.sum(npss_batch_weights)      # (batch, dim)
        npss_loss_baseline = np.sum(npss_batch_loss * npss_weights, axis=-1)   # (batch, )

##############################################
        #my experiment on same data############################################################################################
####################################################

        context_len = config["train"]["context_len"] + 1  # target frame included
        interpolation_window_offset = context_len - 1
        past_context_len = 1
    
        beg_context_len = context_len - past_context_len # 10
        
        target_idx = interpolation_window_offset + trans_len #  25
        interpolation_window_slice = slice(interpolation_window_offset, target_idx)  # 10:25
        processing_window_len = interpolation_window_offset + trans_len + interpolation_window_offset

        beg_context_slice = slice(interpolation_window_offset-beg_context_len, interpolation_window_offset)
        past_context_slice = slice(target_idx, target_idx+past_context_len)

        atten_mask = context_model.get_attention_mask_slice(
            window_len=processing_window_len,
            pred_context_slice=beg_context_slice,
            past_context_slice=past_context_slice,
            device=device)

        (positions, rotations, foot_contact, parents) = (self.positions, self.rotations, self.foot_contact, self.parents)
        parents = parents[0] 
        positions = positions[..., :processing_window_len, :, :] #[batch, window, :,:]
        rotations = rotations[..., :processing_window_len, :, :, :]
        foot_contact = foot_contact[..., :processing_window_len, :]

        positions, rotations = data_utils.to_start_centered_data(
            positions, rotations, interpolation_window_offset)

        pos_new, rot_new = context_model.evaluate_my_experiment(
            model=model,
            positions=positions,
            rotations=rotations,
            seq_slice=interpolation_window_slice,
            indices=indices,
            mean=self.mean,
            std=self.std,
            atten_mask=atten_mask,
            beg_context_slice=beg_context_slice,
            past_context_slice=past_context_slice,
            #past_context_len=past_context_len,
            post_process=post_process)  #[32, 35, 22, 3]), torch.Size([32, 35, 22, 3, 3]) 

        (gpos_batch_loss, gquat_batch_loss,
        npss_batch_loss, npss_batch_weights) = \
            benchmark.get_rmi_style_batch_loss_slice(
                positions=positions, rotations=rotations, pos_new=pos_new, rot_new=rot_new, parents=parents,
                interpolation_window_slice=interpolation_window_slice, mean_rmi=self.mean_rmi, std_rmi=self.std_rmi
            )
            
    def test_model_masks_evaluation(self):
            
        for trans_len in [15,30,45]:
            for past_context_len in range(1,10):
                
                beg_context_len = self.context_len - past_context_len # 10
                
                target_idx = self.interpolation_window_offset + trans_len #  25
                interpolation_window_slice = slice(self.interpolation_window_offset, target_idx)  # 10:25
                processing_window_len = self.interpolation_window_offset + trans_len + self.interpolation_window_offset

                beg_context_slice = slice(self.interpolation_window_offset- beg_context_len, self.interpolation_window_offset)
                past_context_slice = slice(target_idx, target_idx+past_context_len)

                mean, std = context_model.get_train_stats_torch(self.config, self.dtype, self.device)
                mean_rmi, std_rmi = rmi.get_rmi_benchmark_stats_torch(
                    self.config, self.dtype, self.device)

                atten_mask = context_model.get_attention_mask_slice(
                    window_len=processing_window_len,
                    pred_context_slice=beg_context_slice,
                    past_context_slice=past_context_slice,
                    device= self.device)
                
                positions = self.positions[..., :processing_window_len, :, :].clone() #[batch, window, :,:]
                rotations = self.rotations[..., :processing_window_len, :, :, :].clone()

                positions, rotations = data_utils.to_start_centered_data(
                    positions, rotations, self.interpolation_window_offset)

                pos_new_normal, rot_new_normal = context_model.evaluate_my_experiment(
                    model=self.model, 
                    positions=positions, 
                    rotations=rotations, 
                    seq_slice=interpolation_window_slice,
                    indices=self.indices, 
                    mean=mean, 
                    std=std, 
                    atten_mask=atten_mask, 
                    beg_context_slice=beg_context_slice,
                    past_context_slice=past_context_slice, 
                    post_process=False) 
                
                (gpos_batch_loss_normal, gquat_batch_loss_normal,
                npss_batch_loss_normal, npss_batch_weights_normal) = \
                    benchmark.get_rmi_style_batch_loss_slice(
                        positions=positions, rotations=rotations, pos_new=pos_new_normal, rot_new=rot_new_normal, parents=self.parents[0],
                        interpolation_window_slice=interpolation_window_slice, mean_rmi=mean_rmi, std_rmi=std_rmi
                    )
                
                #zeroing whats not inside context               
                positions_zeroed = positions.clone()
                positions_zeroed[..., interpolation_window_slice, 0, :] = 0 #only modifying root joint
                
                rotations_zeroed = rotations.clone()
                rotations_zeroed[:, interpolation_window_slice, ...] = 0
                pos_new_zeroed, rot_new_zeroed = context_model.evaluate_my_experiment(
                    model=self.model, 
                    positions=positions_zeroed, 
                    rotations=rotations_zeroed, 
                    seq_slice=interpolation_window_slice,
                    indices=self.indices, 
                    mean=mean, 
                    std=std, 
                    atten_mask=atten_mask, 
                    beg_context_slice=beg_context_slice,
                    past_context_slice=past_context_slice, 
                    post_process=False)  #[32, 35, 22, 3]), torch.Size([32, 35, 22, 3, 3]) 

                (gpos_batch_loss_zeroed, gquat_batch_loss_zeroed,
                npss_batch_loss_zeroed, npss_batch_weights_zeroed) = \
                    benchmark.get_rmi_style_batch_loss_slice(
                        positions=positions, rotations=rotations, pos_new=pos_new_zeroed, rot_new=rot_new_zeroed, parents=self.parents[0],
                        interpolation_window_slice=interpolation_window_slice, mean_rmi=mean_rmi, std_rmi=std_rmi
                    )
                # positions_delta = torch.abs(pos_new_normal - pos_new_zeroed)
                # rotations_delta = torch.abs(rot_new_normal - rot_new_zeroed)
                # print(positions_delta.shape)
                # plt.imshow((positions_delta)[0])
                # plt.ylabel("frame")
                # plt.xlabel("feature(joint)")
                # plt.show()

                #assert that positions for every joint except root is equal in both in every frame
                for f in range(positions.shape[1]):
                    # Non-root joints should be identical across frames
                    self.assertTrue(
                        torch.allclose(positions[0, f, 1:, :], positions[0, 0, 1:, :], atol=1e-6),
                        msg=f"Non-root joint position prediction mismatch in baseline at frame {f}"
                    )
                    self.assertTrue(
                        torch.allclose(pos_new_normal[0, f, 1:, :], pos_new_normal[0, 0, 1:, :], atol=1e-6),
                        msg=f"Non-root joint position prediction mismatch in new normal at frame {f}"
                    )
                    self.assertTrue(
                        torch.allclose(pos_new_zeroed[0, f, 1:, :], pos_new_zeroed[0, 0, 1:, :], atol=1e-6),
                        msg=f"Non-root joint position prediction mismatch in new zeroed at frame {f}"
                    )

                # Root joint of normal vs zeroed
                for f in range(positions.shape[1]):
                    self.assertTrue(
                        torch.allclose(pos_new_normal[0, f, 0, :], pos_new_zeroed[0, f, 0, :], atol=1e-6),
                        msg=f"Root joint position prediction mismatch at frame {f}"
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
                


if __name__ == "__main__":
    unittest.main()