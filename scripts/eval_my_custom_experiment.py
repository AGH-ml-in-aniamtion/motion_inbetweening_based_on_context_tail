import unittest
import torch
import os
import sys
import json
import argparse
import pandas as pd
import torch
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
package_root = os.path.join(project_root, "packages")
sys.path.append(package_root)


from motion_inbetween.model import transformer
from motion_inbetween import benchmark, visualization
from motion_inbetween.model import ContextTransformer
from motion_inbetween.config import load_config_by_name
from motion_inbetween.train import rmi
from motion_inbetween.train import context_model
from motion_inbetween.train import utils as train_utils
from motion_inbetween.data import utils_torch as data_utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate context model. "
                                     "Post-processing is applied by default.")
    parser.add_argument("config", help="config name")
    parser.add_argument("-s", "--dataset",
                        help="dataset name (default=benchmark_past_ctx)",
                        default="benchmark_past_ctx")
    parser.add_argument("-i", "--index", type=int, help="data index")
    parser.add_argument("-t", "--trans", type=int, default=30,
                        help="transition length (default=30)")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="debug mode")
    parser.add_argument("-p", "--post_processing", action="store_true",
                        default=False, help="apply post-processing")
    args = parser.parse_args()

    config = load_config_by_name(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    indices = config["indices"]
    
    dataset, data_loader, sequence_filenames = train_utils.init_bvh_dataset_w_file_info(
        config, args.dataset, device=device)
    dtype = data_loader.dataset.dtype
    mean, std = context_model.get_train_stats_torch(config, dtype, device)
    mean_rmi, std_rmi = rmi.get_rmi_benchmark_stats_torch(
        config, dtype, device)

    results = []
    model = ContextTransformer(config["model"]).to(device)
    epoch, iteration = train_utils.load_checkpoint(config, model)
    
    debug = args.debug
    post_process = args.post_processing
    save_results = False

    context_len = config["train"]["context_len"] + 1  # target frame included - 11
    past_contexts = range(1, 11)  # minimum 1 (target_frame, maximum 10)
    transitions = [5, 15, 30, 45]


    #i can get per animation result if i compare indexes of dataset with appended batched results
    for trans_len in transitions:
        for past_context_len in past_contexts:
            
            window_offset = past_context_len-1# offsetting whole model input window so it matches with different past context lengths
            beg_context_len = context_len - past_context_len
            
            assert beg_context_len+past_context_len == context_len
            assert window_offset+beg_context_len+1 == context_len
 

            window_len = beg_context_len + trans_len + past_context_len + 1 # 1 extra frame for model predicting how movement continues
            processing_window_slice = slice(window_offset, window_offset+window_len)


            data_indexes = []
            gpos_loss = []
            gquat_loss = []
            npss_loss = []
            npss_weights = []
            animations = []
            
            for i, data in enumerate(data_loader):
                (positions, rotations, _, _,
                    foot_contact, parents, data_idx) = data
                parents = parents[0] 
                
                if positions.shape[1] < window_len + window_offset:
                    print("Skipping sequence ", i, " of length ", positions.shape[1], " because it is shorter than required window length ", window_len + window_offset)
                    continue
                
                
                positions = positions[..., processing_window_slice, :, :] #[batch, window, :,:]
                rotations = rotations[..., processing_window_slice, :, :, :]
                foot_contact = foot_contact[..., processing_window_slice, :]
                
                beg_context_slice = slice(0, beg_context_len)
                target_idx = beg_context_len + trans_len
                past_context_slice = slice(target_idx, target_idx+past_context_len)
                interpolation_window_slice = slice(beg_context_slice.stop, past_context_slice.start)
                
                atten_mask = context_model.get_attention_mask_slice(
                    window_len=window_len,
                    pred_context_slice=beg_context_slice,
                    past_context_slice=past_context_slice,
                    device=device)
                
                positions, rotations = data_utils.to_start_centered_data(
                    positions, rotations, interpolation_window_slice.start)
                
                assert atten_mask.shape[1] == window_len 
                assert positions.shape[1] == window_len and rotations.shape[1] == window_len 
                
                pos_new, rot_new = context_model.evaluate_my_experiment(
                    model=model,
                    positions=positions,
                    rotations=rotations,
                    seq_slice=interpolation_window_slice,
                    indices=indices,
                    mean=mean,
                    std=std,
                    atten_mask=atten_mask,
                    beg_context_slice=beg_context_slice,
                    past_context_slice=past_context_slice,
                    #past_context_len=past_context_len,
                    post_process=post_process)  #[32, 35, 22, 3]), torch.Size([32, 35, 22, 3, 3]) 

                (gpos_batch_loss, gquat_batch_loss,
                npss_batch_loss, npss_batch_weights) = \
                    benchmark.get_rmi_style_batch_loss_slice(
                        positions=positions, rotations=rotations, pos_new=pos_new, rot_new=rot_new, parents=parents,
                        interpolation_window_slice=interpolation_window_slice, mean_rmi=mean_rmi, std_rmi=std_rmi
                    )

                gpos_loss.append(gpos_batch_loss)
                gquat_loss.append(gquat_batch_loss)
                npss_loss.append(npss_batch_loss)
                npss_weights.append(npss_batch_weights)
                data_indexes.extend(data_idx.tolist())
                
                #if not exist make dir 
                if save_results:
                    dirname = "./beg{}_past{}_fixed{}".format(
                        beg_context_len, past_context_len, 0)
                    
                    if os.path.exists(dirname) is False:
                        os.makedirs(dirname)
                        
                    json_path_gt = "output_{}_gt.json".format(i)  
                    json_path_gt = os.path.join(dirname, json_path_gt)
                    visualization.save_data_to_json(
                        json_path_gt, positions[0], rotations[0],
                        foot_contact, parents)

                    json_path = "output_{}.json".format(i)
                    json_path = os.path.join(dirname, json_path)
                    visualization.save_data_to_json(
                        json_path, pos_new[0], rot_new[0],
                        foot_contact, parents)

            gpos_loss = np.concatenate(gpos_loss, axis=0) 
            gquat_loss = np.concatenate(gquat_loss, axis=0)
            npss_loss = np.concatenate(npss_loss, axis=0)           # (batch, dim)
            npss_weights = np.concatenate(npss_weights, axis=0)

            #print(gpos_loss.shape, gquat_loss.shape, npss_loss.shape, npss_weights.shape, len(data_indexes))
            # npss_weights = npss_weights / np.sum(npss_weights)      # (batch, dim)
            # npss_loss = np.sum(npss_loss * npss_weights, axis=-1)   # (batch, )

            for i, data_idx in enumerate(data_indexes):
                results.append((trans_len, past_context_len, gpos_loss[i], gquat_loss[i], npss_loss[i], npss_weights[i], data_idx , sequence_filenames[data_idx]))
    results = pd.DataFrame(results)
    results.columns = ["trans", "past_context", "gpos", "gquat", "npss", "npss_weights", "data_index", "sequence_filename"]
    print(results.head())
    results.to_pickle("results3.pickle")
    