import unittest
import torch
import os
import sys
import json
import argparse
import pandas as pd
import torch

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
                        help="dataset name (defaultwqq=benchmark)",
                        default="benchmark")
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
    dataset, data_loader = train_utils.init_bvh_dataset(
        config, args.dataset, device=device, shuffle=False)

    results = []
    past_context = range(0,config["train"]["context_len"]-1)  #range(1, 11)
    transitions = [15,30,45]

    for t in transitions:
        for i in past_context:
            config["train"]["past_context"] = i #????? train???
            # initialize model
            model = ContextTransformer(config["model"]).to(device)
            # load checkpoint
            epoch, iteration = train_utils.load_checkpoint(config, model)

            if args.index is None:
                res = context_model.eval_on_dataset(
                    config,
                    data_loader,
                    model, 
                    trans_len=t, 
                    beg_contex_len=config["train"]["context_len"] - i, 
                    past_context_len=i,
                    debug=args.debug,
                    post_process=args.post_processing,
                    #fixed_pred_window_start=config["train"]["context_len"],
                    save_results=True)

                if args.debug:
                    gpos_loss, gquat_loss, npss_loss, loss_data = res

                    json_path = "{}_{}_{}_ranking.json".format(
                        args.config, args.dataset, t)
                    with open(json_path, "w") as fh:
                        json.dump(loss_data, fh)
                else:
                    gpos_loss, gquat_loss, npss_loss = res

                print(config["name"])
                print("b_ctx: {}, p_ctx: {},trans {}: gpos: {:.4f}, gquat: {:.4f}, npss: {:.4f}{}".format(
                    config["train"]["context_len"] - i, i,t, gpos_loss, gquat_loss, npss_loss,
                    " (w/ post-processing)" if args.post_processing else ""))
                results.append((t, i, gpos_loss, gquat_loss, npss_loss))
    results = pd.DataFrame(results)
    results.columns = ["trans", "past_context", "gpos", "gquat", "npss"]
    print(results)
    results.to_csv("results.csv")