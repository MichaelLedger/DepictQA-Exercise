import argparse
import logging
import os
import pprint
import random
import time

import numpy as np
import torch
import yaml
from easydict import EasyDict
from tqdm import tqdm

from datasets import load_trainset
from model.build_mps import build_agent

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parser_args():
    parser = argparse.ArgumentParser(description="train parameters for DepictQA")
    parser.add_argument("--cfg", type=str, default="config.yaml")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    return args

def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)

def main(args):
    start_time = time.time()
    set_random_seed(args["seed"])
    
    # Setup device
    device = torch.device("mps")
    
    # Setup training parameters
    num_epochs = args.train["epochs"]
    save_dir = args.train["save_dir"]
    log_dir = args.train["log_dir"]
    batch_size = args.batch_size
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Setup logging
    time_str = time.strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
        level=logging.INFO,
        filename=os.path.join(log_dir, f"train_{time_str}.log"),
        filemode="w",
    )

    # Load dataset
    args.batch_size = batch_size  # Override batch size for dataset loading
    dataloader = load_trainset(args)
    
    # Calculate total steps
    total_steps = num_epochs * len(dataloader.dataset) // batch_size
    args.train["total_steps"] = total_steps
    
    # Build model
    agent = build_agent(args, training=True)
    agent.model.to(device)  # Move model to MPS device
    
    # Save training args
    with open(os.path.join(log_dir, "training_args.yaml"), "w") as fw:
        yaml.dump(args, fw)
    logging.info("args: {}".format(pprint.pformat(args)))

    # Training loop
    pbar = tqdm(total=total_steps)
    step = 0
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Train step
            agent.train_model(batch, step=step, pbar=pbar)
            step += 1
            
            # Save checkpoints
            if args.train["save_freq_step"] and step % args.train["save_freq_step"] == 0:
                agent.save_model(save_dir, epoch + 1, step)
                
        # Save epoch checkpoints
        if epoch % args.train["save_freq_epoch"] == 0:
            agent.save_model(save_dir, epoch + 1)

    # Final save
    agent.save_model(save_dir)
    logging.info(f"[!] Training done. Total time: {time.time() - start_time}")

if __name__ == "__main__":
    args = parser_args()
    with open(args.cfg, "r") as f:
        cfg = EasyDict(yaml.safe_load(f))
    args = vars(args)
    cfg.update(args)
    main(cfg)
