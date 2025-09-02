import datetime
import logging
import os
from collections import OrderedDict

import torch
from torch.utils.tensorboard import SummaryWriter


class MPSAgent:
    def __init__(self, model, args):
        super(MPSAgent, self).__init__()
        self.args = args
        self.model = model
        self.device = torch.device("mps")
        
        # Setup tensorboard writer
        self.writer = SummaryWriter(args.train["log_dir"])
        
        # Load pretrained weights if they exist
        delta_path = args.model["delta_path"]
        if os.path.exists(delta_path):
            delta_ckpt = torch.load(delta_path, map_location=self.device)
            self.model.load_state_dict(delta_ckpt, strict=False)
            logging.info(f"[!] Load pretrained delta ckpt from {delta_path}")
        else:
            logging.info(f"[!] Train from scratch, delta_path ({delta_path}) not exist")

        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.train.optimizer.lr,
            betas=args.train.optimizer.betas,
            eps=args.train.optimizer.eps,
            weight_decay=args.train.optimizer.weight_decay
        )
        
        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=args.train.scheduler.warmup_max_lr,
            total_steps=args.train.scheduler.total_steps,
            pct_start=args.train.scheduler.warmup_steps / args.train.scheduler.total_steps,
            div_factor=args.train.scheduler.warmup_max_lr / args.train.scheduler.warmup_min_lr,
            final_div_factor=args.train.scheduler.warmup_min_lr / args.train.scheduler.warmup_max_lr
        )

    def train_model(self, batch, step=0, pbar=None):
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        self.optimizer.zero_grad()
        loss, acc = self.model(batch)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        # Update progress bar
        pbar.set_description(
            f"[!] loss: {round(loss.item(), 4)}; acc: {round(acc * 100, 2)}"
        )
        pbar.update(1)
        
        # Log metrics
        if step % self.args.train["log_step"] == 0:
            elapsed = pbar.format_dict["elapsed"]
            rate = pbar.format_dict["rate"]
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            self.writer.add_scalar("train/loss", loss.item(), step)
            self.writer.add_scalar("train/acc", acc * 100, step)
            logging.info(
                f"[!] progress: {round(pbar.n / pbar.total, 4)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; acc: {round(acc * 100, 2)}"
            )
        
        acc *= 100
        return acc

    def save_model(self, save_dir, epoch=None, step=None):
        # Get trainable parameters
        trainable_params = [
            k for (k, v) in self.model.named_parameters() if v.requires_grad
        ]
        
        # Get state dict
        state_dict = self.model.state_dict()
        ckpt = OrderedDict((k, state_dict[k]) for k in trainable_params)
        
        # Save checkpoint
        if epoch is None:
            torch.save(ckpt, os.path.join(save_dir, "ckpt.pt"))
        elif step is None:
            torch.save(ckpt, os.path.join(save_dir, f"ckpt_epoch{epoch}.pt"))
        else:
            torch.save(
                ckpt, os.path.join(save_dir, f"ckpt_epoch{epoch}_step{step}.pt")
            )
            
        # Save tokenizer and config
        self.model.tokenizer.save_pretrained(save_dir)
        self.model.llm.config.save_pretrained(save_dir)
        logging.info(f"[!] Save model in {save_dir}")
