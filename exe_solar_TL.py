import argparse
import logging
import torch
import datetime
import json
import yaml
import os
import numpy as np

from dataset_solar import get_solar_dataloader
from main_model import PriSTI
from utils import train, evaluate


class PriSTI_Solar(PriSTI):
    def __init__(self, config, device, target_dim, seq_len=24):
        super(PriSTI_Solar, self).__init__(target_dim, seq_len, config, device)
        self.config = config

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        coeffs = None
        if self.config['model']['use_guide']:
            coeffs = batch["coeffs"].to(self.device).float() if "coeffs" in batch else None
        cond_mask = batch["cond_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)  # [B, K, L]
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)
        for_pattern_mask = observed_mask

        if self.config['model']['use_guide'] and coeffs is not None:
            coeffs = coeffs.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
        )

def load_pretrained_except_nodes(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_dict = model.state_dict()

    filtered_ckpt = {}
    skipped = []

    for k, v in ckpt.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_ckpt[k] = v
        else:
            skipped.append(k)

    model_dict.update(filtered_ckpt)
    model.load_state_dict(model_dict)

    print(f"Loaded {len(filtered_ckpt)} layers")
    print(f"Skipped {len(skipped)} node-specific layers")
    
    
def freeze_early_diffusion_blocks(model, freeze_ratio=0.5):
    #Freeze first N% of NoiseProject layers in the diffusion model
    num_layers = len(model.diffmodel.residual_layers)
    freeze_upto = int(num_layers * freeze_ratio)

    print(f"Freezing first {freeze_upto}/{num_layers} diffusion residual blocks")

    for i in range(freeze_upto):
        for p in model.diffmodel.residual_layers[i].parameters():
            p.requires_grad = False


def unfreeze_embeddings(model):
    #Ensure embeddings trainable (sensor/node embeddings)
    for p in model.embed_layer.parameters():
        p.requires_grad = True


def unfreeze_output_head(model):
    #Ensure the model’s output projections always train when fine-tuning
    for p in model.diffmodel.output_projection1.parameters():
        p.requires_grad = True
    for p in model.diffmodel.output_projection2.parameters():
        p.requires_grad = True


def apply_transfer_freezing(model, freeze_ratio=0.5):
    #Main function to apply all freezing/unfreezing rules
    unfreeze_embeddings(model)
    freeze_early_diffusion_blocks(model, freeze_ratio)
    unfreeze_output_head(model)

    # Print stats
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable}/{total} ({100*trainable/total:.2f}%)")

def main(args):
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["model"]["is_unconditional"] = args.unconditional
    config["model"]["target_strategy"] = args.targetstrategy
    # adjacency for Solar: compute/load adjacency from lat/lon NPZ
    config["diffusion"]["adj_file"] = 'solar'
    config["seed"] = SEED

    print(json.dumps(config, indent=4))

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = (
        "./save/solar_" + args.missing_pattern + '_' + current_time + "/"
    )

    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # load data (infer target_dim from mean scaler)
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_solar_dataloader(
        npz_path=args.npz_path,
        batch_size=config["train"]["batch_size"],
        device=args.device,
        val_len=args.val_len,
        test_len=args.test_len,
        missing_pattern=args.missing_pattern,
        is_interpolate=config["model"]["use_guide"],
        num_workers=args.num_workers,
        target_strategy=args.targetstrategy,
        eval_length=args.eval_length,
    )

    # infer number of nodes from mean_scaler
    try:
        target_dim = int(mean_scaler.shape[0])
    except Exception:
        target_dim = None

    if target_dim is None:
        raise RuntimeError("Could not infer target dimension from mean_scaler")

    model = PriSTI_Solar(config, args.device, target_dim=target_dim, seq_len=args.eval_length).to(args.device)

    if args.modelfolder == "":
        print("Apply full training, no transfer learning")
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )
        
    else:
        print(f"Loading pretrained model from {args.modelfolder}...")
        load_pretrained_except_nodes(
            model,
            f"./save/{args.modelfolder}/model.pth",
            args.device
        )
        print("Applying freezing for transfer learning...")
        apply_transfer_freezing(model, freeze_ratio=0.5)
        
        # Train the non-frozen parameters
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )

    logging.basicConfig(filename=foldername + '/test_model.log', level=logging.DEBUG)
    logging.info("model_name={}".format(args.modelfolder))
    evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PriSTI - Solar")
    parser.add_argument("--config", type=str, default="traffic.yaml")
    parser.add_argument('--device', default='cuda:0', help='Device for training/eval')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument(
        "--targetstrategy", type=str, default="block", choices=["mix", "random", "block"]
    )
    parser.add_argument("--nsample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--missing_pattern", type=str, default="block")     # block|point
    parser.add_argument('--npz_path', type=str, default='./data/solar/data_&_nodeMetaData.npz')
    parser.add_argument('--eval_length', type=int, default=24)
    parser.add_argument('--val_len', type=float, default=0.1)
    parser.add_argument('--test_len', type=float, default=0.2)

    args = parser.parse_args()
    print(args)

    main(args)
