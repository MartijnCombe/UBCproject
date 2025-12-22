import argparse
import torch
import datetime
import json
import yaml
import os
import logging
import numpy as np

from dataset_aqi36 import get_dataloader
from main_model import PriSTI_aqi36
from utils import train, evaluate


# -----------------------------------------------------------------------------------
# FREEZING FUNCTIONS (added)
# -----------------------------------------------------------------------------------

def freeze_early_diffusion_blocks(model, freeze_ratio=0.5):
    """Freeze first N% of NoiseProject layers in the diffusion model."""
    num_layers = len(model.diffmodel.residual_layers)
    freeze_upto = int(num_layers * freeze_ratio)

    print(f"Freezing first {freeze_upto}/{num_layers} diffusion residual blocks")

    for i in range(freeze_upto):
        for p in model.diffmodel.residual_layers[i].parameters():
            p.requires_grad = False


def unfreeze_embeddings(model):
    """Ensure embeddings trainable (sensor/node embeddings)."""
    for p in model.embed_layer.parameters():
        p.requires_grad = True


def unfreeze_output_head(model):
    """Ensure the model’s output projections always train when fine-tuning."""
    for p in model.diffmodel.output_projection1.parameters():
        p.requires_grad = True
    for p in model.diffmodel.output_projection2.parameters():
        p.requires_grad = True


def apply_transfer_freezing(model, freeze_ratio=0.5):
    """Main function to apply all freezing/unfreezing rules."""
    unfreeze_embeddings(model)
    freeze_early_diffusion_blocks(model, freeze_ratio)
    unfreeze_output_head(model)

    # Print stats
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable}/{total} ({100*trainable/total:.2f}%)")


# -----------------------------------------------------------------------------------
# MAIN APP CODE
# -----------------------------------------------------------------------------------

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
    config["diffusion"]["adj_file"] = 'AQI36'
    config["seed"] = SEED

    print(json.dumps(config, indent=4))

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = "./save/pm25_outsample_" + current_time + "/"
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)

    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)

    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        config["train"]["batch_size"],
        device=args.device,
        val_len=args.val_len,
        is_interpolate=config["model"]["use_guide"],
        num_workers=args.num_workers,
        target_strategy=args.targetstrategy,
        mask_sensor=config["model"]["mask_sensor"]
    )

    model = PriSTI_aqi36(config, args.device).to(args.device)


    # --------------------------------------------------------------------------
    # TRAIN OR LOAD MODEL
    # --------------------------------------------------------------------------
    if args.modelfolder == "":  
        # FULL TRAINING (NO FREEZING)
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )

    else:
        # ------------------------------
        # Load pre-trained weights
        # ------------------------------
        print(f"Loading pretrained model from {args.modelfolder}...")
        model.load_state_dict(
            torch.load(f"./save/{args.modelfolder}/model.pth", map_location=args.device)
        )

        # ------------------------------
        # Apply transfer learning freezing
        # ------------------------------
        print("Applying freezing for transfer learning...")
        apply_transfer_freezing(model, freeze_ratio=0.5)

        # Now run fine-tuning
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )

    # --------------------------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PriSTI")
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument("--targetstrategy", type=str, default="hybrid",
                        choices=["hybrid", "random", "historical"])
    parser.add_argument("--val_len", type=float, default=0.1)
    parser.add_argument("--nsample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unconditional", action="store_true")

    args = parser.parse_args()
    print(args)
    main(args)
