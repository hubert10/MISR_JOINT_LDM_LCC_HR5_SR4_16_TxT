import sys

sys.path.append(".")
import torch
import importlib
from typing import Dict
from utils.hparams import hparams, set_hparams
from utils.utils_dataset import read_config
from utils.hparams import hparams
from models.denoiser.unet import Unet
from models.lr_net_branch import LRNet
from models.diffusion.latent_diffusion import LatentDiffusion


def load_weight(weight_path: str) -> Dict[str, torch.Tensor]:
    weight = torch.load(
        weight_path, map_location=torch.device("cpu"), weights_only=False
    )
    if "state_dict" in weight:
        weight = weight["state_dict"]

    pure_weight = {}
    for key, val in weight.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        pure_weight[key] = val

    return pure_weight


class Trainer:
    def __init__(self):
        self.sd_weights = (
            "D:\\kanyamahanga\\Datasets\\CMSRD\\checkpoints\\v2-1_512-ema-pruned.ckpt"
        )
        self.output = "checkpoints/init_weight/init_weight-refsr-o-sgm.pt"

    def load(self):
        dim_mults = hparams["unet_dim_mults"]
        dim_mults = [int(x) for x in dim_mults.split("|")]

        denoise_net = Unet(
            hparams["hidden_size"],
            out_dim=hparams["num_channels_sat"],
            cond_dim=hparams["rrdb_num_feat"],
            dim_mults=dim_mults,
        )

        first_stage_config = {
            "embed_dim": 4,
            "double_z": True,
            "z_channels": 4,
            "resolution": 256,
            "in_channels": 4,
            "out_ch": 4,
            "ch": 128,
            "ch_mult": [1, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
        }
        cond_stage_config = {
            "image_size": 64,
            "in_channels": 8,
            "model_channels": 160,
            "out_channels": 4,
            "num_res_blocks": 2,
            "attention_resolutions": [16, 8],
            "channel_mult": [1, 2, 2, 4],
            "num_head_channels": 32,
        }

        # 2. Cond Encoder with ONLY two encoding block layers
        cond_net = LRNet(
            img_size=hparams["sat_patch_size"],
            block_channels=hparams["block_channels"][:2],
            block_layers=hparams["block_layers"][1:],
            embed_dim=hparams["embed_dim"],
            uper_head_dim=hparams["uper_head_dim"],
            depths=hparams["depths"],
            num_heads=hparams["num_heads"],
            mlp_ratio=hparams["mlp_ratio"],
            num_classes=hparams["num_classes"],
            nbts=hparams["nbts"],
            pool_scales=hparams["pool_scales"],
            spa_temp_att=hparams["spa_temp_att"],
            conv_spa_att=hparams["conv_spa_att"],
            decoder_channels=hparams["decoder_channels"],
            window_size=hparams["window_size"],
            d_model=hparams["d_model"],
            config=hparams,
        )

        model = LatentDiffusion(
            denoise_net=denoise_net,
            cond_net=cond_net,
            first_stage_config=first_stage_config,
            cond_stage_config=cond_stage_config,
            timesteps=hparams["timesteps"],
        )

        sd_weights = load_weight(self.sd_weights)

        scratch_weights = model.state_dict()

        print(
            " ------------------------ scratch_weights.keys() ------------------------------------ :",
            scratch_weights.keys(),
        )

        init_weights = {}
        for weight_name in scratch_weights.keys():
            # find target pretrained weights for this weight
            if weight_name.startswith("control_"):
                suffix = weight_name[len("control_") :]
                target_name = f"latent_diff.diffusion_{suffix}"
                target_model_weights = sd_weights
            else:
                target_name = weight_name
                target_model_weights = sd_weights

            # if target weight exist in pretrained model
            print(f"copy weights: {target_name} -> {weight_name}")
            # if target_name in target_model_weights and ('transformer_blocks' not in target_name):
            if target_name in target_model_weights:
                # get pretrained weight
                target_weight = target_model_weights[target_name]
                target_shape = target_weight.shape
                model_shape = scratch_weights[weight_name].shape
                # print("model_shape:", model_shape)
                # print("target_shape:", target_shape)

                # if pretrained weight has the same shape with model weight, we make a copy
                if model_shape == target_shape:
                    init_weights[weight_name] = target_weight.clone()
                # else we copy pretrained weight with additional channels initialized to zero
                else:
                    print("model_shape:", model_shape)
                    print("target_shape:", target_shape)
                    newly_added_channels = model_shape[1] - target_shape[1]
                    oc, _, h, w = target_shape
                    zero_weight = torch.zeros((oc, newly_added_channels, h, w)).type_as(
                        target_weight
                    )
                    init_weights[weight_name] = torch.cat(
                        (target_weight.clone(), zero_weight), dim=1
                    )
                    print(
                        f"add zero weight to {target_name} in pretrained weights, newly added channels = {newly_added_channels}"
                    )
            else:
                init_weights[weight_name] = scratch_weights[weight_name].clone()
                print(f"These weights are newly added: {weight_name}")

        model.load_state_dict(init_weights, strict=True)
        torch.save(model.state_dict(), self.output)
        print("Done.")


if __name__ == "__main__":
    set_hparams()
    config = read_config(hparams["config_file"])
    pkg = ".".join(hparams["trainer_cls"].split(".")[:-1])
    trainer = Trainer()
    trainer.load()
