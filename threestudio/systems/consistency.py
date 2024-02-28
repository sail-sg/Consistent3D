import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

import numpy as np
import imageio

import random


@threestudio.register("consistency-system")
class consistency(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        refinement: bool = False
        texture: bool = False
        geometry_only: bool = False
        freq: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.cfg.geometry_only:
            render_out = self.renderer(**batch, render_rgb=False)
        else:
            render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        if self.cfg.refinement and not self.cfg.texture:
            # initialize SDF
            # FIXME: what if using other geometry types?
            self.geometry.initialize_shape()

    def training_step(self, batch, batch_idx):
        loss = 0.0
        out = self(batch)
        prompt_utils = self.prompt_processor()

        if not self.cfg.refinement: # instantngp for coarse geometry
            guidance_inp = out["comp_rgb"]
            guidance_out = self.guidance(
                guidance_inp, prompt_utils, **batch, rgb_as_latents=False,
            )
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                loss_orient = (
                    out["weights"].detach()
                    * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                ).sum() / (out["opacity"] > 0).sum()
                self.log("train/loss_orient", loss_orient)
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            if self.C(self.cfg.loss.lambda_normal_smoothness_2d) > 0:
                if "comp_normal" not in out:
                    raise ValueError(
                        "comp_normal is required for 2D normal smoothness loss, no comp_normal is found in the output."
                    )
                normal = out["comp_normal"]
                loss_normal_smoothness_2d = (
                    normal[:, 1:, :, :] - normal[:, :-1, :, :]
                ).square().mean() + (
                    normal[:, :, 1:, :] - normal[:, :, :-1, :]
                ).square().mean()
                self.log("trian/loss_normal_smoothness_2d", loss_normal_smoothness_2d)
                loss += loss_normal_smoothness_2d * self.C(
                    self.cfg.loss.lambda_normal_smoothness_2d
                )
        else: # refinement
            if self.cfg.geometry_only:
                guidance_inp = out["comp_normal"]
            else:
                guidance_inp = out["comp_rgb"]
            guidance_out = self.guidance(
                guidance_inp, prompt_utils, **batch, rgb_as_latents=False,
            )
            if not self.cfg.texture:  # geometry training
                loss_normal_consistency = out["mesh"].normal_consistency()
                self.log("train/loss_normal_consistency", loss_normal_consistency)
                loss += loss_normal_consistency * self.C(
                    self.cfg.loss.lambda_normal_consistency
                )
                if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                    with torch.cuda.amp.autocast(enabled=False):
                        loss_laplacian_smoothness = out["mesh"].laplacian()
                    self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
                    loss += loss_laplacian_smoothness * self.C(
                        self.cfg.loss.lambda_laplacian_smoothness
                    )

        for name, value in guidance_out.items():
            if name == "eval":
                # eval is a dict and cannot be logged
                continue
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx, test_mode=False):
        out = self(batch)
        if test_mode:
            out_tmp = out.copy()
            out = {}
            if "comp_rgb" in out_tmp:
                out["comp_rgb"] = out_tmp["comp_rgb"].detach()
            else:
                out["comp_normal"] = out_tmp["comp_normal"].detach()

        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ]
                if "opacity" in out
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def save_test(self, comp_rgb):
        fn = f"it{self.true_global_step}-test/test.png"
        x = comp_rgb
        x = x.reshape(-1, *x.shape[2:])
        x = x.detach().cpu().numpy()
        imageio.imwrite(self.get_save_path(fn), np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8), compress_level=3) # Low compression for faster saving 

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )