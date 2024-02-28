from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
import random

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *

import math
import numpy as np


@threestudio.register("stable-diffusion-guidance")
class StableDiffusionGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 100.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        loss_type: str = "sds"

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        min_sigma_percent: float = 0.02
        max_sigma_percent: float = 0.98
        max_step_percent_annealed: float = 0.5
        anneal_start_step: Optional[int] = None
        cur_steps: Optional[Any] = field(default_factory=lambda: [0,700,20,5000])
        recon_std_rescale: float = 0.0
        share_noise: bool = True
        sde_flow: bool = False
        perturb_indices: Optional[str] = None
        perturb_factor: int = 200
        end_gap: int = 100

        use_sjc: bool = False
        var_red: bool = True
        weighting_strategy: str = "sds"

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        view_dependent_prompting: bool = True

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        cleanup()

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        if self.cfg.token_merging:
            import tomesd

            tomesd.apply_patch(self.unet, **self.cfg.token_merging_params)

        if self.cfg.use_sjc:
            # score jacobian chaining use DDPM
            self.scheduler = DDPMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
                beta_start=0.00085,
                beta_end=0.0120,
                beta_schedule="scaled_linear",
            )
        else:
            self.scheduler = DDIMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
            )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value
        # self.set_min_max_steps(min_step_percent=self.cfg.min_step_percent, max_step_percent=self.cfg.max_step_percent)  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )
        # if self.cfg.use_sjc:
        #     # score jacobian chaining need mu
        #     self.us: Float[Tensor, "..."] = torch.sqrt((1 - self.alphas) / self.alphas)

        # always compute sigma 
        self.us: Float[Tensor, "..."] = torch.sqrt((1 - self.alphas) / self.alphas)

        self.grad_clip_val: Optional[float] = None

        self.sigma_min = int(self.num_train_timesteps * self.cfg.min_sigma_percent)
        self.sigma_max = int(self.num_train_timesteps * self.cfg.max_sigma_percent)
        threestudio.info(f"sigma_max: {self.sigma_max}, sigma_min: {self.sigma_min}")

        threestudio.info(f"Loaded Stable Diffusion!")

        self.noise = torch.randn(4, 64, 64, device=self.device) # share noise
        
    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        batch_size = elevation.shape[0]

        if self.cfg.share_noise:
            noise = self.noise.repeat(batch_size, 1, 1, 1)
        else:
            noise = torch.randn_like(latents)

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                # noise = torch.randn_like(latents)  # TODO: use torch generator
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        grad = w * (noise_pred - noise)

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
            "grad_latents": -grad,
        }

        return grad, guidance_eval_utils
    
    def compute_grad_consistency(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        t_next: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        batch_size = elevation.shape[0]
        if self.cfg.share_noise:
            noise = self.noise.repeat(batch_size, 1, 1, 1)
        else:
            noise = torch.randn_like(latents)
        sigma_t = ((1 - self.alphas[t]) / self.alphas[t]).sqrt().view(-1, 1, 1, 1)
        sigma_t_next = ((1 - self.alphas[t_next]) / self.alphas[t_next]).sqrt().view(-1, 1, 1, 1)

        if prompt_utils.use_perp_neg:
            # TODO: not used
            raise NotImplementedError("Perp neg not implemented for consistency loss")
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )

            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                y = latents
                zs = y + sigma_t * noise
                scaled_zs = zs / torch.sqrt(1 + sigma_t**2)

                # pred noise
                latent_model_input = torch.cat([scaled_zs] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )

                # perform guidance (high scale from paper!)
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_text + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            with torch.no_grad():
                # add noise
                _, pred_epsilon = self.get_rescaled_sample(zs, t, noise_pred, noise_pred_text)
                zs_next = zs + (sigma_t_next - sigma_t) * pred_epsilon
                scaled_zs_next = zs_next / torch.sqrt(1 + sigma_t_next**2) 

                # pred noise
                latent_model_input_next = torch.cat([scaled_zs_next] * 2, dim=0)
                noise_pred_next = self.forward_unet(
                    latent_model_input_next,
                    torch.cat([t_next] * 2),
                    encoder_hidden_states=text_embeddings,
                )

                # perform guidance (high scale from paper!)
                noise_pred_text_next, noise_pred_uncond_next = noise_pred_next.chunk(2)
                noise_pred_next = noise_pred_text_next + self.guidance_scale * (noise_pred_text_next - noise_pred_uncond_next)

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            grad_weight = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            grad_weight = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            grad_weight = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == 'mix':
            if self.cur_training_ratio >= 0.5:
                grad_weight = (1 - self.alphas[t]).view(-1, 1, 1, 1)
            else:
                grad_weight = 1
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )
        
        _, pred_epsilon_next = self.get_rescaled_sample(scaled_zs_next, t_next, noise_pred_next, noise_pred_text_next)
        pred_next = (sigma_t - sigma_t_next) * pred_epsilon + sigma_t_next * pred_epsilon_next
        pred_next /= sigma_t

        grad_latents = (sigma_t / (sigma_t_next + 1e-9)) * (pred_epsilon - noise_pred) + (noise_pred_next - pred_epsilon)
        grad_latents = grad_weight * grad_latents
        grad_latents = torch.nan_to_num(grad_latents)
        if self.grad_clip_val is not None:
            grad_latents.clamp_(-self.grad_clip_val, self.grad_clip_val)
        target = latents + grad_latents
        loss = 0.5 * F.mse_loss(latents, target.detach(), reduction="sum") / batch_size

        grad_visual = latents + grad_latents / grad_weight
        
        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "t_next": t_next,
            "latents": latents,
            "latents_noisy": scaled_zs,
            "latents_noisy_next": scaled_zs_next,
            "noise_pred": noise_pred,
            "noise_pred_next": noise_pred_next,
            "pred_epsilon": pred_epsilon,
            "pred_epsilon_next": pred_epsilon_next,
            "grad_latents": grad_visual,
        }

        if isinstance(grad_weight, int):
            return_grad_weight = grad_weight
        else:
            return_grad_weight = grad_weight.mean()
        return grad_latents, loss, return_grad_weight, (1 - self.alphas[t]).mean(), guidance_eval_utils

    def get_original_sample(
        self,
        zs: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        noise_pred: Float[Tensor, "B 4 64 64"],
        use_clipped_model_output: bool = False,
    ):
        sigma_t = ((1 - self.alphas[t]) / self.alphas[t]).sqrt().view(-1, 1, 1, 1)
        pred_original_sample = zs - sigma_t * noise_pred
        if use_clipped_model_output:
            pred_original_sample.clamp_(-1, 1) # constrain to [-1, 1]
        pred_epsilon = (zs - pred_original_sample) / sigma_t
        return pred_original_sample, pred_epsilon
    
    def get_rescaled_sample(self, zs, t, noise_pred, noise_pred_text):
        sigma_t = ((1 - self.alphas[t]) / self.alphas[t]).sqrt().view(-1, 1, 1, 1)
        latents_recon = self.get_original_sample(zs, t, noise_pred, False)[0]
        # clip or rescale x0
        if self.cfg.recon_std_rescale > 0:
            latents_recon_nocfg = self.get_original_sample(zs, t, noise_pred_text, False)[0]
            factor = (latents_recon_nocfg.std([1,2,3],keepdim=True) + 1e-8) / (latents_recon.std([1,2,3],keepdim=True) + 1e-8)
            latents_recon_adjust = latents_recon.clone() * factor
            latents_recon = self.cfg.recon_std_rescale * latents_recon_adjust + (1-self.cfg.recon_std_rescale) * latents_recon
        pred_epsilon = (zs - latents_recon) / sigma_t
        return latents_recon, pred_epsilon

    def compute_grad_sjc(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        batch_size = elevation.shape[0]

        sigma = self.us[t]
        sigma = sigma.view(-1, 1, 1, 1)

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                noise = torch.randn_like(latents)
                y = latents
                zs = y + sigma * noise
                scaled_zs = zs / torch.sqrt(1 + sigma**2)
                # pred noise
                latent_model_input = torch.cat([scaled_zs] * 4, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)  # TODO: use torch generator
                y = latents

                zs = y + sigma * noise
                scaled_zs = zs / torch.sqrt(1 + sigma**2)

                # pred noise
                latent_model_input = torch.cat([scaled_zs] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )

                # perform guidance (high scale from paper!)
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

        Ds = zs - sigma * noise_pred

        if self.cfg.var_red:
            grad = -(Ds - y) / sigma
        else:
            grad = -(Ds - zs) / sigma

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": scaled_zs,
            "noise_pred": noise_pred,
        }

        return grad, guidance_eval_utils

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if "consistency" in self.cfg.loss_type:
            t_next = torch.as_tensor([self.cur_steps] * batch_size, dtype=torch.long, device=self.device)
            perturb = torch.randint(self.cfg.end_gap, self.cfg.end_gap + self.cfg.perturb_factor, [batch_size], dtype=torch.long, device=self.device)
            t_next.clamp_(self.sigma_min, self.sigma_max - self.cfg.end_gap)
            t = t_next + perturb
            t.clamp_(self.sigma_min, self.sigma_max)

        if "consistency" in self.cfg.loss_type:
            grad, loss_sds, grad_weight, sds_weight, guidance_eval_utils = self.compute_grad_consistency(
                latents, t, t_next, prompt_utils, elevation, azimuth, camera_distances
            )
        else:
            if self.cfg.use_sjc:
                grad, guidance_eval_utils = self.compute_grad_sjc(
                    latents, t, prompt_utils, elevation, azimuth, camera_distances
                )
            else:
                grad, guidance_eval_utils = self.compute_grad_sds(
                    latents, t, prompt_utils, elevation, azimuth, camera_distances
                )

            grad = torch.nan_to_num(grad)
            # clip grad for stable training?
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

            # loss = SpecifyGradient.apply(latents, grad)
            # SpecifyGradient is not straghtforward, use a reparameterization trick instead
            target = (latents - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

            grad_weight = sds_weight = 1.


        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
            "t": t.float().mean(),
            "t_next": t_next.float().mean(),
            "grad_weight": grad_weight,
            "sds_weight": sds_weight,
        }

        if guidance_eval:
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_noise_pred(
        self,
        latents_noisy,
        t,
        text_embeddings,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        batch_size = latents_noisy.shape[0]

        if use_perp_neg:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 4).to(self.device),
                encoder_hidden_states=text_embeddings,
            )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            # noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
            #     e_pos + accum_grad
            # )
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            )
            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            # noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            #     noise_pred_text - noise_pred_uncond
            # )
            noise_pred = noise_pred_text + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return noise_pred
    
    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        latents_noisy,
        noise_pred,
        use_perp_neg=False,
        neg_guidance_weights=None,
        latents=None,
        t_next=None,
        latents_noisy_next=None,
        noise_pred_next=None,
        pred_epsilon=None,
        pred_epsilon_next=None,
        grad_latents=None,
    ):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[ :bs].unsqueeze(-1)  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        if t_next is not None:
            large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_next[ :bs].unsqueeze(-1)  # sized [bs,50] > [bs,1]
            idxs_next = torch.min(large_enough_idxs, dim=1)[1]
            t_next = self.scheduler.timesteps_gpu[idxs_next]

        # visualize gradient
        if grad_latents is not None:
            grad_visual = self.decode_latents(grad_latents[:bs]).permute(0, 2, 3, 1)
        else:
            grad_visual = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1) # dummy to avoid error

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)
        if latents_noisy_next is not None:
            imgs_noisy_next = self.decode_latents(latents_noisy_next[:bs]).permute(0, 2, 3, 1)
        else:
            imgs_noisy_next = imgs_noisy

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1] # delete eta for ODE step, eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

        if pred_epsilon is not None:
            rescaled_latents_1step = []
            rescaled_pred_1orig = []
            for b in range(bs):
                step_output = self.scheduler.step(
                    pred_epsilon[b : b + 1], t[b], latents_noisy[b : b + 1] # delete eta for ODE step, eta=1
                )
                rescaled_latents_1step.append(step_output["prev_sample"])
                rescaled_pred_1orig.append(step_output["pred_original_sample"])
            rescaled_latents_1step = torch.cat(rescaled_latents_1step)
            rescaled_pred_1orig = torch.cat(rescaled_pred_1orig)
            rescaled_imgs_1step = self.decode_latents(rescaled_latents_1step).permute(0, 2, 3, 1)
            rescaled_imgs_1orig = self.decode_latents(rescaled_pred_1orig).permute(0, 2, 3, 1)
        else:
            # dummy to avoid error
            rescaled_imgs_1step = imgs_1step
            rescaled_imgs_1orig = imgs_1orig


        # get prev latent at t_next
        if t_next is not None:
            latents_1step_next = []
            pred_1orig_next = []
            for b in range(bs):
                step_output = self.scheduler.step(
                    noise_pred_next[b : b + 1], t_next[b], latents_noisy_next[b : b + 1] # delete eta for ODE step, eta=1
                )
                latents_1step_next.append(step_output["prev_sample"])
                pred_1orig_next.append(step_output["pred_original_sample"])
            latents_1step_next = torch.cat(latents_1step_next)
            pred_1orig_next = torch.cat(pred_1orig_next)
            imgs_1step_next = self.decode_latents(latents_1step_next).permute(0, 2, 3, 1)
            imgs_1orig_next = self.decode_latents(pred_1orig_next).permute(0, 2, 3, 1)
            
            if pred_epsilon_next is not None:
                rescaled_latents_1step_next = []
                rescaled_pred_1orig_next = []
                for b in range(bs):
                    step_output = self.scheduler.step(
                        pred_epsilon_next[b : b + 1], t_next[b], latents_noisy_next[b : b + 1] # delete eta for ODE step, eta=1
                    )
                    rescaled_latents_1step_next.append(step_output["prev_sample"])
                    rescaled_pred_1orig_next.append(step_output["pred_original_sample"])
                rescaled_latents_1step_next = torch.cat(rescaled_latents_1step_next)
                rescaled_pred_1orig_next = torch.cat(rescaled_pred_1orig_next)
                rescaled_imgs_1step_next = self.decode_latents(rescaled_latents_1step_next).permute(0, 2, 3, 1)
                rescaled_imgs_1orig_next = self.decode_latents(rescaled_pred_1orig_next).permute(0, 2, 3, 1)
            else:
                # dummy to avoid error
                rescaled_imgs_1step_next = imgs_1step_next
                rescaled_imgs_1orig_next = imgs_1orig_next
        else:
            # dummy to avoid error
            imgs_1step_next = imgs_1step
            imgs_1orig_next = imgs_1orig
            rescaled_imgs_1step_next = imgs_1step_next
            rescaled_imgs_1orig_next = imgs_1orig_next

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1]
            text_emb = (
                text_embeddings[
                    [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
                ]
                if use_perp_neg
                else text_embeddings[[b, b + len(idxs)], ...]
            )
            neg_guid = neg_guidance_weights[b : b + 1] if use_perp_neg else None
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                # pred noise
                noise_pred = self.get_noise_pred(
                    latents, t, text_emb, use_perp_neg, neg_guid
                )
                # get prev latent
                latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
                    "prev_sample"
                ]
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)

        if t_next is not None:
            latents_final_next = []
            for b, i in enumerate(idxs_next):
                latents = latents_1step_next[b : b + 1]
                text_emb = (
                    text_embeddings[
                        [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
                    ]
                    if use_perp_neg
                    else text_embeddings[[b, b + len(idxs)], ...]
                )
                neg_guid = neg_guidance_weights[b : b + 1] if use_perp_neg else None
                for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                    # pred noise
                    noise_pred = self.get_noise_pred(
                        latents, t, text_emb, use_perp_neg, neg_guid
                    )
                    # get prev latent
                    latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
                        "prev_sample"
                    ]
                latents_final_next.append(latents)

            latents_final_next = torch.cat(latents_final_next)
            imgs_final_next = self.decode_latents(latents_final_next).permute(0, 2, 3, 1)
        else:
            # dummy to avoid error
            imgs_final_next = imgs_final

        return {
            "bs": bs,
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_noisy_next": imgs_noisy_next,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "rescaled_imgs_1orig": rescaled_imgs_1orig,
            "imgs_1step_next": imgs_1step_next,
            "imgs_1orig_next": imgs_1orig_next,
            "rescaled_imgs_1orig_next": rescaled_imgs_1orig_next,
            "imgs_final": imgs_final,
            "imgs_final_next": imgs_final_next,
            "grad_visual": grad_visual,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )

        if self.cfg.perturb_indices is not None:
            self.cur_steps = int(self.anneal_t(self.cfg.cur_steps, epoch, global_step))
        else:
            self.cur_steps = int(C(self.cfg.cur_steps, epoch, global_step))
        end_ratio = self.cfg.cur_steps[3]
        self.cur_training_ratio = global_step / end_ratio
        self.guidance_scale = C(self.cfg.guidance_scale, epoch, global_step)
    
    def anneal_t(self, value: Any, epoch: int, global_step: int) -> float:
        start_step, start_value, end_value, end_step = value
        if isinstance(end_step, int):
            current_step = global_step
        else:
            current_step = epoch
        if current_step <= start_step:
            return start_value
        if current_step >= end_step:
            return end_value
        value = start_value + (end_value - start_value) * math.sqrt((current_step - start_step) / (end_step - start_step))
        threestudio.debug(f"iter: {current_step}, value: {value}")
        return value
    
