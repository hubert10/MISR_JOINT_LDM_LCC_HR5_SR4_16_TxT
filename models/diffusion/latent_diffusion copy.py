import torch
import pathlib
from typing import Union
import requests
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import Literal
from trainer import Trainer
import torch.nn.functional as F
from contextlib import contextmanager
from functools import partial
from utils.hparams import hparams
from einops import rearrange
from utils.utils import linear_transform_6b
from utils.utils import assert_tensor_validity
from utils.utils import revert_padding
import torch.utils.checkpoint as checkpoint
from models.diffusion.utils import DDIMSampler
from skimage.exposure import match_histograms
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
from models.autoencoder.autoencoder import AutoencoderKL, DiagonalGaussianDistribution
from models.denoiser.unet import UNetModel
from models.diffusion.utils import (
    LitEma,
    count_params,
    default,
    disabled_train,
    exists,
    extract_into_tensor,
    make_beta_schedule,
    make_convolutional_sample,
)


from losses.srdiff_loss import (
    pixel_wise_closest_sr_sits_aer_loss,
    grad_pixel_wise_closest_sr_sits_aer_loss,
    temp_gradient_magnitude_consistency_loss,
    gray_value_consistency_loss,
)
from losses.focal_smooth import FocalLossWithSmoothing


__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}


class DiffusionWrapper(nn.Module):
    """
    A wrapper around a UNetModel that supports different types of conditioning.

    Args:
        diff_model_config (dict): A dictionary of configuration options for the UNetModel.
        conditioning_key (str, optional): The type of conditioning to use
            (None, 'concat', 'crossattn', 'hybrid', or 'adm'). Defaults to None.

    Raises:
        AssertionError: If the conditioning key is not one of the supported values.

    Example:
        >>> diff_model_config = {'in_channels': 3, 'out_channels': 3, 'num_filters': 32}
        >>> wrapper = DiffusionWrapper(diff_model_config, conditioning_key='concat')
        >>> x = torch.randn(1, 3, 256, 256)
        >>> t = torch.randn(1)
        >>> c_concat = [torch.randn(1, 32, 256, 256)]
        >>> y = wrapper(x, t, c_concat=c_concat)
    """

    def __init__(self, diff_model_config: dict, conditioning_key: Optional[str] = None):
        super().__init__()
        self.diffusion_model = UNetModel(**diff_model_config)
        self.conditioning_key = conditioning_key

        ckey_options = [None, "concat", "crossattn", "hybrid", "adm"]
        assert (
            self.conditioning_key in ckey_options
        ), f"Unsupported conditioning key: {self.conditioning_key}"

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c_concat: Optional[List[torch.Tensor]] = None,
        c_crossattn: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Apply the diffusion model to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            t (torch.Tensor): The diffusion time.
            c_concat (List[torch.Tensor], optional): A list of tensors to concatenate with the input tensor.
                Used when conditioning_key is 'concat'. Defaults to None.
            c_crossattn (List[torch.Tensor], optional): A list of tensors to use for cross-attention.
                Used when conditioning_key is 'crossattn', 'hybrid', or 'adm'. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.

        Raises:
            NotImplementedError: If the conditioning key is not one of the supported values.
        """

        xc = torch.cat([x] + [x], dim=1)

        # xc = torch.cat([x] + c_concat, dim=1)
        # out = self.diffusion_model(xc, t)
        out = self.diffusion_model(xc, c_concat, t)

        print("out:", out.shape)
        return out


class DDPM(nn.Module):
    """This class implements the classic DDPM (Diffusion Models) with Gaussian diffusion
    in image space.

    Args:
        unet_config (dict): A dictionary of configuration options for the UNetModel.
        timesteps (int): The number of diffusion timesteps to use.
        beta_schedule (str): The type of beta schedule to use (linear, cosine, or fixed).
        use_ema (bool): Whether to use exponential moving averages (EMAs) of the model weights during training.
        first_stage_key (str): The key to use for the first stage of the model (either "image" or "noise").
        linear_start (float): The starting value for the linear beta schedule.
        linear_end (float): The ending value for the linear beta schedule.
        cosine_s (float): The scaling factor for the cosine beta schedule.
        given_betas (list): A list of beta values to use for the fixed beta schedule.
        v_posterior (float): The weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta.
        conditioning_key (str): The type of conditioning to use (None, 'concat', 'crossattn', 'hybrid', or 'adm').
        parameterization (str): The type of parameterization to use for the diffusion process (either "eps" or "x0").
        use_positional_encodings (bool): Whether to use positional encodings for the input.

    Methods:
        register_schedule: Registers the schedule for the betas and alphas.
        get_input: Gets the input from the DataLoader and rearranges it.
        decode_first_stage: Decodes the first stage of the model.
        ema_scope: Switches to EMA weights during training.

    Attributes:
        parameterization (str): The type of parameterization used for the diffusion process.
        cond_stage_model (None): The conditioning stage model (not used in this implementation).
        first_stage_key (str): The key used for the first stage of the model.
        use_positional_encodings (bool): Whether positional encodings are used for the input.
        model (DiffusionWrapper): The diffusion model.
        use_ema (bool): Whether EMAs of the model weights are used during training.
        model_ema (LitEma): The EMA of the model weights.
        v_posterior (float): The weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta.

    Example:
        >>> unet_config = {
            'in_channels': 3,
            'model_channels': 160,
            'num_res_blocks': 2,
            'attention_resolutions': [16, 8],
            'channel_mult': [1, 2, 2, 4],
            'num_head_channels': 32
        }
        >>> model = DDPM(
                unet_config, timesteps=1000, beta_schedule='linear',
                use_ema=True, first_stage_key='image'
            )
    """

    def __init__(
        self,
        denoise_net,
        cond_net,
        timesteps: int = 1000,
        loss_type: str = "l2",
        beta_schedule: str = "linear",
        use_ema: bool = True,
        first_stage_key: str = "image",
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        cosine_s: float = 8e-3,
        given_betas: Optional[List[float]] = None,
        original_elbo_weight: float = 0.0,
        v_posterior: float = 0.0,
        l_simple_weight: float = 1.0,
        conditioning_key: Optional[str] = None,
        parameterization: str = "eps",
        use_positional_encodings: bool = False,
        learn_logvar: bool = False,
        logvar_init: float = 0.0,
    ) -> None:
        super().__init__()
        assert parameterization in [
            "eps",
            "x0",
        ], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization

        print(
            f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode"
        )
        self.cond_net = cond_net
        self.cond_stage_model = None
        self.first_stage_key = first_stage_key
        self.use_positional_encodings = use_positional_encodings
        self.denoise_net = denoise_net
        self.loss_type = loss_type
        self.timesteps = timesteps
        self.learn_logvar = learn_logvar

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        self.logvar = torch.full(fill_value=logvar_init, size=(self.timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

        count_params(self.denoise_net, verbose=True)

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.denoise_net)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.v_posterior = v_posterior

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

    def register_schedule(
        self,
        given_betas: Optional[List[float]] = None,
        beta_schedule: str = "linear",
        timesteps: int = 1000,
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        cosine_s: float = 8e-3,
    ) -> None:
        """
        Registers the schedule for the betas and alphas.

        Args:
            given_betas (list, optional): A list of beta values to use for the fixed beta schedule.
                Defaults to None.
            beta_schedule (str, optional): The type of beta schedule to use (linear, cosine, or fixed).
                Defaults to "linear".
            timesteps (int, optional): The number of diffusion timesteps to use. Defaults to 1000.
            linear_start (float, optional): The starting value for the linear beta schedule. Defaults to 1e-4.
            linear_end (float, optional): The ending value for the linear beta schedule. Defaults to 2e-2.
            cosine_s (float, optional): The scaling factor for the cosine beta schedule. Defaults to 8e-3.
        """
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule,
                timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (
                2
                * self.posterior_variance
                * to_torch(alphas)
                * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x0":
            lvlb_weights = (
                0.5
                * np.sqrt(torch.Tensor(alphas_cumprod))
                / (2.0 * 1 - torch.Tensor(alphas_cumprod))
            )
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def get_input(self, batch: Dict[str, torch.Tensor], k: str) -> torch.Tensor:
        """
        Gets the input from the DataLoader and rearranges it.

        Args:
            batch (Dict[str, torch.Tensor]): The batch of data from the DataLoader.
            k (str): The key for the input tensor in the batch.

        Returns:
            torch.Tensor: The input tensor, rearranged and converted to float.
        """

        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]

        x = x.to(memory_format=torch.contiguous_format).float()

        return x

    @contextmanager
    def ema_scope(self, context: Optional[str] = None) -> Generator[None, None, None]:
        """
        A context manager that switches to EMA weights during training.

        Args:
            context (Optional[str]): A string to print when switching to and from EMA weights.

        Yields:
            None
        """
        if self.use_ema:
            self.model_ema.store(self.denoise_net.parameters())
            self.model_ema.copy_to(self.denoise_net)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.denoise_net.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * noise
        )

    def predict_start_from_z_and_v(self, x_t, t, v):
        # self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            * x_t
        )

    def decode_first_stage(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes the first stage of the model.

        Args:
            z (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The decoded output tensor.
        """

        z = 1.0 / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(
                    z, ks, stride, uf=uf
                )

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view(
                    (z.shape[0], -1, ks[0], ks[1], z.shape[-1])
                )  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                output_list = [
                    self.first_stage_model.decode(z[:, :, :, :, i])
                    for i in range(z.shape[-1])
                ]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded

            else:
                return self.first_stage_model.decode(z)

        else:
            return self.first_stage_model.decode(z)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, cond, x, img_lr, closest_idx, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_outputs = []
        sr_outputs = []

        for tps in range(cond[0].shape[1]):
            model_output = self.apply_model(x_noisy, t, [feat[:, tps] for feat in cond])

            # recover clean latent - assuming parameterization == "eps"
            x0_pred = self.predict_start_from_noise(x_noisy, t, model_output)

            # generate SR image for classification
            x_sr = self.decode_first_stage_with_grad(x0_pred)

            print("model_output:", model_output.shape)
            print("x_sr:", x_sr.shape)

            model_outputs.append(model_output)
            sr_outputs.append(x_sr)

            # TODO, make this work for a time series!
            # and the model_output should be used for
            # loss computation

        sr_outputs = torch.stack(sr_outputs, 1)  # torch.Size([4, 6, 64, 10, 10])
        model_outputs = torch.stack(model_outputs, 1)  # torch.Size([4, 6, 64, 10, 10])

        prefix = "train" if self.training else "val"

        if self.parameterization == "x0":  # predicted image
            target = x_start
        elif self.parameterization == "eps":  # predicted noise
            target = noise  # torch.Size([2, 4, 16, 16])
        else:
            raise NotImplementedError()

        print("target:", target.shape)
        print("model_outputs:", model_outputs.shape)

        noise_pred = self.closest_lr_sits_aer(model_outputs, closest_idx)

        if self.loss_type == "l1":
            loss = (target - noise_pred).abs().mean()
        elif self.loss_type == "l2":
            loss = F.mse_loss(target, noise_pred)
        else:
            raise NotImplementedError()

        aux_loss = (
            hparams["px_loss_weight"]
            * pixel_wise_closest_sr_sits_aer_loss(sr_outputs, x, closest_idx)
            + hparams["grad_px_loss_weight"]
            * grad_pixel_wise_closest_sr_sits_aer_loss(sr_outputs, x, closest_idx)
            + hparams["temp_grad_mag_loss_weight"]
            * temp_gradient_magnitude_consistency_loss(sr_outputs)
            + hparams["gray_value_px_loss_weight"]
            * gray_value_consistency_loss(sr_outputs, img_lr)
        )
        final_loss = hparams["main_loss_weight"] * loss + aux_loss

        return final_loss, sr_outputs

    def q_sample(
        self, x_start: torch.Tensor, t: int, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Samples from the posterior distribution at the given timestep.

        Args:
            x_start (torch.Tensor): The starting tensor.
            t (int): The timestep.
            noise (Optional[torch.Tensor]): The noise tensor.

        Returns:
            torch.Tensor: The sampled tensor.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        print("noise:", noise.shape)
        print("x_start:", x_start.shape)

        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )


class LatentDiffusion(DDPM):
    """
    LatentDiffusion is a class that extends the DDPM class and implements a diffusion
    model with a latent variable. The model consists of two stages: a first stage that
    encodes the input tensor into a latent tensor, and a second stage that decodes the
    latent tensor into the output tensor. The model also has a conditional stage that
    takes a conditioning tensor as input and produces a learned conditioning tensor
    that is used to condition the first and second stages of the model. The class
    provides methods for encoding and decoding tensors, computing the output tensor
    and loss, and sampling from the distribution at a given latent tensor and timestep.
    The class also provides methods for registering and applying schedules, and for
    getting and setting the scale factor and conditioning key.

    Methods:
        register_schedule(self, schedule: Schedule) -> None: Registers the given schedule
            with the model.
        make_cond_schedule(self, schedule: Schedule) -> Schedule: Returns a new schedule
            with the given schedule applied to the conditional stage of the model.
        encode_first_stage(self, x: torch.Tensor, t: int) -> torch.Tensor: Encodes the given
            input tensor with the first stage of the model for the given timestep.
        get_first_stage_encoding(self, x: torch.Tensor, t: int) -> torch.Tensor: Returns the
            encoding of the given input tensor with the first stage of the model for the
            given timestep.
        get_learned_conditioning(self, x: torch.Tensor, t: int, y: Optional[torch.Tensor] = None) -> torch.Tensor:
            Returns the learned conditioning tensor for the given input
            tensor, timestep, and conditioning tensor.
        get_input(self, x: torch.Tensor, t: int, y: Optional[torch.Tensor] = None) -> torch.Tensor:
            Returns the input tensor for the given input tensor, timestep, and
            conditioning tensor.
        compute(self, x: torch.Tensor, t: int, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
            Computes the output tensor and loss for the given input tensor,
            timestep, and conditioning tensor.
        apply_model(self, x: torch.Tensor, t: int, y: Optional[torch.Tensor] = None) -> torch.Tensor: Applies
            the model to the given input tensor, timestep, and conditioning tensor.
        get_fold_unfold(self, ks: int, stride: int, vqf: int) -> Tuple[Callable, Callable]: Returns the fold
            and unfold functions for the given kernel size, stride, and vector quantization factor.
        forward(self, x: torch.Tensor, t: int, y: Optional[torch.Tensor] = None) -> torch.Tensor: Computes the
            output tensor for the given input tensor, timestep, and conditioning tensor.
        q_sample(self, z: torch.Tensor, t: int, eps: Optional[torch.Tensor] = None) -> torch.Tensor: Samples
            from the distribution at the given latent tensor and timestep.
    """

    def __init__(
        self,
        first_stage_config: Dict[str, Any],
        cond_stage_config: Union[str, Dict[str, Any]],
        num_timesteps_cond: Optional[int] = None,
        cond_stage_key: str = "image",
        cond_stage_trainable: bool = False,
        concat_mode: bool = True,
        cond_stage_forward: Optional[Callable] = None,
        conditioning_key: Optional[str] = None,
        scale_factor: float = 1.0,
        scale_by_std: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initializes the LatentDiffusion model.

        Args:
            first_stage_config (Dict[str, Any]): The configuration for the first stage of the model.
            cond_stage_config (Union[str, Dict[str, Any]]): The configuration for the conditional stage of the model.
            num_timesteps_cond (Optional[int]): The number of timesteps for the conditional stage of the model.
            cond_stage_key (str): The key for the conditional stage of the model.
            cond_stage_trainable (bool): Whether the conditional stage of the model is trainable.
            concat_mode (bool): Whether to use concatenation or cross-attention for the conditioning.
            cond_stage_forward (Optional[Callable]): A function to apply to the output of the conditional stage of the model.
            conditioning_key (Optional[str]): The key for the conditioning.
            scale_factor (float): The scale factor for the input tensor.
            scale_by_std (bool): Whether to scale the input tensor by its standard deviation.
            *args (Any): Additional arguments.
            **kwargs (Any): Additional keyword arguments.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs["timesteps"]
        self.linear_transform = linear_transform_6b
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        if cond_stage_config == "__is_unconditional__":
            conditioning_key = None

        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))

        self.cond_stage_forward = cond_stage_forward

        # Set Fusion parameters (SIMON)
        # TODO: We only have SISR parameters
        self.sr_type = "SISR"

        # Setup the AutoencoderKL model
        embed_dim = first_stage_config[
            "embed_dim"
        ]  # extract embedded dim from first stage config
        self.first_stage_model = AutoencoderKL(first_stage_config, embed_dim=embed_dim)
        self.first_stage_model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

        # Setup the Unet model
        self.cond_stage_model = torch.nn.Identity()  # Unet
        self.cond_stage_model.eval()
        self.cond_stage_model.train = disabled_train
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False

    def register_schedule(
        self,
        given_betas: Optional[Union[float, torch.Tensor]] = None,
        beta_schedule: str = "linear",
        timesteps: int = 1000,
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        cosine_s: float = 8e-3,
    ) -> None:
        """
        Registers the given schedule with the model.

        Args:
            given_betas (Optional[Union[float, torch.Tensor]]): The betas for the schedule.
            beta_schedule (str): The type of beta schedule to use.
            timesteps (int): The number of timesteps for the schedule.
            linear_start (float): The start value for the linear schedule.
            linear_end (float): The end value for the linear schedule.
            cosine_s (float): The scale factor for the cosine schedule.
        """
        super().register_schedule(
            given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s
        )

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def make_cond_schedule(self) -> None:
        """
        Shortens the schedule for the conditional stage of the model.
        """
        self.cond_ids = torch.full(
            size=(self.num_timesteps,),
            fill_value=self.num_timesteps - 1,
            dtype=torch.long,
        )
        ids = torch.round(
            torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)
        ).long()
        self.cond_ids[: self.num_timesteps_cond] = ids

    # 2023-04-08
    def decode_first_stage_with_grad(
        self, z, predict_cids=False, force_not_quantize=False
    ):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, "b h w c -> b c h w").contiguous()

        z = 1.0 / self.scale_factor * z
        return self.first_stage_model.decode(z)

    def encode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the given input tensor with the first stage of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The encoded output tensor.
        """
        return self.first_stage_model.encode(x)

    def get_first_stage_encoding(
        self, encoder_posterior: Union[DiagonalGaussianDistribution, torch.Tensor]
    ) -> torch.Tensor:
        """
        Returns the encoding of the given input tensor with the first stage of the
        model for the given timestep.

        Args:
            encoder_posterior (Union[DiagonalGaussianDistribution, torch.Tensor]): The
                encoder posterior.

        Returns:
            torch.Tensor: The encoding of the input tensor.
        """
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        return self.scale_factor * z

    def get_learned_conditioning(self, c: torch.Tensor) -> torch.Tensor:
        """
        Returns the learned conditioning tensor for the given input tensor.

        Args:
            c (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The learned conditioning tensor.
        """
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, "encode") and callable(
                self.cond_stage_model.encode
            ):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
                # cond stage model is identity
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_input(
        self,
        batch: torch.Tensor,
        k: int,
        return_first_stage_outputs: bool = False,
        force_c_encode: bool = False,
        cond_key: Optional[str] = None,
        return_original_cond: bool = False,
        bs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns the input tensor for the given batch and timestep.

        z is the latent representation of the HR image
        This z becomes x_start in p_losses

        Args:
            batch (torch.Tensor): The input batch tensor.
            k (int): The timestep.
            return_first_stage_outputs (bool): Whether to return the outputs of the first stage of the model.
            force_c_encode (bool): Whether to force encoding of the conditioning tensor.
            cond_key (Optional[str]): The key for the conditioning tensor.
            return_original_cond (bool): Whether to return the original conditioning tensor.
            bs (Optional[int]): The batch size.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: The input tensor, the outputs of the
            first stage of the model (if `return_first_stage_outputs` is `True`), and the encoded conditioning tensor
            (if `force_c_encode` is `True` and `cond_key` is not `None`).
        """

        # k = first_stage_key on this SR example
        x = super().get_input(batch, k)  # line 333

        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)

        # perform always for HR and for HR only of SISR
        if self.sr_type == "SISR" or k == "image":
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()

        if self.model.conditioning_key is not None:
            # self.model.conditioning_key = "image" in SR example

            if cond_key is None:
                cond_key = self.cond_stage_key

            if cond_key != self.first_stage_key:
                if cond_key in ["caption", "coordinates_bbox"]:
                    xc = batch[cond_key]
                elif cond_key == "class_label":
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    # import pudb; pudb.set_trace()
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]

            # BUG if use_positional_encodings is True
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, "pos_x": pos_x, "pos_y": pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {"pos_x": pos_x, "pos_y": pos_y}
        out = [z, c]

        if return_first_stage_outputs:
            # Important: For training loss computation, the decoded image is NOT used.
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)

        return out

    def compute(
        self, example: torch.Tensor, custom_steps: int = 200, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Performs inference on the given example tensor.

        Args:
            example (torch.Tensor): The example tensor.
            custom_steps (int): The number of steps to perform.
            temperature (float): The temperature to use.

        Returns:
            torch.Tensor: The output tensor.
        """
        guider = None
        ckwargs = None
        ddim_use_x0_pred = False
        temperature = temperature
        eta = 1.0
        custom_shape = None

        if hasattr(self, "split_input_params"):
            delattr(self, "split_input_params")

        logs = make_convolutional_sample(
            example,
            self,
            custom_steps=custom_steps,
            eta=eta,
            quantize_x0=False,
            custom_shape=custom_shape,
            temperature=temperature,
            noise_dropout=0.0,
            corrector=guider,
            corrector_kwargs=ckwargs,
            x_T=None,
            ddim_use_x0_pred=ddim_use_x0_pred,
        )

        return logs["sample"]

    def apply_model(
        self,
        x_noisy: torch.Tensor,
        t: int,
        cond: Optional[torch.Tensor] = None,
        return_ids: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Applies the model to the given noisy (latent) input tensor.

        Args:
            x_noisy (torch.Tensor): The noisy input tensor.
            t (int): The timestep.
            cond (Optional[torch.Tensor]): The conditioning tensor.
            return_ids (bool): Whether to return the IDs of the diffusion process.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: The output tensor, and optionally the IDs of the
            diffusion process.
            returns a latent, which is then used for loss computation
            So the loss is computed directly on the latent space, not the decoded image.
        """

        # diffusion prediction

        x_recon = self.denoise_net(x_noisy, t, cond)

        if isinstance(x_recon, tuple) and not return_ids:
            print("x_recon[0]", x_recon[0].shape)
            return x_recon[0]
        else:
            print("x_recon", x_recon.shape)
            return x_recon

    def get_fold_unfold(
        self, x: torch.Tensor, kernel_size: int, stride: int, uf: int = 1, df: int = 1
    ) -> Tuple[nn.Conv2d, nn.ConvTranspose2d]:
        """
        Returns the fold and unfold convolutional layers for the given input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            kernel_size (int): The kernel size.
            stride (int): The stride.
            uf (int): The unfold factor.
            df (int): The fold factor.

        Returns:
            Tuple[nn.Conv2d, nn.ConvTranspose2d]: The fold and unfold convolutional layers.
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(
                kernel_size=kernel_size, dilation=1, padding=0, stride=stride
            )
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(
                kernel_size[0], kernel_size[1], Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(
                kernel_size=kernel_size, dilation=1, padding=0, stride=stride
            )
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(
                kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                dilation=1,
                padding=0,
                stride=(stride[0] * uf, stride[1] * uf),
            )
            fold = torch.nn.Fold(
                output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2
            )

            weighting = self.get_weighting(
                kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(
                1, 1, h * uf, w * uf
            )  # normalizes the overlap
            weighting = weighting.view(
                (1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx)
            )

        elif df > 1 and uf == 1:
            fold_params = dict(
                kernel_size=kernel_size, dilation=1, padding=0, stride=stride
            )
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(
                kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                dilation=1,
                padding=0,
                stride=(stride[0] // df, stride[1] // df),
            )
            fold = torch.nn.Fold(
                output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2
            )

            weighting = self.get_weighting(
                kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(
                1, 1, h // df, w // df
            )  # normalizes the overlap
            weighting = weighting.view(
                (1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx)
            )

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    def apply_cond_encoder(self, img_lr, dates):
        self.cond_net.eval()
        # Prevents PyTorch from tracking gradients, reducing memory
        # usage and speeding up inference.
        with torch.no_grad():
            cond_net_out, _, _, cond = self.cond_net(img_lr, dates)
        return cond

    def closest_lr_sits_aer(self, img_lr, closest_indices):
        B, T, C, H, W = img_lr.shape
        closest_indices = closest_indices.to(torch.long)

        # Ensure closest_indices is a tensor
        if not torch.is_tensor(closest_indices):
            closest_indices = torch.tensor(closest_indices, device=img_lr.device)

        # Gather closest satellite images
        closest_sat_image = img_lr[
            torch.arange(B), closest_indices
        ]  # shape (B, C, H, W)
        return closest_sat_image

    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        dates: torch.Tensor,
        closest_idx: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.
            c (torch.Tensor): The conditioning tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The output tensor.
        """

        print("input x diffusion model:", x.shape)
        print("input img_lr diffusion model:", img_lr.shape)

        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        print("latent z:", z.shape)

        # @TODO: What happens when the conditioning model should be retrained?
        self.cond_net.eval()  #
        # Prevents PyTorch from tracking gradients, reducing memory
        # usage and speeding up inference.
        with torch.no_grad():
            cond_net_out, _, _, cond = self.cond_net(img_lr, dates)

            print("---------------Conditioning Temp Feats-------------------")
            print("cond 0:", cond[0].shape)
            print("cond 1:", cond[1].shape)
            print("cond 2:", cond[2].shape)
            print()

        t = torch.randint(
            0, self.num_timesteps, (z.shape[0],), device=self.device
        ).long()

        # if self.model.conditioning_key is not None:
        #     assert c is not None
        #     if self.cond_stage_trainable:  # This is FALSE in our case
        #         print("cond_stage_trainable:", self.cond_stage_trainable)
        #         c = self.get_learned_conditioning(c)
        #     if self.shorten_cond_schedule:  # TODO: drop this option # TRUE in our case
        #         print("shorten_cond_schedule:", self.shorten_cond_schedule)
        #         c = self.cond_ids[t].to(self.device)
        #         print("c:", c.shape)
        loss, model_outputs = self.p_losses(
            z, cond, x, img_lr, closest_idx, t, *args, **kwargs
        )
        losses = {"sr": loss}
        return losses, model_outputs

    def _tensor_encode(self, X: torch.Tensor):
        # set copy to model
        self._X = X.clone()
        # normalize image
        X_enc = self.linear_transform(X, stage="norm")
        # encode LR images
        self.encode_conditioning = True
        # try to upsample->encode conditioning
        # X_int = torch.nn.functional.interpolate(X, size=(X.shape[-1]*4,X.shape[-1]*4), mode='bilinear', align_corners=False)
        X_int = X
        # encode conditioning
        X_enc = self.first_stage_model.encode(X_int).sample()
        # move to same device as the model
        X_enc = X_enc.to(self.device)
        return X_enc

    def _tensor_decode(self, X_enc: torch.Tensor, spe_cor: bool = True):
        # Decode
        X_dec = self.decode_first_stage(X_enc)
        X_dec = self.linear_transform(X_dec, stage="denorm")
        # Apply spectral correction
        if spe_cor:
            for i in range(X_dec.shape[1]):
                X_dec[:, i] = self.hq_histogram_matching(X_dec[:, i], self._X[:, i])
        # If the value is negative, set it to 0
        X_dec[X_dec < 0] = 0
        return X_dec

    def _prepare_model(
        self,
        X: torch.Tensor,
        eta: float = 1.0,
        custom_steps: int = 100,
        verbose: bool = False,
    ):
        # Create the DDIM sampler
        ddim = DDIMSampler(self)

        # make schedule to compute alphas and sigmas
        ddim.make_schedule(ddim_num_steps=custom_steps, ddim_eta=eta, verbose=verbose)

        # Create the HR latent image
        latent = torch.randn(X.shape, device=X.device)

        # Create the vector with the timesteps
        timesteps = ddim.ddim_timesteps
        time_range = np.flip(timesteps)

        return ddim, latent, time_range

    def _attribution_methods(
        self,
        X: torch.Tensor,
        grads: torch.Tensor,
        attribution_method: Literal[
            "grad_x_input", "max_grad", "mean_grad", "min_grad"
        ],
    ):
        if attribution_method == "grad_x_input":
            return torch.norm(grads * X, dim=(0, 1))
        elif attribution_method == "max_grad":
            return grads.abs().max(dim=0).max(dim=0)
        elif attribution_method == "mean_grad":
            return grads.abs().mean(dim=0).mean(dim=0)
        elif attribution_method == "min_grad":
            return grads.abs().min(dim=0).min(dim=0)
        else:
            raise ValueError(
                "The attribution method must be one of: grad_x_input, max_grad, mean_grad, min_grad"
            )

    def hq_histogram_matching(
        self, image1: torch.Tensor, image2: torch.Tensor
    ) -> torch.Tensor:
        """Lazy implementation of histogram matching

        Args:
            image1 (torch.Tensor): The low-resolution image (C, H, W).
            image2 (torch.Tensor): The super-resolved image (C, H, W).

        Returns:
            torch.Tensor: The super-resolved image with the histogram of
                the target image.
        """

        # Go to numpy
        np_image1 = image1.detach().cpu().numpy()
        np_image2 = image2.detach().cpu().numpy()

        if np_image1.ndim == 3:
            np_image1_hat = match_histograms(np_image1, np_image2, channel_axis=0)
        elif np_image1.ndim == 2:
            np_image1_hat = match_histograms(np_image1, np_image2, channel_axis=None)
        else:
            raise ValueError("The input image must have 2 or 3 dimensions.")

        # Go back to torch
        image1_hat = torch.from_numpy(np_image1_hat).to(image1.device)

        return image1_hat

    def load_pretrained(self, weights_file: str):
        """
        Loads the pretrained model from the given path.
        """

        # download pretrained model
        # hf_model = "https://huggingface.co/isp-uv-es/opensr-model/resolve/main/sr_checkpoint.ckpt" # original one
        # create download link based on input
        hf_model = str(
            "https://huggingface.co/simon-donike/RS-SR-LTDF/resolve/main/"
            + str(weights_file)
        )

        # download pretrained model
        if not pathlib.Path(weights_file).exists():
            print("Downloading pretrained weights from: ", hf_model)
            with open(weights_file, "wb") as f:
                f.write(requests.get(hf_model).content)

        weights = torch.load(
            weights_file, weights_only=False, map_location=self.device
        )["state_dict"]

        # Remote perceptual tensors from weights
        for key in list(weights.keys()):
            if "loss" in key:
                del weights[key]

        self.model.load_state_dict(weights, strict=True)
        print("Loaded pretrained weights from: ", weights_file)

    def explainer(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        eta: float = 1.0,
        temperature: float = 1.0,
        custom_steps: int = 100,
        steps_to_consider_for_attributions: list = list(range(100)),
        attribution_method: Literal[
            "grad_x_input", "max_grad", "mean_grad", "min_grad"
        ] = "grad_x_input",
        verbose: bool = False,
        enable_checkpoint=True,
        histogram_matching=True,
    ):
        # Normalize and encode the LR image
        X = X.clone()
        Xnorm = self._tensor_encode(X)

        # ddim, latent and time_range
        ddim, latent, time_range = self._prepare_model(
            X=Xnorm, eta=eta, custom_steps=custom_steps, verbose=verbose
        )

        # Iterate over the timesteps
        container = []
        iterator = tqdm(
            time_range, desc="DDIM Sampler", total=custom_steps, disable=True
        )

        for i, step in enumerate(iterator):

            # Activate or deactivate gradient tracking
            if i in steps_to_consider_for_attributions:
                torch.set_grad_enabled(True)
            else:
                torch.set_grad_enabled(False)

            # Compute the latent image
            if enable_checkpoint:
                outs = checkpoint.checkpoint(
                    ddim.p_sample_ddim,
                    latent,
                    Xnorm,
                    step,
                    custom_steps - i - 1,
                    temperature,
                    use_reentrant=False,
                )
            else:
                outs = ddim.p_sample_ddim(
                    x=latent,
                    c=Xnorm,
                    t=step,
                    index=custom_steps - i - 1,
                    temperature=temperature,
                )
            latent, _ = outs

            if i not in steps_to_consider_for_attributions:
                continue

            # Apply the mask
            output_graph = (latent * mask).mean()

            # Compute the gradients
            grads = torch.autograd.grad(output_graph, Xnorm, retain_graph=True)[0]

            # Compute the attribution and save it
            with torch.no_grad():
                to_save = {
                    "latent": self._tensor_decode(latent, spe_cor=histogram_matching),
                    "attribution": self._attribution_methods(
                        Xnorm, grads, attribution_method
                    ),
                }
            container.append(to_save)

        return container

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int],
        img_lr: Dict[str, torch.Tensor],  # low-resolution image used for conditioning
        dates: Dict[str, torch.Tensor],
        x_T: Optional[torch.Tensor] = None,
        eta: float = 1.0,
        temperature: float = 1.0,
        custom_steps: int = 100,
        verbose: bool = False,
        histogram_matching: bool = True,
        save_iterations: bool = False,
    ):

        device = next(self.model.parameters()).device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        self.cond_net.eval()
        # Prevents PyTorch from tracking gradients, reducing memory
        # usage and speeding up inference.
        with torch.no_grad():
            cond_net_out, _, _, cond = self.cond_net(img_lr, dates)

            print("---------------Conditioning Temp Feats-------------------")
            print("cond 0:", cond[0].shape)
            print("cond 1:", cond[1].shape)
            print("cond 2:", cond[2].shape)
            print()

        sr_images = []

        for j in range(img_lr.shape[1]):
            # Each timestep in the reverse diffusion process is conditioned
            # the corresponding temporal features from the LR image

            # Assert shape, size, dimensionality
            X, padding = assert_tensor_validity(img_lr[:, j, :, :])
            print("X gen:", X.shape)  # torch.Size([1, 4, 16, 16])

            # Normalize the image
            X = X.clone()
            Xnorm = self._tensor_encode(X)
            print("Xnorm gen:", Xnorm.shape)  # torch.Size([1, 4, 4, 4])

            # ddim, latent and time_range
            ddim, latent, time_range = self._prepare_model(
                X=Xnorm, eta=eta, custom_steps=custom_steps, verbose=verbose
            )
            iterator = tqdm(
                time_range, desc="DDIM Sampler", total=custom_steps, disable=True
            )

            print("latent gen:", latent.shape)
            # Iterate over the timesteps
            if save_iterations:
                save_iters = []

            for i, step in enumerate(iterator):
                outs = ddim.p_sample_ddim(
                    x=latent,
                    c=[feat[:, j] for feat in cond],
                    t=step,
                    index=custom_steps - i - 1,
                    use_original_steps=False,
                    temperature=temperature,
                )
                latent, _ = outs

                if save_iterations:
                    save_iters.append(
                        self._tensor_decode(latent, spe_cor=histogram_matching)
                    )

            if save_iterations:
                return save_iters

            sr = self._tensor_decode(latent, spe_cor=histogram_matching)
            sr = revert_padding(sr, padding)
            print("sr: ", sr.shape)

            sr_images.append(sr)

        img_sr = torch.stack(sr_images, 1)
        print("img_sr: ", img_sr.shape)
        return img_sr
