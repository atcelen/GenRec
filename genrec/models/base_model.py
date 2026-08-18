from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from diffusers import AutoencoderKL
from diffusers.utils.torch_utils import randn_tensor
import PIL
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
import einops
from typing import Optional, List, Union, Dict, Any, Tuple

from genrec.models import UNetMVMM2DConditionModel
from genrec.models.pose_adapter import RayMapEncoder
from genrec.models.geo_model import GeometryModel, BatchedPrediction
from genrec.models.vae_skip_adapter import VAESkipAdapter
from genrec.utils.logger import setup_logger

import os
import hashlib
from pathlib import Path

logging = setup_logger(__name__)


def _resolve_model_config_path(ckpt):
    """Resolve the vendored ``genrec/model_config`` spec to an absolute path.

    The YAML configs reference the vendored component-config directory as
    ``genrec/model_config``, which is only valid relative to the repo root (and
    would otherwise be misread as a HuggingFace repo id). Resolve it against the
    installed package location; leave every other value (HF repo ids, existing
    paths) untouched.
    """
    if ckpt is None:
        return ckpt
    p = str(ckpt)
    if os.path.isdir(p):
        return p
    if p.replace("\\", "/").rstrip("/").endswith("genrec/model_config"):
        pkg_dir = Path(__file__).resolve().parents[1] / "model_config"
        if pkg_dir.is_dir():
            return str(pkg_dir)
    return p


class GenerativeReconstructionBaseModel(torch.nn.Module, ABC):
    """
    Abstract base class for all generative reconstruction training paradigms.

    Handles shared component creation (VAE, UNet, CLIP, ray encoder, geometry model),
    shared utilities (latent preparation, ray processing, coordinate maps),
    and shared lifecycle (set_train/set_eval, device property).

    Subclasses MUST implement:
        - _create_scheduler(opt) -> tuple
        - forward(...) -> torch.Tensor
        - evaluate(...) -> Dict[str, torch.Tensor]

    Subclasses MAY override:
        - _create_unet(opt) -> Tuple[nn.Module, dict]
        - _create_image_vae(opt), _create_depth_vae(opt)
        - _create_tokenizer(opt), _create_text_encoder(opt)
        - _create_ray_encoder(opt), _create_geometry_model(opt)
        - set_train(), set_eval()
    """

    def __init__(self, opt, mp_mode: str = "no"):
        super().__init__()

        # 1. Device and Dtype Handling
        # Do not force "cuda" here. Let the Trainer handle .to(device).
        # We store the desired dtype for later casting.
        self.mp_mode = mp_mode
        self.mixed_dtype = torch.bfloat16 if mp_mode == "bf16" else (torch.float16 if mp_mode == "fp16" else torch.float32)

        # Resolve the vendored `genrec/model_config` spec (repo-relative) to an
        # absolute path so component creation works from any working directory.
        for _attr in ("pretrained_model_name_or_path", "spatialgen_ckpt", "depth_vae_ckpt", "ray_encoder_ckpt"):
            if hasattr(opt, _attr):
                setattr(opt, _attr, _resolve_model_config_path(getattr(opt, _attr)))

        # 2. Component Initialization
        # We initialize sub-modules but do NOT immediately move them to GPU.
        # This prevents "cuda:0" OOMs in multi-gpu setups before DDP wrapping.
        self.noise_scheduler, self.val_scheduler = self._create_scheduler(opt)
        logging.info("Loaded scheduler")

        self.image_vae = self._create_image_vae(opt)
        logging.info("Loaded VAE image encoder")

        self.depth_vae = self._create_depth_vae(opt)
        logging.info("Loaded VAE depth encoder")

        self.ray_encoder = self._create_ray_encoder(opt)
        logging.info("Loaded ray encoder")

        self.tokenizer = self._create_tokenizer(opt)
        logging.info("Loaded tokenizer")

        self.text_encoder = self._create_text_encoder(opt)
        logging.info("Loaded text encoder")

        self.unet, self._unet_loading_info = self._create_unet(opt)
        logging.info("Loaded UNet model")

        # 3. Freeze parameters (standard practice)
        self.image_vae.requires_grad_(False)
        self.depth_vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.ray_encoder.requires_grad_(False)
        self.freeze_unet = getattr(opt, "freeze_unet", False)
        if self.freeze_unet:
            self.unet.requires_grad_(False)
            logging.info("UNet frozen (freeze_unet=True)")
        else:
            self.unet.requires_grad_(True)

        self.vae_scale_factor = 2 ** (len(self.image_vae.config.block_out_channels) - 1)
        self.default_text_prompt = "high quality, 8k, realistic textures, detailed lighting, ultra realistic"
        self.default_negative_prompt = ""
        self.num_tasks = self.unet.config.num_tasks

        # 4. Integrated GeometryModel Configuration
        self.geometry_model = self._create_geometry_model(opt)
        logging.info("Loaded integrated GeometryModel")

        # Caching configuration
        self.use_geometry_cache = getattr(opt, "use_geometry_cache", False)
        self.geometry_cache_path = getattr(opt, "geometry_cache_path", None)
        if self.use_geometry_cache and self.geometry_cache_path:
            os.makedirs(self.geometry_cache_path, exist_ok=True)
            logging.info(f"Geometry caching enabled at: {self.geometry_cache_path}")

        # Geometry processing parameters
        self.geometry_process_res = getattr(opt, "geometry_process_res", 504)
        self.geometry_infer_gs = getattr(opt, "geometry_infer_gs", False)
        self.depth_filter_thresh = getattr(opt, "depth_filter_thresh", 0.5)
        self.depth_grad_thresh = getattr(opt, "depth_grad_thresh", 0.0)
        self.depth_weight_scale = getattr(opt, "depth_weight_scale", 50.0)
        self.coord_fp32_vae = getattr(opt, "coord_fp32_vae", False)
        self.coord_inverse_depth_weight = getattr(opt, "coord_inverse_depth_weight", False)
        self.source_only_da3 = getattr(opt, "source_only_da3", False)
        self.log_scale_coords = getattr(opt, "log_scale_coords", False)
        self.normalize_coords_with_output_views = getattr(opt, "normalize_coords_with_output_views", False)

        # Learned null token for PC/coord latent masking
        self.use_learned_null_token = getattr(opt, "use_learned_null_token", False)
        if self.use_learned_null_token:
            self.pc_null_token = nn.Parameter(torch.zeros(1, 4, 1, 1))
            self.coord_null_token = nn.Parameter(torch.zeros(1, 4, 1, 1))

        # VAE skip adapter: encoder→decoder skip connections for PC renders.
        # Frozen only in recon-branch-only finetuning (freeze_unet=True); when
        # training the UNet, skip convs + LoRA weights remain trainable.
        self.use_vae_skip = getattr(opt, "use_vae_skip", False)
        if self.use_vae_skip:
            lora_rank_vae = getattr(opt, "lora_rank_vae", 0)
            self.vae_skip_adapter = VAESkipAdapter(self.image_vae, lora_rank=lora_rank_vae)
            if self.freeze_unet:
                # Re-freeze the entire VAE so newly-added skip convs and LoRA
                # params (which add_adapter unfreezes by default) stay frozen.
                self.image_vae.requires_grad_(False)
                logging.info(f"Created VAESkipAdapter as frozen pretrained backbone (LoRA rank={lora_rank_vae})")
            else:
                for i in range(self.vae_skip_adapter.num_skips):
                    getattr(self.image_vae.decoder, f"skip_conv_{i}").requires_grad_(True)
                vae_gradient_checkpointing = getattr(opt, "vae_gradient_checkpointing", True)
                if vae_gradient_checkpointing:
                    self.vae_skip_adapter.gradient_checkpointing = True
                    logging.info("Enabled gradient checkpointing on VAE decoder (skip adapter)")
                logging.info(f"Created VAESkipAdapter with trainable skip projections (LoRA rank={lora_rank_vae})")
        self._pc_skip_features = None  # Populated during encode, consumed during decode

        # Depth VAE skip adapter: encoder→decoder skip connections for coord maps.
        # Same freeze_unet-conditional treatment as the image VAE skip adapter.
        self.use_depth_vae_skip = getattr(opt, "use_depth_vae_skip", False)
        if self.use_depth_vae_skip:
            lora_rank_depth_vae = getattr(opt, "lora_rank_depth_vae", 0)
            self.depth_vae_skip_adapter = VAESkipAdapter(self.depth_vae, lora_rank=lora_rank_depth_vae)
            if self.freeze_unet:
                self.depth_vae.requires_grad_(False)
                logging.info(f"Created depth VAESkipAdapter as frozen pretrained backbone (LoRA rank={lora_rank_depth_vae})")
            else:
                for i in range(self.depth_vae_skip_adapter.num_skips):
                    getattr(self.depth_vae.decoder, f"skip_conv_{i}").requires_grad_(True)
                vae_gradient_checkpointing = getattr(opt, "vae_gradient_checkpointing", True)
                if vae_gradient_checkpointing:
                    self.depth_vae_skip_adapter.gradient_checkpointing = True
                    logging.info("Enabled gradient checkpointing on depth VAE decoder (skip adapter)")
                logging.info(f"Created depth VAESkipAdapter with trainable skip projections (LoRA rank={lora_rank_depth_vae})")
        self._coord_skip_features = None  # Populated during encode, consumed during decode

    # ===========================================================
    # Abstract methods (MUST be implemented by subclasses)
    # ===========================================================

    @abstractmethod
    def _create_scheduler(self, opt) -> tuple:
        """Create and return (noise_scheduler, val_scheduler)."""
        ...

    @abstractmethod
    def forward(self, **kwargs) -> torch.Tensor:
        """Training forward pass. Returns loss tensor."""
        ...

    @abstractmethod
    def evaluate(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Evaluation/inference pass. Returns results dict."""
        ...

    # ===========================================================
    # Shared component creation (overridable)
    # ===========================================================

    @staticmethod
    def _subfolder_has_weights(ckpt, subfolder: str) -> bool:
        """Whether `<ckpt>/<subfolder>/` holds a weight file.

        A local config-only directory (e.g. the vendored ``genrec/model_config``)
        carries only ``config.json``; the actual weights come from the full GenRec
        checkpoint loaded afterwards, so components are then built from config
        alone. Non-local paths (HuggingFace repo ids) are assumed to have weights.
        """
        d = os.path.join(str(ckpt), subfolder)
        if not os.path.isdir(d):
            return True  # HF repo id (or missing dir — let from_pretrained raise)
        weight_names = (
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.bin",
        )
        return any(os.path.exists(os.path.join(d, n)) for n in weight_names)

    def _create_image_vae(self, opt) -> torch.nn.Module:
        ckpt = getattr(opt, "pretrained_model_name_or_path", opt.spatialgen_ckpt)
        if not self._subfolder_has_weights(ckpt, "vae"):
            return AutoencoderKL.from_config(AutoencoderKL.load_config(ckpt, subfolder="vae"))
        vae = AutoencoderKL.from_pretrained(ckpt, subfolder="vae")
        return vae

    def _create_depth_vae(self, opt) -> torch.nn.Module:
        ckpt = getattr(opt, "depth_vae_ckpt", None) or getattr(opt, "pretrained_model_name_or_path", opt.spatialgen_ckpt)
        if not self._subfolder_has_weights(ckpt, "scm_vae"):
            return AutoencoderKL.from_config(AutoencoderKL.load_config(ckpt, subfolder="scm_vae"))
        scm_vae = AutoencoderKL.from_pretrained(ckpt, subfolder="scm_vae")
        return scm_vae

    def _create_ray_encoder(self, opt) -> torch.nn.Module:
        ckpt = getattr(opt, "ray_encoder_ckpt", None) or getattr(opt, "pretrained_model_name_or_path", None)
        # from_pretrained_new falls back to random init when the directory holds
        # no ray_encoder weight file (config-only directory).
        ray_encoder = RayMapEncoder.from_pretrained_new(ckpt, subfolder="ray_encoder")
        return ray_encoder

    def _create_tokenizer(self, opt) -> CLIPTokenizer:
        ckpt = getattr(opt, "pretrained_model_name_or_path", opt.spatialgen_ckpt)
        subfolder = getattr(opt, "tokenizer_subfolder", "tokenizer")
        return CLIPTokenizer.from_pretrained(ckpt, subfolder=subfolder)

    def _create_text_encoder(self, opt) -> CLIPTextModel:
        ckpt = getattr(opt, "pretrained_model_name_or_path", opt.spatialgen_ckpt)
        subfolder = getattr(opt, "text_encoder_subfolder", "text_encoder")
        if not self._subfolder_has_weights(ckpt, subfolder):
            from transformers import CLIPTextConfig
            config = CLIPTextConfig.from_pretrained(ckpt, subfolder=subfolder)
            return CLIPTextModel(config)
        return CLIPTextModel.from_pretrained(ckpt, subfolder=subfolder)

    def _create_unet(self, opt) -> Tuple[torch.nn.Module, dict]:
        ckpt = getattr(opt, "pretrained_model_name_or_path", opt.spatialgen_ckpt)
        from_scratch = getattr(opt, "from_scratch", True)
        out_channels = getattr(opt, "out_channels", 4)

        if not from_scratch:
            unet_subfolder = getattr(opt, "subfolder", "unet")
            return UNetMVMM2DConditionModel.from_pretrained_new(
                ckpt,
                subfolder=unet_subfolder,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=getattr(opt, "ignore_mismatched_sizes", True),
                output_loading_info=getattr(opt, "output_loading_info", True),
                # Config-only instantiation when the directory has no unet weights
                # (they come from the full GenRec checkpoint instead).
                from_scratch=not self._subfolder_has_weights(ckpt, unet_subfolder),
                out_channels=out_channels,
                **opt.unet_from_pretrained_kwargs,
            )

        def _get(name, default): return getattr(opt, name, default)

        return UNetMVMM2DConditionModel(
            sample_size=_get("sample_size", 64),
            in_channels=_get("in_channels", 4),
            out_channels=out_channels,
            down_block_types=_get("down_block_types", ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")),
            up_block_types=_get("up_block_types", ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")),
            block_out_channels=_get("block_out_channels", (320, 640, 1280, 1280)),
            layers_per_block=_get("layers_per_block", 2),
            cross_attention_dim=_get("cross_attention_dim", 768),
            attention_head_dim=_get("attention_head_dim", 8),
            use_linear_projection=_get("use_linear_projection", True),
            addition_embed_type=_get("addition_embed_type", None),
            time_embedding_type=_get("time_embedding_type", "positional"),
            resnet_time_scale_shift=_get("resnet_time_scale_shift", "default"),
        ), {}

    def _create_geometry_model(self, opt) -> GeometryModel:
        """Create and configure the GeometryModel."""
        geometry_ckpt = getattr(opt, "geometry_model_ckpt", "depth-anything/DA3NESTED-GIANT-LARGE")
        geometry_model = GeometryModel.from_pretrained(geometry_ckpt)
        geometry_model.requires_grad_(False)  # Always frozen
        geometry_model.eval()
        return geometry_model

    # ===========================================================
    # Shared properties
    # ===========================================================

    @property
    def device(self):
        """
        Dynamic device property.
        In DDP, 'self.unet.device' will correctly point to the local GPU (e.g., cuda:3).
        """
        return self.unet.device

    # ===========================================================
    # Shared utilities
    # ===========================================================

    def check_inputs(self, image, height, width):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(f"`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List` but is {type(image)}")
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def set_train(self):
        """Sets correct training mode for all submodules."""
        if self.freeze_unet:
            self.unet.eval()
        else:
            self.unet.train()
        self.ray_encoder.train()
        # Skip adapters follow freeze_unet: frozen-backbone in recon-only mode,
        # trainable when the UNet is being trained.
        if self.use_vae_skip:
            if self.freeze_unet:
                self.vae_skip_adapter.eval()
            else:
                self.vae_skip_adapter.train()
        if self.use_depth_vae_skip:
            if self.freeze_unet:
                self.depth_vae_skip_adapter.eval()
            else:
                self.depth_vae_skip_adapter.train()

        # Frozen models stay in eval mode
        self.image_vae.eval()
        self.text_encoder.eval()
        if self.geometry_model is not None:
            self.geometry_model.eval()

    def set_eval(self):
        self.unet.eval()
        self.image_vae.eval()
        self.ray_encoder.eval()
        self.text_encoder.eval()
        if self.use_vae_skip:
            self.vae_skip_adapter.eval()
        if self.use_depth_vae_skip:
            self.depth_vae_skip_adapter.eval()
        if self.geometry_model is not None:
            self.geometry_model.eval()

    def trainable_parameters(self):
        """Yield only parameters that require gradients."""
        return (p for p in self.parameters() if p.requires_grad)

    def trainable_param_groups(self, base_lr, adapter_lr=None, recon_lr=None):
        """Return parameter groups with per-module learning rates.

        Groups:
          - 'adapter': VAE skip convs + LoRA params (adapter_lr)
          - 'recon': dual-recon pathway params (recon_lr)
          - 'base': everything else that requires grad (base_lr)
        """
        if adapter_lr is None:
            adapter_lr = base_lr
        if recon_lr is None:
            recon_lr = base_lr

        adapter_ids = set()
        if self.use_vae_skip:
            for i in range(self.vae_skip_adapter.num_skips):
                for p in getattr(self.image_vae.decoder, f"skip_conv_{i}").parameters():
                    adapter_ids.add(id(p))
            for name, p in self.image_vae.named_parameters():
                if p.requires_grad:
                    adapter_ids.add(id(p))
        if self.use_depth_vae_skip:
            for i in range(self.depth_vae_skip_adapter.num_skips):
                for p in getattr(self.depth_vae.decoder, f"skip_conv_{i}").parameters():
                    adapter_ids.add(id(p))
            for name, p in self.depth_vae.named_parameters():
                if p.requires_grad:
                    adapter_ids.add(id(p))

        recon_ids = set()
        for name, p in self.named_parameters():
            if p.requires_grad and 'recon_branch' in name:
                recon_ids.add(id(p))

        base, adapter, recon = [], [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            if id(p) in adapter_ids:
                adapter.append(p)
            elif id(p) in recon_ids:
                recon.append(p)
            else:
                base.append(p)

        groups = []
        if base:
            groups.append({'params': base, 'lr': base_lr, 'name': 'base'})
        if adapter:
            groups.append({'params': adapter, 'lr': adapter_lr, 'name': 'adapter'})
        if recon:
            groups.append({'params': recon, 'lr': recon_lr, 'name': 'recon'})
        return groups

    def _encode_text(self, prompt):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        text_embed = self.text_encoder(text_input_ids)[0]
        return text_embed

    def prepare_img_latents(self, image, batch_size, dtype, device, generator=None, do_classifier_free_guidance=False, vae=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
             raise ValueError(f"`image` invalid type {type(image)}")

        if isinstance(image, torch.Tensor):
            if image.ndim == 3: image = image.unsqueeze(0)
            if image.min() >= 0 and image.max() <= 1: image = image * 2.0 - 1.0
        else:
             if isinstance(image, (PIL.Image.Image, np.ndarray)): image = [image]
             if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
                 image = [np.array(i.convert("RGB"))[None, :] for i in image]
                 image = np.concatenate(image, axis=0)
             elif isinstance(image, list) and isinstance(image[0], np.ndarray):
                 image = np.concatenate([i[None, :] for i in image], axis=0)
             image = image.transpose(0, 3, 1, 2)
             image = torch.from_numpy(image).float() / 127.5 - 1.0

        image = image.to(device=device, dtype=dtype)

        if vae is None:
            vae = self.image_vae

        # VAE Encoding
        with torch.no_grad():
            if isinstance(generator, list):
                init_latents = [vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = vae.encode(image).latent_dist.mode()

            init_latents = init_latents * vae.config.scaling_factor

        if batch_size > init_latents.shape[0]:
            num_images_per_prompt = batch_size // init_latents.shape[0]
            bs_embed, emb_c, emb_h, emb_w = init_latents.shape
            init_latents = init_latents.unsqueeze(1).repeat(1, num_images_per_prompt, 1, 1, 1)
            init_latents = init_latents.view(bs_embed * num_images_per_prompt, emb_c, emb_h, emb_w)

        return init_latents

    def _process_rays(self, rays: torch.FloatTensor):
        encoded_rays = self.ray_encoder(rays)
        return encoded_rays

    def _mask_latents(self, latents, mask):
        """Mask conditioning latents in unobserved regions (zero or learned null token)."""
        if self.use_learned_null_token:
            null = self.pc_null_token.to(dtype=latents.dtype, device=latents.device)
            return mask * latents + (1 - mask) * null
        return latents * mask

    def _mask_coord_latents(self, latents, mask):
        """Mask coord conditioning latents in unobserved regions (zero or learned null token)."""
        if self.use_learned_null_token:
            null = self.coord_null_token.to(dtype=latents.dtype, device=latents.device)
            return mask * latents + (1 - mask) * null
        return latents * mask

    # ===========================================================
    # VAE skip adapter utilities
    # ===========================================================

    def encode_pc_with_features(self, pc_renders, warp_masks, dtype=None):
        """
        Encode PC renders through the image VAE, capturing encoder features
        for skip connections to the decoder.

        Args:
            pc_renders: (N, 3, H, W) in [0, 1] range
            warp_masks: (N, 1, H, W) binary warp mask
            dtype: optional dtype override (defaults to self.mixed_dtype)

        Returns:
            (latents, skip_features)
        """
        _dtype = dtype if dtype is not None else self.mixed_dtype
        # Normalize to [-1, 1]
        x = pc_renders * 2.0 - 1.0
        x = x.to(device=self.device, dtype=_dtype)
        warp_masks = warp_masks.to(device=self.device, dtype=_dtype)

        latents, skip_features = self.vae_skip_adapter.encode_with_features(
            self.image_vae, x, warp_masks
        )
        self._pc_skip_features = skip_features
        return latents

    def decode_with_skip(self, latents, skip_features=None, dtype=None):
        """
        Decode latents through image VAE, optionally injecting skip features.
        Falls back to standard vae.decode() if no skip features available.

        Args:
            latents: (N, 4, H/8, W/8) — already divided by scaling_factor
            skip_features: optional list of 4 tensors [H/8, H/4, H/2, H].
                           If None, uses self._pc_skip_features.
            dtype: optional dtype to cast latents before decoding (e.g. fp32)

        Returns:
            Decoded pixel tensor (N, 3, H, W) in [-1, 1]
        """
        if dtype is not None:
            latents = latents.to(dtype=dtype)
        sf = skip_features if skip_features is not None else self._pc_skip_features
        if not self.use_vae_skip or sf is None:
            return self.image_vae.decode(latents).sample

        return self.vae_skip_adapter.decode_with_skip(self.image_vae, latents, sf)

    # ===========================================================
    # Depth VAE skip adapter utilities
    # ===========================================================

    def encode_coord_with_features(self, coord_maps, warp_masks, dtype=None):
        """
        Encode coord maps through the depth VAE, capturing encoder features
        for skip connections to the decoder.

        Args:
            coord_maps: (N, 3, H, W) in [-1, 1] range (already normalized)
            warp_masks: (N, 1, H, W) binary warp mask
            dtype: optional dtype override (defaults to self.mixed_dtype)

        Returns:
            latents: (N, 4, H/8, W/8) scaled by depth_vae.config.scaling_factor
        """
        _dtype = dtype if dtype is not None else self.mixed_dtype
        x = coord_maps.to(device=self.device, dtype=_dtype)
        warp_masks = warp_masks.to(device=self.device, dtype=_dtype)

        latents, skip_features = self.depth_vae_skip_adapter.encode_with_features(
            self.depth_vae, x, warp_masks
        )
        self._coord_skip_features = skip_features
        return latents

    def decode_depth_with_skip(self, latents, skip_features=None, dtype=None):
        """
        Decode latents through depth VAE, optionally injecting skip features.
        Falls back to standard depth_vae.decode() if no skip features available.

        Args:
            latents: (N, 4, H/8, W/8) — already divided by scaling_factor
            skip_features: optional list of 4 tensors [H/8, H/4, H/2, H].
                           If None, uses self._coord_skip_features.
            dtype: optional dtype to cast latents before decoding (e.g. fp32)

        Returns:
            Decoded pixel tensor (N, C, H, W) in [-1, 1]
        """
        if dtype is not None:
            latents = latents.to(dtype=dtype)
        sf = skip_features if skip_features is not None else self._coord_skip_features
        if not self.use_depth_vae_skip or sf is None:
            return self.depth_vae.decode(latents).sample

        return self.depth_vae_skip_adapter.decode_with_skip(self.depth_vae, latents, sf)

    # ===========================================================
    # Coordinate map utilities
    # ===========================================================

    def create_coordinate_map(self, depth: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
        """
        Generates a World Coordinate Map from Depth and Camera Params.

        Args:
            depth: Tensor of shape (B, 1, H, W). Depth map.
            K: Tensor of shape (B, 3, 3). Intrinsic matrix.
            c2w: Tensor of shape (B, 4, 4). Camera-to-World extrinsic matrix.

        Returns:
            coord_map: Tensor of shape (B, 3, H, W).
                       The (x, y, z) world coordinates for each pixel.
        """
        B, _, H, W = depth.shape
        device = depth.device

        i, j = torch.meshgrid(torch.arange(H, device=device),
                              torch.arange(W, device=device), indexing='ij')

        ones = torch.ones_like(i)
        uv_grid = torch.stack([j, i, ones], dim=0).float()  # (3, H, W)
        uv_grid = uv_grid.reshape(3, -1).unsqueeze(0).repeat(B, 1, 1)  # (B, 3, H*W)

        K_inv = torch.inverse(K)  # (B, 3, 3)
        cam_coords_normalized = torch.bmm(K_inv, uv_grid)  # (B, 3, H*W)

        depth_flat = depth.reshape(B, 1, -1)
        cam_coords = cam_coords_normalized * depth_flat  # (B, 3, H*W)

        ones_row = torch.ones((B, 1, H * W), device=device)
        cam_coords_hom = torch.cat([cam_coords, ones_row], dim=1)  # (B, 4, H*W)

        world_coords_hom = torch.bmm(c2w, cam_coords_hom)  # (B, 4, H*W)

        coord_map = world_coords_hom[:, :3, :].reshape(B, 3, H, W)

        return coord_map

    def normalize_coords(self, coord_map: torch.Tensor, scene_min: list, scene_max: list) -> torch.Tensor:
        """
        Normalizes coordinates to [0, 1] based on global scene bounds.

        Args:
            coord_map: Tensor of shape (B, 3, H, W)
            scene_min: [min_x, min_y, min_z]
            scene_max: [max_x, max_y, max_z]

        Returns:
            Normalized coord_map clamped to [-1, 1]
        """
        min_v = torch.tensor(scene_min, device=coord_map.device, dtype=coord_map.dtype).view(1, 3, 1, 1)
        max_v = torch.tensor(scene_max, device=coord_map.device, dtype=coord_map.dtype).view(1, 3, 1, 1)

        norm_map = (coord_map - min_v) / (max_v - min_v + 1e-8)
        norm_map = torch.clamp(norm_map, -1, 1)

        return norm_map

    # ===========================================================
    # Geometry processing & caching
    # ===========================================================

    def _generate_cache_key(self, scene_ids: List[str], frame_indices: Optional[List[List[int]]] = None) -> str:
        """Generate a unique cache key for geometry outputs."""
        key_str = "_".join(scene_ids)
        if frame_indices:
            key_str += "_" + "_".join(["-".join(map(str, fi)) for fi in frame_indices])
        key_str += f"_res{self.geometry_process_res}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_geometry_cache(self, cache_key: str) -> Optional[BatchedPrediction]:
        """Load geometry prediction from cache if available."""
        if not self.use_geometry_cache or not self.geometry_cache_path:
            return None

        cache_file = os.path.join(self.geometry_cache_path, f"{cache_key}.pt")
        if os.path.exists(cache_file):
            try:
                cached_data = torch.load(cache_file, map_location='cpu')
                return BatchedPrediction(**cached_data)
            except Exception as e:
                logging.warning(f"Failed to load geometry cache {cache_file}: {e}")
                return None
        return None

    def _save_geometry_cache(self, cache_key: str, prediction: BatchedPrediction) -> None:
        """Save geometry prediction to cache."""
        if not self.use_geometry_cache or not self.geometry_cache_path:
            return

        cache_file = os.path.join(self.geometry_cache_path, f"{cache_key}.pt")
        try:
            cache_data = {
                'processed_images': prediction.processed_images.cpu(),
                'depth': prediction.depth.cpu(),
                'conf': prediction.conf.cpu(),
                'extrinsics': prediction.extrinsics.cpu(),
                'intrinsics': prediction.intrinsics.cpu(),
                'plucker_rays': prediction.plucker_rays.cpu(),
                'pc_renders': prediction.pc_renders.cpu(),
                'coord_maps': prediction.coord_maps.cpu(),
                'full_coord_maps': prediction.full_coord_maps.cpu(),
                'warp_masks': prediction.warp_masks.cpu(),
                'gaussians': None,  # Don't cache gaussians (too large)
                'scales': prediction.scales.cpu() if prediction.scales is not None else None,
            }
            torch.save(cache_data, cache_file)
        except Exception as e:
            logging.warning(f"Failed to save geometry cache {cache_file}: {e}")

    @torch.no_grad()
    def process_geometry(
        self,
        raw_images: torch.Tensor,  # (B, T, H, W, 3) - raw images in [0, 255]
        scene_ids: Optional[List[str]] = None,
        T_in: int = 3,
        T_out: int = 1,
        conf_mask_thresh: float = 0.0,
        gt_extrinsics: torch.Tensor = None,  # (B, T, 4, 4) GT w2c matrices
        gt_intrinsics: torch.Tensor = None,  # (B, T, 3, 3) GT intrinsics at original image res
        depth_filter_thresh: float = 0.5,
        depth_grad_thresh: float = 0.0,
        depth_weight_scale: float = 50.0,
        dyn_masks: torch.Tensor = None,  # (B, T, 1, H, W) 1=dynamic 0=static
        precomputed_depths: torch.Tensor = None,  # (B, T, H, W) precomputed depth maps
        source_only_da3: bool = False,  # only pass source views through DA3
        compute_reliability: bool = False,  # compute viewing-angle reliability maps
    ) -> BatchedPrediction:
        """
        Process raw images through the integrated GeometryModel.

        This method handles batched geometry processing with optional caching.

        Args:
            raw_images: Raw images tensor of shape (B, T, H, W, 3) in [0, 255] range
            scene_ids: Optional list of scene identifiers for caching
            T_in: Number of input views
            T_out: Number of output views
            source_only_da3: Only pass source views through DA3 to prevent cross-view contamination

        Returns:
            BatchedPrediction containing all geometry outputs
        """
        # Check cache if scene_ids provided
        cache_key = None
        if scene_ids and self.use_geometry_cache:
            cache_key = self._generate_cache_key(scene_ids)
            cached = self._load_geometry_cache(cache_key)
            if cached is not None:
                return BatchedPrediction(
                    processed_images=cached.processed_images.to(self.device),
                    depth=cached.depth.to(self.device),
                    conf=cached.conf.to(self.device),
                    extrinsics=cached.extrinsics.to(self.device),
                    intrinsics=cached.intrinsics.to(self.device),
                    plucker_rays=cached.plucker_rays.to(self.device),
                    pc_renders=cached.pc_renders.to(self.device),
                    coord_maps=cached.coord_maps.to(self.device),
                    full_coord_maps=cached.full_coord_maps.to(self.device),
                    warp_masks=cached.warp_masks.to(self.device) if cached.warp_masks is not None else torch.zeros_like(cached.pc_renders[:, :, :1]).to(self.device),
                    gaussians=None,
                    scales=cached.scales.to(self.device) if cached.scales is not None else None,
                )

        # Run batched geometry forward in float32 — coordinate transforms,
        # depth comparisons, and point cloud rendering need full precision.
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            prediction = self.geometry_model.batched_forward(
                images=raw_images,
                process_res=self.geometry_process_res,
                process_res_method="upper_bound_resize",
                infer_gs=self.geometry_infer_gs,
                T_in=T_in,
                T_out=T_out,
                conf_mask_thresh=conf_mask_thresh,
                gt_extrinsics=gt_extrinsics,
                gt_intrinsics=gt_intrinsics,
                depth_filter_thresh=depth_filter_thresh,
                depth_grad_thresh=depth_grad_thresh,
                depth_weight_scale=depth_weight_scale,
                dyn_masks=dyn_masks,
                precomputed_depths=precomputed_depths,
                source_only_da3=source_only_da3,
                compute_reliability=compute_reliability,
                log_scale_coords=self.log_scale_coords,
                normalize_with_all_views=self.normalize_coords_with_output_views,
            )

        # Save to cache if enabled
        if cache_key:
            self._save_geometry_cache(cache_key, prediction)

        return prediction

    # ===========================================================
    # Raw image wrappers (call abstract forward/evaluate)
    # ===========================================================

    def forward_with_raw_images(
        self,
        raw_images: torch.Tensor,  # (B, T, H, W, 3) - raw images in [0, 255]
        scene_ids: Optional[List[str]] = None,
        T_in: int = 3,
        T_out: int = 1,
        **kwargs,
    ):
        """
        Forward pass that processes raw images through integrated geometry model first.

        This is the main entry point when using integrated geometry processing.
        """
        # Process geometry
        geo_pred = self.process_geometry(raw_images, scene_ids, T_in, T_out,
                                         source_only_da3=self.source_only_da3)

        # Extract processed images and convert to model format (B, T, C, H, W) in [0, 1]
        processed_imgs = geo_pred.processed_images.float() / 255.0  # (B, T, H, W, 3)
        processed_imgs = processed_imgs.permute(0, 1, 4, 2, 3)  # (B, T, 3, H, W)

        # Normalize to [-1, 1] for VAE
        processed_imgs = processed_imgs * 2.0 - 1.0

        # Split into input and output
        input_imgs = processed_imgs[:, :T_in]
        output_imgs = processed_imgs[:, T_in:]

        # Get plucker rays
        input_rays = geo_pred.plucker_rays[:, :T_in]
        output_rays = geo_pred.plucker_rays[:, T_in:]

        # Get point cloud renders
        input_pc_renders = geo_pred.pc_renders[:, :T_in]
        output_pc_renders = geo_pred.pc_renders[:, T_in:]

        # Get coordinate maps
        # loo_pred coord maps (from input depth only) — used for conditioning
        input_coord_map = geo_pred.coord_maps[:, :T_in].to(dtype=self.mixed_dtype)
        output_coord_map = geo_pred.coord_maps[:, T_in:].to(dtype=self.mixed_dtype)  # conditioning (loo)
        # all-view coord maps — used for supervision targets (no black holes)
        if self.source_only_da3:
            output_coord_target = None  # target depth is zero; fall back to output_coord_map
        else:
            output_coord_target = geo_pred.full_coord_maps[:, T_in:].to(dtype=self.mixed_dtype)

        # TODO: Remove this
        visualize = False
        if visualize:
            import torchvision.utils as vutils
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import os

            b0 = 0

            def apply_colormap(x, cmap_name='viridis'):
                """Apply colormap to single-channel tensor. Input: (N, 1, H, W) or (N, H, W), Output: (N, 3, H, W)"""
                if x.dim() == 4 and x.shape[1] == 1:
                    x = x.squeeze(1)
                elif x.dim() == 4 and x.shape[1] == 3:
                    x = x.norm(dim=1)
                x_min = x.amin(dim=(-2, -1), keepdim=True)
                x_max = x.amax(dim=(-2, -1), keepdim=True)
                x_norm = (x - x_min) / (x_max - x_min + 1e-8)
                cmap = cm.get_cmap(cmap_name)
                x_np = x_norm.float().cpu().numpy()
                colored = []
                for i in range(x_np.shape[0]):
                    rgba = cmap(x_np[i])
                    rgb = rgba[..., :3]
                    colored.append(torch.from_numpy(rgb).permute(2, 0, 1))
                return torch.stack(colored, dim=0)

            def norm_img_for_vis(x):
                if x.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
                    return x.float().clamp(-1, 1).add(1).div(2).clamp(0, 1)
                else:
                    return x.float() / 255.0

            def norm_pc_for_vis(x):
                if x.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
                    x_min = x.amin()
                    x_max = x.amax()
                    return ((x - x_min) / (x_max - x_min + 1e-8)).float().clamp(0, 1)
                else:
                    return x.float() / 255.0

            nin = input_imgs.shape[1]
            nout = output_imgs.shape[1]

            in_imgs = norm_img_for_vis(input_imgs[b0])
            out_imgs = norm_img_for_vis(output_imgs[b0])
            in_coords = apply_colormap(input_coord_map[b0], cmap_name='turbo')
            out_coords = apply_colormap(output_coord_map[b0], cmap_name='turbo')
            in_pc = norm_pc_for_vis(input_pc_renders[b0])
            out_pc = norm_pc_for_vis(output_pc_renders[b0])

            if in_pc.shape[1] != 3:
                in_pc = in_pc[:, :3] if in_pc.shape[1] > 3 else in_pc.expand(-1, 3, -1, -1)
            if out_pc.shape[1] != 3:
                out_pc = out_pc[:, :3] if out_pc.shape[1] > 3 else out_pc.expand(-1, 3, -1, -1)

            ncols = max(nin, nout)

            def pad_to_ncols(x, ncols):
                if x.shape[0] < ncols:
                    pad = torch.zeros(ncols - x.shape[0], *x.shape[1:], device=x.device, dtype=x.dtype)
                    return torch.cat([x, pad], dim=0)
                return x

            in_imgs_padded = pad_to_ncols(in_imgs.cpu(), ncols)
            in_coords_padded = pad_to_ncols(in_coords.cpu(), ncols)
            in_pc_padded = pad_to_ncols(in_pc.cpu(), ncols)
            out_imgs_padded = pad_to_ncols(out_imgs.cpu(), ncols)
            out_coords_padded = pad_to_ncols(out_coords.cpu(), ncols)
            out_pc_padded = pad_to_ncols(out_pc.cpu(), ncols)

            all_rows = torch.cat([
                in_imgs_padded,
                in_coords_padded,
                in_pc_padded,
                out_imgs_padded,
                out_coords_padded,
                out_pc_padded,
            ], dim=0)

            grid = vutils.make_grid(all_rows.cpu(), nrow=ncols, padding=2, normalize=False)

            vis_dir = getattr(self, "visualize_dir", "./")
            os.makedirs(vis_dir, exist_ok=True)
            out_fn = os.path.join(vis_dir, "last_forward_pass_vis.png")
            try:
                vutils.save_image(grid, out_fn)
                print(f"[VISUALIZE] Saved to {out_fn}")
            except Exception as e:
                print(f"[VISUALIZE] Failed to save visualization: {e}")

        # Extract warp masks for PC latent masking
        input_warp_masks = geo_pred.warp_masks[:, :T_in]
        output_warp_masks = geo_pred.warp_masks[:, T_in:]

        # Call standard forward (dispatches to subclass)
        return self.forward(
            input_imgs=input_imgs,
            input_rays=input_rays,
            input_coord_map=input_coord_map,
            input_pc_renders=input_pc_renders,
            output_imgs=output_imgs,
            output_rays=output_rays,
            output_coord_map=output_coord_map,
            output_pc_renders=output_pc_renders,
            output_coord_target=output_coord_target,
            input_warp_masks=input_warp_masks,
            output_warp_masks=output_warp_masks,
            T_in=T_in,
            T_out=T_out,
            **kwargs,
        )

    @torch.no_grad()
    def evaluate_with_raw_images(
        self,
        raw_images: torch.Tensor,  # (B, T, H, W, 3) - raw images in [0, 255]
        scene_ids: Optional[List[str]] = None,
        T_in: int = 3,
        T_out: int = 1,
        height: int = None,
        width: int = None,
        guidance_scale: float = 1.0,
        generator=None,
        num_inference_steps: int = None,
        conf_mask_thresh: float = 0.0,
        gt_extrinsics: torch.Tensor = None,  # (B, T, 4, 4) GT w2c matrices
        gt_intrinsics: torch.Tensor = None,  # (B, T, 3, 3) GT intrinsics at original image res
        depth_filter_thresh: float = 0.5,
        depth_grad_thresh: float = 0.0,
        depth_weight_scale: float = 50.0,
        source_only_da3: bool = False,
        **kwargs,
    ):
        """
        Evaluate using raw images by first processing through integrated geometry model.
        """
        # Process geometry
        geo_pred = self.process_geometry(
            raw_images, scene_ids, T_in, T_out,
            conf_mask_thresh=conf_mask_thresh,
            gt_extrinsics=gt_extrinsics,
            gt_intrinsics=gt_intrinsics,
            depth_filter_thresh=depth_filter_thresh,
            depth_grad_thresh=depth_grad_thresh,
            depth_weight_scale=depth_weight_scale,
            source_only_da3=source_only_da3,
        )

        # Extract processed images and convert to model format (B, T, C, H, W) in [0, 1]
        processed_imgs = geo_pred.processed_images.float() / 255.0
        processed_imgs = processed_imgs.permute(0, 1, 4, 2, 3)

        # Normalize to [-1, 1] for VAE
        processed_imgs = processed_imgs * 2.0 - 1.0

        # Split into input and output
        input_imgs = processed_imgs[:, :T_in]

        # Get plucker rays
        input_rays = geo_pred.plucker_rays[:, :T_in]
        output_rays = geo_pred.plucker_rays[:, T_in:]

        # Get point cloud renders
        input_pc_renders = geo_pred.pc_renders[:, :T_in]
        output_pc_renders = geo_pred.pc_renders[:, T_in:]

        # Get coordinate maps (loo_pred for conditioning)
        input_coord_map = geo_pred.coord_maps[:, :T_in].to(dtype=self.mixed_dtype)
        output_coord_map = geo_pred.coord_maps[:, T_in:].to(dtype=self.mixed_dtype)

        # Extract warp masks for PC latent masking
        input_warp_masks = geo_pred.warp_masks[:, :T_in]
        output_warp_masks = geo_pred.warp_masks[:, T_in:]

        # Compute geometric index maps for sparse cross-attention in recon branch
        geo_index_map = None
        out_out_index_map = None
        if getattr(self, 'dual_recon', False):
            from genrec.utils.geometric_router import (
                compute_batched_geometric_index_map,
                compute_batched_output_to_output_index_map,
            )
            T_total = T_in + T_out
            # Stage 1 (input→output) runs at pixel res; stage 2 (output→output)
            # matches the recon branch's token resolution: pixel for PixelReconSideBranch,
            # latent for ReconSideBranch (post-downsample).
            vsf_in_out = 1
            vsf_out_out = 1 if getattr(self, 'recon_pixel_space', False) else 8
            geo_index_map = compute_batched_geometric_index_map(
                source_depth=geo_pred.depth[:, :T_in].float(),
                source_w2c=geo_pred.extrinsics[:, :T_in].float(),
                source_intrinsics=geo_pred.intrinsics[:, :T_in].float(),
                target_w2c=geo_pred.extrinsics[:, T_in:T_total].float(),
                target_intrinsics=geo_pred.intrinsics[:, T_in:T_total].float(),
                K=getattr(self, 'recon_K', 8),
                vae_scale_factor=vsf_in_out,
                occlusion_eps=getattr(self, 'recon_occlusion_eps', 0.02),
                search_radius=getattr(self, 'recon_search_radius', 1.5),
            )
            # Skip out_out_index_map when source_only_da3 — target depth is zero
            if T_out > 1 and not source_only_da3:
                out_out_index_map = compute_batched_output_to_output_index_map(
                    output_depth=geo_pred.depth[:, T_in:T_total].float(),
                    output_w2c=geo_pred.extrinsics[:, T_in:T_total].float(),
                    output_intrinsics=geo_pred.intrinsics[:, T_in:T_total].float(),
                    K=getattr(self, 'recon_K', 8),
                    vae_scale_factor=vsf_out_out,
                    occlusion_eps=getattr(self, 'recon_occlusion_eps', 0.02),
                    search_radius=getattr(self, 'recon_search_radius', 1.5),
                )
            # Post-process with 3D distances for geometric attention weights
            from genrec.utils.geometric_router import replace_with_3d_distances
            geo_index_map = replace_with_3d_distances(
                geo_index_map,
                source_world_pos=geo_pred.full_coord_maps[:, :T_in].float(),
                target_world_pos=geo_pred.full_coord_maps[:, T_in:T_total].float(),
                T_out=T_out,
                vae_scale_factor=vsf_in_out,
            )
            if out_out_index_map is not None:
                out_out_index_map = replace_with_3d_distances(
                    out_out_index_map,
                    source_world_pos=geo_pred.full_coord_maps[:, T_in:T_total].float(),
                    target_world_pos=geo_pred.full_coord_maps[:, T_in:T_total].float(),
                    T_out=T_out,
                    vae_scale_factor=vsf_out_out,
                )

        # Call the standard evaluate method (dispatches to subclass)
        results = self.evaluate(
            input_imgs=input_imgs,
            input_rays=input_rays,
            input_pc_renders=input_pc_renders,
            output_rays=output_rays,
            output_pc_renders=output_pc_renders,
            input_coord_map=input_coord_map,
            output_coord_map=output_coord_map,
            input_warp_masks=input_warp_masks,
            output_warp_masks=output_warp_masks,
            T_in=T_in,
            T_out=T_out,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            geo_index_map=geo_index_map,
            out_out_index_map=out_out_index_map,
            **kwargs,
        )
        # Include geometry data for proper GT comparison in validation
        results['processed_images'] = processed_imgs  # (B, T, 3, H, W) in [-1, 1]
        results['coord_maps'] = geo_pred.full_coord_maps   # (B, T, 3, H, W) in [0, 1] (all-view, no holes)
        results['output_pc_renders'] = output_pc_renders  # (B, T_out, 3, H, W)
        results['output_warp_masks'] = output_warp_masks  # (B, T_out, 1, H, W) raw binary
        results['depth'] = geo_pred.depth  # (B, T, H, W) at original image resolution
        results['extrinsics'] = geo_pred.extrinsics  # (B, T, 4, 4) w2c
        results['intrinsics'] = geo_pred.intrinsics  # (B, T, 3, 3) at DA3 process_res
        if geo_pred.coord_norm_min is not None:
            results['coord_norm_min'] = geo_pred.coord_norm_min  # (B,) per-scene scalar
            results['coord_norm_max'] = geo_pred.coord_norm_max  # (B,) per-scene scalar
        return results
