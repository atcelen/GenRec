from typing import *

from dataclasses import dataclass, field
from copy import deepcopy



@dataclass
class Options:
    # Dataset
    input_res: int = 256
        ## Camera
    num_input_views: int = 1
    num_views: int = 8
    trajectory_sampler_type: Literal[
        "line",
        "spiral",
        "panoramic",
        "randomwalk",
    ] = "randomwalk"

    
    # training datasets
    koolai_data_dir: Optional[str] = None
    train_split_file: Optional[str] = None
    invalid_split_file: Optional[str] = None
    prompt_embed_dir: Optional[str] = None  # precompute T5 embedding
    prediction_types: list[str] = field(default_factory=lambda: ["rgb", "depth"])
    
    ## Transformer
    llama_style: bool = True
    patch_size: int = 8
    dim: int = 512
    num_blocks: int = 12
    num_heads: int = 8
    grad_checkpoint: bool = True


    # MVD
    # Component architecture configs (+ CLIP tokenizer files). The vendored
    # `genrec/model_config` directory holds the diffusers/transformers config.json
    # files only — all component weights come from the GenRec checkpoint. A full
    # SpatialGen weights directory (or HF repo id) also works here.
    spatialgen_ckpt: str = "genrec/model_config"
    depth_vae_ckpt: Optional[str] = None  # Separate checkpoint for depth VAE (defaults to pretrained_model_name_or_path)
    ray_encoder_ckpt: Optional[str] = None  # Separate checkpoint for ray encoder (defaults to pretrained_model_name_or_path)
    pretrained_model_name_or_path: str = spatialgen_ckpt
    load_fp16vae_for_sdxl: bool = True
        ## Config
    from_scratch: bool = False
    cfg_dropout_prob: float = 0.05  # Probability of dropping 3D conditioning (PC + obs mask + coords) per sample
    cfg_guidance_scale: float = 1.0  # Inference guidance scale (1.0 = no guidance)
    snr_gamma: float = 0.  # Min-SNR trick; `0.` menas not used
    num_inference_steps: int = 20
    noise_scheduler_type: Literal[
        "ddim",
        "dpmsolver++",
        "sde-dpmsolver++",
    ] = "dpmsolver++"
    prediction_type: Optional[str] = None  # `None` means using default prediction type
    beta_schedule: Optional[str] = None  # `None` means using the default beta schedule
    edm_style_training: bool = False  # EDM scheduling; cf. https://arxiv.org/pdf/2206.00364
    common_tricks: bool = True  # cf. https://arxiv.org/pdf/2305.08891 (including: 1. trailing timestep spacing, 2. rescaling to zero snr)
            ### SD3; cf. https://arxiv.org/pdf/2403.03206
    weighting_scheme: Literal[
        "sigma_sqrt",
        "logit_normal",
        "mode",
        "cosmap",
    ] = "logit_normal"
    logit_mean: float = 0.
    logit_std: float = 1.
    mode_scale: float = 1.29
    precondition_outputs: bool = False  # whether prediction x_0
    ## Model
    trainable_modules: Optional[str] = None  # train all parameters if None
    name_lr_mult: Optional[str] = field(default_factory=lambda: "class_embedding,")
    lr_mult: float = 10.
    ### Conditioning
    input_repaint: bool = False # whether to use repainting
    zero_init_conv_in: bool = True  # whether zero_init new conv_in params
    view_concat_condition: bool = True  # `True` for image-cond
    input_concat_plucker: bool = True
    input_concat_binary_mask: bool = True
    input_concat_gs_image: bool = False
    input_concat_gs_alpha: bool = False
    input_concat_pc_image: bool = True
    input_concat_coord_map: bool = False
    mask_pc_latents: bool = False  # Zero out PC/coord conditioning latents in unobserved regions using warp masks
    use_observation_mask: bool = False  # Replace binary input/output mask with soft warp coverage for observation conditioning
    use_learned_null_token: bool = False  # Use learned null tokens instead of zeros for unobserved PC/coord latents
    depth_filter_thresh: float = 0.5  # Depth discontinuity filter threshold for forward-warp rendering (GEN3C-like: 0.05)
    depth_grad_thresh: float = 0.0  # Relative depth-gradient edge filter for forward-warp rendering (0 = off). Tuned in visualize_pointclouds.py; recommended value 0.01.
    depth_weight_scale: float = 50.0  # Depth-weighted splatting sharpness: higher values make closer points dominate more aggressively (approaches hard z-buffering)
    coord_fp32_vae: bool = False  # Keep coordinate maps in float32 through depth VAE encode/decode (prevents bf16 quantization)
    coord_inverse_depth_weight: bool = False  # Weight coord L1 loss by normalized inverse depth (nearby surfaces get more gradient)
    source_only_da3: bool = False  # Only pass source views through DA3 (prevents cross-view contamination)

    ### Inference
    init_std: float = 0.  # cf. Instant3D inference trick, `0.` means not used
    init_noise_strength: float = 0.98  # used with `init_std`; cf. Instant3D inference trick, `1.` means not used
    init_bg: float = 0.  # used with `init_std` and `init_noise_strength`; gray background for the initialization

    ## LPIPS
    lpips_resize: int = 256  # `0` means no resizing
    lpips_weight: float = 1.0  # lpips weight in GSRecon, GSVAE, GSDiff rendering
    lpips_warmup_start: int = 0
    lpips_warmup_end: int = 0

    ## Region-Specific Loss
    use_region_loss: bool = True  # Enable region-aware losses (observed vs unobserved)
    use_dino_loss: bool = True  # Enable DINO feature loss within region losses
    mask_blur_kernel: int = 21  # Gaussian blur kernel for soft mask transitions
    mask_blur_sigma: float = 5.0  # Gaussian blur sigma
    region_obs_l1_weight: float = 0.05  # L1 loss weight for observed regions
    region_obs_lpips_weight: float = 0.5  # LPIPS weight for observed regions
    region_dino_weight: float = 0.2  # DINO feature loss weight (observed + unobserved)
    dino_extract_layers: List[int] = field(default_factory=lambda: [5, 8, 11])  # DINOv2 layers to extract
    dino_pool_sizes: List[int] = field(default_factory=lambda: [1, 2, 4])       # Paired spatial pool sizes
    dino_unobs_weight: float = 0.5  # Weight multiplier for unobserved relative to observed DINO loss
    region_boundary_tv_weight: float = 0.0  # Total variation weight at boundary
    region_coord_l1_weight: float = 0.0  # L1 loss weight for coord maps in observed regions (Task 1)
    region_loss_warmup_steps: int = 1000  # Linear warmup steps for region losses
    # Soft gating params (shared by region losses and confidence)
    region_gate_center: float = 0.4       # Sigmoid midpoint (t_ratio where gate = 0.5)
    region_gate_temperature: float = 0.05  # Sigmoid sharpness (lower = sharper)
    region_gate_compute_eps: float = 0.01  # Skip VAE decode when max(gate) < this
    region_diffusion_downweight: float = 0.0  # Spatially reduce diffusion MSE in observed regions when gate is active (0.0 = disabled)
    use_latent_l1_loss: bool = False     # Auxiliary L1 on predicted clean latents (Laplacian MLE)
    latent_l1_weight: float = 1.0       # Weight for latent L1 auxiliary loss
    use_reliability_weighting: bool = False  # Weight auxiliary losses by viewing-angle reliability

    ## VAE Skip Adapter
    use_vae_skip: bool = False  # Enable VAE encoder→decoder skip connections for PC renders
    lora_rank_vae: int = 0  # LoRA rank for VAE decoder adaptation (0 = disabled)
    use_depth_vae_skip: bool = False  # Enable skip connections for depth/coord VAE decoder
    lora_rank_depth_vae: int = 0  # LoRA rank for depth VAE decoder adaptation (0 = disabled)
    freeze_unet: bool = False  # Freeze UNet weights (only train adapter modules)

    ## Dual FFN
    dual_ffn: bool = False  # Dual FFN: separate reconstruction vs hallucination paths in transformer blocks

    ## Dual Reconstruction Pathway (side-branch)
    dual_recon: bool = False  # Side-branch reconstruction network producing x0 residual correction
    recon_bottleneck: int = 128  # Bottleneck channel width for ReconSideBranch
    recon_branch_depth: int = 2  # Number of SimpleResBlocks in ReconSideBranch
    recon_l1_weight: float = 1.0  # Weight for latent L1 recon loss (observed regions, high SNR)
    recon_num_attn_heads: int = 4  # Number of attention heads in sparse geometric cross-attention
    recon_K: int = 8  # K-nearest neighbors per target pixel in geometric index map
    recon_search_radius: float = 1.5  # Search radius in latent pixels for geometric correspondence
    recon_occlusion_eps: float = 0.02  # Relative z-buffer tolerance for occlusion filtering
    recon_use_geo_bias: bool = True  # Add geometric weight bias (log(geo_weights)) to attention
    recon_trust_sigma: float = 3.0  # Gaussian sigma for patch trust mask blur
    recon_pixel_space: bool = False  # Pixel-space recon branch (opt-in, latent-space remains default)
    recon_lpips_weight: float = 0.5  # LPIPS weight for pixel-space recon loss
    recon_clean_query: bool = False  # Use only clean signals (pc_render, warp_mask) for cross-attention query
    recon_dropout: float = 0.0  # Dropout rate for FiLMResBlocks in recon branch
    recon_t_threshold: int = 0  # If >0, train recon branch only when sampled t_discrete < threshold (match last-step inference distribution)
    recon_last_step_only: bool = False  # If True, apply pixel recon branch only at the final denoising step (avoids lossy re-encodes)
    train_t_clamp_ratio: Optional[Union[float, str]] = None  # None=use region-gate clamp; "auto"=derive from recon_t_threshold; float=explicit max_t_ratio override
    recon_pixel_lpips_full: bool = False  # Pixel branch only: compute LPIPS on the full image (branch can still only affect observed regions, but LPIPS gets a seam-aware signal across the mask boundary)
    recon_mask_blur_kernel: Optional[int] = None  # Pixel branch only: override obs-mask Gaussian blur kernel. None → falls back to global mask_blur_kernel (backward compatible)
    recon_mask_blur_sigma: Optional[float] = None  # Pixel branch only: override obs-mask Gaussian blur sigma. None → falls back to global mask_blur_sigma (backward compatible)
    recon_grad_weight: float = 0.0  # Pixel branch only: weight for edge-preserving (gradient-domain L1) auxiliary loss. 0.0 → disabled (backward compatible)
    recon_delta_sparsity_weight: float = 0.0  # Pixel branch only: weight for masked L1 sparsity on pixel_delta. 0.0 → disabled (backward compatible)
    recon_delta_sparsity_evidence_tau: float = 0.0  # Pixel branch only: >0 → evidence-weighted sparsity (penalty dies where |pc_render−decoded_x0|>tau). 0 → plain masked L1 sparsity
    recon_gram_weight: float = 0.0  # Pixel branch only: weight for VGG19 Gram-matrix style loss on pixel_corrected vs gt. 0.0 → disabled (backward compatible)
    recon_pixel_source_from_vae: bool = False  # VAE-init source conv for PixelReconSideBranch (mirrors ReconSideBranch)
    recon_use_plucker: bool = False  # Pixel branch only: feed per-pixel Plucker rays into query and source paths (zero-init residual; back-compat when False)
    recon_unrolled: bool = False  # Run partial N-step denoising (no_grad) to produce realistic x0 for recon branch training (requires freeze_unet=True)
    recon_unrolled_steps: int = 15  # Number of denoising steps for the unrolled inference loop

    ## V-Prediction Configuration (Diff2Flow)
    v_as_flow_prediction: bool = True  # Reformulate v-prediction as pseudo-flow prediction

    ## Non-Gaussian Source for Flow Matching (ablation)
    init_from_pc_renders: bool = False  # Flow source = aligned PC render latents instead of N(0, I) noise. Requires v_as_flow_prediction=True.

    ## Drift Model Configuration
    num_drift_samples: int = 4  # Number of noise samples per step for drift loss computation

    ## Integrated GeometryModel Configuration
    geometry_model_ckpt: str = "depth-anything/DA3NESTED-GIANT-LARGE"  # HuggingFace model path
    geometry_process_res: int = 616  # Processing resolution for depth estimation
    geometry_infer_gs: bool = False  # Whether to infer 3D Gaussians (memory intensive)
    
    ## Geometry Caching
    # When enabled, geometry outputs are cached to disk for faster subsequent runs
    use_geometry_cache: bool = False  # Enable disk caching of geometry predictions
    geometry_cache_path: Optional[str] = None  # Path to cache directory
    log_scale_coords: bool = False  # Use log(1+x) normalization for scene coordinates
    normalize_coords_with_output_views: bool = False  # Include output views (GT at eval) in coord normalization bounds

    ## Post-Hoc Confidence Network (DINOv2 + DPT)
    use_posthoc_confidence: bool = False  # Enable post-hoc confidence predictor at eval time
    confidence_network_ckpt: Optional[str] = None  # Path to trained ConfidencePredictor checkpoint
    confidence_dpt_features: int = 256  # DPT decoder channel width
    confidence_dino_model: str = "dinov2_vitb14_reg"  # DINOv2 hub model name
    confidence_extract_layers: List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    confidence_target_tau_l1: float = 0.05  # Temperature for L1 error -> confidence target
    confidence_target_use_dino: bool = True  # Include DINO similarity in confidence target

    # Custom options
    num_tasks: int = 2

    def __post_init__(self):
        self.in_channels = 4 + (16 if self.input_concat_plucker else 0) + (1 if self.input_concat_binary_mask else 0) + (4 if self.input_concat_pc_image else 0) + (4 if self.input_concat_gs_image else 0) + (1 if self.input_concat_gs_alpha else 0) + (4 if self.input_concat_coord_map else 0)
        self.unet_from_pretrained_kwargs = {
            "sample_size": self.input_res // 8,
            "in_channels": self.in_channels,
            "zero_init_conv_in": self.zero_init_conv_in,
            "view_concat_condition": self.view_concat_condition,
            "input_concat_plucker": self.input_concat_plucker,
            "input_concat_binary_mask": self.input_concat_binary_mask,
            "input_concat_warpped_image": self.input_concat_pc_image,
            "num_input_views": self.num_input_views,
            "num_output_views": self.num_views - self.num_input_views,
            "num_tasks": self.num_tasks,
            "cd_attention_mid": self.num_tasks > 1,
            "multiview_attention": True,
            "sparse_mv_attention": False,
            "disable_mv_attention_in_64x64": self.input_res == 512,
            "dual_ffn": self.dual_ffn,
        }



def _update_opt(opt: Options, **kwargs) -> Options:
    new_opt = deepcopy(opt)
    for k, v in kwargs.items():
        setattr(new_opt, k, v)
    return new_opt

def options_to_dict(obj: Options) -> dict:
    """Convert Options dataclass to dictionary."""
    if not isinstance(obj, Options):
        raise ValueError("Input must be an instance of Options dataclass.")
    result = {}
    for field_name in obj.__dataclass_fields__:
        result[field_name] = getattr(obj, field_name)
    return result
