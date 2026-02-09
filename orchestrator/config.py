"""Global configuration for the AI Influencer pipeline.

All hardware-specific settings, model paths, and generation parameters
are centralized here for the RTX 5090 / comfyui:latest-5090 environment.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ContentLane(Enum):
    SFW = "sfw"
    SUGGESTIVE = "suggestive"
    SPICY = "spicy"
    NSFW = "nsfw"


class ImageType(Enum):
    MASTER_PORTRAIT = "master_portrait"
    MASTER_THREE_QUARTER = "master_three_quarter"
    MASTER_FULL_BODY = "master_full_body"
    ONE_OFF = "one_off"
    BUNDLE_3 = "bundle_3"
    BUNDLE_5 = "bundle_5"
    BUNDLE_10 = "bundle_10"


@dataclass
class HardwareConfig:
    """RTX 5090-specific settings for comfyui:latest-5090."""
    gpu_name: str = "RTX 5090"
    vram_gb: int = 32
    # 5090 has 32GB VRAM — run at full resolution without tiling for base gen
    # Enable tiling only for upscale passes
    tiled_vae_encode: bool = False
    tiled_vae_decode: bool = False
    tiled_upscale_decode: bool = True
    # Batch sizes — 5090 can handle larger batches at SDXL resolution
    base_batch_size: int = 1
    # FP16 for inference, BF16 available on 5090 Blackwell arch
    dtype: str = "float16"
    device: str = "cuda:0"


@dataclass
class ModelConfig:
    """Model file paths relative to ComfyUI models directory."""
    # Base checkpoint
    checkpoint: str = "fabricatedXL_v70.safetensors"
    # VAE (optional separate — checkpoint includes one)
    vae_separate: Optional[str] = "SDXL/sdxl_vae.safetensors"
    # IP-Adapter models
    ipadapter_style: str = "noobIPAMARK1_mark1.safetensors"
    ipadapter_faceid: str = "ip-adapter-faceid-plusv2_sdxl.bin"
    # CLIP Vision models
    clip_vision_g: str = "clip_vision_g.safetensors"
    clip_vision_big_g: str = "CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors"
    clip_vision_h: str = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    # ControlNet models
    controlnet_canny: str = "SDXL/control-lora-canny-rank256.safetensors"
    controlnet_openpose: str = "noobaiXLControlnet_openposeModel.safetensors"
    # Upscale model
    upscale_model: str = "4x_foolhardy_Remacri.pth"
    # Detection models for detailers
    detector_hand: str = "bbox/hand_yolov9c.pt"
    detector_body: str = "segm/yolo11m-seg.pt"
    detector_nsfw: str = "segm/ntd11_anime_nsfw_segm_v5-variant1.pt"
    detector_face: str = "bbox/face_yolov9c.pt"
    detector_eyes: str = "bbox/Eyeful_v2-Individual.pt"
    # SAM model
    sam_model: str = "sam_vit_b_01ec64.pth"
    # TIPO prompt model
    tipo_model: str = "KBlueLeaf/TIPO-200M-ft2 | TIPO-200M-ft2-F16.gguf"


@dataclass
class SamplerConfig:
    """Default sampler settings matched to FabricatedXL v7."""
    sampler_name: str = "euler_ancestral"
    scheduler: str = "normal"
    steps: int = 28
    cfg: float = 6.0
    denoise: float = 1.0
    clip_skip: int = -2
    # Second pass (refinement)
    second_pass_enabled: bool = False
    second_pass_steps: int = 6
    second_pass_denoise: float = 0.2


@dataclass
class DetailerConfig:
    """Settings for the detailer chain."""
    # Global detailer settings
    guide_size: float = 512
    max_size: float = 2048
    bbox_crop_factor: float = 3.0
    # Per-detailer denoise strengths
    hand_denoise: float = 0.38
    body_denoise: float = 0.22
    nsfw_denoise: float = 0.28
    face_denoise: float = 0.24
    eyes_denoise: float = 0.20
    # Which detailers to enable per lane
    lane_detailers: dict = field(default_factory=lambda: {
        ContentLane.SFW: ["hand", "body", "face", "eyes"],
        ContentLane.SUGGESTIVE: ["hand", "body", "face", "eyes"],
        ContentLane.SPICY: ["hand", "body", "nsfw", "face", "eyes"],
        ContentLane.NSFW: ["hand", "body", "nsfw", "face", "eyes"],
    })


@dataclass
class ResolutionPreset:
    """Resolution presets for SDXL generation."""
    name: str
    width: int
    height: int
    aspect_ratio: str

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000


# Standard SDXL resolution presets
RESOLUTIONS = {
    "1:1": ResolutionPreset("Square", 1024, 1024, "1:1"),
    "1:1_large": ResolutionPreset("Square Large", 1536, 1536, "1:1"),
    "3:4": ResolutionPreset("Portrait 3:4", 896, 1152, "3:4"),
    "2:3": ResolutionPreset("Portrait 2:3", 1024, 1536, "2:3"),
    "5:8": ResolutionPreset("Portrait 5:8", 832, 1216, "5:8"),
    "9:16": ResolutionPreset("Portrait 9:16", 768, 1344, "9:16"),
    "9:21": ResolutionPreset("Portrait 9:21", 640, 1536, "9:21"),
    "13:24": ResolutionPreset("Portrait 13:24", 832, 1536, "13:24"),
    "4:3": ResolutionPreset("Landscape 4:3", 1152, 896, "4:3"),
    "16:9": ResolutionPreset("Landscape 16:9", 1344, 768, "16:9"),
}

# Master image resolution assignments
MASTER_IMAGE_RESOLUTIONS = {
    ImageType.MASTER_PORTRAIT: RESOLUTIONS["3:4"],      # Close-up portrait
    ImageType.MASTER_THREE_QUARTER: RESOLUTIONS["2:3"],  # 3/4 body shot
    ImageType.MASTER_FULL_BODY: RESOLUTIONS["9:16"],     # Full body
}


@dataclass
class VaultLaneConfig:
    """Image count breakdown per lane (200 images per lane)."""
    one_off_count: int = 80
    bundle_3_count: int = 10     # 10 bundles × 3 = 30
    bundle_5_count: int = 10     # 10 bundles × 5 = 50
    bundle_10_count: int = 4     # 4 bundles × 10 = 40
    # Total: 80 + 30 + 50 + 40 = 200

    @property
    def total_images(self) -> int:
        return (
            self.one_off_count
            + self.bundle_3_count * 3
            + self.bundle_5_count * 5
            + self.bundle_10_count * 10
        )

    @property
    def total_bundles(self) -> int:
        return self.bundle_3_count + self.bundle_5_count + self.bundle_10_count


@dataclass
class UpscaleConfig:
    """Upscale settings for final output."""
    enabled: bool = True
    scale_factor: float = 2.0
    tile_width: int = 512
    tile_height: int = 512
    denoise: float = 0.18
    steps: int = 18


@dataclass
class OutputConfig:
    """Output directory structure."""
    base_dir: str = "/output"
    format: str = "png"
    quality: int = 100
    save_metadata: bool = True

    def character_dir(self, character_id: str) -> str:
        return os.path.join(self.base_dir, character_id)

    def master_dir(self, character_id: str) -> str:
        return os.path.join(self.character_dir(character_id), "master")

    def vault_dir(self, character_id: str, lane: ContentLane) -> str:
        return os.path.join(self.character_dir(character_id), "vault", lane.value)

    def bundle_dir(self, character_id: str, lane: ContentLane,
                   bundle_type: str, bundle_index: int) -> str:
        return os.path.join(
            self.vault_dir(character_id, lane),
            "bundles",
            f"{bundle_type}_{bundle_index:03d}"
        )

    def one_off_dir(self, character_id: str, lane: ContentLane) -> str:
        return os.path.join(self.vault_dir(character_id, lane), "one_offs")


@dataclass
class PipelineConfig:
    """Master configuration combining all sub-configs."""
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    detailer: DetailerConfig = field(default_factory=DetailerConfig)
    upscale: UpscaleConfig = field(default_factory=UpscaleConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    vault_lane: VaultLaneConfig = field(default_factory=VaultLaneConfig)

    # ComfyUI connection
    comfyui_host: str = "127.0.0.1"
    comfyui_port: int = 8188

    # Character roster
    characters_dir: str = "characters/roster"
    total_characters: int = 21

    @property
    def comfyui_url(self) -> str:
        return f"http://{self.comfyui_host}:{self.comfyui_port}"

    @property
    def total_master_images(self) -> int:
        return self.total_characters * 3

    @property
    def total_vault_images_per_character(self) -> int:
        return self.vault_lane.total_images * 4  # 4 lanes

    @property
    def total_vault_images(self) -> int:
        return self.total_vault_images_per_character * self.total_characters

    @property
    def total_images(self) -> int:
        return self.total_master_images + self.total_vault_images


# Negative prompt constants — shared across all generations
UNIVERSAL_NEGATIVE = (
    "bad quality, worst quality, worst detail, sketch, censor, "
    "bad hands, bad anatomy, deformed, artist name, watermark, "
    "signature, patreon, twitter username"
)

# Quality prefix for NoobAI/FabricatedXL
QUALITY_PREFIX = "masterpiece, best quality, amazing quality, absurdres"

# Lane-specific negative additions
LANE_NEGATIVES = {
    ContentLane.SFW: "nsfw, nude, explicit, revealing, cleavage, underwear",
    ContentLane.SUGGESTIVE: "nsfw, nude, explicit, nipples, genitalia",
    ContentLane.SPICY: "explicit genitalia, penetration",
    ContentLane.NSFW: "",
}
