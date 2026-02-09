#!/usr/bin/env bash
# ============================================================================
# Model Setup Script for AI Influencer Pipeline
# ============================================================================
# Downloads all required models for the ComfyUI workflow.
# Run this ONCE before starting the pipeline.
#
# Usage: bash docker/setup_models.sh /path/to/comfyui/models
# ============================================================================

set -euo pipefail

MODELS_DIR="${1:-/comfyui/models}"

echo "============================================"
echo "AI Influencer Pipeline — Model Setup"
echo "Target directory: $MODELS_DIR"
echo "============================================"

# Helper function
download_if_missing() {
    local url="$1"
    local dest="$2"
    local dir
    dir=$(dirname "$dest")
    mkdir -p "$dir"
    if [ ! -f "$dest" ]; then
        echo "Downloading: $(basename "$dest")"
        wget -q --show-progress -O "$dest" "$url"
    else
        echo "Already exists: $(basename "$dest")"
    fi
}

echo ""
echo "--- Required Custom Nodes ---"
echo "Ensure these ComfyUI custom nodes are installed:"
echo "  1. ComfyUI_IPAdapter_plus (cubiq)"
echo "  2. ComfyUI-Impact-Pack (ltdrdata)"
echo "  3. ComfyUI-Impact-Subpack (ltdrdata)"
echo "  4. comfyui_controlnet_aux (Fannovel16)"
echo "  5. ComfyUI_UltimateSDUpscale (ssitu)"
echo "  6. ComfyUI-Easy-Use (yolain)"
echo "  7. ComfyUI-Custom-Scripts (pythongosssss)"
echo "  8. ComfyUI-FBCNN (Miosp)"
echo "  9. ComfyUI-Image-Saver (alexopus)"
echo " 10. rgthree-comfy (rgthree)"
echo " 11. ComfyUI-KJNodes (kijai)"
echo " 12. z-tipo-extension (KohakuBlueleaf)"
echo " 13. comfyui-lora-manager"
echo " 14. comfyui-ppm (AttentionCouplePPM)"
echo ""

echo "--- Checkpoint ---"
echo "Required: fabricatedXL_v70.safetensors"
echo "Source: https://civitai.com/models/633354"
echo "Place in: $MODELS_DIR/checkpoints/"
echo ""

echo "--- IP-Adapter Models ---"
echo "Required models (place in $MODELS_DIR/ipadapter/):"
echo "  - noobIPAMARK1_mark1.safetensors"
echo "  - ip-adapter-faceid-plusv2_sdxl.bin"
echo ""

echo "--- CLIP Vision Models ---"
echo "Required (place in $MODELS_DIR/clip_vision/):"
echo "  - clip_vision_g.safetensors"
echo "  - CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors"
echo "  - CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
echo ""

echo "--- ControlNet Models ---"
echo "Required (place in $MODELS_DIR/controlnet/):"
echo "  - SDXL/control-lora-canny-rank256.safetensors"
echo "  - noobaiXLControlnet_openposeModel.safetensors"
echo ""

echo "--- Upscale Models ---"
echo "Required (place in $MODELS_DIR/upscale_models/):"
echo "  - 4x_foolhardy_Remacri.pth"
echo ""

echo "--- Detection Models ---"
echo "Required (place in $MODELS_DIR/ultralytics/):"
echo "  - bbox/hand_yolov9c.pt"
echo "  - bbox/face_yolov9c.pt"
echo "  - bbox/Eyeful_v2-Individual.pt"
echo "  - segm/yolo11m-seg.pt"
echo "  - segm/ntd11_anime_nsfw_segm_v5-variant1.pt"
echo ""

echo "--- SAM Model ---"
echo "Required (place in $MODELS_DIR/sams/):"
echo "  - sam_vit_b_01ec64.pth"
echo ""

echo "--- InsightFace ---"
echo "Required for IP-Adapter FaceID. Install via:"
echo "  pip install insightface onnxruntime-gpu"
echo ""

echo "============================================"
echo "Verify all models are in place, then start"
echo "the pipeline with: docker compose up -d"
echo "============================================"
