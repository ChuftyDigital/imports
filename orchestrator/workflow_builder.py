"""Workflow builder — transforms the base ComfyUI workflow for each generation task.

Takes the UI-exported workflow JSON and converts it to proper ComfyUI API
format using /object_info for accurate widget-to-input-name mapping. Then
injects character-specific parameters for each generation.

Runtime flow:
    1. Load base UI workflow from JSON
    2. On first use, fetch /object_info from ComfyUI (cached locally)
    3. Convert UI format → API format using node definitions
    4. Clone + inject parameters per generation task

NODE ID REFERENCE (from the base workflow):
    46  - POSITIVE wildcard processor (main prompt input)
    48  - NEGATIVE wildcard processor
    49  - Input Parameters (steps, cfg, sampler, scheduler, denoise)
    58  - Width (easy int)
    61  - Height (easy int)
    65  - Batch Size (easy int)
    73  - CheckpointLoaderSimple (fabricatedXL_v70)
    87  - Seed (rgthree)
    96  - KSampler (primary)
    59  - KSampler (secondary/refinement)
    109 - Image Saver (main output)
    110 - Image Saver (upscaled output)

    -- Detailer Chain --
    28  - HandDetailer (FaceDetailerPipe)
    29  - BodyDetailer (FaceDetailerPipe)
    30  - NSFWDetailer (FaceDetailerPipe)
    31  - FaceDetailer (FaceDetailerPipe)
    32  - EyesDetailer (FaceDetailerPipe)

    -- IP-Adapter --
    158 - Load FaceID Image
    149 - Load Advanced Image
    100 - Load Style Image
    101 - Load Composition Image
    103 - Load ClipVision Image

    -- ControlNet --
    105 - Load OpenPose Image
    107 - Load ControlNet Image
"""

import copy
import json
import logging
import os
import random
import urllib.request
from typing import Optional

from .config import (
    PipelineConfig, ContentLane, ImageType,
    RESOLUTIONS, MASTER_IMAGE_RESOLUTIONS,
)

logger = logging.getLogger(__name__)


# Hardcoded widget-to-input mappings for node types where widgets_values
# contains UI-only fields (like control_after_generate) that must be skipped.
# Format: { "NodeType": [("input_name", value_index), ...] }
# A None input_name means "skip this widget index" (UI-only).
WIDGET_MAPS = {
    "KSampler": [
        ("seed", 0), (None, 1),  # control_after_generate
        ("steps", 2), ("cfg", 3), ("sampler_name", 4),
        ("scheduler", 5), ("denoise", 6),
    ],
    "KSamplerAdvanced": [
        ("add_noise", 0), ("noise_seed", 1), (None, 2),  # control_after_generate
        ("steps", 3), ("cfg", 4), ("sampler_name", 5),
        ("scheduler", 6), ("start_at_step", 7), ("end_at_step", 8),
        ("return_with_leftover_noise", 9),
    ],
    "ImpactWildcardProcessor": [
        ("wildcard_text", 0), ("populated_text", 1), ("mode", 2),
        ("seed", 3), (None, 4),  # control_after_generate
        ("Select to add Wildcard", 5),
    ],
    "ImpactWildcardEncode": [
        ("wildcard_text", 0), ("populated_text", 1), ("mode", 2),
        ("seed", 3), (None, 4),  # control_after_generate
        ("Select to add Wildcard", 5),
    ],
    "Input Parameters (Image Saver)": [
        ("seed", 0), (None, 1),  # control_after_generate
        ("steps", 2), ("cfg", 3), ("sampler_name", 4),
        ("scheduler", 5), ("denoise", 6),
    ],
    "Seed (rgthree)": [
        ("seed", 0),
    ],
    "FaceDetailerPipe": [
        ("guide_size", 0), ("guide_size_for", 1), ("max_size", 2),
        ("seed", 3), (None, 4),  # control_after_generate
        ("steps", 5), ("cfg", 6), ("sampler_name", 7),
        ("scheduler", 8), ("denoise", 9),
        ("feather", 10), ("noise_mask", 11), ("force_inpaint", 12),
        ("bbox_threshold", 13), ("bbox_dilation", 14),
        ("bbox_crop_factor", 15), ("sam_detection_hint", 16),
        ("sam_dilation", 17), ("sam_threshold", 18),
        ("sam_bbox_expansion", 19), ("sam_mask_hint_threshold", 20),
        ("sam_mask_hint_use_negative", 21), ("drop_size", 22),
        ("refiner_ratio", 23), ("cycle", 24),
        ("inpaint_model", 25), ("noise_mask_feather", 26),
        ("tiled_encode", 27), ("tiled_decode", 28),
    ],
    "Image Saver": [
        ("filename_prefix", 0), ("subfolder", 1), ("image_format", 2),
        ("quality", 3), ("counter", 4), ("time_format", 5),
        ("dpi", 6), ("save_metadata", 7), ("extra_metadata", 8),
    ],
    "LoadImage": [
        ("image", 0), ("upload", 1),
    ],
    "CheckpointLoaderSimple": [
        ("ckpt_name", 0),
    ],
    "easy int": [
        ("value", 0),
    ],
    "easy float": [
        ("value", 0),
    ],
    "UltralyticsDetectorProvider": [
        ("model_name", 0),
    ],
    "UltimateSDUpscale": [
        ("upscale_by", 0), ("seed", 1), (None, 2),  # control_after_generate
        ("steps", 3), ("cfg", 4), ("sampler_name", 5),
        ("scheduler", 6), ("denoise", 7), ("mode_type", 8),
        ("tile_width", 9), ("tile_height", 10), ("mask_blur", 11),
        ("tile_padding", 12), ("seam_fix_mode", 13),
        ("seam_fix_denoise", 14), ("seam_fix_width", 15),
        ("seam_fix_mask_blur", 16), ("seam_fix_padding", 17),
        ("force_uniform_tiles", 18), ("tiled_decode", 19),
    ],
    "JPEG artifacts removal FBCNN": [
        ("state", 0), ("quality", 1),
    ],
    "CLIPVisionLoader": [
        ("clip_name", 0),
    ],
    "IPAdapterModelLoader": [
        ("ipadapter_file", 0),
    ],
    "ControlNetLoader": [
        ("control_net_name", 0),
    ],
    "VAELoader": [
        ("vae_name", 0),
    ],
    "CLIPSetLastLayer": [
        ("stop_at_clip_layer", 0),
    ],
}


class WorkflowBuilder:
    """Builds ComfyUI API-format workflows from the base UI workflow.

    Converts the UI-exported workflow (LiteGraph format) to proper ComfyUI
    API format, then provides methods to inject parameters for each
    generation task.

    The conversion uses /object_info from ComfyUI when available (cached
    locally), falling back to hardcoded widget maps for known node types.
    """

    # Node IDs mapped from the base workflow
    NODE_IDS = {
        "positive_prompt": "46",
        "negative_prompt": "48",
        "input_params": "49",
        "width": "58",
        "height": "61",
        "batch_size": "65",
        "checkpoint": "73",
        "seed": "87",
        "ksampler_primary": "96",
        "ksampler_secondary": "59",
        "image_saver_main": "109",
        "image_saver_upscaled": "110",
        # Detailers
        "hand_detailer": "28",
        "body_detailer": "29",
        "nsfw_detailer": "30",
        "face_detailer": "31",
        "eyes_detailer": "32",
        "hand_detector": "10",
        "body_detector": "11",
        "nsfw_detector": "12",
        "face_detector": "13",
        "eyes_detector": "14",
        # IP-Adapter images
        "load_faceid_image": "158",
        "load_advanced_image": "149",
        "load_style_image": "100",
        "load_composition_image": "101",
        "load_clipvision_image": "103",
        # ControlNet images
        "load_openpose_image": "105",
        "load_controlnet_image": "107",
        # Upscaler
        "ultimate_upscale": "112",
        "upscale_by": "120",
        # Compression removal
        "fbcnn_main": "114",
        "fbcnn_upscaled": "117",
    }

    # ComfyUI node mode values
    MODE_ACTIVE = 0
    MODE_MUTED = 2
    MODE_BYPASSED = 4

    CACHE_PATH = "workflows/.node_info_cache.json"

    def __init__(self, base_workflow_path: str,
                 config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        with open(base_workflow_path, "r") as f:
            self.ui_workflow = json.load(f)

        # Node info from /object_info (populated by fetch_node_info or cache)
        self._node_info = None

        # Parse the UI workflow structure
        self._nodes = {str(n["id"]): n for n in self.ui_workflow.get("nodes", [])}
        self._build_link_map()

    def _build_link_map(self):
        """Build link lookup tables from the UI workflow."""
        self._link_map = {}  # link_id -> (source_node_id, source_slot)
        for link in self.ui_workflow.get("links", []):
            # link: [link_id, source_node, source_slot, target_node, target_slot, type]
            self._link_map[link[0]] = (str(link[1]), link[2])

    # -------------------------------------------------------------------------
    # NODE INFO / OBJECT_INFO
    # -------------------------------------------------------------------------

    def fetch_node_info(self, force: bool = False) -> dict:
        """Fetch /object_info from ComfyUI and cache locally.

        This provides the authoritative widget-to-input mapping for all
        node types. Called once, then cached.
        """
        cache_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            self.CACHE_PATH,
        )

        # Try cache first
        if not force and os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    self._node_info = json.load(f)
                logger.info(f"Loaded node info cache ({len(self._node_info)} types)")
                return self._node_info
            except (json.JSONDecodeError, IOError):
                pass

        # Fetch from ComfyUI
        url = f"{self.config.comfyui_url}/object_info"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                self._node_info = json.loads(resp.read().decode("utf-8"))

            # Cache it
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(self._node_info, f)
            logger.info(f"Fetched and cached node info ({len(self._node_info)} types)")
            return self._node_info
        except Exception as e:
            logger.warning(f"Could not fetch /object_info: {e}. Using hardcoded maps.")
            return {}

    def _get_widget_input_names(self, class_type: str) -> list:
        """Get ordered list of (input_name, is_ui_only) for a node's widgets.

        Uses /object_info if available, falls back to WIDGET_MAPS.
        """
        # Try /object_info first
        if self._node_info and class_type in self._node_info:
            info = self._node_info[class_type]
            required = info.get("input", {}).get("required", {})
            optional = info.get("input", {}).get("optional", {})
            # The order in required + optional matches widgets_values order
            names = list(required.keys()) + list(optional.keys())
            return names

        # Fall back to hardcoded maps
        if class_type in WIDGET_MAPS:
            return [entry[0] for entry in WIDGET_MAPS[class_type]]

        return []

    # -------------------------------------------------------------------------
    # UI → API CONVERSION
    # -------------------------------------------------------------------------

    def convert_to_api(self) -> dict:
        """Convert the UI-format workflow to ComfyUI API format.

        API format: { "node_id": { "class_type": ..., "inputs": {...} } }
        Where inputs are either literal values or [source_node_id, slot].
        """
        # Try to get node info for accurate conversion
        if self._node_info is None:
            self.fetch_node_info()

        api = {}

        for node_id, node in self._nodes.items():
            class_type = node.get("type", "")
            if not class_type or class_type == "Reroute":
                # Reroutes are handled by link resolution
                if class_type == "Reroute":
                    api[node_id] = self._convert_reroute(node)
                continue

            api_node = {
                "class_type": class_type,
                "inputs": {},
                "_meta": {"title": node.get("title", class_type)},
            }

            # 1. Get linked inputs from the node's input connectors
            linked_input_names = set()
            for inp in node.get("inputs", []):
                inp_name = inp.get("name", "")
                link_id = inp.get("link")
                if link_id is not None and link_id in self._link_map:
                    src_node, src_slot = self._link_map[link_id]
                    api_node["inputs"][inp_name] = [src_node, src_slot]
                    linked_input_names.add(inp_name)

            # 2. Map widget values to named inputs
            widgets = node.get("widgets_values", [])
            if widgets:
                self._map_widgets_to_inputs(
                    api_node, class_type, widgets, linked_input_names
                )

            # Store mode for detailer enable/disable
            api_node["_mode"] = node.get("mode", 0)

            api[node_id] = api_node

        return api

    def _convert_reroute(self, node: dict) -> dict:
        """Handle Reroute nodes (pass-through)."""
        api_node = {
            "class_type": node.get("type", "Reroute"),
            "inputs": {},
            "_meta": {"title": node.get("title", "Reroute")},
            "_mode": node.get("mode", 0),
        }
        for inp in node.get("inputs", []):
            link_id = inp.get("link")
            if link_id is not None and link_id in self._link_map:
                src_node, src_slot = self._link_map[link_id]
                api_node["inputs"][inp.get("name", "")] = [src_node, src_slot]
        return api_node

    def _map_widgets_to_inputs(self, api_node: dict, class_type: str,
                                widgets: list, linked_names: set):
        """Map widget values to named inputs, skipping UI-only widgets."""
        input_names = self._get_widget_input_names(class_type)

        if input_names:
            # We have a mapping — use it
            wi = 0  # widget index
            for name in input_names:
                if name is None:
                    # UI-only widget, skip
                    wi += 1
                    continue
                if name in linked_names:
                    # This input is connected via a link, skip widget value
                    # But still consume the widget slot if it exists
                    if wi < len(widgets) and name in WIDGET_MAPS.get(class_type, {}):
                        wi += 1
                    continue
                if wi < len(widgets):
                    api_node["inputs"][name] = widgets[wi]
                    wi += 1
        else:
            # No mapping available — store raw widgets for manual handling
            # This is a fallback for unknown node types
            api_node["_widgets_values"] = widgets
            logger.debug(f"No widget map for {class_type}, storing raw values")

    # -------------------------------------------------------------------------
    # WORKFLOW BUILDERS
    # -------------------------------------------------------------------------

    def build_master_workflow(
        self,
        positive_prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        character_id: str,
        master_type: str,
        faceid_image: str = "",
        style_image: str = "",
    ) -> dict:
        """Build a workflow for master image generation.

        Master images use full quality settings — all detailers enabled
        (except NSFW), IP-Adapter FaceID, batch_size=1.
        """
        wf = copy.deepcopy(self.convert_to_api())

        self._set_prompt(wf, positive_prompt, negative_prompt)
        self._set_resolution(wf, width, height)
        self._set_seed(wf, seed)
        self._set_batch_size(wf, 1)
        self._set_sampler_params(wf,
                                  steps=self.config.sampler.steps,
                                  cfg=self.config.sampler.cfg)
        self._configure_detailers(wf, lane=ContentLane.SFW)

        if faceid_image:
            self._set_reference_image(wf, "load_faceid_image", faceid_image)
        if style_image:
            self._set_reference_image(wf, "load_style_image", style_image)
            self._set_reference_image(wf, "load_composition_image", style_image)
            self._set_reference_image(wf, "load_clipvision_image", style_image)

        self._set_output_path(wf, character_id, f"master/{master_type}")

        return self._clean_for_api(wf)

    def build_vault_workflow(
        self,
        positive_prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        character_id: str,
        lane: ContentLane,
        output_subdir: str,
        faceid_image: str = "",
        style_image: str = "",
        batch_size: int = 1,
    ) -> dict:
        """Build a workflow for vault image generation.

        Lane-specific detailer config, IP-Adapter FaceID for consistency.
        """
        wf = copy.deepcopy(self.convert_to_api())

        self._set_prompt(wf, positive_prompt, negative_prompt)
        self._set_resolution(wf, width, height)
        self._set_seed(wf, seed)
        self._set_batch_size(wf, batch_size)
        self._set_sampler_params(wf,
                                  steps=self.config.sampler.steps,
                                  cfg=self.config.sampler.cfg)
        self._configure_detailers(wf, lane)

        if faceid_image:
            self._set_reference_image(wf, "load_faceid_image", faceid_image)
        if style_image:
            self._set_reference_image(wf, "load_style_image", style_image)
            self._set_reference_image(wf, "load_composition_image", style_image)
            self._set_reference_image(wf, "load_clipvision_image", style_image)

        self._set_output_path(
            wf, character_id, f"vault/{lane.value}/{output_subdir}"
        )

        return self._clean_for_api(wf)

    # -------------------------------------------------------------------------
    # PARAMETER INJECTION — all use named inputs
    # -------------------------------------------------------------------------

    def _set_input(self, wf: dict, node_id: str, input_name: str, value):
        """Set a named input on a node. The fundamental injection method."""
        node = wf.get(node_id)
        if node:
            node["inputs"][input_name] = value

    def _set_prompt(self, wf: dict, positive: str, negative: str):
        """Inject positive and negative prompts."""
        pos_id = self.NODE_IDS["positive_prompt"]
        neg_id = self.NODE_IDS["negative_prompt"]

        # ImpactWildcardProcessor takes wildcard_text + populated_text
        self._set_input(wf, pos_id, "wildcard_text", positive)
        self._set_input(wf, pos_id, "populated_text", positive)
        self._set_input(wf, neg_id, "wildcard_text", negative)
        self._set_input(wf, neg_id, "populated_text", negative)

    def _set_resolution(self, wf: dict, width: int, height: int):
        """Set output resolution via easy int nodes."""
        self._set_input(wf, self.NODE_IDS["width"], "value", width)
        self._set_input(wf, self.NODE_IDS["height"], "value", height)

    def _set_seed(self, wf: dict, seed: int):
        """Set generation seed."""
        self._set_input(wf, self.NODE_IDS["seed"], "seed", seed)

        # Also set seed on Input Parameters node
        self._set_input(wf, self.NODE_IDS["input_params"], "seed", seed)

        # Set seed on primary KSampler
        self._set_input(wf, self.NODE_IDS["ksampler_primary"], "seed", seed)

        # Secondary KSampler gets offset seed
        self._set_input(wf, self.NODE_IDS["ksampler_secondary"], "seed", seed + 1)

    def _set_batch_size(self, wf: dict, batch_size: int):
        """Set batch size."""
        self._set_input(wf, self.NODE_IDS["batch_size"], "value", batch_size)

    def _set_sampler_params(self, wf: dict, steps: int = 28,
                             cfg: float = 6.0):
        """Set sampler parameters on Input Parameters node and KSamplers."""
        ip_id = self.NODE_IDS["input_params"]
        self._set_input(wf, ip_id, "steps", steps)
        self._set_input(wf, ip_id, "cfg", cfg)
        self._set_input(wf, ip_id, "sampler_name",
                        self.config.sampler.sampler_name)
        self._set_input(wf, ip_id, "scheduler",
                        self.config.sampler.scheduler)
        self._set_input(wf, ip_id, "denoise", self.config.sampler.denoise)

        # Primary KSampler
        ks_id = self.NODE_IDS["ksampler_primary"]
        self._set_input(wf, ks_id, "steps", steps)
        self._set_input(wf, ks_id, "cfg", cfg)
        self._set_input(wf, ks_id, "sampler_name",
                        self.config.sampler.sampler_name)
        self._set_input(wf, ks_id, "scheduler",
                        self.config.sampler.scheduler)

    def _configure_detailers(self, wf: dict, lane: ContentLane):
        """Enable/disable detailers based on content lane."""
        enabled = self.config.detailer.lane_detailers.get(lane, [])

        detailer_pairs = {
            "hand": ("hand_detailer", "hand_detector"),
            "body": ("body_detailer", "body_detector"),
            "nsfw": ("nsfw_detailer", "nsfw_detector"),
            "face": ("face_detailer", "face_detector"),
            "eyes": ("eyes_detailer", "eyes_detector"),
        }

        # Set denoise per detailer
        denoise_map = {
            "hand": self.config.detailer.hand_denoise,
            "body": self.config.detailer.body_denoise,
            "nsfw": self.config.detailer.nsfw_denoise,
            "face": self.config.detailer.face_denoise,
            "eyes": self.config.detailer.eyes_denoise,
        }

        for name, (det_key, prov_key) in detailer_pairs.items():
            det_id = self.NODE_IDS[det_key]
            prov_id = self.NODE_IDS[prov_key]

            if name in enabled:
                # Active
                if det_id in wf:
                    wf[det_id]["_mode"] = self.MODE_ACTIVE
                    self._set_input(wf, det_id, "denoise", denoise_map[name])
                if prov_id in wf:
                    wf[prov_id]["_mode"] = self.MODE_ACTIVE
            else:
                # Bypassed
                if det_id in wf:
                    wf[det_id]["_mode"] = self.MODE_BYPASSED
                if prov_id in wf:
                    wf[prov_id]["_mode"] = self.MODE_BYPASSED

    def _set_reference_image(self, wf: dict, node_key: str, filename: str):
        """Set a reference image on a LoadImage node."""
        node_id = self.NODE_IDS.get(node_key, "")
        if node_id and node_id in wf:
            wf[node_id]["_mode"] = self.MODE_ACTIVE
            self._set_input(wf, node_id, "image", filename)

    def _set_output_path(self, wf: dict, character_id: str, subdir: str):
        """Set the output save path on Image Saver nodes."""
        for saver_key in ("image_saver_main", "image_saver_upscaled"):
            node_id = self.NODE_IDS[saver_key]
            if node_id in wf:
                self._set_input(wf, node_id, "filename_prefix",
                               f"%time_{character_id}_%seed")
                self._set_input(wf, node_id, "subfolder",
                               f"{character_id}/{subdir}")

    # -------------------------------------------------------------------------
    # EXPORT / CLEANUP
    # -------------------------------------------------------------------------

    def _clean_for_api(self, wf: dict) -> dict:
        """Strip internal metadata and prepare for ComfyUI /prompt submission."""
        clean = {}
        for node_id, node_data in wf.items():
            clean_node = {
                "class_type": node_data["class_type"],
                "inputs": dict(node_data.get("inputs", {})),
            }
            # Include _meta title if present
            if "_meta" in node_data:
                clean_node["_meta"] = node_data["_meta"]
            clean[node_id] = clean_node
        return clean

    def export_workflow(self, workflow: dict, output_path: str):
        """Export a workflow to JSON file for debugging."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(workflow, f, indent=2)
