"""Workflow builder — transforms the base ComfyUI workflow for each generation task.

This module takes the reference workflow JSON (from the UI export) and:
1. Converts it to ComfyUI API format
2. Injects character-specific parameters (prompts, LoRAs, reference images)
3. Sets resolution, seed, sampler parameters
4. Enables/disables detailer nodes based on content lane
5. Configures IP-Adapter with character reference images

NODE ID REFERENCE (from the base workflow):
    46  - POSITIVE wildcard processor (main prompt input)
    48  - NEGATIVE wildcard processor
    49  - Input Parameters (steps, cfg, sampler, scheduler, denoise)
    58  - Width
    61  - Height
    65  - Batch Size
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
    10  - Hand detector provider
    11  - Body detector provider
    12  - NSFW detector provider
    13  - Face detector provider
    14  - Eyes detector provider

    -- IP-Adapter --
    158 - Load FaceID Image
    149 - Load Advanced Image
    100 - Load Style Image
    101 - Load Composition Image
    103 - Load ClipVision Image

    -- ControlNet --
    105 - Load OpenPose Image
    107 - Load ControlNet Image

    -- Image Saver path --
    109 - Primary saver (node mode 0 = active)
    110 - Upscaled saver (node mode 4 = bypassed by default)
"""

import copy
import json
import os
import random
from typing import Optional

from .config import (
    PipelineConfig, ContentLane, ImageType,
    RESOLUTIONS, MASTER_IMAGE_RESOLUTIONS,
)


class WorkflowBuilder:
    """Builds ComfyUI API-format workflows from the base workflow template.

    The base workflow is loaded once, then cloned and modified for each
    generation task. This ensures we never corrupt the template.
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

    # Mode values in ComfyUI
    MODE_ACTIVE = 0
    MODE_BYPASSED = 4

    def __init__(self, base_workflow_path: str,
                 config: Optional[PipelineConfig] = None):
        """Initialize with the base workflow JSON file.

        Args:
            base_workflow_path: Path to the base workflow JSON
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()

        with open(base_workflow_path, "r") as f:
            raw = json.load(f)

        # Convert UI format to API format
        self.base_workflow = self._convert_to_api_format(raw)

    def _convert_to_api_format(self, ui_workflow: dict) -> dict:
        """Convert a ComfyUI UI-exported workflow to API format.

        The UI format has a 'nodes' array with positions and visual data.
        The API format is a flat dict keyed by node ID with just the
        functional data (class_type, inputs).
        """
        api_workflow = {}

        nodes = ui_workflow.get("nodes", [])
        links = ui_workflow.get("links", [])

        # Build link lookup: link_id -> (source_node, source_slot)
        link_map = {}
        for link in links:
            # link format: [link_id, source_node, source_slot, target_node, target_slot, type]
            link_id = link[0]
            source_node = str(link[1])
            source_slot = link[2]
            link_map[link_id] = (source_node, source_slot)

        for node in nodes:
            node_id = str(node["id"])
            class_type = node.get("type", "")

            if not class_type:
                continue

            api_node = {
                "class_type": class_type,
                "inputs": {},
                "_meta": {
                    "title": node.get("title", class_type),
                    "mode": node.get("mode", 0),
                },
            }

            # Process widget values
            widgets = node.get("widgets_values", [])
            inputs_spec = node.get("inputs", [])
            outputs_spec = node.get("outputs", [])

            # Map widget values to input names
            # This is complex because ComfyUI's widget mapping varies by node type
            # We store raw widgets and resolve them during parameter injection
            api_node["_widgets_values"] = widgets
            api_node["_mode"] = node.get("mode", 0)

            # Process linked inputs
            for inp in inputs_spec:
                inp_name = inp.get("name", "")
                link_id = inp.get("link")
                if link_id is not None and link_id in link_map:
                    source_node, source_slot = link_map[link_id]
                    api_node["inputs"][inp_name] = [source_node, source_slot]
                elif "widget" in inp and inp.get("link") is None:
                    # Widget input with no link — use widget value
                    pass

            api_workflow[node_id] = api_node

        return api_workflow

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

        Master images use full quality settings:
        - All detailers enabled (except NSFW)
        - IP-Adapter FaceID for consistency
        - Higher step count
        - No batch (1 image at a time for max quality)

        Args:
            positive_prompt: Complete positive prompt from PromptEngine
            negative_prompt: Complete negative prompt
            width: Image width
            height: Image height
            seed: Generation seed
            character_id: For file naming
            master_type: "portrait", "three_quarter", "full_body"
            faceid_image: FaceID reference image filename
            style_image: Style reference image filename
        """
        wf = copy.deepcopy(self.base_workflow)

        # Set prompts
        self._set_prompt(wf, positive_prompt, negative_prompt)

        # Set resolution
        self._set_resolution(wf, width, height)

        # Set seed
        self._set_seed(wf, seed)

        # Set batch size = 1 for master images
        self._set_batch_size(wf, 1)

        # Set sampler params (higher quality for masters)
        self._set_sampler_params(
            wf,
            steps=self.config.sampler.steps,
            cfg=self.config.sampler.cfg,
        )

        # Enable detailers for master images (skip NSFW detailer)
        self._configure_detailers(wf, lane=ContentLane.SFW)

        # Set IP-Adapter references if provided
        if faceid_image:
            self._set_reference_image(wf, "load_faceid_image", faceid_image)
        if style_image:
            self._set_reference_image(wf, "load_style_image", style_image)
            self._set_reference_image(wf, "load_composition_image", style_image)
            self._set_reference_image(wf, "load_clipvision_image", style_image)

        # Set output path
        self._set_output_path(wf, character_id, f"master/{master_type}")

        return wf

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

        Vault images are generated at scale with lane-specific settings:
        - Detailers configured per lane
        - Batch size can be >1 for throughput
        - IP-Adapter FaceID for character consistency

        Args:
            positive_prompt: Complete positive prompt
            negative_prompt: Complete negative prompt
            width: Image width
            height: Image height
            seed: Generation seed
            character_id: For file naming
            lane: Content lane (SFW/Suggestive/Spicy/NSFW)
            output_subdir: Subdirectory under character vault
            faceid_image: FaceID reference image
            style_image: Style reference image
            batch_size: Number of images per batch
        """
        wf = copy.deepcopy(self.base_workflow)

        # Set prompts
        self._set_prompt(wf, positive_prompt, negative_prompt)

        # Set resolution
        self._set_resolution(wf, width, height)

        # Set seed
        self._set_seed(wf, seed)

        # Set batch size
        self._set_batch_size(wf, batch_size)

        # Set sampler params
        self._set_sampler_params(
            wf,
            steps=self.config.sampler.steps,
            cfg=self.config.sampler.cfg,
        )

        # Configure detailers for this lane
        self._configure_detailers(wf, lane)

        # Set IP-Adapter references
        if faceid_image:
            self._set_reference_image(wf, "load_faceid_image", faceid_image)
        if style_image:
            self._set_reference_image(wf, "load_style_image", style_image)
            self._set_reference_image(wf, "load_composition_image", style_image)
            self._set_reference_image(wf, "load_clipvision_image", style_image)

        # Set output path
        self._set_output_path(
            wf, character_id, f"vault/{lane.value}/{output_subdir}"
        )

        return wf

    # -------------------------------------------------------------------------
    # PARAMETER INJECTION METHODS
    # -------------------------------------------------------------------------

    def _set_prompt(self, wf: dict, positive: str, negative: str):
        """Inject positive and negative prompts into the workflow."""
        pos_node = wf.get(self.NODE_IDS["positive_prompt"], {})
        if pos_node:
            widgets = pos_node.get("_widgets_values", [])
            if len(widgets) >= 2:
                widgets[0] = positive  # Wildcard text
                widgets[1] = positive  # Display text

        neg_node = wf.get(self.NODE_IDS["negative_prompt"], {})
        if neg_node:
            widgets = neg_node.get("_widgets_values", [])
            if len(widgets) >= 2:
                widgets[0] = negative
                widgets[1] = negative

    def _set_resolution(self, wf: dict, width: int, height: int):
        """Set output resolution."""
        w_node = wf.get(self.NODE_IDS["width"], {})
        if w_node:
            widgets = w_node.get("_widgets_values", [])
            if widgets:
                widgets[0] = width

        h_node = wf.get(self.NODE_IDS["height"], {})
        if h_node:
            widgets = h_node.get("_widgets_values", [])
            if widgets:
                widgets[0] = height

    def _set_seed(self, wf: dict, seed: int):
        """Set generation seed."""
        seed_node = wf.get(self.NODE_IDS["seed"], {})
        if seed_node:
            widgets = seed_node.get("_widgets_values", [])
            if widgets:
                widgets[0] = seed

    def _set_batch_size(self, wf: dict, batch_size: int):
        """Set batch size."""
        bs_node = wf.get(self.NODE_IDS["batch_size"], {})
        if bs_node:
            widgets = bs_node.get("_widgets_values", [])
            if widgets:
                widgets[0] = batch_size

    def _set_sampler_params(self, wf: dict, steps: int = 28,
                            cfg: float = 6.0):
        """Set sampler parameters."""
        params_node = wf.get(self.NODE_IDS["input_params"], {})
        if params_node:
            widgets = params_node.get("_widgets_values", [])
            # Widget order: seed, control_after_gen, steps, cfg, sampler, scheduler, denoise
            if len(widgets) >= 7:
                widgets[2] = steps  # steps
                widgets[3] = cfg    # cfg

    def _configure_detailers(self, wf: dict, lane: ContentLane):
        """Enable/disable detailers based on content lane."""
        enabled = self.config.detailer.lane_detailers.get(lane, [])

        detailer_map = {
            "hand": ("hand_detailer", "hand_detector"),
            "body": ("body_detailer", "body_detector"),
            "nsfw": ("nsfw_detailer", "nsfw_detector"),
            "face": ("face_detailer", "face_detector"),
            "eyes": ("eyes_detailer", "eyes_detector"),
        }

        for name, (detailer_key, detector_key) in detailer_map.items():
            mode = self.MODE_ACTIVE if name in enabled else self.MODE_BYPASSED
            detailer_node = wf.get(self.NODE_IDS[detailer_key], {})
            if detailer_node:
                detailer_node["_mode"] = mode
            detector_node = wf.get(self.NODE_IDS[detector_key], {})
            if detector_node:
                detector_node["_mode"] = mode

    def _set_reference_image(self, wf: dict, node_key: str, filename: str):
        """Set a reference image on a LoadImage node."""
        node_id = self.NODE_IDS.get(node_key, "")
        node = wf.get(node_id, {})
        if node:
            node["_mode"] = self.MODE_ACTIVE
            widgets = node.get("_widgets_values", [])
            if widgets:
                widgets[0] = filename

    def _set_output_path(self, wf: dict, character_id: str, subdir: str):
        """Set the output save path via Image Saver nodes."""
        for saver_key in ("image_saver_main", "image_saver_upscaled"):
            node = wf.get(self.NODE_IDS[saver_key], {})
            if node:
                widgets = node.get("_widgets_values", [])
                if len(widgets) >= 2:
                    # filename_prefix format
                    widgets[0] = f"%time_{character_id}_%seed"
                    # subfolder
                    widgets[1] = f"{character_id}/{subdir}"

    # -------------------------------------------------------------------------
    # EXPORT
    # -------------------------------------------------------------------------

    def export_workflow(self, workflow: dict, output_path: str):
        """Export a workflow to JSON file for debugging or manual execution."""
        # Clean internal metadata before export
        clean = {}
        for node_id, node_data in workflow.items():
            clean_node = {k: v for k, v in node_data.items()
                         if not k.startswith("_")}
            clean[node_id] = clean_node

        with open(output_path, "w") as f:
            json.dump(clean, f, indent=2)
