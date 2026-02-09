"""Workflow 1: Master Image Generation Pipeline.

Generates 3 canonical reference images per character (63 total for 21 characters):
    1. Portrait — Close-up headshot establishing face identity
    2. Three-Quarter — Mid-body shot establishing build and style
    3. Full Body — Complete look establishing height and proportions

These master images serve as:
    - IP-Adapter FaceID reference for all subsequent vault generation
    - Character consistency anchors
    - Style/composition references
    - Quality benchmarks

Pipeline flow per character:
    1. Load character profile from YAML
    2. Build character-specific prompts via PromptEngine
    3. Enhance prompts with photorealism tokens
    4. Inject into workflow template
    5. Execute on ComfyUI
    6. Save master images to character directory
    7. Register master images as FaceID references for vault pipeline
"""

import json
import os
import logging
import time
from typing import Optional

from .config import (
    PipelineConfig, ContentLane, ImageType,
    MASTER_IMAGE_RESOLUTIONS, RESOLUTIONS,
)
from .comfyui_client import ComfyUIClient
from .workflow_builder import WorkflowBuilder
from .bundle_manager import BundleManager

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.engine import PromptEngine, CharacterData
from prompts.enhancer import PromptEnhancer

logger = logging.getLogger(__name__)


class MasterPipeline:
    """Orchestrates master image generation for all characters.

    This is Workflow 1 — run this FIRST before the vault pipeline.
    The master images produced here become the FaceID references
    for character consistency in Workflow 2.
    """

    MASTER_TYPES = ["portrait", "three_quarter", "full_body"]

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        base_workflow_path: str = "workflows/base_workflow.json",
        characters_dir: str = "characters/roster",
    ):
        self.config = config or PipelineConfig()
        self.client = ComfyUIClient(self.config)
        self.builder = WorkflowBuilder(base_workflow_path, self.config)
        self.enhancer = PromptEnhancer()
        self.characters_dir = characters_dir

        # Track generated master images for vault pipeline
        self.master_registry = {}

    def run_all(self, seed_base: int = 42) -> dict:
        """Run master image generation for ALL characters.

        Args:
            seed_base: Base seed (each character gets a deterministic offset)

        Returns:
            Registry dict mapping character_id -> master image paths
        """
        characters = CharacterData.load_roster(self.characters_dir)
        logger.info(f"Loaded {len(characters)} characters from {self.characters_dir}")

        total = len(characters) * 3
        completed = 0

        for char_id, char_data in characters.items():
            try:
                result = self.run_character(char_data, seed_base)
                self.master_registry[char_id] = result
                completed += 3
                logger.info(
                    f"[{completed}/{total}] Completed master images for {char_id}"
                )
            except Exception as e:
                logger.error(f"Failed master images for {char_id}: {e}")
                continue

        # Save registry for vault pipeline
        registry_path = os.path.join(
            self.config.output.base_dir, "master_registry.json"
        )
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        with open(registry_path, "w") as f:
            json.dump(self.master_registry, f, indent=2)

        logger.info(
            f"Master pipeline complete: {completed}/{total} images generated"
        )
        return self.master_registry

    def run_character(
        self,
        character: CharacterData,
        seed_base: int = 42,
    ) -> dict:
        """Generate all 3 master images for a single character.

        Returns dict with paths to the 3 master images.
        """
        char_id = character.character_id
        engine = PromptEngine(character)

        logger.info(f"Generating master images for: {char_id}")

        results = {}
        output_dir = self.config.output.master_dir(char_id)
        os.makedirs(output_dir, exist_ok=True)

        # Get existing reference images if available
        faceid_ref = character.get("generation.faceid_reference", "")
        style_ref = character.get("generation.style_reference", "")

        for i, master_type in enumerate(self.MASTER_TYPES):
            seed = seed_base + hash(f"{char_id}_{master_type}") % 100000

            # Get resolution for this master type
            img_type = getattr(ImageType, f"MASTER_{master_type.upper()}")
            resolution = MASTER_IMAGE_RESOLUTIONS[img_type]

            # Build prompt
            positive, negative = engine.build_master_prompt(master_type)

            # Enhance with photorealism tokens
            positive = self.enhancer.enhance(
                positive,
                shot_type=master_type,
                lane=ContentLane.SFW,
                realism_level=0.9,  # High realism for master images
                seed=seed,
            )
            negative = self.enhancer.enhance_negative(negative)

            logger.info(
                f"  [{i+1}/3] {master_type}: {resolution.width}x{resolution.height}"
            )
            logger.debug(f"  Positive: {positive[:120]}...")

            # Build workflow
            workflow = self.builder.build_master_workflow(
                positive_prompt=positive,
                negative_prompt=negative,
                width=resolution.width,
                height=resolution.height,
                seed=seed,
                character_id=char_id,
                master_type=master_type,
                faceid_image=faceid_ref,
                style_image=style_ref,
            )

            # Execute
            try:
                exec_result = self.client.execute_workflow(
                    workflow,
                    timeout=600,
                    progress_callback=self._progress_callback,
                )

                # Save images
                saved = self.client.save_images(exec_result, output_dir)
                results[master_type] = {
                    "paths": saved,
                    "seed": seed,
                    "resolution": f"{resolution.width}x{resolution.height}",
                    "prompt": positive,
                }

                # After first master image, use it as FaceID ref for subsequent
                if saved and not faceid_ref:
                    faceid_ref = os.path.basename(saved[0])
                    logger.info(f"  Using {faceid_ref} as FaceID reference")

            except Exception as e:
                logger.error(f"  Failed {master_type} for {char_id}: {e}")
                results[master_type] = {"error": str(e)}

        return results

    def run_single(
        self,
        character_id: str,
        master_type: str = "portrait",
        seed: int = 42,
    ) -> dict:
        """Generate a single master image (for testing/iteration)."""
        characters = CharacterData.load_roster(self.characters_dir)
        if character_id not in characters:
            raise ValueError(f"Character {character_id} not found in roster")

        character = characters[character_id]
        engine = PromptEngine(character)

        img_type = getattr(ImageType, f"MASTER_{master_type.upper()}")
        resolution = MASTER_IMAGE_RESOLUTIONS[img_type]

        positive, negative = engine.build_master_prompt(master_type)
        positive = self.enhancer.enhance(
            positive, shot_type=master_type,
            lane=ContentLane.SFW, realism_level=0.9, seed=seed,
        )
        negative = self.enhancer.enhance_negative(negative)

        workflow = self.builder.build_master_workflow(
            positive_prompt=positive,
            negative_prompt=negative,
            width=resolution.width,
            height=resolution.height,
            seed=seed,
            character_id=character_id,
            master_type=master_type,
            faceid_image=character.get("generation.faceid_reference", ""),
            style_image=character.get("generation.style_reference", ""),
        )

        result = self.client.execute_workflow(workflow, timeout=600)
        output_dir = self.config.output.master_dir(character_id)
        saved = self.client.save_images(result, output_dir)

        return {
            "paths": saved,
            "seed": seed,
            "positive_prompt": positive,
            "negative_prompt": negative,
        }

    @staticmethod
    def _progress_callback(current: int, total: int):
        """Log generation progress."""
        pct = (current / max(total, 1)) * 100
        logger.info(f"  Progress: {current}/{total} ({pct:.0f}%)")
