"""Workflow 2: Image Vault Generation Pipeline.

Mass-produces the image vault for each character:
    4 Lanes: SFW, Suggestive, Spicy, NSFW
    Per lane (200 images):
        80 one-off images
        10 bundles × 3 images = 30 images
        10 bundles × 5 images = 50 images
        4 bundles × 10 images = 40 images

    Total per character: 800 images
    Total for 21 characters: 16,800 images

This pipeline REQUIRES master images to exist first (run Workflow 1).
It uses the master portrait as the FaceID reference to maintain
character consistency across all vault images.

Pipeline flow per character per lane:
    1. Load character profile
    2. Load master image as FaceID reference
    3. Generate one-off images (varied prompts, scenes, outfits)
    4. Generate bundled images (themed sets with shared context)
    5. Organize output into directory structure
"""

import json
import os
import logging
import time
from typing import Optional

from .config import (
    PipelineConfig, ContentLane, RESOLUTIONS,
)
from .comfyui_client import ComfyUIClient
from .workflow_builder import WorkflowBuilder
from .bundle_manager import BundleManager, ImageSpec, BundleSpec

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.engine import PromptEngine, CharacterData
from prompts.enhancer import PromptEnhancer

logger = logging.getLogger(__name__)


class VaultPipeline:
    """Orchestrates mass image vault generation for all characters.

    This is Workflow 2 — run AFTER the master pipeline (Workflow 1).
    Uses master images as FaceID anchors for character consistency.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        base_workflow_path: str = "workflows/base_workflow.json",
        characters_dir: str = "characters/roster",
        master_registry_path: str = "",
    ):
        self.config = config or PipelineConfig()
        self.client = ComfyUIClient(self.config)
        self.builder = WorkflowBuilder(base_workflow_path, self.config)
        self.enhancer = PromptEnhancer()
        self.bundle_mgr = BundleManager(self.config)
        self.characters_dir = characters_dir

        # Load master image registry
        self.master_registry = {}
        reg_path = master_registry_path or os.path.join(
            self.config.output.base_dir, "master_registry.json"
        )
        if os.path.exists(reg_path):
            with open(reg_path, "r") as f:
                self.master_registry = json.load(f)

    def _get_faceid_reference(self, character_id: str) -> str:
        """Get the FaceID reference image path for a character."""
        master = self.master_registry.get(character_id, {})
        portrait = master.get("portrait", {})
        paths = portrait.get("paths", [])
        if paths:
            return os.path.basename(paths[0])
        return ""

    def _get_style_reference(self, character_id: str) -> str:
        """Get style reference from master three-quarter image."""
        master = self.master_registry.get(character_id, {})
        tq = master.get("three_quarter", {})
        paths = tq.get("paths", [])
        if paths:
            return os.path.basename(paths[0])
        return ""

    # -------------------------------------------------------------------------
    # FULL ROSTER RUN
    # -------------------------------------------------------------------------

    def run_all(self, seed_base: int = 12345) -> dict:
        """Run vault generation for ALL characters, ALL lanes.

        This is the big one: 21 characters × 4 lanes × 200 images = 16,800 images.
        """
        characters = CharacterData.load_roster(self.characters_dir)
        logger.info(f"Starting vault generation for {len(characters)} characters")

        total_chars = len(characters)
        results = {}

        for idx, (char_id, char_data) in enumerate(characters.items()):
            logger.info(
                f"[{idx+1}/{total_chars}] Processing character: {char_id}"
            )
            try:
                result = self.run_character(char_data, seed_base)
                results[char_id] = result
            except Exception as e:
                logger.error(f"Failed vault for {char_id}: {e}")
                results[char_id] = {"error": str(e)}

        # Save vault manifest
        manifest_path = os.path.join(
            self.config.output.base_dir, "vault_manifest.json"
        )
        with open(manifest_path, "w") as f:
            json.dump(results, f, indent=2)

        total_generated = sum(
            r.get("total_generated", 0) for r in results.values()
            if isinstance(r, dict) and "error" not in r
        )
        logger.info(f"Vault pipeline complete: {total_generated} images generated")
        return results

    def run_character(
        self,
        character: CharacterData,
        seed_base: int = 12345,
    ) -> dict:
        """Run vault generation for a single character, all 4 lanes."""
        char_id = character.character_id
        engine = PromptEngine(character)

        faceid_ref = self._get_faceid_reference(char_id)
        style_ref = self._get_style_reference(char_id)

        if not faceid_ref:
            logger.warning(
                f"No FaceID reference for {char_id} — "
                "character consistency may be reduced"
            )

        results = {}
        total_generated = 0

        for lane in ContentLane:
            logger.info(f"  Lane: {lane.value}")
            try:
                lane_result = self.run_lane(
                    character, engine, lane, seed_base,
                    faceid_ref, style_ref,
                )
                results[lane.value] = lane_result
                total_generated += lane_result.get("generated", 0)
            except Exception as e:
                logger.error(f"  Failed lane {lane.value} for {char_id}: {e}")
                results[lane.value] = {"error": str(e)}

        results["total_generated"] = total_generated
        return results

    def run_lane(
        self,
        character: CharacterData,
        engine: PromptEngine,
        lane: ContentLane,
        seed_base: int,
        faceid_ref: str,
        style_ref: str,
    ) -> dict:
        """Run vault generation for one lane of one character."""
        char_id = character.character_id
        lane_seed = seed_base + hash(f"{char_id}_{lane.value}") % 100000

        # Generate the complete plan for this lane
        plan = self.bundle_mgr.generate_lane_plan(char_id, lane, lane_seed)

        generated = 0
        errors = 0

        # Process one-off images
        logger.info(f"    Generating {len(plan['one_offs'])} one-off images...")
        for img_spec in plan["one_offs"]:
            try:
                self._generate_single_image(
                    engine, img_spec, lane, char_id,
                    faceid_ref, style_ref,
                    output_subdir="one_offs",
                )
                generated += 1
            except Exception as e:
                logger.error(f"    One-off {img_spec.image_id} failed: {e}")
                errors += 1

            if generated % 20 == 0:
                logger.info(f"    Progress: {generated} images generated")

        # Process bundles
        for bundle_type, bundles in [
            ("bundle_3", plan["bundles_3"]),
            ("bundle_5", plan["bundles_5"]),
            ("bundle_10", plan["bundles_10"]),
        ]:
            logger.info(
                f"    Generating {len(bundles)} {bundle_type} bundles..."
            )
            for bundle in bundles:
                try:
                    count = self._generate_bundle(
                        engine, bundle, lane, char_id,
                        faceid_ref, style_ref,
                    )
                    generated += count
                except Exception as e:
                    logger.error(
                        f"    Bundle {bundle.bundle_id} failed: {e}"
                    )
                    errors += bundle.size

        logger.info(
            f"    Lane {lane.value} complete: "
            f"{generated} generated, {errors} errors"
        )
        return {"generated": generated, "errors": errors, "plan_total": plan["total_images"]}

    # -------------------------------------------------------------------------
    # IMAGE GENERATION
    # -------------------------------------------------------------------------

    def _generate_single_image(
        self,
        engine: PromptEngine,
        img_spec: ImageSpec,
        lane: ContentLane,
        character_id: str,
        faceid_ref: str,
        style_ref: str,
        output_subdir: str = "",
        bundle_tokens: str = "",
    ) -> str:
        """Generate a single vault image.

        Returns the path to the saved image.
        """
        # Build prompt
        positive, negative = engine.build_vault_prompt(
            lane=lane,
            shot_type=img_spec.shot_type,
            bundle_theme=bundle_tokens + (
                f", {img_spec.extra_tokens}" if img_spec.extra_tokens else ""
            ),
            seed=img_spec.seed,
        )

        # Enhance with photorealism
        positive = self.enhancer.enhance(
            positive,
            shot_type=img_spec.shot_type,
            lane=lane,
            realism_level=0.75,
            seed=img_spec.seed,
        )
        negative = self.enhancer.enhance_negative(negative)

        # Get resolution
        resolution = RESOLUTIONS.get(img_spec.resolution_key, RESOLUTIONS["3:4"])

        # Build workflow
        workflow = self.builder.build_vault_workflow(
            positive_prompt=positive,
            negative_prompt=negative,
            width=resolution.width,
            height=resolution.height,
            seed=img_spec.seed,
            character_id=character_id,
            lane=lane,
            output_subdir=output_subdir,
            faceid_image=faceid_ref,
            style_image=style_ref,
        )

        # Execute
        result = self.client.execute_workflow(workflow, timeout=600)

        # Save
        output_dir = os.path.join(
            self.config.output.vault_dir(character_id, lane),
            output_subdir,
        )
        saved = self.client.save_images(result, output_dir)
        return saved[0] if saved else ""

    def _generate_bundle(
        self,
        engine: PromptEngine,
        bundle: BundleSpec,
        lane: ContentLane,
        character_id: str,
        faceid_ref: str,
        style_ref: str,
    ) -> int:
        """Generate all images in a bundle.

        Returns count of successfully generated images.
        """
        count = 0
        bundle_subdir = f"bundles/{bundle.bundle_type}/{bundle.bundle_id}"

        for img_spec in bundle.images:
            try:
                self._generate_single_image(
                    engine, img_spec, lane, character_id,
                    faceid_ref, style_ref,
                    output_subdir=bundle_subdir,
                    bundle_tokens=bundle.shared_tokens,
                )
                count += 1
            except Exception as e:
                logger.error(
                    f"      Image {img_spec.image_id} in bundle "
                    f"{bundle.bundle_id} failed: {e}"
                )

        return count

    # -------------------------------------------------------------------------
    # SELECTIVE RUNS
    # -------------------------------------------------------------------------

    def run_single_lane(
        self,
        character_id: str,
        lane: ContentLane,
        seed_base: int = 12345,
    ) -> dict:
        """Run vault generation for a single character + single lane."""
        characters = CharacterData.load_roster(self.characters_dir)
        if character_id not in characters:
            raise ValueError(f"Character {character_id} not found")

        character = characters[character_id]
        engine = PromptEngine(character)

        return self.run_lane(
            character, engine, lane, seed_base,
            self._get_faceid_reference(character_id),
            self._get_style_reference(character_id),
        )

    def run_test(
        self,
        character_id: str,
        lane: ContentLane = ContentLane.SFW,
        count: int = 5,
        seed: int = 42,
    ) -> list:
        """Generate a small test batch for validation before full run."""
        characters = CharacterData.load_roster(self.characters_dir)
        character = characters[character_id]
        engine = PromptEngine(character)

        plan = self.bundle_mgr.generate_lane_plan(character_id, lane, seed)
        test_specs = plan["one_offs"][:count]

        results = []
        for spec in test_specs:
            try:
                path = self._generate_single_image(
                    engine, spec, lane, character_id,
                    self._get_faceid_reference(character_id),
                    self._get_style_reference(character_id),
                    output_subdir="test",
                )
                results.append({"image_id": spec.image_id, "path": path})
            except Exception as e:
                results.append({"image_id": spec.image_id, "error": str(e)})

        return results
