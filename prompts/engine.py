"""Core prompt generation engine for AI influencer image generation.

This module builds detailed, character-specific prompts by combining:
1. Character persona data (from YAML profiles)
2. Shot-type templates (portrait, full body, etc.)
3. Lane-specific modifiers (SFW → NSFW)
4. Scene/environment context
5. Quality and technical tokens

The prompts are designed for FabricatedXL v7 (NoobAI-based SDXL) and follow
the NoobAI quality tag conventions.
"""

import random
import re
import os
from dataclasses import dataclass, field
from typing import Optional

import yaml

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orchestrator.config import (
    ContentLane, ImageType, QUALITY_PREFIX, UNIVERSAL_NEGATIVE,
    LANE_NEGATIVES, RESOLUTIONS,
)


@dataclass
class CharacterData:
    """Parsed character profile with all attributes accessible."""
    raw: dict
    character_id: str = ""
    display_name: str = ""

    def __post_init__(self):
        self.character_id = self.raw.get("character_id", "")
        self.display_name = self.raw.get("display_name", "")

    def get(self, dotpath: str, default=""):
        """Access nested YAML data with dot notation: 'face.shape'."""
        keys = dotpath.split(".")
        val = self.raw
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            else:
                return default
        return val if val is not None else default

    @classmethod
    def from_yaml(cls, filepath: str) -> "CharacterData":
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return cls(raw=data)

    @classmethod
    def load_roster(cls, roster_dir: str) -> dict:
        """Load all character profiles from the roster directory."""
        characters = {}
        for fname in sorted(os.listdir(roster_dir)):
            if fname.endswith(".yaml") or fname.endswith(".yml"):
                char = cls.from_yaml(os.path.join(roster_dir, fname))
                characters[char.character_id] = char
        return characters


class PromptEngine:
    """Builds detailed, character-aware prompts for ComfyUI generation.

    This is the backbone of the generation pipeline — every image starts here.
    The engine ensures character consistency by injecting the full physical
    description, style tokens, and lane-appropriate content into every prompt.
    """

    def __init__(self, character: CharacterData):
        self.char = character

    # -------------------------------------------------------------------------
    # IDENTITY BLOCK — physical description tokens
    # -------------------------------------------------------------------------

    def build_face_tokens(self) -> str:
        """Build detailed face description from character profile."""
        parts = []
        face = self.char.get("face", {})
        if isinstance(face, dict):
            skin = face.get("skin_tone", "")
            if skin:
                parts.append(f"{skin} skin")
            texture = face.get("skin_texture", "")
            if texture:
                parts.append(texture)
            shape = face.get("shape", "")
            if shape:
                parts.append(f"{shape} face shape")
            jawline = face.get("jawline", "")
            if jawline:
                parts.append(f"{jawline} jawline")
            cheekbones = face.get("cheekbones", "")
            if cheekbones:
                parts.append(f"{cheekbones} cheekbones")
            details = face.get("complexion_details", "")
            if details:
                parts.append(details)
        return ", ".join(parts)

    def build_eye_tokens(self) -> str:
        """Build detailed eye description."""
        parts = []
        eyes = self.char.get("eyes", {})
        if isinstance(eyes, dict):
            color = eyes.get("color", "")
            shape = eyes.get("shape", "")
            if color and shape:
                parts.append(f"{color} {shape} eyes")
            elif color:
                parts.append(f"{color} eyes")
            lashes = eyes.get("lashes", "")
            if lashes:
                parts.append(f"{lashes} eyelashes")
            brow = eyes.get("brow_shape", "")
            brow_color = eyes.get("brow_color", "")
            if brow:
                parts.append(f"{brow} eyebrows")
            distinctive = eyes.get("distinctive", "")
            if distinctive:
                parts.append(distinctive)
        return ", ".join(parts)

    def build_hair_tokens(self, style_override: str = "") -> str:
        """Build hair description, optionally overriding the style."""
        parts = []
        hair = self.char.get("hair", {})
        if isinstance(hair, dict):
            color = hair.get("color", "")
            if color:
                parts.append(f"{color} hair")
            length = hair.get("length", "")
            if length:
                parts.append(f"{length}")
            texture = hair.get("texture", "")
            if texture:
                parts.append(texture)
            volume = hair.get("volume", "")
            if volume:
                parts.append(volume)
            if style_override:
                parts.append(style_override)
            else:
                style = hair.get("signature_style", "")
                if style:
                    parts.append(style)
            bangs = hair.get("bangs", "")
            if bangs and bangs.lower() != "none":
                parts.append(bangs)
        return ", ".join(parts)

    def build_body_tokens(self, include_full: bool = False) -> str:
        """Build body description tokens. Full detail for full-body shots."""
        parts = []
        body = self.char.get("body", {})
        if isinstance(body, dict):
            build = body.get("build", "")
            if build:
                parts.append(f"{build} body")
            height = body.get("height_description", "")
            if height and include_full:
                parts.append(height)
            bust = body.get("bust", "")
            if bust:
                parts.append(f"{bust} bust")
            if include_full:
                waist = body.get("waist", "")
                if waist:
                    parts.append(f"{waist} waist")
                hips = body.get("hips", "")
                if hips:
                    parts.append(f"{hips} hips")
                legs = body.get("legs", "")
                if legs:
                    parts.append(f"{legs} legs")
            skin_details = body.get("skin_details", "")
            if skin_details:
                parts.append(skin_details)
        return ", ".join(parts)

    def build_lips_tokens(self) -> str:
        """Build lip description."""
        parts = []
        lips = self.char.get("lips", {})
        if isinstance(lips, dict):
            shape = lips.get("shape", "")
            color = lips.get("color", "")
            if shape:
                parts.append(f"{shape} lips")
            if color:
                parts.append(f"{color} lip color")
        return ", ".join(parts)

    def build_nose_tokens(self) -> str:
        """Build nose description."""
        nose = self.char.get("nose", {})
        if isinstance(nose, dict):
            shape = nose.get("shape", "")
            size = nose.get("size", "")
            if shape and size:
                return f"{size} {shape} nose"
            elif shape:
                return f"{shape} nose"
        return ""

    # Budget for the identity block within the total prompt.
    # Leaves room for framing, clothing, environment, and enhancement tokens.
    IDENTITY_BUDGET = 600

    def build_identity_block(self, include_full_body: bool = False,
                              hair_style_override: str = "") -> str:
        """Assemble the identity description using the curated summary first.

        Uses the pre-written identity_tokens (concise ~200 char summary) as
        the core, then fills remaining budget with expanded detail tokens
        in priority order: face > eyes > hair > body > nose > lips.
        """
        mandatory = self.char.get("generation.mandatory_tokens", "1girl, solo")
        identity_summary = self.char.get("generation.identity_tokens", "")

        # Core: mandatory + curated summary (always included)
        core = ", ".join(s for s in [mandatory, identity_summary] if s)

        # If no curated summary, fall back to full expansion
        if not identity_summary:
            age = self.char.get("age_appearance", "")
            ethnicity = self.char.get("ethnicity", "")
            sections = [
                mandatory,
                f"{age} year old" if age else "",
                ethnicity if ethnicity else "",
                self.build_face_tokens(),
                self.build_eye_tokens(),
                self.build_nose_tokens(),
                self.build_lips_tokens(),
                self.build_hair_tokens(style_override=hair_style_override),
                self.build_body_tokens(include_full=include_full_body),
            ]
            return ", ".join(s for s in sections if s)

        # Budget-aware expansion: add detail tokens if room permits
        remaining = self.IDENTITY_BUDGET - len(core)
        extras = []

        # Hair override takes priority (for vault variety)
        if hair_style_override:
            extras.append(hair_style_override)

        # Add expanded details in priority order
        detail_builders = [
            self.build_face_tokens,
            self.build_eye_tokens,
            lambda: self.build_hair_tokens(style_override=hair_style_override),
            lambda: self.build_body_tokens(include_full=include_full_body),
        ]

        for builder in detail_builders:
            tokens = builder()
            if tokens and remaining > len(tokens) + 2:
                extras.append(tokens)
                remaining -= len(tokens) + 2

        if extras:
            return f"{core}, {', '.join(extras)}"
        return core

    # -------------------------------------------------------------------------
    # STYLE & ENVIRONMENT BLOCK
    # -------------------------------------------------------------------------

    def build_style_tokens(self) -> str:
        """Build style/aesthetic tokens."""
        parts = []
        style = self.char.get("style", {})
        if isinstance(style, dict):
            aesthetic = style.get("overall_aesthetic", "")
            if aesthetic:
                parts.append(aesthetic)
            makeup = style.get("makeup_style", "")
            if makeup:
                parts.append(makeup)
            jewelry = style.get("jewelry_signature", "")
            if jewelry:
                parts.append(jewelry)
        return ", ".join(parts)

    def build_expression_tokens(self, override: str = "") -> str:
        """Build expression/mood tokens."""
        if override:
            return override
        personality = self.char.get("personality", {})
        if isinstance(personality, dict):
            expr = personality.get("default_expression", "")
            eye_contact = personality.get("eye_contact", "")
            parts = [expr, eye_contact]
            return ", ".join(p for p in parts if p)
        return ""

    def build_environment_tokens(self, location: str = "", lighting: str = "",
                                  background: str = "") -> str:
        """Build environment/setting tokens."""
        parts = []
        if location:
            parts.append(location)
        if lighting:
            parts.append(lighting)
        elif self.char.get("settings.lighting_preference", ""):
            parts.append(self.char.get("settings.lighting_preference"))
        if background:
            parts.append(background)
        return ", ".join(parts)

    # -------------------------------------------------------------------------
    # CLOTHING BLOCK — lane-aware
    # -------------------------------------------------------------------------

    def get_clothing_for_lane(self, lane: ContentLane,
                               specific_outfit: str = "") -> str:
        """Get appropriate clothing based on content lane."""
        if specific_outfit:
            return specific_outfit
        lane_key = f"lanes.{lane.value}.clothing_styles"
        styles = self.char.get(lane_key, [])
        if isinstance(styles, list) and styles:
            return random.choice(styles)
        return ""

    def get_scenario_for_lane(self, lane: ContentLane) -> str:
        """Get a random scenario appropriate for the content lane."""
        lane_key = f"lanes.{lane.value}.typical_scenarios"
        scenarios = self.char.get(lane_key, [])
        if isinstance(scenarios, list) and scenarios:
            return random.choice(scenarios)
        return ""

    def get_lane_mood(self, lane: ContentLane) -> str:
        """Get the mood/energy for a specific lane."""
        return self.char.get(f"lanes.{lane.value}.mood", "")

    def get_lane_composition(self, lane: ContentLane) -> str:
        """Get composition notes for a lane."""
        return self.char.get(f"lanes.{lane.value}.composition_notes", "")

    # -------------------------------------------------------------------------
    # PROMPT ASSEMBLY — the main output functions
    # -------------------------------------------------------------------------

    def build_positive_prompt(
        self,
        shot_type: str = "portrait",
        lane: ContentLane = ContentLane.SFW,
        clothing: str = "",
        scenario: str = "",
        expression_override: str = "",
        hair_style_override: str = "",
        lighting: str = "",
        background: str = "",
        additional_tokens: str = "",
        framing: str = "",
        include_quality_prefix: bool = True,
    ) -> str:
        """Build a complete positive prompt for image generation.

        This is the primary function called by the pipeline.

        Args:
            shot_type: "portrait", "three_quarter", "full_body", or "closeup"
            lane: Content lane (SFW/Suggestive/Spicy/NSFW)
            clothing: Specific outfit override
            scenario: Scene description override
            expression_override: Override default expression
            hair_style_override: Override default hairstyle
            lighting: Lighting description
            background: Background description
            additional_tokens: Extra prompt tokens to append
            framing: Camera framing description
            include_quality_prefix: Whether to prepend quality tags
        """
        include_full_body = shot_type in ("full_body", "three_quarter")

        sections = []

        # 1. Quality prefix (NoobAI convention)
        if include_quality_prefix:
            sections.append(QUALITY_PREFIX)

        # 2. Character identity
        sections.append(
            self.build_identity_block(
                include_full_body=include_full_body,
                hair_style_override=hair_style_override,
            )
        )

        # 3. Expression
        expr = self.build_expression_tokens(override=expression_override)
        if expr:
            sections.append(expr)

        # 4. Framing/composition (custom framing fully replaces default)
        if framing:
            sections.append(framing)
        else:
            if shot_type == "portrait":
                sections.append("close-up portrait, head and shoulders, looking at viewer")
            elif shot_type == "three_quarter":
                sections.append("upper body, three quarter view, looking at viewer")
            elif shot_type == "full_body":
                sections.append("full body shot, standing, looking at viewer")
            elif shot_type == "closeup":
                sections.append("extreme close-up, face detail, looking at viewer")

        # 5. Clothing
        outfit = clothing or self.get_clothing_for_lane(lane)
        if outfit:
            sections.append(outfit)

        # 6. Style tokens
        style = self.build_style_tokens()
        if style:
            sections.append(style)

        # 7. Scenario/environment
        scene = scenario or self.get_scenario_for_lane(lane)
        env = self.build_environment_tokens(
            location=scene, lighting=lighting, background=background
        )
        if env:
            sections.append(env)

        # 8. Lane mood
        mood = self.get_lane_mood(lane)
        if mood:
            sections.append(mood)

        # 9. Additional tokens
        if additional_tokens:
            sections.append(additional_tokens)

        # Assemble and clean
        prompt = ", ".join(s for s in sections if s)
        prompt = self._clean_prompt(prompt)
        return prompt

    def build_negative_prompt(self, lane: ContentLane = ContentLane.SFW) -> str:
        """Build negative prompt combining universal + character + lane negatives."""
        parts = [UNIVERSAL_NEGATIVE]

        # Character-specific negatives
        char_neg = self.char.get("generation.character_negatives", "")
        if char_neg:
            parts.append(char_neg)

        # Lane-specific negatives
        lane_neg = LANE_NEGATIVES.get(lane, "")
        if lane_neg:
            parts.append(lane_neg)

        return ", ".join(parts)

    def build_master_prompt(self, master_type: str) -> tuple:
        """Build prompt for a master/canonical image.

        Args:
            master_type: "portrait", "three_quarter", or "full_body"

        Returns:
            (positive_prompt, negative_prompt) tuple
        """
        master_key = f"master_images.{master_type}"
        master = self.char.get(master_key, {})

        if not isinstance(master, dict):
            master = {}

        framing = master.get("framing", "")
        expression = master.get("expression", "")
        background = master.get("background", "")
        lighting = master.get("lighting", "")
        clothing = master.get("clothing", "")
        additional = master.get("additional_prompt", "")

        positive = self.build_positive_prompt(
            shot_type=master_type,
            lane=ContentLane.SFW,
            clothing=clothing,
            expression_override=expression,
            lighting=lighting,
            background=background,
            framing=framing,
            additional_tokens=additional,
        )

        negative = self.build_negative_prompt(ContentLane.SFW)

        return positive, negative

    def build_vault_prompt(
        self,
        lane: ContentLane,
        shot_type: str = "",
        bundle_theme: str = "",
        seed: Optional[int] = None,
    ) -> tuple:
        """Build prompt for a vault image with controlled randomization.

        Args:
            lane: Content lane
            shot_type: Override shot type (random if empty)
            bundle_theme: If part of a bundle, the unifying theme
            seed: Random seed for reproducible variation

        Returns:
            (positive_prompt, negative_prompt) tuple
        """
        if seed is not None:
            random.seed(seed)

        # Randomize shot type if not specified
        if not shot_type:
            weights = {
                "portrait": 0.3,
                "three_quarter": 0.35,
                "full_body": 0.25,
                "closeup": 0.1,
            }
            shot_type = random.choices(
                list(weights.keys()), weights=list(weights.values()), k=1
            )[0]

        # Randomize hair style occasionally
        hair_override = ""
        alt_styles = self.char.get("hair.alternative_styles", [])
        if isinstance(alt_styles, list) and alt_styles and random.random() < 0.3:
            hair_override = random.choice(alt_styles)

        # Bundle theme integration
        additional = ""
        if bundle_theme:
            additional = bundle_theme

        positive = self.build_positive_prompt(
            shot_type=shot_type,
            lane=lane,
            hair_style_override=hair_override,
            additional_tokens=additional,
        )

        negative = self.build_negative_prompt(lane)

        return positive, negative

    # -------------------------------------------------------------------------
    # BUNDLE THEME GENERATORS
    # -------------------------------------------------------------------------

    def generate_bundle_theme(self, lane: ContentLane,
                               bundle_size: int, seed: int = 0) -> dict:
        """Generate a cohesive theme for a bundle of images.

        Returns a dict with:
            - theme_name: Human-readable theme
            - shared_tokens: Tokens shared across all images in bundle
            - per_image_variants: List of variant tokens for each image
        """
        random.seed(seed)

        themes = self._get_themes_for_lane(lane)
        theme = random.choice(themes)

        variants = []
        for i in range(bundle_size):
            variant_seed = seed + i + 1
            random.seed(variant_seed)

            shot_types = ["portrait", "three_quarter", "full_body"]
            if bundle_size >= 5:
                shot_types.append("closeup")

            shot = shot_types[i % len(shot_types)]

            variant_clothing = self.get_clothing_for_lane(lane)
            variant_expression = ""
            moods = self.char.get("personality.mood_range", [])
            if isinstance(moods, list) and moods:
                variant_expression = random.choice(moods)

            variants.append({
                "shot_type": shot,
                "clothing": variant_clothing,
                "expression": variant_expression,
                "index": i,
            })

        return {
            "theme_name": theme["name"],
            "shared_tokens": theme["tokens"],
            "per_image_variants": variants,
        }

    def _get_themes_for_lane(self, lane: ContentLane) -> list:
        """Get available themes for a content lane."""
        base_themes = [
            {"name": "Golden Hour", "tokens": "golden hour lighting, warm tones, sun flare"},
            {"name": "Urban Exploration", "tokens": "urban cityscape, street photography, modern architecture"},
            {"name": "Studio Elegance", "tokens": "professional studio, dramatic lighting, clean backdrop"},
            {"name": "Natural Beauty", "tokens": "natural outdoor setting, soft natural light, green foliage"},
            {"name": "Cozy Indoor", "tokens": "cozy interior, warm ambient light, lifestyle setting"},
            {"name": "Rainy Day", "tokens": "rainy atmosphere, wet surfaces, moody lighting, reflections"},
            {"name": "Rooftop Views", "tokens": "rooftop setting, city skyline background, open sky"},
            {"name": "Beach Day", "tokens": "beach setting, ocean waves, sandy shore, blue sky"},
            {"name": "Night Life", "tokens": "nighttime, neon lights, urban night, city lights bokeh"},
            {"name": "Garden Party", "tokens": "lush garden, flower blooms, dappled sunlight, greenery"},
            {"name": "Vintage Mood", "tokens": "vintage aesthetic, retro tones, film grain look, nostalgic"},
            {"name": "Minimalist", "tokens": "minimalist setting, clean lines, neutral tones, modern"},
        ]

        if lane in (ContentLane.SUGGESTIVE, ContentLane.SPICY, ContentLane.NSFW):
            base_themes.extend([
                {"name": "Hotel Suite", "tokens": "luxury hotel room, elegant interior, soft lighting"},
                {"name": "Mirror Play", "tokens": "mirror reflection, intimate setting, self-aware pose"},
                {"name": "Silk and Satin", "tokens": "silk sheets, satin fabric, luxurious textures, soft focus"},
                {"name": "Morning After", "tokens": "morning light, bedroom, messy sheets, natural glow"},
            ])

        if lane in (ContentLane.SPICY, ContentLane.NSFW):
            base_themes.extend([
                {"name": "Boudoir Classic", "tokens": "boudoir setting, dramatic shadows, intimate atmosphere"},
                {"name": "Bath Time", "tokens": "bathroom setting, steam, wet skin, warm water"},
                {"name": "Candlelight", "tokens": "candlelight, warm flickering light, dark atmosphere, intimate"},
            ])

        return base_themes

    # -------------------------------------------------------------------------
    # UTILITY
    # -------------------------------------------------------------------------

    @staticmethod
    def _clean_prompt(prompt: str) -> str:
        """Clean up prompt: remove double commas, extra spaces, deduplicate tokens."""
        # Fix comma issues
        prompt = re.sub(r",\s*,", ",", prompt)
        prompt = re.sub(r"\s+", " ", prompt)
        prompt = re.sub(r"^,\s*", "", prompt)
        prompt = re.sub(r",\s*$", "", prompt)
        prompt = prompt.strip()

        # Deduplicate: if the same token appears twice, keep only the first
        seen = set()
        deduped = []
        for token in prompt.split(","):
            normalized = token.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(token.strip())
        prompt = ", ".join(deduped)

        return prompt
