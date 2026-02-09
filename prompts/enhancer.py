"""Prompt enhancer — adds photorealistic detail tokens and technical quality boosters.

This module post-processes prompts from the engine to inject:
- Photography-specific technical language
- Skin/texture realism tokens
- Lighting detail tokens
- Anti-AI-artifact tokens
- Resolution and detail level tokens

These enhancements are what push output from "AI-looking" to "photorealistic".
"""

import random
from typing import Optional

from orchestrator.config import ContentLane, ImageType


class PromptEnhancer:
    """Enhances base prompts with photorealism and technical photography tokens."""

    # Photography technique tokens — randomly sampled for variety
    PHOTOGRAPHY_TOKENS = [
        "professional photography",
        "DSLR quality",
        "shallow depth of field",
        "bokeh background",
        "sharp focus on subject",
        "high dynamic range",
        "natural color grading",
        "shot on Canon EOS R5",
        "85mm lens",
        "f/1.8 aperture",
        "professional portrait photography",
        "editorial photography",
        "fashion photography",
        "lifestyle photography",
    ]

    # Skin realism tokens — critical for avoiding the "AI skin" look
    SKIN_REALISM_TOKENS = [
        "realistic skin texture",
        "natural skin pores",
        "subtle skin imperfections",
        "natural skin subsurface scattering",
        "realistic skin sheen",
        "natural skin undertones",
    ]

    # Lighting quality tokens
    LIGHTING_QUALITY_TOKENS = [
        "professional lighting",
        "natural light and shadow interplay",
        "subtle rim lighting",
        "catch lights in eyes",
        "soft shadow gradients",
        "three-point lighting",
        "natural ambient occlusion",
    ]

    # Anti-artifact tokens (things that make images look less AI)
    ANTI_ARTIFACT_TOKENS = [
        "photorealistic",
        "hyperrealistic",
        "ultra detailed",
        "8k resolution",
        "crisp details",
        "lifelike",
    ]

    # Detail tokens per body zone
    DETAIL_TOKENS = {
        "face": [
            "detailed facial features",
            "realistic eye reflections",
            "natural lip texture",
            "individual eyelashes visible",
            "realistic eyebrow hairs",
        ],
        "hair": [
            "individual hair strands visible",
            "natural hair shine",
            "realistic hair flow",
            "detailed hair texture",
        ],
        "body": [
            "natural body proportions",
            "realistic skin texture on body",
            "natural muscle definition",
            "realistic body lighting",
        ],
        "hands": [
            "detailed fingers",
            "correct hand anatomy",
            "natural finger proportions",
            "realistic hand pose",
        ],
    }

    def enhance(
        self,
        prompt: str,
        shot_type: str = "portrait",
        lane: ContentLane = ContentLane.SFW,
        realism_level: float = 0.8,
        seed: Optional[int] = None,
    ) -> str:
        """Enhance a prompt with photorealism tokens.

        Args:
            prompt: Base prompt from PromptEngine
            shot_type: Type of shot (affects which detail tokens to add)
            lane: Content lane
            realism_level: 0.0-1.0, how many enhancement tokens to add
            seed: Random seed for reproducible enhancement

        Returns:
            Enhanced prompt string
        """
        if seed is not None:
            random.seed(seed)

        enhancements = []

        # Always add core realism tokens
        n_photo = max(1, int(len(self.PHOTOGRAPHY_TOKENS) * realism_level * 0.3))
        enhancements.extend(random.sample(self.PHOTOGRAPHY_TOKENS, n_photo))

        # Skin realism (always important for photorealistic humans)
        n_skin = max(1, int(len(self.SKIN_REALISM_TOKENS) * realism_level * 0.4))
        enhancements.extend(random.sample(self.SKIN_REALISM_TOKENS, n_skin))

        # Lighting quality
        n_light = max(1, int(len(self.LIGHTING_QUALITY_TOKENS) * realism_level * 0.3))
        enhancements.extend(random.sample(self.LIGHTING_QUALITY_TOKENS, n_light))

        # Anti-artifact tokens
        n_anti = max(1, int(len(self.ANTI_ARTIFACT_TOKENS) * realism_level * 0.4))
        enhancements.extend(random.sample(self.ANTI_ARTIFACT_TOKENS, n_anti))

        # Shot-type-specific detail tokens
        if shot_type in ("portrait", "closeup"):
            zones = ["face", "hair"]
        elif shot_type == "three_quarter":
            zones = ["face", "hair", "body"]
        else:
            zones = ["face", "hair", "body", "hands"]

        for zone in zones:
            zone_tokens = self.DETAIL_TOKENS.get(zone, [])
            if zone_tokens:
                n = max(1, int(len(zone_tokens) * realism_level * 0.3))
                enhancements.extend(random.sample(zone_tokens, n))

        enhancement_str = ", ".join(enhancements)
        return f"{prompt}, {enhancement_str}"

    def enhance_negative(self, negative: str) -> str:
        """Enhance negative prompt with anti-AI-artifact tokens."""
        anti_ai = [
            "artificial looking skin",
            "plastic skin",
            "airbrushed",
            "uncanny valley",
            "mannequin-like",
            "doll-like face",
            "wax figure",
            "blurry eyes",
            "asymmetric eyes",
            "malformed hands",
            "extra fingers",
            "fused fingers",
            "too many fingers",
            "mutated hands",
            "poorly drawn face",
            "mutation",
            "deformed",
            "ugly",
            "blurry",
            "bad anatomy",
            "extra limbs",
            "cloned face",
            "disfigured",
            "cross-eyed",
        ]
        return f"{negative}, {', '.join(anti_ai)}"
