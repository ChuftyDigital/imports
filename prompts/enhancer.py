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

    # SDXL CLIP encodes in 3 passes of 77 tokens = 231 tokens max.
    # ~4 chars per token on average, so ~900 chars of real content.
    # With commas/spaces overhead, cap total prompt at 1800 chars.
    MAX_PROMPT_CHARS = 1800

    # Photography technique tokens — randomly sampled for variety
    PHOTOGRAPHY_TOKENS = [
        "professional photography",
        "DSLR quality",
        "shallow depth of field",
        "bokeh background",
        "sharp focus on subject",
        "natural color grading",
        "shot on Canon EOS R5",
        "85mm lens",
        "f/1.8 aperture",
        "editorial photography",
    ]

    # Skin realism tokens — critical for avoiding the "AI skin" look
    SKIN_REALISM_TOKENS = [
        "realistic skin texture",
        "natural skin pores",
        "subtle skin imperfections",
        "realistic skin sheen",
    ]

    # Lighting quality tokens
    LIGHTING_QUALITY_TOKENS = [
        "professional lighting",
        "catch lights in eyes",
        "soft shadow gradients",
        "natural ambient occlusion",
    ]

    # Anti-artifact tokens (things that make images look less AI)
    ANTI_ARTIFACT_TOKENS = [
        "photorealistic",
        "ultra detailed",
        "crisp details",
        "lifelike",
    ]

    # Detail tokens per body zone
    DETAIL_TOKENS = {
        "face": [
            "detailed facial features",
            "realistic eye reflections",
            "individual eyelashes visible",
        ],
        "hair": [
            "individual hair strands visible",
            "realistic hair flow",
        ],
        "body": [
            "natural body proportions",
            "realistic skin texture on body",
        ],
        "hands": [
            "detailed fingers",
            "correct hand anatomy",
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
        """Enhance a prompt with photorealism tokens, budget-aware.

        Only adds enhancement tokens if the base prompt has room under the
        MAX_PROMPT_CHARS budget. Prioritizes: photography > skin > lighting >
        anti-artifact > zone detail. Stops adding once budget is reached.

        Args:
            prompt: Base prompt from PromptEngine
            shot_type: Type of shot (affects which detail tokens to add)
            lane: Content lane
            realism_level: 0.0-1.0, how many enhancement tokens to add
            seed: Random seed for reproducible enhancement

        Returns:
            Enhanced prompt string, capped at MAX_PROMPT_CHARS
        """
        if seed is not None:
            random.seed(seed)

        remaining = self.MAX_PROMPT_CHARS - len(prompt)
        if remaining < 50:
            # Base prompt already near limit — skip enhancements
            return prompt[:self.MAX_PROMPT_CHARS]

        # Collect candidate tokens in priority order
        candidates = []

        # Core photography (pick 2)
        candidates.extend(random.sample(self.PHOTOGRAPHY_TOKENS,
                                        min(2, len(self.PHOTOGRAPHY_TOKENS))))

        # Skin realism (pick 1-2)
        n_skin = max(1, int(2 * realism_level))
        candidates.extend(random.sample(self.SKIN_REALISM_TOKENS,
                                        min(n_skin, len(self.SKIN_REALISM_TOKENS))))

        # Lighting (pick 1)
        candidates.extend(random.sample(self.LIGHTING_QUALITY_TOKENS, 1))

        # Anti-artifact (pick 1-2)
        n_anti = max(1, int(2 * realism_level))
        candidates.extend(random.sample(self.ANTI_ARTIFACT_TOKENS,
                                        min(n_anti, len(self.ANTI_ARTIFACT_TOKENS))))

        # Shot-type-specific detail tokens (pick 1 per zone)
        if shot_type in ("portrait", "closeup"):
            zones = ["face", "hair"]
        elif shot_type == "three_quarter":
            zones = ["face", "hair", "body"]
        else:
            zones = ["face", "hair", "body", "hands"]

        for zone in zones:
            zone_tokens = self.DETAIL_TOKENS.get(zone, [])
            if zone_tokens:
                candidates.extend(random.sample(zone_tokens, 1))

        # Add tokens one by one until budget is hit
        added = []
        budget = remaining - 2  # account for leading ", "
        for token in candidates:
            cost = len(token) + 2  # ", " separator
            if budget >= cost:
                added.append(token)
                budget -= cost

        if not added:
            return prompt

        return f"{prompt}, {', '.join(added)}"

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
