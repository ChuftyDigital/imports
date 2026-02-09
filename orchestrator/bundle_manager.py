"""Bundle manager — organizes vault images into coherent themed bundles.

Handles the vault structure:
    Per lane (200 images):
        80 one-off images
        10 bundles × 3 images = 30 images
        10 bundles × 5 images = 50 images
        4 bundles × 10 images = 40 images

Each bundle shares a theme (location, mood, outfit style, lighting) while
varying pose, expression, and exact composition across images.
"""

import os
import random
from dataclasses import dataclass, field
from typing import Optional

from .config import ContentLane, PipelineConfig, RESOLUTIONS


@dataclass
class BundleSpec:
    """Specification for a single bundle."""
    bundle_id: str
    bundle_type: str          # "bundle_3", "bundle_5", "bundle_10"
    size: int                 # 3, 5, or 10
    lane: ContentLane
    theme_name: str
    shared_tokens: str        # Tokens shared across all images
    base_seed: int
    images: list = field(default_factory=list)  # List of ImageSpec


@dataclass
class ImageSpec:
    """Specification for a single image within the vault."""
    image_id: str
    index: int
    seed: int
    shot_type: str            # "portrait", "three_quarter", "full_body", "closeup"
    resolution_key: str       # Maps to RESOLUTIONS dict
    bundle_id: Optional[str] = None
    extra_tokens: str = ""    # Per-image variant tokens
    clothing_override: str = ""
    expression_override: str = ""


class BundleManager:
    """Generates the complete image specification for a character's vault.

    Creates a deterministic plan for all 200 images per lane,
    including bundle groupings and per-image variations.
    """

    # Shot type distributions per image category
    SHOT_WEIGHTS = {
        "one_off": {
            "portrait": 0.30,
            "three_quarter": 0.30,
            "full_body": 0.25,
            "closeup": 0.15,
        },
        "bundle": {
            "portrait": 0.25,
            "three_quarter": 0.30,
            "full_body": 0.30,
            "closeup": 0.15,
        },
    }

    # Resolution distributions per shot type
    RESOLUTION_MAP = {
        "portrait": ["3:4", "1:1", "5:8"],
        "three_quarter": ["2:3", "3:4", "13:24"],
        "full_body": ["9:16", "2:3", "9:21"],
        "closeup": ["1:1", "3:4"],
    }

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def generate_lane_plan(
        self,
        character_id: str,
        lane: ContentLane,
        base_seed: int = 42,
    ) -> dict:
        """Generate the complete image plan for one lane of one character.

        Returns a dict with:
            - one_offs: List of 80 ImageSpec
            - bundles_3: List of 10 BundleSpec (3 images each)
            - bundles_5: List of 10 BundleSpec (5 images each)
            - bundles_10: List of 4 BundleSpec (10 images each)
            - total_images: 200
        """
        rng = random.Random(base_seed)
        vc = self.config.vault_lane
        image_counter = 0

        # Generate one-off images
        one_offs = []
        for i in range(vc.one_off_count):
            seed = rng.randint(1, 2**53)
            shot = self._pick_shot_type(rng, "one_off")
            res = self._pick_resolution(rng, shot)

            one_offs.append(ImageSpec(
                image_id=f"{character_id}_{lane.value}_oneoff_{i:04d}",
                index=image_counter,
                seed=seed,
                shot_type=shot,
                resolution_key=res,
            ))
            image_counter += 1

        # Generate bundles
        bundles_3 = self._generate_bundles(
            character_id, lane, "bundle_3", 3,
            vc.bundle_3_count, rng, image_counter,
        )
        image_counter += vc.bundle_3_count * 3

        bundles_5 = self._generate_bundles(
            character_id, lane, "bundle_5", 5,
            vc.bundle_5_count, rng, image_counter,
        )
        image_counter += vc.bundle_5_count * 5

        bundles_10 = self._generate_bundles(
            character_id, lane, "bundle_10", 10,
            vc.bundle_10_count, rng, image_counter,
        )
        image_counter += vc.bundle_10_count * 10

        return {
            "character_id": character_id,
            "lane": lane,
            "one_offs": one_offs,
            "bundles_3": bundles_3,
            "bundles_5": bundles_5,
            "bundles_10": bundles_10,
            "total_images": image_counter,
        }

    def _generate_bundles(
        self,
        character_id: str,
        lane: ContentLane,
        bundle_type: str,
        bundle_size: int,
        count: int,
        rng: random.Random,
        start_index: int,
    ) -> list:
        """Generate a set of bundles with themed images."""
        bundles = []
        idx = start_index

        for b in range(count):
            bundle_seed = rng.randint(1, 2**53)
            bundle_rng = random.Random(bundle_seed)

            # Pick theme for this bundle
            theme = self._pick_bundle_theme(bundle_rng, lane)

            images = []
            # Ensure variety of shot types within bundle
            shot_sequence = self._generate_shot_sequence(bundle_rng, bundle_size)

            for i in range(bundle_size):
                img_seed = bundle_rng.randint(1, 2**53)
                shot = shot_sequence[i]
                res = self._pick_resolution(bundle_rng, shot)

                images.append(ImageSpec(
                    image_id=f"{character_id}_{lane.value}_{bundle_type}_{b:03d}_{i:02d}",
                    index=idx,
                    seed=img_seed,
                    shot_type=shot,
                    resolution_key=res,
                    bundle_id=f"{character_id}_{lane.value}_{bundle_type}_{b:03d}",
                    extra_tokens=theme.get("per_image_tokens", [""])[i % len(theme.get("per_image_tokens", [""]))],
                ))
                idx += 1

            bundles.append(BundleSpec(
                bundle_id=f"{character_id}_{lane.value}_{bundle_type}_{b:03d}",
                bundle_type=bundle_type,
                size=bundle_size,
                lane=lane,
                theme_name=theme["name"],
                shared_tokens=theme["tokens"],
                base_seed=bundle_seed,
                images=images,
            ))

        return bundles

    def _generate_shot_sequence(self, rng: random.Random,
                                 size: int) -> list:
        """Generate a varied sequence of shot types for a bundle.

        Ensures we don't get all the same shot type in a bundle.
        """
        required_types = ["portrait", "three_quarter", "full_body"]

        if size <= 3:
            return rng.sample(required_types, size)

        sequence = list(required_types)
        remaining = size - len(sequence)
        all_types = ["portrait", "three_quarter", "full_body", "closeup"]

        for _ in range(remaining):
            sequence.append(rng.choice(all_types))

        rng.shuffle(sequence)
        return sequence

    def _pick_shot_type(self, rng: random.Random, category: str) -> str:
        """Pick a shot type based on weighted distribution."""
        weights = self.SHOT_WEIGHTS.get(category, self.SHOT_WEIGHTS["one_off"])
        types = list(weights.keys())
        probs = list(weights.values())
        return rng.choices(types, weights=probs, k=1)[0]

    def _pick_resolution(self, rng: random.Random, shot_type: str) -> str:
        """Pick an appropriate resolution for a shot type."""
        options = self.RESOLUTION_MAP.get(shot_type, ["3:4"])
        return rng.choice(options)

    def _pick_bundle_theme(self, rng: random.Random,
                            lane: ContentLane) -> dict:
        """Pick a theme for a bundle based on lane."""
        themes = [
            {
                "name": "Golden Hour Outdoors",
                "tokens": "golden hour, warm sunlight, outdoor setting, lens flare",
                "per_image_tokens": [
                    "standing in sunlight",
                    "sitting on bench, sun behind",
                    "walking, candid moment",
                    "leaning against wall, backlit",
                    "looking over shoulder, sun flare",
                ],
            },
            {
                "name": "City Streets",
                "tokens": "urban street, city backdrop, modern architecture, street style",
                "per_image_tokens": [
                    "crossing street, confident stride",
                    "leaning on railing, city view",
                    "sitting at outdoor cafe",
                    "standing on sidewalk, shop fronts",
                    "walking through alley, moody light",
                ],
            },
            {
                "name": "Studio Professional",
                "tokens": "professional studio, clean backdrop, controlled lighting, editorial",
                "per_image_tokens": [
                    "centered pose, direct gaze",
                    "profile view, dramatic shadow",
                    "dynamic pose, movement",
                    "seated, relaxed professional",
                    "standing, hands in pockets",
                ],
            },
            {
                "name": "Coastal Vibes",
                "tokens": "beach, ocean, sandy, coastal, sea breeze, natural light",
                "per_image_tokens": [
                    "walking along shoreline",
                    "sitting on rocks, waves behind",
                    "standing in shallow water",
                    "lying on sand, relaxed",
                    "wind in hair, looking at ocean",
                ],
            },
            {
                "name": "Home Comfort",
                "tokens": "modern interior, cozy home, warm lighting, lifestyle",
                "per_image_tokens": [
                    "on sofa, casual pose",
                    "by window, natural light",
                    "in kitchen, candid moment",
                    "on bed, relaxed morning",
                    "mirror selfie, bathroom",
                ],
            },
            {
                "name": "Fitness & Active",
                "tokens": "athletic, active, gym, workout, energetic",
                "per_image_tokens": [
                    "mid-workout, dynamic",
                    "post-workout, towel on shoulders",
                    "stretching, flexibility",
                    "mirror shot, gym background",
                    "outdoor run, action shot",
                ],
            },
            {
                "name": "Night Out",
                "tokens": "nighttime, city lights, glamorous, evening, neon",
                "per_image_tokens": [
                    "standing under neon sign",
                    "seated at bar, moody lighting",
                    "dancing, motion blur",
                    "walking in evening dress",
                    "close-up, dramatic city lights bokeh",
                ],
            },
            {
                "name": "Garden Paradise",
                "tokens": "lush garden, flowers, greenery, natural beauty, dappled light",
                "per_image_tokens": [
                    "among flowers, soft focus background",
                    "sitting on garden bench",
                    "walking through garden path",
                    "smelling flowers, eyes closed",
                    "lying in grass, overhead shot",
                ],
            },
        ]

        # Add intimate themes for higher lanes
        if lane in (ContentLane.SUGGESTIVE, ContentLane.SPICY, ContentLane.NSFW):
            themes.extend([
                {
                    "name": "Hotel Luxury",
                    "tokens": "luxury hotel suite, elegant interior, mood lighting, opulent",
                    "per_image_tokens": [
                        "on hotel bed, luxurious sheets",
                        "by floor-to-ceiling window, city view",
                        "in marble bathroom, mirror",
                        "draped on chaise lounge",
                        "room service scene, silk robe",
                    ],
                },
                {
                    "name": "Pool & Spa",
                    "tokens": "swimming pool, spa, water, wet skin, tropical",
                    "per_image_tokens": [
                        "pool edge, legs in water",
                        "emerging from pool, wet",
                        "lounging poolside, sunglasses",
                        "in hot tub, steam",
                        "showering off, outdoor shower",
                    ],
                },
            ])

        if lane in (ContentLane.SPICY, ContentLane.NSFW):
            themes.extend([
                {
                    "name": "Boudoir Elegance",
                    "tokens": "boudoir, intimate, sensual, soft lighting, silk, lace",
                    "per_image_tokens": [
                        "lying on bed, looking at camera",
                        "sitting on edge of bed",
                        "standing by vanity mirror",
                        "draped in sheets, artistic",
                        "back view, looking over shoulder",
                    ],
                },
                {
                    "name": "Candlelight Romance",
                    "tokens": "candlelight, warm glow, romantic, intimate atmosphere, shadows",
                    "per_image_tokens": [
                        "bathed in candlelight, front",
                        "side lit by candles, dramatic",
                        "relaxing in bathtub, candles around",
                        "lying down, candle glow on skin",
                        "standing, silhouette with candle backlight",
                    ],
                },
            ])

        return rng.choice(themes)

    def get_full_character_plan(
        self,
        character_id: str,
        base_seed: int = 42,
    ) -> dict:
        """Generate the complete vault plan for all 4 lanes of a character.

        Returns:
            Dict with lane plans and summary statistics.
        """
        plans = {}
        total = 0

        for lane in ContentLane:
            lane_seed = base_seed + hash(lane.value) % 10000
            plan = self.generate_lane_plan(character_id, lane, lane_seed)
            plans[lane] = plan
            total += plan["total_images"]

        return {
            "character_id": character_id,
            "lanes": plans,
            "total_images": total,
            "expected_total": self.config.vault_lane.total_images * 4,
        }
