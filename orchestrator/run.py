#!/usr/bin/env python3
"""Main entry point for the AI Influencer image generation pipeline.

Usage:
    # Generate master images for all characters (Workflow 1)
    python -m orchestrator.run master --all

    # Generate master images for a single character
    python -m orchestrator.run master --character aria_nova

    # Generate vault images for all characters (Workflow 2)
    python -m orchestrator.run vault --all

    # Generate vault for a single character, single lane
    python -m orchestrator.run vault --character aria_nova --lane sfw

    # Test run (5 images) to validate setup
    python -m orchestrator.run test --character aria_nova --lane sfw --count 5

    # Show pipeline stats
    python -m orchestrator.run stats

    # Generate prompts only (no image generation) for review
    python -m orchestrator.run prompts --character aria_nova --type master
"""

import argparse
import json
import logging
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from orchestrator.config import PipelineConfig, ContentLane
from orchestrator.master_pipeline import MasterPipeline
from orchestrator.vault_pipeline import VaultPipeline
from orchestrator.bundle_manager import BundleManager
from prompts.engine import PromptEngine, CharacterData
from prompts.enhancer import PromptEnhancer


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_master(args, config):
    """Run Workflow 1: Master Image Generation."""
    pipeline = MasterPipeline(
        config=config,
        base_workflow_path=args.workflow,
        characters_dir=args.characters_dir,
    )

    if args.all:
        pipeline.run_all(seed_base=args.seed)
    elif args.character:
        pipeline.run_single(
            character_id=args.character,
            master_type=args.type or "portrait",
            seed=args.seed,
        )
    else:
        print("Specify --all or --character <id>")
        sys.exit(1)


def cmd_vault(args, config):
    """Run Workflow 2: Image Vault Generation."""
    pipeline = VaultPipeline(
        config=config,
        base_workflow_path=args.workflow,
        characters_dir=args.characters_dir,
    )

    if args.all:
        pipeline.run_all(seed_base=args.seed)
    elif args.character:
        if args.lane:
            lane = ContentLane(args.lane)
            pipeline.run_single_lane(args.character, lane, args.seed)
        else:
            characters = CharacterData.load_roster(args.characters_dir)
            char = characters[args.character]
            pipeline.run_character(char, args.seed)
    else:
        print("Specify --all or --character <id>")
        sys.exit(1)


def cmd_test(args, config):
    """Run a test batch for validation."""
    pipeline = VaultPipeline(
        config=config,
        base_workflow_path=args.workflow,
        characters_dir=args.characters_dir,
    )

    lane = ContentLane(args.lane or "sfw")
    results = pipeline.run_test(
        character_id=args.character,
        lane=lane,
        count=args.count or 5,
        seed=args.seed,
    )

    print(f"\nTest results for {args.character} ({lane.value}):")
    for r in results:
        status = "OK" if "path" in r else f"FAIL: {r.get('error', '?')}"
        print(f"  {r['image_id']}: {status}")


def cmd_stats(args, config):
    """Show pipeline statistics."""
    characters = CharacterData.load_roster(args.characters_dir)
    n_chars = len(characters)
    vault = config.vault_lane

    print("\n" + "=" * 60)
    print("AI INFLUENCER PIPELINE — GENERATION STATISTICS")
    print("=" * 60)
    print(f"\nCharacters:        {n_chars}")
    print(f"\nWorkflow 1: Master Images")
    print(f"  Per character:   3 images (portrait, 3/4, full body)")
    print(f"  Total:           {n_chars * 3} images")
    print(f"\nWorkflow 2: Image Vault")
    print(f"  Lanes:           4 (SFW, Suggestive, Spicy, NSFW)")
    print(f"  Per lane:")
    print(f"    One-offs:      {vault.one_off_count}")
    print(f"    3-packs:       {vault.bundle_3_count} × 3 = {vault.bundle_3_count * 3}")
    print(f"    5-packs:       {vault.bundle_5_count} × 5 = {vault.bundle_5_count * 5}")
    print(f"    10-packs:      {vault.bundle_10_count} × 10 = {vault.bundle_10_count * 10}")
    print(f"    Lane total:    {vault.total_images}")
    print(f"  Per character:   {vault.total_images * 4} images (4 lanes)")
    print(f"  All characters:  {vault.total_images * 4 * n_chars} images")
    print(f"\nGRAND TOTAL:       {n_chars * 3 + vault.total_images * 4 * n_chars} images")
    print(f"\nHardware:          {config.hardware.gpu_name} ({config.hardware.vram_gb}GB)")
    print(f"Base model:        {config.models.checkpoint}")
    print(f"Sampler:           {config.sampler.sampler_name} / {config.sampler.scheduler}")
    print(f"Steps:             {config.sampler.steps}")
    print(f"CFG:               {config.sampler.cfg}")
    print("=" * 60)

    # List characters
    print(f"\nCharacter Roster ({n_chars}):")
    for cid, cdata in characters.items():
        name = cdata.display_name
        eth = cdata.get("ethnicity", "?")
        age = cdata.get("age_appearance", "?")
        print(f"  {cid:25s}  {name:20s}  {eth:30s}  age {age}")


def cmd_prompts(args, config):
    """Generate and display prompts without running generation."""
    characters = CharacterData.load_roster(args.characters_dir)
    char_id = args.character
    if char_id not in characters:
        print(f"Character {char_id} not found")
        sys.exit(1)

    char = characters[char_id]
    engine = PromptEngine(char)
    enhancer = PromptEnhancer()

    prompt_type = args.type or "master"

    print(f"\nPrompts for: {char.display_name} ({char_id})")
    print("=" * 70)

    if prompt_type == "master":
        for mt in ["portrait", "three_quarter", "full_body"]:
            pos, neg = engine.build_master_prompt(mt)
            pos = enhancer.enhance(pos, shot_type=mt, seed=args.seed)
            neg = enhancer.enhance_negative(neg)

            print(f"\n--- {mt.upper()} ---")
            print(f"POSITIVE:\n{pos}\n")
            print(f"NEGATIVE:\n{neg}\n")

    elif prompt_type == "vault":
        lane = ContentLane(args.lane or "sfw")
        for shot in ["portrait", "three_quarter", "full_body"]:
            pos, neg = engine.build_vault_prompt(
                lane=lane, shot_type=shot, seed=args.seed,
            )
            pos = enhancer.enhance(pos, shot_type=shot, lane=lane, seed=args.seed)
            neg = enhancer.enhance_negative(neg)

            print(f"\n--- {lane.value.upper()} / {shot.upper()} ---")
            print(f"POSITIVE:\n{pos}\n")
            print(f"NEGATIVE:\n{neg}\n")


def main():
    parser = argparse.ArgumentParser(
        description="AI Influencer Image Generation Pipeline"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--workflow", default="workflows/base_workflow.json",
        help="Path to base ComfyUI workflow JSON",
    )
    parser.add_argument(
        "--characters-dir", default="characters/roster",
        help="Path to character profiles directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="/output")

    subparsers = parser.add_subparsers(dest="command")

    # master command
    p_master = subparsers.add_parser("master", help="Run Workflow 1: Master Images")
    p_master.add_argument("--all", action="store_true")
    p_master.add_argument("--character", type=str)
    p_master.add_argument("--type", type=str, choices=["portrait", "three_quarter", "full_body"])

    # vault command
    p_vault = subparsers.add_parser("vault", help="Run Workflow 2: Vault Generation")
    p_vault.add_argument("--all", action="store_true")
    p_vault.add_argument("--character", type=str)
    p_vault.add_argument("--lane", type=str, choices=["sfw", "suggestive", "spicy", "nsfw"])

    # test command
    p_test = subparsers.add_parser("test", help="Run test batch")
    p_test.add_argument("--character", required=True)
    p_test.add_argument("--lane", type=str, default="sfw")
    p_test.add_argument("--count", type=int, default=5)

    # stats command
    subparsers.add_parser("stats", help="Show pipeline statistics")

    # prompts command
    p_prompts = subparsers.add_parser("prompts", help="Preview prompts")
    p_prompts.add_argument("--character", required=True)
    p_prompts.add_argument("--type", choices=["master", "vault"], default="master")
    p_prompts.add_argument("--lane", type=str, default="sfw")

    args = parser.parse_args()
    setup_logging(args.verbose)

    config = PipelineConfig()
    config.output.base_dir = args.output_dir

    commands = {
        "master": cmd_master,
        "vault": cmd_vault,
        "test": cmd_test,
        "stats": cmd_stats,
        "prompts": cmd_prompts,
    }

    if args.command in commands:
        commands[args.command](args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
