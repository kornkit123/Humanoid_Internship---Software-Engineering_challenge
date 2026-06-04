#!/usr/bin/env python3
"""
main.py — Run the LLM agent in a virtual world.

Usage:
    python main.py                        # interactive scenario picker
    python main.py --scenario key_door    # run a specific scenario
    python main.py --scenario navigate    # maze navigation
    python main.py --scenario messenger   # NPC messenger quest
    python main.py --log run.json         # save transcript to JSON
"""

import argparse
import json
import sys
from pathlib import Path

from world.scenarios import SCENARIOS
from agent.llm_agent  import LLMAgent


# ------------------------------------------------------------------
# Goal-check functions (one per scenario)
# ------------------------------------------------------------------

def check_key_door(world) -> bool:
    """Goal: reach the chest in the top room."""
    agent = world.get_agent()
    # Chest is removed when opened; or agent stands on tile y < 4
    if agent and agent.y <= 3:
        chest = world.entities.get("chest1")
        if chest and agent.x == chest.x and agent.y == chest.y:
            return True
    return False


def check_messenger(world) -> bool:
    """Goal: agent has talked to the blacksmith (evidenced by an event log entry)."""
    return any("Gareth" in msg for msg in world.messages)


def check_navigate(world) -> bool:
    """Goal: agent picks up the red beacon."""
    agent = world.get_agent()
    if agent:
        inv = agent.data.get("inventory", [])
        if any(i["id"] == "red_beacon" for i in inv):
            return True
        # Also succeeds if agent stands on it
        if any(e.id == "red_beacon" for e in world.entities_at(agent.x, agent.y)):
            return True
    return False


GOAL_CHECKS = {
    "key_door":  check_key_door,
    "messenger": check_messenger,
    "navigate":  check_navigate,
}


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def pick_scenario_interactive() -> str:
    print("\nAvailable scenarios:")
    for i, name in enumerate(SCENARIOS, 1):
        print(f"  {i}. {name}")
    choice = input("Pick scenario number (or name): ").strip()
    if choice.isdigit():
        keys = list(SCENARIOS.keys())
        idx  = int(choice) - 1
        if 0 <= idx < len(keys):
            return keys[idx]
    if choice in SCENARIOS:
        return choice
    print("Invalid choice. Defaulting to 'key_door'.")
    return "key_door"


def save_log(history: list[dict], path: str):
    clean = []
    for h in history:
        clean.append({
            "turn":      h["turn"],
            "reasoning": h["reasoning"],
            "action":    h["action"],
            "success":   h["success"],
            "message":   h["message"],
        })
    Path(path).write_text(json.dumps(clean, indent=2))
    print(f"\nTranscript saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="LLM Agent in a Virtual World")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()),
                        help="Which scenario to run")
    parser.add_argument("--max-turns", type=int, default=40,
                        help="Maximum agent turns (default: 40)")
    parser.add_argument("--log", type=str, default=None,
                        help="Path to save JSON transcript")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-step output")
    args = parser.parse_args()

    scenario_name = args.scenario or pick_scenario_interactive()

    factory     = SCENARIOS[scenario_name]
    world, goal = factory()
    goal_check  = GOAL_CHECKS.get(scenario_name)

    agent   = LLMAgent(world, goal, max_turns=args.max_turns)
    history = agent.run(goal_check=goal_check, verbose=not args.quiet)

    # Summary
    successes = sum(1 for h in history if h["success"])
    print(f"\n{'='*60}")
    print(f"  Scenario : {scenario_name}")
    print(f"  Turns    : {len(history)}")
    print(f"  Success  : {successes} / {len(history)} actions succeeded")
    goal_met = goal_check and goal_check(world)
    print(f"  Goal met : {'YES 🎉' if goal_met else 'no'}")
    print(f"{'='*60}\n")

    if args.log:
        save_log(history, args.log)
    else:
        # Always save to logs/ directory
        log_path = f"logs/{scenario_name}_latest.json"
        save_log(history, log_path)


if __name__ == "__main__":
    main()
