"""
LLM Agent Harness
Connects the Claude API to the world loop.
The harness is responsible for:
  1. Translating world observations into a rich, structured prompt
  2. Parsing the LLM's JSON action response
  3. Feeding the result back into the world
"""

import json
import re
import time
from typing import Optional

import anthropic

from world.grid_world import GridWorld


# ── Anthropic client (reads ANTHROPIC_API_KEY from env) ──────────────
_client = anthropic.Anthropic()

MODEL = "claude-opus-4-5"


SYSTEM_PROMPT = """You are an intelligent agent exploring a 2D grid world.
Your goal is to accomplish the given task by taking deliberate, strategic actions.

WORLD CONVENTIONS
-----------------
- The world is a 2-D tile grid. North = y decreases, South = y increases.
- '@' on the minimap = you.
- '#' = wall  '.' = floor  'd' = open door  'D' = locked door  '~' = water
- Capital letters on the minimap = entities (K=key, C=chest, N=npc, S=sign, I=item)

AVAILABLE ACTIONS
-----------------
You must respond with EXACTLY ONE JSON object per turn:

  {"action": "move",    "direction": "<north|south|east|west>"}
  {"action": "pickup",  "target": "<item name or id>"}
  {"action": "use",     "item": "<item in inventory>", "target": "<target name or door direction>"}
  {"action": "examine", "target": "<entity or tile name>"}
  {"action": "talk",    "target": "<npc name>"}
  {"action": "wait"}

REASONING GUIDELINES
--------------------
Before choosing an action, briefly reason through:
1. Where am I and what do I see?
2. What is the goal and what is my current progress?
3. What is the best next action to make progress?
4. Are there any obstacles? How do I overcome them?

Output format (strictly):
<reasoning>
Your internal monologue here — a few sentences max.
</reasoning>
<action>
{"action": "...", ...}
</action>

Do NOT output anything outside these two XML tags."""


def _build_user_prompt(observation: dict, goal: str, history: list[dict]) -> str:
    """
    Convert the world observation dict into a detailed, structured text prompt.
    This is the critical 'observation representation' design choice.
    """
    obs = observation
    agent = obs["agent"]
    adj   = obs["adjacent_cells"]

    lines = []

    # ── Turn and goal ──────────────────────────────────────────────
    lines.append(f"=== TURN {obs['turn']} ===")
    lines.append(f"GOAL: {goal}")
    lines.append("")

    # ── Agent status ──────────────────────────────────────────────
    lines.append("YOUR STATUS")
    lines.append(f"  Position : ({agent['position']['x']}, {agent['position']['y']})")
    lines.append(f"  Health   : {agent['health']}")
    inv = agent["inventory"]
    if inv:
        inv_str = ", ".join(f"{i['name']} ({i['id']})" for i in inv)
        lines.append(f"  Inventory: {inv_str}")
    else:
        lines.append("  Inventory: (empty)")
    lines.append(f"  Standing on: {obs['current_tile']}")
    lines.append("")

    # ── Immediate surroundings ────────────────────────────────────
    lines.append("ADJACENT CELLS")
    for direction in ["north", "east", "south", "west"]:
        cell = adj.get(direction, {})
        tile_desc = cell.get("tile", "unknown")
        passable  = cell.get("passable", False)
        locked    = cell.get("locked_door", False)
        ents      = cell.get("entities", [])
        status    = "PASSABLE" if passable else ("LOCKED DOOR" if locked else "BLOCKED")
        ent_str   = f"  [entities: {', '.join(ents)}]" if ents else ""
        lines.append(f"  {direction.upper():5s} → {tile_desc} ({status}){ent_str}")
    lines.append("")

    # ── Minimap ───────────────────────────────────────────────────
    lines.append("MINIMAP (@ = you, # = wall, . = floor, ~ = water, D = locked door)")
    for row in obs["minimap"].split("\n"):
        lines.append("  " + row)
    lines.append("")

    # ── Visible entities ──────────────────────────────────────────
    nearby_ents = []
    for tile_info in obs.get("visible_tiles", []):
        for e in tile_info.get("entities", []):
            nearby_ents.append((tile_info["relative"], e))

    if nearby_ents:
        lines.append("NEARBY ENTITIES")
        for rel, e in nearby_ents:
            pickable = " [PICKABLE]" if e["pickable"] else ""
            talkable = " [TALKABLE]" if e["talkable"] else ""
            lines.append(f"  {e['name']} ({e['id']}) — {rel}{pickable}{talkable}")
            lines.append(f"    ↳ {e['description']}")
        lines.append("")

    # ── Recent events ─────────────────────────────────────────────
    recent = obs.get("recent_events", [])
    if recent:
        lines.append("RECENT EVENTS")
        for evt in recent:
            lines.append(f"  • {evt}")
        lines.append("")

    # ── Action history summary (last 6 turns) ─────────────────────
    if history:
        lines.append("YOUR RECENT ACTIONS")
        for h in history[-6:]:
            result_str = "✓" if h["success"] else "✗"
            lines.append(f"  Turn {h['turn']:3d}: [{result_str}] {h['action']} → {h['message'][:80]}")
        lines.append("")

    return "\n".join(lines)


def _parse_response(text: str) -> tuple[Optional[str], Optional[dict]]:
    """Extract <reasoning> and <action> from the LLM response."""
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    action_match    = re.search(r"<action>\s*(\{.*?\})\s*</action>", text, re.DOTALL)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    action    = None
    if action_match:
        try:
            action = json.loads(action_match.group(1))
        except json.JSONDecodeError:
            pass

    return reasoning, action


class LLMAgent:
    """
    Wraps the Claude API and maintains the conversation / history loop.
    """

    def __init__(self, world: GridWorld, goal: str, max_turns: int = 30):
        self.world     = world
        self.goal      = goal
        self.max_turns = max_turns
        self.history: list[dict] = []   # action history
        self.conversation: list[dict] = []  # multi-turn messages

    def step(self) -> dict:
        """Take one agent step: observe → reason → act → return result."""
        obs     = self.world.get_observation()
        prompt  = _build_user_prompt(obs, self.goal, self.history)

        # Multi-turn conversation: append the new observation as a user message
        self.conversation.append({"role": "user", "content": prompt})

        response = _client.messages.create(
            model=MODEL,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=self.conversation,
        )

        raw_text = response.content[0].text
        # Keep the assistant turn in conversation history
        self.conversation.append({"role": "assistant", "content": raw_text})

        reasoning, action = _parse_response(raw_text)

        if action is None:
            result = {"success": False, "message": "LLM returned unparseable action."}
            action = {"action": "wait"}
        else:
            result = self.world.execute_action(action)

        record = {
            "turn":      obs["turn"],
            "reasoning": reasoning,
            "action":    action,
            "success":   result["success"],
            "message":   result.get("message", ""),
            "raw":       raw_text,
        }
        self.history.append(record)
        return record

    def run(self, goal_check=None, verbose: bool = True) -> list[dict]:
        """
        Run the agent loop until max_turns or goal_check returns True.
        goal_check(world) -> bool
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"  GOAL: {self.goal}")
            print(f"{'='*60}\n")
            print(self.world.render())
            print()

        for turn_i in range(self.max_turns):
            record = self.step()

            if verbose:
                self._print_step(record, turn_i + 1)
                print()
                print(self.world.render())
                print()

            if goal_check and goal_check(self.world):
                if verbose:
                    print("\n🎉  GOAL ACHIEVED!\n")
                break

            time.sleep(0.3)   # small delay to be polite to the API

        return self.history

    def _print_step(self, record: dict, step_num: int):
        print(f"── Step {step_num} / Turn {record['turn']} ──────────────────────────")
        if record["reasoning"]:
            print(f"🧠  Reasoning: {record['reasoning']}")
        action = record["action"]
        print(f"⚡  Action: {json.dumps(action)}")
        status = "✓" if record["success"] else "✗"
        print(f"{status}  Result: {record['message']}")
