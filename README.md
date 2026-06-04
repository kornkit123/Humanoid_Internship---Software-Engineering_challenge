# LLM Agent in a Virtual World

An intelligent agent powered by Claude that perceives, reasons, and acts inside a 2D tile-based world to accomplish goal-directed tasks.

```
# # # # # # # # # # # #
# . . . . . . . . . . #
# . S . . . C . . . . #    S = Sign     C = Chest
# . . . . . . . . . . #    D = Locked Door
# # # # # # D # # # # #    K = Key      @ = Agent
# . . . . . . . . . . #
# . @ . . . . . . K . #
# . . . . . . . . . . #
# # # # # # # # # # # #
```

---

## Requirements

- Python 3.10+
- An Anthropic API key ([get one here](https://console.anthropic.com/))

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/llm-agent-world
cd llm-agent-world

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 4. Run a scenario
python main.py --scenario key_door
```

---

## Scenarios

| Scenario | Description | Goal |
|---|---|---|
| `key_door` | Classic dungeon room | Find the brass key, unlock the door, reach the chest |
| `messenger` | Town with two NPCs | Talk to the innkeeper, deliver her message to the blacksmith |
| `navigate` | Maze map | Navigate through a maze of walls to reach the red beacon |

```bash
python main.py --scenario key_door      # dungeon key puzzle
python main.py --scenario messenger     # NPC dialogue quest
python main.py --scenario navigate      # maze pathfinding
```

### Options

```
--scenario    Which scenario to run (key_door | messenger | navigate)
--max-turns   Max agent turns before stopping (default: 40)
--log PATH    Save JSON transcript to a custom path
--quiet       Suppress per-step terminal output
```

---

## Project Structure

```
llm-agent-world/
├── main.py                  # Entry point & goal checkers
├── requirements.txt
│
├── world/
│   ├── grid_world.py        # 2D world engine: tiles, entities, actions, observations
│   └── scenarios.py         # Three hand-crafted scenario builders
│
├── agent/
│   └── llm_agent.py         # LLM harness: prompt builder, API calls, response parser
│
└── logs/
    └── key_door_example.json  # Example run transcript
```

---

## Design Choices

### 1. Observation Representation

The agent receives a **richly-structured text observation** each turn, not raw JSON. The key insight is that LLMs reason better over human-readable text than over nested data structures.

The observation includes:

- **Agent status** — position, health, inventory
- **Adjacent cells summary** — what's immediately north/south/east/west with passability flags
- **ASCII minimap** — a spatial overview centred on the agent (radius 4)
- **Nearby entities list** — name, description, and affordances (pickable/talkable)
- **Recent events** — last 5 world events as a short log
- **Action history** — last 6 turns with success/fail status

The adjacent-cells summary is the most critical piece: it tells the agent *exactly* what it can and cannot do right now, preventing futile wall-bumping.

### 2. Action Space Design

Six verbs cover the full interaction space without overwhelming the model:

```
move(direction)        — spatial navigation
pickup(target)         — item collection
use(item, target)      — item-environment interaction (keys, tools)
examine(target)        — information gathering
talk(target)           — NPC dialogue
wait                   — explicit no-op
```

Each action maps to a clean JSON object. The harness returns a `{success, message}` result that the agent reads in its next observation via the event log — closing the feedback loop.

### 3. Multi-Turn Conversation

Rather than starting a fresh API call each turn, the harness accumulates the **full conversation history** and sends it with each request. This means the agent has access to all its previous reasoning, not just the current observation — mimicking working memory.

### 4. Structured Reasoning Format

The agent is prompted to wrap its response in `<reasoning>...</reasoning>` and `<action>...</action>` XML tags. This:
- Makes chain-of-thought explicit and inspectable
- Forces the model to reason before committing to an action
- Makes the action trivially parseable (no brittle regex on free text)

### 5. What Worked

- **Adjacency summary** was the biggest win — once the agent knew what it *can't* do, it stopped wasting turns on walls
- **Event log feedback** closes the loop elegantly: failed actions explain *why* they failed, and the agent self-corrects
- **Multi-turn conversation** gives the agent a sense of continuity and lets it remember what it has already tried

### 6. What Could Be Improved

- **No backtracking memory** — the agent doesn't maintain an internal map, so in the maze scenario it can re-visit dead ends
- **Goal decomposition** — complex multi-step goals could benefit from an explicit sub-goal planner layer
- **Richer entity interactions** — combat, crafting, or multi-item puzzles would stress-test the harness further

---

## 📋 Example Output

```
============================================================
  GOAL: Find the brass key, use it to unlock the door, reach the chest.
============================================================

── Step 1 / Turn 1 ──────────────────────────────────────
    Reasoning: I can see the locked door (D) to my north. The K entity
    on the minimap is to the east. I'll head east to find the key.
    Action: {"action": "move", "direction": "east"}
✓  Result: You move east and stand on stone floor.

── Step 9 / Turn 9 ──────────────────────────────────────
    Reasoning: The Brass Key is here. Time to pick it up!
    Action: {"action": "pickup", "target": "brass key"}
✓  Result: You pick up Brass Key.

── Step 15 / Turn 15 ─────────────────────────────────────
    Reasoning: I can see the locked door to my north. I have the Brass
    Key — let me use it to unlock the door.
    Action: {"action": "use", "item": "Brass Key", "target": "door"}
✓  Result: You use Brass Key to unlock the door to the north!

    GOAL ACHIEVED!
```

See `logs/key_door_example.json` for a full transcript.

---

##    License

MIT
