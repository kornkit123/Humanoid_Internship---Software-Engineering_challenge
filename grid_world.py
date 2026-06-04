"""
Grid World Engine
A 2D tile-based world where an LLM agent can exist and interact.
"""

from dataclasses import dataclass, field
from typing import Optional
import random
import json


# --- Tile Types ---
TILE_FLOOR   = "floor"
TILE_WALL    = "wall"
TILE_DOOR    = "door"      # passable once unlocked
TILE_WATER   = "water"     # impassable


# --- Entity Types ---
ENT_AGENT    = "agent"
ENT_KEY      = "key"
ENT_CHEST    = "chest"
ENT_NPC      = "npc"
ENT_SIGN     = "sign"
ENT_ITEM     = "item"


@dataclass
class Entity:
    id: str
    type: str
    x: int
    y: int
    name: str
    description: str
    pickable: bool = False
    talkable: bool = False
    data: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "description": self.description,
            "position": {"x": self.x, "y": self.y},
            "pickable": self.pickable,
            "talkable": self.talkable,
            "data": self.data,
        }


@dataclass
class Tile:
    type: str
    symbol: str
    passable: bool
    description: str
    locked: bool = False       # for doors
    lock_key: Optional[str] = None  # which key unlocks this door


class GridWorld:
    """
    The world state. Manages tiles, entities, and world rules.
    """

    DIRECTIONS = {
        "north": (0, -1),
        "south": (0,  1),
        "east":  (1,  0),
        "west":  (-1, 0),
    }

    def __init__(self, width: int, height: int):
        self.width  = width
        self.height = height
        self.tiles: list[list[Tile]] = []
        self.entities: dict[str, Entity] = {}
        self.agent_id: Optional[str] = None
        self.turn: int = 0
        self.messages: list[str] = []   # world-event log
        self._build_empty()

    # ------------------------------------------------------------------
    # World construction helpers
    # ------------------------------------------------------------------

    def _build_empty(self):
        self.tiles = [
            [Tile(TILE_FLOOR, ".", True, "stone floor") for _ in range(self.width)]
            for _ in range(self.height)
        ]

    def set_tile(self, x: int, y: int, tile: Tile):
        if self._in_bounds(x, y):
            self.tiles[y][x] = tile

    def place_walls(self, coords: list[tuple[int,int]]):
        wall = Tile(TILE_WALL, "#", False, "solid stone wall")
        for x, y in coords:
            self.set_tile(x, y, wall)

    def border_walls(self):
        for x in range(self.width):
            self.set_tile(x, 0, Tile(TILE_WALL, "#", False, "solid stone wall"))
            self.set_tile(x, self.height - 1, Tile(TILE_WALL, "#", False, "solid stone wall"))
        for y in range(self.height):
            self.set_tile(0, y, Tile(TILE_WALL, "#", False, "solid stone wall"))
            self.set_tile(self.width - 1, y, Tile(TILE_WALL, "#", False, "solid stone wall"))

    def place_door(self, x: int, y: int, locked: bool = False, lock_key: Optional[str] = None):
        symbol = "D" if locked else "d"
        desc   = "a locked wooden door" if locked else "an open doorway"
        self.set_tile(x, y, Tile(TILE_DOOR, symbol, not locked, desc, locked=locked, lock_key=lock_key))

    def add_entity(self, entity: Entity):
        self.entities[entity.id] = entity
        if entity.type == ENT_AGENT:
            self.agent_id = entity.id

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def tile_at(self, x: int, y: int) -> Optional[Tile]:
        if self._in_bounds(x, y):
            return self.tiles[y][x]
        return None

    def entities_at(self, x: int, y: int) -> list[Entity]:
        return [e for e in self.entities.values() if e.x == x and e.y == y]

    def get_agent(self) -> Optional[Entity]:
        if self.agent_id:
            return self.entities.get(self.agent_id)
        return None

    # ------------------------------------------------------------------
    # Observation builder  (this is the key harness piece)
    # ------------------------------------------------------------------

    def get_observation(self, vision_radius: int = 4) -> dict:
        """
        Build a rich, structured observation for the LLM agent.
        Includes: agent state, nearby tiles, visible entities, inventory,
        recent events, and a compact ASCII mini-map.
        """
        agent = self.get_agent()
        if not agent:
            return {}

        ax, ay = agent.x, agent.y

        # 1. Visible tiles in the vision radius
        visible_tiles = []
        for dy in range(-vision_radius, vision_radius + 1):
            for dx in range(-vision_radius, vision_radius + 1):
                nx, ny = ax + dx, ay + dy
                if not self._in_bounds(nx, ny):
                    continue
                tile = self.tiles[ny][nx]
                rel  = self._relative_direction(dx, dy)
                ents = [e.to_dict() for e in self.entities_at(nx, ny)
                        if e.id != self.agent_id]
                entry = {
                    "position": {"x": nx, "y": ny},
                    "offset": {"dx": dx, "dy": dy},
                    "relative": rel,
                    "tile_type": tile.type,
                    "tile_description": tile.description,
                    "passable": tile.passable,
                    "entities": ents,
                }
                if tile.type == TILE_DOOR:
                    entry["locked"] = tile.locked
                visible_tiles.append(entry)

        # 2. Adjacent cells (immediate surroundings summary)
        adjacents = {}
        for dir_name, (dx, dy) in self.DIRECTIONS.items():
            nx, ny = ax + dx, ay + dy
            tile   = self.tile_at(nx, ny)
            if tile:
                ents  = [e.name for e in self.entities_at(nx, ny) if e.id != self.agent_id]
                adjacents[dir_name] = {
                    "tile": tile.description,
                    "passable": tile.passable,
                    "entities": ents,
                    "locked_door": (tile.type == TILE_DOOR and tile.locked),
                }
            else:
                adjacents[dir_name] = {"tile": "void (out of bounds)", "passable": False}

        # 3. Inventory
        inventory = agent.data.get("inventory", [])

        # 4. ASCII mini-map
        mini_map = self._render_minimap(ax, ay, vision_radius)

        # 5. Current tile
        current_tile = self.tiles[ay][ax]

        return {
            "turn": self.turn,
            "agent": {
                "position": {"x": ax, "y": ay},
                "inventory": inventory,
                "health": agent.data.get("health", 100),
            },
            "current_tile": current_tile.description,
            "adjacent_cells": adjacents,
            "visible_tiles": visible_tiles,
            "minimap": mini_map,
            "recent_events": self.messages[-5:],  # last 5 world events
        }

    def _relative_direction(self, dx: int, dy: int) -> str:
        """Convert dx/dy offset to a readable direction phrase."""
        parts = []
        if dy < 0:
            parts.append(f"{abs(dy)} north")
        elif dy > 0:
            parts.append(f"{dy} south")
        if dx > 0:
            parts.append(f"{dx} east")
        elif dx < 0:
            parts.append(f"{abs(dx)} west")
        if not parts:
            return "here (your position)"
        return ", ".join(parts)

    def _render_minimap(self, cx: int, cy: int, radius: int) -> str:
        """Render an ASCII view centred on the agent."""
        lines = []
        for dy in range(-radius, radius + 1):
            row = []
            for dx in range(-radius, radius + 1):
                nx, ny = cx + dx, cy + dy
                if dx == 0 and dy == 0:
                    row.append("@")
                elif not self._in_bounds(nx, ny):
                    row.append(" ")
                else:
                    tile  = self.tiles[ny][nx]
                    ents  = self.entities_at(nx, ny)
                    non_agent = [e for e in ents if e.id != self.agent_id]
                    if non_agent:
                        row.append(non_agent[0].type[0].upper())
                    else:
                        row.append(tile.symbol)
            lines.append(" ".join(row))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Action execution  (the harness action interface)
    # ------------------------------------------------------------------

    def execute_action(self, action: dict) -> dict:
        """
        Execute an action dict returned by the LLM.
        Returns a result dict with success, message, and updated state.
        """
        self.turn += 1
        verb = action.get("action", "").lower()

        if verb == "move":
            return self._action_move(action.get("direction", ""))
        elif verb == "pickup":
            return self._action_pickup(action.get("target", ""))
        elif verb == "use":
            return self._action_use(action.get("item", ""), action.get("target", ""))
        elif verb == "examine":
            return self._action_examine(action.get("target", ""))
        elif verb == "talk":
            return self._action_talk(action.get("target", ""))
        elif verb == "wait":
            self._log("You wait a moment.")
            return {"success": True, "message": "You wait a moment."}
        else:
            return {"success": False, "message": f"Unknown action: '{verb}'. Valid actions: move, pickup, use, examine, talk, wait."}

    def _action_move(self, direction: str) -> dict:
        agent = self.get_agent()
        direction = direction.lower().strip()
        if direction not in self.DIRECTIONS:
            return {"success": False, "message": f"Invalid direction '{direction}'. Use: north, south, east, west."}

        dx, dy = self.DIRECTIONS[direction]
        nx, ny = agent.x + dx, agent.y + dy
        tile   = self.tile_at(nx, ny)

        if tile is None:
            return {"success": False, "message": "You can't move that way — void beyond the world."}
        if not tile.passable:
            if tile.type == TILE_DOOR and tile.locked:
                msg = f"The door to the {direction} is locked. You need a key."
            else:
                msg = f"You can't move {direction} — {tile.description} blocks the way."
            return {"success": False, "message": msg}

        agent.x, agent.y = nx, ny
        msg = f"You move {direction} and stand on {tile.description}."
        # pick up auto-collectables?
        ents = self.entities_at(nx, ny)
        notes = []
        for e in ents:
            if e.id != self.agent_id:
                notes.append(f"You see {e.name} here.")
        if notes:
            msg += " " + " ".join(notes)
        self._log(msg)
        return {"success": True, "message": msg}

    def _action_pickup(self, target: str) -> dict:
        agent = self.get_agent()
        ents  = [e for e in self.entities_at(agent.x, agent.y)
                 if e.id != self.agent_id and e.pickable
                 and (target.lower() in e.name.lower() or target.lower() in e.id.lower())]
        if not ents:
            return {"success": False, "message": f"No pickable item matching '{target}' here."}
        item = ents[0]
        inv  = agent.data.setdefault("inventory", [])
        inv.append({"id": item.id, "name": item.name, "description": item.description, "data": item.data})
        del self.entities[item.id]
        msg = f"You pick up {item.name}."
        self._log(msg)
        return {"success": True, "message": msg, "picked_up": item.name}

    def _action_use(self, item_name: str, target: str) -> dict:
        agent = self.get_agent()
        inv   = agent.data.get("inventory", [])
        item  = next((i for i in inv if item_name.lower() in i["name"].lower()), None)
        if not item:
            return {"success": False, "message": f"You don't have '{item_name}' in your inventory."}

        # Check adjacent tiles for locked door matching target
        for dir_name, (dx, dy) in self.DIRECTIONS.items():
            nx, ny = agent.x + dx, agent.y + dy
            tile   = self.tile_at(nx, ny)
            if tile and tile.type == TILE_DOOR and tile.locked:
                if (target.lower() in dir_name or
                        target.lower() in tile.description.lower() or
                        target.lower() == "door"):
                    if tile.lock_key and tile.lock_key == item["id"]:
                        tile.locked   = False
                        tile.passable = True
                        tile.symbol   = "d"
                        tile.description = "an unlocked wooden door"
                        msg = f"You use {item['name']} to unlock the door to the {dir_name}!"
                        self._log(msg)
                        return {"success": True, "message": msg}
                    else:
                        return {"success": False, "message": "This key doesn't fit that lock."}

        # Check entities on current tile
        adj_ents = []
        for dir_name, (dx, dy) in self.DIRECTIONS.items():
            adj_ents += self.entities_at(agent.x + dx, agent.y + dy)
        adj_ents += self.entities_at(agent.x, agent.y)

        for e in adj_ents:
            if target.lower() in e.name.lower() or target.lower() in e.id.lower():
                if e.type == ENT_CHEST and "chest_key" in item.get("id", ""):
                    e.data["opened"] = True
                    contents = e.data.get("contents", "nothing")
                    msg = f"You open the chest with {item['name']}! Inside: {contents}."
                    self._log(msg)
                    return {"success": True, "message": msg}

        return {"success": False, "message": f"You can't use {item['name']} on '{target}' right now."}

    def _action_examine(self, target: str) -> dict:
        agent  = self.get_agent()
        # Check inventory first
        inv    = agent.data.get("inventory", [])
        inv_item = next((i for i in inv if target.lower() in i["name"].lower()), None)
        if inv_item:
            return {"success": True, "message": f"{inv_item['name']}: {inv_item['description']}"}

        # Check nearby entities
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                for e in self.entities_at(agent.x + dx, agent.y + dy):
                    if target.lower() in e.name.lower() or target.lower() in e.id.lower():
                        return {"success": True, "message": f"{e.name}: {e.description}"}

        # Check current / adjacent tiles
        tile = self.tile_at(agent.x, agent.y)
        if target.lower() in tile.description.lower() or target.lower() in tile.type.lower():
            return {"success": True, "message": f"The tile you stand on: {tile.description}."}

        return {"success": False, "message": f"You don't see anything called '{target}' nearby."}

    def _action_talk(self, target: str) -> dict:
        agent = self.get_agent()
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                for e in self.entities_at(agent.x + dx, agent.y + dy):
                    if e.talkable and (target.lower() in e.name.lower() or target.lower() in e.id.lower()):
                        dialogue = e.data.get("dialogue", "…")
                        msg = f"{e.name} says: \"{dialogue}\""
                        self._log(msg)
                        return {"success": True, "message": msg}
        return {"success": False, "message": f"No one named '{target}' nearby to talk to."}

    def _log(self, msg: str):
        self.messages.append(msg)

    # ------------------------------------------------------------------
    # Full-world render (for debug / human viewer)
    # ------------------------------------------------------------------

    def render(self) -> str:
        agent = self.get_agent()
        rows  = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                ents = self.entities_at(x, y)
                non_agent = [e for e in ents if e.id != self.agent_id]
                if agent and agent.x == x and agent.y == y:
                    row.append("@")
                elif non_agent:
                    e = non_agent[0]
                    row.append(e.type[0].upper())
                else:
                    row.append(self.tiles[y][x].symbol)
            rows.append(" ".join(row))
        return "\n".join(rows)
