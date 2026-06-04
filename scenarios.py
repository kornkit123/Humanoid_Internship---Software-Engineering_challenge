"""
Scenarios — hand-crafted levels for the LLM agent to explore.
Each returns a (world, goal_description) pair.
"""

from world.grid_world import (
    GridWorld, Entity, Tile,
    ENT_AGENT, ENT_KEY, ENT_CHEST, ENT_NPC, ENT_SIGN, ENT_ITEM,
    TILE_WALL, TILE_DOOR, TILE_WATER,
)


def _make_agent(x: int, y: int) -> Entity:
    return Entity(
        id="agent", type=ENT_AGENT, x=x, y=y,
        name="You", description="The agent.",
        data={"inventory": [], "health": 100},
    )


# ------------------------------------------------------------------
# Scenario 1 — "Find the key and unlock the door"
# ------------------------------------------------------------------

def scenario_key_door() -> tuple[GridWorld, str]:
    """
    Classic dungeon room: find the brass key, unlock the north door,
    reach the chest beyond it.
    """
    W, H = 12, 10
    world = GridWorld(W, H)
    world.border_walls()

    # Inner dividing wall with a locked door
    for x in range(1, W - 1):
        world.set_tile(x, 4, Tile(TILE_WALL, "#", False, "stone wall"))
    world.place_door(6, 4, locked=True, lock_key="brass_key")

    # Agent starts at bottom-left area
    world.add_entity(_make_agent(2, 7))

    # Brass key somewhere in the bottom room
    world.add_entity(Entity(
        id="brass_key", type=ENT_KEY, x=9, y=7,
        name="Brass Key", description="A small tarnished brass key.",
        pickable=True,
        data={"key_for": "brass_key"},
    ))

    # Sign with a hint
    world.add_entity(Entity(
        id="sign1", type=ENT_SIGN, x=2, y=2,
        name="Stone Sign", description="Carved text: 'The treasure lies beyond the locked door.'",
        talkable=True,
        data={"dialogue": "The treasure lies beyond the locked door. Seek the brass key."},
    ))

    # Treasure chest in the top room
    world.add_entity(Entity(
        id="chest1", type=ENT_CHEST, x=6, y=2,
        name="Treasure Chest", description="A heavy wooden chest bound in iron.",
        data={"contents": "100 gold coins", "opened": False},
    ))

    goal = (
        "Find the brass key somewhere in this room, use it to unlock the locked door "
        "in the wall to your north, then reach the treasure chest beyond it."
    )
    return world, goal


# ------------------------------------------------------------------
# Scenario 2 — "Explore and deliver a message"
# ------------------------------------------------------------------

def scenario_messenger() -> tuple[GridWorld, str]:
    """
    A town-square map. Talk to the Innkeeper to get a message,
    then deliver it to the Blacksmith.
    """
    W, H = 16, 14
    world = GridWorld(W, H)
    world.border_walls()

    # Buildings — just rooms with doors
    # Inn (top-left)
    for x in range(2, 7):
        world.set_tile(x, 2, Tile(TILE_WALL, "#", False, "stone wall"))
        world.set_tile(x, 6, Tile(TILE_WALL, "#", False, "stone wall"))
    for y in range(2, 7):
        world.set_tile(2, y, Tile(TILE_WALL, "#", False, "stone wall"))
        world.set_tile(6, y, Tile(TILE_WALL, "#", False, "stone wall"))
    world.place_door(4, 6)   # inn entrance

    # Blacksmith (bottom-right)
    for x in range(9, 14):
        world.set_tile(x, 8,  Tile(TILE_WALL, "#", False, "stone wall"))
        world.set_tile(x, 12, Tile(TILE_WALL, "#", False, "stone wall"))
    for y in range(8, 13):
        world.set_tile(9,  y, Tile(TILE_WALL, "#", False, "stone wall"))
        world.set_tile(13, y, Tile(TILE_WALL, "#", False, "stone wall"))
    world.place_door(11, 8)  # blacksmith entrance

    # Water feature — decorative pond
    for x in range(7, 10):
        for y in range(4, 7):
            world.set_tile(x, y, Tile(TILE_WATER, "~", False, "shimmering water"))

    # Agent in the town square
    world.add_entity(_make_agent(8, 10))

    # Innkeeper NPC
    world.add_entity(Entity(
        id="innkeeper", type=ENT_NPC, x=4, y=4,
        name="Innkeeper Mira", description="A stout woman with kind eyes.",
        talkable=True,
        data={"dialogue": "Thank goodness! I need you to deliver a message to Gareth the Blacksmith. Tell him: 'The shipment arrives at dawn.'"},
    ))

    # Blacksmith NPC
    world.add_entity(Entity(
        id="blacksmith", type=ENT_NPC, x=11, y=10,
        name="Gareth the Blacksmith", description="A broad-shouldered man covered in soot.",
        talkable=True,
        data={"dialogue": "A message from Mira? Excellent news — I'll have everything prepared by dawn."},
    ))

    # Signpost in square
    world.add_entity(Entity(
        id="signpost", type=ENT_SIGN, x=8, y=8,
        name="Town Signpost",
        description="Points: INN (north-west) | SMITHY (south-east)",
        talkable=True,
        data={"dialogue": "Town Square. Inn is to the north-west. Blacksmith's forge is to the south-east."},
    ))

    goal = (
        "Talk to Innkeeper Mira inside the Inn to the north-west to get her message, "
        "then find Gareth the Blacksmith in his forge to the south-east and talk to him "
        "to deliver the message."
    )
    return world, goal


# ------------------------------------------------------------------
# Scenario 3 — "Navigate to the red marker"
# ------------------------------------------------------------------

def scenario_navigate() -> tuple[GridWorld, str]:
    """
    A maze-like map. Navigate from start to reach the red beacon.
    Tests pure pathfinding + spatial reasoning.
    """
    W, H = 18, 14
    world = GridWorld(W, H)
    world.border_walls()

    # Maze walls
    walls = [
        # horizontal corridors
        *[(x, 3)  for x in range(2, 10)],
        *[(x, 7)  for x in range(6, 16)],
        *[(x, 10) for x in range(2, 12)],
        # vertical walls
        *[(4,  y) for y in range(4, 8)],
        *[(10, y) for y in range(1, 6)],
        *[(13, y) for y in range(8, 12)],
        *[(7,  y) for y in range(8, 11)],
    ]
    world.place_walls(walls)

    # Agent at top-left
    world.add_entity(_make_agent(2, 2))

    # Red beacon at bottom-right corner
    world.add_entity(Entity(
        id="red_beacon", type=ENT_ITEM, x=15, y=12,
        name="Red Beacon", description="A glowing red crystal mounted on a pedestal.",
        pickable=True,
        data={"colour": "red"},
    ))

    # Helpful guide NPC midway
    world.add_entity(Entity(
        id="guide", type=ENT_NPC, x=8, y=6,
        name="Wandering Guide", description="A hooded figure with a lantern.",
        talkable=True,
        data={"dialogue": "The red beacon is to the south-east. Look for the gaps in the walls!"},
    ))

    goal = (
        "Navigate through the maze from your starting position to reach the Red Beacon "
        "in the south-east corner of the map and pick it up."
    )
    return world, goal


SCENARIOS = {
    "key_door":  scenario_key_door,
    "messenger": scenario_messenger,
    "navigate":  scenario_navigate,
}
