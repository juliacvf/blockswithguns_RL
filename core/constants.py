"""Shared constants for Blocks With Guns RL."""

# World
GRID_SIZE = 100            # 100x100 cells
CELL = 1.0                 # world units per cell
WORLD = GRID_SIZE * CELL   # 100 world units

# Players
PLAYER_RADIUS = 0.35
PLAYER_SPEED = 3.5         # world units / second
PLAYER_LIVES = 8

# Shooting
BULLET_SPEED = 14.0
BULLET_RADIUS = 0.15
BULLET_TTL = 6.0           # seconds
BULLET_RANGE = 20.0        # max travel distance in blocks
SHOOT_COOLDOWN = 0.65      # seconds (30% slower between shots)
QUICKSHOT_COOLDOWN = 0.35  # 40% longer than the original 0.25 s powerup cooldown

# Powerups
POWERUP_FIRST_DELAY = 1.0   # 2.5x the original spawn frequency
POWERUP_INTERVAL = 1.0      # 150% more often than the original 2.5 s cadence
POWERUP_TTL = 25.0          # expires if not picked up within 25 s
POWERUP_RADIUS = 0.5
QUICKSHOT_DURATION = 5.0
SPEED_DURATION = 5.0
SPEED_MULTIPLIER = 1.5

# Simulation
FIXED_DT = 1.0 / 30.0      # engine tick
MAX_EPISODE_SECONDS = 120.0

# Heat zone: safe square centered on the map, shrinking HEAT_SHRINK_RATE
# blocks of half-width per second; players outside lose 1 life every
# HEAT_DAMAGE_PERIOD seconds.
HEAT_SHRINK_RATE = 0.25  # 50% slower shrink
HEAT_DAMAGE_PERIOD = 5.0

# Mud ponds: sparse perlin ponds on open grass that slow movement
MUD_SCALE = 14.0         # perlin scale for pond shapes
MUD_THRESHOLD = 0.74     # high = sparse ponds
MUD_SLOW = 0.5           # speed multiplier while standing in mud

# Trees: decorative scatter on open grass cells
TREE_DENSITY = 0.012     # fraction of open grass cells that get a tree

# Retreat: once per match, when a bullet drops a player to exactly
# RETREAT_LIVES, the victim teleports to a random free cell inside the
# heat-safe zone, far from the opponent (hits at 2 or 1 life do not trigger it)
RETREAT_LIVES = 3
RETREAT_MIN_DIST = 25.0  # min blocks from the opponent after a retreat

# Spawns (world units, center of cell)
SPAWN_A = (15.5, 15.5)
SPAWN_B = (84.5, 84.5)

# Powerup type ids (stable order used in one-hot encodings)
POWERUP_TYPES = ["LIFE", "QUICKSHOT", "SPEED", "SWAP"]
