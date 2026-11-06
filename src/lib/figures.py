"""
Properties of the figures of the units, shared by the server and the client.

Every animation in catalog/animations.json has "movement_speed": how fast the figure moves over the ground
while the animation is played at its usual pace, in the units of the models per second. The server gives every unit
the speed of the animation it plays, see server.simulation.Simulation.unit_speed(), and the client plays
the animation as fast as the unit really moves, see client.cli.GameClient.animation_rate(), so the feet
of a slow zombie and of a fast tiger both stay on the ground.

The catalog has no flag for the figures which fly: they are listed in FLYING_FIGURES.
Figures which walk are placed so their lowest point is on the ground, see client.scen.Scene.figure_ground_offset().
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from lib import const
from lib import models

#------------------------------------------------------------------------------

# models are drawn MODELS_SCALE_FACTOR times bigger, a segment is SEGMENT_SIZE units of the scene:
# a speed in the units of the models per second becomes segments per tick
SPEED_TO_SEGMENTS_PER_TICK = const.MODELS_SCALE_FACTOR / const.SEGMENT_SIZE / models.TICK_RATE

# all units move that many times faster than their animations were recorded for, their animations
# are played that much faster too. With 1.6 a man runs 0.12 segments per tick
MOVEMENT_SPEED_SCALE = 1.6

# used for the animations which do not move the figure, segments per tick
DEFAULT_SPEED = models.MAX_SPEED

# no unit is faster than that, segments per tick: a faster one could jump over a tree in one tick,
# the obstacles are checked only where the unit is at the end of the tick, see lib.physics
FASTEST_SPEED = 0.25

# figures which fly or float above the ground, they stay at the height recorded in their animations
FLYING_FIGURES = (
    'unanwiba',  # bat
    'unmoba1',  # banshee
    'unmoba2',  # banshee
    'unmobe0',  # beholder
    'unmobe1',  # beholder
    'unmobi',  # bird
    'unmocu',  # curse
    'unmodg',  # dragon
    'unmosu',  # succubus
    'unmowi',  # will-o'-wisp
)

#------------------------------------------------------------------------------


def is_flying(model_name):
    return model_name in FLYING_FIGURES


def animation_info(animations_catalog, model_name, animation_name):
    for action in animations_catalog.get(model_name, {}).get('actions', []):
        if action['action_name'] == animation_name:
            return action
    return None


def natural_speed(animations_catalog, model_name, animation_name, size=1.0):
    """
    Segments per tick a unit of that size (see lib.zone.unit_size()) moves while the animation is played
    at its usual pace, 0.0 when the animation does not move the figure. A bigger unit makes longer steps.
    """
    action = animation_info(animations_catalog, model_name, animation_name)
    if not action:
        return 0.0
    return float(action.get('movement_speed') or 0.0) * SPEED_TO_SEGMENTS_PER_TICK * size


def max_speed(animations_catalog, model_name, animation_name, size=1.0):
    """
    Segments per tick a unit moves with that animation, see MOVEMENT_SPEED_SCALE.
    """
    speed = natural_speed(animations_catalog, model_name, animation_name, size)
    if not speed:
        return DEFAULT_SPEED
    return round(min(speed * MOVEMENT_SPEED_SCALE, FASTEST_SPEED), 4)
