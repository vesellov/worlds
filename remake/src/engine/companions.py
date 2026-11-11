"""
The other units of the party in the zone: where they stand behind the hero, the points of a group order
and their following of the hero.

RedeployParty (590700) brings the whole current party into the zone; the units of the computer follow their
leader when the party goes together (the swarm of the options, "Party: together"), apart they stand until an
order. The remake has no CParty::Swarm of game.exe; the distance and the slots of the followers are its own.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math

#------------------------------------------------------------------------------

from engine import stats
from engine import unit as unit_module

#------------------------------------------------------------------------------

# the slots of the followers behind the hero: the first pair at this distance, the next pairs farther
FOLLOW_ROW = 1.6
# the side of the odd and the even followers of a row
FOLLOW_SIDE = 1.0
# a follower goes when it is farther than this from its slot
FOLLOW_DISTANCE = 1.0
# farther than this it runs instead of walking
RUN_DISTANCE = 6.0
# the radius of the ring of a group order
FORMATION_RADIUS = 1.2

#------------------------------------------------------------------------------


def follow_slot(hero, index):
    """
    The point behind the hero for the follower number index (1 for the first).
    """
    row = (index + 1) // 2
    side = FOLLOW_SIDE if index % 2 else -FOLLOW_SIDE
    forward_x, forward_y = math.sin(hero.yaw), -math.cos(hero.yaw)
    right_x, right_y = math.cos(hero.yaw), math.sin(hero.yaw)
    distance = FOLLOW_ROW * row
    return (hero.x - forward_x * distance + right_x * side,
            hero.y - forward_y * distance + right_y * side)


def formation_points(x, y, count, radius=FORMATION_RADIUS):
    """
    The points of a group order around (x, y): the first unit goes to the point, the others around it.
    """
    if count <= 0:
        return []
    if count == 1:
        return [(x, y)]
    points = [(x, y)]
    for index in range(1, count):
        angle = 2.0 * math.pi * (index - 1) / (count - 1)
        points.append((x + math.cos(angle) * radius, y + math.sin(angle) * radius))
    return points


def can_follow(unit):
    """
    A follower which stands and has no foe goes to its slot; a dead, frozen or fighting one stays.
    """
    if unit.is_dead or unit.foe is not None or unit.is_frozen:
        return False
    return unit.mode not in (unit_module.MODE_ATTACK, unit_module.MODE_ACTION)


def update_follow(hero, followers, together=True):
    """
    The followers behind the hero when the party goes together; False when the party goes apart.
    """
    if not together or hero is None or hero.is_dead:
        return False
    for index, unit in enumerate(followers, 1):
        if not can_follow(unit):
            continue
        x, y = follow_slot(hero, index)
        distance = math.hypot(x - unit.x, y - unit.y)
        if distance <= FOLLOW_DISTANCE or unit.mode != unit_module.MODE_IDLE:
            continue
        gait = stats.SPEED_RUN if distance > RUN_DISTANCE else stats.SPEED_WALK
        unit.go_to(x, y, gait=gait)
    return True
