"""
Choice of the animation for an action of a unit, as the server of game.exe does it.

matching()      sub_5190c0: the weapons of the action must include the weapons asked for,
                all other bits of the key must be equal.
choose()        several matching actions: one is picked at random, weighted by action_probability.
find()          sub_5301b0: when nothing matches, the weapons are dropped, then the state REST
                is replaced by NEUTRAL and by ATTACK, then the modifier and the stage are ignored,
                at last the state too (the looser search of sub_519380, its exact rules are not known yet).
                With no animation at all the original writes "Fucking ShITTT!!! ArtIsTs MuSt diE!!!" to its log.
schedule()      sub_550100: the action lasts animation_length / speed ticks
                and the hit comes hit_frame / speed ticks after the start.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import random

#------------------------------------------------------------------------------

from eilib import adb

#------------------------------------------------------------------------------

STATE_NEUTRAL = adb.STATES.index('NEUTRAL')
STATE_REST = adb.STATES.index('REST')
STATE_ATTACK = adb.STATES.index('ATTACK')

TYPE_SPECIAL = adb.TYPES.index('SPECIAL')
TYPE_ATTACK = adb.TYPES.index('ATTACK')
TYPE_CAST = adb.TYPES.index('CAST')
TYPE_RUN = adb.TYPES.index('RUN')
TYPE_WALK = adb.TYPES.index('WALK')
TYPE_IDLE = adb.TYPES.index('IDLE')
TYPE_DEATH = adb.TYPES.index('DEATH')
TYPE_SUFFER = adb.TYPES.index('SUFFER')
TYPE_CROSS = adb.TYPES.index('CROSS')

STAGE_UNIQUE = 0
STAGE_START = 1
STAGE_CYCLE = 2
STAGE_END = 3

WEAPON_BITS = {name: 1 << i for i, name in enumerate(adb.WEAPONS)}

MODIFIER_MASK = 0xff << adb.MODIFIER_SHIFT

STAGE_MASK = 3 << adb.STAGE_SHIFT

STATE_MASK = 7 << adb.STATE_SHIFT

# sub_5327d0: the action types which have their own speed in the unit, everything else plays at 1.0
SPEED_TYPES = ('ATTACK', 'CAST', 'SUFFER', 'DEATH', 'CROSS', 'SPECIAL')

#------------------------------------------------------------------------------


def matching(database, key, weapons=0, ignore_mask=0):
    found = []
    other = ~(adb.WEAPONS_MASK | ignore_mask) & 0xffffffff
    for action in database.actions:
        if (action.key & other) != (key & other):
            continue
        if (action.weapons & weapons) != weapons:
            continue
        found.append(action)
    return found


def choose(candidates, rng=random):
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    total = sum(a.probability for a in candidates)
    if total <= 0:
        return candidates[0]
    roll = rng.randrange(100) * total / 100.0
    for action in candidates:
        roll -= action.probability
        if roll < 0:
            return action
    return candidates[-1]


def find(database, state, action_type, modifier=0, stage=STAGE_UNIQUE, weapons=0, rng=random):
    """
    The action of the database for the wanted state, type, modifier, stage and weapons, None when there is none.
    """
    states = [state]
    if state == STATE_REST:
        states += [STATE_NEUTRAL, STATE_ATTACK]
    attempts = []
    for s in states:
        key = adb.make_key(state=s, action_type=action_type, modifier=modifier, stage=stage)
        attempts.append((key, weapons, 0))
        attempts.append((key, 0, 0))
    for s in states:
        key = adb.make_key(state=s, action_type=action_type, modifier=modifier, stage=stage)
        attempts.append((key, weapons, MODIFIER_MASK))
        attempts.append((key, 0, MODIFIER_MASK | STAGE_MASK))
    key = adb.make_key(state=state, action_type=action_type, modifier=modifier, stage=stage)
    attempts.append((key, weapons, STATE_MASK))
    attempts.append((key, 0, STATE_MASK | MODIFIER_MASK))
    attempts.append((key, 0, STATE_MASK | MODIFIER_MASK | STAGE_MASK))
    for key, weapons_wanted, ignore_mask in attempts:
        action = choose(matching(database, key, weapons_wanted, ignore_mask), rng=rng)
        if action is not None:
            return action
    if _Debug:
        print('no animation for state=%d type=%d modifier=%d stage=%d weapons=%x' % (
            state, action_type, modifier, stage, weapons))
    return None


def speed_multiplier(action, unit_speeds=None):
    """
    unit_speeds: action type name -> speed of that kind of actions of the unit (+0x4b0..+0x4c4 in CUnitServer).
    """
    if not unit_speeds or action is None:
        return 1.0
    return unit_speeds.get(action.type_name, 1.0) if action.type_name in SPEED_TYPES else 1.0


def schedule(action, speed=1.0):
    """
    (duration, hit time) of the action in ticks from its start.
    """
    speed = speed or 1.0
    return action.length / speed, action.hit_frame / speed


def weapon_bits(weapon_name):
    if not weapon_name:
        return 0
    return WEAPON_BITS[weapon_name.upper()]
