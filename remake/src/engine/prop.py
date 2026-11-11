"""
An animated object without the logic of a unit: the stone columns of the main and the escape menus.

game.exe finds the clips of such objects in their .adb like for the units: 518fc0 takes the first action
of the type SPECIAL with the wanted modifier (the state does not matter), 512b20 the action with an exact key,
for example 0x80590000 - the cycle of the idle "cidle".
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import numpy as np

#------------------------------------------------------------------------------

from eilib import adb
from engine import animation
from engine import quat

#------------------------------------------------------------------------------

TYPE_SPECIAL = adb.TYPES.index('SPECIAL')

# game.exe 6333c0: the idle of the menu columns
IDLE_KEY = 0x80590000

KEY_MASK = ~adb.WEAPONS_MASK & 0xffffffff

#------------------------------------------------------------------------------


class Prop(object):

    def __init__(self, figure, world_matrix=None):
        self.figure = figure
        self.matrix = np.identity(4) if world_matrix is None else np.asarray(world_matrix, dtype=np.float64)
        self.player = animation.AnimationPlayer()
        # HideObject of the scripts: the prop is not drawn
        self.hidden = False

    def __repr__(self):
        return 'Prop(%s)' % self.figure.model_name

    def special_action(self, modifier):
        """
        The first action of the type SPECIAL with the modifier, game.exe 518fc0.
        """
        for action in self.figure.actions():
            if action.type == TYPE_SPECIAL and action.modifier == modifier:
                return action
        return None

    def action_by_key(self, key):
        for action in self.figure.actions():
            if action.key & KEY_MASK == key & KEY_MASK:
                return action
        return None

    def play(self, action):
        if action is None:
            return False
        clip = self.figure.clip(action)
        if clip is None:
            return False
        self.player.play(clip)
        return True

    def play_special(self, modifier):
        return self.play(self.special_action(modifier))

    def idle(self):
        return self.play(self.action_by_key(IDLE_KEY))

    @property
    def is_playing(self):
        return self.player.clip is not None and not self.player.finished

    def update(self, dt_ms, clock):
        self.player.add_time(clock.frames(dt_ms))

    def pose(self):
        return self.figure.skeleton.pose(self.player.tracks, self.player.time)

    def world_matrix(self):
        return self.matrix

#------------------------------------------------------------------------------


def placement_matrix(position, rotation, scale=1.0):
    """
    4x4 matrix of an object at the position with the rotation quaternion (w, x, y, z).
    """
    m = np.identity(4)
    m[0:3, 0:3] = quat.to_matrix(rotation) * scale
    m[0:3, 3] = position
    return m
