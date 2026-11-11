"""
The cursor over the world and the windows: the ground, an enemy, a peaceful unit with topics, a lever or a
body, the sky and the target of a spell of the Ctrl key.

game.exe loads the cursors in LoadCursors (5daea0): cursor_default, cursor_attack, cursor_talk, cursor_use,
cursor_move, cursor_steal, cursor_spell, cursor_spellcancel, cursor_wait, cursor_camera, cursor_cancel and the
eight cursor_scroll_* with their hot points (engine.ui.CURSOR_HOT_POINTS). The rules of the choice are kept
by the cursor manager of the game; the remake chooses them from the target under the mouse and keeps the
default one over the panels, the windows and the menus. The six cursors of the hit locations
(cursor_attack_hd/bd/lh/rh/ll/rl) need the part of the body under the mouse which the main flow does not
pick yet, so the remake shows cursor_attack over a whole unit.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from engine import magic
from client import picking
from client import ui

#------------------------------------------------------------------------------

DEFAULT = 'cursor_default'
MOVE = 'cursor_move'
ATTACK = 'cursor_attack'
TALK = 'cursor_talk'
USE = 'cursor_use'
SPELL = 'cursor_spell'
SPELL_CANCEL = 'cursor_spellcancel'

# the scroll cursors of the borders of the screen (LoadCursors 5daea0, hot points of engine.ui)
EDGE_SIZE = 6
EDGE_CURSORS = {
    (-1, -1): 'cursor_scroll_ul', (0, -1): 'cursor_scroll_u', (1, -1): 'cursor_scroll_ur',
    (-1, 0): 'cursor_scroll_l', (1, 0): 'cursor_scroll_r',
    (-1, 1): 'cursor_scroll_dl', (0, 1): 'cursor_scroll_d', (1, 1): 'cursor_scroll_dr',
}

#------------------------------------------------------------------------------


class CursorChooser(object):
    """
    The name of the cursor for the point of the mouse; the ground pick is done only when the mouse moved.
    """

    def __init__(self, app):
        self.app = app
        # the position (halved) and the name of the cursor which is shown
        self.point = None
        self.name = None

    def update(self, pos, over_interface=False):
        """
        pos is the position of the mouse in the pixels of the window.
        """
        app = self.app
        if over_interface or app.tutorial_open() or getattr(app, 'movie', None) is not None:
            self.set(DEFAULT, pos)
            return
        edge = self.edge_at(pos)
        if edge is not None:
            self.set(EDGE_CURSORS[edge], pos)
            return
        self.set(self.name_at(pos), pos)

    def edge_at(self, pos):
        """
        (-1, 0, 1) of the border of the screen under the mouse, None away from the borders.
        """
        app = self.app
        if getattr(app, 'battle', None) is None:
            return None
        x, y = app.interface.virtual_point(pos[0], pos[1])
        horizontal = -1 if x < EDGE_SIZE else (1 if x > ui.WIDTH - EDGE_SIZE else 0)
        vertical = -1 if y < EDGE_SIZE else (1 if y > ui.HEIGHT - EDGE_SIZE else 0)
        if horizontal == 0 and vertical == 0:
            return None
        return horizontal, vertical

    def set(self, name, pos=(0.0, 0.0)):
        key = (int(pos[0]) // 2, int(pos[1]) // 2, name)
        if key == self.point:
            return
        self.point = key
        self.name = name
        self.app.interface.cursor.set(name)

    def ctrl_down(self):
        return bool(self.app.keys_down & {'ctrl', 'lctrl', 'rctrl'})

    def name_at(self, pos):
        app = self.app
        if app.scene is None or app.terrain is None or app.unit is None:
            return DEFAULT
        part_hit = picking.pick_part(app, pos)
        point = app.scene.pick_ground(pos[0], pos[1])
        foe = app.unit_at(point) if point is not None else None
        if part_hit is not None and foe is None:
            foe = part_hit[0]
        if self.ctrl_down() and app.spell is not None:
            if foe is not None or app.spell.flags & magic.TARGET_POINT:
                return SPELL
            return SPELL_CANCEL
        if part_hit is not None:
            unit, location = part_hit
            if app.battle is not None and app.battle.can_talk(unit):
                return TALK
            return picking.cursor_for(app, unit, location)
        if foe is not None:
            if app.battle is not None and app.battle.can_talk(foe):
                return TALK
            return ATTACK
        if point is None:
            return DEFAULT
        if app.corpse_at(point) is not None:
            return USE
        if app.lever_at(point) is not None:
            return USE
        return MOVE
