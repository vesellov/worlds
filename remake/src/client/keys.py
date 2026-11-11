"""
The hotkeys of the game: config/keyboard.ini through the options of the remake (engine.keys holds the
actions and the names of the keys) and the actions of the client.

The actions of the client which the remake has: the walk, the run, the sneak, the crawl, the pause,
the follow of the camera, the spells, the called shots, the windows of the battle screen (the text, the
information modes, the minimap), the camera (the normal view, four preset views, the track of the hero
and the zoom). The actions without a behaviour yet (the items, the weapons, the group of the party,
the swarm, the science, the quick save) answer "not made yet" in the message window. The arrows are
left to client.main (the camera pans while they are held).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math

#------------------------------------------------------------------------------

from eilib import keys as keys_module
from engine import equipment as equipment_module
from engine import hud as hud_module
from engine import quickitems as quickitems_module
from engine import saveload as saveload_module
from engine import stats as stats_module
from client import picking


#------------------------------------------------------------------------------

# the modes of the movement dial (649150), the order of engine.hud.MODE_DIAL_SECTORS
MODE_CRAWL = 0
MODE_SNEAK = 1
MODE_WALK = 2
MODE_RUN = 3

# the preset views of camera1..4 (the remake's reading: the camera has no saved presets in game.exe)
CAMERA_PRESET_YAW = {
    'camera1': 0.0,
    'camera2': 90.0,
    'camera3': 180.0,
    'camera4': 270.0,
}
CAMERA_ZOOM_STEP = 2.0
# the height of the chest of the hero, the target of client.main (TARGET_HEIGHT)
CAMERA_TARGET_HEIGHT = 1.0

# S of use_science: how far the remake looks for a lever around the hero
SCIENCE_SEARCH_DISTANCE = 8.0

# the windows of the battle screen: the action -> the flag of client.hud.BattleScreen.hidden
WINDOW_FLAGS = {
    'w_text1': 'text',
    'w_text2': 'info',
    'w_minimap': 'minimap',
}

# the actions which client.main still reads from the held keys (the arrows of the camera)
HELD_ACTIONS = ('camera_up', 'camera_down', 'camera_left', 'camera_right')

# the named parts of the called shots, client.picking.CALLED_SHOT_NAMES gives the numbers
CALLED_SHOT_LABELS = {
    'cs_head': 'head',
    'cs_body': 'body',
    'cs_lhand': 'left hand',
    'cs_rhand': 'right hand',
    'cs_lleg': 'left leg',
    'cs_rleg': 'right leg',
}

#------------------------------------------------------------------------------


class Keys(object):
    """
    The bindings of the options and the actions of the client.
    """

    def __init__(self, app):
        self.app = app
        self.bindings = {}
        self.by_kivy = {}
        self.reload()

    def __repr__(self):
        return 'Keys(%d bindings)' % len(self.bindings)

    def reload(self):
        """
        The bindings are read again after the options are confirmed.
        """
        self.bindings = keys_module.bindings(self.app.settings.values)
        self.by_kivy = {}
        for action, key in self.bindings.items():
            kivy = keys_module.kivy_name(key)
            if kivy:
                self.by_kivy[kivy] = action

    def action_for(self, name):
        return self.by_kivy.get(name)

    def on_key(self, name):
        app = self.app
        if app.args.kind == 'model' or app.main_menu is not None or app.battle is None:
            return False
        if app.esc_menu is not None and app.esc_menu.is_open:
            return False
        if app.keys_down & {'ctrl', 'lctrl', 'rctrl'}:
            # the keys of the remake with Ctrl (the camp, the inventory, the debug views)
            return False
        action = self.action_for(name)
        if action is None:
            return False
        if action in HELD_ACTIONS:
            return False
        return self.do(action)

    def message(self, text):
        self.app.show_message(text)

    def do(self, action):
        app = self.app
        if action in ('run', 'walk', 'sneak', 'crawl'):
            return self.set_mode(action)
        if action == 'pause':
            app.paused = not app.paused
            return True
        if action == 'tutorial_script':
            return self.show_tutorial()
        if action == 'obj':
            return self.toggle_object_names()
        if action == 'swarm':
            return self.toggle_swarm()
        if action == 'use_science':
            return self.use_science()
        if action.startswith('select'):
            return self.select_members(action)
        if action.startswith('weapon'):
            return self.choose_weapon(action)
        if action.startswith('item'):
            return self.use_quick_item(action)
        if action in ('quicksave', 'quickload'):
            return self.quick_save(action)
        if action in CALLED_SHOT_LABELS:
            return self.called_shot(action)
        if action.startswith('spell'):
            return self.choose_spell(action)
        if action in WINDOW_FLAGS or action in ('w_info1', 'w_info2', 'w_info3', 'w_info4'):
            return self.window_action(action)
        if action == 'camera_norm':
            app.reset_camera()
            return True
        if action in CAMERA_PRESET_YAW:
            return self.camera_preset(action)
        if action == 'camera_track':
            return self.camera_track()
        if action == 'camera_zoom_in':
            app.scene.camera.zoom_by(-CAMERA_ZOOM_STEP)
            return True
        if action == 'camera_zoom_out':
            app.scene.camera.zoom_by(CAMERA_ZOOM_STEP)
            return True
        self.message('%s: not made yet' % action.replace('_', ' '))
        return True

    def set_mode(self, action):
        battle = self.app.battle
        if battle is None:
            return False
        mode = {'crawl': MODE_CRAWL, 'sneak': MODE_SNEAK, 'walk': MODE_WALK, 'run': MODE_RUN}[action]
        battle.mode_index = mode
        self.app.gait = {MODE_CRAWL: stats_module.SPEED_CRAWL, MODE_SNEAK: stats_module.SPEED_SNEAK,
                         MODE_WALK: stats_module.SPEED_WALK, MODE_RUN: stats_module.SPEED_RUN}[mode]
        self.app.scene.set_hud_layer('mode', hud_module.mode_dial(mode).elements)
        return True

    def show_tutorial(self):
        tutorial = getattr(self.app, 'tutorial', None)
        if tutorial is None or tutorial.window.last is None:
            self.message('Tutorial: no page')
            return True
        tutorial.window.show_last()
        return True

    def toggle_object_names(self):
        """
        TAB: the names and the health of the units around the hero over the world (the remake's reading of
        the action "obj"; the rule of game.exe is not found).
        """
        app = self.app
        app.show_object_names = not getattr(app, 'show_object_names', False)
        self.message('Object names: %s' % ('on' if app.show_object_names else 'off'))
        if app.battle is not None:
            app.battle.update_object_names()
        return True

    def toggle_swarm(self):
        app = self.app
        app.swarm = not getattr(app, 'swarm', False)
        self.message('Party: %s' % ('together' if app.swarm else 'apart'))
        return True

    def use_science(self):
        """
        S: the hero goes to the nearest lever and tries his Use/Steal on it (engine.lever.Lever.use).
        """
        app = self.app
        if app.unit is None:
            return False
        best = None
        for lever in app.levers:
            distance = math.hypot(lever.position[0] - app.unit.x, lever.position[1] - app.unit.y)
            if best is None or distance < best[0]:
                best = (distance, lever)
        if best is None or best[0] > SCIENCE_SEARCH_DISTANCE:
            self.message('Use/Steal: no lever near')
            return True
        app.go_to_lever(best[1])
        return True

    def select_members(self, action):
        """
        F1..F4: the members of the party in the zone (RedeployParty brings all of them); the chosen units
        take the orders of the mouse.
        """
        app = self.app
        if action == 'select_all':
            units = app.select_party()
            self.message('Party selected: %d' % len(units))
            return True
        try:
            index = int(action[len('select'):]) - 1
        except ValueError:
            return False
        units = app.select_party(index)
        if units:
            self.message('Member %d: %s' % (index + 1, units[0].name))
        else:
            self.message('Member %d: none' % (index + 1))
        return True

    def choose_weapon(self, action):
        """
        Q/W/E/R: the weapon of the hero by the order of the weapons in his things and in the party; the old
        one goes back to the party, the new one is taken from it (engine.equipment, app.replace_hero).
        """
        app = self.app
        item_names = getattr(app, 'hero_item_names', None)
        if app.unit is None:
            return False
        item_names = list(item_names or [])
        try:
            index = int(action[len('weapon'):]) - 1
        except ValueError:
            return False
        items = app.game_data.items()
        party_items = getattr(app.party, 'items', None)
        if party_items is None:
            return False
        equipment = getattr(app.unit.figure, 'equipment', None)
        current_weapon = equipment.weapon.name if equipment is not None and equipment.weapon is not None else None
        armor_names = [item.name for item in equipment.armors.values()] if equipment is not None else []

        def weapon(name):
            item = equipment_module.make_item(items, name)
            return item is not None and item.is_weapon

        candidates = []
        for name in item_names + ([current_weapon] if current_weapon else []) + list(party_items):
            if weapon(name) and name not in candidates:
                candidates.append(name)
        if not 0 <= index < len(candidates):
            self.message('Weapon %d: none' % (index + 1))
            return True
        target = candidates[index]
        worn = [name for name in item_names if not weapon(name)]
        for name in armor_names:
            if name not in worn:
                worn.append(name)
        old = [name for name in item_names if weapon(name)]
        if current_weapon is not None and current_weapon not in old:
            old.append(current_weapon)
        if target in old:
            self.message('Weapon: %s' % target)
            return True
        if old:
            party_items.append(old[0])
        if target in party_items:
            party_items.remove(target)
        new_names = worn + [target]
        app.hero_item_names = list(new_names)
        app.replace_hero(new_names)
        self.message('Weapon: %s' % target)
        return True

    def use_quick_item(self, action):
        """
        P/O/I/U: the potions of the belt. The quick items are the things of the party with a record in the
        section 4 of items.idb; the used one leaves the party (engine.quickitems).
        """
        app = self.app
        if app.unit is None:
            return False
        try:
            index = int(action[len('item'):]) - 1
        except ValueError:
            return False
        names = getattr(app.party, 'items', None)
        if names is None:
            return False
        belt = quickitems_module.belt(names, app.game_data.quick_items(), order=app.quick_belt_names())
        if not 0 <= index < len(belt):
            self.message('Quick item %d: none' % (index + 1))
            return True
        quick_item = belt[index][0]
        used, healed = quickitems_module.use(app.unit, quick_item, app.game_data.spells())
        if not used:
            self.message('%s: can not use' % quick_item.name)
            return True
        quickitems_module.take(names, quick_item)
        self.sync_party_strip()
        if app.battle is not None:
            app.battle.update_static()
        if healed > 0.0:
            self.message('%s: +%d health' % (quick_item.name, round(healed)))
        else:
            self.message('%s: used' % quick_item.name)
        return True

    def sync_party_strip(self):
        strip = getattr(getattr(self.app, 'battle', None), 'inventory', None)
        if strip is None:
            return
        items = [equipment_module.make_item(self.app.game_data.items(), name) for name in self.app.party.items]
        strip.items = [item for item in items if item is not None]

    def quick_save(self, action):
        """
        F5/F8: the quick save and load through the API of the main flow (app.save_game, app.load_game),
        which is not made yet; the stub reports it.
        """
        app = self.app
        method = getattr(app, 'save_game' if action == 'quicksave' else 'load_game', None)
        if method is None:
            self.message('%s: not made yet' % action.replace('_', ' '))
            return True
        method(saveload_module.QUICK_SLOT_NAME)
        return True

    def called_shot(self, action):
        location = picking.CALLED_SHOT_NAMES.get(action)
        if location is None:
            return False
        if self.app.aim_part == location:
            self.app.aim_part = None
            self.message('Called shot: off')
        else:
            self.app.aim_part = location
            self.message('Called shot: %s' % CALLED_SHOT_LABELS[action])
        self.app.cursor.point = None
        return True

    def choose_spell(self, action):
        app = self.app
        try:
            index = int(action[len('spell'):]) - 1
        except ValueError:
            return False
        spells = list(getattr(getattr(app, 'demo', None), 'spells', []) or [])
        if app.spell is not None and app.spell not in spells:
            spells.insert(0, app.spell)
        if not 0 <= index < len(spells):
            self.message('Spell %d: not known' % (index + 1))
            return True
        app.spell = spells[index]
        self.message('Spell: %s' % app.spell.name)
        return True

    def window_action(self, action):
        battle = self.app.battle
        if battle is None:
            return False
        if action in WINDOW_FLAGS:
            flag = WINDOW_FLAGS[action]
            battle.hidden ^= {flag}
            battle.update_static()
            battle.update_minimap(0)
            return True
        modes = ('w_info1', 'w_info2', 'w_info3', 'w_info4')
        battle.info_mode = modes.index(action)
        battle.hidden.discard('info')
        battle.update_static()
        return True

    def camera_preset(self, action):
        camera = self.app.scene.camera
        camera.yaw = CAMERA_PRESET_YAW[action]
        self.app.reset_camera()
        return True

    def camera_track(self):
        """
        Home, the Tracking camera of game.exe: the camera tracks the selected character until the key is
        pressed again ("tip action_camera_track"); it is off at the start.
        """
        app = self.app
        if app.unit is None:
            return False
        app.args.follow = not app.args.follow
        if app.args.follow:
            app.scene.camera.target[:] = (app.unit.x, app.unit.y, app.unit.z + CAMERA_TARGET_HEIGHT)
        self.message('Tracking camera: %s' % ('on' if app.args.follow else 'off'))
        return True
