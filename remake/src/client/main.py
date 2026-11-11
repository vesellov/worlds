"""
Entry point of the remake client.

    PYTHONPATH=src venv/bin/python -m client.main demo
    PYTHONPATH=src venv/bin/python -m client.main menu
    PYTHONPATH=src venv/bin/python -m client.main model unhuma
    PYTHONPATH=src venv/bin/python -m client.main map bz10k --unit unhuma --weapon SWORD
    PYTHONPATH=src venv/bin/python -m client.main model unhuma --item "hadagan high2 plate.adamantium" --item "saber.steel"

Model viewer: Left/Right - previous/next action of the .adb, Space - pause, B - next body proportions,
F - the fast clock of the battle (27 ms ticks), mouse drag - turn the camera, wheel - zoom.
Map: click - walk there, Shift + click - run, click on a unit - attack it, Ctrl + click on a unit - cast
the spell of --spell at it (a spell of the point also on the ground), 1 attack, 2 cast, 3 suffer, 4 death, 5 special, 0 idle,
arrows - move the camera, Q/E - turn it, F - fast clock, wheel - zoom, [ and ] - the hour of the day back and forth,
T - the time of the day runs fast (an hour in 10 seconds), Esc - the escape menu, H - the debug text.
The battle screen (client.hud): pause and speed on the dial of the lower right corner, walk/run on the left one,
the zoom buttons of the mini-map, the tooltips of texts.res.
Main menu: the buttons of the column as in game.exe, "New game" opens --start-map, R - outline the button regions.
Demo (client.demo): the main menu, "New game" brings the dressed hero with the book of the spells to zone1,
F1 - its keys: Z/X the spell, C cast it at the hero, W the next weapon, M full mana and health.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math
import os
import random
import sys
import time
import argparse

import numpy as np
from PIL import Image
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label

#------------------------------------------------------------------------------

from eilib import adb
from eilib import gamedata
from eilib import idb
from eilib import mapdata
from eilib import script as script_lang
from eilib import zones as zones_module
from engine import acks as acks_module
from engine import actions
from engine import ai
from engine import ambient as ambient_module
from engine import camera as camera_module
from engine import combat
from engine import companions as companions_module
from engine import difficulty as difficulty_module
from engine import equipment as equipment_module
from engine import experience
from engine import flyinghp as flyinghp_module
from engine import figure
from engine import lever as lever_module
from engine import lighting
from engine import lightning
from engine import magic
from engine import mercenaries as mercenaries_module
from engine import menus
from engine import missiles
from engine import music as music_module
from engine import options as options_module
from engine import party as party_module
from engine import particles
from engine import pathing
from engine import questlog as questlog_module
from engine import sounds
from engine import prop as prop_module
from engine import roster as roster_module
from engine import saveload as saveload_module
from engine import script
from engine import stats
from engine import terrain as terrain_module
from engine import timing
from engine import trade as trade_module
from engine import trap as trap_module
from engine import water
from engine import world
from engine import unit as unit_module
from engine import weather as weather_module
from client import audio
from client import cursor as cursor_view
from client import demo
from client import flyinghp as flyinghp_view
from client import scene
from client import menu
from client import music as music_view
from client import picking as picking_view
from client import hud as battle
from client import keys as keys_view
from client import mapscript
from client import movie as movie_view
from client import tutorial as tutorial_view
from client import weather as weather_view
from client import ui

#------------------------------------------------------------------------------

BODY_PRESETS = [(0.5, 0.5, 0.5), (0.0, 0.0, 0.0), (1.0, 0.0, 0.5), (0.0, 1.0, 0.5), (1.0, 1.0, 1.0)]

HOUR_STEP = 0.5

# how long an arrow stays in the body it hit (the remake's choice for the effect 0x5b)
ARROW_STICK_MS = 8000.0

# the ticks of engine.ambient in one frame at most
AMBIENT_MAX_TICKS = 2

AI_REG = 'config/ai.reg'

# 5f78c0 (partly, the key handler of the interface): Scroll Lock (the scan code 0x46) saves the screen as "shot%4d.bmp" with a counter from 0 (7c7e14)
SHOT_KEYS = ('screenlock', 'scrolllock')
SHOT_FILE_NAME = 'shot%4d.bmp'

TILE_DESC_REG = 'tileDesc.reg'

# the health part of the legs set by the key L: below DamageLevel1 and DamageLevelRunLimit of ai.reg
WOUNDED_LEGS = 0.4

# how far from a unit a click still picks it, map units
PICK_MARGIN = 0.5

# the remake's choices: a lever is picked this near to its point, the hero uses it from this distance
LEVER_PICK_RADIUS = 1.5
LEVER_USE_DISTANCE = 2.0

# the science of the hero for the levers: the flags 1, 2, 4 at any level, the keys are the quest items of the party
HERO_LEVER_FLAGS = lever_module.LEVEL_FLAGS | lever_module.KEY_FLAG
HERO_LEVER_LEVEL = 1000

# the texts.res lines of the briefings (6051xx) which the remake also shows when a script gives or takes a thing
PARTY_MESSAGES = {
    'item': 'string brief_get_item',
    'quest_item': 'string brief_get_item',
    'quest_item_lost': 'string brief_lost_item',
    'money': 'string brief_get_money',
    'quest': 'string brief_complete_quest',
}

# the remake's messages: texts.res has none for the levers
LEVER_LOCKED_MESSAGE = 'The lever does not move.'
LEVER_KEY_MESSAGE = 'A key is needed.'

# hours per second when the time of the day runs fast
FAST_HOURS_PER_SECOND = 0.1

ACTION_KEYS = {
    '1': actions.TYPE_ATTACK,
    '2': actions.TYPE_CAST,
    '3': actions.TYPE_SUFFER,
    '4': actions.TYPE_DEATH,
    '5': actions.TYPE_SPECIAL,
}

#------------------------------------------------------------------------------


class RemakeApp(App):

    def __init__(self, args, **kwargs):
        super(RemakeApp, self).__init__(**kwargs)
        self.args = args
        self.game_data = gamedata.GameData(args.data_dir_path)
        # 468a00: the noise of the lightning bolts
        gfx = self.game_data.archive('res/gfx.res')
        if lightning.NOISE_FILE_NAME in gfx:
            lightning.load_noise(gfx.read(lightning.NOISE_FILE_NAME))
        self.audio = audio.AudioPlayer(self.game_data, enabled=args.sound)
        # the tracks of data/stream by res/music.reg
        self.music = music_view.MusicPlayer(self.game_data, enabled=args.sound)
        self.clock = timing.TickClock()
        self.paused = False
        self.movie = None
        self.last_time = None
        # the frames counted for the answers kept for one frame (client.picking), the last move of the mouse
        self.frame_index = 0
        self.ambient = None
        self.ambient_random = random.Random()
        self.mouse_pending = None
        # GSSetVar / GSGetVar of the scripts, they live through the maps
        self.game_vars = {}
        # engine.questlog.QuestLog: the messages and the sounds of the changed variables of the quests
        self.quest_log = questlog_module.QuestLog()
        # engine.difficulty.NOVICE or NORMAL: the choice of the window of a new game (633770)
        self.difficulty = difficulty_module.DEFAULT
        # engine.options.Settings: the volumes, the difficulty and the other options of the remake file
        self.settings = options_module.load_settings()
        # 7c7e14: the number of the next picture of the screen (5f78c0 (partly))
        self.shot_index = 0
        options_module.load_hotkeys(self.settings, self.game_data.config_text('keyboard.ini'))
        self.settings.apply(self)
        # client.keys.Keys: the hotkeys of the options and config/keyboard.ini
        self.keys = keys_view.Keys(self)
        # the money, the items and the quests of the player, they live through the maps
        self.party = party_module.Party(self.game_data.items(), self.game_data.quests())
        self.party.on_change = self.on_party_change
        # engine.roster.Roster: the parties of the player between the zones, made by the first map of a game
        self.roster = None
        # the engine.roster.UnitRecord of the hero on the map
        self.hero_record = None
        self.unit = None
        # engine.unit.Unit of the other members of the current party in the zone
        self.companions = []
        # the units the orders of the mouse go to (F1..F4 choose them)
        self.selected_units = []
        # the party goes together (the followers go behind the hero) or apart
        self.swarm = True
        self.map_units = []
        self.props = []
        self.props_by_nid = {}
        # engine.camera.obstacles of the static objects of the map
        self.camera_obstacles = None
        # the logic of the units of the computer (engine.ai)
        self.ai = ai.Controller()
        # (engine.missiles.Arrow, its prop, its view) of the arrows in flight
        self.arrows = []
        # (prop, view, target, offset, matrix, ms left) of the arrows in a body
        self.stuck_arrows = []
        # engine.magic.AreaEffect of the spells on the ground
        self.spell_effects = []
        # engine.particles.Emitter which live alone: the smoke of a missile which came, the flashes of the hits
        self.loose_emitters = []
        # magic.SpellLight of the missiles which hit
        self.loose_lights = []
        # id of a unit -> the spells on it the last time (the sounds of the enchantments, 5112a0)
        self.enchantments_seen = {}
        # (engine.particles.Emitter, engine.unit.Unit) of the spells which follow their units
        self.unit_emitters = []
        # the part of a tick of the particles which went by
        self.particle_ticks = 0.0
        # engine.magic.Spell of the hero for Ctrl + click
        self.spell = None
        self.terrain = None
        self.path_grid = None
        # client.mapscript.MapFigure of the static objects of the map, their parts make the grid of the paths
        self.figures = []
        # client.mapscript.LeverObject of the map, their parts go into the grid of the paths by their state
        self.levers = []
        # engine.trap.MagicTrap of the map
        self.traps = []
        # the LeverObject the hero goes to use
        self.lever_to_use = None
        self.map_name = None
        # the .mob of the map (its camera is camera/<mob>.cam when the zone has no own one)
        self.mob_name = None
        self.world_light = None
        self.hour = world.DEFAULT_HOUR
        self.fast_hours = False
        self.body_index = 0
        self.action_index = 0
        self.keys_down = set()
        self.drag_start = None
        # the cursor under the mouse (client.cursor)
        self.cursor = cursor_view.CursorChooser(self)
        self.main_menu = None
        self.esc_menu = None
        self.battle = None
        # client.weather.WeatherView of the map or of the main menu
        self.weather = None
        # the hit location of the called shot (client.picking.CALLED_SHOT_NAMES), None for a usual blow
        self.aim_part = None
        # the gait of the movement dial: engine.stats.SPEED_RUN/WALK/SNEAK/CRAWL
        self.gait = stats.SPEED_WALK
        self.show_debug_text = args.kind == 'model'
        self.message = ''
        self.message_time_ms = 0.0
        self.help_time_ms = 0.0
        self.scene_camera_class = scene.FixedCamera
        # the model and the prototype of the hero on the map
        self.hero_model_name = args.unit
        self.hero_prototype_name = args.prototype
        # the prototype whose numbers the hero has (engine.roster.UnitRecord.stats_name), None for its own
        self.hero_stats_name = None
        # client.demo.Demo of "make demo", None in the other kinds
        self.demo = demo.Demo(self) if args.kind == 'demo' else None

    def build(self):
        self.title = 'Evil Islands remake - %s %s' % (self.args.kind, self.args.name or '')
        self.root_layout = FloatLayout()
        self.hud = Label(text='', halign='left', valign='top', markup=False, size_hint=(1, 1),
                         padding=(10, 10), font_size='14sp')
        self.hud.bind(size=lambda w, s: setattr(w, 'text_size', s))
        # the keys of the demo: low on the left, above the dial of the actions, away from the messages
        self.help_label = Label(text='', halign='left', valign='bottom', markup=False, size_hint=(1, 1),
                                padding=(10, 10, 10, 100), font_size='14sp', outline_width=1,
                                outline_color=(0, 0, 0))
        self.help_label.bind(size=lambda w, s: setattr(w, 'text_size', s))
        self.new_scene()
        if self.args.kind == 'model':
            self.setup_model(self.args.name)
        elif self.args.kind in ('menu', 'demo'):
            self.open_main_menu()
        else:
            self.setup_map(self.args.name)
        Window.bind(on_key_down=self.on_key_down, on_key_up=self.on_key_up, mouse_pos=self.on_mouse_pos)
        Clock.schedule_interval(self.on_frame, 0)
        if self.args.screenshot_file_path:
            Clock.schedule_once(self.take_screenshot, self.args.screenshot_delay)
        return self.root_layout

    def new_scene(self):
        """
        A fresh 3D view with the interface over it, the old ones are thrown away.
        """
        self.root_layout.clear_widgets()
        self.scene = scene.SceneWidget(self.game_data)
        self.interface = ui.Interface(self.game_data)
        self.interface.show_regions = self.args.show_regions
        self.root_layout.add_widget(self.scene)
        # engine.flyinghp: the lost health and the gained experience over the units
        self.flying_numbers = flyinghp_module.FlyingNumbers()
        self.flying_view = flyinghp_view.FlyingNumbersView(self.game_data)
        self.root_layout.add_widget(self.flying_view)
        self.root_layout.add_widget(self.interface)
        self.root_layout.add_widget(self.hud)
        self.root_layout.add_widget(self.help_label)
        self.scene.bind(on_touch_down=self.on_touch_down, on_touch_move=self.on_touch_move, on_touch_up=self.on_touch_up)
        Window.show_cursor = False
        self.unit = None
        self.companions = []
        self.selected_units = []
        self.map_units = []
        self.props = []
        self.props_by_nid = {}
        # engine.camera.obstacles of the static objects of the map
        self.camera_obstacles = None
        # the logic of the units of the computer (engine.ai)
        self.ai = ai.Controller()
        # (engine.missiles.Arrow, its prop, its view) of the arrows in flight
        self.arrows = []
        # (prop, view, target, offset, matrix, ms left) of the arrows in a body
        self.stuck_arrows = []
        # engine.magic.AreaEffect of the spells on the ground
        self.spell_effects = []
        # engine.particles.Emitter which live alone: the smoke of a missile which came, the flashes of the hits
        self.loose_emitters = []
        # magic.SpellLight of the missiles which hit
        self.loose_lights = []
        # id of a unit -> the spells on it the last time (the sounds of the enchantments, 5112a0)
        self.enchantments_seen = {}
        # (engine.particles.Emitter, engine.unit.Unit) of the spells which follow their units
        self.unit_emitters = []
        # the part of a tick of the particles which went by
        self.particle_ticks = 0.0
        # engine.magic.Spell of the hero for Ctrl + click
        self.spell = None
        self.terrain = None
        self.path_grid = None
        # client.mapscript.MapFigure of the static objects of the map, their parts make the grid of the paths
        self.figures = []
        # client.mapscript.LeverObject of the map, their parts go into the grid of the paths by their state
        self.levers = []
        # engine.trap.MagicTrap of the map
        self.traps = []
        # the LeverObject the hero goes to use
        self.lever_to_use = None
        self.map_name = None
        self.mob_name = None
        self.world_light = None
        self.main_menu = None
        self.esc_menu = None
        self.battle = None
        if self.weather is not None:
            self.weather.close()
            self.weather = None
        # engine.script.ScriptWorld of the map and its client.mapscript.ClientHost
        self.scripts = None
        self.script_host = mapscript.ClientHost(self)
        # (zones.Zone, the index of its exit) which LeaveToZone asked for
        self.next_zone = None
        # engine.ambient.Ambient of the map: the loops and the single sounds of the land around the camera
        self.ambient = None
        self.audio.set_ambient({})
        # the ticks of the world in this zone and the autosave of the entry, which only a new game and
        # an entry into a zone ask for, not a load (CBattleWindow::Activate 6353da)
        self.zone_ticks = 0
        self.zone_time_ms = 0.0
        self.autosave_pending = False
        # the tutorial window of the scripts, the game stands while it is open
        self.tutorial = tutorial_view.TutorialView(self)

    def open_main_menu(self):
        self.new_scene()
        self.title = 'Evil Islands remake - main menu'
        self.main_menu = menu.MainMenu(self)
        self.main_menu.setup()
        self.weather = weather_view.WeatherView(self)
        self.weather.open_menu()
        self.music.system.set_main_menu()

    def start_game(self):
        """
        New game: the first map of the campaign with the hero.
        """
        self.roster = None
        self.quest_log.reset()
        if self.demo is not None:
            self.demo.start()
        else:
            self.new_scene()
            self.setup_map(self.args.start_map)
        self.autosave_pending = True

    def save_game(self, name=None, quiet=False):
        """
        The state of the remake in saves/<name>: remake.json and the game-compatible info.sav. The name is
        the next free saveN when it is not given. quiet: no message (the autosaves of game.exe write none).
        """
        if self.unit is None or self.map_name is None:
            if not quiet:
                self.show_message('Nothing to save')
            return False
        name = name or saveload_module.next_name(self.game_data.data_dir_path)
        state = saveload_module.capture(self)
        saveload_module.save_slot(saveload_module.slot_dir(self.game_data.data_dir_path, name), state)
        if self.battle is not None and not quiet:
            self.battle.add_message('Saved %s' % name)
        return True

    def autosave(self):
        """
        The string event "autosave" (5f8070, of the world on its 10th tick or of a script): when the entry
        into the zone asked for it, the game goes to savea if the option is on (6360b0, the flag is
        dropped either way).
        """
        if not self.autosave_pending:
            return False
        self.autosave_pending = False
        if not self.settings.values.get('autosave'):
            return False
        return self.save_game(saveload_module.AUTO_ENTER_SLOT_NAME, quiet=True)

    def autosave_exit(self):
        """
        The map of the allod opens after the exit from a zone: the game goes to saveb if the option is on
        (62d366 and 62eb94; game.exe also skips it in a network game and in one more state of the world,
        [world + 0x104] == 5, which is not known yet).
        """
        if not self.settings.values.get('autosave'):
            return False
        return self.save_game(saveload_module.AUTO_EXIT_SLOT_NAME, quiet=True)

    def load_game(self, name):
        """
        The state of saves/<name>: the map is built again and the party, the hero, the units, the levers,
        the variables and the camera take their places from remake.json.
        """
        dir_path = saveload_module.slot_dir(self.game_data.data_dir_path, name)
        if saveload_module.game_save_is_newer(dir_path):
            game_save = saveload_module.read_game_save(dir_path, self.game_data)
            if game_save is not None:
                return self.load_game_save(name, game_save)
        state = saveload_module.load_slot(dir_path)
        if state is None or not state.get('map_name'):
            self.show_message('Cannot load %s' % name)
            return False
        self.new_scene()
        self.game_vars = dict(state.get('game_vars') or {})
        self.quest_log.reset()
        saveload_module.restore_party(self.party, state.get('party'), self.game_data)
        self.roster = saveload_module.restore_roster(self.roster or roster_module.Roster(), state.get('roster'),
                                                     self.game_data)
        self.take_lead()
        position = (state.get('hero', {}).get('x'), state.get('hero', {}).get('y'))
        if position[0] is None:
            position = None
        self.setup_map(state['map_name'], position=position, model_name=self.hero_model_name,
                       prototype_name=self.hero_prototype_name,
                       item_names=list(self.hero_item_names or ()), yaw=state.get('hero', {}).get('yaw', 0.0),
                       mob_name=state.get('mob_name'), hour=state.get('hour'))
        self.fast_hours = bool(state.get('fast_hours', False))
        saveload_module.restore_world(self, state)
        if self.battle is not None:
            self.battle.add_message('Loaded %s' % name)
            self.battle.update_static()
        return True

    def load_game_save(self, name, game_save):
        """
        A save of game.exe (engine.saveload.GameSave): the map of its zone with its hour, the party of its
        roster led by the hero in its place; the other objects come from the .mob of the map, the variables,
        the quests and the things of the other units are not read yet.
        """
        self.new_scene()
        self.game_vars = {}
        self.quest_log.reset()
        self.party.items = list(game_save.other_item_names)
        self.party.quest_items = []
        self.roster = game_save.roster
        self.take_lead()
        position, yaw = None, 0.0
        if game_save.hero_place is not None:
            (x, y, _), yaw = game_save.hero_place
            position = (x, y)
        self.setup_map(game_save.map_name, position=position, model_name=self.hero_model_name,
                       prototype_name=self.hero_prototype_name, item_names=list(self.hero_item_names or ()),
                       yaw=yaw, mob_name=game_save.mob_name, hour=game_save.hour)
        if self.battle is not None:
            self.battle.add_message('Loaded %s' % name)
            self.battle.update_static()
        return True

    def show_help(self, text, time_ms):
        """
        Lines of text over the screen for a while (the keys of the demo).
        """
        self.help_label.text = text
        self.help_time_ms = time_ms

    def show_message(self, text):
        if self.battle is not None:
            self.battle.add_message(text)
            return
        self.message = text
        self.message_time_ms = 3000.0

    def update_pause_filter(self):
        """
        The world and the HUD of game.exe go gray and dark while a modal window is up; the menu
        columns, the cursor and the windows themselves keep their colors (engine.menus.PAUSE_FILTER).
        """
        if self.scene is None:
            return
        paused = ((self.main_menu is not None and self.main_menu.has_modal) or
                  (self.esc_menu is not None and self.esc_menu.has_modal))
        self.scene.set_pause_filter(*(menus.PAUSE_FILTER if paused else menus.PAUSE_NONE))

    def ui_sound(self, name):
        """
        A sound of a button or of a window: "buttons/Camp/Put_on.wav" of sfx.res (5f7e60).
        """
        audio = getattr(self, 'audio', None)
        if audio is not None:
            audio.play_interface(name)

    def on_party_change(self, kind, value):
        """
        A script gave or took a thing of the party: "Item gained: Traps key".
        """
        if kind in ('quest_item', 'quest_item_lost'):
            name = (self.game_data.text('QUESTITEM %s' % value.name) or value.name).splitlines()[0]
        elif kind == 'quest':
            name = (self.game_data.text('quest %s' % value.name) or value.name).splitlines()[0]
        else:
            name = str(value)
        head = (self.game_data.text(PARTY_MESSAGES[kind]) or '').strip()
        self.show_message('%s %s' % (head, name.strip()) if head else name.strip())

    def hero_science(self):
        """
        (flags, keys, level) of the hero for engine.lever.Lever.use.
        """
        return (HERO_LEVER_FLAGS, self.party.keys(), HERO_LEVER_LEVEL)

    def make_equipment(self, model_name, prototype_name=None, item_names=()):
        """
        Equipment for the humanoids (the unit types of the class 0x32 in units.udb), None for other figures.
        """
        if model_name not in figure.HUMANOID_MODELS:
            return None
        armor_names, weapon_names = [], []
        for item_name in item_names:
            item = equipment_module.make_item(self.game_data.items(), item_name)
            if item is not None:
                (weapon_names if item.is_weapon else armor_names).append(item_name)
        return equipment_module.from_unit(self.game_data, model_name, prototype_name, armor_names, weapon_names)

    def make_unit(self, model_name, position=(0.0, 0.0), coefs=None, yaw=0.0, texture_name=None, weapon=None, gear=None,
                  unit_stats=None, prototype_name=None, in_party=False, removable=False, hero_stats=None):
        """
        unit_stats: engine.stats.UnitStats of the unit (of the map); without it the numbers of the prototype
        (prototype_name or the first one of the unit type). in_party: the unit is of the player.
        hero_stats: engine.experience.HeroStats of a hero or a mercenary.
        """
        coefs = coefs or BODY_PRESETS[self.body_index % len(BODY_PRESETS)]
        model = figure.FigureModel(self.game_data, model_name, coefs=coefs, equipment=gear)
        if weapon is None and gear is not None:
            weapon = gear.weapon_name
        u = unit_module.Unit(model, terrain=self.terrain, position=position, yaw=yaw, weapon=weapon)
        registry = self.game_data.registry(AI_REG)
        u.wounds = stats.WoundLevels.from_registry(registry)
        unit_type = self.unit_type(model_name)
        if unit_stats is None:
            prototype = self.prototype(model_name, prototype_name)
            if prototype is not None:
                unit_stats = stats.from_prototype(prototype, unit_type)
            elif unit_type is not None:
                unit_stats = stats.UnitStats(0, 0, 0, 0, 0, 0, 0, 0, 0, speeds=unit_type.speeds)
        u.set_stats(unit_stats)
        prototype = self.prototype(model_name, prototype_name)
        u.prototype_name = prototype.name if prototype is not None else None
        u.kill_experience = prototype.kill_experience if prototype is not None else 0.0
        if unit_stats is not None and unit_stats.max_health > 0:
            u.fighter = combat.Fighter(unit_stats, unit_type, in_party=in_party,
                                       difficulty=combat.Difficulty.from_registry(registry, self.difficulty_level()),
                                       radius=combat.unit_radius(u.half_size(), in_party))
            if prototype is not None and prototype.strike_delay > 0:
                u.fighter.strike_delay = prototype.strike_delay
            if gear is not None:
                u.fighter.equip(gear, hero_stats)
            if missiles.is_shooting(u.weapon_code):
                # a shooter strikes from the range of its weapon (the tag 23 of the shape)
                shape = gear.weapon.shape if gear is not None and gear.weapon is not None else None
                u.fighter.reach = shape.range if shape is not None and shape.range > 0 else missiles.DEFAULT_RANGE
            elif u.shoots:
                # a monster which shoots without a weapon (54b6c0) strikes from its range of the attack;
                # 54f450 makes a missile only for a bow or a crossbow item, its blow comes at once
                u.fighter.reach = u.attack_range
        u.path_class = unit_type.path_class if unit_type is not None else pathing.HUMAN_CLASS
        # the class of the humanoids: six layers of the wounds and the armor which hides them (engine.woundskin)
        u.humanoid = unit_type is not None and unit_type.unit_class == idb.UNIT_CLASS_HUMANOID
        if unit_type is not None:
            weapon_code = adb.WEAPONS.index(weapon) if weapon in adb.WEAPONS else None
            voice = sounds.Voice.from_unit_type(unit_type, self.prototype(model_name, prototype_name), weapon_code,
                                                sound_count=self.game_data.sound_count)
            u.sound_tracker = sounds.SoundTracker(voice, rng=u.rng)
            u.on_sounds = self.audio.on_unit_sounds
        u.on_shoot = self.add_arrow
        u.on_experience = self.on_experience
        u.neighbours = self.living_units
        u.on_spell_effect = self.spell_effects.append
        u.on_spell_sound = self.audio.play_sound
        u.on_spell_emitter = self.add_unit_emitter
        u.on_signal = self.on_unit_signal
        self.scene.add_unit(u, texture_name=texture_name, removable=removable)
        return u

    def prototype(self, model_name, prototype_name=None):
        """
        The prototype of units.udb with the name, otherwise the first one of the unit type with the model.
        """
        units = self.game_data.units()
        if prototype_name:
            return units.prototype(prototype_name)
        for prototype in units.prototypes:
            unit_type = units.unit_type(prototype.type_name)
            if unit_type is not None and unit_type.model_name == model_name:
                return prototype
        return None

    def difficulty_level(self):
        """
        DAT_0079beb0: the index of [DifficultyLevels] of ai.reg for the difficulty of the game (0 normal, 1 novice),
        the multipliers which CUnitServer__ApplyDifficulty (52e0d0) gives to the units out of the party.
        """
        return difficulty_module.GAME_INDEX.get(self.difficulty, 0)

    def unit_type(self, model_name):
        """
        The first unit type of units.udb with the model, None when there is none.
        """
        for unit_type in self.game_data.units().types:
            if unit_type.model_name == model_name:
                return unit_type
        return None

    def spawn_map_objects(self, map_name):
        """
        Trees, buildings and units of the .mob file of the map, the units stand and play their idles.
        """
        root = self.game_data.map_mob(map_name)
        if root is None:
            return
        self.figures = []
        self.ai = ai.Controller(ai.Diplomacy.from_mob(root), ai.Settings.from_registry(self.game_data.registry(AI_REG)))
        self.spawn_objects(root, map_name)

    def spawn_objects(self, root, mob_name):
        """
        The objects of a .mob: the units go to the logic, the static objects to the obstacles of the paths.
        Gives the new units and whether a static object was added.
        """
        skipped = 0
        new_units = []
        static_added = False
        for obj in world.map_objects(root):
            if obj.is_trap:
                self.traps.append(self.make_trap(obj))
                self.script_host.add_object(obj.nid, obj.name, self.traps[-1])
                continue
            if not self.game_data.has_model(obj.template):
                self.script_host.add_object(obj.nid, obj.name, obj)
                skipped += 1
                continue
            texture_name = obj.texture if obj.texture and obj.texture != 'default0' else None
            x, y, z = obj.position
            if obj.is_unit:
                gear = self.make_equipment(obj.template, obj.prototype, list(obj.armors) + list(obj.weapons))
                u = self.make_unit(obj.template, position=(x, y), coefs=obj.complection, yaw=obj.yaw,
                                   texture_name=texture_name, gear=gear, unit_stats=stats.from_mob(obj.stats),
                                   prototype_name=obj.prototype)
                u.quest_items = [q for q in (self.game_data.items().quest_item(n) for n in obj.quest_items) if q]
                self.map_units.append(u)
                new_units.append(u)
                self.script_host.add_object(obj.nid, obj.name, u)
                logics = [ai.UnitLogic.from_node(n) for n in obj.logic_nodes]
                self.ai.add(u, obj.player or 0, logics=[lg or ai.UnitLogic(use=False) for lg in logics] or None)
                continue
            model = figure.FigureModel(self.game_data, obj.template, coefs=obj.complection)
            matrix = gl_pose(x, y, self.terrain.height_at(x, y) + z, obj.rotation_matrix())
            texture_name = texture_name or model.default_texture_name()
            second_texture_name = obj.second_texture if obj.second_texture and obj.second_texture != 'default0' else None
            if obj.template in menus.COLUMN_MODELS:
                # game.exe 667e80 marks the menu columns, they play the actions of their .adb
                p = prop_module.Prop(model, matrix)
                self.scene.add_prop(p, texture_name, second_texture_name)
                self.props.append(p)
                self.props_by_nid[obj.nid] = p
                self.script_host.add_object(obj.nid, obj.name, p)
                continue
            if obj.is_lever:
                lever = mapscript.LeverObject(obj, lever_module.Lever.from_object(obj), model, matrix)
                lever.view = self.scene.add_lever(model, lever.lever, matrix, texture_name, second_texture_name)
                self.levers.append(lever)
                self.script_host.add_object(obj.nid, obj.name, lever)
                static_added = True
                continue
            map_figure = mapscript.MapFigure(obj, model, matrix)
            map_figure.view = self.scene.add_static_object(model, matrix, texture_name, second_texture_name)
            map_figure.parts()
            self.figures.append(map_figure)
            self.script_host.add_object(obj.nid, obj.name, map_figure)
            static_added = True
        if _Debug:
            print('%d objects of %s were skipped' % (skipped, mob_name))
        if static_added:
            self.camera_obstacles = camera_module.obstacles(f.parts() for f in self.figures)
        self.set_camera_floor()
        return new_units, static_added

    def listener_point(self):
        """
        40fb30: the sounds are heard from the target of the camera, 2 over the ground under it.
        """
        x, y = float(self.scene.camera.target[0]), float(self.scene.camera.target[1])
        ground = self.terrain.height_at(x, y) if self.terrain is not None else 0.0
        return x, y, ground + camera_module.LISTENER_HEIGHT

    def set_camera_floor(self):
        """
        40e4c0: the camera of the player stays above the land and the objects of the map.
        """
        camera = self.scene.camera
        if isinstance(camera, camera_module.Camera):
            camera.terrain = self.terrain
            camera.obstacles = self.camera_obstacles
            camera.floor_point = None

    def make_trap(self, obj):
        """
        A magic trap: it casts like a unit, the marker of the editor is not drawn.
        """
        t = trap_module.from_object(obj, self.game_data.spells(), land_height=self.terrain.height_at)
        t.on_shoot = self.add_arrow
        t.on_spell_effect = self.spell_effects.append
        t.player_of = lambda u: self.ai.player_of(u)
        return t

    def add_mob(self, mob_name):
        """
        AddMob of the scripts: the objects of another .mob of the folder maps join the map.
        Gives the root of the .mob, None when there is no such file.
        """
        name = mob_name.lower()
        if name.endswith('.mob'):
            name = name[:-4]
        root = self.game_data.map_mob(name)
        if root is None or self.terrain is None:
            return None
        new_units, static_added = self.spawn_objects(root, name)
        if static_added:
            self.build_path_grid()
        for u in new_units:
            u.path_grid = self.path_grid
        return root

    def setup_model(self, model_name):
        gear = self.make_equipment(model_name, self.args.prototype, self.args.item_names)
        self.unit = self.make_unit(model_name, texture_name=self.args.texture, weapon=self.args.weapon, gear=gear)
        self.scene.camera.target[:] = (0.0, 0.0, 1.0 * self.unit.figure.scale)
        self.scene.camera.distance = 4.5
        self.scene.camera.pitch = 15.0
        if self.args.action:
            action = self.unit.figure.database.by_name(self.args.action)
            if action:
                self.action_index = self.unit.figure.database.actions.index(action)
                self.unit.do_action(action)

    def setup_world(self, map_name, hour=None, with_unit=True, mob_name=None):
        """
        The land, the water, the light and the objects of a map; mob_name is the .mob of the objects when it is
        not the one of the map (the zone bz1g of map.txt loads basegipat.mpr and basecam.mob).
        """
        mob_name = mob_name or map_name
        self.terrain = terrain_module.Terrain(self.game_data, map_name)
        self.map_name = map_name
        self.scene.set_terrain(self.terrain)
        settings = world.world_settings(self.game_data.map_mob(mob_name))
        if hour is None:
            hour = settings.hour if self.args.hour is None else self.args.hour
        self.hour = hour
        self.world_light = lighting.for_map(self.game_data, map_name)
        zone = self.game_data.zone_of_map(map_name)
        self.ambient = ambient_module.Ambient(
            self.terrain, ambient_module.sound_set(zone.allod if zone else None, zone is not None and zone.is_cave),
            lambda name: self.game_data.sound_name(name) is not None)
        self.scene.set_light(self.world_light.at(self.hour))
        self.scene.set_water(self.terrain, water.Wind(settings.wind_direction, settings.wind_strength))
        if self.args.objects:
            self.spawn_map_objects(mob_name)

    def setup_map(self, map_name, position=None, model_name=None, prototype_name=None, item_names=None, yaw=0.0,
                  camera_yaw=None, mob_name=None, hour=None):
        """
        The map with the hero: by the options of the command line, the arguments take their place.
        yaw is the angle of the hero in radians, camera_yaw the angle of the camera in degrees; hour is
        the hour of a loaded save instead of WS_TIME.
        """
        mob_name = mob_name or map_name
        self.mob_name = mob_name
        self.title = 'Evil Islands remake - map %s' % map_name
        self.setup_world(map_name, hour=hour, mob_name=mob_name)
        self.esc_menu = menu.EscMenu(self)
        x, y = position or self.args.position or (self.terrain.width / 2.0, self.terrain.height / 2.0)
        self.hero_model_name = model_name or self.args.unit
        self.hero_prototype_name = prototype_name or self.args.prototype
        self.hero_item_names = list(self.args.item_names if item_names is None else item_names)
        if self.roster is None:
            self.roster = roster_module.new_roster(self.hero_prototype_name, self.hero_item_names, self.hero_model_name)
            self.hero_record = self.roster.default.lead()
            self.hero_stats_name = None
        self.unit = self.make_hero((x, y), self.hero_item_names, yaw=yaw)
        self.spawn_companions()
        self.select_party(0)
        self.spell = self.make_hero_spell(self.args.spell)
        self.build_path_grid()
        self.scene.camera.target[:] = (x, y, self.unit.z + camera_module.TARGET_HEIGHT)
        if camera_yaw is not None:
            self.scene.camera.yaw = camera_yaw
        self.reset_camera()
        self.battle = battle.BattleScreen(self)
        self.battle.trade.units = self.party_fighters
        self.battle.setup()
        self.interface.set_regions(self.battle.regions())
        self.start_scripts(mob_name)
        self.recalc_mercenaries()
        self.hide_party_merc_objects()
        self.start_zone_music(map_name)
        self.weather = weather_view.WeatherView(self)
        self.weather.open_map(map_name)

    def start_zone_music(self, map_name):
        """
        SetZoneMode (567ab0): the music of the allod of the zone of map.txt whose .mpr is the map and of its sky.
        """
        found = [z for z in self.game_data.zones() if z.mpr_name == map_name.lower()]
        if not found:
            self.music.system.set_zone(zones_module.DEFAULT_ALLOD, False)
            return
        zone = found[0]
        self.music.system.set_zone(music_module.zone_allod(zone.name, zone.allod or zones_module.DEFAULT_ALLOD),
                                   zone.sky == zones_module.SKY_CAVE)

    def enter_zone_number(self, zone, exit_number):
        """
        The party enters the zone by the exit of the number of map.txt (1, 2...), not the index of the list.
        """
        index = next((i for i, zone_exit in enumerate(zone.exits) if zone_exit.number == exit_number), 0)
        self.enter_zone(zone, index)

    def advance_hours(self, hours):
        """
        The "#passtime" of a way: the hours of the world go by.
        """
        if hours:
            self.set_hour(self.hour + float(hours))

    def open_global_map(self, edge_zone, pass_time=0.0):
        """
        A road zone without a .mpr: the map of the allod with the roads out of it.
        """
        if self.battle is None or edge_zone is None:
            return False
        return self.battle.open_global_map(edge_zone, pass_time)

    def travel_through(self, option, pass_time=0.0):
        """
        A road of the global map: the party comes into the zone of the road by its exit, or another road is
        shown when the zone has no .mpr; the hours of the way go by.
        """
        zone = getattr(option, 'zone', None)
        if zone is None:
            return False
        if self.battle is not None and getattr(self.battle, 'globalmap', None) is not None:
            # the window closed itself before the travel; the game of the zone must not stay paused
            self.paused = self.battle.globalmap_paused
        self.advance_hours(pass_time)
        exit_number = option.exit.neighbour_exit if option.exit is not None else 0
        if zone.mpr_name:
            self.enter_zone_number(zone, exit_number)
        else:
            self.open_global_map(zone, 0.0)
        return True

    def enter_zone(self, zone, exit_index):
        """
        The party comes into the zone of map.txt by its exit (an index of the list, 590700): the hero stands
        in the middle of the deploy rect and looks by #deployangle or to the middle of the map.
        """
        zone_exit = zone.exits[exit_index] if 0 <= exit_index < len(zone.exits) else None
        width, height = (n * mapdata.SECTOR_CELLS for n in zone.sectors) if zone.sectors else (0, 0)
        position = zone_exit.deploy_center if zone_exit is not None and zone_exit.deploy is not None else None
        yaw = zone_exit.deploy_yaw(width, height) if position is not None and width else 0.0
        camera_yaw = zone_exit.camera_yaw if zone_exit is not None else None
        self.new_scene()
        self.take_lead()
        self.setup_map(zone.mpr_name, position=position, model_name=self.hero_model_name,
                       prototype_name=self.hero_prototype_name, item_names=self.hero_item_names, yaw=yaw,
                       camera_yaw=camera_yaw, mob_name=zone.mob_name, hour=self.hour)
        self.autosave_pending = True

    def leave_to_zone(self, zone_name, exit_index):
        """
        LeaveToZone of the scripts (66f2d0): the world only takes the zone and the exit, the zone is loaded after
        the tick of the scripts.
        """
        zones = [z for z in self.game_data.zones() if z.name.lower() == zone_name.lower()]
        if not zones or not zones[0].mpr_name:
            self.show_message('no zone %s' % zone_name)
            return
        self.next_zone = (zones[0], exit_index)

    def build_path_grid(self):
        """
        The grid of the paths with the objects of the map and the levers in their states; every unit gets it.
        """
        parts = []
        for map_figure in self.figures + self.levers:
            parts.extend(map_figure.parts())
        self.path_grid = pathing.PassGrid(self.terrain, pathing.tile_descriptions(self.game_data.registry(TILE_DESC_REG)),
                                          parts)
        for u in self.zone_party() + self.map_units:
            u.path_grid = self.path_grid

    def push_units_out(self):
        """
        5ba9d0: the units which stand in the closed points of the new grid go to the nearest open ones
        and forget their orders (554450).
        """
        for u in ([self.unit] if self.unit else []) + self.map_units:
            if unit_module.push_out(u):
                if u.brain is not None:
                    u.brain.order = ai.ORDER_NONE

    def start_scripts(self, map_name):
        """
        The scripts of the .mob: WorldScript starts at once, the rest goes with the ticks of the world.
        """
        self.scripts = None
        if not self.args.objects:
            return
        try:
            program = mapscript.program_of(self.game_data.map_mob(map_name))
        except script_lang.ScriptError as error:
            self.show_message('the scripts of %s: %s' % (map_name, error))
            return
        if program is None:
            return
        self.scripts = script.ScriptWorld(program, self.script_host, self.game_vars, controller=self.ai)
        self.scripts.start()

    def take_lead(self):
        """
        The hero of the map is the first unit of the current party of the roster (SetCurrentParty): its model,
        prototype, numbers and items.
        """
        record = self.roster.current().lead() if self.roster is not None else None
        if record is None:
            return
        self.hero_record = record
        self.hero_model_name = record.model_name or self.prototype_model_name(record.prototype_name) or self.args.unit
        self.hero_prototype_name = record.prototype_name
        self.hero_stats_name = record.stats_name if record.stats_name != record.prototype_name else None
        self.hero_item_names = list(record.item_names or ())

    def prototype_model_name(self, prototype_name):
        """
        The model of the unit type of a prototype of units.udb, None when it is not known.
        """
        units = self.game_data.units()
        prototype = units.prototype(prototype_name) if prototype_name else None
        unit_type = units.unit_type(prototype.type_name) if prototype is not None else None
        return unit_type.model_name if unit_type is not None else None

    def redeploy_party(self):
        """
        RedeployParty (0xd3): the current party comes into the zone again, the first unit of it in the place
        of the hero.
        """
        old = self.unit
        if old is None or self.roster is None:
            return False
        self.clear_companions()
        self.take_lead()
        self.swap_hero(self.make_hero((old.x, old.y), self.hero_item_names, yaw=old.yaw))
        self.spawn_companions((old.x, old.y))
        self.select_party(0)
        return True

    def clear_companions(self):
        """
        The units of the other members of the party leave the zone with their views.
        """
        for unit in self.companions:
            for view in [v for v in self.scene.unit_views if v.unit is unit]:
                self.scene.remove_prop(view)
            self.ai.players.pop(unit, None)
        self.companions = []

    def record_hero_stats(self, record, prototype_name):
        """
        engine.experience.HeroStats of a record of the roster, made of the hero template of its prototype (5136d0)
        when it first comes into a zone; None without a template. prototype_name is taken when the record
        has none (the hero of the command line is found by its model).
        """
        if record is None:
            return None
        if record.hero_stats is None:
            settings = experience.RpgSettings.from_registry(self.game_data.registry(AI_REG))
            record.hero_stats = experience.from_prototype(self.game_data.units(), record.stats_name or prototype_name,
                                                          self.game_data.perks(), settings)
        return record.hero_stats

    def make_hero(self, position, item_names, yaw=0.0):
        return self.make_record_unit(self.hero_record, position, yaw,
                                     model_name=self.hero_model_name,
                                     prototype_name=self.hero_prototype_name,
                                     stats_name=self.hero_stats_name, item_names=item_names,
                                     texture_name=self.args.texture, weapon=self.args.weapon)

    def make_record_unit(self, record, position, yaw=0.0, model_name=None, prototype_name=None,
                         stats_name=None, item_names=None, texture_name=None, weapon=None):
        """
        The unit of a record of the roster: its model, its prototype, the numbers of its stats_name, its
        items and its experience. The hero and the other members of the party come into the zone this way.
        """
        if model_name is None:
            model_name = (record.model_name if record is not None else None) or self.prototype_model_name(
                record.prototype_name if record is not None else None) or self.args.unit
        prototype_name = prototype_name if prototype_name is not None else getattr(record, 'prototype_name', None)
        stats_name = stats_name if stats_name is not None else getattr(record, 'stats_name', None)
        if item_names is None:
            item_names = list(getattr(record, 'item_names', None) or ())
        gear = self.make_equipment(model_name, prototype_name, item_names)
        if gear is not None and record is not None:
            record.worn_items = gear.keep_wear(record.worn_items)
        unit_type = self.unit_type(model_name)
        prototype = self.prototype(model_name, stats_name or prototype_name)
        unit_stats = stats.from_prototype(prototype, unit_type) if prototype is not None else None
        hero_stats = self.record_hero_stats(record, prototype.name if prototype is not None else None)
        if unit_stats is not None and hero_stats is not None:
            unit_stats.health = unit_stats.max_health = hero_stats.max_health
            unit_stats.mana = unit_stats.max_mana = hero_stats.max_mana
        unit = self.make_unit(model_name, position=position, yaw=yaw, texture_name=texture_name,
                              weapon=weapon, gear=gear, unit_stats=unit_stats,
                              prototype_name=prototype_name, in_party=True, removable=True,
                              hero_stats=hero_stats)
        unit.hero_stats = hero_stats
        unit.record = record
        if record is not None:
            unit.name = record.name
        unit.on_blow = self.on_blow
        unit.on_kill = self.on_kill
        unit.on_item_wear = self.on_item_wear
        unit.on_signal = self.on_unit_signal
        unit.on_cant_run = self.on_cant_run
        self.ai.add_player_unit(unit)
        return unit

    def spawn_companions(self, position=None):
        """
        The other records of the current party come into the zone behind the hero (RedeployParty).
        """
        self.companions = []
        if self.unit is None or self.roster is None:
            return self.companions
        records = list(self.roster.current().units)
        try:
            start = records.index(self.hero_record) + 1
        except ValueError:
            start = 1
        position = position or (self.unit.x, self.unit.y)
        for index, record in enumerate(records[start:], 1):
            x, y = companions_module.follow_slot(self.unit, index)
            unit = self.make_record_unit(record, (x, y), yaw=self.unit.yaw)
            self.companions.append(unit)
        return self.companions

    def recalc_mercenaries(self):
        """
        RecalcMercBriefings (591800): the topics of the mercenaries from the party and the base of the map.
        """
        if self.roster is None:
            return
        zone = self.game_data.zone_of_map(self.map_name) if self.map_name else None
        if zone is None and self.map_name:
            zone = next((z for z in self.game_data.zones() if z.mob_name.lower() == self.map_name.lower()), None)
        records = self.roster.current().units
        mercs_count = max(len(self.zone_party()) - 1, 0)
        mercenaries_module.recalc(self.game_vars, [record.name for record in records], mercs_count,
                                  zone.name if zone is not None else '')

    def mercenary_object(self, name):
        """
        The object "merc<N>" of the map by its name without the case, None when the map has none.
        """
        if self.script_host is None:
            return None
        lower = name.lower()
        for object_name, obj in self.script_host.objects_by_name.items():
            if object_name.lower() == lower:
                return obj
        return None

    def mercenary_position(self, obj):
        if obj is None:
            return None
        if hasattr(obj, 'x'):
            return (obj.x, obj.y, getattr(obj, 'z', 0.0))
        position = getattr(obj, 'position', None)
        return tuple(position) if position is not None else None

    def hide_mercenary_object(self, obj):
        """
        The object of a hired mercenary leaves the map like RemoveUnitFromServer (5935c0).
        """
        if obj is None or not hasattr(obj, 'hidden'):
            return False
        obj.hidden = True
        if obj in self.map_units:
            self.map_units.remove(obj)
        brain = getattr(obj, 'brain', None)
        if brain is not None and brain in self.ai.brains:
            self.ai.brains.remove(brain)
        return True

    def show_mercenary_object(self, name, position=None):
        """
        A mercenary left in its base stands on the map again.
        """
        obj = self.mercenary_object(name)
        if obj is None or not hasattr(obj, 'hidden'):
            return False
        obj.hidden = False
        if position is not None and hasattr(obj, 'x'):
            obj.x, obj.y = float(position[0]), float(position[1])
            obj.z = obj.ground_height() if self.terrain is not None else getattr(obj, 'z', 0.0)
        if hasattr(obj, 'fighter') and obj not in self.map_units:
            self.map_units.append(obj)
        brain = getattr(obj, 'brain', None)
        if brain is not None and brain not in self.ai.brains:
            self.ai.brains.append(brain)
        return True

    def hide_party_merc_objects(self):
        """
        The objects of the mercenaries of the party do not stand in the zone beside their party units.
        """
        for unit in self.companions:
            self.hide_mercenary_object(self.mercenary_object(getattr(unit, 'name', '')))

    def party_path(self, name):
        party = self.roster.current() if self.roster is not None else None
        if party is None or party is self.roster.default:
            return name
        return party.name + roster_module.SEPARATOR + name

    def hire_mercenary(self, index):
        """
        591de0 with the topic n1 or n3: the record of the mercenary joins the party, its unit comes to the
        hero; the object of the map goes away until the mercenary leaves the party again.
        """
        if self.roster is None:
            return None
        name = mercenaries_module.name_of(index)
        definitions = mercenaries_module.mercenary(index)
        if definitions is None:
            return None
        base, fallback_prototype = definitions
        party = self.roster.current()
        record = party.unit(name) if party is not None else None
        obj = self.mercenary_object(name)
        if record is None:
            prototype_name = getattr(obj, 'prototype_name', None) or fallback_prototype
            model_name = obj.figure.model_name if obj is not None and hasattr(obj, 'figure') else None
            record = self.roster.add_unit(self.party_path(name), prototype_name, model_name=model_name)
            if record is None:
                return None
            if obj is not None and hasattr(obj, 'fighter'):
                names = [item.name for item in trade_module.worn_items(obj)]
                record.item_names = names or None
        position = self.mercenary_position(obj)
        if position is None:
            hero = self.unit
            position = companions_module.follow_slot(hero, len(self.companions) + 1) if hero is not None else (0.0, 0.0)
        unit = self.make_record_unit(record, (position[0], position[1]),
                                     yaw=getattr(obj, 'yaw', 0.0) if obj is not None else 0.0)
        unit.path_grid = self.path_grid
        self.companions.append(unit)
        self.hide_mercenary_object(obj)
        self.show_message('%s joins the party' % name)
        return unit

    def dismiss_mercenary(self, index, remove_object=False):
        """
        591de0 with the topic n2 (the mercenary stays in its base) or n10 (it leaves the map too).
        """
        if self.roster is None:
            return False
        name = mercenaries_module.name_of(index)
        unit = next((c for c in self.companions if getattr(c, 'name', '').lower() == name.lower()), None)
        position = (unit.x, unit.y, unit.z) if unit is not None else None
        if unit is not None:
            for view in [v for v in self.scene.unit_views if v.unit is unit]:
                self.scene.remove_prop(view)
            self.ai.players.pop(unit, None)
            self.companions.remove(unit)
            self.selected_units = [chosen for chosen in self.selected_units if chosen is not unit]
        self.roster.remove_unit(self.party_path(name))
        if not remove_object:
            self.show_mercenary_object(name, position)
        self.show_message('%s leaves the party' % name)
        return True

    def remove_mercenary_record(self, unit):
        """
        A mercenary of the party died: its record leaves the party, its body stays in the zone.
        """
        if self.roster is not None and getattr(unit, 'name', None):
            self.roster.remove_unit(self.party_path(unit.name))
        unit.party_removed = True
        self.show_message('%s has died' % getattr(unit, 'name', 'A mercenary'))

    def complete_merc_briefing(self, variable):
        """
        591de0: the end of the briefing of a mercenary hires it, leaves it in the base or takes it away.
        """
        result = mercenaries_module.complete(self.game_vars, variable)
        if result is None:
            return
        index, action = result
        if action == 'hire':
            self.hire_mercenary(index)
        elif action == 'leave':
            self.dismiss_mercenary(index, remove_object=False)
        elif action == 'leave_zone':
            self.dismiss_mercenary(index, remove_object=True)

    def zone_party(self):
        """
        The living units of the party in the zone, the hero first.
        """
        units = []
        if self.unit is not None:
            units.append(self.unit)
        units.extend(self.companions)
        return [unit for unit in units if not unit.is_dead]

    def select_party(self, index=None):
        """
        F1..F3 choose a member, F4 all of them; None selects every living member of the party.
        """
        party = self.zone_party()
        if index is None:
            self.selected_units = list(party)
            return self.selected_units
        if not 0 <= index < len(party):
            return []
        self.selected_units = [party[index]]
        return self.selected_units

    def replace_hero(self, item_names):
        """
        The hero with other items at the same place (the demo changes the weapon this way): the health and the
        mana go over, the monsters which fought the old one fight the new one.
        """
        old = self.unit
        if old is None:
            return
        if self.hero_record is not None:
            self.hero_record.item_names = list(item_names)
        hero = self.make_hero((old.x, old.y), item_names, yaw=old.yaw)
        if old.fighter is not None and hero.fighter is not None:
            hero.fighter.max_mana = max(hero.fighter.max_mana, old.fighter.max_mana)
            hero.fighter.mana = old.fighter.mana
            for new_part, old_part in zip(hero.fighter.parts, old.fighter.parts):
                new_part.health = min(old_part.health, new_part.max_health)
                new_part.state = old_part.state
        self.swap_hero(hero)

    def replace_companion(self, unit, record=None):
        """
        The unit of a member of the party is built again with its worn things (the camp), its health and its
        mana stay; the monsters which fought it fight the new one.
        """
        if unit not in self.companions:
            return unit
        index = self.companions.index(unit)
        record = record or getattr(unit, 'record', None)
        if record is None:
            return unit
        record.item_names = list(record.item_names or ())
        new = self.make_record_unit(record, (unit.x, unit.y), yaw=unit.yaw)
        if unit.fighter is not None and new.fighter is not None:
            new.fighter.max_mana = max(new.fighter.max_mana, unit.fighter.max_mana)
            new.fighter.mana = unit.fighter.mana
            for new_part, old_part in zip(new.fighter.parts, unit.fighter.parts):
                new_part.health = min(old_part.health, new_part.max_health)
                new_part.state = old_part.state
        new.path_grid = unit.path_grid
        for view in [v for v in self.scene.unit_views if v.unit is unit]:
            self.scene.remove_prop(view)
        self.ai.players.pop(unit, None)
        self.companions[index] = new
        self.selected_units = [new if chosen is unit else chosen for chosen in self.selected_units]
        for brain in self.ai.brains:
            if brain.target is unit:
                brain.target = new
        for other in self.map_units:
            if other.foe is unit:
                other.foe = new
        return new

    def swap_hero(self, hero):
        """
        The new hero takes the place of the old one: its view goes away, the monsters which fought the old one
        fight the new one.
        """
        old = self.unit
        for view in [v for v in self.scene.unit_views if v.unit is old]:
            self.scene.remove_prop(view)
        self.ai.players.pop(old, None)
        hero.path_grid = old.path_grid
        for brain in self.ai.brains:
            if brain.target is old:
                brain.target = hero
        for u in self.map_units:
            if u.foe is old:
                u.foe = hero
        self.unit = hero

    def make_hero_spell(self, text):
        """
        "Fire Arrow+r1+e1": a prototype of spells.sdb by its name or code with the codes of the modifiers.
        """
        if not text:
            return None
        words = text.split('+')
        spell = magic.make_spell(self.game_data.spells(), words[0].strip(), [w.strip() for w in words[1:]])
        if spell is None:
            if _Debug:
                print('unknown spell %r' % text)
        elif not spell.sends_missile() and spell.index not in magic.INSTANT_SPELLS:
            if _Debug:
                print('the flight of the spell %r is not known yet, it hits at once' % spell.name)
        return spell

    def reset_camera(self):
        """
        The default camera of 40fb30: the distance and the pitch of [Camera Settings], the pitch and the
        zoom are clamped by engine.camera.
        """
        camera = self.scene.camera
        distance = camera_module.DEFAULT_DISTANCE if self.args.distance is None else self.args.distance
        camera.zoom_to(distance)
        pitch = camera_module.DEFAULT_PITCH if self.args.pitch is None else self.args.pitch
        camera.pitch = camera_module.clamp_pitch(pitch)
        self.set_camera_floor()

    def rebuild_model(self):
        self.scene.canvas.clear()
        self.scene.unit_views = []
        self.scene.static_views = []
        self.setup_model(self.args.name)

    def update_hover(self):
        """
        The unit under the mouse and the cursor for the last position of the mouse of this frame.
        """
        pending = self.mouse_pending
        self.mouse_pending = None
        if pending is None or self.battle is None:
            return
        pos, x, y = pending
        self.hover_unit, self.hover_part = picking_view.hover_at(self, (x, y))
        self.cursor.update(pos, self.battle.on_mouse(x, y))

    def on_frame(self, dt):
        self.frame_index += 1
        self.update_hover()
        now = time.perf_counter()
        dt_ms = 0.0 if self.last_time is None else (now - self.last_time) * 1000.0
        self.last_time = now
        if self.paused or self.tutorial_open() or (self.movie is not None and self.movie.is_open):
            dt_ms = 0.0
        if self.message_time_ms > 0.0:
            self.message_time_ms -= dt * 1000.0
        if self.help_time_ms > 0.0:
            self.help_time_ms -= dt * 1000.0
            if self.help_time_ms <= 0.0:
                self.help_label.text = ''
        self.interface.update(dt * 1000.0)
        if self.main_menu is not None:
            self.main_menu.update(dt * 1000.0)
        self.music.update(now * 1000.0, self.ai.in_danger(ai.PARTY_PLAYER))
        if self.battle is not None:
            self.battle.update(dt * 1000.0)
        if self.esc_menu is not None:
            self.esc_menu.update(dt * 1000.0, self.clock)
            if self.esc_menu.is_open:
                # the world stands while the escape menu is open
                dt_ms = 0.0
        self.update_pause_filter()
        ticks = self.clock.advance(dt_ms)
        if self.map_name is not None:
            # 5641f0 (partly) counts its calls before it looks at the pause, so the open tutorial does not stop the
            # count: the 10th tick of the world in the zone sends "autosave"
            ticks_before = self.zone_ticks
            self.zone_time_ms += dt * 1000.0
            self.zone_ticks = int(self.zone_time_ms / self.clock.tick_ms)
            if saveload_module.autosave_due(ticks_before, self.zone_ticks):
                self.autosave()
        self.scene.wave_clock.update(ticks)
        if self.fast_hours and dt_ms:
            self.set_hour(self.hour + FAST_HOURS_PER_SECOND * dt_ms / 1000.0)
        self.move_camera(dt, ticks)
        self.update_night_senses()
        self.ai.effects = self.spell_effects
        self.ai.update(ticks)
        for u in self.map_units:
            u.unseen = not self.ai.is_visible(u)
        if self.unit is not None:
            companions_module.update_follow(self.unit, self.companions, self.swarm)
        if self.scripts is not None:
            levels_moved = False
            for _ in range(ticks):
                self.scripts.tick()
                if self.next_zone is not None:
                    break
                levels_moved = self.scripts.water_levels.tick() or levels_moved
            if levels_moved:
                self.scene.set_water_levels(self.scripts.water_levels.offsets())
        self.update_quest_log()
        if self.next_zone is not None and not (self.movie is not None and self.movie.is_open):
            zone, exit_index = self.next_zone
            self.next_zone = None
            self.enter_zone(zone, exit_index)
            return
        if self.traps:
            units = self.living_units()
            for _ in range(ticks):
                for t in self.traps:
                    t.update(units)
        for lever in self.levers:
            lever.lever.update(dt_ms / self.clock.tick_ms)
        self.update_lever_use()
        for u in self.map_units + self.companions:
            u.update(dt_ms, self.clock)
        for u in self.companions:
            if u.is_dead and not getattr(u, 'party_removed', False):
                self.remove_mercenary_record(u)
        for p in self.props:
            p.update(dt_ms, self.clock)
        if self.unit:
            self.unit.update(dt_ms, self.clock)
            if self.terrain is not None and self.args.follow:
                self.scene.camera.target[:] = (self.unit.x, self.unit.y, self.unit.z + camera_module.TARGET_HEIGHT)
        self.update_arrows(dt_ms)
        effects = list(self.spell_effects)
        self.spell_effects[:] = unit_module.update_spell_effects(self.spell_effects, self.living_units(),
                                                                 dt_ms / self.clock.tick_ms)
        self.update_particles(dt_ms)
        self.audio.update(([self.unit] if self.unit else []) + self.companions + self.map_units,
                          self.listener_point())
        self.update_spell_sounds(effects + [e for e in self.spell_effects if e not in effects])
        self.update_ambient(ticks)
        if ticks:
            for view in self.scene.unit_views:
                self.scene.update_wounds(view)
        self.scene.redraw()
        self.update_flying_numbers()
        self.update_hud()

    def update_flying_numbers(self):
        """
        52ef90 (partly) and 5e18e0: the numbers of the lost health fly over the units while the option
        show_flying_hp is on.
        """
        now_ms = time.perf_counter() * 1000.0
        units = ([self.unit] if self.unit is not None else []) + self.companions + self.map_units
        # 5117f0: only the units the player sees give their numbers
        self.flying_view.watch(units, self.flying_numbers, now_ms, shown=lambda u: not u.unseen)
        self.flying_view.draw(self.flying_numbers, self.scene, now_ms,
                              enabled=bool(self.settings.values.get('show_flying_hp', 1)))

    def on_experience(self, unit, share):
        """
        66a440: a unit of the party shows the experience it gains.
        """
        stats = getattr(unit, 'hero_stats', None)
        if stats is None:
            return
        value = flyinghp_module.shown_experience(share, stats.intelligence)
        self.flying_numbers.add(unit, value, flyinghp_module.EXP, time.perf_counter() * 1000.0)

    def update_quest_log(self):
        """
        636990 and 638240 (partly): the changed variables of the quests give their lines to the messages of the battle
        window, the window plays the sound of the quests once.
        """
        if self.battle is None:
            return
        for message in self.quest_log.update(self.game_vars, self.game_data.text):
            self.battle.add_message(message)
        sound_name = self.quest_log.take_sound()
        if sound_name:
            self.audio.play_interface(sound_name)

    def update_ambient(self, ticks):
        """
        6cb350 on every tick of the world: the loops of the land around the camera and its single sounds.
        """
        if self.ambient is None or not ticks:
            return
        x, y = self.scene.camera.target[0], self.scene.camera.target[1]
        loops = {}
        # a long frame (a load) gives many ticks at once; their single sounds would come together
        for _ in range(min(ticks, AMBIENT_MAX_TICKS)):
            loops, shots = self.ambient.tick(x, y, self.hour, self.ambient_random)
            for name, position in shots:
                self.audio.play_sound(magic.SpellSound(name, position, ambient_module.SHOT_NEAR,
                                                       ambient_module.SHOT_FAR))
        self.audio.set_ambient(loops)

    def set_hour(self, hour):
        self.hour = hour % lighting.HOURS
        if self.world_light is not None:
            self.scene.set_light(self.world_light.at(self.hour))

    def update_night_senses(self):
        """
        58ccd0 and 5536b0 (partly): the darkness of the hour and the rain or the snow take the sight of the units.
        """
        dark = unit_module.darkness(self.hour)
        weather = getattr(self, 'weather', None)
        schedule = getattr(weather, 'schedule', None) if weather is not None else None
        wet = schedule is not None and schedule.falling != weather_module.NONE
        for unit in self.zone_party() + list(self.map_units):
            unit.set_light(dark, wet)

    def move_camera(self, dt, ticks=0):
        """
        The arrows scroll the camera on a map (40f230, 40fb30); in the model viewer they choose the actions.
        The main loop of game.exe (456440, partly) calls 40fb30 for every tick of the world and once more for
        the real time of the frame, so while the world goes the camera lives twice as fast.
        """
        camera = self.scene.camera
        if self.terrain is not None and isinstance(camera, camera_module.Camera):
            right = int('right' in self.keys_down) - int('left' in self.keys_down)
            forward = int('up' in self.keys_down) - int('down' in self.keys_down)
            values = getattr(getattr(self, 'settings', None), 'values', {}) or {}
            camera.scroll_keys(right, forward, options_module.keyboard_power(values))
            camera_ticks = (dt * 1000.0 + ticks * self.clock.tick_ms) / timing.TICK_MS
            camera.advance_scroll(camera_ticks)
        if self.ctrl_down() and 'q' in self.keys_down:
            camera.rotate_by(-90.0 * dt)
        if self.ctrl_down() and 'e' in self.keys_down:
            camera.rotate_by(90.0 * dt)

    def update_hud(self):
        u = self.unit
        lines = []
        if u is not None:
            clip = u.player.clip
            action = clip.action if clip else None
            lines.append('%s  scale=%.3f  body=%s' % (u.figure.model_name, u.figure.scale, u.figure.coefs))
            if action is not None:
                lines.append('%s  %s %s:%d %s  weapons=%s' % (
                    action.name, action.state_name, action.type_name, action.modifier, action.stage_name,
                    ','.join(action.weapon_names) or '-'))
                lines.append('frame %.2f / %d  hit_frame=%d  movement_speed=%.3f' % (
                    u.player.time, clip.length, action.hit_frame, action.movement_speed))
            lines.append('mode=%s  tick=%d ms (%.2f fps)%s' % (
                u.mode, self.clock.tick_ms, 1000.0 / self.clock.tick_ms, '  PAUSED' if self.paused else ''))
        if self.world_light is not None:
            lines.append('%02d:%02d%s  wind %.2f' % (int(self.hour), int(self.hour * 60) % 60,
                                                     '  (fast)' if self.fast_hours else '', self.scene.wind.strength))
        if not self.show_debug_text:
            lines = []
        if self.message_time_ms > 0.0:
            lines.append(self.message)
        self.hud.text = '\n'.join(lines)

    def select_action(self, step):
        database = self.unit.figure.database
        if not database or not database.actions:
            return
        self.action_index = (self.action_index + step) % len(database.actions)
        self.unit.do_action(database.actions[self.action_index])
        if database.actions[self.action_index].is_cycle:
            self.unit.mode = unit_module.MODE_IDLE

    def play_movie(self, name, on_end=None):
        """
        PlayMovie of the scripts: False when the file is missing or Kivy has no video provider;
        on_end is called once when the movie ends or a key stops it.
        """
        if self.movie is None:
            self.movie = movie_view.MoviePlayer(self.root_layout)
        self.movie.volume = self.settings.values['volume_stream'] / 100.0 if self.args.sound else 0.0
        self.movie.windowed = bool(self.settings.values.get('movie_window', 0))
        return self.movie.play(movie_view.movie_file_path(self.game_data.data_dir_path, name), on_end)

    def show_briefing(self, name):
        """
        The event "briefing <name>" of the scripts: the modal window of client.briefing with the music
        Briefing. When it closes, the variable "b.<owner>.<name>" of the briefing is completed (592090).
        """
        if self.battle is None:
            return False
        if not self.battle.show_briefing(name):
            return False
        variable = self.scripts.briefing_variable(name) if self.scripts is not None else None
        if variable is not None:
            view = self.battle.briefing
            closed = view.on_close

            def _closed():
                if closed is not None:
                    closed()
                if self.scripts is not None:
                    self.scripts.complete_briefing(0, variable)

            view.on_close = _closed
        return True

    def save_shot(self):
        """
        5f78c0 (partly): the picture of the screen in the folder of the remake's settings, the counter of the run goes up.
        """
        dir_path = options_module.DEFAULT_DIR_PATH
        os.makedirs(dir_path, exist_ok=True)
        shot_file_path = os.path.join(dir_path, SHOT_FILE_NAME % self.shot_index)
        self.shot_index += 1
        png_file_path = shot_file_path[:-len('.bmp')] + '.png'
        saved_path = Window.screenshot(name=png_file_path)
        if not saved_path:
            return None
        Image.open(saved_path).convert('RGB').save(shot_file_path, 'BMP')
        os.remove(saved_path)
        if _Debug:
            print('the screen is saved in %s' % shot_file_path)
        return shot_file_path

    def tutorial_enabled(self):
        """
        The tutorial windows pause the world, so the remake shows them only with --tutorial.
        """
        return bool(self.args.tutorial) and bool(self.settings.values.get('show_tutorial', 1))

    def tutorial_open(self):
        tutorial = getattr(self, 'tutorial', None)
        return tutorial is not None and tutorial.is_open

    def on_key_down(self, window, key, scancode, codepoint, modifiers):
        name = window._system_keyboard.keycode_to_string(key)
        self.keys_down.add(name)
        if name in SHOT_KEYS:
            self.save_shot()
            return True
        if self.movie is not None and self.movie.on_key(name):
            return True
        if self.main_menu is not None and self.main_menu.on_key(name):
            return True
        if self.battle is not None and self.battle.on_key(name):
            return True
        if self.esc_menu is not None and self.esc_menu.on_key(name):
            return True
        if self.battle is not None and self.tutorial.on_key(name):
            return True
        if self.demo is not None and (self.esc_menu is None or not self.esc_menu.is_open) and self.demo.on_key(name):
            return True
        if self.keys is not None and self.keys.on_key(name):
            return True
        if name == 'spacebar':
            self.paused = not self.paused
        elif name == 'f' and self.ctrl_down():
            self.clock.set_fast(self.clock.tick_ms == timing.TICK_MS)
        elif self.args.kind == 'model' and name == 'right':
            self.select_action(1)
        elif self.args.kind == 'model' and name == 'left':
            self.select_action(-1)
        elif self.args.kind == 'model' and name == 'b':
            self.body_index += 1
            self.rebuild_model()
        elif name in ACTION_KEYS and self.unit is not None:
            self.unit.do_action(self.unit.find_action(ACTION_KEYS[name], stage=actions.STAGE_UNIQUE))
        elif name == '0' and self.unit is not None:
            self.unit.idle()
        elif name == '[':
            self.set_hour(self.hour - HOUR_STEP)
        elif name == ']':
            self.set_hour(self.hour + HOUR_STEP)
        elif name == 't':
            self.fast_hours = not self.fast_hours
        elif name == 'h' and self.ctrl_down():
            self.show_debug_text = not self.show_debug_text
        elif name == 'i' and self.ctrl_down() and self.battle is not None:
            self.battle.toggle_inventory()
        elif name == 'k' and self.ctrl_down() and self.battle is not None:
            self.battle.toggle_camp()
        elif name == 'l' and self.ctrl_down() and self.unit is not None:
            self.unit.legs_part = WOUNDED_LEGS if self.unit.legs_part == 1.0 else 1.0
        elif name == 'r' and self.ctrl_down():
            self.interface.show_regions = not self.interface.show_regions
            self.interface.set_regions(self.interface.regions)
        elif name == 'escape':
            if self.esc_menu is not None:
                if self.esc_menu.is_open:
                    self.esc_menu.close()
                else:
                    self.esc_menu.open()
            elif self.main_menu is None and self.args.kind in ('menu', 'demo'):
                self.open_main_menu()
            else:
                self.stop()
        return True

    def shift_down(self):
        return bool(self.keys_down & {'shift', 'lshift', 'rshift'})

    def ctrl_down(self):
        return bool(self.keys_down & {'ctrl', 'lctrl', 'rctrl'})

    def on_mouse_pos(self, window, pos):
        pos = ui.mouse_window_pos(pos)
        x, y = self.interface.virtual_point(*pos)
        if self.main_menu is not None:
            self.main_menu.on_mouse(x, y)
            self.cursor.update(pos, True)
        elif self.esc_menu is not None and self.esc_menu.is_open:
            self.esc_menu.on_mouse(x, y)
            self.cursor.update(pos, True)
        elif self.battle is not None:
            # the bodies under the mouse are found once a frame (update_hover), as CCursor__Update reads the mouse
            self.mouse_pending = (tuple(pos), x, y)

    def on_key_up(self, window, key, *args):
        self.keys_down.discard(window._system_keyboard.keycode_to_string(key))
        return True

    def on_touch_down(self, widget, touch):
        if not touch.is_mouse_scrolling:
            x, y = self.interface.virtual_point(touch.x, touch.y)
            if self.main_menu is not None:
                self.main_menu.on_click(x, y)
                return True
            if self.esc_menu is not None and self.esc_menu.is_open:
                self.esc_menu.on_click(x, y)
                return True
            if self.tutorial.on_click(x, y):
                return True
            if self.battle is not None and self.battle.camp_press(x, y):
                return True
            if self.battle is not None and self.battle.on_click(x, y):
                return True
        if self.main_menu is not None:
            return True
        if touch.is_mouse_scrolling:
            factor = 0.9 if touch.button == 'scrolldown' else 1.1
            self.scene.camera.zoom_to(self.scene.camera.distance * factor)
            return True
        self.drag_start = (touch.x, touch.y, False)
        return True

    def on_touch_move(self, widget, touch):
        if self.drag_start is None:
            return False
        dx, dy = touch.dx, touch.dy
        if abs(touch.x - self.drag_start[0]) + abs(touch.y - self.drag_start[1]) > 4:
            self.drag_start = (self.drag_start[0], self.drag_start[1], True)
        camera = self.scene.camera
        values = getattr(getattr(self, 'settings', None), 'values', {}) or {}
        # Inverse camera left/right rotation and up/down movement (CameraFlipXAxis, CameraFlipYAxis)
        camera.reverse_x = bool(values.get('camera_reverse_x', 0))
        camera.reverse_y = bool(values.get('camera_reverse_y', 0))
        camera.drag(dx, dy)
        return True

    def on_touch_up(self, widget, touch):
        if self.battle is not None:
            self.battle.on_release()
            x, y = self.interface.virtual_point(touch.x, touch.y)
            if self.battle.camp_release(x, y):
                return True
        if self.drag_start is None:
            return False
        dragged = self.drag_start[2]
        self.drag_start = None
        if not dragged and self.terrain is not None and self.unit is not None:
            point = self.scene.pick_ground(touch.x, touch.y)
            aim_hit = picking_view.pick_part(self, (touch.x, touch.y)) if self.aim_part is not None else None
            if point is not None or aim_hit is not None:
                run = 'shift' in Window.modifiers if hasattr(Window, 'modifiers') else False
                run = run or 'shift' in self.keys_down or 'lshift' in self.keys_down or 'rshift' in self.keys_down
                run = run or (self.battle is not None and self.battle.run)
                foe = self.unit_at(point) if point is not None else None
                if foe is None and aim_hit is not None:
                    foe = aim_hit[0]
                ctrl = 'ctrl' in self.keys_down or 'lctrl' in self.keys_down or 'rctrl' in self.keys_down
                corpse = self.corpse_at(point) if point is not None and foe is None and not ctrl else None
                if ctrl and self.spell is not None and (foe is not None or (
                        point is not None and self.spell.flags & magic.TARGET_POINT)):
                    if foe is None:
                        foe = magic.Spot(float(point[0]), float(point[1]), float(point[2]))
                    if not self.unit.cast(foe, self.spell, run=run):
                        if _Debug:
                            print('can not cast %s, mana %.1f' % (self.spell, self.unit.fighter.mana))
                elif corpse is not None and self.battle is not None:
                    self.battle.open_loot(corpse)
                elif foe is not None and self.talk_to(foe):
                    pass
                elif foe is not None:
                    self.unit.aim_part = aim_hit[1] if aim_hit is not None else self.aim_part
                    for unit in self.selected_party():
                        unit.attack(foe, run=run)
                elif point is not None and self.lever_at(point) is not None:
                    self.go_to_lever(self.lever_at(point), run)
                elif point is not None:
                    self.lever_to_use = None
                    gait = stats.SPEED_RUN if run and self.gait == stats.SPEED_WALK else self.gait
                    self.order_party_to(point[0], point[1], gait, run)
        return True

    def quick_belt_names(self):
        """
        The names of the belt of the hero (UNIT_QUICK_ITEMS), the quick items among his things when the
        record has no belt of its own.
        """
        record = self.hero_record
        if record is not None and record.quick_items is not None:
            return list(record.quick_items)
        database = self.game_data.quick_items()
        return [name for name in (self.hero_item_names or []) if database.item(name) is not None]

    def set_quick_belt(self, slot, name=None):
        """
        The belt of the hero: the name of a quick item in the slot (0..3), None for an empty slot.
        """
        record = self.hero_record
        if record is None or not 0 <= slot < 4:
            return False
        if name is not None and self.game_data.quick_items().item(name) is None:
            return False
        names = self.quick_belt_names()
        while len(names) < 4:
            names.append(None)
        names[slot] = name
        record.quick_items = [item_name for item_name in names if item_name is not None]
        if self.battle is not None:
            self.battle.quick_key = None
            self.battle.update_quick_items()
        return True

    def selected_party(self):
        """
        The living units of the choice; the hero alone when nothing is chosen.
        """
        units = [unit for unit in self.selected_units if not unit.is_dead]
        if units:
            return units
        return [self.unit] if self.unit is not None and not self.unit.is_dead else []

    def order_party_to(self, x, y, gait, run=False):
        """
        A click on the ground: the chosen units go to the points of the group order around (x, y).
        """
        units = self.selected_party()
        for unit, (px, py) in zip(units, companions_module.formation_points(x, y, len(units))):
            unit.go_to(px, py, run=run, gait=gait)

    def rebuild_corpse(self, unit):
        """
        461cb0: the things taken from a body build its figure again without them.
        """
        if unit is None or getattr(unit, 'figure', None) is None:
            return False
        old = unit.figure
        unit.figure = figure.FigureModel(self.game_data, old.model_name, coefs=old.coefs,
                                         equipment=old.equipment)
        self.scene.replace_unit_view(unit)
        return True

    def living_units(self):
        """
        The living units of the world, the spells spread over them.
        """
        return [u for u in self.map_units if not u.is_dead] + self.zone_party()

    def lever_at(self, point):
        """
        The lever of the map nearest to the point of the ground, None when none is near.
        """
        best = None
        for lever in self.levers:
            x, y = lever.position[0], lever.position[1]
            distance = math.hypot(x - point[0], y - point[1])
            if distance <= LEVER_PICK_RADIUS and (best is None or distance < best[0]):
                best = (distance, lever)
        return best[1] if best else None

    def go_to_lever(self, lever, run=False):
        """
        The hero walks to the lever and uses it there (update_lever_use).
        """
        if not lever.lever.usable:
            self.show_message(LEVER_LOCKED_MESSAGE)
            return
        x, y = lever.position[0], lever.position[1]
        dx, dy = self.unit.x - x, self.unit.y - y
        length = math.hypot(dx, dy) or 1.0
        self.lever_to_use = lever
        self.unit.go_to(x + dx / length * (LEVER_USE_DISTANCE - 0.5), y + dy / length * (LEVER_USE_DISTANCE - 0.5),
                        run=run)

    def update_lever_use(self):
        lever = self.lever_to_use
        hero = self.unit
        if lever is None or hero is None or hero.mode in (unit_module.MODE_WALK, unit_module.MODE_RUN):
            return
        self.lever_to_use = None
        if math.hypot(hero.x - lever.position[0], hero.y - lever.position[1]) > LEVER_USE_DISTANCE:
            return
        hero.target_yaw = ai.look_yaw(lever.position[0] - hero.x, lever.position[1] - hero.y)
        if lever.lever.use(self.hero_science()):
            self.script_host.lever_switched(lever)
        else:
            self.show_message(LEVER_KEY_MESSAGE)

    def unit_at(self, point):
        """
        The living unit of the map nearest to the point of the ground within its radius, None when there is none.
        """
        best = None
        for u in self.map_units:
            if u.fighter is None or u.is_dead:
                continue
            distance = math.hypot(u.x - point[0], u.y - point[1])
            if distance <= u.fighter.radius + PICK_MARGIN and (best is None or distance < best[0]):
                best = (distance, u)
        return best[1] if best else None

    def talk_to(self, unit):
        """
        The click on a peaceful unit with topics of the game "b.<owner>.<name>" opens the conversation
        instead of an attack (game.exe 607320 and 608b20).
        """
        if self.battle is None or self.unit is None:
            return False
        if self.ai.hostile(self.unit, unit):
            return False
        return self.battle.open_talk(unit)

    def corpse_at(self, point):
        """
        The dead unit of the map nearest to the point of the ground within its radius (the click on a body
        opens its things, game.exe 5942d0).
        """
        best = None
        for u in self.map_units:
            if not u.is_dead:
                continue
            radius = u.fighter.radius if u.fighter is not None else 0.5
            distance = math.hypot(u.x - point[0], u.y - point[1])
            if distance <= radius + PICK_MARGIN and (best is None or distance < best[0]):
                best = (distance, u)
        return best[1] if best else None

    def add_arrow(self, arrow):
        for sound in arrow.sounds:
            self.audio.play_sound(sound)
        arrow.sounds = []
        model_name = arrow.model_name
        if model_name is None and arrow.spell is not None and arrow.spell.index in magic.MISSILE_TRAILS:
            # CEffectArrow 0x4e: the fire arrow and the magic of Rick fly as the arrow of a bow
            model_name = missiles.ARROW_MODEL
        if model_name is None:
            # the missile of a spell: its particles are not made yet
            self.arrows.append((arrow, None, None))
            return
        model = figure.FigureModel(self.game_data, model_name)
        p = prop_module.Prop(model, arrow.world_matrix())
        view = self.scene.add_prop(p, missiles.ARROW_TEXTURE, removable=True)
        self.arrows.append((arrow, p, view))

    def update_stuck_arrows(self, dt_ms):
        """
        The arrows in the bodies: they go with the unit and go away after ARROW_STICK_MS.
        """
        left = []
        for p, view, target, offset, matrix, time_left in self.stuck_arrows:
            time_left -= dt_ms
            if time_left <= 0.0:
                if view is not None:
                    self.scene.remove_prop(view)
                continue
            if target is not None and hasattr(target, 'x'):
                moved = matrix.copy()
                moved[0:3, 3] = (target.x + offset[0], target.y + offset[1], target.z + offset[2])
                p.matrix = moved
            left.append([p, view, target, offset, matrix, time_left])
        self.stuck_arrows = left

    def update_arrows(self, dt_ms):
        """
        The flight of the arrows; an arrow which came gives its blow to the target (67bde0).
        """
        ticks = dt_ms / self.clock.tick_ms
        self.update_stuck_arrows(dt_ms)
        flying = []
        for arrow, p, view in self.arrows:
            if arrow.update(ticks):
                if p is not None:
                    p.matrix = arrow.world_matrix()
                flying.append((arrow, p, view))
                continue
            stuck = False
            if p is not None and arrow.spell is None and arrow.target is not None and hasattr(arrow.target, 'x'):
                # the effect 0x5b: the arrow stays in the body for a while
                target = arrow.target
                offset = (arrow.x - target.x, arrow.y - target.y, arrow.z - target.z)
                matrix = arrow.world_matrix()
                self.stuck_arrows.append([p, view, target, offset, matrix, ARROW_STICK_MS])
                stuck = True
            if view is not None and not stuck:
                self.scene.remove_prop(view)
            if arrow.trail is not None:
                arrow.trail.stop()
                self.loose_emitters.append(arrow.trail)
            if arrow.flash is not None:
                code, scale = arrow.flash
                self.loose_emitters.append(particles.Emitter(code, (arrow.x, arrow.y, arrow.z), scale=scale))
            if arrow.spell is not None:
                light = magic.flash_light(arrow.spell, (arrow.x, arrow.y, arrow.z))
                if light is not None:
                    self.loose_lights.append(light)
                # 681b70: the sound of the hit takes the place of the one of the flight
                sound = magic.end_sound(arrow.spell, (arrow.x, arrow.y, arrow.z))
                if sound is not None:
                    self.audio.play_sound(sound)
            # a missile of a trap or of the scripts (engine.trap.Caster) has no unit behind it
            shooter = arrow.shooter
            if arrow.target is not None and (shooter.fighter is not None or isinstance(shooter, trap_module.Caster)):
                shooter.deliver_blow(arrow.target, arrow.hit)
        self.arrows = flying

    def add_unit_emitter(self, emitter, unit):
        """
        The particles of a spell which follow the unit (the healing).
        """
        self.unit_emitters.append((emitter, unit))

    def update_spell_sounds(self, effects):
        """
        The sounds which the effects of the spells made this frame, the loops of the effects, the missiles
        and the units, the sounds of the enchantments which came to the units or left them (5112a0, 510890).
        """
        loops = []
        for effect in effects:
            for sound in effect.sounds:
                self.audio.play_sound(sound)
            effect.sounds = []
            if effect.loop is not None and not effect.finished:
                loops.append(effect.loop)
        for arrow, p, view in self.arrows:
            if arrow.loop is not None:
                arrow.loop.position = (arrow.x, arrow.y, arrow.z)
                loops.append(arrow.loop)
        seen = {}
        for u in self.living_units():
            position = (u.x, u.y, u.z)
            found, seen[id(u)] = magic.enchantment_sounds(self.enchantments_seen.get(id(u), set()), u.fighter,
                                                          position)
            for sound in found:
                self.audio.play_sound(sound)
            loop = magic.enchantment_loop(u.fighter, position)
            if loop is not None:
                loops.append(loop)
            warning = magic.enchantment_warning(u.fighter, position)
            if warning is not None:
                self.audio.play_sound(warning, key=('warning', id(u)))
        self.enchantments_seen = seen
        if self.scripts is not None:
            loops.extend(self.scripts.sources.sound_loops())
        self.audio.set_loops(loops)

    def update_particles(self, dt_ms):
        """
        The particles of the missiles and the loose ones run their whole ticks (the ones of the effects
        on the ground run with them); the scene draws all of them between the ticks.
        """
        self.particle_ticks += dt_ms / self.clock.tick_ms
        while self.particle_ticks >= 1.0:
            self.particle_ticks -= 1.0
            for arrow, p, view in self.arrows:
                if arrow.trail is not None:
                    arrow.trail.update((arrow.x, arrow.y, arrow.z))
            self.loose_emitters = [e for e in self.loose_emitters if e.update()]
            self.unit_emitters = [(e, u) for e, u in self.unit_emitters if e.update((u.x, u.y, u.z))]
            if self.weather is not None:
                self.weather.tick()
            self.loose_lights = [light for light in self.loose_lights if light.update()]
            if self.scripts is not None:
                self.scripts.sources.update()
        emitters = list(self.loose_emitters)
        emitters.extend(e for e, u in self.unit_emitters)
        emitters.extend(arrow.trail for arrow, p, view in self.arrows if arrow.trail is not None)
        bolts = []
        lights = list(self.loose_lights)
        if self.scripts is not None:
            emitters.extend(self.scripts.sources.emitters())
            lights.extend(self.scripts.sources.spell_lights())
            bolts.extend(self.scripts.sources.bolts())
        for effect in self.spell_effects:
            emitters.extend(effect.emitters)
            bolts.extend(effect.bolts())
            lights.extend(effect.lights)
        fraction = min(self.particle_ticks, 1.0)
        precipitation = None
        if self.weather is not None:
            emitters.extend(self.weather.particles())
            bolts.extend(self.weather.bolts())
            precipitation = self.weather.precipitation(fraction)
        self.scene.set_particles(emitters, fraction, bolts, precipitation=precipitation)
        self.scene.set_point_lights([(light.at(fraction), light.color, light.radius) for light in lights])

    def on_blow(self, attacker, target, hit, dealt):
        if self.scene is not None and target is not None:
            self.scene.camera.shake(point=(target.x, target.y, target.z + target.height() * 0.5))
        if dealt is None:
            if _Debug:
                print('%s misses %s' % (attacker, target))
        else:
            if _Debug:
                print('%s hits %s: %.1f, health %.1f' % (attacker, target, dealt, target.fighter.health))

    def on_kill(self, killer, victim):
        """
        553f40: the party shares the experience of a monster killed by its unit (5911c0).
        """
        if not experience.is_rewarded_kill(killer, victim):
            return
        shares = experience.reward_units(self.script_host.party_units(0), victim.kill_experience)
        if _Debug:
            print('%s killed %s: experience %g, shares %s' % (killer, victim, victim.kill_experience, shares))

    def on_item_wear(self, unit, item, message):
        """
        5253e0 -> 552780: a unit of the player tells that its item is worn out or broken; the group of
        engine.acks gives the voice and its text.
        """
        if _Debug:
            print('%s: %s durability %.1f/%.1f, message 0x%x' % (unit, item, item.durability, item.max_durability,
                                                                  message))
        if self.battle is not None and getattr(unit, 'fighter', None) is not None and unit.fighter.in_party:
            self.battle.say_unit_ack(unit, acks_module.item_message_group(message))

    def on_cant_run(self, unit, reason):
        """
        548520 (partly): a unit of the player told to run says why it walks (the message 0x15 for the legs, 0x14 for
        the weight).
        """
        if self.battle is not None:
            self.battle.say_unit_ack(unit, acks_module.run_limit_group(reason))

    def on_unit_signal(self, unit, level):
        """
        A mood signal of engine.signals (5aea80) starts: its nearest group of acks.db gives a voice.
        """
        if self.battle is not None:
            self.battle.say_unit_ack(unit, acks_module.signal_group(level))

    def party_fighters(self):
        """
        The units of the party for the deals of the camp (engine.trade).
        """
        return self.script_host.party_units(0)

    def fix_items(self):
        """
        FixItems: every item of the world is whole again, the armor of the units is built again.
        """
        units = self.zone_party() + list(self.map_units)
        for unit in units:
            for item in trade_module.worn_items(unit):
                trade_module.repair_item(unit, item)
        for record in (self.roster.records() if self.roster is not None else []):
            for item in record.worn_items.values():
                item.durability = item.max_durability

    def take_screenshot(self, *args):
        Window.screenshot(name=self.args.screenshot_file_path)
        if self.args.exit_after_screenshot:
            self.stop()

#------------------------------------------------------------------------------


def gl_pose(x, y, z, rotation_matrix):
    m = np.identity(4)
    m[0:3, 0:3] = rotation_matrix
    m[0:3, 3] = (x, y, z)
    return m


def parse_position(text):
    x, y = text.split(',')
    return float(x), float(y)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Evil Islands remake on Python + Kivy')
    p.add_argument('kind', choices=['model', 'map', 'menu', 'demo'])
    p.add_argument('name', nargs='?', default=None, help='model name (unhuma, unmogo...) or map name (bz10k...)')
    p.add_argument('--start-map', dest='start_map', default='zone1', help='the map the "new game" of the main menu opens')
    p.add_argument('--show-regions', dest='show_regions', action='store_true',
                   help='outline the regions of the menus (the key R switches them)')
    p.add_argument('--data', dest='data_dir_path', metavar='DATA_PATH', default=None,
                   help='folder of the original game, by default EI_DATA_PATH')
    p.add_argument('--unit', default='unhuma', help='model of the unit on the map')
    p.add_argument('--weapon', default=None, choices=[None] + list(actions.WEAPON_BITS))
    p.add_argument('--texture', default=None)
    p.add_argument('--item', dest='item_names', action='append', default=[], metavar='ITEM',
                   help='armor or weapon of items.idb like "hadagan high2 plate.adamantium" or "saber.steel", can repeat')
    p.add_argument('--spell', default=None, metavar='SPELL',
                   help='the spell of the hero for Ctrl + click: a name or a code of spells.sdb and modifiers, "Fire Arrow+r1+e1"')
    p.add_argument('--prototype', default=None, help='unit prototype of units.udb for the skin and the hair')
    p.add_argument('--position', type=parse_position, default=None, metavar='X,Y',
                   help='where the unit stands on the map, the middle of the map by default')
    p.add_argument('--hour', type=float, default=None, help='the hour of the day on the map, WS_TIME of the .mob by default')
    p.add_argument('--distance', type=float, default=None, help='distance of the camera from the unit on the map')
    p.add_argument('--pitch', type=float, default=None, help='angle of the camera above the ground, degrees')
    p.add_argument('--action', default=None, help='action of the .adb to start with')
    # the Tracking camera of game.exe is off at the start; Home switches it (keys.camera_track)
    p.add_argument('--follow', dest='follow', action='store_true',
                   help='the camera tracks the hero from the start')
    p.add_argument('--no-sound', dest='sound', action='store_false', help='do not play the sounds of sfx.res')
    p.add_argument('--no-objects', dest='objects', action='store_false', help='do not load the objects of the .mob file')
    p.add_argument('--screenshot', dest='screenshot_file_path', metavar='SCREENSHOT_PATH', default=None)
    p.add_argument('--screenshot-delay', dest='screenshot_delay', type=float, default=2.0)
    p.add_argument('--exit', dest='exit_after_screenshot', action='store_true')
    p.add_argument('--tutorial', action='store_true', help='show the tutorial windows of the scripts')
    return p.parse_args(argv)


def main():
    RemakeApp(parse_args(sys.argv[1:])).run()


if __name__ == '__main__':
    main()
