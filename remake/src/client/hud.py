"""
The battle screen of the client: the panels of engine.hud in the 3D interface of the scene, the tooltips,
the heads of the portraits, the mini-map with its compass, the strip of items (engine.inventory, opened by the key I,
in game.exe it belongs to the camp screens) and the buttons which already do something.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math

import numpy as np

#------------------------------------------------------------------------------

from engine import ai
from engine import acks as acks_module
from engine import figure
from engine import equipment
from engine import hud
from engine import inventory
from engine import loot as loot_module
from engine import options
from engine import prop as prop_module
from engine import quickitems as quickitems_module
from engine import quat
from engine import spellvisual
from engine import stats as stats_module
from engine import timing
from engine import trade as trade_module
from engine import unit as unit_module
from client import briefing as briefing_view
from client import camp as camp_view
from client import footprints as footprints_view
from client import gl
from client import loot as loot_view
from client import globalmap as globalmap_view
from client import questmap as questmap_view
from client import talk as talk_view
from client import ui
from client import zoneexit as zoneexit_view

#------------------------------------------------------------------------------

# 65e540 (partly): the tooltip shows after the mouse stands over a button for ToolTip Time (79be70): the slider
# of the option times 0.25 s (73c3b0); at 0 there are no tooltips
TOOLTIP_STEP_MS = 250.0
TOOLTIP_COLOR = (228, 215, 167)
TOOLTIP_SHADE = (0.0, 0.0, 0.0, 0.7)
TOOLTIP_PADDING = 4
TOOLTIP_OFFSET = 20

TEXT_COLOR = (228, 215, 167)

SPEED_PAUSE = 0
SPEED_NORMAL = 1
SPEED_FAST = 2

MODE_WALK = 2
MODE_RUN = 3

# 640270: the head of a portrait is turned by pi about x, the y of the interface goes down
HEAD_ROTATION = (0.0, 1.0, 0.0, 0.0)

ZOOM_ACTIONS = {'zoom in': 1, 'zoom out': -1}

# TAB (the action "obj"): the names of the units this near to the hero
OBJECT_NAMES_DISTANCE = 18.0
OBJECT_NAMES_COLOR = (190, 235, 255)

# 63ba70 (the window of the battle interface, 80..240 x 520..600): four regions of 40 x 80 for the quick
# items of the hero; the slots themselves are drawn by battle00, the window adds only the icons and counts
QUICK_SLOT_RECTS = tuple((80 + i * 40, 520, 120 + i * 40, 600) for i in range(4))
QUICK_LAYER = 'quick items'
QUICK_TEXT_COLOR = (228, 215, 167)

#------------------------------------------------------------------------------


class BattleScreen(object):

    def __init__(self, app):
        self.app = app
        self.hidden = set()
        self.mode_index = MODE_WALK
        self.hovered = None
        self.hover_ms = 0.0
        self.mouse = (0.0, 0.0)
        self.time_s = 0.0
        self.speed_index = None
        self.hour = None
        # the health of the hero and of the parts of its body which the panels show
        self.shown_state = None
        self.shown_enchantments = None
        self.messages = []
        self.heads = []
        # the shares of the bars of the portraits drawn last
        self.shown_bars = None
        self.info_figure = None
        self.info_figure_view = None
        self.portrait_count = 0
        self.minimap = None
        self.zoom_direction = 0
        self.compass = None
        self.compass_view = None
        self.compass_offset = np.identity(4)
        self.info_mode = hud.INFO_GENERAL
        # the prints of the steps of the units (G9)
        self.footprints = footprints_view.Footprints(app)
        self.zone_exit = zoneexit_view.ZoneExit(app)
        self.quick_key = None
        # 649ab0: the text window shows the messages or the collected items
        self.item_messages = []
        self.text_mode = 'messages'
        # the answers of the hero to the orders (engine.acks)
        self.ack_unit = None
        self.ack_player = None
        self.ack_mode = None
        self.ack_foe = None
        self.inventory = None
        self.camp = None
        self.camp_paused = False
        self.camp_opened_inventory = False
        # the camp covers the battle screen: the panels and the models of the battle HUD go away, the strip stays
        self.camp_hides_hud = False
        self.questmap = None
        self.globalmap = None
        self.globalmap_paused = False
        self.briefing = None
        self.briefing_variable = None
        self.loot = None
        self.talk = None
        self.talk_paused = False
        self.modal_paused = False
        # item index -> (view, prop, offset matrix) of the models in the slots of the strip
        self.item_views = {}
        # the deals of the camp: the prices and the repair of the things with the money of the party
        self.trade = trade_module.Trade(app.party, app.game_data.items())
        # the mouse is over a panel of the battle screen (the cursor stays the default one)
        self.over_interface = False

    @property
    def run(self):
        return self.mode_index == MODE_RUN

    def setup(self):
        self.update_static()
        self.update_portraits()
        self.setup_minimap()
        self.update(0.0)

    def setup_minimap(self):
        """
        The texture and the size of the map come from "#maps" of map.txt, without them the map is not drawn.
        """
        app = self.app
        zone = app.game_data.zone_of_map(app.map_name) if app.map_name else None
        if zone is None or zone.sectors is None or not zone.minimap_texture_name:
            return
        self.minimap = hud.Minimap(zone.minimap_texture_name, zone.sectors[0], zone.sectors[1])
        self.add_compass()

    def add_compass(self):
        game_data = self.app.game_data
        if not game_data.has_model(hud.COMPASS_MODEL):
            return
        battle00 = game_data.texture(hud.BATTLE_TEXTURE)
        if battle00 is None:
            return
        model = figure.FigureModel(game_data, hud.COMPASS_MODEL)
        self.compass = prop_module.Prop(model)
        # the mesh of in25arrow already lies on the ring around its origin, the offset 4.15 of its .bon
        # would put it far to the right, so it is taken back (a guess, game.exe does not show it)
        bone = game_data.bones(hud.COMPASS_MODEL).get(hud.COMPASS_MODEL)
        if bone is not None:
            self.compass_offset[0:3, 3] = -bone[0]
        texture = self.app.scene.composed_texture('compass:' + hud.BATTLE_TEXTURE,
                                                  hud.shifted_texture(battle00, hud.COMPASS_TEXTURE_OFFSET))
        self.compass_view = self.app.scene.add_prop(self.compass, hud.BATTLE_TEXTURE, interface=True, texture=texture)
        self.place_compass(0.0)

    def place_compass(self, angle):
        position = gl.interface_point(hud.COMPASS_POINT[0], hud.COMPASS_POINT[1], hud.COMPASS_DEPTH)
        rotation = quat.multiply(quat.from_axis_angle((0.0, 0.0, 1.0), angle), HEAD_ROTATION)
        self.compass.matrix = prop_module.placement_matrix(position, rotation, hud.COMPASS_SCALE) @ self.compass_offset

    def update_minimap(self, dt_ms):
        if self.minimap is None or self.camp_hides_hud:
            return
        if self.zoom_direction:
            self.minimap.zoom_by(self.zoom_direction, dt_ms / 1000.0)
        scene = self.app.scene
        if 'minimap' in self.hidden:
            scene.set_hud_layer('minimap map', [])
            return
        camera = (float(scene.camera.target[0]), float(scene.camera.target[1]))
        units = []
        hero = self.app.unit
        own_player = self.app.ai.player_of(hero) if hero is not None else ai.PARTY_PLAYER
        for u in self.app.map_units:
            # 643d80: the marks of the units the player sees (+0x50d8)
            if u.is_dead or u.unseen:
                continue
            relation = self.app.ai.diplomacy.relation(own_player, self.app.ai.player_of(u))
            if relation == ai.SAME_SIDE:
                kind = hud.MARK_OWN
            elif relation == ai.WAR:
                kind = hud.MARK_ENEMY
            else:
                kind = hud.MARK_NEUTRAL
            units.append((u.x, u.y, u.yaw, kind))
        if hero is not None:
            units.append((hero.x, hero.y, hero.yaw, hud.MARK_OWN))
        scene.set_hud_layer('minimap map', self.minimap.elements(camera, units))
        if self.compass is not None:
            self.place_compass(hud.compass_angle(scene.camera.yaw))

    def panels(self):
        """
        The panels with mouse regions, the hidden ones keep only their open/close button.
        """
        result = []
        if self.inventory is not None:
            # the strip takes the bottom of the screen, like the camp screens of game.exe
            result.append(hud.Panel('inventory', [], self.inventory.regions()))
        for panel in (self.info_panel(), hud.text_panel(), hud.minimap_panel(), self.pause_panel(), hud.mode_dial(self.mode_index)):
            if self.inventory is not None and panel.name in ('pause', 'mode'):
                continue
            if panel.name in self.hidden:
                panel = hud.Panel(panel.name, [], panel.regions[:1])
            result.append(panel)
        if self.inventory is None:
            for x in hud.portrait_slots(self.portrait_count):
                result.append(hud.portrait(x))
        return result

    def pause_panel(self):
        pulse = hud.pulse(self.time_s) if self.speed_index == SPEED_PAUSE else 1.0
        speed = SPEED_NORMAL if self.speed_index is None else self.speed_index
        return hud.pause_dial(self.app.hour, speed, pulse)

    def update_static(self):
        scene = self.app.scene
        self.shown_enchantments = self.enchantments_key()
        if self.camp_hides_hud:
            self.clear_battle_hud()
            return
        for panel in (self.info_panel(), hud.text_panel(), hud.minimap_panel()):
            elements = [] if panel.name in self.hidden else panel.elements
            if panel.name == 'info' and self.info_mode == hud.INFO_SPELL_EFFECTS and panel.name not in self.hidden:
                elements = elements + hud.spell_icons(self.spell_icon_textures())
            scene.set_hud_layer(panel.name, elements)
        self.update_info_text()
        scene.set_hud_layer('mode', [] if self.inventory is not None else hud.mode_dial(self.mode_index).elements)
        if self.compass_view is not None:
            # the compass is a model, it goes away with the panel
            self.app.scene.show_interface_prop(self.compass_view, 'minimap' not in self.hidden)
        if self.info_figure_view is not None:
            # the figure of the information window is a model too
            self.app.scene.show_interface_prop(self.info_figure_view, 'info' not in self.hidden)
        self.update_text()

    def clear_battle_hud(self):
        """
        The camp covers the battle screen: its panels, portraits and models are not drawn under it. The strip
        of items stays (the camp uses it) and comes back with client.hud.update_inventory.
        """
        scene = self.app.scene
        for name in ('info', 'text', 'minimap', 'minimap map', 'pause', 'mode', 'portraits'):
            scene.set_hud_layer(name, [])
        self.app.interface.set_layer('info text', [])
        self.app.interface.set_layer('tooltip', [])
        if self.compass_view is not None:
            scene.show_interface_prop(self.compass_view, False)
        if self.info_figure_view is not None:
            scene.show_interface_prop(self.info_figure_view, False)

    def set_camp_hides_hud(self, hidden):
        """
        The camp is open (True) or closed (False): hide or bring back the battle HUD, the camp draws its own.
        """
        self.camp_hides_hud = hidden
        if hidden:
            self.clear_battle_hud()
        else:
            self.update_portraits()
            if self.inventory is not None:
                for view in self.heads:
                    self.app.scene.show_interface_prop(view, False)
                self.app.scene.set_hud_layer('portraits', [])
            self.update_static()
            self.update_minimap(0)

    def enchantments_key(self):
        """
        The spells on the hero in the order of the slots of the information window.
        """
        fighter = self.app.unit.fighter if self.app.unit is not None else None
        if fighter is None:
            return ()
        return tuple(e.index for e in fighter.enchantments[:hud.SPELL_ICON_SLOTS])

    def spell_icon_textures(self):
        """
        The textures "spell%04d" of the spells on the hero by their texture_type (0x60 of the record of
        game.exe), None for a spell without a texture.
        """
        game_data = self.app.game_data
        spells = game_data.spells().spells
        result = []
        for index in self.enchantments_key():
            spell = spells[index] if 0 <= index < len(spells) else None
            texture_name = 'spell%04d' % spell.texture_type if spell is not None else None
            if texture_name is not None and game_data.texture(texture_name) is None:
                texture_name = None
            result.append(texture_name)
        return result

    def hero_stats(self):
        """
        engine.stats.UnitStats of the hero from its prototype, None without it.
        """
        return self.unit_stats(self.app.unit)

    def unit_stats(self, unit):
        """
        engine.stats.UnitStats of a unit from its prototype, None without it.
        """
        prototype = self.unit_prototype(unit)
        return stats_module.from_prototype(prototype) if prototype is not None else None

    def info_unit(self):
        """
        The unit of the information window: the one under the mouse, otherwise the hero (G12).
        """
        unit = getattr(self.app, 'hover_unit', None)
        if unit is not None and unit.fighter is not None:
            return unit
        return self.app.unit

    def unit_state(self, unit):
        """
        (health, mana, hit locations) of a unit: (current, maximum) from its parts or its prototype,
        the mana of the fighter when it has one.
        """
        prototype = self.unit_prototype(unit)
        if prototype is None:
            return None, None, []
        unit_stats = stats_module.from_prototype(prototype)
        mana = (unit_stats.mana, unit_stats.max_mana)
        fighter = unit.fighter if unit is not None else None
        if fighter is not None:
            mana = (max(0.0, fighter.mana), fighter.max_mana)
        if fighter is not None and fighter.parts:
            health = (max(0.0, fighter.health), fighter.max_health)
            return health, mana, [(part.name, max(0.0, part.health_part)) for part in fighter.parts]
        unit_type = self.app.game_data.units().unit_type(prototype.type_name)
        locations = unit_type.hit_locations if unit_type is not None else []
        health = (unit_stats.health, unit_stats.max_health)
        return health, mana, [(location.name, 1.0) for location in locations]

    def hero_state(self):
        """
        (health, mana, hit locations) of the hero.
        """
        return self.unit_state(self.app.unit)

    def unit_prototype(self, unit):
        """
        The prototype of a unit ("--prototype" or the one of the demo for the hero of the command line,
        otherwise the first one of the unit type with the model of the unit).
        """
        app = self.app
        if unit is None:
            prototype_name = app.hero_prototype_name
            if not prototype_name:
                return None
            return app.prototype(None, prototype_name)
        prototype_name = getattr(unit, 'prototype_name', None) or app.hero_prototype_name
        return app.prototype(unit.figure.model_name, prototype_name)

    def hero_prototype(self):
        return self.unit_prototype(self.app.unit)

    def info_panel(self):
        health, mana, locations = self.unit_state(self.info_unit())
        return hud.info_panel(self.info_mode, health[0] / float(health[1]) if health and health[1] else 0.0,
                              mana[0] / float(mana[1]) if mana and mana[1] else None,
                              [part for _, part in locations])

    def update_info_text(self):
        items = []
        if 'info' not in self.hidden and not self.camp_hides_hud:
            unit = self.info_unit()
            health, mana, locations = self.unit_state(unit)
            game_data = self.app.game_data
            names = list(hud.INFO_LABELS) + [hud.HIT_LOCATION_LABEL % n for n, _ in locations]
            labels = {}
            for name in names:
                text = game_data.text(name)
                if text:
                    labels[name] = text.replace('\r', '').strip()
            unit_stats = self.unit_stats(unit)
            attack = unit_stats.attack if unit_stats else None
            defense = unit_stats.defense if unit_stats else None
            damage = (unit_stats.damage_min, unit_stats.damage_max) if unit_stats else None
            for text, rect, color in hud.info_texts(self.info_mode, labels, health, mana,
                                                    [(n, part <= 0.0) for n, part in locations],
                                                    attack, defense, damage):
                items.append(ui.text_item(text, rect, ui.FONT_SMALL, color))
            if unit is not None and unit is not self.app.unit and health is not None:
                name = getattr(unit, 'name', None) or unit.figure.model_name
                items.append(ui.text_item('%s %d/%d' % (name, round(max(health[0], 0)), round(health[1])),
                                          (20, 20, 180, 36), ui.FONT_SMALL, hud.INFO_TEXT_COLOR))
        self.app.interface.set_layer('info text', items)

    def update_portraits(self):
        """
        A portrait of every member of the party in the zone: the frame, the bars and the head model of 6407a0;
        the whole figure of the information window (6456a0) goes with the hero.
        """
        scene = self.app.scene
        for view in self.heads:
            scene.remove_interface_prop(view)
        self.heads = []
        if self.info_figure_view is not None:
            scene.remove_interface_prop(self.info_figure_view)
        self.info_figure = None
        self.info_figure_view = None
        units = self.app.zone_party()
        self.portrait_count = len(units)
        for x, unit in zip(hud.portrait_slots(len(units)), units):
            view = self.add_head(unit, x)
            if view is not None:
                self.heads.append(view)
        info_unit = self.info_unit()
        if info_unit is not None:
            self.add_info_figure(info_unit)
        self.shown_bars = None
        self.update_portrait_bars()

    def update_portrait_bars(self):
        """
        641040 (partly): the bars of the health and the mana of the portraits follow the shares of the units on
        every tick (u = (share * 64 + 9) / 256 of the texture); the flash of a wound and the moves of the heads
        are not here.
        """
        units = self.app.zone_party()
        shares = tuple(hud.bar_shares(unit) for unit in units)
        if shares == self.shown_bars:
            return
        self.shown_bars = shares
        elements = []
        for x, (health, mana) in zip(hud.portrait_slots(len(units)), shares):
            elements.extend(hud.portrait(x, health, mana).elements)
        self.app.scene.set_hud_layer('portraits', elements)

    def update_info_figure(self):
        """
        The whole figure of the information window follows the unit under the mouse.
        """
        scene = self.app.scene
        if self.info_figure_view is not None:
            scene.remove_interface_prop(self.info_figure_view)
        self.info_figure = None
        self.info_figure_view = None
        unit = self.info_unit()
        if unit is not None:
            self.add_info_figure(unit)

    def add_head(self, unit, x):
        game_data = self.app.game_data
        model_name = unit.figure.model_name
        head_name = hud.head_model_name(model_name, unit.figure.coefs[1], 1)
        if not game_data.has_model(head_name):
            return None
        model = figure.FigureModel(game_data, head_name)
        position = gl.interface_point(x + hud.PORTRAIT_WIDTH * 0.5, hud.PORTRAIT_HEAD_Y, hud.PORTRAIT_HEAD_DEPTH)
        rotation = HEAD_ROTATION
        head = prop_module.Prop(model, prop_module.placement_matrix(position, rotation, hud.portrait_head_scale()))
        face_name = hud.face_texture_name(model_name, 0)
        face = game_data.texture(face_name)
        if face is None:
            return None
        texture = self.app.scene.composed_texture('portrait:' + face_name, hud.portrait_face_texture(face))
        return self.app.scene.add_prop(head, face_name, interface=True, texture=texture)

    def add_info_figure(self, unit):
        """
        6456a0: the whole figure of the unit in the information window. game.exe draws it into its own render
        target of the region (20, 0, 180, 220); the remake puts the model itself into the 3D interface.
        """
        game_data = self.app.game_data
        model_name = unit.figure.model_name
        if not game_data.has_model(model_name):
            return
        model = figure.FigureModel(game_data, model_name, coefs=unit.figure.coefs, equipment=unit.figure.equipment)
        info = prop_module.Prop(model)
        position = gl.interface_point(hud.INFO_FIGURE_POINT[0], hud.INFO_FIGURE_POINT[1], hud.INFO_FIGURE_DEPTH)
        info.matrix = prop_module.placement_matrix(position, hud.INFO_FIGURE_ROTATION, hud.INFO_FIGURE_SCALE)
        info.idle()
        texture_name = unit.figure.default_texture_name()
        if texture_name is None:
            return
        textures = {group: self.app.scene.composed_texture('info figure:%s:%s' % (model_name, key), rgba)
                    for group, (key, rgba) in model.composed_textures().items()}
        view = self.app.scene.add_prop(info, texture_name, interface=True, textures=textures)
        self.info_figure = info
        self.info_figure_view = view

    def update_text(self):
        items = []
        lines = self.item_messages if self.text_mode == 'items' else self.messages
        if 'text' not in self.hidden and lines and not self.camp_hides_hud:
            text = '\n'.join(lines[-4:])
            items.append(ui.text_item(text, hud.TEXT_BOX, ui.FONT_SMALL, TEXT_COLOR, valign='bottom'))
        self.app.interface.set_layer('text window', items)

    def update_ground_marks(self):
        """
        The rings under the units: the hero of the party (green), the unit under the mouse (yellow, red for
        an enemy) and the foe of the hero (red). game.exe draws its own rings, their geometry is not found.
        """
        scene = getattr(self.app, 'scene', None)
        if scene is None or self.app.terrain is None or self.camp_hides_hud:
            if scene is not None:
                scene.set_ground_marks([])
            return
        marks = []
        seen = set()
        hero = self.app.unit

        def add(unit, color):
            if unit is None or unit.fighter is None or unit.is_dead or id(unit) in seen:
                return
            seen.add(id(unit))
            radius = max(0.65, unit.fighter.radius) * 1.15
            marks.append((unit.x, unit.y, unit.z, radius, color))

        add(hero, (0.30, 1.0, 0.35, 0.55))
        for unit in getattr(self.app, 'selected_units', ()):
            if unit is not hero:
                add(unit, (0.35, 0.75, 1.0, 0.50))
        if hero is not None:
            add(getattr(hero, 'foe', None), (1.0, 0.25, 0.20, 0.55))
        hover = getattr(self.app, 'hover_unit', None)
        if hover is not None and hover is not hero:
            hostile = hero is not None and self.app.ai.hostile(hero, hover)
            add(hover, (1.0, 0.35, 0.25, 0.40) if hostile else (1.0, 0.88, 0.30, 0.40))
        scene.set_ground_marks(marks)

    def update_figure_scales(self):
        """
        37 and 38: the figure of a unit grows or shrinks while the spell is on it (engine.spellvisual).
        """
        units = self.app.zone_party() + list(self.app.map_units)
        spellvisual.update_units(units)

    def update_object_names(self):
        """
        TAB: the name and the health of the units near the hero over their heads (the remake's reading of
        the action "obj").
        """
        scene = getattr(self.app, 'scene', None)
        hero = getattr(self.app, 'unit', None)
        if not getattr(self.app, 'show_object_names', False) or scene is None or hero is None or self.camp_hides_hud:
            self.app.interface.set_layer('object names', [])
            return
        items = []
        seen = set()
        for unit in [hero] + list(self.app.map_units):
            if unit.fighter is None or unit.is_dead or id(unit) in seen:
                continue
            seen.add(id(unit))
            if unit is not hero and math.hypot(unit.x - hero.x, unit.y - hero.y) > OBJECT_NAMES_DISTANCE:
                continue
            point = scene.screen_point((unit.x, unit.y, unit.z + unit.height() + 0.35))
            if point is None or not (0 <= point[0] <= ui.WIDTH and 0 <= point[1] <= ui.HEIGHT):
                continue
            name = getattr(unit, 'name', None) or unit.figure.model_name
            text = '%s %d' % (name, max(0, round(unit.fighter.health)))
            width = 90
            items.append(ui.text_item(text, (point[0] - width * 0.5, point[1] - 14, point[0] + width * 0.5, point[1]),
                                      ui.FONT_SMALL, OBJECT_NAMES_COLOR, align='center'))
        self.app.interface.set_layer('object names', items)

    def quick_belt(self):
        """
        [(QuickItem, count)] of the belt of the hero (UNIT_QUICK_ITEMS): the counts are those of the things
        of the party, the order that of the belt; without a belt, the quick items of the things of the hero.
        """
        party = getattr(self.app, 'party', None)
        names = list(getattr(party, 'items', []) or [])
        order = self.app.quick_belt_names()
        return quickitems_module.belt(names, self.app.game_data.quick_items(), order=order)

    def update_quick_items(self):
        """
        63ba70: the four slots of the quick items with the icons qitem<number> of items.idb.
        """
        if self.camp_hides_hud:
            key = ('hidden', )
            belt = []
        else:
            belt = self.quick_belt()
            key = tuple((quick_item.name, count) for quick_item, count in belt)
        if key == self.quick_key:
            return
        self.quick_key = key
        items = []
        for index, rect in enumerate(QUICK_SLOT_RECTS):
            if index >= len(belt):
                continue
            quick_item, count = belt[index]
            texture_name = 'qitem%04d' % quick_item.icon
            if self.app.game_data.texture(texture_name) is not None:
                icon = (rect[0] + 4, rect[1] + 6, rect[2] - 4, rect[1] + 36)
                items.append(ui.sprite_item(texture_name, icon))
            items.append(ui.text_item('%d' % count, (rect[0] + 2, rect[3] - 16, rect[2] - 2, rect[3] - 2),
                                      ui.FONT_SMALL, QUICK_TEXT_COLOR, align='right'))
        self.app.interface.set_layer(QUICK_LAYER, items)

    def quick_slot_at(self, x, y):
        if self.camp_hides_hud or self.briefing is not None:
            return None
        for index, rect in enumerate(QUICK_SLOT_RECTS):
            if rect[0] <= x < rect[2] and rect[1] <= y < rect[3]:
                return index
        return None

    def show_quick_tooltip(self, index, x, y):
        """
        The name of the quick item under the mouse (infoitem of game.exe).
        """
        belt = self.quick_belt()
        if not 0 <= index < len(belt):
            self.app.interface.set_layer('tooltip', [])
            return
        quick_item, count = belt[index]
        text = '%s (%d)' % (quick_item.name, count)
        width, height = self.app.interface.text_size(text, ui.FONT_SMALL)
        width += TOOLTIP_PADDING * 2
        height += TOOLTIP_PADDING * 2
        x0 = min(max(x, 0), ui.WIDTH - width)
        y0 = max(0, y - height - TOOLTIP_OFFSET)
        box = (x0, y0, x0 + width, y0 + height)
        inner = (x0 + TOOLTIP_PADDING, y0 + TOOLTIP_PADDING, box[2] - TOOLTIP_PADDING,
                 box[3] - TOOLTIP_PADDING)
        self.app.interface.set_layer('tooltip', [ui.box_item(box, TOOLTIP_SHADE),
                                                 ui.text_item(text, inner, ui.FONT_SMALL, TOOLTIP_COLOR)])

    def update_acks(self):
        """
        10..18 of acks.db: the hero answers a new order. The remake hears the order in its mode: walk or run
        is Move, attack is Attack; the looting is said by BattleScreen.open_loot.
        """
        hero = self.app.unit
        if hero is None:
            self.ack_unit = None
            self.ack_player = None
            return
        if hero is not self.ack_unit:
            self.ack_unit = hero
            self.ack_player = acks_module.player_for(self.app.game_data, hero, getattr(hero, 'rng', None))
            self.ack_mode = None
            self.ack_foe = None
        if self.ack_player is None:
            return
        mode = hero.mode
        foe = hero.foe
        group = None
        if mode != self.ack_mode:
            if mode in (unit_module.MODE_WALK, unit_module.MODE_RUN):
                group = acks_module.GROUP_MOVE
            elif mode == unit_module.MODE_ATTACK:
                group = acks_module.GROUP_ATTACK
        elif mode == unit_module.MODE_ATTACK and foe is not None and foe is not self.ack_foe:
            group = acks_module.GROUP_ATTACK
        self.ack_mode = mode
        self.ack_foe = foe
        if group is not None:
            self.say_ack(group)

    def say_unit_ack(self, unit, group):
        """
        A phrase of a group of acks.db at the place of a unit of the party (its own prototype and voice).
        """
        if unit is None or group is None:
            return None
        player = acks_module.player_for(self.app.game_data, unit, getattr(unit, 'rng', None))
        if player is None:
            return None
        phrase = player.choose(group)
        if phrase is None:
            return None
        sound = acks_module.PhraseSound(phrase, (unit.x, unit.y, unit.z))
        audio = getattr(self.app, 'audio', None)
        if audio is not None:
            audio.play_sound(sound, key=('ack', id(unit), group))
        if phrase.text:
            self.add_message(phrase.text.strip())
        return phrase

    def say_ack(self, group):
        """
        A phrase of a group of acks.db at the place of the hero.
        """
        hero = self.app.unit
        if hero is None or self.ack_player is None:
            return None
        phrase = self.ack_player.choose(group)
        if phrase is None:
            return None
        sound = acks_module.PhraseSound(phrase, (hero.x, hero.y, hero.z))
        audio = getattr(self.app, 'audio', None)
        if audio is not None:
            audio.play_sound(sound, key=('ack', id(hero), group))
        if phrase.text:
            self.add_message(phrase.text.strip())
        return phrase

    def add_message(self, text):
        self.messages.append(text)
        self.messages = self.messages[-20:]
        if self.text_mode == 'messages':
            self.update_text()

    def add_item_message(self, text):
        """
        10202: the collected things in the text window.
        """
        self.item_messages.append(text)
        self.item_messages = self.item_messages[-20:]
        if self.text_mode == 'items':
            self.update_text()

    def current_speed(self):
        if self.app.paused:
            return SPEED_PAUSE
        return SPEED_FAST if self.app.clock.tick_ms != timing.TICK_MS else SPEED_NORMAL

    def update(self, dt_ms):
        self.time_s += dt_ms / 1000.0
        if self.briefing is not None:
            self.briefing.update(dt_ms)
            self.clear_battle_hud()
            self.app.scene.set_hud_layer('inventory', [])
            self.app.interface.set_layer(QUICK_LAYER, [])
            self.quick_key = None
            return
        self.update_ground_marks()
        self.update_figure_scales()
        self.footprints.update(dt_ms)
        self.zone_exit.update(dt_ms)
        self.update_quick_items()
        self.update_acks()
        if getattr(self.app, 'show_object_names', False):
            self.update_object_names()
        if self.questmap is not None and not self.questmap.is_open:
            self.questmap = None
            self.app.paused = self.modal_paused
        if self.globalmap is not None and not self.globalmap.is_open:
            self.globalmap = None
            self.app.paused = self.globalmap_paused
        if self.loot is not None and not self.loot.is_open:
            self.loot = None
            self.app.paused = self.modal_paused
        if self.talk is not None and not self.talk.is_open:
            self.talk = None
            self.app.paused = self.talk_paused
        if self.camp is not None:
            self.camp.update(dt_ms)
        if self.questmap is not None:
            self.questmap.update()
        if self.globalmap is not None:
            self.globalmap.update()
        if self.briefing is not None:
            self.briefing.update(dt_ms)
        if self.loot is not None:
            self.loot.update()
        if self.talk is not None:
            self.talk.update()
        speed = self.current_speed()
        if speed != self.speed_index or speed == SPEED_PAUSE or self.app.hour != self.hour:
            self.speed_index = speed
            self.hour = self.app.hour
            if not self.camp_hides_hud:
                self.app.scene.set_hud_layer('pause',
                                             [] if self.inventory is not None else self.pause_panel().elements)
        state = self.hero_state()
        if state != self.shown_state:
            if self.shown_state is not None:
                self.update_static()
            self.shown_state = state
        enchantments = self.enchantments_key()
        if enchantments != self.shown_enchantments:
            self.shown_enchantments = enchantments
            if self.info_mode == hud.INFO_SPELL_EFFECTS:
                self.update_static()
        if self.info_figure is not None and not self.camp_hides_hud:
            self.info_figure.update(dt_ms, self.app.clock)
        self.update_minimap(dt_ms)
        self.update_inventory(dt_ms)
        if not self.camp_hides_hud and self.inventory is None and self.portrait_count:
            self.update_portrait_bars()
        delay = self.tooltip_delay_ms()
        if self.hovered is not None and self.hovered.tip is not None and delay > 0.0:
            before = self.hover_ms
            self.hover_ms += dt_ms
            if before < delay <= self.hover_ms:
                self.show_tooltip()

    def tooltip_delay_ms(self):
        """
        65e540 (partly): the time the mouse stands over a button before its tooltip (the fade of game.exe is not
        here); 0 switches the tooltips off.
        """
        settings = getattr(self.app, 'settings', None)
        value = settings.values.get('tooltip_time', options.TOOLTIP_TIME) if settings is not None else \
            options.TOOLTIP_TIME
        return float(value) * TOOLTIP_STEP_MS

    def show_tooltip(self):
        text = self.app.game_data.text(hud.TIP_PREFIX % self.hovered.tip)
        if not text:
            return
        text = '\n'.join(line.rstrip() for line in text.replace('\r', '').strip().split('\n'))
        interface = self.app.interface
        width, height = interface.text_size(text, ui.FONT_SMALL)
        # the label wraps the lines when its box is even a little narrower than the text
        width += TOOLTIP_PADDING
        x, y = self.mouse
        x0 = min(max(x, TOOLTIP_PADDING), ui.WIDTH - width - TOOLTIP_PADDING * 2)
        y0 = y + TOOLTIP_OFFSET
        if y0 + height + TOOLTIP_PADDING * 2 > ui.HEIGHT:
            y0 = y - TOOLTIP_OFFSET - height - TOOLTIP_PADDING * 2
        box = (x0, y0, x0 + width + TOOLTIP_PADDING * 2, y0 + height + TOOLTIP_PADDING * 2)
        inner = (box[0] + TOOLTIP_PADDING, box[1] + TOOLTIP_PADDING, box[2] - TOOLTIP_PADDING, box[3] - TOOLTIP_PADDING)
        interface.set_layer('tooltip', [ui.box_item(box, TOOLTIP_SHADE),
                                         ui.text_item(text, inner, ui.FONT_SMALL, TOOLTIP_COLOR)])

    def on_mouse(self, x, y):
        """
        True when the mouse is over the battle screen and not over the world. The camp takes the mouse
        above its strip of items.
        """
        if self.briefing is not None:
            self.over_interface = True
            return True
        self.mouse = (x, y)
        self.over_interface = False
        unit = self.info_unit()
        if unit is not getattr(self, 'info_target', None):
            self.info_target = unit
            if not self.camp_hides_hud:
                self.update_static()
                self.update_info_figure()
        if self.questmap is not None:
            self.over_interface = True
            return True
        if self.globalmap is not None:
            self.over_interface = True
            return True
        if self.loot is not None:
            self.over_interface = True
            return True
        if self.talk is not None:
            self.over_interface = True
            return True
        if self.camp is not None and not self.on_strip(x, y):
            self.hovered = None
            self.hover_ms = 0.0
            self.app.interface.set_layer('tooltip', [])
            self.over_interface = True
            return self.camp.on_mouse(x, y)
        if self.camp is None:
            index = self.quick_slot_at(x, y)
            if index is not None:
                self.show_quick_tooltip(index, x, y)
                self.over_interface = True
                return True
        region = hud.region_at(self.panels(), x, y)
        if region is not self.hovered and (region is None or self.hovered is None or region.points != self.hovered.points):
            self.hovered = region
            self.hover_ms = 0.0
            self.app.interface.set_layer('tooltip', [])
        self.over_interface = region is not None
        return region is not None

    def on_strip(self, x, y):
        """
        The strip of items keeps working in the camp: it is the bottom 100 pixels of the screen.
        """
        return self.inventory is not None and y >= inventory.LOWER_TOP

    def regions(self):
        return [r.points for p in self.panels() for r in p.regions]

    def on_click(self, x, y):
        if self.briefing is not None:
            return self.briefing.on_click(x, y)
        if self.questmap is not None:
            return self.questmap.on_click(x, y)
        if self.globalmap is not None:
            return self.globalmap.on_click(x, y)
        if self.loot is not None:
            return self.loot.on_click(x, y)
        if self.talk is not None:
            return self.talk.on_click(x, y)
        if self.camp is None:
            index = self.quick_slot_at(x, y)
            if index is not None:
                keys = getattr(self.app, 'keys', None)
                if keys is not None:
                    keys.use_quick_item('item%d' % (index + 1))
                    self.quick_key = None
                    self.update_quick_items()
                return True
        if self.camp is not None and not self.on_strip(x, y):
            self.app.interface.set_layer('tooltip', [])
            return self.camp.on_click(x, y)
        if self.camp is not None and self.camp.on_strip_click(x, y, self.inventory):
            return True
        region = hud.region_at(self.panels(), x, y)
        if region is None:
            return False
        self.app.interface.set_layer('tooltip', [])
        self.hover_ms = float('inf')
        action = region.action
        app = self.app
        if action in ('info', 'text', 'minimap'):
            self.hidden ^= {action}
            self.update_static()
        elif action in ('text messages', 'text items'):
            self.text_mode = 'items' if action == 'text items' else 'messages'
            self.hidden.discard('text')
            self.update_text()
        elif action == hud.TIP_PAUSE:
            app.paused = True
        elif action == hud.TIP_NORMAL_SPEED:
            app.paused = False
            app.clock.set_fast(False)
        elif action == hud.TIP_FAST_SPEED:
            app.paused = False
            app.clock.set_fast(True)
        elif action == 'default camera':
            app.reset_camera()
        elif action == hud.TIP_QUESTS:
            self.toggle_questmap()
        elif action in hud.INFO_MODE_ACTIONS:
            self.info_mode = hud.INFO_MODE_ACTIONS.index(action)
            self.update_static()
        elif action in ZOOM_ACTIONS:
            self.zoom_direction = ZOOM_ACTIONS[action]
        elif self.inventory is not None and self.inventory.on_click(action, jump=self.app.shift_down()):
            if action.startswith('scroll'):
                app.ui_sound('buttons/Camp/Scroll.wav' if self.camp_hides_hud else 'buttons/Battle/Scroll.wav')
            elif action.startswith('category'):
                app.ui_sound('buttons/Camp/Mode.wav' if self.camp_hides_hud else 'buttons/Battle/Weapon.wav')
        elif action in (hud.TIP_CRAWL, hud.TIP_SNEAK, hud.TIP_WALK, hud.TIP_RUN):
            self.mode_index = {hud.TIP_CRAWL: 0, hud.TIP_SNEAK: 1, hud.TIP_WALK: 2,
                               hud.TIP_RUN: 3}[action]
            app.gait = {0: stats_module.SPEED_CRAWL, 1: stats_module.SPEED_SNEAK,
                        2: stats_module.SPEED_WALK, 3: stats_module.SPEED_RUN}[self.mode_index]
            app.scene.set_hud_layer('mode', hud.mode_dial(self.mode_index).elements)
        elif action == 'portrait':
            pass
        else:
            text = app.game_data.text(hud.TIP_PREFIX % region.tip) if region.tip else None
            name = text.replace('\r', '').split('\n')[0].strip() if text else str(action)
            self.add_message('%s: not made yet' % name)
        if _Debug:
            print('battle screen: %s' % action)
        return True

    def toggle_camp(self):
        """
        The camp screen over the battle: the strip of items at the bottom and the quarters of the party.
        game.exe opens it from its scripts (a campfire), the remake does it by the key K.
        """
        app = self.app
        if self.camp is not None:
            self.camp.close()
            self.camp = None
            app.paused = self.camp_paused
            if self.camp_opened_inventory and self.inventory is not None:
                self.toggle_inventory()
            self.set_camp_hides_hud(False)
            app.interface.set_regions(self.regions())
            return
        self.camp_opened_inventory = self.inventory is None
        app.show_help('', 0)
        self.camp_paused = app.paused
        app.paused = True
        self.camp = camp_view.CampView(app)
        # the things of the party go to the strip, the worn things stand on the paper doll of the camp
        self.camp.open()
        if self.inventory is None:
            self.toggle_inventory(list(getattr(app.party, 'items', [])))
        else:
            self.camp.sync_strip()
        self.set_camp_hides_hud(True)
        app.interface.set_regions(self.camp.regions())

    def show_briefing(self, name, variable=None):
        """
        The event "briefing <name>" and a topic of the talk: the modal window of client.briefing with the
        music of the allod. The end of the briefing completes its game variable (592090).
        """
        if self.briefing is not None:
            return False
        if variable is None and getattr(self.app, 'scripts', None) is not None:
            variable = self.app.scripts.briefing_variable(name)
        view = briefing_view.BriefingView(self.app)
        if not view.show(name):
            return False
        self.modal_paused = self.app.paused
        self.app.paused = True
        self.app.show_help('', 0)
        self.clear_battle_hud()
        for head_view in self.heads:
            self.app.scene.show_interface_prop(head_view, False)
        self.briefing = view
        self.briefing_variable = variable
        self.briefing.on_close = self.on_briefing_closed
        return True

    def on_briefing_closed(self):
        variable = self.briefing_variable
        self.briefing_variable = None
        self.briefing = None
        if variable is not None and getattr(self.app, 'scripts', None) is not None:
            self.app.scripts.complete_briefing(0, variable)
        self.app.paused = self.modal_paused
        self.update_portraits()
        self.update_static()
        self.update_minimap(0)
        self.update_ground_marks()

    def camp_press(self, x, y):
        """
        The mouse went down: client.camp.CampView starts the drag of a thing of the camp.
        """
        return self.camp is not None and self.camp.on_press(x, y)

    def camp_release(self, x, y):
        """
        The mouse went up: the dragged thing goes to its slot or back to the strip.
        """
        return self.camp is not None and self.camp.on_release(x, y)

    def can_talk(self, unit):
        """
        True when the unit is peaceful and has topics (the cursor of the conversation over it).
        """
        if unit is None or self.app.unit is None or self.talk is not None:
            return False
        if self.app.ai.hostile(self.app.unit, unit):
            return False
        return talk_view.has_topics(self.app, unit)

    def open_talk(self, unit):
        """
        The topics of a peaceful unit: the window of client.talk; a chosen topic plays its briefing.
        """
        if self.talk is not None or unit is None:
            return False
        view = talk_view.TalkView(self.app)
        if not view.open(unit):
            return False
        self.talk_paused = self.app.paused
        self.app.paused = True
        self.app.show_help('', 0)
        self.talk = view
        self.talk.on_topic = self.on_talk_topic
        return True

    def on_talk_topic(self, topic):
        self.talk = None
        self.app.paused = self.talk_paused
        self.show_briefing(topic.name, topic.variable)

    def open_loot(self, unit):
        """
        The things of a dead unit: engine.loot builds them (461cb0 marks the body looted at once) and the
        window of client.loot takes them into the party.
        """
        if self.loot is not None or unit is None or getattr(unit, 'looted', False):
            return False
        loot = loot_module.make_loot(self.app.game_data, unit, rng=getattr(unit, 'rng', None))
        self.modal_paused = self.app.paused
        self.app.paused = True
        self.app.show_help('', 0)
        view = loot_view.LootView(self.app)
        view.open(loot)
        self.loot = view
        self.say_ack(acks_module.GROUP_LOOT)
        return True

    def toggle_questmap(self):
        """
        The button of the quests of the disk of the pause (10401): the map of the allod and the journal.
        """
        app = self.app
        if self.questmap is not None:
            self.questmap.close()
            self.questmap = None
            app.paused = self.modal_paused
            return
        self.modal_paused = app.paused
        app.paused = True
        app.show_help('', 0)
        self.questmap = questmap_view.QuestMapView(app)
        self.questmap.open()

    def open_global_map(self, edge_zone, pass_time=0.0):
        """
        The party walked into a road: the map of the allod with the roads out of its zone (client.globalmap).
        """
        app = self.app
        if self.globalmap is None:
            app.autosave_exit()
            self.globalmap_paused = app.paused
            app.paused = True
            app.show_help('', 0)
            self.globalmap = globalmap_view.GlobalMapView(app)
        self.globalmap.open(edge_zone, pass_time)
        return True

    def on_key(self, name):
        """
        The keys of the open windows: the briefing and the map of the allod take all of them, the camp too;
        Escape closes the window or the camp, the key K closes the camp.
        """
        if self.briefing is not None:
            return self.briefing.on_key(name)
        if self.questmap is not None:
            return self.questmap.on_key(name)
        if self.globalmap is not None:
            return self.globalmap.on_key(name)
        if self.loot is not None:
            return self.loot.on_key(name)
        if self.talk is not None:
            return self.talk.on_key(name)
        if self.camp is None:
            return False
        if name == 'k':
            self.toggle_camp()
            return True
        return self.camp.on_key(name) or True

    def toggle_inventory(self, item_names=None):
        """
        The strip with the items of the hero (the items given with --item, worn or not); item_names replaces
        them (the things of the party in the camp).
        """
        app = self.app
        scene = app.scene
        if self.inventory is None:
            names = app.args.item_names if item_names is None else item_names
            items = [equipment.make_item(app.game_data.items(), name) for name in names]
            self.inventory = inventory.InventoryStrip([item for item in items if item is not None])
            for view in self.heads:
                scene.show_interface_prop(view, False)
            scene.set_hud_layer('portraits', [])
        else:
            self.inventory = None
            for view, _, _ in self.item_views.values():
                scene.remove_interface_prop(view)
            self.item_views = {}
            scene.set_hud_layer('inventory', [])
            for view in self.heads:
                scene.show_interface_prop(view, True)
            self.update_portraits()
        self.speed_index = None
        self.update_static()
        app.interface.set_regions(self.regions())

    def update_inventory(self, dt_ms):
        strip = self.inventory
        if strip is None:
            return
        strip.update(dt_ms / 1000.0)
        scene = self.app.scene
        scene.set_hud_layer('inventory', strip.elements())
        shown = strip.shown_items()
        visible = {}
        for item, x, y, depth in strip.slots():
            visible[shown.index(item)] = (item, x, y, depth)
        for index in list(self.item_views):
            if index not in visible:
                scene.remove_interface_prop(self.item_views.pop(index)[0])
        for index, (item, x, y, depth) in visible.items():
            if index not in self.item_views:
                made = self.add_item_model(item, depth)
                if made is None:
                    continue
                self.item_views[index] = made
            view, item_prop, offset = self.item_views[index]
            position = gl.interface_point(x, y, depth)
            item_prop.matrix = prop_module.placement_matrix(position, inventory.item_rotation(), 1.0) @ offset

    def add_item_model(self, item, depth):
        """
        5eca10: the figure of the item with its bounds in the middle of the slot, the larger side 50 pixels.
        """
        game_data = self.app.game_data
        model_name = inventory.item_model_name(item)
        if model_name is None or not game_data.has_model(model_name):
            return None
        model = figure.FigureModel(game_data, model_name)
        points = model.figure(model_name).variant_positions((0.0, 0.0, 0.0))
        low, high = points.min(axis=0), points.max(axis=0)
        scale = inventory.item_scale(high[0] - low[0], high[1] - low[1], depth)
        offset = np.identity(4)
        offset[0:3, 0:3] *= scale
        offset[0:3, 3] = -(low + high) * 0.5 * scale
        bone = game_data.bones(model_name).get(model_name)
        if bone is not None:
            # the figure stands at its own origin in the slot, like the compass
            offset[0:3, 3] -= bone[0] * scale
        item_prop = prop_module.Prop(model)
        texture_name = inventory.item_texture_name(item)
        rgba = game_data.texture(texture_name) if texture_name else None
        texture = self.app.scene.composed_texture('item:%s' % texture_name, inventory.item_texture(rgba))
        view = self.app.scene.add_prop(item_prop, model_name, interface=True, texture=texture)
        return view, item_prop, offset

    def on_release(self):
        self.zoom_direction = 0
        if self.inventory is not None:
            self.inventory.release()

    def clear(self):
        if self.briefing is not None:
            self.briefing.close()
            self.briefing = None
        if self.questmap is not None:
            self.questmap.close()
            self.questmap = None
        if self.loot is not None:
            self.loot.close()
            self.loot = None
        if self.talk is not None:
            self.talk.close()
            self.talk = None
        if self.camp is not None:
            self.camp.close()
            self.camp = None
        for view in self.heads:
            self.app.scene.remove_interface_prop(view)
        self.heads = []
        if self.info_figure_view is not None:
            self.app.scene.remove_interface_prop(self.info_figure_view)
        self.info_figure = None
        self.info_figure_view = None
        if self.compass_view is not None:
            self.app.scene.remove_interface_prop(self.compass_view)
        self.compass_view = None
        self.compass = None
        self.minimap = None
        for view, _, _ in self.item_views.values():
            self.app.scene.remove_interface_prop(view)
        self.item_views = {}
        self.inventory = None
        self.app.scene.clear_hud()
        self.app.interface.set_layer('tooltip', [])
        self.app.interface.set_layer('text window', [])
        self.app.interface.set_layer('info text', [])
