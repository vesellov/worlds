"""
The camp screen of the client: the quarters of the party of engine.camp with the textures campm1..4 and
camp1..4, the information about the chosen member, the windows of the skills, of the trade, of the repair
and of the constructors and the remake's column of their buttons at the right.

The window of the camp goes over the battle screen in the 2D interface layer. The strip of items at the
bottom stays the 3D Strip of client.hud (the camp draws nothing over it). The buttons of the windows do not
exist in game.exe (a shop or a campfire of the scripts opens them); the column is the remake's way to reach
them until the scripts do it.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import numpy

#------------------------------------------------------------------------------

from engine import camp as camp_module
from engine import equipment as equipment_module
from engine import experience as experience_module
from engine import figure
from engine import hud as hud_module
from engine import inventory as inventory_module
from engine import prop as prop_module
from engine import skills as skills_module
from client import gl
from client import ui

#------------------------------------------------------------------------------

LAYER_NAME = 'camp'
# the black card behind the figure of the left panel (the hud is drawn under the 2D layer of the camp,
# its depth stays behind the figure, so it covers the world and not the hero)
BACKDROP_LAYER = 'camp backdrop'
BACKDROP_DEPTH = 30.0

TEXT_COLOR = (228, 215, 167)
TITLE_COLOR = (255, 179, 49)
SELECTED_COLOR = (255, 215, 120)
DIM_COLOR = (120, 115, 90)

# the remake's column of the windows at the right of the camp
TAB_X = (605, 795)
TAB_TOP = 110
TAB_HEIGHT = 28
TAB_STEP = 32
TAB_LABELS = (
    (camp_module.CAMP_WINDOW, None),
    (camp_module.INFO_WINDOW, None),
    (camp_module.SKILLS_WINDOW, 'string skills'),
    (camp_module.TRADE_WINDOW, 'string mode_itemtrade'),
    (camp_module.REPAIR_WINDOW, 'string mode_itemrepair'),
    (camp_module.ITEM_CONSTRUCTOR_WINDOW, 'string mode_itemconstr'),
    (camp_module.SPELL_CONSTRUCTOR_WINDOW, 'string mode_spellconstr'),
)

TAB_TEXTS = ('Camp', 'Information', 'Skills', 'Item shop', 'Items repair', 'Item constructor',
             'Spell constructor')

# the texts of the remake for the windows whose data is not read yet
NOT_MADE = 'not made yet'

TITLE_BOX = (210, 108, 590, 132)
INFO_TITLE_BOX = (6, 104, 194, 128)
CLOSE_LABEL = 'X'

# the whole figure of the hero in the free left panel of the camp (the remake stands it in the interface)
FIGURE_POINT = (100.0, 374.0)
FIGURE_DEPTH = 8.0
FIGURE_SCALE = 1.2
SLOT_TEXT_COLOR = (255, 236, 190)
PICK_COLOR = (255, 220, 120)

# the drag of a thing with the mouse: the distance which makes a click a drag
DRAG_THRESHOLD = 4.0
# the tooltip of a slot of the paper doll
TOOLTIP_LAYER = 'camp tooltip'
TOOLTIP_BOX = (0, 0, 190, 34)

# the sounds of the camp (buttons/Camp/... of sfx.res, 5f7e60)
SOUND_MODE = 'buttons/Camp/Mode.wav'
SOUND_SLOT = 'buttons/Camp/Slot.wav'
SOUND_CHANGE = 'buttons/Camp/Change.wav'
SOUND_PUT_ON = 'buttons/Camp/Put_on.wav'
SOUND_PUT_OFF = 'buttons/Camp/Put_off.wav'
SOUND_BUY = 'buttons/Camp/Buy.wav'
SOUND_PERK = 'buttons/Camp/Perk.wav'

# the rows of the skills and of the abilities in the skills window (624ed0, 6268e0)
SKILL_ROW_HEIGHT = 16
SKILL_LIST_TOP = 166
SKILL_VALUE_X = 372
SKILL_PRICE_X = 400
SKILL_PRICE_RIGHT = 590
ABILITY_ROW_HEIGHT = 15
KNOWN_LIST_TOP = 340
AVAILABLE_LIST_TOP = 410
MAX_KNOWN_ROWS = 3
MAX_AVAILABLE_ROWS = 3

#------------------------------------------------------------------------------


def tab_rect(index):
    top = TAB_TOP + index * TAB_STEP
    return (TAB_X[0], top, TAB_X[1], top + TAB_HEIGHT)

#------------------------------------------------------------------------------


class CampView(object):
    """
    The camp over the battle screen: the quarters of up to four members, the information and the windows.
    """

    def __init__(self, app):
        self.app = app
        self.camp = None
        self.hovered_tab = None
        self.shown_key = None
        self.figure = None
        self.figure_view = None
        # (name, slot) of the thing which the mouse drags, None when nothing is dragged
        self.dragging = None
        self.drag_source = None
        self.press_point = None
        self.mouse = (0.0, 0.0)
        self.tooltip_text = None
        # (rect, kind, value) of the rows of the skills window
        self.skill_hits = []

    @property
    def is_open(self):
        return self.camp is not None

    def __repr__(self):
        return 'CampView(%s)' % (self.camp.window if self.camp is not None else 'closed')

    def open(self):
        """
        The party of the roster, the things of the party in the strip and the worn things on the paper doll;
        the spells of the demo when they are known.
        """
        app = self.app
        names = []
        if app.roster is not None:
            names = [record.name for record in app.roster.current().units]
        self.unpack_worn_items()
        units = app.zone_party()
        spells = list(app.demo.spells) if getattr(app, 'demo', None) is not None else []
        if getattr(app, 'spell', None) is not None and app.spell not in spells:
            spells.insert(0, app.spell)
        repair_items = [item.name for item in camp_module.equipped_items(app.unit).values()]
        self.camp = camp_module.Camp(units, list(getattr(app.party, 'items', [])), spells, names,
                                     party=getattr(app, 'party', None), repair_items=repair_items)
        self.shown_key = None
        self.add_backdrop()
        self.add_figure()
        self.update()

    def unpack_worn_items(self):
        """
        The things of --item which the hero does not wear go to the party (the strip shows only them), the
        worn things stay the items of the hero.
        """
        app = self.app
        if app.unit is None:
            return
        worn = [item.name for item in camp_module.equipped_items(app.unit).values()]
        for name in list(getattr(app, 'hero_item_names', []) or []):
            if name not in worn and name not in app.party.items:
                app.party.items.append(name)
        app.hero_item_names = worn

    def close(self):
        self.remove_figure()
        self.remove_backdrop()
        self.camp = None
        self.hovered_tab = None
        self.shown_key = None
        self.dragging = None
        self.drag_source = None
        self.press_point = None
        self.app.interface.set_layer(LAYER_NAME, [])
        self.clear_tooltip()

    def add_backdrop(self):
        """
        The world under the left panel goes black: a black card of the hud in front of the world and behind
        the figure of the hero (the depth of the figure wins).
        """
        self.app.scene.set_hud_layer(BACKDROP_LAYER,
                                     [hud_module.quad(BACKDROP_DEPTH, camp_module.INFO_RECT, None,
                                                      color=(0.0, 0.0, 0.0, 1.0))])

    def remove_backdrop(self):
        self.app.scene.set_hud_layer(BACKDROP_LAYER, [])

    def add_figure(self):
        """
        6456a0 of the camp: the whole figure of the hero stands in the free left panel of the camp. game.exe
        draws it into a render target of the region; the remake puts the model into the 3D interface.
        """
        app = self.app
        unit = app.unit
        if unit is None or getattr(app, 'scene', None) is None:
            return
        game_data = app.game_data
        model_name = unit.figure.model_name
        if not game_data.has_model(model_name):
            return
        model = figure.FigureModel(game_data, model_name, coefs=unit.figure.coefs,
                                   equipment=unit.figure.equipment)
        hero = prop_module.Prop(model)
        position = gl.interface_point(FIGURE_POINT[0], FIGURE_POINT[1], FIGURE_DEPTH)
        hero.matrix = prop_module.placement_matrix(position, hud_module.INFO_FIGURE_ROTATION, FIGURE_SCALE)
        hero.idle()
        texture_name = unit.figure.default_texture_name()
        if texture_name is None:
            return
        textures = {group: app.scene.composed_texture('camp figure:%s:%s' % (model_name, key), rgba)
                    for group, (key, rgba) in model.composed_textures().items()}
        self.figure = hero
        self.figure_view = app.scene.add_prop(hero, texture_name, interface=True, textures=textures)

    def remove_figure(self):
        if self.figure_view is not None and getattr(self.app, 'scene', None) is not None:
            self.app.scene.remove_interface_prop(self.figure_view)
        self.figure = None
        self.figure_view = None
        # (name, slot) of the thing which the mouse drags, None when nothing is dragged
        self.dragging = None
        self.drag_source = None
        self.press_point = None
        self.mouse = (0.0, 0.0)

    def refresh_figure(self):
        self.remove_figure()
        self.add_figure()

    def text(self, key, default):
        return (self.app.game_data.text(key) or default).replace('\r', '').strip()

    def update(self, dt_ms=0):
        """
        The layer is built again only when the state of the camp changes (the texts are expensive); the figure
        of the hero is animated every frame.
        """
        if self.camp is None:
            return
        if self.figure is not None:
            self.figure.update(dt_ms, self.app.clock)
        camp = self.camp
        key = (camp.window, camp.selected, camp.member_count(), tuple(camp.cell_items()),
               tuple(spell.name for spell in camp.spells), camp.picked, camp.money,
               tuple((slot, item.name) for slot, item in camp_module.equipped_items(self.camp.unit).items()),
               tuple(getattr(self.app.party, 'items', [])), self.dragging, self.drag_source,
               (round(self.mouse[0]), round(self.mouse[1])) if self.dragging else None)
        if key == self.shown_key:
            return
        self.shown_key = key
        self.app.interface.set_layer(LAYER_NAME, self.elements())

    def elements(self):
        # the left panel is a hole: the whole figure of the hero stands there in the 3D interface
        info = camp_module.INFO_RECT
        items = [ui.box_item((0, 0, 800, info[1]), (0.0, 0.0, 0.0, 1.0)),
                 ui.box_item((0, info[3], info[2], 500), (0.0, 0.0, 0.0, 1.0)),
                 ui.box_item((info[2], info[1], 800, 500), (0.0, 0.0, 0.0, 1.0))]
        items.extend(self.member_items())
        items.extend(self.tab_items())
        items.extend(self.window_items())
        items.extend(self.picked_items())
        items.extend(self.drag_items())
        return items

    def member_items(self):
        """
        The four dolls of the camp: the backgrounds and the frames are there for empty quarters too, the
        paper doll of the things hangs in the quarter of its member.
        """
        camp = self.camp
        items = []
        for index, rect in enumerate(camp_module.MEMBER_RECTS):
            items.append(ui.sprite_item(camp_module.MEMBER_BACKGROUNDS[index], rect,
                                        camp_module.MEMBER_TEXELS))
            items.append(ui.sprite_item(camp_module.MEMBER_TEXTURES[index], rect,
                                        camp_module.MEMBER_TEXELS))
            member = index < camp.member_count()
            color = SELECTED_COLOR if member and index == camp.selected else DIM_COLOR
            items.extend(ui.panel_items(rect, 2, color, color, background=(0.0, 0.0, 0.0, 0.0))[1:])
            name = camp.member_name(index) if member else ''
            if name:
                items.append(ui.text_item(name, (rect[0] + 6, rect[1] + 4, rect[2] - 6, rect[1] + 24),
                                          ui.FONT_MEDIUM, TEXT_COLOR))
            if self.member_available(index):
                items.extend(self.slot_items(index))
        return items

    def member_unit(self, index):
        camp = self.camp
        if 0 <= index < len(camp.units):
            return camp.units[index]
        return None

    def member_record(self, index):
        """
        engine.roster.UnitRecord of the member; the hero is the first record too.
        """
        roster = getattr(self.app, 'roster', None)
        records = getattr(getattr(roster, 'current', None), 'units', None) or []
        if 0 <= index < len(records):
            return records[index]
        return None

    def member_stats(self, index):
        """
        engine.experience.HeroStats of a member: of its unit in the zone or of its record (the main flow
        makes the units and the stats of the other members).
        """
        unit = self.member_unit(index)
        stats = getattr(unit, 'hero_stats', None) if unit is not None else None
        if stats is None:
            stats = getattr(self.member_record(index), 'hero_stats', None)
        return skills_module.attach(stats)

    def selected_stats(self):
        return self.member_stats(self.camp.selected)

    def record_equipped(self, index):
        """
        slot -> Item of the things of a record (the main flow will make their units and equip_record).
        """
        record = self.member_record(index)
        result = {}
        if record is None:
            return result
        for name in list(getattr(record, 'item_names', []) or []):
            item = equipment_module.make_item(self.app.game_data.items(), name)
            slot = camp_module.slot_of(item)
            if slot is not None:
                result[slot] = item
        return result

    def member_equipped(self, quarter):
        unit = self.member_unit(quarter)
        if unit is not None:
            return camp_module.equipped_items(unit)
        return self.record_equipped(quarter)

    def member_available(self, quarter):
        return self.member_unit(quarter) is not None or self.member_record(quarter) is not None

    def slot_items(self, quarter):
        """
        The paper doll of a member: eight squares of the slots (SLOT_ORDER) with the icon of the worn
        thing (the textures of the 3D models of the strip, 5ec020); the slot under the mouse has a tooltip.
        """
        equipped = self.member_equipped(quarter)
        items = []
        for slot, rect in camp_module.slot_rects(quarter):
            item = equipped.get(slot)
            occupied = item is not None
            color = PICK_COLOR if occupied else DIM_COLOR
            items.append(ui.box_item(rect, (0.10, 0.09, 0.07, 0.85)))
            items.extend(ui.panel_items(rect, 1, color, color, background=(0.0, 0.0, 0.0, 0.0)))
            if occupied:
                items.extend(self.icon_items(rect, item))
        return items

    def icon_items(self, rect, item):
        """
        The texture of an item fitted into the slot; the transparent edges of the texture are cut by
        its alpha so the thing takes the most of the square.
        """
        texture_name = inventory_module.item_texture_name(item)
        if texture_name is None:
            return []
        texture = self.app.game_data.texture(texture_name)
        if texture is None:
            return []
        alpha = texture[:, :, 3]
        rows, columns = numpy.nonzero(alpha)
        if len(rows) == 0:
            return []
        top, bottom = int(rows.min()), int(rows.max()) + 1
        left, right = int(columns.min()), int(columns.max()) + 1
        width, height = texture.shape[1], texture.shape[0]
        inner = min(rect[2] - rect[0], rect[3] - rect[1]) - 5
        scale = min(inner / float(right - left), inner / float(bottom - top))
        draw_width = (right - left) * scale
        draw_height = (bottom - top) * scale
        center_x = (rect[0] + rect[2]) * 0.5
        center_y = (rect[1] + rect[3]) * 0.5
        box = (center_x - draw_width * 0.5, center_y - draw_height * 0.5,
               center_x + draw_width * 0.5, center_y + draw_height * 0.5)
        return [ui.sprite_item(texture_name, box, (left, top, right, bottom))]

    def picked_items(self):
        """
        The strip of the camp: the chosen thing is outlined and waits for a slot of the paper doll.
        """
        if self.camp is None or self.camp.picked is None:
            return []
        strip = getattr(self.app.battle, 'inventory', None)
        if strip is None:
            return []
        items = []
        for item, x, y, depth in strip.slots():
            if item.name == self.camp.picked:
                rect = (x - 50, y - 50, x + 50, y + 50)
                items.extend(ui.panel_items(rect, 2, PICK_COLOR, PICK_COLOR,
                                            background=(0.0, 0.0, 0.0, 0.0))[1:])
        return items

    def item_by_name(self, name):
        """
        The Item of the party with the name (a fresh one from items.idb when the strip has none).
        """
        strip = getattr(getattr(self.app, 'battle', None), 'inventory', None)
        if strip is not None:
            for item in strip.items:
                if item.name == name:
                    return item
        return equipment_module.make_item(self.app.game_data.items(), name)

    def strip_item_at(self, x, y, strip):
        """
        The thing of the strip under the point, None when there is none.
        """
        if strip is None or not strip.top <= y < strip.top + inventory_module.STRIP_HEIGHT:
            return None
        for item, sx, sy, depth in strip.slots():
            if abs(x - sx) <= inventory_module.SLOT_SIZE * 0.5:
                if item.name in self.app.party.items:
                    return item
        return None

    def strip(self):
        return getattr(getattr(self.app, 'battle', None), 'inventory', None)

    def on_strip_click(self, x, y, strip):
        """
        A thing of the strip is picked for the paper doll (a second click releases it); the arrows and the
        buttons of the strip stay to client.hud when False.
        """
        if self.camp is None:
            return False
        item = self.strip_item_at(x, y, strip)
        if item is None:
            return False
        self.play(SOUND_SLOT)
        self.camp.picked = None if self.camp.picked == item.name else item.name
        self.update()
        return True

    def on_press(self, x, y):
        """
        The mouse goes down on a thing of the strip or on a slot of the doll: it starts to drag it.
        """
        if self.camp is None or self.camp.window != camp_module.CAMP_WINDOW:
            return False
        item = self.strip_item_at(x, y, self.strip())
        if item is not None:
            self.press_point = (x, y)
            self.dragging = item.name
            self.drag_source = ('strip', None, None)
            self.mouse = (x, y)
            self.play(SOUND_SLOT)
            return True
        hit = camp_module.slot_at(x, y, self.camp.member_count())
        if hit is None:
            return False
        quarter, slot = hit
        if not self.member_available(quarter):
            return False
        item = self.member_equipped(quarter).get(slot)
        if item is None:
            return False
        self.press_point = (x, y)
        self.dragging = item.name
        self.drag_source = ('slot', slot, quarter)
        self.mouse = (x, y)
        return True

    def on_release(self, x, y):
        """
        The mouse goes up: a short press is a click, a moved thing goes to the slot under the mouse or,
        from the doll to the strip, back to the party.
        """
        if self.dragging is None:
            return False
        source, source_slot, source_quarter = self.drag_source
        moved = self.press_point is not None and (
            abs(x - self.press_point[0]) + abs(y - self.press_point[1]) >= DRAG_THRESHOLD)
        name = self.dragging
        self.dragging = None
        self.drag_source = None
        self.press_point = None
        self.clear_tooltip()
        if not moved:
            if source == 'strip':
                self.play(SOUND_SLOT)
                self.camp.picked = None if self.camp.picked == name else name
            else:
                self.unequip(source_slot, source_quarter)
            self.update()
            return True
        if source == 'strip':
            hit = camp_module.slot_at(x, y, self.camp.member_count())
            if hit is not None:
                quarter, slot = hit
                if self.member_available(quarter):
                    self.equip(name, slot, quarter)
            self.update()
            return True
        if y >= inventory_module.LOWER_TOP:
            self.unequip(source_slot, source_quarter)
        self.update()
        return True

    def drag_items(self):
        """
        The icon of the dragged thing under the mouse.
        """
        if self.dragging is None:
            return []
        item = self.item_by_name(self.dragging)
        if item is None:
            return []
        x, y = self.mouse
        rect = (x - camp_module.SLOT_SIZE * 0.5, y - camp_module.SLOT_SIZE * 0.5,
                x + camp_module.SLOT_SIZE * 0.5, y + camp_module.SLOT_SIZE * 0.5)
        items = [ui.box_item(rect, (0.0, 0.0, 0.0, 0.45))]
        items.extend(self.icon_items(rect, item))
        return items

    def clear_tooltip(self):
        self.tooltip_text = None
        self.app.interface.set_layer(TOOLTIP_LAYER, [])

    def on_slot_tooltip(self, x, y):
        """
        The name of the slot and of its thing under the mouse (the icons alone do not say it).
        """
        if self.camp is None or self.camp.window != camp_module.CAMP_WINDOW:
            self.clear_tooltip()
            return
        hit = camp_module.slot_at(x, y, self.camp.member_count())
        if hit is None:
            self.clear_tooltip()
            return
        quarter, slot = hit
        if not self.member_available(quarter):
            self.clear_tooltip()
            return
        item = self.member_equipped(quarter).get(slot)
        text = '%s: %s' % (camp_module.SLOT_LABELS[slot], item.name if item is not None else '-')
        if text == self.tooltip_text:
            return
        self.tooltip_text = text
        box = (x + 12, y + 10, x + 12 + TOOLTIP_BOX[2], y + 10 + TOOLTIP_BOX[3])
        self.app.interface.set_layer(TOOLTIP_LAYER, [
            ui.box_item(box, (0.0, 0.0, 0.0, 0.75)),
            ui.text_item(text, (box[0] + 4, box[1] + 4, box[2] - 2, box[3]), ui.FONT_SMALL, TEXT_COLOR)])

    def on_slot_click(self, quarter, slot):
        """
        A slot of the paper doll: the picked thing goes on, a full slot gives its thing back to the party.
        """
        unit = self.member_unit(quarter)
        if unit is None:
            return False
        equipped = camp_module.equipped_items(unit)
        if self.camp.picked is not None:
            return self.equip(self.camp.picked, slot, quarter)
        if slot in equipped:
            return self.unequip(slot, quarter)
        return False

    def equip(self, name, slot, quarter=0):
        """
        62d0c0 with 1 (put on): the thing leaves the strip and takes the slot; the thing of the slot goes back
        to the party. The hero is built again with the worn things (client.main.replace_hero).
        """
        unit = self.member_unit(quarter)
        item = self.item_by_name(name)
        if item is None or camp_module.slot_of(item) != slot:
            return False
        if name not in self.app.party.items:
            return False
        if unit is None:
            return self.equip_record(name, slot, quarter)
        record = getattr(unit, 'record', None) or self.member_record(quarter)
        worn = [item_name for item_name in (getattr(record, 'item_names', None) or []) if item_name != name]
        equipped = camp_module.equipped_items(unit)
        old = equipped.get(slot)
        if old is not None:
            worn = [item_name for item_name in worn if item_name != old.name]
            if old.name not in self.app.party.items:
                self.app.party.items.append(old.name)
        worn.append(name)
        self.app.party.items.remove(name)
        self.play(SOUND_PUT_ON)
        return self.apply_worn_items(unit, worn)

    def unequip(self, slot, quarter=0):
        """
        62d0c0 with 5 (put off): the thing of the slot goes back to the party.
        """
        unit = self.member_unit(quarter)
        if unit is None:
            return self.unequip_record(slot, quarter)
        item = camp_module.equipped_items(unit).get(slot)
        if item is None:
            return False
        record = getattr(unit, 'record', None) or self.member_record(quarter)
        worn = [name for name in (getattr(record, 'item_names', None) or []) if name != item.name]
        if item.name not in self.app.party.items:
            self.app.party.items.append(item.name)
        self.play(SOUND_PUT_OFF)
        return self.apply_worn_items(unit, worn)

    def equip_record(self, name, slot, quarter):
        """
        The thing of the party to the record of a member without a unit (the main flow will make their
        units and equip_record); the thing of the slot goes back to the party.
        """
        record = self.member_record(quarter)
        if record is None:
            return False
        names = list(getattr(record, 'item_names', []) or [])
        old = self.record_slot_name(record, slot)
        if old is not None:
            names = [item_name for item_name in names if item_name != old]
            if old not in self.app.party.items:
                self.app.party.items.append(old)
        names.append(name)
        record.item_names = names
        self.app.party.items.remove(name)
        self.play(SOUND_PUT_ON)
        self.after_record_change()
        return True

    def unequip_record(self, slot, quarter):
        record = self.member_record(quarter)
        item = self.record_equipped(quarter).get(slot)
        if record is None or item is None:
            return False
        record.item_names = [name for name in (record.item_names or []) if name != item.name]
        if item.name not in self.app.party.items:
            self.app.party.items.append(item.name)
        self.play(SOUND_PUT_OFF)
        self.after_record_change()
        return True

    def record_slot_name(self, record, slot):
        for name in list(getattr(record, 'item_names', []) or []):
            item = equipment_module.make_item(self.app.game_data.items(), name)
            if item is not None and camp_module.slot_of(item) == slot:
                return name
        return None

    def after_record_change(self):
        if self.camp is not None:
            self.camp.items = list(self.app.party.items)
            self.camp.trade_items = list(self.app.party.items)
            self.camp.picked = None
        self.sync_strip()
        self.shown_key = None
        self.update()

    def apply_worn_items(self, unit, worn):
        """
        The member is built again with the worn things (the hero with replace_hero, another unit with
        replace_companion), its record takes the names; the strip takes the things of the party.
        """
        app = self.app
        record = getattr(unit, 'record', None)
        if record is not None:
            record.item_names = list(worn)
        if unit is app.unit:
            app.hero_item_names = list(worn)
            app.replace_hero(list(worn))
        elif unit in app.companions:
            new = app.replace_companion(unit, record)
            if self.camp is not None:
                self.camp.units = [new if member is unit else member for member in self.camp.units]
        else:
            return False
        if self.camp is not None:
            self.camp.units[0] = app.unit
            self.camp.items = list(app.party.items)
            self.camp.trade_items = list(app.party.items)
            self.camp.repair_items = [item.name for item in camp_module.equipped_items(app.unit).values()]
            self.camp.picked = None
        # the prototype gives a weapon back when no weapon is named, so the worn things are read again
        app.hero_item_names = camp_module.equipped_names(app.unit)
        self.sync_strip()
        self.refresh_figure()
        battle = getattr(app, 'battle', None)
        if battle is not None:
            battle.update_portraits()
            battle.update_static()
        self.shown_key = None
        self.update()
        return True

    def sync_strip(self):
        """
        The strip shows the things of the party (the worn things are on the paper doll, not in the strip).
        """
        strip = getattr(getattr(self.app, 'battle', None), 'inventory', None)
        if strip is None:
            return
        items = [equipment_module.make_item(self.app.game_data.items(), name) for name in self.app.party.items]
        strip.items = [item for item in items if item is not None]

    def tab_items(self):
        items = []
        for index, (window, key) in enumerate(TAB_LABELS):
            rect = tab_rect(index)
            selected = self.camp.window == window
            color = SELECTED_COLOR if selected else DIM_COLOR
            items.extend(ui.panel_items(rect, 1, color, color, background=(0.15, 0.14, 0.10, 1.0)))
            text = self.text(key, TAB_TEXTS[index]) if key else TAB_TEXTS[index]
            items.append(ui.text_item(text, (rect[0] + 4, rect[1] + 4, rect[2] - 4, rect[3] - 2),
                                      ui.FONT_SMALL, TEXT_COLOR))
        return items

    def window_items(self):
        camp = self.camp
        if camp.window == camp_module.CAMP_WINDOW:
            return []
        if camp.window == camp_module.INFO_WINDOW:
            return self.info_items()
        if camp.window == camp_module.SKILLS_WINDOW:
            return self.skills_items()
        return self.grid_items()

    def info_items(self):
        rect = camp_module.INFO_RECT
        items = ui.panel_items(rect, 2, (90, 90, 66), (132, 130, 90), background=(0.08, 0.08, 0.07, 1.0))
        items.append(ui.text_item('Information', INFO_TITLE_BOX, ui.FONT_MEDIUM, TITLE_COLOR))
        labels = {key: self.text(key, key) for key in camp_module.INFO_TEXT_KEYS}
        lines = self.camp.stat_lines(labels=labels)
        if not lines:
            items.append(ui.text_item(NOT_MADE, (12, 140, 188, 160), ui.FONT_SMALL, DIM_COLOR))
            return items
        for index, (caption, value) in enumerate(lines):
            top = 140 + index * 20
            items.append(ui.text_item(caption, (10, top, 120, top + 18), ui.FONT_SMALL, TEXT_COLOR))
            items.append(ui.text_item(value, (120, top, 195, top + 18), ui.FONT_SMALL, TEXT_COLOR,
                                      align='right'))
        return items

    def skills_items(self):
        """
        624ed0: the rows of the skills with their levels and the price of the next level, the known
        abilities and those which the unit can take now (6268e0, 5260a0).
        """
        items = self.panel_items()
        self.skill_hits = []
        items.append(ui.text_item(self.text('string skills', 'Skills'), TITLE_BOX, ui.FONT_MEDIUM, TITLE_COLOR))
        stats = self.selected_stats()
        if stats is None:
            items.append(ui.text_item(NOT_MADE, (220, 140, 590, 158), ui.FONT_SMALL, DIM_COLOR))
            return items
        items.append(ui.text_item('%s %d' % (self.text('string infoskill_4', 'Experience:').rstrip(':'),
                                             stats.experience), (210, 138, 400, 156), ui.FONT_SMALL, TEXT_COLOR))
        items.append(ui.text_item('%s %d' % (self.text('string infoskill_23', 'Additional points:').rstrip(':'),
                                             skills_module.free_experience(stats)),
                                  (400, 138, 590, 156), ui.FONT_SMALL, TEXT_COLOR, align='right'))
        top = SKILL_LIST_TOP
        for index, level, price in skills_module.skill_lines(stats):
            name = dict(skills_module.SKILL_ROWS).get(index, str(index))
            rect = (210, top, 590, top + SKILL_ROW_HEIGHT)
            items.append(ui.text_item(name, (220, top, SKILL_VALUE_X - 10, top + SKILL_ROW_HEIGHT - 1),
                                      ui.FONT_SMALL, TEXT_COLOR))
            items.append(ui.text_item(str(level), (SKILL_VALUE_X, top, SKILL_VALUE_X + 20,
                                                   top + SKILL_ROW_HEIGHT - 1),
                                      ui.FONT_SMALL, SELECTED_COLOR, align='right'))
            if price is not None:
                items.append(ui.text_item(str(price), (SKILL_PRICE_X, top, SKILL_PRICE_RIGHT,
                                                       top + SKILL_ROW_HEIGHT - 1),
                                          ui.FONT_SMALL, DIM_COLOR, align='right'))
                self.skill_hits.append((rect, 'skill', index))
            top += SKILL_ROW_HEIGHT
        known = skills_module.known_perks(stats)
        items.append(ui.text_item(self.text('string perks', 'Known Abilities'), (210, 322, 590, 338),
                                  ui.FONT_SMALL, TITLE_COLOR))
        for row, perk in enumerate(known[:MAX_KNOWN_ROWS]):
            line_top = KNOWN_LIST_TOP + row * ABILITY_ROW_HEIGHT
            items.append(ui.text_item(perk.name, (220, line_top, 590, line_top + ABILITY_ROW_HEIGHT - 1),
                                      ui.FONT_SMALL, TEXT_COLOR))
        available = skills_module.available_perks(stats)
        items.append(ui.text_item(self.text('string perks_av', 'Available Abilities'), (210, 390, 590, 406),
                                  ui.FONT_SMALL, TITLE_COLOR))
        for row, (perk, price) in enumerate(available[:MAX_AVAILABLE_ROWS]):
            line_top = AVAILABLE_LIST_TOP + row * ABILITY_ROW_HEIGHT
            items.append(ui.text_item(perk.name, (220, line_top, 470, line_top + ABILITY_ROW_HEIGHT - 1),
                                      ui.FONT_SMALL, TEXT_COLOR))
            items.append(ui.text_item(str(price), (470, line_top, 590, line_top + ABILITY_ROW_HEIGHT - 1),
                                      ui.FONT_SMALL, DIM_COLOR, align='right'))
            self.skill_hits.append(((210, line_top, 590, line_top + ABILITY_ROW_HEIGHT), 'perk', perk.code))
        if not available:
            items.append(ui.text_item(NOT_MADE, (220, AVAILABLE_LIST_TOP, 590, AVAILABLE_LIST_TOP + 16),
                                      ui.FONT_SMALL, DIM_COLOR))
        return items

    def on_skill_click(self, index):
        """
        The click on a skill: its next level for the free experience (624ed0, 528ab0).
        """
        stats = self.selected_stats()
        if not skills_module.raise_skill(stats, index):
            self.play(SOUND_CHANGE)
            self.message('Not enough experience')
            return
        self.after_stats_change(stats)

    def on_perk_click(self, code):
        """
        The click on an available ability: it is taken and its price goes to the experience of the perks
        (6268e0, 526450, 528050).
        """
        stats = self.selected_stats()
        perk = stats.perk_database.perk(code) if stats is not None and stats.perk_database is not None else None
        if perk is None or not skills_module.buy_perk(stats, perk):
            self.play(SOUND_CHANGE)
            self.message('Not enough experience')
            return
        self.after_stats_change(stats)

    def after_stats_change(self, stats):
        """
        524560: the health and the stamina of the unit of the member are counted again.
        """
        self.play(SOUND_PERK)
        unit = self.member_unit(self.camp.selected)
        if unit is not None:
            experience_module.update_fighter(unit.fighter, stats)
            if unit.hero_stats is None:
                unit.hero_stats = stats
            self.refresh_figure()
        self.message('Bought')
        self.shown_key = None
        self.update()

    def trade(self):
        return getattr(getattr(self.app, 'battle', None), 'trade', None)

    def money_text(self):
        label = self.text('string camp_current_money', 'Money:').rstrip(':').strip()
        return '%s\n%d' % (label, self.camp.money)

    def cell_text(self, name):
        """
        The name of the thing and its deal value (the trade window) or its repair price (the repair window).
        """
        deal = self.trade()
        if deal is None:
            return name
        if self.camp.window == camp_module.REPAIR_WINDOW:
            price = deal.repair_price(name)
            return '%s\n%s' % (name, 'in order' if price <= 0 else price)
        return '%s\n%d' % (name, deal.sale_price(name))

    def grid_items(self):
        camp = self.camp
        items = self.panel_items()
        key = camp_module.TITLE_KEYS.get(camp.window, 'string skills')
        title = self.text(key, camp.window.title())
        items.append(ui.text_item(title, TITLE_BOX, ui.FONT_MEDIUM, TITLE_COLOR))
        names = camp.cell_items()
        deal = camp.window in (camp_module.TRADE_WINDOW, camp_module.REPAIR_WINDOW)
        money_index = len(camp_module.CELLS) - 1
        if deal:
            names = names[:money_index]
        for index, rect in enumerate(camp_module.CELLS):
            textures = (camp_module.TRADE_TEXTURES if camp.window == camp_module.TRADE_WINDOW
                        else camp_module.REPAIR_TEXTURES)
            texture = textures[(index // 4 + index % 2) % len(textures)]
            items.append(ui.sprite_item(texture, rect, camp_module.CELL_TEXELS))
            items.extend(ui.panel_items(rect, 1, DIM_COLOR, DIM_COLOR, background=(0.0, 0.0, 0.0, 0.0))[1:])
            if deal and index == money_index:
                items.append(ui.text_item(self.money_text(), (rect[0] + 8, rect[1] + 8, rect[2] - 8, rect[3] - 8),
                                          ui.FONT_SMALL, TEXT_COLOR))
            elif index < len(names):
                items.append(ui.text_item(self.cell_text(names[index]),
                                          (rect[0] + 8, rect[1] + 8, rect[2] - 8, rect[3] - 8),
                                          ui.FONT_SMALL, TEXT_COLOR))
        if not names and not deal:
            items.append(ui.text_item(NOT_MADE, (220, 140, 580, 160), ui.FONT_SMALL, DIM_COLOR))
        close = camp_module.CLOSE_RECT
        items.append(ui.box_item(close, (0.35, 0.30, 0.20, 1.0)))
        items.append(ui.text_item(CLOSE_LABEL, (close[0] + 6, close[1] + 2, close[2], close[3]), ui.FONT_MEDIUM,
                                  TEXT_COLOR))
        return items

    def panel_items(self):
        rect = camp_module.MAIN_RECT
        return ui.panel_items(rect, 3, (90, 90, 66), (132, 130, 90), background=(0.08, 0.08, 0.07, 1.0))

    def regions(self):
        result = [camp_module.MEMBER_RECTS[index] for index in range(self.camp.member_count())]
        result.extend(tab_rect(index) for index in range(len(TAB_LABELS)))
        if self.camp.window in (camp_module.TRADE_WINDOW, camp_module.REPAIR_WINDOW,
                                camp_module.ITEM_CONSTRUCTOR_WINDOW, camp_module.SPELL_CONSTRUCTOR_WINDOW):
            result.append(camp_module.CLOSE_RECT)
            result.extend(camp_module.CELLS)
        return result

    def on_mouse(self, x, y):
        """
        True when the point is over the camp; the tab under the mouse is remembered.
        """
        if self.camp is None:
            return False
        self.hovered_tab = None
        self.mouse = (x, y)
        for index in range(len(TAB_LABELS)):
            rect = tab_rect(index)
            if rect[0] <= x < rect[2] and rect[1] <= y < rect[3]:
                self.hovered_tab = index
                break
        if self.dragging is not None:
            self.clear_tooltip()
            self.shown_key = None
            self.update()
        else:
            self.on_slot_tooltip(x, y)
        return True

    def on_click(self, x, y):
        if self.camp is None:
            return False
        for index, (window, key) in enumerate(TAB_LABELS):
            rect = tab_rect(index)
            if rect[0] <= x < rect[2] and rect[1] <= y < rect[3]:
                self.play(SOUND_MODE)
                if window == camp_module.CAMP_WINDOW:
                    self.camp.close_window()
                else:
                    self.camp.open(window)
                self.update()
                return True
        if self.camp.window == camp_module.SKILLS_WINDOW:
            for rect, kind, value in self.skill_hits:
                if rect[0] <= x < rect[2] and rect[1] <= y < rect[3]:
                    if kind == 'skill':
                        self.on_skill_click(value)
                    else:
                        self.on_perk_click(value)
                    return True
        if self.camp.window == camp_module.CAMP_WINDOW:
            hit = camp_module.slot_at(x, y, self.camp.member_count())
            if hit is not None:
                self.on_slot_click(*hit)
                return True
        action = self.camp.click(x, y)
        if action is not None:
            if _Debug:
                print('camp: %s' % action)
            if action == 'select':
                self.play(SOUND_CHANGE)
            elif action.startswith('cell ') and self.camp.window in (camp_module.TRADE_WINDOW,
                                                                    camp_module.REPAIR_WINDOW):
                self.on_cell(int(action.split()[1]))
            self.update()
            return True
        return True

    def on_cell(self, index):
        """
        The trade window sells the thing of the cell, the repair window repairs it; the last cell is the
        money of the party (its clicks do nothing).
        """
        camp = self.camp
        deal = self.trade()
        if deal is None or index >= len(camp_module.CELLS) - 1:
            return
        names = camp.cell_items()
        if index >= len(names):
            return
        name = names[index]
        self.play(SOUND_BUY)
        if camp.window == camp_module.TRADE_WINDOW:
            price = deal.sell(name)
            if price is None:
                self.message('Cannot sell %s' % name)
                return
            self.message('Sold %s for %d' % (name, price))
            camp.items = list(self.app.party.items)
            camp.trade_items = list(self.app.party.items)
            self.sync_strip()
            return
        result = deal.repair(name)
        if result is None:
            self.message('%s needs no repair' % name)
        elif result is False:
            self.message('Not enough money to repair %s' % name)
        else:
            self.message('Repaired %s for %d' % (name, result))

    def message(self, text):
        battle = getattr(self.app, 'battle', None)
        if battle is not None:
            battle.add_message(text)

    def play(self, name):
        self.app.ui_sound(name)

    def on_key(self, name):
        """
        Escape closes the open window or the camp; the digits of the windows of the remake.
        """
        if self.camp is None:
            return False
        if name == 'escape':
            if self.camp.window != camp_module.CAMP_WINDOW:
                self.camp.close_window()
            else:
                self.close()
            self.update()
            return True
        if self.camp.picked is not None and name in ('1', '2', '3', '4'):
            # a thing of the strip goes to a slot of the belt of the hero (UNIT_QUICK_ITEMS)
            self.app.set_quick_belt(int(name) - 1, self.camp.picked)
            self.play(SOUND_PUT_ON)
            self.camp.picked = None
            self.update()
            return True
        return False
