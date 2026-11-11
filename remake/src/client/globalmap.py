"""
The window of the global map of an allod (game.exe 64cbd0): the texture "<allod>mapg" with the markers of
the zones and the roads out of the current zone of the roads. A click on a road takes the party to its
zone and adds the hours of the way (the "#passtime" of the exits); Escape closes the window and the party
stays in its zone.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from engine import globalmap as globalmap_module
from client import ui

#------------------------------------------------------------------------------

LAYER_NAME = 'global map'

SHADE = (0.0, 0.0, 0.0, 0.80)
MAP_RECT = (150, 60, 650, 560)
TITLE_BOX = (150, 20, 650, 52)
TITLE_COLOR = (255, 179, 49)
TEXT_COLOR = (228, 215, 167)
DIM_COLOR = (120, 120, 120)
ROAD_COLOR = (255, 215, 120)
CURRENT_COLOR = (110, 220, 130)
CLOSE_RECT = (740, 540, 780, 580)

MARKER_SIZE = 4
ROAD_SIZE = 8
ROAD_HIT_SIZE = 14

# the sounds of the global map (buttons/Globalmap of sfx.res)
SOUND_SELECT = 'buttons/Globalmap/Zone.wav'
SOUND_MOVE = 'buttons/Globalmap/Move.wav'
SOUND_CLOSE = 'buttons/Globalmap/Ok.wav'

# the texture of the map of an allod with the markers
MAP_TEXTURE_SUFFIX = 'mapg'

#------------------------------------------------------------------------------


def _color(rgb, alpha=1.0):
    return tuple(channel / 255.0 for channel in rgb) + (alpha, )


def zone_title(game_data, zone):
    """
    The title "zone <name>" of texts.res, the name of the zone without it.
    """
    text = game_data.text('zone %s' % zone.name) if game_data is not None else None
    if text:
        first = text.replace('\r', '').split('\n', 1)[0].strip()
        if first:
            return first
    return zone.name


class GlobalMapView(object):
    """
    The modal map of the allod with the roads out of the current zone.
    """

    def __init__(self, app):
        self.app = app
        self.model = None
        self.pass_time = 0.0
        self.shown_key = None
        self.road_rects = []

    def __repr__(self):
        return 'GlobalMapView(%s)' % (self.model.edge_zone.name if self.model is not None else 'closed')

    @property
    def is_open(self):
        return self.model is not None

    def open(self, edge_zone, pass_time=0.0):
        self.model = globalmap_module.GlobalMap.from_game(self.app.game_data, edge_zone)
        self.pass_time = float(pass_time)
        self.shown_key = None
        self.update()
        return True

    def close(self):
        self.model = None
        self.shown_key = None
        self.road_rects = []
        self.app.interface.set_layer(LAYER_NAME, [])

    def map_point(self, zone):
        marker = self.model.marker(zone)
        if marker is None:
            return None
        left, top, right, bottom = MAP_RECT
        return (left + marker[0] * (right - left), top + marker[1] * (bottom - top))

    def update(self):
        if self.model is None:
            return
        key = (self.model.edge_zone.name, tuple((option.zone.name, option.exit.number)
                                                for option in self.model.options))
        if key == self.shown_key:
            return
        self.shown_key = key
        self.app.interface.set_layer(LAYER_NAME, self.elements())

    def elements(self):
        app = self.app
        model = self.model
        allod = model.allod or ''
        texture_name = allod + MAP_TEXTURE_SUFFIX
        if app.game_data.texture(texture_name) is None:
            texture_name = allod + 'map'
        items = [ui.box_item((0, 0, 800, 600), SHADE),
                 ui.sprite_item(texture_name, MAP_RECT),
                 ui.text_item('Global map: %s' % allod, TITLE_BOX, ui.FONT_LARGE, TITLE_COLOR,
                              align='center', valign='middle')]
        for zone in model.zones:
            point = self.map_point(zone)
            if point is None:
                continue
            color = DIM_COLOR
            size = MARKER_SIZE
            items.append(ui.box_item((point[0] - size, point[1] - size, point[0] + size, point[1] + size),
                                     _color(color)))
        current = self.map_point(model.edge_zone) if model.edge_zone is not None else None
        if current is not None:
            items.append(ui.box_item((current[0] - ROAD_SIZE, current[1] - ROAD_SIZE, current[0] + ROAD_SIZE,
                                      current[1] + ROAD_SIZE), _color(CURRENT_COLOR)))
            items.append(ui.text_item(zone_title(app.game_data, model.edge_zone),
                                      (current[0] + 10, current[1] - 8, current[0] + 150, current[1] + 10),
                                      ui.FONT_SMALL, CURRENT_COLOR))
        self.road_rects = []
        for option in model.options:
            point = self.map_point(option.zone)
            if point is None:
                continue
            rect = (point[0] - ROAD_HIT_SIZE, point[1] - ROAD_HIT_SIZE, point[0] + ROAD_HIT_SIZE,
                    point[1] + ROAD_HIT_SIZE)
            self.road_rects.append((rect, option))
            items.append(ui.box_item((point[0] - ROAD_SIZE, point[1] - ROAD_SIZE, point[0] + ROAD_SIZE,
                                      point[1] + ROAD_SIZE), _color(ROAD_COLOR)))
            items.append(ui.text_item(zone_title(app.game_data, option.zone),
                                      (point[0] + 10, point[1] - 8, point[0] + 160, point[1] + 10),
                                      ui.FONT_SMALL, ROAD_COLOR))
        close = CLOSE_RECT
        items.append(ui.box_item(close, (0.35, 0.30, 0.20, 1.0)))
        items.append(ui.text_item('X', (close[0] + 6, close[1] + 2, close[2], close[3]), ui.FONT_MEDIUM,
                                  TEXT_COLOR))
        return items

    def option_at(self, x, y):
        for (left, top, right, bottom), option in self.road_rects:
            if left <= x < right and top <= y < bottom:
                return option
        return None

    def on_click(self, x, y):
        if not self.is_open:
            return False
        if _inside(CLOSE_RECT, x, y):
            self.app.ui_sound(SOUND_CLOSE)
            self.close()
            return True
        option = self.option_at(x, y)
        if option is None:
            return True
        self.app.ui_sound(SOUND_MOVE)
        total = self.pass_time + option.pass_time
        self.close()
        self.app.travel_through(option, total)
        return True

    def on_key(self, name):
        if not self.is_open:
            return False
        if name in ('escape', 'x'):
            self.app.ui_sound(SOUND_CLOSE)
            self.close()
        return True


def _inside(rect, x, y):
    return rect[0] <= x < rect[2] and rect[1] <= y < rect[3]
