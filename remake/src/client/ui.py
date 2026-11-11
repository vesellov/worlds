"""
The 2D interface over the 3D view, in the coordinates of game.exe: a screen of 800 x 600 with y going down.

The virtual screen is scaled by the height of the window and centered, the projection of the 3D view keeps
the vertical angle of the game (client.gl), so the rects of menus.reg cover the stones of the columns
at any width of the window.

Text: game.exe (CInterface3D::Create 5f4990) makes three "Times New Roman" fonts of 10.4, 11.2 and 14.4 points
for the width 800 (the sizes grow with the width), Liberation Serif has the same metrics.
Cursor: cursor_*.mmp are strips of 32 x 32 frames, black is transparent, the frame changes every 125 ms (5e52e0).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import os

import numpy as np
from kivy.core.text import Label as CoreLabel
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, Line, InstructionGroup
from kivy.uix.widget import Widget

#------------------------------------------------------------------------------

from engine import tooltip as tooltip_module
from client import gl

#------------------------------------------------------------------------------

WIDTH = 800
HEIGHT = 600

FONT_FILE_PATHS = (
    '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf',
    '/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf',
    '/usr/share/fonts/liberation-serif/LiberationSerif-Regular.ttf',
)

# tenths of a point at the width 800, game.exe 5f4990
FONT_TENTHS = (104, 112, 144)
FONT_SMALL = 0
FONT_MEDIUM = 1
FONT_LARGE = 2

CURSOR_FRAME = 32
CURSOR_FRAME_MS = 125.0

# hot points of the cursors (game.exe 5daea0), the others point with the top left corner
CURSOR_HOT_POINTS = {
    'cursor_scroll_u': (16, 0),
    'cursor_scroll_ur': (32, 0),
    'cursor_scroll_r': (32, 16),
    'cursor_scroll_dr': (32, 32),
    'cursor_scroll_d': (16, 32),
    'cursor_scroll_dl': (0, 32),
    'cursor_scroll_l': (0, 16),
    'cursor_scroll_ul': (0, 0),
}

#------------------------------------------------------------------------------


class VirtualScreen(object):

    def __init__(self, window_width, window_height):
        self.scale = window_height / float(HEIGHT)
        self.offset_x = (window_width - WIDTH * self.scale) * 0.5
        self.window_height = window_height

    def to_window(self, x, y):
        return self.offset_x + x * self.scale, self.window_height - y * self.scale

    def to_virtual(self, window_x, window_y):
        return (window_x - self.offset_x) / self.scale, (self.window_height - window_y) / self.scale

    def window_rect(self, left, top, right, bottom):
        """
        (x, y, width, height) in the window for a rect of the virtual screen.
        """
        x0, y0 = self.to_window(left, bottom)
        return x0, y0, (right - left) * self.scale, (bottom - top) * self.scale


class Cursor(object):

    def __init__(self, game_data):
        self.game_data = game_data
        self.textures = {}
        self.name = None
        self.frames = 1
        self.time_ms = 0.0
        self.group = InstructionGroup()
        self.color = Color(1.0, 1.0, 1.0, 1.0)
        self.rectangle = Rectangle(size=(CURSOR_FRAME, CURSOR_FRAME))
        self.group.add(self.color)
        self.group.add(self.rectangle)
        self.set('cursor_default')

    def texture(self, name):
        if name not in self.textures:
            rgba = self.game_data.texture(name)
            if rgba is None:
                self.textures[name] = None
            else:
                self.textures[name] = gl.make_texture(color_key(rgba), mipmap=False, wrap='clamp_to_edge')
        return self.textures[name]

    def set(self, name):
        if name == self.name or self.texture(name) is None:
            return
        self.name = name
        texture = self.texture(name)
        self.frames = max(texture.width // CURSOR_FRAME, 1)
        self.rectangle.texture = texture
        self.time_ms = 0.0
        self._show_frame()

    def _show_frame(self):
        frame = int(self.time_ms // CURSOR_FRAME_MS) % self.frames
        u0, u1 = frame / float(self.frames), (frame + 1) / float(self.frames)
        # the textures are uploaded from the top row, so v = 0 is the top of the image
        self.rectangle.tex_coords = (u0, 1.0, u1, 1.0, u1, 0.0, u0, 0.0)

    def update(self, dt_ms, screen, window_x, window_y):
        self.time_ms += dt_ms
        self._show_frame()
        hot_x, hot_y = CURSOR_HOT_POINTS.get(self.name, (0, 0))
        size = CURSOR_FRAME * screen.scale
        self.rectangle.size = (size, size)
        self.rectangle.pos = (window_x - hot_x * screen.scale, window_y - size + hot_y * screen.scale)


class Interface(Widget):
    """
    Sprites and texts placed on the virtual screen, the outlines of the regions for debugging and the cursor.
    """

    def __init__(self, game_data, **kwargs):
        super(Interface, self).__init__(**kwargs)
        self.game_data = game_data
        self.font_file_path = next((p for p in FONT_FILE_PATHS if os.path.isfile(p)), None)
        self.items = []
        self.layers = {}
        self.textures = {}
        self.show_regions = False
        self.regions = []
        self.content = InstructionGroup()
        self.outlines = InstructionGroup()
        self.cursor = Cursor(game_data)
        self.canvas.add(self.content)
        self.canvas.add(self.outlines)
        self.canvas.add(self.cursor.group)
        self.bind(size=self._relayout, pos=self._relayout)

    @property
    def screen(self):
        return VirtualScreen(max(self.width, 1), max(self.height, 1))

    def texture(self, texture_name):
        if texture_name not in self.textures:
            rgba = self.game_data.texture(texture_name)
            self.textures[texture_name] = gl.make_texture(rgba, mipmap=False, wrap='clamp_to_edge') if rgba is not None else None
        return self.textures[texture_name]

    def clear(self):
        self.items = []
        self.layers = {}
        self.regions = []
        self._relayout()

    def set_layer(self, name, items):
        """
        Replaces a named group of items made by sprite_item(), text_item() and box_item(), the layers go over the items.
        """
        self.layers[name] = list(items)
        self._relayout()

    def add_sprite(self, texture_name, rect, texels=None):
        """
        A piece of an atlas: rect (left, top, right, bottom) on the screen, texels the same in the pixels of the texture.
        """
        self.items.append(sprite_item(texture_name, rect, texels))
        self._relayout()

    def add_text(self, text, box, font=FONT_MEDIUM, color=(255, 255, 255), align='left', valign='top'):
        self.items.append(text_item(text, box, font, color, align, valign))
        self._relayout()

    def set_regions(self, regions):
        """
        Outlines for debugging: rects (left, top, right, bottom) or polygons [(x, y), ...].
        """
        self.regions = [tuple(r) for r in regions]
        self._relayout()

    def font_size(self, font):
        return FONT_TENTHS[font] / 10.0 * 96.0 / 72.0 * self.screen.scale

    def _relayout(self, *args):
        screen = self.screen
        self.content.clear()
        items = list(self.items)
        for layer in self.layers.values():
            items.extend(layer)
        for kind, payload, rect, extra in items:
            if kind == 'sprite':
                texels, color, turn = extra
                self._draw_sprite(screen, payload, rect, texels, color, turn)
            elif kind == 'box':
                x, y, w, h = screen.window_rect(*rect)
                self.content.add(Color(*payload))
                self.content.add(Rectangle(pos=(x, y), size=(w, h)))
            else:
                self._draw_text(screen, payload, rect, extra)
        self.outlines.clear()
        if self.show_regions:
            self.outlines.add(Color(1.0, 0.9, 0.2, 0.8))
            for region in self.regions:
                if len(region) == 4 and not isinstance(region[0], (tuple, list)):
                    x, y, w, h = screen.window_rect(*region)
                    self.outlines.add(Line(rectangle=(x, y, w, h), width=1.2))
                else:
                    points = [c for point in region for c in screen.to_window(*point)]
                    self.outlines.add(Line(points=points, close=True, width=1.2))

    def _draw_sprite(self, screen, texture_name, rect, texels, color=None, turn=False):
        texture = self.texture(texture_name)
        if texture is None:
            return
        x, y, w, h = screen.window_rect(*rect)
        self.content.add(Color(*(color or (1.0, 1.0, 1.0, 1.0))))
        if texels is None:
            texels = (0, 0, texture.width, texture.height)
        u0, v0 = texels[0] / float(texture.width), texels[1] / float(texture.height)
        u1, v1 = texels[2] / float(texture.width), texels[3] / float(texture.height)
        if turn:
            # the corners from the bottom left counterclockwise: (u1, v1), (u1, v0), (u0, v0), (u0, v1)
            tex_coords = (u1, v1, u1, v0, u0, v0, u0, v1)
        else:
            tex_coords = (u0, v1, u1, v1, u1, v0, u0, v0)
        self.content.add(Rectangle(texture=texture, pos=(x, y), size=(w, h), tex_coords=tex_coords))

    def _draw_text(self, screen, text, box, extra):
        font, color, align, valign = extra
        x, y, w, h = screen.window_rect(*box)
        options = dict(text=text, font_size=self.font_size(font), halign=align, valign=valign,
                       text_size=(w, h), color=tuple(c / 255.0 for c in color) + (1.0, ))
        if self.font_file_path:
            options['font_name'] = self.font_file_path
        label = CoreLabel(**options)
        label.refresh()
        texture = label.texture
        if texture is None:
            return
        self.content.add(Color(1.0, 1.0, 1.0, 1.0))
        self.content.add(Rectangle(texture=texture, pos=(x, y + h - texture.height), size=texture.size))

    def text_size(self, text, font=FONT_MEDIUM):
        """
        (width, height) of the text on the virtual screen.
        """
        options = dict(text=text, font_size=self.font_size(font))
        if self.font_file_path:
            options['font_name'] = self.font_file_path
        label = CoreLabel(**options)
        label.refresh()
        scale = self.screen.scale
        return label.texture.width / scale, label.texture.height / scale

    def text_height(self, text, width, font=FONT_MEDIUM):
        """
        The height of the text wrapped into the width, on the virtual screen.
        """
        scale = self.screen.scale
        options = dict(text=text, font_size=self.font_size(font), text_size=(width * scale, None))
        if self.font_file_path:
            options['font_name'] = self.font_file_path
        label = CoreLabel(**options)
        label.refresh()
        return label.texture.height / scale

    def update(self, dt_ms):
        window_x, window_y = mouse_window_pos(Window.mouse_pos)
        self.cursor.update(dt_ms, self.screen, window_x - self.x, window_y - self.y)

    def virtual_point(self, window_x, window_y):
        return self.screen.to_virtual(window_x - self.x, window_y - self.y)

#------------------------------------------------------------------------------


def mouse_window_pos(pos):
    """
    The point of Window.mouse_pos as the touches give it: Kivy writes the mouse y as height - 1 - y
    (window_sdl2 _fix_mouse_pos) and the y of a touch as height - y, so the mouse was a pixel lower.
    """
    return pos[0], pos[1] + 1


def sprite_item(texture_name, rect, texels=None, color=None, turn=False):
    """
    A piece of an atlas: texels (left, top, right, bottom) in the pixels of the texture, a top below
    the bottom draws the piece upside down; color (r, g, b, a) multiplies its pixels. A turned piece is
    rotated by a quarter clockwise (the elements of 65b870 turned around z): the left of the texels goes
    to the top of the rect, their top to its right.
    """
    return ('sprite', texture_name, tuple(rect), (texels, tuple(color) if color else None, turn))


def text_item(text, box, font=FONT_MEDIUM, color=(255, 255, 255), align='left', valign='top'):
    return ('text', text, tuple(box), (font, tuple(color), align, valign))


def box_item(rect, rgba):
    return ('box', tuple(rgba), tuple(rect), None)


def bevel_items(rect, width, high_width, color, high_color):
    """
    A frame of width pixels outside the rect: the outer pixels are color, the inner high_width pixels
    are high_color (the bevel of the frames of the windows of game.exe, 5e7d60).
    """
    x0, y0, x1, y1 = rect
    x2, y2 = x1 + high_width, y1 + high_width
    return [
        box_item((x0 - width, y0 - width, x1 + width, y0 - high_width), color),
        box_item((x0 - width, y2, x1 + width, y1 + width), color),
        box_item((x0 - width, y0 - high_width, x0 - high_width, y2), color),
        box_item((x2, y0 - high_width, x1 + width, y2), color),
        box_item((x0 - high_width, y0 - high_width, x2, y0), high_color),
        box_item((x0 - high_width, y1, x2, y2), high_color),
        box_item((x0 - high_width, y0, x0, y1), high_color),
        box_item((x1, y0, x2, y1), high_color),
    ]


def border_items(rect, thickness, color, high_color):
    """
    A border of the color and a thin light line inside it, without the background (the windows of game.exe).
    """
    left, top, right, bottom = rect
    items = [
        box_item((left, top, right, top + thickness), color),
        box_item((left, bottom - thickness, right, bottom), color),
        box_item((left, top + thickness, left + thickness, bottom - thickness), color),
        box_item((right - thickness, top + thickness, right, bottom - thickness), color),
    ]
    h = 1
    items.extend([
        box_item((left + thickness, top + thickness, right - thickness, top + thickness + h), high_color),
        box_item((left + thickness, bottom - thickness - h, right - thickness, bottom - thickness), high_color),
        box_item((left + thickness, top + thickness + h, left + thickness + h, bottom - thickness - h),
                 high_color),
        box_item((right - thickness - h, top + thickness + h, right - thickness, bottom - thickness - h),
                 high_color),
    ])
    return items


def panel_items(rect, thickness, color, high_color, background=(0.0, 0.0, 0.0, 1.0)):
    """
    A black panel with a border of the color and a thin light line inside it (the windows of game.exe).
    """
    return [box_item(rect, background)] + border_items(rect, thickness, color, high_color)


def tooltip_items(interface, text, mouse):
    """
    65e540 (partly): the items of the tooltip of the text of texts.res for the mouse: the frame, the
    background, the title line and the other lines.
    """
    lines = tooltip_module.lines_of(text)
    if not lines:
        return []
    width, height = tooltip_module.size(lines, lambda line: int(interface.text_size(line, FONT_SMALL)[0]) + 1)
    left, top, right, bottom = tooltip_module.box(mouse[0], mouse[1], width, height)
    items = [
        box_item((left, top, right, bottom), tuple(c / 255.0 for c in tooltip_module.FRAME_COLOR) + (1.0, )),
        box_item((left + 1, top + 1, right - 1, bottom - 1),
                 tuple(c / 255.0 for c in tooltip_module.BACKGROUND_COLOR) + (1.0, )),
    ]
    y = top + tooltip_module.TOP_PADDING
    for index, line in enumerate(lines):
        color = tooltip_module.TITLE_COLOR if index == 0 else tooltip_module.TEXT_COLOR
        items.append(text_item(line, (left + tooltip_module.PADDING, y, right, y + tooltip_module.LINE_HEIGHT),
                               FONT_SMALL, color, valign='middle'))
        y += tooltip_module.LINE_HEIGHT
    return items


def color_key(rgba):
    """
    The black pixels of a DirectDraw surface with a color key become transparent.
    """
    result = np.array(rgba, copy=True)
    black = np.all(result[:, :, 0:3] == 0, axis=2)
    result[black, 3] = 0
    return result
