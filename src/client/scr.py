"""
Screen widgets: the root container of the application, the joystick button and the minimap.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math

from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line, Rectangle, Triangle, InstructionGroup
from kivy.metrics import dp, sp, cm

#------------------------------------------------------------------------------

from lib import system

#------------------------------------------------------------------------------

# how often the dots of the units on the minimap are moved, seconds
MINIMAP_UPDATE_INTERVAL = 0.2

# kind of the dot on the minimap -> (color, radius in dp), see MiniMap.show_units()
MINIMAP_DOTS = {
    'me': ((1.0, 0.9, 0.1, 1.0), 3.0),
    'hero': ((0.2, 0.8, 1.0, 1.0), 2.5),
    'unit': ((1.0, 0.25, 0.2, 1.0), 2.0),
}

#------------------------------------------------------------------------------


class RootContainer(FloatLayout):
    pass

#------------------------------------------------------------------------------


class JoystickRoundButton(Button):

    def __init__(self, **kwargs):
        self.move_callback = kwargs.pop('move_callback', None)
        super(JoystickRoundButton, self).__init__(**kwargs)
        self.background_normal = ''
        self.background_color = (0, 0, 0, 0)
        self.btn_color = (0.5, 0.5, 0.5, 0.3)
        self.letter_color = (1, 1, 1, 0.7)
        self.arrow_color = (0.6, 0.6, 0.6, 0.5)
        self.label_w = Label(text="" if system.is_android() else "", color=self.letter_color, font_size=sp(11), bold=True)
        self.label_a = Label(text="" if system.is_android() else "", color=self.letter_color, font_size=sp(11), bold=True)
        self.label_s = Label(text="" if system.is_android() else "", color=self.letter_color, font_size=sp(11), bold=True)
        self.label_d = Label(text="" if system.is_android() else "", color=self.letter_color, font_size=sp(11), bold=True)
        self.add_widget(self.label_w)
        self.add_widget(self.label_a)
        self.add_widget(self.label_s)
        self.add_widget(self.label_d)
        self.bind(pos=self.update_layout, size=self.update_layout)

    def update_layout(self, *args):
        self.canvas.before.clear()
        diameter = min(self.width, self.height)
        radius = diameter / 2.0
        x = self.x + (self.width - diameter) / 2
        y = self.y + (self.height - diameter) / 2
        cx = self.x + self.width / 2
        cy = self.y + self.height / 2
        arrow_size = diameter * 0.18
        arrow_offset = radius * 0.95
        letter_offset = radius * 0.74
        with self.canvas.before:
            if self.state == 'normal':
                Color(*self.btn_color)
            else:
                Color(self.btn_color[0]*0.7, self.btn_color[1]*0.7, self.btn_color[2]*0.7, self.btn_color[3])
            Ellipse(pos=(x, y), size=(diameter, diameter))
            Color(*self.arrow_color)
            Triangle(points=[
                cx, cy + arrow_offset,
                cx - arrow_size, cy + arrow_offset - arrow_size,
                cx + arrow_size, cy + arrow_offset - arrow_size,
            ])
            Triangle(points=[
                cx, cy - arrow_offset,
                cx - arrow_size, cy - arrow_offset + arrow_size,
                cx + arrow_size, cy - arrow_offset + arrow_size,
            ])
            Triangle(points=[
                cx - arrow_offset, cy,
                cx - arrow_offset + arrow_size, cy + arrow_size,
                cx - arrow_offset + arrow_size, cy - arrow_size,
            ])
            Triangle(points=[
                cx + arrow_offset, cy,
                cx + arrow_offset - arrow_size, cy + arrow_size,
                cx + arrow_offset - arrow_size, cy - arrow_size,
            ])
        self.label_w.center = (cx, cy + letter_offset)
        self.label_s.center = (cx, cy - letter_offset)
        self.label_a.center = (cx - letter_offset, cy)
        self.label_d.center = (cx + letter_offset, cy)

    def on_state(self, instance, value):
        self.update_layout()

    def collide_point(self, x, y):
        center_x = self.x + self.width / 2
        center_y = self.y + self.height / 2
        radius = min(self.width, self.height) / 2
        distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        is_colliding = distance <= radius
        return is_colliding

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            touch.grab(self)
            if self.move_callback:
                local_x, local_y = self.to_local(*touch.pos)
                local_center_x = self.width / 2
                local_center_y = self.height / 2
                rel_x = local_x - local_center_x
                rel_y = local_y - local_center_y
                self.move_callback(rel_x, rel_y)
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if touch.grab_current is self and self.collide_point(*touch.pos):
            if self.move_callback:
                local_x, local_y = self.to_local(*touch.pos)
                local_center_x = self.width / 2
                local_center_y = self.height / 2
                rel_x = local_x - local_center_x
                rel_y = local_y - local_center_y
                self.move_callback(rel_x, rel_y)
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            if self.move_callback:
                self.move_callback(0, 0)
            return True
        return super().on_touch_up(touch)

#------------------------------------------------------------------------------


class MiniMap(Widget):
    """
    The whole map of the biome in the corner of the screen: minimap.png of the biome and the dots of all units
    of the world. A pixel (x, y) of the image is the segment (w=x, h=y) of the map, the top row of the image is h=0.
    units_source() returns [(x, y, kind)], positions are continuous map coordinates, kind is a key of MINIMAP_DOTS.
    """

    def __init__(self, image_file_path, map_width, map_height, units_source, **kwargs):
        super(MiniMap, self).__init__(**kwargs)
        self.map_width = map_width
        self.map_height = map_height
        self.units_source = units_source
        texture = CoreImage(image_file_path).texture
        # every pixel of the small image stays sharp when it is drawn bigger
        texture.mag_filter = 'nearest'
        with self.canvas:
            Color(1.0, 1.0, 1.0, 0.85)
            self.picture = Rectangle(texture=texture, pos=self.pos, size=self.size)
            Color(0.0, 0.0, 0.0, 0.6)
            self.frame = Line(rectangle=(self.x, self.y, self.width, self.height), width=1.0)
        self.dots = InstructionGroup()
        self.canvas.add(self.dots)
        self.bind(pos=self.update_layout, size=self.update_layout)
        self.update_event = Clock.schedule_interval(self.update, MINIMAP_UPDATE_INTERVAL)

    def update_layout(self, *args):
        self.picture.pos = self.pos
        self.picture.size = self.size
        self.frame.rectangle = (self.x, self.y, self.width, self.height)
        self.update()

    def update(self, *args):
        self.show_units(self.units_source() if self.units_source else [])

    def show_units(self, units):
        self.dots.clear()
        # the own hero is drawn the last, on top of the others
        for x, y, kind in sorted(units, key=lambda u: u[2] == 'me'):
            color, radius = MINIMAP_DOTS.get(kind, MINIMAP_DOTS['unit'])
            radius = dp(radius)
            px = self.x + (x % self.map_width) / self.map_width * self.width
            py = self.top - (y % self.map_height) / self.map_height * self.height
            self.dots.add(Color(*color))
            self.dots.add(Ellipse(pos=(px - radius, py - radius), size=(2 * radius, 2 * radius)))

    def stop(self):
        if self.update_event:
            self.update_event.cancel()
            self.update_event = None
