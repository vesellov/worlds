import math

from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Triangle
from kivy.metrics import dp, sp, cm

import system

class RootContainer(FloatLayout):
    pass


class JoystickRoundButton(Button):

    def __init__(self, **kwargs):
        self.move_callback = kwargs.pop('move_callback', None)
        super(JoystickRoundButton, self).__init__(**kwargs)
        self.background_normal = ''
        self.background_color = (0, 0, 0, 0)
        self.btn_color = (0.5, 0.5, 0.5, 0.3)
        self.letter_color = (1, 1, 1, 0.7)
        self.arrow_color = (0.6, 0.6, 0.6, 0.5)
        self.label_w = Label(text="" if system.is_android() else "W", color=self.letter_color, font_size=sp(12), bold=True)
        self.label_a = Label(text="" if system.is_android() else "A", color=self.letter_color, font_size=sp(12), bold=True)
        self.label_s = Label(text="" if system.is_android() else "S", color=self.letter_color, font_size=sp(12), bold=True)
        self.label_d = Label(text="" if system.is_android() else "D", color=self.letter_color, font_size=sp(12), bold=True)
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
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if touch.grab_current is self and self.collide_point(*touch.pos):
            local_x, local_y = self.to_local(*touch.pos)
            local_center_x = self.width / 2
            local_center_y = self.height / 2
            rel_x = local_x - local_center_x
            rel_y = local_y - local_center_y
            if self.move_callback:
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
