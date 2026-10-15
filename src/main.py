import os
import sys
import json
import random
import math

_Debug = True

os.environ['KIVY_WINDOW'] = 'sdl2'
os.environ['KIVY_TEXT'] = 'sdl2'
os.environ['KIVY_IMAGE'] = 'sdl2'

import system

ROOT_PATH = ''
if system.is_android():
    ROOT_PATH = os.path.abspath(os.environ['ANDROID_ARGUMENT'])
elif system.is_osx():
    ROOT_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
elif system.is_ios():
    ROOT_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
else:
    ROOT_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

from kivy.config import Config
Config.set('graphics', 'top', '100')
Config.set('graphics', 'left', '100')
Config.set('graphics', 'width', '1400')
Config.set('graphics', 'height', '700')

if system.is_android():
    Config.set('kivy', 'keyboard_mode', 'managed')

from kivy.clock import Clock
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.progressbar import ProgressBar
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse
from kivy.metrics import dp, sp

import res
import rend
import dat
import scen



class JoystickRoundButton(Button):

    def __init__(self, **kwargs):
        super(JoystickRoundButton, self).__init__(**kwargs)
        self.background_normal = ''
        self.background_color = (0, 0, 0, 0)
        self.btn_color = (0.2, 0.6, 1, 0.2) 
        self.label_w = Label(text="W", font_size=sp(12), bold=True)
        self.label_a = Label(text="A", font_size=sp(12), bold=True)
        self.label_s = Label(text="S", font_size=sp(12), bold=True)
        self.label_d = Label(text="D", font_size=sp(12), bold=True)
        self.add_widget(self.label_w)
        self.add_widget(self.label_a)
        self.add_widget(self.label_s)
        self.add_widget(self.label_d)
        self.bind(pos=self.update_layout, size=self.update_layout)
        # self.state = 'down'

    def update_layout(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            if self.state == 'normal':
                Color(*self.btn_color)
            else:
                Color(self.btn_color[0]*0.7, self.btn_color[1]*0.7, self.btn_color[2]*0.7, self.btn_color[3])
            diameter = min(self.width, self.height)
            x = self.x + (self.width - diameter) / 2
            y = self.y + (self.height - diameter) / 2
            Ellipse(pos=(x, y), size=(diameter, diameter))
        center_x = self.x + self.width / 2
        center_y = self.y + self.height / 2
        radius = diameter / 2.0
        offset = radius * 0.8
        self.label_w.center = (center_x, center_y + offset)
        self.label_s.center = (center_x, center_y - offset)
        self.label_a.center = (center_x - offset, center_y)
        self.label_d.center = (center_x + offset, center_y)

    def on_state(self, instance, value):
        self.update_layout()

    def collide_point(self, x, y):
        if _Debug:
            print(f'    joystick.collide_point {x}, {y}')
        center_x = self.x + self.width / 2
        center_y = self.y + self.height / 2
        radius = min(self.width, self.height) / 2
        distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        return distance <= radius

    def on_touch_down(self, touch):
        if _Debug:
            print(f'            joystick.on_touch_down {touch} is_touch={touch.is_touch} type_id={touch.type_id} profile={touch.profile} button={touch.button} shape={touch.shape} is_double_tap={touch.is_double_tap} is_triple_tap={touch.is_triple_tap}')
        return super(JoystickRoundButton, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if _Debug:
            print(f'            joystick.on_touch_move {touch} is_touch={touch.is_touch} type_id={touch.type_id} profile={touch.profile} button={touch.button} shape={touch.shape} is_double_tap={touch.is_double_tap} is_triple_tap={touch.is_triple_tap}')
        return super(JoystickRoundButton, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        if _Debug:
            print(f'            joystick.on_touch_up {touch} is_touch={touch.is_touch} type_id={touch.type_id} profile={touch.profile} button={touch.button} shape={touch.shape} is_double_tap={touch.is_double_tap} is_triple_tap={touch.is_triple_tap}')
        return super(JoystickRoundButton, self).on_touch_up(touch)


class RootContainer(FloatLayout):

    round_button = None

    # def on_size(self, *args):
    #     print(f'RootContainer.on_size {args}')
        # if self.round_button:
        #     fixed_offset = dp(100)
        #     w, h = args[1]
        #     self.round_button.x = self.round_button.width + fixed_offset
        #     self.round_button.y = self.round_button.height + fixed_offset

    # def on_update_round_button_position(self, instance, size):
    #     fixed_offset = dp(500)
    #     self.round_button.x = self.width - self.round_button.width - fixed_offset
    #     self.round_button.y = self.height - self.round_button.height - fixed_offset
    #     print(f'on_update_round_button_position {instance} {size} {self.round_button.x} {self.round_button.y}')


class WorldsApp(App):

    known_templates = {}
    known_figures_parts = {}

    def build(self):
        if system.is_android():
            Window.keyboard_mode = 'managed'
        # Window.bind(on_request_close=self.on_request_close)
        # url_prefix = 'https://raw.githubusercontent.com/eigamer/ei/refs/heads/main/eng2001/res/'
        # url_prefix = 'https://raw.githubusercontent.com/eigamer/ei/refs/heads/main/astral2006/res/'
        self.inventory_list = [
            ('data', 'figures.res', ['figures_res_0', 'figures_res_1', ], 'https://raw.githubusercontent.com/eigamer/ei/refs/heads/main/astral2006/res/'),
            ('data', 'textures.res', ['textures_res_0', 'textures_res_1', 'textures_res_2', ], 'https://raw.githubusercontent.com/eigamer/ei/refs/heads/main/astral2006/res/'),
            ('data', 'redress.res', ['redress_res_0', 'redress_res_1', ], 'https://raw.githubusercontent.com/eigamer/ei/refs/heads/main/astral2006/res/'),
            ('data', 'textures2.res', ['textures_res_0', 'textures_res_1', 'textures_res_2', ], 'https://raw.githubusercontent.com/eigamer/ei/refs/heads/main/eng2001/res/'),
            ('catalog', 'animations.json', [], 'https://raw.githubusercontent.com/bitdust-io/worlds/refs/heads/main/catalog/'),
            ('catalog', 'armors.json', [], 'https://raw.githubusercontent.com/bitdust-io/worlds/refs/heads/main/catalog/'),
            ('catalog', 'buildings.json', [], 'https://raw.githubusercontent.com/bitdust-io/worlds/refs/heads/main/catalog/'),
            ('catalog', 'figures.json', [], 'https://raw.githubusercontent.com/bitdust-io/worlds/refs/heads/main/catalog/'),
            ('catalog', 'figures_names.json', [], 'https://raw.githubusercontent.com/bitdust-io/worlds/refs/heads/main/catalog/'),
            ('catalog', 'figures_parts.json', [], 'https://raw.githubusercontent.com/bitdust-io/worlds/refs/heads/main/catalog/'),
            ('catalog', 'figures_samples.json', [], 'https://raw.githubusercontent.com/bitdust-io/worlds/refs/heads/main/catalog/'),
            ('catalog', 'materials.json', [], 'https://raw.githubusercontent.com/bitdust-io/worlds/refs/heads/main/catalog/'),
            ('catalog', 'plants.json', [], 'https://raw.githubusercontent.com/bitdust-io/worlds/refs/heads/main/catalog/'),
            ('catalog', 'textures.json', [], 'https://raw.githubusercontent.com/bitdust-io/worlds/refs/heads/main/catalog/'),
            ('catalog', 'weapons.json', [], 'https://raw.githubusercontent.com/bitdust-io/worlds/refs/heads/main/catalog/'),
            ('assets', 'water1.png', [], 'https://raw.githubusercontent.com/bitdust-io/worlds/refs/heads/main/assets/'),
            ('assets', 'sky1.png', [], 'https://raw.githubusercontent.com/bitdust-io/worlds/refs/heads/main/assets/'),
            # TODO: the following needs to move from "assets" to "map" sub dir
            ('assets', 'tiles.json', [], 'https://raw.githubusercontent.com/bitdust-io/worlds/refs/heads/main/assets/'),
            ('assets', 'catalog.json', [], 'https://raw.githubusercontent.com/bitdust-io/worlds/refs/heads/main/assets/'),
            ('assets', 'map.png', [], 'https://raw.githubusercontent.com/bitdust-io/worlds/refs/heads/main/assets/'),
            ('assets', 'heightmap.png', [], 'https://raw.githubusercontent.com/bitdust-io/worlds/refs/heads/main/assets/'),
            ('assets', 'encoded.png', [], 'https://raw.githubusercontent.com/bitdust-io/worlds/refs/heads/main/assets/'),
            ('assets', 'plants.json', [], 'https://raw.githubusercontent.com/bitdust-io/worlds/refs/heads/main/assets/'),
            ('assets', 'buildings.json', [], 'https://raw.githubusercontent.com/bitdust-io/worlds/refs/heads/main/assets/'),
        ]
        for i in range(0, 29):
            self.inventory_list.append(('assets/land', f'{i}.png', [], 'https://raw.githubusercontent.com/bitdust-io/worlds/refs/heads/main/assets/land/'))
        for tex in ['flag_east', 'flag_north', 'flag_south', 'flag_undead', 'gipat2sys', 'gipatsys', 'ingossys', 'jigransys', 'quater01', 'suslangersys', ]:
            self.inventory_list.append(('textures/model', f'{tex}.png', [], 'https://raw.githubusercontent.com/bitdust-io/worlds/refs/heads/main/assets/flags/'))
        self.downloading_list = []
        for folder, file_name, file_parts, url_prefix in self.inventory_list:
            dest_file_path = os.path.join(res.data_path(folder), file_name)
            if os.path.isfile(dest_file_path):
                if _Debug:
                    print(f'        file already exists: {dest_file_path}')
                continue
            self.downloading_list.append((folder, file_name, file_parts, url_prefix))
        self.root = RootContainer()
        self.downloading_popup = None
        if not self.downloading_list:
            Clock.schedule_once(self.do_start)
            return self.root
        self.downloading_list = self.inventory_list
        self.progress_bar = ProgressBar(value=0, max=len(self.downloading_list), height=32, size_hint=(1, None))
        self.downloading_popup = Popup(title='Downloading', separator_height=0, content=self.progress_bar, size_hint=(0.8, 0.2), auto_dismiss=False)
        self.downloading_popup.open()
        self.task = Clock.schedule_once(self.do_download_next_file)
        return self.root

    def on_request_close(self, *args):
        if self.task:
            self.task.cancel()
            self.task = None
        return True

    def do_download_next_file(self, dt):
        folder, file_name, file_parts, url_prefix = self.downloading_list.pop(0)
        if _Debug:
            print(f'downloading {folder}/{file_name} from {url_prefix}')
        self.task = Clock.schedule_once(lambda _: res.download_file(folder, file_name, file_parts, url_prefix, callback=self.on_file_downloaded))

    def on_file_downloaded(self, file_path):
        self.progress_bar.value += 1
        if self.downloading_list:
            delay = 0
            if file_path:
                delay = random.randint(1, 3)
            self.task = Clock.schedule_once(self.do_download_next_file, delay)
        else:
            self.task = None
            if self.downloading_popup:
                self.downloading_popup.dismiss()
            Clock.schedule_once(self.do_start)

    def do_start(self, dt):
        if _Debug:
            print('starting the app')
        self.root.clear_widgets()
        catalog = dat.CatalogData()
        catalog.load_figures(figures_file_name=res.data_path('catalog/figures.json'))
        catalog.load_animations(animations_file_name=res.data_path('catalog/animations.json'))
        catalog.load_armors(armors_file_name=res.data_path('catalog/armors.json'))
        catalog.load_weapons(weapons_file_name=res.data_path('catalog/weapons.json'))
        catalog.load_materials(materials_file_name=res.data_path('catalog/materials.json'))
        land = dat.LandData()
        land.load_heightmap_file(heightmap_file_name=res.data_path('assets/heightmap.png'))
        land.load_tilemap_file(tilemap_file_name=res.data_path('assets/encoded.png'))
        land.load_cache_tiles_textures(textures_dir_path=res.data_path('assets/land'))
        land.load_plants_data(plants_data_file_name=res.data_path('assets/plants.json'))
        land.load_buildings_data(buildings_data_file_name=res.data_path('assets/buildings.json'))
        scene = scen.Scene(land=land, catalog=catalog)
        scene.calculate_land_vertices()
        scene.calculate_scaled_elevation_map()
        renderer = rend.Renderer(app_root=self, scene=scene)
        self.known_templates = json.loads(open(res.data_path('catalog/figures_samples.json'), 'rt').read())
        scene.renderer = renderer
        scene.init_scene(257, 340)
        # scene.init_scene()
        # self.create_human_hero(scene)
        scene.create_hero(
            # model_name='unorma', weapon='stone battle axe.granite',
            # model_name='unmogo', texture='goblin00',
            # model_name='unmori', texture='rick',
            # model_name='unmosu', texture='succubus02',
            # model_name='unmotr', texture='troll02',
            model_name='unhuma',
            skin=41,
            hair=0,
            wears=[
              "hadagan brigand pants.thin",
              "hadagan brigand boots.thin",
              "hadagan brigand gloves.thin",
              "hadagan brigand leggins.thick",
              "hadagan brigand helm.thick",
              "hadagan brigand plate.thick"
            ],
            weapon='cheat dagger.bronze',
            # elevation_correction=0.5,
        )
        scene.renderer.camera_unit_lock = scene.hero.unit_name
        self.root.add_widget(renderer)
        self.root.round_button = JoystickRoundButton(
            text="", 
            # font_size=sp(16),
            size_hint=(None, None),
            size=(dp(100), dp(100)),
        )
        fixed_offset = dp(5)
        self.root.round_button.x = fixed_offset
        self.root.round_button.y = fixed_offset
        self.root.add_widget(self.root.round_button)
        # self.root.round_button.bind(on_press=self.on_round_button_click)

    # def on_round_button_click(self, instance):
    #     if _Debug:
    #         print(f"Round button clicked: {instance}")


def main():
    res.DATA_PATH = system.get_app_data_path()
    if _Debug:
        print(f'data path: {res.DATA_PATH}')
        print(f'root path: {ROOT_PATH}')
    if not os.path.exists(res.DATA_PATH):
        os.makedirs(res.DATA_PATH)
    WorldsApp().run()


if __name__ == '__main__':
    main()
