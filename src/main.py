import os
import sys
import json
import random

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

import res
import rend
import dat
import scen


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
                    print(f'{dest_file_path}')
                continue
            self.downloading_list.append((folder, file_name, file_parts, url_prefix))
        self.root = FloatLayout()
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
        overlay_button = Button(
            text="OK",
            size_hint=(0.1, 0.1),
            pos_hint={'center_x': 0.05, 'center_y': 0.05}
        )
        self.root.add_widget(overlay_button)


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
