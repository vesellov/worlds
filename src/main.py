import os
import sys
import gc
import json
import threading
import traceback

_Debug = True

os.environ['KIVY_WINDOW'] = 'sdl2'
os.environ['KIVY_TEXT'] = 'sdl2'
os.environ['KIVY_IMAGE'] = 'sdl2'


PLATFORM = None

kivy_build = os.environ.get('KIVY_BUILD', '')
if kivy_build in {'android', 'ios'}:
    PLATFORM = kivy_build
elif 'P4A_BOOTSTRAP' in os.environ:
    PLATFORM = 'android'
elif 'ANDROID_ARGUMENT' in os.environ:
    PLATFORM = 'android'
elif sys.platform in ('win32', 'cygwin'):
    PLATFORM = 'win'
elif sys.platform == 'darwin':
    PLATFORM = 'macosx'
elif sys.platform.startswith('linux'):
    PLATFORM = 'linux'
elif sys.platform.startswith('freebsd'):
    PLATFORM = 'linux'
else:
    raise Exception(f'Unsupported platform: {sys.platform}')


import system
system.current_platform(known_platform=PLATFORM)

ROOT_PATH = ''
if system.is_android():
    ROOT_PATH = os.path.abspath(os.environ['ANDROID_ARGUMENT'])
elif system.is_osx():
    ROOT_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
elif system.is_ios():
    ROOT_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
else:
    ROOT_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

if not system.is_android():
    from screeninfo import get_monitors
    this_monitor = None
    for m in get_monitors():
        print(f'display monitor: {m.name} {m.width}x{m.height} primary={m.is_primary}')
        if m.is_primary:
            this_monitor = m
    if this_monitor:
        display_width, display_height = this_monitor.width, this_monitor.height
        if display_width >= 3200:
            target_dpi = 192
            target_density = 2.0
            target_fontscale = 1.2
        elif display_width <= 1280:
            target_dpi = 72
            target_density = 0.75
            target_fontscale = 0.9
        else:
            target_dpi = 96
            target_density = 1.0
            target_fontscale = 1.0
        os.environ['KIVY_DPI'] = f'{target_dpi}'
        os.environ['KIVY_METRICS_DENSITY'] = f'{target_density}'
        os.environ['KIVY_METRICS_FONTSCALE'] = f'{target_fontscale}'
    else:
        display_width, display_height = 800, 600

from kivy.config import Config
if not system.is_android():
    Config.set('graphics', 'top', f'{int(display_width/4)}')
    Config.set('graphics', 'left', f'{int(display_height/4)}')
    Config.set('graphics', 'width', f'{int(display_width/2)}')
    Config.set('graphics', 'height', f'{int(display_width/4)}')

from kivy.core.window import Window
from kivy.metrics import dp, sp, cm, Metrics

if system.is_android():
    display_width, display_height = Window.system_size

# if system.is_android():
#     Config.set('kivy', 'keyboard_mode', 'managed')

from kivy.clock import Clock
from kivy.app import App
from kivy.uix.progressbar import ProgressBar
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button

import res
import rend
import dat
import scen
import scr
import cli
import loader

class WorldsApp(App):

    known_templates = {}
    known_figures_parts = {}
    game_client = None

    def build(self):
        if _Debug:
            print(f'building app screen dpi={Metrics.dpi} density={Metrics.density} fontscale={Metrics.fontscale} display={display_width}x{display_height}')
        # if system.is_android():
        #     Window.keyboard_mode = 'managed'
        # Window.bind(on_request_close=self.on_request_close)
        self.inventory_list = []
        if not loader.FIGURES_URL_PREFIX:
            # models are downloaded one by one when the game server tells which are needed, see loader.ModelsLoader
            # until the separate models are published the whole figures.res is downloaded and models are taken from it
            self.inventory_list.append(('data', 'figures.res', ['figures_res_0', 'figures_res_1', ], 'https://raw.githubusercontent.com/eigamer/ei/refs/heads/main/astral2006/res/'))
        self.inventory_list += [
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
                # if _Debug:
                #     print(f'        file already exists: {dest_file_path}')
                continue
            self.downloading_list.append((folder, file_name, file_parts, url_prefix))
        self.root = scr.RootContainer()
        self.downloading_popup = None
        if not self.downloading_list:
            Clock.schedule_once(self.do_start)
            return self.root
        self.downloading_list = self.inventory_list
        self.progress_bar = ProgressBar(value=0, max=len(self.downloading_list), height=32, size_hint=(1, None))
        self.downloading_popup = Popup(title='downloading...', separator_height=0, content=self.progress_bar, size_hint=(0.8, 0.2), auto_dismiss=False)
        self.downloading_popup.open()
        self.task = Clock.schedule_once(self.do_download_next_file)
        return self.root

    def on_request_close(self, *args):
        if self.task:
            self.task.cancel()
            self.task = None
        return True

    def do_download_next_file(self, dt):
        self.task = None
        folder, file_name, file_parts, url_prefix = self.downloading_list.pop(0)
        if _Debug:
            print(f'downloading {folder}/{file_name} from {url_prefix}')
        # pauses between requests and retries take time, the window must not freeze meanwhile
        threading.Thread(target=self.download_in_background, args=(folder, file_name, file_parts, url_prefix), daemon=True).start()

    def download_in_background(self, folder, file_name, file_parts, url_prefix):
        dest_file_path = os.path.join(res.data_path(folder), file_name)
        urls = [url_prefix + part for part in (file_parts or [file_name, ])]
        try:
            loader.download_file(dest_file_path, urls)
        except Exception:
            traceback.print_exc()
            Clock.schedule_once(lambda dt: self.on_download_failed(file_name))
            return
        Clock.schedule_once(lambda dt: self.on_file_downloaded(dest_file_path))

    def on_download_failed(self, file_name):
        # the game can not start without the file, the next start continues from this file
        if self.downloading_popup:
            self.downloading_popup.title = f'failed to download {file_name}, please restart the game later'

    def on_file_downloaded(self, file_path):
        self.progress_bar.value += 1
        if self.downloading_list:
            # the pause between requests is added by loader.download_file
            self.task = Clock.schedule_once(self.do_download_next_file)
        else:
            self.task = None
            if self.downloading_popup:
                self.downloading_popup.dismiss()
            Clock.schedule_once(self.do_start)

    def do_start(self, dt):
        if _Debug:
            print(f'starting the app')
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
        self.catalog = catalog
        self.land = land
        self.scene = None
        self.models_popup = None
        # scene is rendered only after the game server said which models are used in the world and all of them are on the device
        self.game_client = cli.GameClient(url=os.environ.get('WORLDS_SERVER_URL'), on_welcome=self.on_welcome)
        self.show_models_popup(title='connecting to the game server...')
        self.game_client.start()

    def show_models_popup(self, title, maximum=1):
        if not self.models_popup:
            self.models_progress_bar = ProgressBar(value=0, max=maximum, height=32, size_hint=(1, None))
            self.models_popup = Popup(title=title, separator_height=0, content=self.models_progress_bar, size_hint=(0.8, 0.2), auto_dismiss=False)
            self.models_popup.open()
        self.models_popup.title = title
        self.models_progress_bar.max = max(1, maximum)
        self.models_progress_bar.value = 0

    def hide_models_popup(self):
        if self.models_popup:
            self.models_popup.dismiss()
            self.models_popup = None

    def on_welcome(self, templates):
        self.required_templates = templates
        models_loader = loader.ModelsLoader(templates, on_progress=self.on_models_progress, on_done=self.on_models_ready)
        if models_loader.missing:
            self.show_models_popup(title='downloading models...', maximum=len(models_loader.missing))
        models_loader.start()

    def on_models_progress(self, done, total, template):
        if self.models_popup:
            self.models_progress_bar.value = done
            self.models_popup.title = f'downloading models... {done}/{total}'

    def on_models_ready(self, failed):
        self.hide_models_popup()
        if failed:
            # the world can not be rendered without all of its models, game does not start
            self.show_models_error(failed)
            return
        if not self.scene:
            self.build_scene()
        self.game_client.attach(scene=self.scene, renderer=self.renderer)

    def show_models_error(self, failed):
        content = BoxLayout(orientation='vertical', spacing=dp(10), padding=dp(10))
        names = ', '.join(failed[:10]) + (f' and {len(failed) - 10} more' if len(failed) > 10 else '')
        message = Label(
            text=f'Failed to download {len(failed)} models required by the game server:\n{names}\n\nCheck the network connection and try again.',
            halign='center',
            valign='middle',
        )
        message.bind(size=lambda instance, value: setattr(instance, 'text_size', value))
        content.add_widget(message)
        retry_button = Button(text='try again', size_hint=(1, None), height=dp(44))
        content.add_widget(retry_button)
        error_popup = Popup(title='models were not downloaded', content=content, size_hint=(0.9, 0.6), auto_dismiss=False)

        def _retry(*args):
            error_popup.dismiss()
            # messages from the server keep waiting in the queue until the scene is ready, see cli.GameClient.attach()
            self.on_welcome(self.required_templates)

        retry_button.bind(on_release=_retry)
        error_popup.open()

    def build_scene(self):
        catalog = self.catalog
        land = self.land
        scene = scen.Scene(land=land, catalog=catalog)
        self.scene = scene
        scene.calculate_land_vertices()
        scene.calculate_scaled_elevation_map()
        self.renderer = rend.Renderer(app_root=self, scene=scene)
        self.known_templates = json.loads(open(res.data_path('catalog/figures_samples.json'), 'rt').read())
        scene.renderer = self.renderer
        scene.init_scene(250, 350)
        # all units are created by the game server, client only renders them, see cli.GameClient
        self.renderer.game_client = self.game_client
        # must add renderer the first to properly propagate touch events and draw widgets on top of it
        self.root.add_widget(self.renderer)
        self.renderer.joystick_button = scr.JoystickRoundButton(
            move_callback=self.renderer.on_joystick_move,
            text="", 
            # font_size=sp(16),
            size_hint=(None, None),
            size=(cm(3), cm(3)),
        )
        fixed_offset = dp(5)
        self.renderer.joystick_button.x = fixed_offset
        self.renderer.joystick_button.y = fixed_offset
        self.root.add_widget(self.renderer.joystick_button)
        # the loaded catalog, map and models are millions of long-living Python objects, a full garbage
        # collection has to walk all of them and it freezes the app for hundreds of milliseconds on a phone,
        # move them out of the garbage collector reach so periodic collections only check new objects
        gc.collect()
        gc.freeze()

    def on_stop(self):
        if self.game_client:
            self.game_client.stop()


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
