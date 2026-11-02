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


from lib import system
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

from lib import res
from lib import models
from client import rend
from client import dat
from client import scen
from client import scr
from client import cli
from client import loader

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
        self.inventory_list = loader.inventory_list()
        self.downloading_list = []
        for folder, file_name, url_prefix in self.inventory_list:
            dest_file_path = os.path.join(res.data_path(folder), file_name)
            if os.path.isfile(dest_file_path):
                # if _Debug:
                #     print(f'        file already exists: {dest_file_path}')
                continue
            self.downloading_list.append((folder, file_name, url_prefix))
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
        folder, file_name, url_prefix = self.downloading_list.pop(0)
        if _Debug:
            print(f'downloading {folder}/{file_name} from {url_prefix}')
        # the download and its retries take time, the window must not freeze meanwhile
        threading.Thread(target=self.download_in_background, args=(folder, file_name, url_prefix), daemon=True).start()

    def download_in_background(self, folder, file_name, url_prefix):
        dest_file_path = os.path.join(res.data_path(folder), file_name)
        try:
            loader.download_file(dest_file_path, url_prefix + file_name)
        except Exception:
            traceback.print_exc()
            Clock.schedule_once(lambda dt: self.on_download_failed(file_name))
            return
        Clock.schedule_once(lambda dt: self.on_file_downloaded(dest_file_path))

    def on_download_failed(self, file_name):
        # the game can not start without the file, the next start continues from this file
        if self.downloading_popup:
            self.downloading_popup.title = f'failed to download {file_name}'

    def on_file_downloaded(self, file_path):
        self.progress_bar.value += 1
        if self.downloading_list:
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
        catalog.load_figures(figures_file_name=res.catalog_path('figures.json'))
        catalog.load_animations(animations_file_name=res.catalog_path('animations.json'))
        catalog.load_armors(armors_file_name=res.catalog_path('armors.json'))
        catalog.load_weapons(weapons_file_name=res.catalog_path('weapons.json'))
        catalog.load_materials(materials_file_name=res.catalog_path('materials.json'))
        self.catalog = catalog
        self.land = None
        self.scene = None
        self.renderer = None
        self.models_popup = None
        self.server_name = None
        self.biomes = []
        # the land comes from the game server: the archive of the biome is downloaded before anything is rendered,
        # then the scene is rendered only after all models used in the biome are on the device
        self.game_client = cli.GameClient(
            url=os.environ.get('WORLDS_SERVER_URL'),
            on_welcome=self.on_welcome,
            on_server_info=self.on_server_info,
            on_biome=self.on_biome,
            on_biome_progress=self.on_biome_progress,
        )
        self.show_models_popup(title='connecting to the game server...')
        self.game_client.start()

    #--- BIOMES ---

    def on_server_info(self, server_name, biomes):
        self.server_name = server_name
        self.biomes = biomes
        if not biomes:
            self.show_models_popup(title=f'game server {server_name} has no biomes')
            return
        # the game always starts in the first biome of the server
        self.enter_biome(biomes[0].id)

    def enter_biome(self, biome_id):
        self.game_client.enter(biome_id)
        self.close_scene()
        name = next((b.name for b in self.biomes if b.id == biome_id), biome_id)
        self.show_models_popup(title=f'entering {name}...')

    def next_biome(self):
        # debug: moves the hero into the next biome of the server
        if len(self.biomes) < 2:
            if _Debug:
                print('the game server has only one biome')
            return
        ids = [b.id for b in self.biomes]
        current = self.game_client.biome_id or self.game_client.entering
        index = (ids.index(current) + 1) % len(ids) if current in ids else 0
        self.enter_biome(ids[index])

    def close_scene(self):
        if self.renderer:
            self.renderer.stop()
        if self.scene:
            self.scene.stop()
        self.root.clear_widgets()
        self.scene = None
        self.renderer = None
        self.land = None
        # objects of the previous biome were moved out of the garbage collector reach, see build_scene()
        gc.unfreeze()
        gc.collect()

    def on_biome_progress(self, done, total):
        self.show_models_popup(title=f'downloading the land... {done}/{total}', maximum=total)
        self.models_progress_bar.value = done

    def on_biome(self, location_id, biome_dir, error):
        if error or not biome_dir:
            self.show_models_popup(title=f'failed to receive the land: {error}')
            return
        self.show_models_popup(title='loading the land...')
        # let the popup be drawn before the land is loaded
        Clock.schedule_once(lambda dt: self.load_land(biome_dir))

    def load_land(self, biome_dir):
        land = dat.LandData()
        land.load_heightmap_file(heightmap_file_name=os.path.join(biome_dir, 'heightmap.png'))
        land.load_tilemap_file(tilemap_file_name=os.path.join(biome_dir, 'encoded.png'), tiles_file_name=os.path.join(biome_dir, 'tiles.json'))
        land.load_cache_tiles_textures(textures_dir_path=res.data_path('textures/land'))
        land.load_plants_data(plants_data_file_name=os.path.join(biome_dir, 'plants.json'))
        land.load_buildings_data(buildings_data_file_name=os.path.join(biome_dir, 'buildings.json'))
        self.land = land
        if _Debug:
            print(f'loaded the land of biome {self.game_client.location_id} from {biome_dir}')
        self.game_client.biome_loaded()

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
        world_templates = []
        for t in templates:
            # [template, store_id], older servers send only the name of the template
            if isinstance(t, str):
                world_templates.append((t, models.DEFAULT_STORE_ID))
            else:
                world_templates.append((t[0], t[1]))
        files = loader.missing_files(templates=world_templates, textures=self.world_textures())
        if _Debug:
            print(f'on_welcome missing files: {files}')
        store_loader = loader.StoreLoader(files, on_progress=self.on_models_progress, on_done=self.on_models_ready)
        if files:
            self.show_models_popup(title='downloading models...', maximum=len(files))
        store_loader.start()

    def world_textures(self):
        # textures of buildings and plants, the land is rendered while the camera moves so all of them are needed
        # before the scene is created, those are taken from the default store
        textures = set()
        for building_list in self.land.buildings_map_data.values():
            for building in building_list:
                textures.add(building['t'])
        for plant_variant in self.land.plants_variants.values():
            textures.add(plant_variant['t'])
        return sorted((t, models.DEFAULT_STORE_ID) for t in textures if t)

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
        self.known_templates = json.loads(open(res.catalog_path('figures_samples.json'), 'rt').read())
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


def catalog_path():
    # in the Android package catalog/ is next to main.py, see installer/android/Makefile,
    # when running from the repository it is in the root folder: src/client/main.py -> catalog/
    for path in (os.path.join(ROOT_PATH, 'catalog'), os.path.join(ROOT_PATH, '..', '..', 'catalog')):
        if os.path.isdir(path):
            return os.path.abspath(path)
    raise Exception(f'catalog folder not found near {ROOT_PATH}')


def main():
    res.DATA_PATH = system.get_app_data_path()
    res.CATALOG_PATH = catalog_path()
    if _Debug:
        print(f'data path: {res.DATA_PATH}')
        print(f'root path: {ROOT_PATH}')
        print(f'catalog path: {res.CATALOG_PATH}')
    if not os.path.exists(res.DATA_PATH):
        os.makedirs(res.DATA_PATH)
    WorldsApp().run()


if __name__ == '__main__':
    main()
