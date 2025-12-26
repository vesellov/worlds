import os
import sys


_Debug = True


from kivy.config import Config
# Config.set('graphics', 'window_state', 'maximized')

from kivy.app import App

import rend
import dat


class AppRoot(App):

    def prepare_test_unit(self, scene, test=2, save_json=False):
        template = None
        if test == 1:
            template = 'unmogo'
            if template not in scene.models:
                m = dat.ModelData(template=template)
                m.unpack_figure_data('figures.res', 'models', save_json=save_json)
                scene.add_model(m)
                return scene.create_unit_from_model_data(
                    template=template,
                    coefs=[0, 0, 0],
                    excluded_parts=['rh3.pike00', ],
                    selected_animations=[],
                    textures={'*': 'goblin01.png'},
                )

        if test == 2:
            template = 'unmoba2'
            if template not in scene.models:
                m = dat.ModelData(template=template)
                m.unpack_figure_data('figures.res', 'models', save_json=save_json)
                scene.add_model(m)
                return scene.create_unit_from_model_data(
                    template=template,
                    coefs=[0, 0, 0],
                    textures={'*': 'banshee02.png'},
                )

        if test == 3:
            template = 'unhufe'
            if template not in scene.models:
                m = dat.ModelData(template=template)
                m.unpack_figure_data('figures.res', 'models', save_json=save_json)
                scene.add_model(m)
                return scene.create_unit_from_model_data(
                    template=template,
                    coefs=[0, 0, 0],
                    selected_parts=[
                        'hp',
                        'bd',
                        'hd',
                        'rh1',
                        'rh2',
                        'rh3',
                        'lh1',
                        'lh2',
                        'lh3',
                        'll1',
                        'll2',
                        'll3',
                        'rl1',
                        'rl2',
                        'rl3',
                        'hr.01',
                    ],
                    selected_animations=[
                        'cidle07',
                        'crun01',
                        'ccrawl01',
                        'crest',
                        'cspecial14',
                        'cwalk01',
                        'cwalk02',
                        'cwalk05',
                        'uattack01',
                        'uattack02',
                        'uattack08',
                        'ubriefing06',
                        'ucast03',
                        'ucross06',
                        'udeath06',
                        'uhit14',
                        'uspecial05',
                        'udeath15',
                    ],
                    textures={'*': 'unhufeskin_08.png'},
                )
        return None

    def build(self):
        land = dat.LandData()
        land.load_elevation_file(elevation_file_name='elevation_island.png')
        land.save_elevation_memmap('elevation_island', destination_dir='.')
        scene = dat.SceneData(land=land)
        renderer = rend.Renderer(app_root=self, scene=scene)
        self.test_id = 1
        unit = self.prepare_test_unit(scene=scene, test=self.test_id)
        if unit:
            unit.animation_playing = unit.animations_loaded[0]
            renderer.add_unit(unit.name)
        renderer.add_land()
        return renderer


if __name__ == '__main__':
    if not os.path.isfile('figures.res'):
        print('Please copy "figures.res" file into the current folder')
        sys.exit(-1)
    AppRoot().run()
