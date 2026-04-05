import os
import sys
import json


_Debug = True


from kivy.config import Config
# Config.set('graphics', 'window_state', 'maximized')

from kivy.app import App

import res
import rend
import dat
import scen


class AppRoot(App):

    known_templates = {}

    def prepare_test_unit(self, scene, test=0, template=None, save_json=False):
        if test == -1:
            template = template or 'unmogo'
            if template not in scene.models:
                m = dat.ModelData()
                m.unpack_figure_data('data/figures.res', 'models', template=template, save_json=save_json)
                scene.add_model_template(template, m)
            return scene.create_object_data_from_model_data(
                template=template,
                coefs=[0, 0, 0],
                excluded_parts=['rh3.pike00', ],
                selected_animations='*',
                textures={'*': 'goblin01'},
            )
        elif test == -2:
            template = template or 'unmoba2'
            if template not in scene.models:
                m = dat.ModelData()
                m.unpack_figure_data('data/figures.res', 'models', template=template, save_json=save_json)
                scene.add_model_template(template, m)
            return scene.create_object_data_from_model_data(
                template=template,
                coefs=[0, 0, 0],
                selected_animations='*',
                textures={'*': 'banshee02'},
            )

        elif test == -3:
            template = template or 'unhufe'
            selected_parts = [
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
            ]
            selected_animations = [
                'cwalk02',
                'cidle07',
                'crun01',
                # 'ccrawl01',
                # 'crest',
                # 'cspecial14',
                # 'cwalk01',
                # 'cwalk05',
                # 'uattack01',
                # 'uattack02',
                # 'uattack08',
                # 'ubriefing06',
                # 'ucast03',
                # 'ucross06',
                # 'udeath06',
                # 'uhit14',
                # 'uspecial05',
                # 'udeath15',
            ]
            if template not in scene.models:
                m = dat.ModelData()
                m.unpack_figure_data('data/figures.res', 'models', template=template, selected_parts=selected_parts, selected_animations=selected_animations, save_json=save_json)
                scene.add_model_template(template, m)
            return scene.create_object_data_from_model_data(
                template=template,
                coefs=[0, 0, 0],
                selected_parts=selected_parts,
                selected_animations='*',
                textures={'*': 'unhufeskin_08'},
            )
        else:
            # if self.known_templates is None:
            #     self.known_templates = json.loads(open('models.json', 'rt').read())
                # print('known templates', sorted(self.known_templates.keys()))
            template = template or sorted(self.known_templates.keys())[test]
            model_data = self.known_templates[template][0]
            if _Debug:
                print(f'preparing test unit {test} with template "{template}"')
            selected_parts = []
            selected_animations = []
            if template not in scene.models:
                m = dat.ModelData()
                m.unpack_figure_data('data/figures.res', 'models', template=template, save_json=save_json)
                if not os.path.isfile('textures/model/' + model_data['t']+'.png'):
                    m.unpack_texture('data/textures.res', 'textures/model', model_data['t'])
                scene.add_model_template(template, m)
                selected_parts = selected_parts or res.flat_tree(m.links[template]['ordered'])
                selected_animations = selected_animations or list(m.animations.keys())
            else:
                m = scene.models[template]
            return scene.create_object_data_from_model_data(
                template=template,
                coefs=model_data['c'],
                selected_parts=selected_parts or model_data['p'],
                selected_animations=list(m.animations.keys()),
                textures={'*': model_data['t']},
            )
        return None

    def build(self):
        land = dat.LandData()
        land.load_heightmap_file(heightmap_file_name='assets/heightmap.png', sea_level=0.001)
        land.load_tilemap_file(tilemap_file_name='assets/encoded.png')
        land.load_cache_tiles_textures(textures_dir_path='assets/land')
        land.load_plants_data(plants_data_file_name='assets/trees.json')
        # land.save_elevation_memmap('island4_heightmap', destination_dir='.')
        scene = scen.Scene(land=land)
        scene.calculate_land_vertices()
        renderer = rend.Renderer(app_root=self, scene=scene)
        self.known_templates = json.loads(open('assets/models.json', 'rt').read())
        scene.renderer = renderer
        self.test_id = 299
        # unit = self.prepare_test_unit(scene=scene, test=self.test_id)
        # if unit:
        #     if unit.animations_loaded:
        #         unit.animation_playing = unit.animations_loaded[0]
        #     renderer.add_unit(unit.name)
        # self.prepare_trees(scene)
        scene.prepare_land(277, 829, land.width, land.height) # int(land.width / 2), int(land.height / 2),
        return renderer


def main():
    res.download_res_file('data', 'figures.res', ['figures_res_0', 'figures_res_1', ])
    res.download_res_file('data', 'textures.res', ['textures_res_0', 'textures_res_1', 'textures_res_2', ])
    res.download_res_file('data', 'redress.res', ['redress_res_0', 'redress_res_1', ])
    AppRoot().run()


if __name__ == '__main__':
    main()
