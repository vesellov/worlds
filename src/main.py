import os
import sys
import json

from kivy.config import Config
from kivy.core.window import Window
from kivy.app import App

import res
import rend
import dat
import scen

Window.size = (1400, 700)
Window.top = 100
Window.left = 100

_Debug = True


class AppRoot(App):

    known_templates = {}
    known_figures_parts = {}

    def create_human_hero(self, scene):
        # selected_animations = '*'
        # [
            # 'uattack01', 'uattack02', 'uattack03', 'uattack14', 'uattack15', 'uattack16',
            # 'cidle01',
            # 'cwalk02', 'cidle07', 'crun01',
        # ]
        # compiled_models = json.loads(open('repack/EIrepack/compiled.json', 'rt').read())
        # m = compiled_models['zone9 Ogre']
        # zone9 Human Mage1 F
        # zone9 Orc Archer1 F
        # zone9 Orc Fighter1 M
        # zone35 Undead Archer M
        # zone3 Human Archer1 F
        # zone23 Human Fighter3 M
        # zone23 Human Fighter3 M
        # zone2 Human Fighter2 M
        template = scene.catalog.build_template_data(
            model_name='unhuma',
            skin=41,
            hair=0,
            wears=['hadagan medium pants.wool', ],
            weapon='cheat dagger.bronze',
        )
        selected_parts = [p.replace('.hidden', '') for p in template['parts']]
        hidden_parts = [p.replace('.hidden', '') for p in template['parts'] if p.endswith('.hidden')]        
        unit = scene.place_animated_unit_on_land(
            template=template['model_name'],
            coefs=[0.5, 0.5, 0.5],
            # scale=scale,
            map_w=140,
            map_h=244,
            shift_w=0.5,
            shift_h=0.5,
            direction=0, # random.randint(0, 360),
            # elevation_correction=-5.0,
            selected_parts=selected_parts,
            hidden_parts=hidden_parts,
            textures=template['textures'],
            single_texture=True,
            selected_animations=template['animations'],
        )
        return unit

    def build(self):
        catalog = dat.CatalogData()
        catalog.load_figures(figures_file_name='catalog/figures.json')
        catalog.load_animations(animations_file_name='catalog/animations.json')
        catalog.load_armors(armors_file_name='catalog/armors.json')
        catalog.load_weapons(weapons_file_name='catalog/weapons.json')
        catalog.load_materials(materials_file_name='catalog/materials.json')
        land = dat.LandData()
        land.load_heightmap_file(heightmap_file_name='assets/heightmap.png')
        land.load_tilemap_file(tilemap_file_name='assets/encoded.png')
        land.load_cache_tiles_textures(textures_dir_path='assets/land')
        land.load_plants_data(plants_data_file_name='assets/plants.json')
        land.load_buildings_data(buildings_data_file_name='assets/buildings.json')
        scene = scen.Scene(land=land, catalog=catalog)
        scene.calculate_land_vertices()
        scene.calculate_scaled_elevation_map()
        renderer = rend.Renderer(app_root=self, scene=scene)
        self.known_templates = json.loads(open('assets/templates.json', 'rt').read())
        # self.known_figures_parts = json.loads(open('assets/catalog_figures.json', 'rt').read())
        scene.renderer = renderer
        scene.init_scene(140,244)
        # scene.init_scene()
        self.create_human_hero(scene)
        return renderer


def main():
    res.download_res_file('data', 'figures.res', ['figures_res_0', 'figures_res_1', ])
    res.download_res_file('data', 'textures.res', ['textures_res_0', 'textures_res_1', 'textures_res_2', ])
    res.download_res_file('data', 'redress.res', ['redress_res_0', 'redress_res_1', ])
    AppRoot().run()


if __name__ == '__main__':
    main()
