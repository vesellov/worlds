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


class AppRoot(App):

    known_templates = None
    # known_templates = '''unhufe,unhuma,ingm3,unorfe,unorma,unmoog,unmosk2,stwa35,unmodg,stwa1,unmocu,unmosu,stwa34,stst20,unmotr,jstatue00,unmogo,stbuho36,jbuho00,fireztower00,jvhouse00,nafltr66,unmori,unanwiba,unanhoco,unanwiti,unanhodo,unanwiwo,unanwihy,unmozo1,goldpile00,stbr10,stst154,unmosp,stbuho59,unmocy,unmoog2,unmoog1,unmodr,unmoli,unmoto,stbuho15,unanwide,stbuho12,unanwibo,stbuho30,stbuho60,stst67,jstatue01,unmozo0,stst106,stst107,unanhopi,stst131,stst136,stst138,stst135,stst137,stst134,stst160,stbuho46,stst19,unmosk,unanwira,unanwigi,stst44,jbr00,unmoun,nafltr76,unanhoho,stbuho64,stst114,nafltr67,stbr19,stst56,jigranwalls1,jiganwalls1,unmomi,unanhoha,jprops00,unmoba2,unmogo2,unmoba1,stst113,stbuto1,jfield02,unanwicr,stst112,stbuho4,stst40,stbuho47,stbuho50,unmobe1,stbuho57,stst61,unmoel0,unmoel1,unmoel2,unmobe0,stbr17,stbuho22,stbr1,stbuho6,stbuho61,stst001,stst002,stbuho38,stbuho40,stst123,stbuho24,stst122,stbuho32,stbuho3,stbuho29,stst63,stst70,stbuho62,stst50,stst65,stst88,stbuho48,stwa11,stbuho5,stbr14,stst155,stbuho1,stga1,stga8,nafltr65,sttr131,sttr134,sttr133,sttr129,sttr132,sttr135,sttr130,stbuho21,unmogo1,stst42,stbuho2,stst58,nafltr75,jfobj00,stst8,stst141,stbr15,co3,stbuho55,stst148,stst125,stst116,stst84,stwa4,stst57,stga4,stbuho28,stbr18,sttr144,sttr143,sttr142,sttr145,sttr147,sttr146,stbuho56,stbuho14,nafltr60,sttr136,sttr139,sttr138,sttr140,sttr137,sttr141,stst108,stst33,stst92,stst32,stwa3,stst97,stst51,stbuho7,stst38,stbuho27,stst101,stst129,stbuho51,stbr20,stbr2,stst105,nafltr73,stst60,nast21,stbuho63,stbuho23,sttr128,sttr126,sttr127,sttr125,stst26,stst14,stst90,stst41,stst25,stbuho39,unmobi,stst18,stbuho11,stst12,stbuto2,stst144,stga5,stst100,stst149,stst28,stbuho58,stst13,stst23,stst6,stst10,nast16,nast12,nast14,nast10,stst147,stst15,stst24,stst89,stst121,nafltr72,stga6,nafltr59,tentjig,stbuho20,stst62,stst11,stst1,stst30,stst53,stst103,stbuho8,jcrypt00,stwa13,goldchest00,stst91,stbuho53,stbr3,stga7,stbuho19,stwa2,stst54,stst9,stwa10,stga2,stst2,stbuho10,stst119,stbuho18,stst72,stst27,stbuho16,stst34,stst120,nafltr56,nafltr68,nafltr71,stst31,stga3,stbuho34,stst146,stst130,stto8,naflbu11,naflbu14,stbuho54,stst124,nast13,nast15,nast11,nast9,stst39,stbuho33,stst111,stst143,stbuho9,stst115,stst4,stwa12,stbr5,stst128,nafltr78,stwa14,stst55,nafltr57,nafltr69,stst87,nast25,nast23,nast22,nast24,stbuho65,naflbu12,co4,naflbu15,stbuho49,stst16,stbr6,stwe3,stst17,stbr4,stbuho25,nafltr77,stbuho44,stto11,stbuho13,stwa6,stst118,nawa2,stst94,nawa1,stst110,stst117,stbuho17,stst80,unmosh,stwa7,stwa9,stbr16,stto6,stst156,stst93,stbuho52,naflbu7,stbuho67,stst142,stst81,stwe2,unmowi,stbuho26,stbuho41,stbr9,stst95,naflte1,naflte2,naflte3,stst64,stbr8,stbuho37,stbr7,stst76,stbuho43,stbuho42,stst3,nafltr79,stst71,stwe1,stto2,nafltr82,nafltr80,stst45,stwali2,stst29,stst99,jbr03,stst126,stst85,stst36,stst37,stst5,stst52,stst48,stst49,stst153,stst98,stsi5,stbr12,stsi3,jbr01,jbr02,stst132,stst43,stst133,stst157,naflbu19,stst127,stst66,stst159,naflbu10,naflbu13,naflbu6,naflbu9,stst35,nast4,nast2,stwali1,stbr11,nafltr74,nafltr23,nafltr86,stst21,stst59,nafltr81,naflli2,stsi1,stst47,nast18,nast19,stwame1,stst109,stst140,stst102,nast17,stst139,stst46,nast7,nast8,nast5,nast6,nast26,nast27,stst22,stbuho68,stbuho31,nafltr21,nafltr84,nafltr85,nafltr22,stwa8,nafltr83,nafltr70,nafltr20,stto9,stto1,stto5,stbuho45,stst86,stto10,stbuho35,naflbu8,stbr13,stwa15,stst151,stst152,nast20,stst74,stst75,nast3,co1,stst83,nast1,stst79,naflbu16,naflbu20,naflbu17,stst161,stst73,stst68,stwa5,stst69,stsi4,stst145,stst78,stst77,stto7,stst96,stst158,naflbu18,naflbu21,co2,stbuho66,stst82,stst104,stst150,naflli1,stsi2'''.split(',')

    def prepare_test_unit(self, scene, test=0, template=None, save_json=False):
        if test == -1:
            template = template or 'unmogo'
            if template not in scene.models:
                m = dat.ModelData()
                m.unpack_figure_data('figures.res', 'models', template=template, save_json=save_json)
                scene.add_model(template, m)
            return scene.create_unit_from_model_data(
                template=template,
                coefs=[0, 0, 0],
                excluded_parts=['rh3.pike00', ],
                selected_animations=[],
                textures={'*': 'goblin01.png'},
            )
        elif test == -2:
            template = template or 'unmoba2'
            if template not in scene.models:
                m = dat.ModelData()
                m.unpack_figure_data('figures.res', 'models', template=template, save_json=save_json)
                scene.add_model(template, m)
            return scene.create_unit_from_model_data(
                template=template,
                coefs=[0, 0, 0],
                textures={'*': 'banshee02.png'},
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
                m.unpack_figure_data('figures.res', 'models', template=template, selected_parts=selected_parts, selected_animations=selected_animations, save_json=save_json)
                scene.add_model(template, m)
            return scene.create_unit_from_model_data(
                template=template,
                coefs=[0, 0, 0],
                selected_parts=selected_parts,
                selected_animations=selected_animations,
                textures={'*': 'unhufeskin_08.png'},
            )
        else:
            if self.known_templates is None:
                self.known_templates = json.loads(open('models.json', 'rt').read())
                print('known templates', sorted(self.known_templates.keys()))
            template = template or sorted(self.known_templates.keys())[test]
            model_data = self.known_templates[template][0]
            if _Debug:
                print(f'preparing test unit {test} with template "{template}"')
            selected_parts = []
            selected_animations = []
            if template not in scene.models:
                m = dat.ModelData()
                m.unpack_figure_data('figures.res', 'models', template=template, save_json=save_json)
                if not os.path.isfile('textures/model/' + model_data['t']+'.png'):
                    m.unpack_texture('textures.res', 'textures/model', model_data['t'])
                scene.add_model(template, m)
                selected_parts = selected_parts or res.flat_tree(m.links[template]['ordered'])
                selected_animations = selected_animations or list(m.animations.keys())
            return scene.create_unit_from_model_data(
                template=template,
                coefs=model_data['c'],  # [0, 0, 0],
                selected_parts=selected_parts or model_data['p'],
                selected_animations=[selected_animations[0], ] if selected_animations else [],
                textures={'*': 'textures/model/' + model_data['t']+'.png', },
            )
        return None

    def prepare_trees(self, scene):
        for coord in scene.land.plants_map_data.keys():
            for i in range(len(scene.land.plants_map_data[coord])):
                plant = scene.land.plants_map_data[coord][i]
                template = plant['m']
                coefs = plant['c']
                texture = plant['t']
                if template not in scene.models:
                    m = dat.ModelData()
                    m.unpack_figure_data('figures.res', 'models', template=template)
                    if not os.path.isfile('textures/model/' + texture + '.png'):
                        m.unpack_texture('textures.res', 'textures/model', texture)
                    scene.add_model(template, m)
                so = scene.create_static_object_from_model_data(
                    template=template,
                    coefs=coefs,
                    textures={'*': 'textures/model/' + texture + '.png', },
                )
                scene.land.plants_map_data[coord][i]['so'] = so.name

    def build(self):
        land = dat.LandData()
        land.load_heightmap_file(heightmap_file_name='heightmap.png', sea_level=0.001)
        land.load_tilemap_file(tilemap_file_name='encoded.png')
        land.load_cache_tiles_textures(textures_dir_path='textures/land')
        land.load_plants_data(plants_data_file_name='trees.json')
        # land.save_elevation_memmap('island4_heightmap', destination_dir='.')
        scene = dat.SceneData(land=land)
        renderer = rend.Renderer(app_root=self, scene=scene)
        self.test_id = 299
        # unit = self.prepare_test_unit(scene=scene, test=self.test_id)
        # if unit:
        #     if unit.animations_loaded:
        #         unit.animation_playing = unit.animations_loaded[0]
        #     renderer.add_unit(unit.name)
        # self.prepare_trees(scene)
        renderer.prepare_land(287, 829, land.width, land.height) # int(land.width / 2), int(land.height / 2),
        # renderer.update_land()
        return renderer


if __name__ == '__main__':
    if not os.path.isfile('figures.res'):
        print('Please copy "figures.res" file from Evil Islands Res folder into the current folder')
        sys.exit(-1)
    AppRoot().run()
