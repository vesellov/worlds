"""
Access to the files of the original game: the archives in res/ and maps/ are opened once
and the parsed objects are cached.

The folder of the game is given by the environment variable EI_DATA_PATH,
it is the folder with game.exe, res/ and maps/.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import os
import collections

#------------------------------------------------------------------------------

from eilib import res
from eilib import fig
from eilib import skel
from eilib import anm
from eilib import adb
from eilib import mmp
from eilib import mapdata
from eilib import mob
from eilib import reg
from eilib import cam
from eilib import idb
from eilib import perkflags
from eilib import quickdb
from eilib import zones

#------------------------------------------------------------------------------

DEFAULT_DATA_PATH = '/run/media/veselin/STUFF/evil_islands/EvilIslands/ENG/game_info/data'

# the models of the units and objects, then the stone columns of the main and the escape menus
FIGURE_ARCHIVES = ('res/figures.res', 'res/menus.res')

SFX_ARCHIVE = 'res/sfx.res'

#------------------------------------------------------------------------------


def resolve_path(dir_path, relative_path):
    """
    The path of relative_path ("res/texts.res") under dir_path, every part of it compared without the case
    when the exact one is missing: the add-on keeps its data in "Res/" and "Maps/".
    """
    path = dir_path
    for part in relative_path.replace('\\', '/').split('/'):
        exact = os.path.join(path, part)
        if not os.path.exists(exact) and os.path.isdir(path):
            for name in os.listdir(path):
                if name.lower() == part.lower():
                    exact = os.path.join(path, name)
                    break
        path = exact
    return path

#------------------------------------------------------------------------------


class GameData(object):

    def __init__(self, data_dir_path=None):
        self.data_dir_path = data_dir_path or os.environ.get('EI_DATA_PATH') or DEFAULT_DATA_PATH
        if not os.path.isdir(resolve_path(self.data_dir_path, 'res')):
            raise IOError('game data not found in %r, set EI_DATA_PATH' % self.data_dir_path)
        self._archives = {}
        self._cache = {}

    def archive(self, archive_name):
        """
        archive_name is relative to the data folder, for example "res/figures.res" or "maps/bz10k.mpr".
        """
        if archive_name not in self._archives:
            self._archives[archive_name] = res.ResArchive.open(resolve_path(self.data_dir_path, archive_name))
        return self._archives[archive_name]

    def cached(self, key, loader):
        if key not in self._cache:
            self._cache[key] = loader()
        return self._cache[key]

    @property
    def figures(self):
        return self.archive(FIGURE_ARCHIVES[0])

    def figures_archive(self, model_name):
        """
        The archive with the .mod, .bon and .anm of a model, None when no archive has it
        (CFigureManager::LoadModelResources 5aa5f0).
        """
        for archive_name in FIGURE_ARCHIVES:
            archive = self.archive(archive_name)
            if model_name + '.mod' in archive:
                return archive
        return None

    def loose_figure_archive(self, model_name):
        """
        The archive with a figure lying alone as <name>.fig and <name>.bon, like the heads of the portraits
        "infaunhumame1face", None when there is none (LoadLooseFigure 5ac1a0).
        """
        for archive_name in FIGURE_ARCHIVES:
            archive = self.archive(archive_name)
            if model_name + '.fig' in archive:
                return archive
        return None

    def linked_figure_archive(self, model_name):
        """
        The archive with a model lying as <name>.lnk and its parts as <name><part>.fig and .bon, like the arrow
        "efar1" (efar1arrow.fig), None when there is none.
        """
        for archive_name in FIGURE_ARCHIVES:
            archive = self.archive(archive_name)
            if model_name + '.lnk' in archive:
                return archive
        return None

    def has_model(self, model_name):
        return (self.figures_archive(model_name) is not None or self.loose_figure_archive(model_name) is not None or
                self.linked_figure_archive(model_name) is not None)

    def model_names(self):
        names = set()
        for archive_name in FIGURE_ARCHIVES:
            names.update(name[:-4] for name in self.archive(archive_name).names_with_suffix('.mod'))
        return sorted(names)

    def model_archive(self, model_name):
        return self.cached(('mod', model_name), lambda: self.figures_archive(model_name).archive(model_name + '.mod'))

    def parts_tree(self, model_name):
        """
        The parts of the .mod of a model, a figure lying alone is a model of one part with its own name
        (the .lnk: CFigureManager::LoadSkeleton 5ac770).
        """
        def _load():
            linked = self.linked_figure_archive(model_name) if self.figures_archive(model_name) is None else None
            if linked is not None:
                return skel.parse_lnk(linked.read(model_name + '.lnk'))
            if self.figures_archive(model_name) is None and self.loose_figure_archive(model_name) is not None:
                tree = skel.PartsTree()
                tree.add(model_name, None)
                return tree
            return skel.parse_lnk(self.model_archive(model_name).read(model_name))
        return self.cached(('lnk', model_name), _load)

    def part_figure(self, model_name, part_name):
        """
        The .fig of a part: 5ac880 inside the .mod, 5ac2f0 for "<model><part>.fig" lying alone.
        """
        def _load():
            if self.figures_archive(model_name) is None:
                linked = self.linked_figure_archive(model_name)
                if linked is not None:
                    return fig.parse(linked.read(model_name + part_name + '.fig'))
                return fig.parse(self.loose_figure_archive(model_name).read(part_name + '.fig'))
            return fig.parse(self.model_archive(model_name).read(part_name))
        return self.cached(('fig', model_name, part_name), _load)

    def bones(self, model_name):
        """
        part name -> (variants, 3) offsets of the part from its parent.
        """
        def _load():
            figures = self.figures_archive(model_name)
            linked = self.linked_figure_archive(model_name) if figures is None else None
            if linked is not None:
                tree = skel.parse_lnk(linked.read(model_name + '.lnk'))
                return {name: skel.parse_bon(linked.read(model_name + name + '.bon'))
                        for name in tree.order if model_name + name + '.bon' in linked}
            loose = self.loose_figure_archive(model_name) if figures is None else None
            if loose is not None and model_name + '.bon' in loose:
                return {model_name: skel.parse_bon(loose.read(model_name + '.bon'))}
            if figures is None or model_name + '.bon' not in figures:
                return {}
            archive = figures.archive(model_name + '.bon')
            return {name: skel.parse_bon(archive.read(name)) for name in archive.names}
        return self.cached(('bon', model_name), _load)

    def clips_archive(self, model_name):
        def _load():
            figures = self.figures_archive(model_name)
            if figures is None or model_name + '.anm' not in figures:
                return None
            return figures.archive(model_name + '.anm')
        return self.cached(('anm', model_name), _load)

    def clip(self, model_name, clip_name):
        """
        part name -> anm.Track, None when the model has no such clip (CFigureManager::LoadAnimation 5accf0).
        """
        def _load():
            clips = self.clips_archive(model_name)
            if clips is None or clip_name not in clips:
                return None
            archive = clips.archive(clip_name)
            return {name: anm.parse(archive.read(name)) for name in archive.names}
        return self.cached(('clip', model_name, clip_name), _load)

    def animation_database(self, model_name):
        def _load():
            database = self.archive('res/database.res')
            if model_name + '.adb' not in database:
                return None
            return adb.parse(database.read(model_name + '.adb'))
        return self.cached(('adb', model_name), _load)

    def items(self):
        """
        idb.ItemDatabase: materials, weapon and armor shapes of items.idb.
        """
        return self.cached(('idb', 'items'), lambda: idb.parse_items(self.archive('res/database.res').read('items.idb')))

    def quests(self):
        """
        name -> idb.Quest of quests.qdb.
        """
        return self.cached(('qdb', ), lambda: idb.parse_quests(self.archive('res/database.res').read('quests.qdb')))

    def quick_items(self):
        """
        eilib.quickdb.QuickItemDatabase: potions, wands and spell containers of items.idb.
        """
        return self.cached(('idb', 'quick items'),
                           lambda: quickdb.parse_quick_items(self.archive('res/database.res').read('items.idb')))

    def acks(self):
        """
        The name of a prototype -> idb.Acks of acks.db: the phrases of the units.
        """
        return self.cached(('acks', ), lambda: idb.parse_acks(self.archive('res/database.res').read('acks.db')))

    def units(self):
        """
        idb.UnitDatabase: unit types and prototypes of units.udb.
        """
        return self.cached(('idb', 'units'), lambda: idb.parse_units(self.archive('res/database.res').read('units.udb')))

    def perks(self):
        """
        idb.PerkDatabase: the skills and the perks of perks.pdb; flags of the perks in the camp (tag 22).
        """
        def _load():
            data = self.archive('res/database.res').read('perks.pdb')
            database = idb.parse_perks(data)
            database.flags = perkflags.parse_flags(data)
            return database
        return self.cached(('idb', 'perks'), _load)

    def spells(self):
        """
        idb.SpellDatabase: spell prototypes, modifiers and templates of spells.sdb.
        """
        return self.cached(('idb', 'spells'), lambda: idb.parse_spells(self.archive('res/database.res').read('spells.sdb')))

    def sound_names(self):
        """
        The names of the sounds of sfx.res by their lowercase form with "/": game.exe writes them with "\\"
        and in other cases ("monsters\\ogre\\attack\\1.wav" is "monsters/Ogre/Attack/1.wav").
        """
        return self.cached(('sfx', ), lambda: {n.lower(): n for n in self.archive(SFX_ARCHIVE).names})

    def sound_name(self, name):
        """
        The name of a sound in sfx.res as the game writes it, None when there is no such file.
        """
        return self.sound_names().get(name.replace('\\', '/').lower())

    def sound_count(self, folder_name):
        """
        The number of the sounds 1.wav, 2.wav... in a folder of sfx.res (the steps of 534590 pick one of them).
        """
        prefix = folder_name.replace('\\', '/').lower().rstrip('/') + '/'
        def _count():
            return sum(1 for name in self.sound_names() if name.startswith(prefix) and '/' not in name[len(prefix):])
        return self.cached(('sfx count', prefix), _count)

    def sound(self, name):
        """
        The bytes of the .wav file of a sound, None when there is no such file.
        """
        real_name = self.sound_name(name)
        if real_name is None:
            return None
        return self.archive(SFX_ARCHIVE).read(real_name)

    def zones(self):
        """
        zones.Zone of map.txt in texts.res.
        """
        return self.cached(('zones', ), lambda: zones.parse(self.archive('res/texts.res').read('map.txt')))

    def zone_of_map(self, map_name):
        """
        The first zone which loads the .mpr of the map, None when map.txt does not name it.
        """
        for zone in self.zones():
            if zone.mpr_name == map_name.lower():
                return zone
        return None

    def config_text(self, file_name):
        """
        Text of a file of the config/ folder, None when there is none. The names are compared without the case.
        """
        config_dir_path = resolve_path(self.data_dir_path, 'config')
        if not os.path.isdir(config_dir_path):
            return None
        for name in os.listdir(config_dir_path):
            if name.lower() == file_name.lower():
                with open(os.path.join(config_dir_path, name), 'rb') as f:
                    return f.read().decode('cp1251', errors='replace')
        return None

    def texture(self, texture_name):
        """
        (height, width, 4) RGBA of a texture from textures.res or redress.res, None when there is none.
        """
        def _load():
            file_name = texture_name if texture_name.lower().endswith('.mmp') else texture_name + '.mmp'
            for archive_name in ('res/textures.res', 'res/redress.res', 'res/menus.res'):
                archive = self.archive(archive_name)
                if file_name in archive:
                    return mmp.decode(archive.read(file_name))
            return None
        return self.cached(('mmp', texture_name.lower()), _load)

    def texture_names(self, prefix=''):
        names = []
        for archive_name in ('res/textures.res', 'res/redress.res'):
            names.extend(n[:-4] for n in self.archive(archive_name).names if n.lower().startswith(prefix.lower()))
        return sorted(names)

    def registry(self, file_name):
        """
        Sections of a binary .reg: from an archive of res/ ("menus.reg" is in menus.res) or a file of the data folder
        like "config/ai.reg" or "res/music.reg" ("tileDesc.reg" is in aiinfo.res). None when there is none.
        """
        def _load():
            file_path = resolve_path(self.data_dir_path, file_name)
            if os.path.isfile(file_path):
                with open(file_path, 'rb') as f:
                    return reg.parse(f.read())
            for archive_name in ('res/menus.res', 'res/aiinfo.res'):
                archive = self.archive(archive_name)
                if file_name in archive:
                    return reg.parse(archive.read(file_name))
            return None
        return self.cached(('reg', file_name.lower()), _load)

    def camera(self, camera_name):
        """
        cam.CameraKey list of camera/<name>.cam, None when there is none.
        """
        def _load():
            camera_dir_path = resolve_path(self.data_dir_path, 'camera')
            if not os.path.isdir(camera_dir_path):
                return None
            for name in os.listdir(camera_dir_path):
                if name.lower() == camera_name.lower() + '.cam':
                    with open(os.path.join(camera_dir_path, name), 'rb') as f:
                        return cam.parse(f.read())
            return None
        return self.cached(('cam', camera_name.lower()), _load)

    def text(self, text_name):
        """
        A text of texts.res like "string game_paused", None when there is none.
        """
        def _load():
            archive = self.archive('res/texts.res')
            if text_name not in archive:
                return None
            return bytes(archive.read(text_name)).decode('cp1251', errors='replace')
        return self.cached(('text', text_name), _load)

    def map_names(self):
        maps_dir_path = resolve_path(self.data_dir_path, 'maps')
        return sorted(file_name[:-4] for file_name in os.listdir(maps_dir_path) if file_name.lower().endswith('.mpr'))

    def map_mob(self, map_name):
        """
        Root mob.Node of the objects of the map, None when the map has no .mob file.
        """
        def _load():
            mob_file_path = resolve_path(self.data_dir_path, 'maps/' + map_name + '.mob')
            if not os.path.isfile(mob_file_path):
                return None
            with open(mob_file_path, 'rb') as f:
                return mob.parse(f.read())
        return self.cached(('mob', map_name), _load)

    def model_textures(self):
        """
        model name -> texture names in the order of how often the maps use them with that model.
        The textures of the monsters are not in their models, the objects of the maps choose them.
        """
        def _load():
            counts = collections.defaultdict(collections.Counter)
            for map_name in self.map_names():
                root = self.map_mob(map_name)
                if root is None:
                    continue
                for name in ('OBJECT', 'UNIT'):
                    for node in root.walk(name):
                        if node.children is None:
                            continue
                        template, texture = node.get('OBJTEMPLATE'), node.get('OBJPRIMTXTR')
                        if template and texture:
                            counts[template.lower()][texture.lower()] += 1
            return {model: [t for t, _ in c.most_common()] for model, c in counts.items()}
        return self.cached(('model_textures', ), _load)

    def map_header(self, map_name):
        return self.cached(('mp', map_name), lambda: mapdata.parse_mp(self.archive('maps/%s.mpr' % map_name).read(map_name + '.mp')))

    def map_sector(self, map_name, x, y):
        def _load():
            archive = self.archive('maps/%s.mpr' % map_name)
            sector_name = mapdata.sector_name(map_name, x, y)
            if sector_name not in archive:
                return None
            return mapdata.parse_sec(archive.read(sector_name), self.map_header(map_name).max_altitude)
        return self.cached(('sec', map_name, x, y), _load)
