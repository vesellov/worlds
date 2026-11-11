"""
Reader of map.txt of texts.res: the zones of the game, game.exe 45ea20.

A zone starts with "#zone <name> <allod> <type>" and has blocks like "#res" (the .mpr and .mob files),
"#maps" (the size in sectors "Xsize Ysize", then "MinimapTexture ObjTexture"), "#sky" (normal or cave), "#weather" (none, rain or snow), "#exit N" and others.

"#exit N" opens an entrance of the zone: the line after it names the neighbour zone and its exit ("none 0" for the
entrance of the start of the game), "#deploy x1 y1 x2 y2" is the rectangle the party appears in (its center),
"#remove" the rectangle which takes the party out, "#passtime" the hours of the way, "#view" the angle of the camera
and "#deployangle" the angle of the party in degrees (game.exe 45ea20 keeps them in radians, the record of 0x3c bytes:
+0 passtime, +4 deploy, +0x14 remove, +0x30 view, +0x34 has deployangle, +0x38 deployangle). Without "#deployangle"
the party looks from the entrance to the middle of the map (590700 before 58ef20).
"##" starts a comment. The allod (gipat, ingos, suslanger) and the sky choose the lights of the map:
config/Lights[Cave]<allod>.ini (game.exe 567ab0).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math

#------------------------------------------------------------------------------

ENCODING = 'cp1251'

DEFAULT_ALLOD = 'gipat'

SKY_NORMAL = 'normal'
SKY_CAVE = 'cave'

# the blocks of map.txt which belong to the last "#exit N"
EXIT_BLOCKS = ('exit', 'deploy', 'remove', 'passtime', 'view', 'deployangle')

#------------------------------------------------------------------------------


class Allod(object):
    """
    The block "#allod <name>" of map.txt: the 3D models of its markers, in the order of the game
    (the arrow of the current zone, the ball of a quest, the skull of a failed quest, the down arrow
    and the armor of a shop; see FUN_0064cbd0).
    """

    def __init__(self, name, model_names=()):
        self.name = name
        self.model_names = list(model_names)

    def __repr__(self):
        return 'Allod(%s, %d models)' % (self.name, len(self.model_names))

#------------------------------------------------------------------------------


class Zone(object):

    def __init__(self, name, allod, zone_type):
        self.name = name
        self.allod = allod
        self.type = zone_type
        self.mpr_name = None
        self.mob_name = None
        self.sky = SKY_NORMAL
        self.weather = 'rain'
        self.sectors = None
        self.minimap_texture_name = None
        self.quest_texture_name = None
        self.exits = []

    def __repr__(self):
        return 'Zone(%s, %s, %s, %s)' % (self.name, self.allod, self.mpr_name, self.sky)

    def exit(self, number):
        for zone_exit in self.exits:
            if zone_exit.number == number:
                return zone_exit
        return None

    @property
    def is_cave(self):
        return self.sky == SKY_CAVE

    def lights_file_name(self):
        return 'lights%s%s.ini' % ('cave' if self.is_cave else '', self.allod or DEFAULT_ALLOD)

#------------------------------------------------------------------------------


class Exit(object):
    """
    An entrance of a zone: where the party appears when it comes from the neighbour zone.
    """

    def __init__(self, number):
        self.number = number
        self.neighbour_name = None
        self.neighbour_exit = 0
        self.deploy = None
        self.remove = None
        self.pass_time = 0.0
        self.view = 0.0
        self.deploy_angle = None

    def __repr__(self):
        return 'Exit(%d %s:%d deploy %s)' % (self.number, self.neighbour_name, self.neighbour_exit, self.deploy)

    @property
    def deploy_center(self):
        if self.deploy is None:
            return None
        x1, y1, x2, y2 = self.deploy
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    def deploy_yaw(self, map_width, map_height):
        """
        The angle of the party in radians (the angle of its rotation around z, as OBJROTATION of a .mob):
        "#deployangle" or the look from the entrance to the middle of the map, atan2 - pi/2 of 590700.
        """
        if self.deploy_angle is not None:
            return math.radians(self.deploy_angle)
        x, y = self.deploy_center
        return math.atan2(y - map_height * 0.5, x - map_width * 0.5) - math.pi * 0.5

    def contains_remove(self, x, y):
        """
        The point is in the #remove rectangle of the entrance: the party walks out of the zone there.
        """
        if self.remove is None:
            return False
        x1, y1, x2, y2 = self.remove
        return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)

    @property
    def camera_yaw(self):
        """
        The angle of the camera: #view is the direction it looks, the camera stands behind the target.
        """
        return (self.view + 180.0) % 360.0

#------------------------------------------------------------------------------


def parse_allods(data):
    """
    Allod of map.txt (the "#allod <name>" blocks and the model names under them).
    """
    text = data if isinstance(data, str) else bytes(data).decode(ENCODING, errors='replace')
    allods = []
    current = None
    for raw_line in text.splitlines():
        line = raw_line.split('//', 1)[0].strip()
        if not line or line.startswith('##'):
            continue
        if line.startswith('#'):
            words = line[1:].split()
            current = None
            if words and words[0].lower() == 'allod' and len(words) >= 2:
                current = Allod(words[1].lower())
                allods.append(current)
            continue
        if current is not None:
            current.model_names.append(line.split()[0].lower())
    return allods


def parse(data):
    text = data if isinstance(data, str) else bytes(data).decode(ENCODING, errors='replace')
    zones = []
    zone = None
    block = None
    block_lines = []
    for raw_line in text.splitlines():
        line = raw_line.split('//', 1)[0].strip()
        if not line or line.startswith('##'):
            continue
        if line.startswith('#'):
            _finish_block(zone, block, block_lines)
            words = line[1:].split()
            block, block_lines = words[0].lower(), []
            if block == 'zone' and len(words) >= 2:
                zone = Zone(words[1].lower(), words[2].lower() if len(words) > 2 else None,
                            words[3].lower() if len(words) > 3 else None)
                zones.append(zone)
            elif block == 'allod':
                zone = None
            elif block == 'exit' and zone is not None and len(words) >= 2:
                try:
                    zone.exits.append(Exit(int(words[1])))
                except ValueError:
                    pass
            continue
        block_lines.append(line)
    _finish_block(zone, block, block_lines)
    if _Debug:
        print('map.txt: %d zones' % len(zones))
    return zones


def remove_exit(zone, x, y):
    """
    The entrance of a zone whose #remove rectangle holds the point, None when the point walks out of no
    entrance (590700: the party leaves the zone through it).
    """
    if zone is None:
        return None
    for zone_exit in zone.exits:
        if zone_exit.contains_remove(x, y):
            return zone_exit
    return None


def _finish_block(zone, block, lines):
    if zone is None or not lines:
        return
    words = lines[0].split()
    if block == 'res' and words:
        zone.mpr_name = words[0].lower()
        zone.mob_name = words[1].lower() if len(words) > 1 else zone.mpr_name
    elif block == 'maps' and len(words) >= 2:
        try:
            zone.sectors = (int(words[0]), int(words[1]))
        except ValueError:
            pass
        if len(lines) > 1:
            names = lines[1].split()
            zone.minimap_texture_name = names[0].lower() if names else None
            zone.quest_texture_name = names[1].lower() if len(names) > 1 else None
    elif block == 'sky' and words:
        zone.sky = words[0].lower()
    elif block == 'weather' and words:
        zone.weather = words[0].lower()
    elif zone.exits and block in EXIT_BLOCKS:
        _finish_exit_block(zone.exits[-1], block, words)


def _finish_exit_block(zone_exit, block, words):
    try:
        if block == 'exit' and words:
            zone_exit.neighbour_name = words[0].lower()
            zone_exit.neighbour_exit = int(words[1]) if len(words) > 1 else 0
        elif block in ('deploy', 'remove') and len(words) >= 4:
            setattr(zone_exit, block, tuple(float(w) for w in words[:4]))
        elif block == 'passtime':
            zone_exit.pass_time = float(words[0])
        elif block == 'view':
            zone_exit.view = float(words[0])
        elif block == 'deployangle':
            zone_exit.deploy_angle = float(words[0])
    except ValueError:
        pass
