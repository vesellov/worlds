import math


#--- CAMERA ---
CAMERA_DISTANCE_TO_CENTER_INITIAL = 10.0

CAMERA_VIEW_CLIP_NEAR = 2.0

CAMERA_VIEW_CLIP_FAR = 720.0

SCALE_INITIAL = 5.0

SCALE_MIN = 0.35

SCALE_MAX = 7.0

SCALE_SPEED_FACTOR = 0.2

ROTATE_SPEED = 1.0

ROTATE_VERTICAL_MIN = 5.0

ROTATE_VERTICAL_MAX = 85.0

ROTATE_VERTICAL_INITIAL = 45


#--- PLANET ---
SEGMENT_SIZE = 5.0

PLANET_EQUATOR_SEGMENTS = 360

PLANET_EQUATOR_LENGTH = SEGMENT_SIZE * PLANET_EQUATOR_SEGMENTS

PLANET_RADIUS = PLANET_EQUATOR_LENGTH / (2.0 * math.pi)

SEGMENT_ANGLE = 360.0 / PLANET_EQUATOR_SEGMENTS

SEGMENT_ANGLE_HALF = SEGMENT_ANGLE / 2.0

SEGMENT_ANGLE_HALF_RADIANS = math.radians(SEGMENT_ANGLE_HALF)

SEGMENT_HALF_SIN = math.sin(SEGMENT_ANGLE_HALF_RADIANS)

SEGMENT_HALF_COS = math.cos(SEGMENT_ANGLE_HALF_RADIANS)

SEGMENT_ANGLE_RADIANS = math.radians(SEGMENT_ANGLE/math.sqrt(2.0))

SEGMENT_SIN = math.sin(SEGMENT_ANGLE_RADIANS)

SEGMENT_COS = math.cos(SEGMENT_ANGLE_RADIANS)

PI_4_SIN = math.sin(math.pi / 4.0)

PI_4_COS = math.cos(math.pi / 4.0)

VISIBLE_AREA_SIZE_SEGMENTS = 40

VISIBLE_AREA_SIZE_SEGMENTS_HALF = int(VISIBLE_AREA_SIZE_SEGMENTS / 2.0)

MODELS_SCALE_FACTOR = 2.0

LAND_MOVE_SPEED = 0.75

LAND_ROTATE_SPEED = 1.0


#--- ELEVATION ---
ELEVATION_FACTOR = PLANET_RADIUS * 0.4

ELEVATION_CORRECTION = 2.0

INPUT_WATER_LEVEL = 20

ELEVATION_UNPACK_EXPONENT = 2.0

ELEVATION_UNPACK_UNDERWATER_FACTOR = 50.0

WATER_LEVEL = 11

WATER_LEVEL_ELEVATION = PLANET_RADIUS + (WATER_LEVEL / 100.0) * ELEVATION_FACTOR


#--- TEXTURES ---
# Groups of textures.res, every group is written into a separate <group>.res file by tools/split_textures_res.py.
# Rules are checked in this order, the first matching regex (against the lower case texture name) wins.
# Textures which do not match any rule go to the "objects" group (props: barrels, chests, carpets and so on).
TEXTURE_GROUPS = [
    # terrain of every zone goes to its own file: zone11000 -> zone11, zone11dun1map -> zone11dun1
    ('zone:', r'^(zone.+?)(\d{3}|map|quest|questm|radar)$'),
    ('terrain', r'^bz\d+[a-z]\d{3}$'),
    ('maps', r'^(basegipat|gipat|gipat2|ingos|jigran|suslanger)(\d{3}|map|mapg|sys)$'),
    ('trees', r'^(tree|oak|bush|mushroom)'),
    ('nature', r'^(stone|sky|earth|ice|vulkan|cave\d|rainsnow)'),
    ('armor', r'^(bt|gl|hl|lg|pl|pt|sh)_\d'),
    ('weapons', r'^(ax|bw|cb|dg|hm|sp|sw)_\d'),
    ('units', r'^un(an|mo)'),
    ('faces', r'^(face|orc_face)'),
    ('creatures', r'^(banshee|bat\d|beholder|bird|boar|cow|crab|cyclop|deer|dog\d|dragon\d|driad\d|eagle|fox|goblin\d'
                  r'|golem\d|hare|horse|hyena|jlion|jogre|jskeleton|jwolf|lizardman|mindflayer|ogre\d|pig|rat\d'
                  r'|skeleton|spider|squirrel|succubus|tiger|toad|troll|unicorn|willowisp|wolf|zombie)'),
    ('items', r'^(litem|qitem|quitem|material|modifier|prototype|mod\d|trap\d)'),
    ('spells', r'^(spell\d|skill\d)'),
    ('cursors', r'^(sm_)?cursor_'),
    ('effects', r'^(acidray|antimagic|astral|bloodprints|curse\d|demonlight|explosion|eye$|fire|footprints|healing$'
                r'|light$|lightning|particles|poisoncloud|protection|rikarrowsmoke|shapechange|shadow|smtglight|stench'
                r'|stun|teleport$|tendri|tornado|visionfog)'),
    ('interface', r'^(camp|casting|changehero|check|constr|inventory|logo|pointer|repair|saveload|scrollbar|select'
                  r'|sound|trade|link$|level\d|zoneexit|zonemap|ambush|battle|cross$|flag$|rick$|default)'),
    ('buildings', r'.*(house|tower|wall|bridge|gate|tent|fence|shop|crypt|ruins|temple|door|prison|shipyard|veranda'
                  r'|cellar|cave|mine|dome|arc\d|pyramid|hall|headquaters|quater|vivary|well\d|fontain|pool|sortir'
                  r'|smith|throne|tipy|ladder|climb|column|zagon|aquarium|akvarium|statue|idol|altar|cage|waterlock)'),
]
