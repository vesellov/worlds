"""
Reader of the .adb animation databases (database.res), one per model.

The header is 0x2c bytes, every action is 0x58 bytes, the same offsets game.exe uses.
The packed field of an action is the key the server looks the animations up by (sub_5190c0):
    bits 0-6    weapons (SWORD, AXE, DAGGER, SPEAR, HAMMER, BOW, CROSSBOW)
    bits 15-17  state of the unit
    bits 18-21  type of the action
    bits 22-29  modifier of the action
    bits 30-31  stage of the animation
    bits 32-37  forms (HIDES, STEPS, HIT, SFXES, LOWSHAPE, HIGHSHAPE)
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import struct

#------------------------------------------------------------------------------

MAGIC = b'ADB\0'

HEADER_SIZE = 0x2c

ACTION_SIZE = 0x58

ENCODING = 'cp1251'

WEAPONS = ('SWORD', 'AXE', 'DAGGER', 'SPEAR', 'HAMMER', 'BOW', 'CROSSBOW')

STATES = ('NEUTRAL', 'REST', 'ATTACK', 'UNK3', 'WARRY', 'UNK5', 'LIE', 'ALL')

TYPES = ('UNK0', 'SPECIAL', 'ATTACK', 'CAST', 'RUN', 'WALK', 'IDLE', 'DEATH',
         'SUFFER', 'CROSS', 'UNK10', 'UNK11', 'UNK12', 'UNK13', 'UNK14', 'ALL')

STAGES = ('UNIQUE', 'START', 'CYCLE', 'END')

FORMS = ('HIDES', 'STEPS', 'HIT', 'SFXES', 'LOWSHAPE', 'HIGHSHAPE')

WEAPONS_MASK = 0x7f

STATE_SHIFT = 15
TYPE_SHIFT = 18
MODIFIER_SHIFT = 22
STAGE_SHIFT = 30

STAGE_MASK = 0xc0000000

STAGE_CYCLE_BITS = 0x80000000

#------------------------------------------------------------------------------


class Action(object):

    def __init__(self):
        self.name = ''
        self.number = 0
        self.packed = 0
        self.probability = 0
        self.length = 0
        self.movement_speed = 0.0
        self.show_hide_frames = (0, 0)
        self.sound_frames = (0, 0, 0, 0)
        self.hit_frame = 0
        self.special_sound_frame = 0
        self.sound_ids = (0, 0, 0, 0)

    def __repr__(self):
        return 'Action(%s %s %s:%d %s %s len=%d)' % (
            self.name, self.state_name, self.type_name, self.modifier, self.stage_name,
            ','.join(self.weapon_names) or '-', self.length)

    @property
    def key(self):
        """
        The lower 32 bits, the part of the packed field the animations are searched by.
        """
        return self.packed & 0xffffffff

    @property
    def weapons(self):
        return self.packed & WEAPONS_MASK

    @property
    def weapon_names(self):
        return [w for i, w in enumerate(WEAPONS) if self.weapons & (1 << i)]

    @property
    def state(self):
        return (self.packed >> STATE_SHIFT) & 7

    @property
    def state_name(self):
        return STATES[self.state]

    @property
    def type(self):
        return (self.packed >> TYPE_SHIFT) & 15

    @property
    def type_name(self):
        return TYPES[self.type]

    @property
    def modifier(self):
        return (self.packed >> MODIFIER_SHIFT) & 255

    @property
    def stage(self):
        return (self.packed >> STAGE_SHIFT) & 3

    @property
    def stage_name(self):
        return STAGES[self.stage]

    @property
    def is_cycle(self):
        # CAnimationSystem2 (0x50d600) loops a clip only for the stage CYCLE
        return (self.packed & STAGE_MASK) == STAGE_CYCLE_BITS

    @property
    def forms(self):
        return [f for i, f in enumerate(FORMS) if (self.packed >> 32) & (1 << i)]


class AnimationDatabase(object):

    def __init__(self):
        self.model_name = ''
        self.min_height = 0.0
        self.avg_height = 0.0
        self.max_height = 0.0
        self.actions = []

    def __repr__(self):
        return 'AnimationDatabase(%s, %d actions)' % (self.model_name, len(self.actions))

    def by_name(self, name):
        for action in self.actions:
            if action.name.lower() == name.lower():
                return action
        return None

    def size_scale(self, height_coef):
        """
        Scale of the figure for the height coefficient of the unit, the same as CUnitClientSpecific (sub_535850).
        """
        if self.avg_height == 0:
            return 1.0
        return (self.min_height + (self.max_height - self.min_height) * height_coef) / self.avg_height

#------------------------------------------------------------------------------


def make_key(state=0, action_type=0, modifier=0, stage=0, weapons=0):
    return ((weapons & WEAPONS_MASK) | (state << STATE_SHIFT) | (action_type << TYPE_SHIFT)
            | (modifier << MODIFIER_SHIFT) | (stage << STAGE_SHIFT)) & 0xffffffff


def _cstr(raw):
    return bytes(raw).split(b'\0', 1)[0].decode(ENCODING)


def parse(data):
    data = bytes(data)
    if data[:4] != MAGIC:
        raise ValueError('not an .adb file')
    db = AnimationDatabase()
    count = struct.unpack_from('<I', data, 4)[0]
    db.model_name = _cstr(data[8:32])
    db.min_height, db.avg_height, db.max_height = struct.unpack_from('<3f', data, 32)
    # a few files of the game are cut short (unmozozo, unmoco3, unmoco4, inmm1), only the whole actions are read
    count = min(count, (len(data) - HEADER_SIZE) // ACTION_SIZE)
    for i in range(count):
        offset = HEADER_SIZE + i * ACTION_SIZE
        a = Action()
        a.name = _cstr(data[offset:offset + 16])
        a.number, a.packed, a.probability, a.length, a.movement_speed = struct.unpack_from('<IQIIf', data, offset + 16)
        values = struct.unpack_from('<12I', data, offset + 0x28)
        a.show_hide_frames = values[0:2]
        a.sound_frames = values[2:6]
        a.hit_frame = values[6]
        a.special_sound_frame = values[7]
        a.sound_ids = values[8:12]
        db.actions.append(a)
    return db
