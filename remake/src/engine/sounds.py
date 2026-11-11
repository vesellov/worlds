"""
The sounds of the units, as the client of game.exe picks them.

The voice (534760): every action of .adb has a frame of the sound (+0x44, special_sound_frame) and up to four
numbers of the files (+0x48..+0x54, sound_ids, the list ends at the first 0). The sound starts when the frame
of the clip passes it: previous frame <= frame of the sound < current frame. One of the numbers is taken
at random and the name is made by the type of the action from the folder of the voice (tag 34 of the unit type
in units.udb, +0x124, replaced by tag 8 of the prototype, +0x34, when it is there):
- IDLE: "<folder>\\idle\\<n>.wav" with the chance of tag 37 of the type (+0x150) in percents;
- ATTACK: "<folder>\\attack\\<n>.wav" with the chance of tag 38 (+0x154); a human with a bow or a crossbow
  plays "weapons\\<n>.wav" instead;
- DEATH: "<folder>\\death\\<n>.wav"; SUFFER: "<folder>\\hit\\<n>.wav";
- SPECIAL: the sounds of the skills by the modifier (19 tame, 20 steal, 21 science, 22, 48, 52, 53 loot);
- a human whose blow missed plays "weapons\\miss.wav" at the frame of the sound of the attack (the flag +0x256
  of the unit is set before it); the remake decides the blow at hit_frame, so it plays the miss then.

534b60 runs every frame of the client for a unit. It remembers the action and the previous frame (+0x30,
+0x2c, the frame is 0 when the action changes) and plays the sound with the priority 1000 for SPECIAL,
ATTACK and DEATH, 0 for the others. The voice of an attack does not start while the previous one plays (+0x90),
ATTACK and DEATH stop the idle voice (+0x88), DEATH stops the attack voice too.

The steps of WALK and RUN (534b60, 534590): the values of +0x30..+0x3c of the action (sound_frames) minus 1 are
the frames of the steps, the list ends at the value 1; a step sounds when previous frame < its frame <= current
frame, then the others are not looked at. The folder is the one of the type of the tile under the unit
(engine.terrain.tile_type_at, 6c8d00) in the list of tag 35 of the unit type, the number is random among
the files of the folder: "<folder>\\<n>.wav". A unit higher than 0.01 above the land (on a bridge) steps
as on the tile type 2; a crawling unit plays "steps\\human\\crawl\\<1..6>.wav" except on the types 6 and 14
(the remake has no crawl yet). An effect 0x10 on the unit (50f630) silences the steps.

The blow (CUnitClientSpecific::vf46, 537500): the target plays "weapons\\<kind>\\<material>\\1.wav" when
the damage is above 0.0001. The kind is by the code of the weapon: the type of the weapon item (0 sword, 1 axe,
2 dagger...) or, for a unit without one, "real weapon typeID" of its prototype (tag 39, +0x7c). The material
is the skin of the type of the target (tag 44, +0x118), for a human the material of its armor on the part:
Bone, Crystal and Stone sound as stone, Metal as metal, the rest as leather.

All of them play by 6b85c0 at the position of the unit with the distances 8 and 40 (73ec1c, 73ec20):
the full volume up to 8, silence beyond 40, between them the volume falls evenly (6b7f80 (partly)).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import random

#------------------------------------------------------------------------------

from engine import actions

#------------------------------------------------------------------------------

HUMANOID_CLASS = 0x32

# 534b60: the priorities of 6b85c0
PRIORITY_LOW = 0
PRIORITY_HIGH = 1000

# 6b7cd0: the channels of the sounds of the world
CHANNELS = 15

# 73ec1c and 73ec20: the distances of the sounds of the units
MIN_DISTANCE = 8.0
MAX_DISTANCE = 40.0

# 7440dc and 7440e0: the sounds fade while the listener is between these heights and are silent above
LISTENER_FULL_HEIGHT = 30.0
LISTENER_SILENT_HEIGHT = 70.0

# 537500: no sound for a blow which did not take more than this
MIN_DAMAGE = 0.0001

# 534590: a unit higher above the land than this steps as on the tile type BRIDGE_TILE_TYPE
STEP_HEIGHT = 0.01
BRIDGE_TILE_TYPE = 2

CHANNEL_IDLE = 'idle'
CHANNEL_ATTACK = 'attack'

MISS_SOUND = 'weapons\\miss.wav'

# 534760: the modifiers of SPECIAL and the sounds of the skills
SKILL_SOUNDS = {
    0x13: 'heroes\\human\\skills\\tame.wav',
    0x14: 'heroes\\human\\skills\\steal.wav',
    0x15: 'heroes\\human\\skills\\science.wav',
    0x16: 'heroes\\human\\skills\\loot.wav',
    0x30: 'heroes\\human\\skills\\loot.wav',
    0x34: 'heroes\\human\\skills\\loot.wav',
    0x35: 'heroes\\human\\skills\\loot.wav',
}

# 537500: the folders of the sounds of a blow by the code of the weapon
BLOW_KINDS = {
    0: 'slash', 1: 'slash', 13: 'slash',
    2: 'pierce', 3: 'pierce', 5: 'pierce', 6: 'pierce', 10: 'pierce',
    4: 'crash', 11: 'crash',
    9: 'hands1', 12: 'hands4', 14: 'hands6', 15: 'hands7',
}

STONE_MATERIALS = ('Bone', 'Crystal', 'Stone')
METAL_MATERIAL = 'Metal'

# the codes of the bow and the crossbow (adb.WEAPONS)
SHOOTING_WEAPONS = (5, 6)

#------------------------------------------------------------------------------


class Sound(object):
    """
    A sound to play: the name in sfx.res as game.exe writes it, the priority of 6b85c0, the channel
    which the following sounds of the unit check or stop (None for the others).
    """

    def __init__(self, name, priority=PRIORITY_LOW, channel=None):
        self.name = name
        self.priority = priority
        self.channel = channel

    def __repr__(self):
        return 'Sound(%s, %d, %s)' % (self.name, self.priority, self.channel)


class Voice(object):
    """
    The sounds of a unit: the folder, the chances of the idle and the attack voices, the material of the skin,
    the code of the weapon of its blows (537500), whether it is a human (the miss and the shots).
    """

    def __init__(self, sfx_path='', idle_chance=0, attack_chance=0, material='', weapon_code=None, humanoid=False,
                 step_paths=(), step_counts=()):
        self.sfx_path = sfx_path
        # the folders of the steps by the type of the tile and the numbers of their files
        self.step_paths = list(step_paths)
        self.step_counts = list(step_counts)
        self.idle_chance = idle_chance
        self.attack_chance = attack_chance
        self.material = material
        self.weapon_code = weapon_code
        self.humanoid = humanoid

    def __repr__(self):
        return 'Voice(%s, idle %d%%, attack %d%%, %s, weapon %r)' % (
            self.sfx_path, self.idle_chance, self.attack_chance, self.material, self.weapon_code)

    @classmethod
    def from_unit_type(cls, unit_type, prototype=None, weapon_code=None, sound_count=None):
        """
        The voice of units.udb; weapon_code is the type of the weapon item (adb.WEAPONS), else the one
        of the prototype; sound_count(folder) gives the number of the files of a folder of the steps
        (eilib.gamedata.GameData.sound_count), without it there are no steps.
        """
        sfx_path = unit_type.sfx_path
        if prototype is not None and prototype.sfx_path:
            sfx_path = prototype.sfx_path
        if weapon_code is None and prototype is not None:
            weapon_code = prototype.weapon_kind
        step_paths = unit_type.step_paths if sound_count is not None else []
        step_counts = [sound_count(path) for path in step_paths]
        return cls(sfx_path, unit_type.idle_sound_chance, unit_type.attack_sound_chance, unit_type.skin_material,
                   weapon_code, unit_type.unit_class == HUMANOID_CLASS, step_paths, step_counts)


class SoundTracker(object):
    """
    534b60: follows the clip of one unit and gives the sounds which its frames pass.
    """

    def __init__(self, voice, rng=None):
        self.voice = voice
        self.rng = rng or random.Random()
        self.action = None
        self.frame = 0.0
        # the action changed and its first frame was not handled yet: the channels stop then
        self.fresh = False

    def update(self, action, frame, busy_channels=(), step_tile=None):
        """
        The sounds which start now and the channels which stop: (sounds, stopped channels).
        busy_channels are the channels whose sound still plays, step_tile() gives the type of the tile
        for the steps.
        """
        if action is not self.action:
            self.action = action
            self.frame = 0.0
            self.fresh = True
        previous = self.frame
        if frame < previous:
            # the clip of a cycle went round: the remake keeps one clip, the game starts the action again
            previous = 0.0
        self.frame = frame
        if action is None or frame == previous:
            return [], []
        changed = self.fresh
        self.fresh = False
        sounds = []
        stopped = []
        action_type = action.type
        if action_type == actions.TYPE_ATTACK:
            if changed:
                stopped.append(CHANNEL_IDLE)
            if CHANNEL_ATTACK not in busy_channels:
                name = self.voice_name(action, previous, frame)
                if name:
                    sounds.append(Sound(name, PRIORITY_HIGH, CHANNEL_ATTACK))
        elif action_type in (actions.TYPE_SPECIAL, actions.TYPE_DEATH):
            if action_type == actions.TYPE_DEATH and changed:
                stopped.extend((CHANNEL_IDLE, CHANNEL_ATTACK))
            name = self.voice_name(action, previous, frame)
            if name:
                sounds.append(Sound(name, PRIORITY_HIGH))
        elif action_type in (actions.TYPE_WALK, actions.TYPE_RUN):
            name = self.step_name(action, previous, frame, step_tile)
            if name:
                sounds.append(Sound(name, PRIORITY_LOW))
        elif action_type in (actions.TYPE_IDLE, actions.TYPE_SUFFER):
            name = self.voice_name(action, previous, frame)
            if name:
                channel = CHANNEL_IDLE if action_type == actions.TYPE_IDLE else None
                sounds.append(Sound(name, PRIORITY_LOW, channel))
        if sounds or stopped:
            if _Debug:
                print('%s frames %.1f..%.1f: %r, stop %r' % (action.name, previous, frame, sounds, stopped))
        return sounds, stopped

    def step_name(self, action, previous, frame, step_tile):
        """
        534b60 and 534590: the sound of the first step whose frame lies in (previous, frame].
        """
        for value in action.sound_frames:
            step_frame = value - 1
            if step_frame == 0:
                break
            if previous < step_frame <= frame:
                if step_tile is None:
                    return None
                tile = step_tile()
                counts = self.voice.step_counts
                if not 0 <= tile < len(counts) or not counts[tile]:
                    return None
                number = self.rng.randrange(counts[tile]) + 1
                return '%s\\%d.wav' % (self.voice.step_paths[tile], number)
        return None

    def voice_name(self, action, previous, frame):
        """
        534760: the name of the voice of the action when its frame of the sound lies in [previous, frame).
        """
        sound_frame = action.special_sound_frame
        if not sound_frame or sound_frame < previous or frame <= sound_frame:
            return None
        numbers = []
        for number in action.sound_ids:
            if not number:
                break
            numbers.append(number)
        if not numbers:
            return None
        number = numbers[self.rng.randrange(len(numbers))]
        folder = self.voice.sfx_path
        action_type = action.type
        if action_type == actions.TYPE_IDLE:
            if self.rng.randrange(100) >= self.voice.idle_chance:
                return None
            return '%s\\idle\\%d.wav' % (folder, number)
        if action_type == actions.TYPE_SPECIAL:
            return SKILL_SOUNDS.get(action.modifier)
        if action_type == actions.TYPE_ATTACK:
            if self.voice.humanoid and self.voice.weapon_code in SHOOTING_WEAPONS:
                return 'weapons\\%d.wav' % number
            if self.rng.randrange(100) >= self.voice.attack_chance:
                return None
            return '%s\\attack\\%d.wav' % (folder, number)
        if action_type == actions.TYPE_DEATH:
            return '%s\\death\\%d.wav' % (folder, number)
        if action_type == actions.TYPE_SUFFER:
            return '%s\\hit\\%d.wav' % (folder, number)
        return None

#------------------------------------------------------------------------------


def blow_sound(weapon_code, material, damage):
    """
    537500: the sound of a blow on the target, None when there is none.
    """
    if damage is None or damage <= MIN_DAMAGE:
        return None
    kind = BLOW_KINDS.get(weapon_code)
    if kind is None:
        return None
    if material in STONE_MATERIALS:
        material = 'stone'
    elif material != METAL_MATERIAL:
        material = 'leather'
    return Sound('weapons\\%s\\%s\\1.wav' % (kind, material), PRIORITY_HIGH)


def miss_sound(voice):
    """
    534760: a human whose blow missed.
    """
    if voice is None or not voice.humanoid:
        return None
    return Sound(MISS_SOUND, PRIORITY_HIGH)


def take_channel(priorities, priority):
    """
    6b7cd0: the channel for a new sound of the priority; priorities are those of the CHANNELS channels, None for
    a free one. The first free channel, else the first of the channels with the lowest priority when the new
    sound is above it (its sound is cut), else -1: the sound is not played.
    """
    lowest = None
    for index, busy in enumerate(priorities):
        if busy is None:
            return index
        if lowest is None or busy < priorities[lowest]:
            lowest = index
    if lowest is None or priority <= priorities[lowest]:
        return -1
    return lowest


def height_factor(height):
    """
    6b7f80 (partly): the part of the volume which is left at the height of the listener (the ground under the target
    of the camera plus 2, 40fb30): all of it up to 30, nothing from 70.
    """
    if height >= LISTENER_SILENT_HEIGHT:
        return 0.0
    if height <= LISTENER_FULL_HEIGHT:
        return 1.0
    return 1.0 - (height - LISTENER_FULL_HEIGHT) / (LISTENER_SILENT_HEIGHT - LISTENER_FULL_HEIGHT)


def volume(distance, near=MIN_DISTANCE, far=MAX_DISTANCE, height=0.0):
    """
    6b7f80 (partly): the volume 0..1 of a sound at the distance on the map from the listener (6b85c0, the units
    play with the distances 8 and 40): full up to near, silence beyond far, between them it falls evenly by whole
    percents; the height of the listener lowers it (height_factor). game.exe also counts the pan of the sound
    from the side of the listener, the remake has no pan.
    """
    if distance > far:
        return 0.0
    if distance < near or far <= near:
        percents = 100
    else:
        percents = 100 - int(round((distance - near) / (far - near) * 100.0))
    return percents * 0.01 * height_factor(height)
