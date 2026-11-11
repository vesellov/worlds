"""
The numbers of a unit: the 43 dwords of UNIT_STATS of a .mob (kept by the object of the map at +0x210, game.exe
46c010) or the fields of its prototype in units.udb (52d740) when the map gives none.

The meaning of the fields was found by comparing the units of zone1.mob with their prototypes, whose fields are
named by EIrepack (Formats/db.py, MIT license, (c) 2018 Asp Adm):

    0, 1    health, its maximum         prototype 9 (HP)
    2, 3    mana, its maximum           prototype 10 (mana), "Stamina" in the information window
    15      range of the attack         prototype 14 (attack range), 8 for the toads of zone1, 0 for the others
    16      to hit (attack)             prototype 15 (stats to hit)
    17      parry (defense)             prototype 16 (stats parry)
    18      weight of the weapon        prototype 17 (tuning weapon weight), 1.0 for the units of zone1
    19, 20  damage, min and spread      prototype 19, 20 (the damage is min + 0..spread, 54f630)
    21..27  parts of the 7 types of the damage, the tag 11 of the unit type
    28      absorption (armor)          prototype 11
    29..34  six senses                  prototype 25 (sight, infra, life, hearing, smell, tracking)
    35..40  six detections              prototype 26

    4       the multiplier of the speed  1.0 for the monsters of zone1, 1.25 for its "Runner" heroes
    6..9    the speeds of the run, the walk, the sneak and the crawl, map units per tick (55 ms),
            the same as the tag 7 of the unit type in units.udb

The fields 0..3, 14 and 28 are integers, the others floats (checked on all the units of zone1;
41 and 42 hold garbage). 477b00 copies the record into the unit of the server: 0..3 the health and the mana,
4..12 +0x208..+0x22c, 13 +0x244, the bytes of 14 the classes of the paths, 16 the attack +0x254, 17 the parry
+0x25c, 18 the weight of the weapon +0x260, 19 and 20 the damage +0x264 and +0x268, 21..27 the parts of the types
of the damage +0x270, 28 the absorption times the 7 factors of the unit type. 15 goes to +0x250: a unit
without a weapon item whose value is above 1 shoots (54b6c0). 10 (+0x224) is the cosine of the half of the angle
of the sight and 11 (+0x228) the radius of the peripheral vision (5ca5a0 (partly), see engine.ai): -0.0 and 1.7 for most
units of zone1, 0.707 and 5.0 for three of them. 5, 12 and 13 are not known yet.

The speed. 522ce0 packs the stats of the server for the clients: the speeds +0x34..+0x40 of the stats object
(run, walk, sneak, crawl) as bytes of speed * 255 / 2 at +0x340..+0x343 of the unit, the multiplier
+0x28 = +0x24 * the factor of the legs as a byte of 25.5 * it at +0x344. 52fa20 and 52f920 give the speed
of the way of moving +0x264 (3 run, 2 and 4 walk, 1 sneak, 0 crawl): multiplier * speed of the way, units
per tick (the track of the server, 5bb190 and 5c7430, counts the ticks); the second value of 52f920 is the speed
of turning: the same (the run for the walks 2 and 4) times RotateSpeedMult of ai.reg (7c5428, 100).
A unit without the stats of a map takes the multiplier from the tag 13 of its prototype (+0x60, copied by 5136d0
to +0x24 of the numbers), 1.0 for all the prototypes of units.udb.

Wounded legs slow a unit down. The levels come from the section [RPG] of config/ai.reg:

    524060  the speed factor: the smallest health part of the legs (hit locations of the type 3)
            <= DamageLevel1 gives DamageLevel1Value, else <= DamageLevel2 gives DamageLevel2Value, else 1;
            the speed of the unit is its base speed times the factor (524560, 524cd0: +0x28 = +0x24 * +0x58)
    5511a0  the modifier of the walk cycle: factor <= DamageLevel1Value is 2, factor <= DamageLevel2Value is 1,
            else 0 (the limping walks cwalk06..11 of unhuma)
    5241c0  a leg below DamageLevelRunLimit does not let the unit run
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import struct

#------------------------------------------------------------------------------

STATS_COUNT = 43

HEALTH = 0
MAX_HEALTH = 1
MANA = 2
MAX_MANA = 3
VIEW_COS = 10
PERIPHERAL = 11
ATTACK_RANGE = 15
TO_HIT = 16
PARRY = 17
WEAPON_WEIGHT = 18
DAMAGE_MIN = 19
DAMAGE_SPREAD = 20
DAMAGE_TYPES = 21
DAMAGE_TYPES_COUNT = 7
ABSORPTION = 28
SENSES = 29
DETECTIONS = 35
SENSES_COUNT = 6
SPEED_SCALE = 4
SPEEDS = 6

SPEED_RUN = 0
SPEED_WALK = 1
SPEED_SNEAK = 2
SPEED_CRAWL = 3
SPEEDS_COUNT = 4

INTEGER_FIELDS = frozenset((0, 1, 2, 3, 14, 28))

WOUNDS_SECTION = 'RPG'
DAMAGE_LEVEL_1 = 0.5
DAMAGE_LEVEL_1_VALUE = 0.5
DAMAGE_LEVEL_2 = 0.0
DAMAGE_LEVEL_2_VALUE = 0.33
DAMAGE_LEVEL_RUN_LIMIT = 0.5

#------------------------------------------------------------------------------


class UnitStats(object):

    def __init__(self, health, max_health, mana, max_mana, attack, defense, damage_min, damage_max, armor,
                 speed_scale=1.0, speeds=None, weapon_weight=1.0, damage_spread=None, damage_types=None,
                 senses=None, detections=None, attack_range=0.0, view_cos=-1.0, peripheral=0.0):
        """
        damage_max is damage_min + damage_spread, damage_types the 7 parts of the types of the damage
        (None: those of the unit type).
        """
        self.speed_scale = speed_scale
        self.weapon_weight = weapon_weight
        self.attack_range = attack_range
        # the cosine of the half of the angle of the sight (-1: all around) and the radius of the peripheral vision
        self.view_cos = view_cos
        self.peripheral = peripheral
        # the six senses (sight, infra, life, hearing, smell, tracking) and the six detections of the unit
        self.senses = tuple(senses) if senses else (0.0, ) * SENSES_COUNT
        self.detections = tuple(detections) if detections else (1.0, ) * SENSES_COUNT
        self.damage_spread = float(damage_max - damage_min) if damage_spread is None else damage_spread
        self.damage_types = tuple(damage_types) if damage_types else None
        # run, walk, sneak and crawl, map units per tick
        self.speeds = tuple(speeds) if speeds else None
        self.health = health
        self.max_health = max_health
        self.mana = mana
        self.max_mana = max_mana
        self.attack = attack
        self.defense = defense
        self.damage_min = damage_min
        self.damage_max = damage_max
        self.armor = armor

    def __repr__(self):
        return 'UnitStats(health %d/%d, mana %d/%d, attack %d, defense %d, damage %d-%d, armor %d)' % (
            self.health, self.max_health, self.mana, self.max_mana, self.attack, self.defense,
            self.damage_min, self.damage_max, self.armor)

    @property
    def health_part(self):
        return self.health / float(self.max_health) if self.max_health > 0 else 0.0

    @property
    def mana_part(self):
        return self.mana / float(self.max_mana) if self.max_mana > 0 else 0.0

#------------------------------------------------------------------------------


class WoundLevels(object):
    """
    How wounded legs change the walk of a unit, the levels of [RPG] of ai.reg (stored at 7c542c..7c543c).
    """

    def __init__(self, level1=DAMAGE_LEVEL_1, level1_value=DAMAGE_LEVEL_1_VALUE, level2=DAMAGE_LEVEL_2,
                 level2_value=DAMAGE_LEVEL_2_VALUE, run_limit=DAMAGE_LEVEL_RUN_LIMIT):
        self.level1 = level1
        self.level1_value = level1_value
        self.level2 = level2
        self.level2_value = level2_value
        self.run_limit = run_limit

    @classmethod
    def from_registry(cls, registry):
        """
        The levels of the sections of ai.reg (as parsed by eilib.reg), the defaults when they are not there.
        """
        keys = (registry or {}).get(WOUNDS_SECTION) or {}
        return cls(
            level1=float(keys.get('DamageLevel1', DAMAGE_LEVEL_1)),
            level1_value=float(keys.get('DamageLevel1Value', DAMAGE_LEVEL_1_VALUE)),
            level2=float(keys.get('DamageLevel2', DAMAGE_LEVEL_2)),
            level2_value=float(keys.get('DamageLevel2Value', DAMAGE_LEVEL_2_VALUE)),
            run_limit=float(keys.get('DamageLevelRunLimit', DAMAGE_LEVEL_RUN_LIMIT)),
        )

    def speed_factor(self, legs_part):
        if legs_part <= self.level1:
            return self.level1_value
        if legs_part <= self.level2:
            return self.level2_value
        return 1.0

    def walk_modifier(self, speed_factor):
        if speed_factor <= self.level1_value:
            return 2
        if speed_factor <= self.level2_value:
            return 1
        return 0

    def can_run(self, legs_part):
        return not legs_part < self.run_limit

#------------------------------------------------------------------------------


def decode(dwords):
    """
    The values of UNIT_STATS as read by eilib.mob (unsigned dwords): floats where they are floats.
    """
    result = []
    for index, value in enumerate(dwords):
        if index in INTEGER_FIELDS:
            result.append(struct.unpack('<i', struct.pack('<I', value & 0xffffffff))[0])
        else:
            result.append(struct.unpack('<f', struct.pack('<I', value & 0xffffffff))[0])
    return result


def from_mob(dwords):
    """
    UnitStats of the UNIT_STATS of a unit of a map, None when the record is not complete.
    """
    if dwords is None or len(dwords) < STATS_COUNT:
        return None
    v = decode(dwords)
    return UnitStats(v[HEALTH], v[MAX_HEALTH], v[MANA], v[MAX_MANA], int(round(v[TO_HIT])), int(round(v[PARRY])),
                     int(round(v[DAMAGE_MIN])), int(round(v[DAMAGE_MIN] + v[DAMAGE_SPREAD])), v[ABSORPTION],
                     speed_scale=v[SPEED_SCALE], speeds=v[SPEEDS:SPEEDS + SPEEDS_COUNT], weapon_weight=v[WEAPON_WEIGHT],
                     damage_spread=v[DAMAGE_SPREAD], damage_types=v[DAMAGE_TYPES:DAMAGE_TYPES + DAMAGE_TYPES_COUNT],
                     senses=v[SENSES:SENSES + SENSES_COUNT], detections=v[DETECTIONS:DETECTIONS + SENSES_COUNT],
                     attack_range=v[ATTACK_RANGE], view_cos=v[VIEW_COS], peripheral=v[PERIPHERAL])


def from_prototype(prototype, unit_type=None):
    """
    UnitStats of a fresh unit of the prototype of units.udb: the health and the mana are full,
    the speeds are those of unit_type (idb.UnitType) when it is given, the multiplier of the speed is
    the tag 13 of the prototype (5136d0).
    """
    health = int(round(prototype.health))
    mana = int(round(prototype.mana))
    return UnitStats(health, health, mana, mana, int(round(prototype.attack)), int(round(prototype.defense)),
                     int(round(prototype.damage_min)), int(round(prototype.damage_min + prototype.damage_max)),
                     int(round(prototype.armor)), speed_scale=prototype.speed_scale,
                     speeds=unit_type.speeds if unit_type is not None else None,
                     weapon_weight=prototype.weapon_weight or 1.0, damage_spread=prototype.damage_max,
                     senses=prototype.senses, detections=prototype.detections, attack_range=prototype.attack_range)
