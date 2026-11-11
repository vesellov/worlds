"""
The spells, as the server of game.exe builds and casts them.

A spell (CSpell, 68bf20) is a prototype of spells.sdb (eilib.idb.SpellPrototype) with modifiers
(eilib.idb.SpellModifier) in the order of the scroll. The fields start as those of the prototype, then every
modifier works by the first letter of its code, its value v and the prototype p:

    r  range     += v * p.range,      not below 0
    a  area      += v * p.area,       not below 0
    e  effect    += v * p.effect,     not below 0
    m  mana      += v * p.mana,       not below 0
    d  duration  += int(v * p.duration), not below 0
    t  targets   += int(v * p.targets)
    f  the filter of the targets by the second letter: f friends (1), e enemies (2), h not humans (4),
       g not goblins (8), o not orcs (0x10), l not lizardmen (0x20)
    i  it a weapon trigger (0x10000), ic an armor which works all the time (0x20000)

and every modifier adds 1 to the slots, its mana (+0x10 of the modifier) to the mana, its price to the price and
its level (+0x18) to the complexity. At the end an area above 0 becomes the radius sqrt(area / pi), and the mana
is not less than the mana of the prototype. A modifier is allowed (68be80) when the flag of its letter in
the tag 20 of the prototype is set: r 0, t 1, a 2, e 3, d 4; the other letters are always allowed.

The filter (68b480) is checked only when one of the bits 0..15 is set; a target which is not a unit is then
skipped. The friends and the enemies are told apart by the masks +0x244 and +0x248 of both units: a unit of the
player which attacks a unit that is not at war with it gets its bit there (52e940), so these are the units
which became enemies; the remake takes the diplomacy of the map as well (engine.ai.Diplomacy).
The races are told by the model: unhuma and unhufe, unmogo, unorma and unorfe, unmoli.

The blow of a spell (683910) is the packet of a blow of a weapon (engine.combat.Hit) with the attack FLT_MAX,
so it always hits (CUnitServer::vf20 skips the roll 553df0), and the part of the body 6: 5254b0 takes
the absorption of the unit first and then spreads the damage over the parts so that the health of the unit
falls by exactly the rest. Only the 12 spells below hurt, by one type of the damage each, all others give
the parts 0 (their effects are not found yet). The roll is the effect; a spell which lasts (the duration above 1)
gives effect * 4 / duration a time, and its absorption is taken in the same part.

Casting (554f50 at a unit, 554d00 at a point, 54ee10 at the moment of the spell): only for a unit
of the player (+0x24c of the unit) the skill is checked and the mana is spent. The skill of the school
(527eb0, the object +0x1e0 of the unit of the server) is
    +0x20 - 25 + the byte of the group of the school + the bonus of the level of the school
(the groups: fire, lightning, acid at +0x6ce; illusion, divination at +0x6cf; enchantments, healing,
domination at +0x6d0); the spell is refused when round(skill) is below the complexity. The mana spent is
mana * 25 / +0x20, kept between 0 and the maximum. +0x20 is not known yet (25 is the plain value, a hero
with 25 pays the mana of the spell); the units of the computer pay nothing. A spell of a scroll takes its mana
from the scroll (552c60).

The spells without damage (the effect of a spell is a state machine, 67e690: 0 the start 67f830, 1 the flight
681b70, 2 the lasting 682610, 3 the end 67ef80, each switched by the number of the prototype):

- the healing (24) calls CUnitServer::vf21 (5543b0) with the effect: 525800 spreads it over the parts
  (engine.combat.heal);
- the spells 9..17, 19, 25, 26, 29..34, 37, 38 go to the list of the spells on the target (vcall 0xb4 of the unit,
  50f1e0): a record of 20 bytes copied from the spell (the number, the ticks of the duration, the effect...).
  Every tick (CMapObjectServer::vf06, 515820) a record loses a tick, the last one takes it away; then 524560
  counts the numbers of the unit again (engine.combat.Fighter.recount) and, with them, the senses:
      12 eagle sight, 13 infravision, 14 detect life   the sight, the infra, the life sense grow by the effect
      15 invisibility   the detection by the sight falls by effect * 0.01, not below 0
      16 silence        the detection by the hearing falls by the effect, not below 0
      17 lichdom        the detection by the life sense falls by the effect, not below 0
      19 stench         the detection by the smell grows by the effect
  The stun (25) and the teleport (28, on its caster after the move, Teleport) freeze the unit (0x1000000 of +8
  of the unit, 52f7f0): its server does nothing (5499f0) while one of them is on it. The speed and the slow
  (33, 34) give speed - slow to +0x30 of the numbers, the speed of the strikes (engine.combat.Fighter.strike_speed); 31 regeneration multiplies the healing of the regeneration (523740, engine.combat.Fighter.regenerate);
  the others are not known yet.
- 58e670: the stench, the stun, the weaken, the feeblemind, the slow and the shrink (19, 25, 30, 32, 34, 38)
  provoke the target like a blow.
- 58e850 spreads a spell over its targets: one target when the spell has no filter and one target;
  otherwise the nearest units to the point within the range of the spell which pass the filter (and are
  enemies of a caster of the computer). Possession, link and charm (21, 27, 36) are not cast this way.

The flight (67f830, 681b70): the fire arrow (spells 0 and 40) and the acid ray (2) send the missile
CEffectArrow (0x4e) which homes on the target like an arrow (engine.missiles) with the step 0.667 a tick,
the blow comes when it arrives; the lightning (1, 41) hits at once, its arc (CEffectLightning 0x4d by
67f3d0, engine.lightning) goes from the hand of the caster to 1 above the feet of the target with the power
5 * the visual strength (41 with the textures of the demons) and follows them for 7 ticks (LightningArc).

The spells of the point make an effect on the ground (AreaEffect; the states of 67e690 are 0 the start
67f830, 1 the flight 681b70, 2 the lasting 682610, 3 the end 67ef80), its radius is sqrt(area / pi):
    3 fireball            flies from the caster to the point with the speed of the spell over 15 ticks
                          (5 without one) and bursts in its circle (6835c0)
    4 invoke lightning    hits its circle at once
    5 column of acid      hits its circle on the 15th of its 30 ticks
    6, 7 walls            hit every 4th tick of their duration a strip 1 wide across the way from the
                          caster, as long as the radius to both sides (6836f0)
    8 acid fog            hits its circle every 4th tick of its duration
    39 campfire           hits its circle every 4th tick and burns until it is taken away
The units of the circle (5c0bd0) pass the filter of the spell; 681b40 also wants +0x324 of the unit or the
field +0x24 below 0.5 of the record of 530930 (0x120 bytes, not known yet), not checked here. After the first
blow of a lasting effect (+0xc | 1) the blows come without the attacker and the victims do not answer them.

The lightning of the effects (67f830, 682610): the invoke lightning strikes from 30 above to its point
(3 below the ground) with the power -7 * the strength for one tick, and once more with the power -7 on its
third tick; the lightning wall stands between the ends of the wall 1 above the ground with the power 0
while it lasts.

The light of a spell (67f780): the point light CEffectPointLight (0x4b, the light 0x31 of the client) with the
color (+0x64) and the radius (+0x70) of the prototype moves by its speed every tick (vf06 678170); the spell
keeps one (+0x98), a new one takes the place of the old. 68d560/6c3180 light the ground with it: the color
times 1 - d^2 / r^2, full on the side which faces it and dot + 1 on the other, added to the light of the
vertex and taken by the largest of the lights. It goes out by rising: the speed (0, 0, radius / fadeout) with
the fadeout of the prototype (+0x74, at least 1) takes it above the ground in fadeout ticks.
    0, 2, 40 missiles   where they hit, 1.5 * the radius, rising 1.5 times faster (681b70)
    1, 41 arc           at the target while the arc lasts, then rises (682610 case 1)
    3 fireball          flies with the ball; where it bursts as the missiles (681b70 case 3)
    4 invoke lightning  2 above the ground of the point, rising at once
    5 column of acid    at the point, rising at once
    6, 7, 8 walls, fog  1 above the ground, the radius of the area + its own, rises when the effect ends

The sounds of a spell (67f830 the start, 681b70 the flight, 682610 the lasting): 67f540 makes the sound
effect 0x48 of one file, 67f620 the effect 0x63 which plays its start once and then loops the files of
its list (one taken at random). The spell effect keeps one sound (+0x94): a new one stops the old, so the
loop ends with the sound of the end. They sound at 10 and 50 (the full volume up to 10, falling evenly
to nothing at 50, 6b7f80), the strike of Invoke Lightning at 1000 over the whole map:
    0, 40 fire arrow     magic/Firearrow/start at the cast, circle on the missile, end where it hits
    1, 41 lightning      magic/lightning/start
    2 acid ray           magic/Acidray/ start, circle, end as the fire arrow
    3 fireball           magic/Fireball/start and the circle with the ball when it flies at least 1.25 s
                         (distance >= speed * 1.25), otherwise magic/FireBall/start alone; fireball/end
                         where it bursts
    4 invoke lightning   magic/InvokeLight/start
    5 column of acid     magic/Acidcolumn/start
    6 wall of fire       magic/Firewall/ start and circle while it lasts, end
    7 lightning wall     magic/Lightwall/circle while it lasts
    8 acid fog           magic/poison/ start and circle while it lasts, end
    20 clairvoyance      magic/Clairvoyence/start at the point
    21 possession        magic/Possession/start at the target (no such file in sfx.res)
    23 vision fog        magic/VisionFog/start at the target
    24 healing           magic/Healing/start at the target (not for a target of the spread with the full health)
    27 link              magic/Link/ start, then circle while it lasts (67f620), at the target
    36 charm             magic/Charm/start at the target
    28 teleport          magic/Teleport/start at the target
The enchantments sound on the client of the unit (5112a0, 5114d0): the type which comes to its list plays
"magic/<name>/start.wav", the type which leaves it "magic/<name>/end.wav" (ENCHANTMENT_SOUNDS, at 10
and 50); the feeblemind (32) loops its circle while it is on. 510890: a unit of the party (the ids from 10^9)
plays magic/end1.wav (at 8 and 20) when an enchantment has less than 30 ticks left, not over itself.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math

#------------------------------------------------------------------------------

from eilib import idb
from engine import combat
from engine import lightning
from engine import particles

#------------------------------------------------------------------------------

# the types of the damage of the spells (683910)
FIRE = 3
ACID = 4
LIGHTNING = 5

# the index of a prototype in spells.sdb -> the type of its damage and whether it lasts (+0x30 of the packet)
DAMAGE_SPELLS = {
    0: (FIRE, False),       # Fire Arrow
    3: (FIRE, False),       # Fireball
    40: (FIRE, False),      # Rick Magic
    1: (LIGHTNING, False),  # Lightning Arc
    4: (LIGHTNING, False),  # Invoke Lightning
    41: (LIGHTNING, False), # Curse Magic
    2: (ACID, False),       # Acid Ray
    5: (ACID, True),        # Column of Acid
    8: (ACID, True),        # Acid Fog
    6: (FIRE, True),        # Wall of Fire
    39: (FIRE, True),       # Campfire
    7: (LIGHTNING, True),   # Smtg Lightning
}

# 73baa0: a lasting spell gives effect * 4 / duration a time
LASTING_SHARE = 4.0

# 73e124: the plain value of +0x20 of the unit of the server, mana * 25 / it is spent
PLAIN_MIND = 25.0

# the bits of the targets (+0x6c of the spell)
TARGET_UNIT = 0x10000000
TARGET_POINT = 0x20000000
FILTER_FRIENDS = 0x1
FILTER_ENEMIES = 0x2
FILTER_HUMANS = 0x4
FILTER_GOBLINS = 0x8
FILTER_ORCS = 0x10
FILTER_LIZARDMEN = 0x20
FILTERS_MASK = 0xffff
ITEM_TRIGGER = 0x10000
ITEM_CONTINUOUS = 0x20000

FILTER_LETTERS = {'f': FILTER_FRIENDS, 'e': FILTER_ENEMIES, 'h': FILTER_HUMANS, 'g': FILTER_GOBLINS,
                  'o': FILTER_ORCS, 'l': FILTER_LIZARDMEN}
ITEM_LETTERS = {'t': ITEM_TRIGGER, 'c': ITEM_CONTINUOUS}

# 68b480: the models which a filter of a race skips
RACE_MODELS = (
    (FILTER_HUMANS, ('unhuma', 'unhufe')),
    (FILTER_GOBLINS, ('unmogo', )),
    (FILTER_ORCS, ('unorma', 'unorfe')),
    (FILTER_LIZARDMEN, ('unmoli', )),
)

# 68be80: the letter of a modifier -> the index of its flag in the tag 20 of the prototype
MODIFIER_FLAGS = {'r': 0, 't': 1, 'a': 2, 'e': 3, 'd': 4}

# 527eb0: the subtype of a spell (+0x14 of the prototype) -> the group of the skill
SCHOOL_GROUPS = (0, 0, 0, 1, 1, 2, 2, 2)

# 67aa70: the step of the missile CEffectArrow (0x4e) a tick
MISSILE_STEP = 0.6666667

# 67f830: the speed of a spell is the distance of 15 ticks (_DAT_0073c518, _DAT_00740544), 5 without one
SPEED_TICKS = 15.0
DEFAULT_SPEED = 5.0

# the spells which make an effect on the ground (6835c0, 6836f0): the fireball flies to its point and bursts
# there, the invoke lightning hits at once, the column of acid on its 15th tick of 30, the walls of fire and
# lightning hit their strip and the acid fog and the campfire their circle every 4th tick
FIREBALL_SPELL = 3
INVOKE_LIGHTNING_SPELL = 4
ACID_COLUMN_SPELL = 5
WALL_SPELLS = (6, 7)
ACID_FOG_SPELL = 8
CAMPFIRE_SPELL = 39
AREA_SPELLS = (FIREBALL_SPELL, INVOKE_LIGHTNING_SPELL, ACID_COLUMN_SPELL) + WALL_SPELLS + (ACID_FOG_SPELL,
                                                                                         CAMPFIRE_SPELL)
COLUMN_TICKS = 30
COLUMN_HIT_TICK = 15
LASTING_HIT_STEP = 4
# 6836f0: the strip of a wall is 1 wide (_DAT_0073cdb8 = 2 over the length of its normal)
WALL_HALF_WIDTH = 0.5

# 58e850 does not spread the possession, the link and the charm
UNSPREAD_SPELLS = (21, 27, 36)

# 67f830: the strength of a visual grows as the cube root of the effect over that of the prototype (1 to 8)
VISUAL_POWER = 0.33
VISUAL_LEAST = 0.2
VISUAL_MOST = 8.0
# the scales of the particles over the strength: the smoke of a fireball, the flash of a hit (681b70)
FIREBALL_SMOKE_SCALE = 0.3
HIT_FLASH_SCALE = 0.7
# 67f830 case 3: the fireball flies 1 above the ground of its point
FIREBALL_HEIGHT = 1.0
# the particles of the missiles: the fire arrow and the "rick magic" (67f830 cases 0 and 40),
# the flashes where they and the acid ray hit (681b70)
MISSILE_TRAILS = {0: particles.FIRE_ARROW_SMOKE, 40: particles.RIK_ARROW_SMOKE}
HIT_FLASHES = {0: particles.FIRE_ARROW_HIT, 40: particles.FIRE_ARROW_HIT, 2: particles.ACID_RAY_HIT}

# the spells which send the missile 0x4e (67f830)
MISSILE_SPELLS = (0, 2, 40)

# the spells which hit the target at once (67f830)
INSTANT_SPELLS = (1, 41)
# 67f830 case 41: the arc with the textures of the demons
DEMON_ARC_SPELL = 41
# 67f3d0: the power of the arc is 5 * the visual strength, it lives 7 ticks (+0xa0, 682610 case 1)
ARC_POWER = 5.0
ARC_TICKS = 7
# vf06 67a8b0: the arc ends 1 above the feet of the target
ARC_TARGET_HEIGHT = 1.0
# the remake's reading of the hand of the caster (532860 takes the base of the weapon): the middle of the unit
ARC_HAND_HEIGHT = 0.5
# 67f830 case 4: the strike of the invoke lightning from 30 above its point, 3 below the ground
STRIKE_HEIGHT = 30.0
STRIKE_DEPTH = 3.0
STRIKE_POWER = -7.0
# 682610 case 4: the second strike comes on the third tick, each one lives one tick
SECOND_STRIKE_TICK = 3
STRIKE_TICKS = 1
# 67f830 case 7: the lightning of the wall 1 above the ground
WALL_LIGHTNING_HEIGHT = 1.0
# 681b70: the light where a missile or a fireball bursts is larger and fades faster (743998)
FLASH_LIGHT_SCALE = 1.5
# 67f830: the lights of the walls and the fog stand 1 above the ground, the one of the invoke lightning 2
GROUND_LIGHT_HEIGHT = 1.0
STRIKE_LIGHT_HEIGHT = 2.0

# the distances of the sounds of the spells (67f540, 67f620) and of the strike of Invoke Lightning
SOUND_NEAR = 10.0
SOUND_FAR = 50.0
SOUND_EVERYWHERE = 1000.0
# 67f830 case 3: the fireball which flies at least this long (in the seconds of its speed) loops its sound
FIREBALL_LOOP_FLIGHT = 1.25
START_SOUNDS = {
    0: 'magic\\Firearrow\\start.wav', 40: 'magic\\Firearrow\\start.wav', 1: 'magic\\lightning\\start.wav',
    41: 'magic\\lightning\\start.wav', 2: 'magic\\Acidray\\start.wav', 3: 'magic\\FireBall\\start.wav',
    4: 'magic\\InvokeLight\\start.wav', 5: 'magic\\Acidcolumn\\start.wav', 6: 'magic\\Firewall\\start.wav',
    8: 'magic\\poison\\start.wav', 24: 'magic\\Healing\\start.wav', 28: 'magic\\Teleport\\start.wav',
    20: 'magic\\Clairvoyence\\start.wav', 21: 'magic\\Possession\\start.wav', 23: 'magic\\VisionFog\\start.wav',
    27: 'magic\\Link\\start.wav', 36: 'magic\\Charm\\start.wav',
}
FLYING_FIREBALL_SOUND = 'magic\\Fireball\\start.wav'
CIRCLE_SOUNDS = {
    0: 'magic\\Firearrow\\circle.wav', 40: 'magic\\Firearrow\\circle.wav', 2: 'magic\\Acidray\\circle.wav',
    3: 'magic\\Fireball\\circle.wav', 6: 'magic\\Firewall\\circle.wav', 7: 'magic\\Lightwall\\circle.wav',
    8: 'magic\\poison\\circle.wav', 27: 'magic\\Link\\circle.wav',
}
END_SOUNDS = {
    0: 'magic\\Firearrow\\end.wav', 40: 'magic\\Firearrow\\end.wav', 2: 'magic\\Acidray\\end.wav',
    3: 'magic\\fireball\\end.wav', 6: 'magic\\Firewall\\end.wav', 8: 'magic\\poison\\end.wav',
    18: 'magic\\Firework\\end.wav', 20: 'magic\\Clairvoyence\\end.wav', 27: 'magic\\Link\\end.wav',
}
# 5114d0: the folders of the sounds of the enchantments by the spell; the stench (19) has none
ENCHANTMENT_SOUNDS = {
    9: 'magic\\Fireshield', 10: 'magic\\Lightsheild', 11: 'magic\\PoisonShield', 12: 'magic\\Eaglesight',
    13: 'magic\\Infravision', 14: 'magic\\Detectlife', 15: 'magic\\Invisibility', 16: 'magic\\Silence',
    17: 'magic\\Lichdom', 25: 'magic\\Stun', 26: 'magic\\Antimagic', 29: 'magic\\Strengthen', 30: 'magic\\Weaken',
    31: 'magic\\Regeneration', 32: 'magic\\FeebleMind', 33: 'magic\\Speed', 34: 'magic\\Slow', 37: 'magic\\Enlarge',
    38: 'magic\\Shrink',
}
FEEBLEMIND_SPELL = 32
# 510890: the warning of the end of an enchantment of the party
ENCHANTMENT_WARNING_SOUND = 'magic\\end1.wav'
ENCHANTMENT_WARNING_TICKS = 30
WARNING_NEAR = 8.0
WARNING_FAR = 20.0

HEALING_SPELL = 24

# the remake's choice: the particles of the fireworks, the clairvoyance and the link (67f830 cases 0x12, 0x14,
# 0x1b, engine.particles.SPELL_PARTICLES) are born for the duration of the spell, at least this many ticks
SPELL_PARTICLES_TICKS = 45
PARTICLE_SPELLS = tuple(sorted(particles.SPELL_PARTICLES))
FIREWORKS_SPELL = 18
CLAIRVOYANCE_SPELL = 20
POSSESSION_SPELL = 21
LINK_SPELL = 27
# 682610 case 0x14: the clairvoyance sounds its end when its counter (+0xa0) comes to this
CLAIRVOYANCE_END_TICK = 10
# 682610 case 0x12: the threat of the fireworks forgets itself by this a tick (the weight is the effect)
FIREWORKS_THREAT_STEP = -1

# 67f830: the spells which go to the list of the spells on the target
VISION_FOG_SPELL = 23
COMPREHEND_SPELL = 35
CHARM_SPELL = 36

# the sight of a unit in the fog of visions (the share of the sight left)
VISION_FOG_FACTOR = 0.5

ENCHANTMENT_SPELLS = (9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 23, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36,
                      37, 38)

# 58e670: the spells which provoke the target
HOSTILE_SPELLS = (19, 25, 30, 32, 34, 38)

# 515820: the spells which freeze the unit
FREEZING_SPELLS = (25, 28)
TELEPORT_SPELL = 28
# 67f830 case 28: the caster goes 30 ticks after the cast (+0xa0 = duration + 30), then the freeze of
# the duration; the sparks fall into the point from 0.2 below its ground
TELEPORT_DELAY = 30
TELEPORT_SINK = 0.2

# 524560: the spells which make a sense sharper: the number -> the sense (engine.stats, 0 sight, 1 infra, 2 life)
SENSE_SPELLS = {12: 0, 13: 1, 14: 2}

# 524560: the spells which change how well the unit is detected: the number -> (the sense, the factor of the effect)
DETECTION_SPELLS = {15: (0, -0.01), 16: (3, -1.0), 17: (2, -1.0), 19: (4, 1.0)}

#------------------------------------------------------------------------------


class Spell(object):
    """
    A spell built of a prototype and modifiers (68bf20).
    """

    def __init__(self, prototype, modifiers=()):
        self.prototype = prototype
        self.modifiers = []
        for modifier in modifiers:
            if modifier is not None and self.allows(modifier):
                self.modifiers.append(modifier)
        self.build()

    def __repr__(self):
        return 'Spell(%r%s, effect %g, mana %g)' % (
            self.prototype.name, ''.join(' +' + m.code for m in self.modifiers), self.effect, self.mana)

    @property
    def index(self):
        return self.prototype.index

    @property
    def name(self):
        return self.prototype.name

    def allows(self, modifier):
        """
        68be80: whether the prototype takes the modifier.
        """
        index = MODIFIER_FLAGS.get(modifier.code[:1])
        if index is None:
            return True
        flags = self.prototype.modifier_flags
        return index < len(flags) and flags[index]

    def build(self):
        p = self.prototype
        self.speed = p.speed
        self.range = p.range
        self.targets = p.targets
        self.flags = 0
        if p.target == idb.SPELL_TARGET_UNIT:
            self.flags = TARGET_UNIT
        elif p.target == idb.SPELL_TARGET_POINT:
            self.flags = TARGET_POINT
        self.mana = p.mana
        self.slots = p.slots
        self.price = p.price
        self.actions = p.actions
        self.effect = p.effect
        self.area = p.area
        self.duration = p.duration
        self.complexity = p.complex
        for m in self.modifiers:
            letter, second = m.code[:1], m.code[1:2]
            if letter == 'a':
                self.area = max(0.0, self.area + m.value * p.area)
            elif letter == 'd':
                self.duration = max(0, self.duration + int(p.duration * m.value))
            elif letter == 'e':
                self.effect = max(0.0, self.effect + m.value * p.effect)
            elif letter == 'f':
                self.flags |= FILTER_LETTERS.get(second, 0)
            elif letter == 'i':
                self.flags |= ITEM_LETTERS.get(second, 0)
            elif letter == 'm':
                self.mana = max(0.0, self.mana + m.value * p.mana)
            elif letter == 'r':
                self.range = max(0.0, self.range + m.value * p.range)
            elif letter == 't':
                self.targets += int(p.targets * m.value)
            self.slots += 1
            self.complexity += m.level
            self.mana += m.mana
            self.price += m.price
        if self.area > 0.0:
            self.area = math.sqrt(self.area / math.pi)
        self.mana = max(self.mana, p.mana)

    @property
    def radius(self):
        """
        The radius of the area, 0 for a spell of one target.
        """
        return self.area

    def affects(self, model_name, hostile):
        """
        68b480: whether the filter lets the spell work on a unit of the model, hostile to the caster or not.
        """
        flags = self.flags & FILTERS_MASK
        if not flags:
            return True
        if flags & FILTER_FRIENDS and hostile:
            return False
        if flags & FILTER_ENEMIES and not hostile:
            return False
        for bit, models in RACE_MODELS:
            if flags & bit and model_name in models:
                return False
        return True

    def hit(self):
        """
        683910: the packet of the blow of the spell, the parts of the damage are all 0 for a spell which does
        not hurt.
        """
        parts = [0.0] * combat.DAMAGE_TYPES_COUNT
        damage = DAMAGE_SPELLS.get(self.index)
        lasting = False
        if damage is not None:
            parts[damage[0]] = 1.0
            lasting = damage[1]
        roll = self.effect
        factor = 1.0
        if self.duration > 1:
            factor = LASTING_SHARE / self.duration
            roll *= factor
        hit = combat.Hit(parts, roll, location=combat.ALL_PARTS, factor=factor, attack=combat.SURE_ATTACK)
        hit.lasting = lasting
        return hit

    def hurts(self):
        return self.index in DAMAGE_SPELLS

    def heals(self):
        return self.index == HEALING_SPELL

    def enchants(self):
        return self.index in ENCHANTMENT_SPELLS

    def provokes(self):
        return self.index in HOSTILE_SPELLS

    def sends_missile(self):
        return self.index in MISSILE_SPELLS

    def mana_cost(self, mind=PLAIN_MIND):
        """
        54ee10: the mana which a unit of the player spends, +0x20 of its object +0x1e0 is the mind
        (UnitRpg::ManaFactor 523cd0 gives 25 / mind).
        """
        return self.mana * PLAIN_MIND / mind if mind > 0 else self.mana

    def can_cast(self, skill):
        """
        554f50: a unit of the player can cast the spell when its skill of the school is not below the complexity.
        """
        return int(round(skill)) >= self.complexity

    @property
    def school_group(self):
        subtype = self.prototype.subtype_id
        return SCHOOL_GROUPS[subtype] if 0 <= subtype < len(SCHOOL_GROUPS) else None

#------------------------------------------------------------------------------


def visual_strength(value, base):
    """
    67f830: 2^(0.33 * log2(value / base within 1..8)) - 0.2, 0.8 for the plain spell, 1 without a base.
    """
    if base == 0.0:
        return 1.0
    share = min(max(value / base, 1.0), VISUAL_MOST)
    return 2.0 ** (VISUAL_POWER * math.log2(share)) - VISUAL_LEAST


def missile_particles(spell, position, rng=None):
    """
    67f830 cases 0 and 40, 681b70: the smoke behind the missile of a spell (its scale is the strength of the
    visual) and the flash where it hits (0.7 of it), None for the ones not made.
    """
    strength = visual_strength(spell.effect, spell.prototype.effect)
    trail = None
    code = MISSILE_TRAILS.get(spell.index)
    if code is not None:
        trail = particles.Emitter(code, position, scale=strength, rng=rng)
    flash = None
    code = HIT_FLASHES.get(spell.index)
    if code is not None:
        flash = (code, strength * HIT_FLASH_SCALE)
    return trail, flash


def flash_light(spell, position):
    """
    681b70: the light where a missile hits or a fireball bursts, 1.5 * the radius, rising 1.5 times faster.
    """
    return SpellLight.of(spell, position, scale=FLASH_LIGHT_SCALE,
                         velocity=(0.0, 0.0, fade_speed(spell, FLASH_LIGHT_SCALE)))


def unit_particles(spell, unit, rng=None):
    """
    67f830 case 24: the healing hangs its particles (0x2006) on the unit it heals (679240), their circle
    is by the box of the unit (485280: the middle of its height, the half of its diagonal).
    """
    if spell.index != HEALING_SPELL:
        return None
    emitter = particles.Emitter(particles.HEALING, (unit.x, unit.y, unit.z),
                                scale=visual_strength(spell.effect, spell.prototype.effect), rng=rng)
    size = unit.size()
    radius = 0.5 * math.sqrt(sum(float(v) * float(v) for v in size)) if size is not None else 0.1
    emitter.holder = (unit.height() * 0.5, radius)
    return emitter


def has_light(spell):
    return any(spell.prototype.color)


def fade_speed(spell, scale=1.0):
    """
    67f830, 682610: the rise of a light which goes out, radius / fadeout of the prototype a tick.
    """
    prototype = spell.prototype
    return prototype.light_radius / max(prototype.fadeout_time, 1.0) * scale

#------------------------------------------------------------------------------


class SpellSound(object):
    """
    A sound of a spell at a place: the name in sfx.res and the distances of its volume.
    """

    def __init__(self, name, position, near=SOUND_NEAR, far=SOUND_FAR):
        self.name = name
        self.position = tuple(position)
        self.near = near
        self.far = far

    def __repr__(self):
        return 'SpellSound(%s at %.1f,%.1f,%.1f)' % ((self.name, ) + tuple(self.position[:3]))


def sound_of(table, spell, position):
    name = table.get(spell.index)
    return SpellSound(name, position) if name else None


def start_sound(spell, position):
    """
    67f830: the sound of the cast (the strike of Invoke Lightning is heard over the whole map).
    """
    sound = sound_of(START_SOUNDS, spell, position)
    if sound is not None and spell.index == INVOKE_LIGHTNING_SPELL:
        sound.near = sound.far = SOUND_EVERYWHERE
    return sound


def circle_sound(spell, position):
    """
    67f620: the sound which loops while the missile flies or the effect lasts.
    """
    return sound_of(CIRCLE_SOUNDS, spell, position)


def end_sound(spell, position):
    """
    681b70, 682610: the sound of the hit or of the end of the effect.
    """
    return sound_of(END_SOUNDS, spell, position)


def enchantment_sounds(seen, fighter, position):
    """
    5112a0: the sounds of the enchantments which came to the unit or left it since the last look; seen is
    the set of the spells on it then, the new set comes back with the sounds.
    """
    now = set(e.index for e in fighter.enchantments) if fighter is not None else set()
    found = []
    for index, suffix in [(i, 'start') for i in sorted(now - seen)] + [(i, 'end') for i in sorted(seen - now)]:
        folder = ENCHANTMENT_SOUNDS.get(index)
        if folder:
            found.append(SpellSound('%s\\%s.wav' % (folder, suffix), position))
    return found, now


def enchantment_loop(fighter, position):
    """
    5114d0 case 32: the feeblemind loops its circle while it is on the unit.
    """
    if fighter is not None and any(e.index == FEEBLEMIND_SPELL for e in fighter.enchantments):
        return SpellSound(ENCHANTMENT_SOUNDS[FEEBLEMIND_SPELL] + '\\circle.wav', position)
    return None


def enchantment_warning(fighter, position):
    """
    510890: magic\\end1.wav when an enchantment of a unit of the party is about to end.
    """
    if fighter is None or not fighter.in_party:
        return None
    if any(0 < e.ticks < ENCHANTMENT_WARNING_TICKS for e in fighter.enchantments):
        return SpellSound(ENCHANTMENT_WARNING_SOUND, position, WARNING_NEAR, WARNING_FAR)
    return None

#------------------------------------------------------------------------------


class SpellLight(object):
    """
    The point light of a spell (CEffectPointLight, 67f780): its point moves by its speed every tick; a light
    which rises goes away when it has risen its radius (its light no longer reaches the ground below it).
    """

    def __init__(self, position, color, radius, velocity=(0.0, 0.0, 0.0)):
        self.position = tuple(float(v) for v in position)
        self.last = self.position
        self.color = tuple(float(v) for v in color)
        self.radius = abs(float(radius))
        self.velocity = tuple(float(v) for v in velocity)
        self.risen = 0.0

    def __repr__(self):
        return 'SpellLight(%.1f, %.1f, %.1f r %.2f)' % (self.position + (self.radius,))

    @classmethod
    def of(cls, spell, position, scale=1.0, extra_radius=0.0, velocity=(0.0, 0.0, 0.0)):
        """
        The light of the spell or None when its prototype has no color.
        """
        if not has_light(spell):
            return None
        return cls(position, spell.prototype.color, spell.prototype.light_radius * scale + extra_radius, velocity)

    @property
    def alive(self):
        return self.risen < self.radius

    def rise(self, speed):
        self.velocity = (0.0, 0.0, speed)

    def update(self):
        self.last = self.position
        vx, vy, vz = self.velocity
        x, y, z = self.position
        self.position = (x + vx, y + vy, z + vz)
        if vz > 0.0:
            self.risen += vz
        return self.alive

    def at(self, fraction):
        return tuple(a + (b - a) * fraction for a, b in zip(self.last, self.position))

#------------------------------------------------------------------------------


class Spot(object):
    """
    A point of the ground which a spell of the point (the target "t") is cast at, it stands for a unit
    in the chase of the caster.
    """

    is_dead = False
    fighter = None

    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return 'Spot(%.1f, %.1f)' % (self.x, self.y)


class AreaEffect(object):
    """
    The effect of a spell on the ground (CEffect of 67e690 with its states 67f830, 681b70, 682610): it hits the
    units around its point by its own clock. update() gives back the (unit, attacker) pairs which the blow
    of the spell (683910) comes to; after the first blow of a lasting one the attacker is None (+0xc | 1),
    so the victims do not answer it.
    """

    def __init__(self, spell, caster, origin, point, ground=None, rng=None):
        self.spell = spell
        self.caster = caster
        self.hit = spell.hit()
        self.radius = spell.radius
        self.point = (point[0], point[1])
        self.position = (origin[0], origin[1])
        # called with (x, y): the height of the ground, for the particles
        self.ground = ground or (lambda x, y: 0.0)
        self.rng = rng
        # engine.particles.Emitter of the visual, they live on after the effect while they have particles
        self.emitters = []
        self.smoke = None
        self.height = 0.0
        self.strength = visual_strength(spell.effect, spell.prototype.effect)
        self.velocity = (0.0, 0.0)
        self.direction = (0.0, 1.0)
        self.started = False
        self.done = False
        self.counter = 0
        self.quiet = False
        self.ticks = 0.0
        self.origin = (origin[0], origin[1])
        # [engine.lightning.Lightning, the ticks it has left or None while the effect lasts]
        self.lightnings = []
        # the whole ticks since the start
        self.age = -1
        # SpellLight, at most one (+0x98 of the effect)
        self.lights = []
        # SpellSound which came and wait for the client to play them, the one which loops (+0x94)
        self.sounds = []
        self.loop = None
        dx, dy = point[0] - origin[0], point[1] - origin[1]
        distance = math.hypot(dx, dy)
        if distance > 0.0:
            self.direction = (dx / distance, dy / distance)
        if spell.index == FIREBALL_SPELL:
            speed = spell.speed if spell.speed > 0.0 else DEFAULT_SPEED
            # 67f830 case 3: it bursts when the counter goes below 0, so on the tick distance / speed * 15
            self.counter = int(round(distance / speed * SPEED_TICKS)) - 2
            step = speed / SPEED_TICKS
            self.velocity = (self.direction[0] * step, self.direction[1] * step)
        elif spell.index == ACID_COLUMN_SPELL:
            self.position = self.point
            self.counter = COLUMN_TICKS
        else:
            self.position = self.point
            self.counter = spell.duration

    def __repr__(self):
        return 'AreaEffect(%s at %.1f, %.1f)' % (self.spell.name, self.point[0], self.point[1])

    @property
    def finished(self):
        """
        The effect is over and its particles are gone.
        """
        return (self.done and not self.emitters and not self.lightnings and not self.lights and
                self.age >= self.last_strike_tick())

    def last_strike_tick(self):
        if self.spell.index == INVOKE_LIGHTNING_SPELL:
            return SECOND_STRIKE_TICK
        return 0

    def update(self, ticks, units):
        """
        Runs the whole ticks which came, the blows of them.
        """
        self.ticks += ticks
        blows = []
        while self.ticks >= 1.0 and not self.finished:
            self.ticks -= 1.0
            self.update_lightnings()
            if not self.done:
                before = list(self.emitters)
                blows.extend(self.tick(units))
                if self.done:
                    # the particles of the effect fade, the ones of its end (the burst) only start
                    for emitter in before:
                        emitter.stop()
                    if self.spell.index in WALL_SPELLS + (ACID_FOG_SPELL,):
                        # 682610: the light rises away when the counter comes to -1, the end sounds
                        for light in self.lights:
                            light.rise(fade_speed(self.spell))
                        self.play_end(self.point)
            self.update_emitters()
            self.lights = [light for light in self.lights if light.update()]
        return blows

    def update_lightnings(self):
        """
        682610: the bolts of the last tick go when their ticks are over (the one of the wall when the effect
        is done); the invoke lightning strikes once more on its third tick.
        """
        self.age += 1
        kept = []
        for bolt, left in self.lightnings:
            bolt.update()
            if left is None:
                if not self.done:
                    kept.append([bolt, None])
            elif left > 1:
                kept.append([bolt, left - 1])
        self.lightnings = kept
        if self.spell.index == INVOKE_LIGHTNING_SPELL and self.age == SECOND_STRIKE_TICK:
            self.strike(STRIKE_POWER)

    def strike(self, power):
        """
        67f830 case 4: a bolt from 30 above the point to 3 below its ground.
        """
        x, y = self.point
        z = self.ground(x, y) - STRIKE_DEPTH
        bolt = lightning.Lightning((x, y, z + STRIKE_HEIGHT), (x, y, z), power, rng=self.rng, time=self.age)
        self.lightnings.append([bolt, STRIKE_TICKS])

    def bolts(self):
        return [bolt for bolt, left in self.lightnings]

    def play_end(self, point):
        """
        The sound of the end takes the place of the loop (+0x94).
        """
        self.loop = None
        x, y = point
        sound = end_sound(self.spell, (x, y, self.ground(x, y)))
        if sound is not None:
            self.sounds.append(sound)

    def start_sounds(self, position):
        index = self.spell.index
        if index == FIREBALL_SPELL:
            speed = self.spell.speed if self.spell.speed > 0.0 else DEFAULT_SPEED
            distance = math.hypot(self.point[0] - self.origin[0], self.point[1] - self.origin[1])
            if distance >= speed * FIREBALL_LOOP_FLIGHT:
                self.sounds.append(SpellSound(FLYING_FIREBALL_SOUND, position))
                self.loop = circle_sound(self.spell, position)
            else:
                self.sounds.append(start_sound(self.spell, position))
            return
        sound = start_sound(self.spell, position)
        if sound is not None:
            self.sounds.append(sound)
        self.loop = circle_sound(self.spell, position)

    def set_light(self, light):
        """
        67f780: the new light takes the place of the old one (67ef00 puts it into the world, 67edc0 takes
        it away).
        """
        self.lights = [light] if light is not None else []

    def emitter(self, code, position, scale=1.0, intensity=1.0):
        """
        CEffectSpell::MakeParticle (67f220): the effect 0x4a (0x5a for the walls 0x2010 and 0x2014) of the code.
        """
        emitter = particles.Emitter(code, position, scale=scale, intensity=intensity, rng=self.rng,
                                    ground=self.ground)
        self.emitters.append(emitter)
        return emitter

    def start_visual(self):
        """
        67f830: the particles of the start.
        """
        index = self.spell.index
        x, y = self.point
        ground = self.ground(x, y)
        if index == FIREBALL_SPELL:
            ox, oy = self.position
            self.height = self.ground(ox, oy) + FIREBALL_HEIGHT
            self.start_sounds((ox, oy, self.height))
        else:
            self.start_sounds((x, y, ground))
        if index == FIREBALL_SPELL:
            self.smoke = self.emitter(particles.FIRE_SMOKE, (ox, oy, self.height),
                                      scale=self.strength * FIREBALL_SMOKE_SCALE)
            # the light flies with the ball
            self.set_light(SpellLight.of(self.spell, (ox, oy, self.height),
                                         velocity=(self.velocity[0], self.velocity[1], 0.0)))
        elif index == INVOKE_LIGHTNING_SPELL:
            self.strike(STRIKE_POWER * self.strength)
            self.set_light(SpellLight.of(self.spell, (x, y, ground + STRIKE_LIGHT_HEIGHT),
                                         velocity=(0.0, 0.0, fade_speed(self.spell))))
            self.emitter(particles.TENDRI_LIGHT, (x, y, ground), scale=self.radius, intensity=self.strength)
        elif index == ACID_COLUMN_SPELL:
            self.emitter(particles.ACID_COLUMN, (x, y, ground))
            self.set_light(SpellLight.of(self.spell, (x, y, ground), velocity=(0.0, 0.0, fade_speed(self.spell))))
        elif index == WALL_SPELLS[0]:
            # 67f830 case 6: the wall stands across the way, the radius to both sides (679bf0)
            wall = self.emitter(particles.FIRE_WALL, (x, y, ground), scale=self.strength, intensity=self.strength)
            nx, ny = self.direction
            wall.half = (-ny * self.radius, nx * self.radius, 0.0)
        elif index == WALL_SPELLS[1]:
            # 67f830 case 7: the bolt between the ends of the wall
            nx, ny = self.direction
            ends = []
            for side in (-1.0, 1.0):
                ex, ey = x - ny * self.radius * side, y + nx * self.radius * side
                ends.append((ex, ey, self.ground(ex, ey) + WALL_LIGHTNING_HEIGHT))
            bolt = lightning.Lightning(ends[0], ends[1], 0.0, rng=self.rng, time=self.age)
            self.lightnings.append([bolt, None])
        elif index == ACID_FOG_SPELL:
            self.emitter(particles.ACID_FOG, (x, y, ground), scale=self.radius, intensity=self.strength)
        if index in WALL_SPELLS + (ACID_FOG_SPELL,):
            # 67f830 cases 6, 7, 8: the light covers the area
            self.set_light(SpellLight.of(self.spell, (x, y, ground + GROUND_LIGHT_HEIGHT), extra_radius=self.radius))

    def burst_visual(self):
        """
        681b70 case 3: the fireball bursts, the larger strength of the effect and of the area.
        """
        if self.smoke is not None:
            self.smoke.stop()
        x, y = self.point
        area = self.radius * self.radius * math.pi
        scale = max(visual_strength(area, self.spell.prototype.area), self.strength)
        self.emitter(particles.EXPLOSION, (x, y, self.ground(x, y) + FIREBALL_HEIGHT), scale=scale)
        self.set_light(flash_light(self.spell, (x, y, self.ground(x, y) + FIREBALL_HEIGHT)))
        self.play_end(self.point)

    def update_emitters(self):
        if self.smoke is not None and not self.done:
            x, y = self.position
            px, py = self.point
            # the height goes from that of the caster to that over the point
            total = math.hypot(px - self.origin[0], py - self.origin[1]) or 1.0
            t = min(math.hypot(x - self.origin[0], y - self.origin[1]) / total, 1.0)
            z = self.height + (self.ground(px, py) + FIREBALL_HEIGHT - self.height) * t
            self.smoke.position = (x, y, z)
            if self.loop is not None:
                # the sound flies with the ball
                self.loop.position = (x, y, z)
        self.emitters = [e for e in self.emitters if e.update()]

    def tick(self, units):
        index = self.spell.index
        if not self.started:
            # 67f830: the start
            self.started = True
            self.start_visual()
            if index == INVOKE_LIGHTNING_SPELL:
                self.done = True
                return self.circle(units)
            return []
        if index == FIREBALL_SPELL:
            # 681b70 case 3
            counter = self.counter
            self.counter -= 1
            if counter >= 0:
                self.position = (self.position[0] + self.velocity[0], self.position[1] + self.velocity[1])
                return []
            self.position = self.point
            self.done = True
            self.burst_visual()
            return self.circle(units)
        if index == ACID_COLUMN_SPELL:
            # 682610 case 5
            counter = self.counter
            self.counter -= 1
            if counter < 0:
                self.done = True
            elif self.counter == COLUMN_HIT_TICK:
                return self.circle(units)
            return []
        if index == CAMPFIRE_SPELL:
            # 682610 case 39: the counter grows, the fire burns until it is taken away
            counter = self.counter
            self.counter += 1
            if counter % LASTING_HIT_STEP == 0:
                return self.circle(units)
            return []
        # 682610 cases 6, 7, 8: the walls and the fog, the visual fades 80 ticks more
        counter = self.counter
        self.counter -= 1
        if counter < 0:
            self.done = True
            return []
        if counter % LASTING_HIT_STEP:
            return []
        if index in WALL_SPELLS:
            blows = self.strip(units)
        else:
            blows = self.circle(units)
        self.quiet = True
        return blows

    def attacker(self):
        if self.quiet:
            return None
        return self.caster

    def passes(self, unit):
        if unit.is_dead or unit.fighter is None:
            return False
        caster = self.caster
        if caster is None:
            return True
        return self.spell.affects(unit.figure.model_name, caster.is_hostile(unit))

    def circle(self, units):
        """
        6835c0: the units nearer than the radius to the point (5c0bd0 measures on the ground).
        """
        attacker = self.attacker()
        x, y = self.point
        return [(u, attacker) for u in units
                if (u.x - x) ** 2 + (u.y - y) ** 2 < self.radius * self.radius and self.passes(u)]

    def strip(self, units):
        """
        6836f0: the units of the circle not farther than 0.5 from the line of the wall, which stands
        across the way from the caster to the point.
        """
        attacker = self.attacker()
        x, y = self.point
        nx, ny = self.direction
        blows = []
        for u in units:
            dx, dy = u.x - x, u.y - y
            if dx * dx + dy * dy >= self.radius * self.radius:
                continue
            if abs(dx * nx + dy * ny) > WALL_HALF_WIDTH:
                continue
            if self.passes(u):
                blows.append((u, attacker))
        return blows


#------------------------------------------------------------------------------


class LightningArc(object):
    """
    The arc of the lightning (1, 41): CEffectLightning made by 67f3d0, its ends follow the hand of the caster
    and the target every tick (67a8b0), it goes after 7 ticks (682610 case 1). The blow came at once.
    """

    def __init__(self, spell, caster, target, rng=None):
        self.spell = spell
        self.caster = caster
        self.target = target
        self.hit = None
        self.emitters = []
        self.left = ARC_TICKS
        self.ticks = 0.0
        self.age = 0
        strength = visual_strength(spell.effect, spell.prototype.effect)
        bolt = lightning.Lightning(self.hand(), self.aim(), ARC_POWER * strength,
                                   demon=spell.index == DEMON_ARC_SPELL, rng=rng)
        self.lightnings = [[bolt, None]]
        # 67f830 case 1: the light at the target
        light = SpellLight.of(spell, self.aim())
        self.lights = [light] if light is not None else []
        sound = start_sound(spell, self.hand())
        self.sounds = [sound] if sound is not None else []
        self.loop = None

    def __repr__(self):
        return 'LightningArc(%s, %d ticks left)' % (self.spell.name, self.left)

    @property
    def finished(self):
        return self.left <= 0 and not self.lights

    def hand(self):
        c = self.caster
        return (c.x, c.y, c.z + c.height() * ARC_HAND_HEIGHT)

    def aim(self):
        t = self.target
        return (t.x, t.y, t.z + ARC_TARGET_HEIGHT)

    def update(self, ticks, units):
        self.ticks += ticks
        while self.ticks >= 1.0 and not self.finished:
            self.ticks -= 1.0
            self.left -= 1
            self.age += 1
            if self.left == 0:
                # 682610 case 1: the arc goes, its light rises away
                self.lightnings = []
                for light in self.lights:
                    light.rise(fade_speed(self.spell))
            for bolt, left in self.lightnings:
                bolt.update()
                bolt.follow(self.hand(), self.aim())
            self.lights = [light for light in self.lights if light.update()]
        return []

    def bolts(self):
        return [bolt for bolt, left in self.lightnings]


class SpellParticles(object):
    """
    The particles of a spell at its target (67f830 cases 0x12 the fireworks, 0x14 the clairvoyance, 0x1b the link
    from the caster): the emitter follows the target, stops after the duration and goes when its particles die.

    The lasting state 682610 counts the duration down (+0xa0, the counter here):
    - 0x12 the fireworks: while the counter is not below 0 every unit in the circle remembers the threat at the
      point with the weight of the effect (5cbbe0); at -1 magic/Firework/end sounds;
    - 0x14 the clairvoyance: while it lasts the units in the circle of a caster of the player are seen by
      the player (4b2ef0, see sight_circle); at 10 magic/Clairvoyence/end sounds;
    - 0x1b the link: it breaks with magic/Link/end when the caster or the target dies or the counter goes
      below 0 (a caster of the player also gets the message 0x19 of 552780, not in the remake).
    58d3d0 around the point marks the objects there for the client of the player (the remake draws them all).
    """

    def __init__(self, spell, caster, target, rng=None, ground=None):
        self.spell = spell
        self.caster = caster
        self.target = target
        self.hit = None
        self.left = max(int(spell.duration), SPELL_PARTICLES_TICKS)
        self.counter = int(spell.duration)
        self.broken = False
        self.ticks = 0.0
        self.lightnings = []
        self.loop = None
        strength = visual_strength(spell.effect, spell.prototype.effect)
        origin = (caster.x, caster.y, caster.z + caster.height() * ARC_HAND_HEIGHT)
        emitter = particles.spell_emitter(spell.index, self.aim(), origin=origin, scale=strength, rng=rng,
                                          ground=ground)
        self.emitters = [emitter] if emitter is not None else []
        light = SpellLight.of(spell, self.aim(), velocity=(0.0, 0.0, fade_speed(spell)))
        self.lights = [light] if light is not None else []
        sound = start_sound(spell, self.aim())
        self.sounds = [sound] if sound is not None else []
        # 67f620: the link loops its circle while it lasts
        self.loop = circle_sound(spell, self.aim())

    def __repr__(self):
        return 'SpellParticles(%s, %d ticks left)' % (self.spell.name, self.left)

    @property
    def finished(self):
        return not self.emitters and not self.lights

    def aim(self):
        t = self.target
        return (t.x, t.y, t.z)

    def update(self, ticks, units):
        self.ticks += ticks
        while self.ticks >= 1.0 and not self.finished:
            self.ticks -= 1.0
            self.left -= 1
            self.counter -= 1
            self.last(units)
            if self.left == 0:
                self.stop()
            if self.loop is not None:
                self.loop.position = self.aim()
            self.emitters = [e for e in self.emitters if e.update(self.aim())]
            self.lights = [light for light in self.lights if light.update()]
        return []

    def stop(self):
        for emitter in self.emitters:
            emitter.stop()
        self.loop = None

    def end(self):
        sound = end_sound(self.spell, self.aim())
        if sound is not None:
            self.sounds.append(sound)

    def last(self, units):
        """
        682610 cases 0x12, 0x14, 0x1b: a tick of the counter.
        """
        index = self.spell.index
        if index == FIREWORKS_SPELL:
            if self.counter >= 0:
                point = self.aim()
                weight = int(self.spell.effect)
                for u in units:
                    brain = getattr(u, 'brain', None)
                    if brain is None or u.is_dead:
                        continue
                    if math.hypot(u.x - point[0], u.y - point[1]) <= self.spell.radius:
                        brain.remember_threat(point, weight, FIREWORKS_THREAT_STEP)
            elif self.counter == -1:
                self.end()
        elif index == CLAIRVOYANCE_SPELL:
            if self.counter == CLAIRVOYANCE_END_TICK:
                self.end()
        elif index == LINK_SPELL and not self.broken:
            if self.caster.is_dead or self.target.is_dead or self.counter < 0:
                self.broken = True
                self.end()
                self.left = 0

    def sight_circle(self):
        """
        682610 case 0x14: (point, radius) of the circle which the clairvoyance shows to the player of its
        caster while it lasts, else None.
        """
        if self.spell.index != CLAIRVOYANCE_SPELL or self.counter < 0:
            return None
        return self.aim(), self.spell.radius

    def sight_eye(self):
        return None

    def bolts(self):
        return []


class Possession(object):
    """
    The possession (21, 682610 case 0x15): while the counter (+0xa0, the duration) lasts and the caster
    belongs to the player, the target (a unit) goes to the list of the units the player sees (+0x50f8 of
    the player, 4b2ef0) together with the units in its lists of the seen (+0xa90, +0xaac of its logic, 4b31d0),
    it becomes the unit the player looks through (+0x24 of the player, the flag 0x2000000 of the target) and
    58d3d0 marks the objects around it in max(sight * light, life sense) * 2 + 20 for the client. The remake
    lets the player see the target and what it sees (engine.ai.Controller.update_player_sight); the effect
    has no particles, the start sound comes with the spell on the target.
    """

    def __init__(self, spell, caster, target):
        self.spell = spell
        self.caster = caster
        self.target = target
        self.hit = None
        self.counter = int(spell.duration)
        self.ticks = 0.0
        self.sounds = []
        self.loop = None
        self.emitters = []
        self.lights = []
        self.lightnings = []

    def __repr__(self):
        return 'Possession(%s, %d ticks left)' % (self.target, self.counter)

    @property
    def finished(self):
        return self.counter < 0 or self.target is None or self.target.is_dead

    def update(self, ticks, units):
        self.ticks += ticks
        while self.ticks >= 1.0 and not self.finished:
            self.ticks -= 1.0
            self.counter -= 1
        return []

    def sight_circle(self):
        return None

    def sight_eye(self):
        """
        The unit through which the player of the caster sees, None when the possession is over.
        """
        return None if self.finished else self.target

    def bolts(self):
        return []


class Teleport(object):
    """
    The teleport (28): the effect of 67f830 case 28 and 682610 case 28 takes its caster (+0x20) to the point
    (+0x24); 67ec70 fills these fields. At the cast the sparks rise where the caster stands (0x2019 with the scale 1, the emitter of the
    effect +0x90) and fall into the point (the scale -1), the sound and the light are at the point. After
    30 ticks, when the caster lives and can stand at the point (5b7a40), it is there (5c11d0): the rising
    sparks stop, the freeze of the spell is put on it for the duration, the light rises away. Otherwise
    it gets the message 0x10 (552780) and stays.
    """

    def __init__(self, spell, caster, point, ground=None, can_stand=None, rng=None):
        self.spell = spell
        self.caster = caster
        self.hit = None
        self.point = (point[0], point[1])
        self.ground = ground or (lambda x, y: 0.0)
        # called with (unit, x, y): whether the unit can stand there
        self.can_stand = can_stand
        self.ticks = 0.0
        self.counter = max(spell.duration, 1) + TELEPORT_DELAY
        self.moved = False
        self.lightnings = []
        x, y = self.point
        z = self.ground(x, y)
        cx, cy = caster.x, caster.y
        self.rising = particles.Emitter(particles.TELEPORT, (cx, cy, self.ground(cx, cy)), rng=rng)
        falling = particles.Emitter(particles.TELEPORT, (x, y, z - TELEPORT_SINK), scale=-1.0, rng=rng)
        self.emitters = [self.rising, falling]
        sound = start_sound(spell, (x, y, z))
        self.sounds = [sound] if sound is not None else []
        self.loop = None
        light = SpellLight.of(spell, (x, y, z))
        self.lights = [light] if light is not None else []

    def __repr__(self):
        return 'Teleport(%s to %.1f, %.1f)' % (self.caster, self.point[0], self.point[1])

    @property
    def finished(self):
        return self.counter < 0 and not self.emitters and not self.lights

    def bolts(self):
        return []

    def update(self, ticks, units):
        self.ticks += ticks
        while self.ticks >= 1.0 and not self.finished:
            self.ticks -= 1.0
            self.tick()
            self.emitters = [e for e in self.emitters if e.update()]
            self.lights = [light for light in self.lights if light.update()]
        return []

    def tick(self):
        """
        682610 case 28.
        """
        counter = self.counter
        self.counter -= 1
        if self.counter == -1 and not self.moved:
            # the end of the effect: the light of a teleport which did not happen rises away too
            # (the remake's choice, 67ef80 is not read)
            for light in self.lights:
                light.rise(fade_speed(self.spell))
        caster = self.caster
        if counter < 0 or caster is None or caster.is_dead:
            return
        if self.counter != max(self.spell.duration, 1):
            return
        x, y = self.point
        if self.can_stand is not None and not self.can_stand(caster, x, y):
            if _Debug:
                print('%s can not stand at %.1f, %.1f' % (caster, x, y))
            return
        caster.teleport_to(x, y)
        self.moved = True
        self.rising.stop()
        if caster.fighter is not None:
            caster.fighter.enchant(self.spell.index, self.spell.duration, self.spell.effect)
            caster.refresh_senses()
        for light in self.lights:
            light.rise(fade_speed(self.spell))


def spread(spell, caster, main, units):
    """
    58e850: the targets of a spell, units or the spot of the point. One target, the main one, when the spell
    has no filter and fewer than 2 targets. Otherwise a spell of the point makes its effect at the main
    one first and the rest go to the living units within the range of the caster which pass the filter:
    the main one first, then the nearest ones to the main one. The sight of the side of the caster
    (+0x12c64, 4b3330) and the trace of 54bbc0 are not checked here.
    """
    if spell.index in UNSPREAD_SPELLS:
        return []
    count = spell.targets
    if not spell.flags & FILTERS_MASK and count < 2:
        return [main]
    targets = []
    if spell.flags & TARGET_POINT and not spell.flags & ITEM_CONTINUOUS:
        count -= 1
        targets.append(main)
    cx, cy = (caster.x, caster.y) if caster is not None else (main.x, main.y)
    candidates = [u for u in units if not u.is_dead and u.fighter is not None and
                  (u.x - cx) ** 2 + (u.y - cy) ** 2 < spell.range * spell.range]
    while count > 0 and candidates:
        if main in candidates:
            chosen = main
        else:
            chosen = min(candidates, key=lambda u: (u.x - main.x) ** 2 + (u.y - main.y) ** 2 +
                         (u.z - main.z) ** 2)
        candidates.remove(chosen)
        hostile = caster.is_hostile(chosen) if caster is not None else False
        if spell.affects(chosen.figure.model_name, hostile):
            targets.append(chosen)
            count -= 1
    return targets


def apply(spell, target):
    """
    67f830: the healing or the record on the list of the spells of the target (engine.combat.Fighter),
    the health which came back for the healing.
    """
    fighter = target.fighter
    if fighter is None:
        return 0.0
    if spell.heals():
        return combat.heal(fighter, spell.effect)
    if spell.enchants():
        fighter.enchant(spell.index, spell.duration, spell.effect)
    return 0.0


def senses(base_senses, base_detections, fighter):
    """
    524560: the senses and the detections of a unit with the spells on it.
    """
    result = list(base_senses)
    detections = list(base_detections)
    if fighter is None or not fighter.enchantments:
        return tuple(result), tuple(detections)
    for index, sense in SENSE_SPELLS.items():
        if sense < len(result):
            result[sense] += fighter.enchantment(index)
    for index, (sense, factor) in DETECTION_SPELLS.items():
        if sense < len(detections):
            value = detections[sense] + factor * fighter.enchantment(index)
            detections[sense] = value if factor > 0 else max(0.0, value)
    if any(e.index == VISION_FOG_SPELL for e in fighter.enchantments) and result:
        result[0] *= VISION_FOG_FACTOR
    return tuple(result), tuple(detections)


def is_frozen(fighter):
    return fighter is not None and any(e.index in FREEZING_SPELLS for e in fighter.enchantments)


def is_charmed(fighter):
    """
    The charm spell (36): the unit is not at war and does not fight while it lasts.
    """
    return fighter is not None and any(e.index == CHARM_SPELL for e in fighter.enchantments)


def make_spell(spells, name, modifier_codes=()):
    """
    A spell of the database (idb.SpellDatabase) by the name or the code of the prototype and the codes
    of the modifiers, None when there is no such prototype. game.exe builds it by CSpellConstructor:
    BeginConstruction (6841a0), AddModifier (684260) for every modifier, EndConstruction (684210).
    """
    prototype = spells.spell(name)
    if prototype is None:
        return None
    return Spell(prototype, [spells.modifier(code) for code in modifier_codes])
