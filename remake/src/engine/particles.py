"""
The particles of game.exe (CParticle, 0x484000-0x499000).

An emitter (CParticle) is made by its type, a code 0x2000..0x2052 (CParticle::vf02, 485540): the type sets
the numbers of the emitter and its three functions, the birth of a particle (+0x1e0), its move (+0x1e4)
and the move of the sources (+0x1e8), the records of 0x38 bytes at +0x154 which some types fly around
(the fragments of an explosion, the tendrils of a lightning). The texture has the name of the type
(textures.res "FireSmoke"), it is an atlas of +0xe0 = 4 columns, +0xdc frames of it are used (16 by default).

An emitter runs once a tick (CParticle::vf06, 496eb0):
    - its point is that of the effect which holds it, the step since the last tick is its move (+0xa4),
      its length +0xc8; the side (+0xb0, the move turned in the ground) and the up (+0xbc) make a basis;
    - the sources move (+0x1e8);
    - every particle keeps its last state (for the drawing between the ticks) and moves (+0x1e4),
      the one whose move gives 0 dies;
    - while the emitter emits (bit 1 of +8) it makes max(round(step / +0xd4), +0xd0) tries, each one gives
      a particle (+0x1e0) with the chance +0xcc of 100 while there are fewer than +0xd8 of them.
An emitter which does not emit lives while it has particles.

A particle (the numbers of its record of floats):
    [0..2] the point, [3] the size, [4..7] the point and the size of the last tick, [8..10] the speed,
    [11..16] the numbers of its type, [0x11] the frame, [0x13] the ticks of the frame, [0x14] the ticks
    of its life, [0x15] the kind within the type, [0x16] the color 0xAARRGGBB, [0x17] the last color.
The drawing (4976b0) puts a square of the size (the half of its side here, the remake's reading) facing
the camera at the point between the last and the new state by the part of the tick which went by,
with the frame of the last tick; +0x150 = 1 draws it adding the light (the healing, the casts,
the tendrils of the lightning).

The effect of a spell (CEffectParticle, 678e80) gives its emitter the scale (+0x15c of the effect divides
the speeds, the sizes and the step, 485480) and the intensity (+0x118, its alpha for the clouds).

The random numbers of game.exe come from a table of 256 (7a9d38) refilled by 49ee70: rnd() * 2^-32
is [0, 1), * 2^-31 - 1 is [-1, 1); here a random.Random gives them.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math
import random

#------------------------------------------------------------------------------

# the types of the particles (CParticle::vf02) which the spells of the remake make
FIRE_SMOKE = 0x2000
EXPLOSION = 0x2002
ACID_FOG = 0x2007
POISON_CLOUD = 0x2008
FIREWORK = 0x2009
HEALING = 0x2006
ACID_COLUMN = 0x200c
FIRE_WALL = 0x2010
LINK = 0x2014
CLAY_RING = 0x2018
TELEPORT = 0x2019
FIRE_ARROW_SMOKE = 0x2011
TENDRI_LIGHT = 0x2030
FIRE_ARROW_HIT = 0x2041
ACID_RAY_HIT = 0x2042
RIK_ARROW_SMOKE = 0x204e
PORTAL_STAR = 0x203d
PORTAL = 0x203e
START_TRANS = 0x2051
TRANSFORM = 0x2052

# the texture of a type (500a00(6, name))
TEXTURE_NAMES = {
    FIRE_SMOKE: 'firesmoke',
    EXPLOSION: 'explosion',
    ACID_FOG: 'poisoncloud',
    POISON_CLOUD: 'poisoncloud',
    FIREWORK: 'fireworks',
    HEALING: 'healing',
    ACID_COLUMN: 'acidray',
    FIRE_WALL: 'tendrismoke',
    LINK: 'link',
    CLAY_RING: 'clayring',
    TELEPORT: 'teleport',
    FIRE_ARROW_SMOKE: 'firearrowsmoke',
    TENDRI_LIGHT: 'tendrilight',
    FIRE_ARROW_HIT: 'firearrowsmoke',
    ACID_RAY_HIT: 'acidray',
    RIK_ARROW_SMOKE: 'rikarrowsmoke',
    PORTAL_STAR: 'casting6',
    START_TRANS: 'zoneexit',
    TRANSFORM: 'changehero',
    PORTAL: 'teleport',
}

# CreateParticleSource (4bd9a0): the names of the types (78b120, compared without the case) -> (code, ticks of the
# record of the script world, -1 for ever)
SCRIPT_TYPES = {
    'acidray': (0x2012, -1), 'firearrow': (0x2011, -1), 'fireball': (0x2000, -1), 'campfire': (0x2001, -1),
    'fireblast': (0x2002, 200), 'lightningblast': (0x2030, 200), 'fire': (0x2003, -1), 'smoke': (0x2004, -1),
    'vulcansmoke': (0x2005, -1), 'healing': (0x2006, 200), 'poisonfog': (0x2007, -1), 'geyser': (0x2009, -1),
    'tornado': (0x200a, -1), 'casting': (0x200b, 200), 'nuke': (0x200c, 200), 'mushroom': (0x200e, 200),
    'link': (0x2014, -1), 'bluegas': (0x2013, -1), 'sphereacid': (0x2015, -1), 'sphereelectricity': (0x2016, -1),
    'spherefire': (0x2017, -1), 'clayring': (0x2018, 200), 'teleport': (0x2019, 200),
    'modifier1': (0x201b, 200), 'modifier2': (0x201c, 200), 'modifier3': (0x201d, 200), 'modifier4': (0x201e, 200),
    'modifier5': (0x201f, 200), 'modifier6': (0x2020, 200), 'modifier7': (0x2021, 200), 'modifier8': (0x2022, 200),
    'modifier9': (0x2023, 200), 'modifier10': (0x2024, 200), 'modifier11': (0x2025, 200),
    'modifier12': (0x2026, 200), 'antimagic': (0x201a, -1), 'castingfire': (0x2027, -1),
    'castingelectricity': (0x2028, -1), 'castingacid': (0x2029, -1), 'castingdivination': (0x202a, -1),
    'castingillusion': (0x202b, -1), 'castingdomination': (0x202c, -1), 'castingenchantment': (0x202d, -1),
    'castinghealing': (0x202e, -1), 'zoneexit': (0x2038, -1), 'portal': (0x203e, -1), 'portalstar': (0x203d, -1),
    'cylinder1': (0x203f, 200), 'cylinder2': (0x2040, 200), 'firestar': (0x2041, 20), 'acidstar': (0x2042, 20),
    'sparks': (0x2043, 60), 'visionstar1': (0x2044, -1), 'visionstar2': (0x2045, -1), 'visionstar3': (0x2046, -1),
    'regeneration': (0x2047, -1), 'silence': (0x2048, -1), 'feeblemind': (0x2049, -1), 'feetcloud1': (0x204a, -1),
    'feetcloud2': (0x204b, -1), 'cursestars': (0x204f, -1), 'curseholder': (0x2050, -1), 'starttrans': (0x2051, 200),
    'transform': (0x2052, 200),
}

# 0x203d PortalStar (493d10, 493d50): one star at the point whose size is (0.8 + 0.4 * random) * scale every tick
PORTAL_STAR_SIZE = 0.8
PORTAL_STAR_FLICKER = 0.4

# 0x203e Portal (494470, 494770, 494930): the sparks circle around the point and float up, a glow stays in the middle
PORTAL_LEAST = 10.0             # the births of a tick: scale * 10
PORTAL_RADIUS = 2.5
PORTAL_TURN = 0.1               # radians a tick, the inner sparks turn the other way
PORTAL_SIZE = 0.3               # the size (2 - r) * 0.3 at the part r of the radius
PORTAL_RISE = 0.015             # the speed up (4 - r) * 2.5 * 0.015, * 1.03 every tick
PORTAL_RISE_GROWTH = 1.03
PORTAL_HEIGHT = 8.0             # a spark goes away above it, its alpha goes down on the way
PORTAL_ALPHA = 200.0
PORTAL_GLOW_HEIGHT = 6.4        # 8 * 0.8 over the point
PORTAL_GLOW_SIZE = 1.2          # + 0.6 * random, only at the scale 1 and above
PORTAL_GLOW_FLICKER = 0.6
PORTAL_GLOW_COLOR = 0x60ffffff
PORTAL_SPARK_COLOR = 0x40ffffff
PORTAL_INNER_FRAMES = 4         # the frames 0..3 of the inner sparks, 4..14 of the outer ones
PORTAL_FRAMES = 15

# the columns of the atlas of a texture (+0xe0)
ATLAS_COLUMNS = 4

WHITE = 0xffffffff
# 0x00ffffff: white and clear
CLEAR = 0x00ffffff

# 0078a5dc: the alpha of the cloud of an explosion by its frame
EXPLOSION_ALPHAS = (255, 255, 207, 175, 128, 128, 128, 128, 128, 175, 207, 192, 128, 48, 0, 0)
# 0078a9f8: the alpha (<< 22, so a quarter of it) of a spark of the lightning by the ticks of its life left
TENDRI_ALPHAS = (0, 80, 176, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255)

# the lowest move of an emitter which gives a direction (_DAT_0073cdf0)
STILL_STEP = 1e-05

# 0x2006 Healing (48b630, 48bbe0, 48bce0): the balls fly around the unit for 36 ticks (the radius grows for
# the first 18, then shrinks), turning 10 degrees a tick, from the bottom of its sphere to the top
HEALING_TICKS = 36
HEALING_TURN = math.radians(10.0)
# the kind of a ball which fades: the alpha goes down by 31 a tick for 8 ticks
HEALING_FADING = 0x10000
HEALING_FADE_TICKS = 8

# 0x2051 StartTrans (vf02 at 485540, the birth 4965e0 of ZoneExit, 496840, 496990): the sparks are born on a ball
# of the radius scale around the point and come into it in 18 ticks, the first particle is a glow in the middle;
# the source emits for 31 ticks (+0x34 from 30), then leaves its holder (4849f0)
START_TRANS_TICKS = 30
START_TRANS_LIFE = 15
START_TRANS_GLOW_LIFE = 4
START_TRANS_GLOW_FRAME = 12
START_TRANS_TRIES = 10
START_TRANS_SPARK_SIZE = 0.05       # 73c52c times the scale
START_TRANS_GROWTH = 0.02           # 73ccd8 times the scale a tick: the size of the glow
START_TRANS_MORE = 0.4              # 73b868 a tick: the births of a tick, up to 8
START_TRANS_MOST_BIRTHS = 8
START_TRANS_GLOW_SHRINK = 0.6       # 73cde0
START_TRANS_FLICKER = 0.4           # 73ccb0 = 0.4 / 2^32 and 73cca8: the glow is (0.8 .. 1.2) * its size
START_TRANS_FLICKER_BASE = 0.8
# 78acf8: the alpha of a particle by its ticks left
LIFE_ALPHA = (0, 80, 176, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 128, 64)

# 0x2052 Transform (the mushroom 0x200e too; 4889d0, 488db0, 488f00): one source rises 0.12 * scale a tick and
# widens by ln(ticks + 2) * 0.2 * scale for 30 ticks; the particles fly out of it in all the directions upward
TRANSFORM_TICKS = 30
TRANSFORM_RISE = 0.12               # 73ccc0
TRANSFORM_WIDTH = 0.2               # 73cce0
TRANSFORM_SPEED = 0.06              # 73ccb8
TRANSFORM_UP = 0.1                  # 73b850: the upward part of the direction from 0.1 to 1.0 (73ccc8 = 0.9 / 2^32)
TRANSFORM_UP_SPREAD = 0.9
TRANSFORM_SIZE = 0.1                # 73b850 + ticks * 0.008 (73ccbc)
TRANSFORM_SIZE_GROWTH = 0.008
TRANSFORM_GROW = 1.1                # 73bac4 while the frame is below 8
TRANSFORM_SLOW = 0.7                # 73c538
TRANSFORM_PULL = 0.02               # 73ccd8 times the scale: to the axis while below the top of the source
TRANSFORM_COLOR = 0x40ffffff        # 7.9999995 as a float
TRANSFORM_LAST_FRAME = 14

# 0x2019 Teleport (the firework 0x2009 too; 48fdb0, 490060, 490110): it emits for 36 ticks; with the scale
# above 0 the sparks rise faster and faster and fade, below 0 they fall into the point and show up
TELEPORT_TICKS = 35
TELEPORT_RISE = 1.2
TELEPORT_FALL = 0.8333333
TELEPORT_RISE_SPEED = 0.15
TELEPORT_FALL_SPEED = 2.25
# the alpha of a falling spark grows by 19 a tick from 1, of a rising one goes down by 1
TELEPORT_SHOW = 19

# 0x2009 Fireworks (48a8f0, 48ada0): three fountains of sparks which fly apart, fall and fade
FIREWORK_LEAST = 20
FIREWORK_CHANCE = 88
FIREWORK_LIFE = 45
FIREWORK_FRAME_DELAY = 2
FIREWORK_SIZE = 0.3
FIREWORK_SPEED = 4.0
FIREWORK_FADE = 1.03
FIREWORK_GRAVITY = 0.3
FIREWORK_FRAMES = 16

# 0x2014 Link (48d8d0, 48de30): a stream of sparks between two points
LINK_LEAST = 10
LINK_CHANCE = 80
LINK_LIFE = 45
LINK_FRAME_DELAY = 3
LINK_RADIUS = 0.2

# 0x2018 ClayRing (48f7f0, 48fab0): a ring of glowing clay particles, additive
CLAY_RING_MOST = 80
CLAY_RING_LIFE = 15
CLAY_RING_FRAME_DELAY = 2
CLAY_RING_SIZE = 0.35

#------------------------------------------------------------------------------


class Particle(object):

    __slots__ = ('x', 'y', 'z', 'size', 'last', 'vx', 'vy', 'vz', 'b', 'c', 'd', 'ex', 'ey', 'ez',
                 'frame', 'last_frame', 'frame_ticks', 'life', 'kind', 'color', 'last_color', 'dead')

    def __init__(self):
        self.x = self.y = self.z = 0.0
        self.size = 1.0
        self.last = (0.0, 0.0, 0.0, 0.0)
        self.vx = self.vy = self.vz = 0.0
        # [0xb], [0xc], [0xd]: the numbers of a type (the next size, the frame of floats, its growth)
        self.b = self.c = self.d = 0.0
        # [0xe], [0xf], [0x10]: the second speed of a type
        self.ex = self.ey = self.ez = 0.0
        self.frame = 0
        self.last_frame = 0
        self.frame_ticks = 0
        self.life = 0
        self.kind = 0
        self.color = WHITE
        self.last_color = WHITE
        self.dead = False

    def keep(self):
        self.last = (self.x, self.y, self.z, self.size)
        self.last_frame = self.frame
        self.last_color = self.color

    def restore(self):
        self.x, self.y, self.z, self.size = self.last
        self.frame = self.last_frame
        self.color = self.last_color


class Source(object):
    """
    A record of 0x38 bytes at +0x154 of an emitter: a point which flies by its own rule (+0x1e8).
    """

    def __init__(self):
        # +0x00 the point, +0x0c the speed, +0x18.. the numbers of a type, +0x30 and +0x34 its counters
        self.p = [0.0, 0.0, 0.0]
        self.v = [0.0, 0.0, 0.0]
        self.f = [0.0] * 6
        self.counter = 0
        self.ticks = 0


class Emitter(object):
    """
    CParticle: the particles of one type around a point.
    """

    def __init__(self, code, position, scale=1.0, intensity=1.0, rng=None, ground=None):
        self.code = code
        self.rng = rng or random.Random()
        # called with (x, y): the height of the ground (CWorldClient::vf28)
        self.ground = ground or (lambda x, y: 0.0)
        self.position = tuple(position)
        self.current = tuple(position)
        self.move = (0.0, 0.0, 0.0)
        self.step = 0.0
        self.side = (0.0, 0.0, 0.0)
        self.up = (0.0, 0.0, 0.0)
        self.particles = []
        self.sources = []
        self.emitting = True
        self.only_attached = False
        self.keep_alive = False
        # +0x140..+0x14c: the wind (7af5cc), none in the remake
        self.wind = (0.0, 0.0, 0.0, 0.0)
        # +0x130..+0x138: the half of a wall
        self.half = (0.0, 0.0, 0.0)
        self.additive = False
        # (the height of the middle, the radius) of the box of the unit which holds the emitter (485280)
        self.holder = None
        # +0x94: the emitter is on its holder (StartTrans leaves it)
        self.attached = True
        self.texture_name = TEXTURE_NAMES.get(code)
        self.first_emitted = False
        self.setup(code)
        self.set_scale(scale)
        self.intensity = intensity

    def __repr__(self):
        return 'Emitter(0x%x, %d particles)' % (self.code, len(self.particles))

    def random(self):
        return self.rng.random()

    def uniform(self, low, high):
        return low + (high - low) * self.rng.random()

    def setup(self, code):
        """
        CParticle::vf02 (485540): the numbers of the type, after the defaults of 4847e0.
        """
        self.chance = 100
        self.least = 0
        self.distance = 1000.0
        self.most = 500
        self.frames = 16
        self.frame_delay = 10000
        self.life = -1
        self.spread = [0.0] * 6
        self.offset = 0.0
        self.size = 1.0
        self.lift = 0.0
        self.fade = 0.0
        self.scale = 1.0
        self.gravity = 0.0
        self.link_from = None
        if code == FIRE_SMOKE:
            self.life = 15
            self.frames = 15
            self.spread = [-0.2, -0.2, -1.0, 0.2, 0.2, 0.0]
            self.chance = 88
            self.least = 1
            self.distance = 0.09
            self.frame_delay = 1
            self.size = 0.8
            self.fade = 0.94
            self.born, self.moves = self.born_fire_smoke, self.move_fire_smoke
        elif code == FIRE_WALL:
            self.spread = [-0.2, -0.2, 0.0, 0.2, 0.2, 0.0]
            self.chance = 88
            self.least = 20
            self.most = 1500
            self.life = 32
            self.frames = 15
            self.frame_delay = 1
            self.size = 0.8
            self.lift = 0.3
            self.fade = 0.85
            self.born, self.moves = self.born_fire_wall, self.move_fire_wall
            self.sources = [Source()]
            self.source_moves = self.source_fire_wall
        elif code == EXPLOSION:
            self.spread = [-0.02, -0.02, -0.02, 0.02, 0.02, 0.0]
            self.chance = 100
            self.least = 60
            self.most = 600
            self.life = 15
            self.frame_delay = 1
            self.offset = 0.02
            self.size = 0.6
            self.lift = 1.0
            self.fade = 0.8
            self.born, self.moves = self.born_explosion, self.move_explosion
            self.source_moves = self.source_explosion
            for _ in range(15):
                self.sources.append(self.explosion_fragment())
        elif code in (ACID_FOG, POISON_CLOUD):
            self.chance = 50
            self.least = 1
            self.most = 0x7fffffff
            self.life = 0xca4
            self.frames = 15
            self.frame_delay = 1
            self.born, self.moves = self.born_cloud, self.move_cloud
        elif code == ACID_COLUMN:
            self.least = 0x80
            self.most = 400
            self.life = -1
            self.frame_delay = 1
            self.lift = 1.0
            self.born, self.moves = self.born_acid_column, self.move_acid_column
            self.source_moves = self.source_acid_column
            self.sources = [Source()]
        elif code in (RIK_ARROW_SMOKE, FIRE_ARROW_SMOKE):
            self.life = 15
            self.frames = 15
            self.spread = [-0.2, -0.2, -0.2, 0.2, 0.2, 0.0]
            self.least = 1
            self.distance = 0.06
            self.frame_delay = 1
            self.offset = 0.2
            self.size = 0.3
            self.fade = 1.03
            self.born, self.moves = self.born_arrow_smoke, self.move_arrow_smoke
        elif code == TENDRI_LIGHT:
            self.life = 15
            self.frames = 15
            self.spread = [-0.2, -0.2, -0.2, 0.2, 0.2, 0.0]
            self.chance = 88
            self.least = 250
            self.most = 600
            self.offset = 0.2
            self.frame_delay = 1
            self.size = 1.0
            self.additive = True
            self.born, self.moves = self.born_tendri_light, self.move_tendri_light
            self.source_moves = self.source_tendri_light
        elif code == HEALING:
            # bits 3 of +8, +0x150 = 1: the light adds; the birth makes the sources and sets the tries
            self.additive = True
            self.least = 0x7fffffff
            self.life = 8
            self.frame_delay = 1
            self.size = 0.5
            self.head = 0
            self.turn_growth = 0.0
            self.size_growth = 0.0
            self.born, self.moves = self.born_healing, self.move_healing
            self.source_moves = self.source_healing
        elif code == PORTAL_STAR:
            self.least = 1
            self.most = 1
            self.born, self.moves = self.born_portal_star, self.move_portal_star
        elif code == PORTAL:
            self.most = 800
            self.life = 10
            self.frame_delay = 3
            self.born, self.moves = self.born_portal, self.move_portal
        elif code == FIREWORK:
            # 485540 case 0x2009: 3 sources, 20 births, sparks of the atlas of the fireworks
            self.spread = [-0.15, -0.15, -0.04, 0.15, 0.15, 0.04]
            self.chance = FIREWORK_CHANCE
            self.least = FIREWORK_LEAST
            self.most = 500
            self.life = FIREWORK_LIFE
            self.frames = FIREWORK_FRAMES
            self.frame_delay = FIREWORK_FRAME_DELAY
            self.size = FIREWORK_SIZE
            self.fade = FIREWORK_FADE
            self.additive = True
            self.gravity = FIREWORK_GRAVITY
            self.born, self.moves = self.born_firework, self.move_firework
            self.sources = [Source(), Source(), Source()]
            self.source_moves = self.source_firework
        elif code == LINK:
            # 485540 case 0x2014: the sparks fly along the line between the two units
            self.spread = [-0.2, -0.2, -0.2, 0.2, 0.2, 0.2]
            self.chance = LINK_CHANCE
            self.least = LINK_LEAST
            self.most = 200
            self.life = LINK_LIFE
            self.frames = 15
            self.frame_delay = LINK_FRAME_DELAY
            self.size = 1.0
            self.additive = True
            self.link_from = None
            self.born, self.moves = self.born_link, self.move_link
        elif code == CLAY_RING:
            # 485540 case 0x2018: a burst of glowing particles around the point
            self.chance = 100
            self.least = CLAY_RING_MOST
            self.most = 500
            self.life = CLAY_RING_LIFE
            self.frames = 16
            self.frame_delay = CLAY_RING_FRAME_DELAY
            self.size = CLAY_RING_SIZE
            self.additive = True
            self.born, self.moves = self.born_clay_ring, self.move_clay_ring
        elif code == TELEPORT:
            self.least = 20
            self.most = 320
            self.life = 14
            self.frame_delay = 1
            self.born, self.moves = self.born_teleport, self.move_teleport
            self.source_moves = self.source_teleport
            self.sources = [Source()]
        elif code == START_TRANS:
            # bits 0x13 of +8 and +0x13c = 7: on its holder at the point "bd"
            self.least = 1
            self.most = 300
            self.born, self.moves = self.born_start_trans, self.move_start_trans
            self.source_moves = self.source_start_trans
            source = Source()
            source.ticks = START_TRANS_TICKS
            source.f[1] = 1.0
            self.sources = [source]
        elif code == TRANSFORM:
            self.least = 10
            self.most = 4000
            self.life = -1
            self.frame_delay = 1
            self.size = 1.0
            self.born, self.moves = self.born_transform, self.move_transform
            self.source_moves = self.source_transform
        elif code in (FIRE_ARROW_HIT, ACID_RAY_HIT):
            # bits 0x13 of +8: emits, only on its holder, keeps living
            self.only_attached = False
            self.least = 1
            self.life = 15
            self.born, self.moves = self.born_hit, self.move_hit
        else:
            raise ValueError('unknown type of particles 0x%x' % code)

    def set_scale(self, scale):
        """
        485480: the scale multiplies the spread, the offset, the size, the lift and the step of the birth.
        """
        self.spread = [s * scale for s in self.spread]
        self.offset *= scale
        self.size *= scale
        self.lift *= scale
        self.distance *= scale
        self.scale *= scale
        if self.code == PORTAL:
            # 494930 every tick
            self.least = int(self.scale * PORTAL_LEAST)

    #------------------------------------------------------------------------------

    def update(self, position=None):
        """
        CParticle::vf06 (496eb0): one tick. False when the emitter is over.
        """
        if position is not None:
            self.position = tuple(position)
        x, y, z = self.position
        cx, cy, cz = self.current
        self.move = (x - cx, y - cy, z - cz)
        self.current = (x, y, z)
        mx, my, mz = self.move
        self.step = math.sqrt(mx * mx + my * my + mz * mz)
        if self.step <= 0.0:
            self.side = (0.0, 0.0, 0.0)
            self.up = (0.0, 0.0, 0.0)
        else:
            sx, sy, sz = my, -mx, 0.0
            length = math.hypot(sx, sy)
            if length < 1e-6:
                sx, sy, sz = my - mz, -mx, mx
                length = math.sqrt(sx * sx + sy * sy + sz * sz)
            sx, sy, sz = sx / length, sy / length, sz / length
            ux, uy, uz = my * sz - mz * sy, mz * sx - mx * sz, mx * sy - my * sx
            length = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
            self.side = (sx, sy, sz)
            self.up = (ux / length, uy / length, uz / length)
        if self.sources and self.source_moves is not None:
            for source in self.sources:
                self.source_moves(source)
        alive = []
        for p in self.particles:
            p.keep()
            if not self.moves(p):
                p.dead = True
                p.restore()
                continue
            alive.append(p)
        self.particles = alive
        tries = max(int(round(self.step / self.distance)) if self.distance > 0.0 else 0, self.least)
        if self.emitting:
            made = 0
            for index in range(tries):
                if len(self.particles) >= self.most or not self.emitting:
                    break
                if self.rng.randrange(100) < self.chance:
                    # the birth gets the number of the particles made before it in this tick
                    self.emit(made)
                    made += 1
            if not made and self.chance > 50 and tries > 0 and len(self.particles) < self.most:
                self.emit(0)
        if self.emitting or self.keep_alive:
            return True
        return bool(self.particles)

    source_moves = None

    def emit(self, index):
        p = Particle()
        if self.born(p, index):
            p.keep()
            self.particles.append(p)

    def stop(self):
        self.emitting = False

    #------------------------------------------------------------------------------
    # 0x2009 Fireworks (48a8f0, 48ada0)

    def source_firework(self, source):
        """
        The three fountains of the type turn and wait between bursts (48aec0); a small pause is enough here.
        """
        source.counter += 1
        if source.counter % 9 == 0:
            source.ticks = 1

    def born_firework(self, p, index):
        cx, cy, cz = self.current
        p.x = cx + self.uniform(self.spread[0], self.spread[3])
        p.y = cy + self.uniform(self.spread[1], self.spread[4])
        p.z = cz + self.uniform(self.spread[2], self.spread[5])
        angle = self.uniform(0.0, 2.0 * math.pi)
        elevation = self.uniform(0.1, 1.3)
        speed = self.uniform(1.0, 1.0 + FIREWORK_SPEED) * self.scale
        p.vx = math.cos(angle) * math.cos(elevation) * speed
        p.vy = math.sin(angle) * math.cos(elevation) * speed
        p.vz = math.sin(elevation) * speed
        p.frame = self.rng.randrange(12)
        p.frame_ticks = self.frame_delay
        p.life = FIREWORK_LIFE
        p.size = FIREWORK_SIZE * self.scale
        p.color = WHITE
        return True

    def move_firework(self, p):
        p.x += p.vx
        p.y += p.vy
        p.z += p.vz
        p.vz -= FIREWORK_GRAVITY
        p.size *= FIREWORK_FADE
        p.life -= 1
        p.frame_ticks -= 1
        if p.frame_ticks <= 0:
            p.frame_ticks = self.frame_delay
            p.frame = (p.frame + 1) % FIREWORK_FRAMES
        return p.life > 0 and p.size > 0.01

    #------------------------------------------------------------------------------
    # 0x2014 Link (48d8d0, 48de30)

    def born_link(self, p, index):
        cx, cy, cz = self.current
        if self.link_from is None:
            p.x, p.y, p.z = cx, cy, cz
        else:
            t = self.random()
            x0, y0, z0 = self.link_from
            p.x = x0 + (cx - x0) * t
            p.y = y0 + (cy - y0) * t
            p.z = z0 + (cz - z0) * t
        p.x += self.uniform(self.spread[0], self.spread[3])
        p.y += self.uniform(self.spread[1], self.spread[4])
        p.z += self.uniform(self.spread[2], self.spread[5])
        p.size = self.size * self.scale
        p.frame = self.rng.randrange(15)
        p.frame_ticks = self.frame_delay
        p.life = LINK_LIFE
        p.color = WHITE
        return True

    def move_link(self, p):
        cx, cy, cz = self.current
        dx, dy, dz = cx - p.x, cy - p.y, cz - p.z
        length = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        speed = LINK_RADIUS * self.scale
        p.x += dx / length * speed
        p.y += dy / length * speed
        p.z += dz / length * speed
        p.life -= 1
        p.frame_ticks -= 1
        if p.frame_ticks <= 0:
            p.frame_ticks = self.frame_delay
            p.frame = (p.frame + 1) % 15
        return p.life > 0

    #------------------------------------------------------------------------------
    # 0x2018 ClayRing (48f7f0, 48fab0)

    def born_clay_ring(self, p, index):
        cx, cy, cz = self.current
        angle = self.uniform(0.0, 2.0 * math.pi)
        radius = self.uniform(0.5, 1.0) * self.scale
        p.x = cx + math.cos(angle) * radius
        p.y = cy + math.sin(angle) * radius
        p.z = cz + self.uniform(0.0, 0.2)
        p.vx = math.cos(angle) * 0.02 * self.scale
        p.vy = math.sin(angle) * 0.02 * self.scale
        p.vz = self.uniform(0.0, 0.05) * self.scale
        p.size = CLAY_RING_SIZE * self.scale
        p.frame = self.rng.randrange(16)
        p.frame_ticks = self.frame_delay
        p.life = CLAY_RING_LIFE
        p.color = WHITE
        return True

    def move_clay_ring(self, p):
        p.x += p.vx
        p.y += p.vy
        p.z += p.vz
        p.size *= 1.01
        p.life -= 1
        p.frame_ticks -= 1
        if p.frame_ticks <= 0:
            p.frame_ticks = self.frame_delay
            p.frame = (p.frame + 1) % 16
        return p.life > 0

    #------------------------------------------------------------------------------
    # 0x203d PortalStar (493d10, 493d50)

    def born_portal_star(self, p, index):
        p.x, p.y, p.z = self.current
        p.size = 0.0
        p.frame = 0
        p.life = 15
        p.color = CLEAR
        return True

    def move_portal_star(self, p):
        if not self.emitting:
            return False
        p.color = WHITE
        p.size = (self.random() * PORTAL_STAR_FLICKER + PORTAL_STAR_SIZE) * self.scale
        return True

    #------------------------------------------------------------------------------
    # 0x203e Portal

    def born_portal(self, p, index):
        cx, cy, cz = self.current
        if not self.particles:
            # the glow in the middle (+0x1c0, the number of the particles, is 0)
            p.x, p.y, p.z = cx, cy, cz + PORTAL_GLOW_HEIGHT
            p.size = 0.0
            p.color = PORTAL_GLOW_COLOR
            p.kind = 1
            p.frame = PORTAL_FRAMES
            return True
        p.kind = 0
        p.color = PORTAL_SPARK_COLOR
        p.life = self.life
        p.frame_ticks = self.frame_delay
        u = self.random()
        r = u * (u + 1.0) * 0.5
        angle = self.random() * 2.0 * math.pi
        p.b = r * PORTAL_RADIUS * math.sin(angle)
        p.c = r * PORTAL_RADIUS * math.cos(angle)
        p.x, p.y, p.z = cx + p.b, cy + p.c, cz
        p.size = 0.0
        p.d = (2.0 - r) * PORTAL_SIZE
        if self.random() <= r:
            p.frame = self.rng.getrandbits(32) & 3
            turn = -PORTAL_TURN
        else:
            p.frame = self.rng.getrandbits(32) % 11 + PORTAL_INNER_FRAMES
            turn = PORTAL_TURN
        p.vx, p.vy = math.cos(turn), math.sin(turn)
        p.vz = (4.0 - r) * PORTAL_RADIUS * PORTAL_RISE
        p.z += (self.random() - 0.5) * p.vz
        return True

    def move_portal(self, p):
        if p.kind != 0:
            if not self.emitting:
                return False
            if self.scale >= 1.0:
                p.size = self.random() * PORTAL_GLOW_FLICKER + PORTAL_GLOW_SIZE
            else:
                p.size = 0.0
            return True
        if p.life == 0:
            return False
        cx, cy, cz = self.current
        b = p.vx * p.b + p.c * p.vy
        c = p.c * p.vx - p.b * p.vy
        p.b, p.c = b, c
        p.x, p.y = cx + b, cy + c
        p.frame_ticks -= 1
        p.z += self.scale * p.vz
        p.vz *= PORTAL_RISE_GROWTH
        if p.frame_ticks == 0:
            p.frame += 1
            if p.frame == PORTAL_INNER_FRAMES:
                p.frame = 0
            if p.frame == PORTAL_FRAMES:
                p.frame = PORTAL_INNER_FRAMES
            p.frame_ticks = self.frame_delay
        height = p.z - cz
        if height > PORTAL_HEIGHT:
            p.size = 0.0
            p.life = 0
            p.color = CLEAR
            return True
        p.size = p.d
        alpha = min((int(round(PORTAL_ALPHA / PORTAL_HEIGHT * (PORTAL_HEIGHT - height))) + 0x37) >> 2, 0x40)
        p.color = (alpha << 24) | 0xffffff
        return True

    #------------------------------------------------------------------------------
    # 0x2000 FireSmoke (48c800, 48cad0): the smoke behind the fireball

    def born_fire_smoke(self, p, index):
        p.color = WHITE
        p.frame = 0
        p.life = self.life
        p.frame_ticks = self.frame_delay
        p.x, p.y, p.z = self.current
        base = self.size * 0.5
        if index == 0:
            # the first one of a tick is a big puff which halves on its first move
            p.size = self.size * 1.2
            p.kind = 1
            t = 0.0
        else:
            p.size = base
            p.kind = 0
            t = self.random() - 1.0
        if self.step <= STILL_STEP:
            p.vx = p.vy = p.vz = 0.0
            return True
        mx, my, mz = self.move
        p.x += t * mx
        p.y += t * my
        p.z += t * mz
        self.side_speed(p, base)
        if p.kind == 0:
            p.x += p.vx
            p.y += p.vy
            p.z += p.vz
        return True

    def side_speed(self, p, base):
        low = self.spread[0] - base
        a = (base + self.spread[3] - low) * self.random() + low
        sx, sy, sz = self.side
        p.vx, p.vy, p.vz = a * sx, a * sy, a * sz
        low = self.spread[1] - base
        b = (base + self.spread[4] - low) * self.random() + low
        ux, uy, uz = self.up
        p.vx += b * ux
        p.vy += b * uy
        p.vz += b * uz

    def tick_frames(self, p):
        """
        The life and the frames of a particle: False when it is over.
        """
        p.life -= 1
        if p.life == 0:
            return False
        p.frame_ticks -= 1
        if p.frame_ticks == 0:
            p.frame += 1
            if p.frame >= self.frames:
                return False
            p.frame_ticks = self.frame_delay
        return True

    def move_fire_smoke(self, p):
        if not self.tick_frames(p):
            return False
        p.size *= self.fade
        if p.kind == 0:
            p.x -= p.vx
            p.y -= p.vy
            p.z -= p.vz
        else:
            p.size *= 0.5
            p.kind = 0
        p.vx *= 0.95
        p.vy *= 0.95
        p.vz *= 0.95
        p.x += p.vx
        p.y += p.vy
        p.z += p.vz
        return True

    #------------------------------------------------------------------------------
    # 0x204e RikArrowSmoke, 0x2011 FireArrowSmoke (48cba0, 48ced0): the smoke behind a missile

    def born_arrow_smoke(self, p, index):
        p.color = WHITE
        p.life = self.life
        p.frame_ticks = self.frame_delay
        p.x, p.y, p.z = self.current
        if index == 0 and not self.first_emitted:
            # the head: it stays in front of the missile
            self.first_emitted = True
            p.kind = 2
            p.frame = 0
            p.size = self.size * 0.4
            t = 0.0
        else:
            p.size = self.size * 0.2
            p.kind = 1 if self.rng.randrange(100) > 70 else 0
            p.frame = self.rng.randrange(2) + 2
            t = self.random() - 1.0
        if self.step <= STILL_STEP:
            p.vx = p.vy = p.vz = 0.0
            return True
        mx, my, mz = self.move
        p.x += t * mx
        p.y += t * my
        p.z += t * mz
        a = (self.spread[3] - self.spread[0]) * self.random() + self.spread[0]
        sx, sy, sz = self.side
        p.vx, p.vy, p.vz = a * sx, a * sy, a * sz
        b = (self.spread[4] - self.spread[1]) * self.random() + self.spread[1]
        ux, uy, uz = self.up
        p.vx += b * ux
        p.vy += b * uy
        p.vz += b * uz
        if p.kind == 0:
            p.x += p.vx
            p.y += p.vy
            p.z += p.vz
        return True

    def move_arrow_smoke(self, p):
        if p.kind == 2:
            mx, my, mz = self.move
            cx, cy, cz = self.current
            p.x, p.y, p.z = cx + mx, cy + my, cz + mz
            p.frame = self.rng.randrange(3)
            return self.emitting
        if not self.tick_frames(p):
            return False
        if p.kind != 0:
            if p.frame < 6:
                p.size *= 1.4
            p.vx *= 0.95
            p.vy *= 0.95
            p.vz *= 0.95
            return True
        if p.frame < 6:
            p.size *= 1.4
        else:
            p.size *= self.fade
        p.x -= p.vx
        p.y -= p.vy
        p.z -= p.vz
        p.vx *= 1.1
        p.vy *= 1.1
        p.vz *= 1.1
        p.x += p.vx
        p.y += p.vy
        p.z += p.vz
        return True

    #------------------------------------------------------------------------------
    # 0x2041, 0x2042 (494950, 4949b0): the flash where a fire arrow or an acid ray hits

    def born_hit(self, p, index):
        self.emitting = False
        p.color = CLEAR
        p.life = 3
        p.frame = self.life
        p.x, p.y, p.z = self.current
        p.size = 0.0
        p.vx = self.size
        return True

    def move_hit(self, p):
        p.life -= 1
        if p.life < 1:
            p.color = CLEAR
            p.size = 0.0
            return False
        p.color = WHITE
        p.size = p.vx
        p.vx *= 0.9
        return True

    #------------------------------------------------------------------------------
    # 0x2010 TendriSmoke (48e9e0, 48ec20, 48ed60, 48ee20): the fire of a wall

    def source_fire_wall(self, source):
        hx, hy, hz = self.half
        count = int(math.sqrt(hx * hx + hy * hy + hz * hz) * 14.285714)
        self.least = min(max(count, 6), self.most // 16)

    def born_fire_wall(self, p, index):
        p.color = WHITE
        p.life = self.life
        p.frame = 0
        p.frame_ticks = self.frame_delay
        p.size = self.size
        extra = 0.0
        if self.rng.randrange(100) + 1 < 41:
            p.size = self.size * 0.5
            extra = self.size * 0.3
        low = self.spread[0] - extra
        p.vx = (extra + self.spread[3] - low) * self.random() + low
        low = self.spread[1] - extra
        p.vy = (extra + self.spread[4] - low) * self.random() + low
        p.vz = 0.0
        p.vx *= 0.5
        p.vy *= 0.5
        cx, cy, cz = self.current
        p.x = cx + p.vx
        p.y = cy + p.vy
        p.z = self.random() * self.lift + cz
        p.ex = p.ey = 0.0
        p.ez = self.lift
        # 48ed60: along the wall, on the ground
        t = self.random() * 2.0 - 1.0
        p.x += t * self.half[0]
        p.y += t * self.half[1]
        p.z = p.z - cz + self.ground(p.x, p.y)
        return True

    def move_fire_wall(self, p):
        p.life -= 1
        if p.life == 0:
            return False
        p.frame_ticks -= 1
        if p.frame_ticks == 0:
            p.frame += 1
            if p.frame >= self.frames:
                return False
            p.frame_ticks = self.frame_delay
            if p.frame > 7:
                # the wall of fire (0x2010) and 0x2003 end their flames at the 8th frame
                return False
            if p.frame == 8:
                p.ez *= 0.5
        z = p.z
        p.z = p.ez + z
        if p.frame < 2:
            p.x += p.vx
            p.y += p.vy
            p.z = p.ez + z + p.vz
        if p.frame < 7:
            p.size *= self.fade
            return True
        p.size *= 1.06
        k = p.size * self.wind[3] * 0.15
        p.x += k * self.wind[0]
        p.y += k * self.wind[1]
        p.z += k * self.wind[2]
        return True

    #------------------------------------------------------------------------------
    # 0x2007, 0x2008 PoisonCloud (48b080, 48b570): a still cloud which fills its circle at once

    def born_cloud(self, p, index):
        if self.particles:
            return False
        radius = self.size
        n = int(radius + 1.0)
        cx, cy, cz = self.current
        for j in range(-n, n + 1):
            for i in range(-n, n + 1):
                x = self.uniform(-0.25, 0.25) + i
                y = self.uniform(-0.25, 0.25) + j
                if x * x + y * y >= radius * radius:
                    continue
                q = Particle()
                q.x, q.y = x + cx, y + cy
                q.size = self.random() * 0.5 + 1.0 - 0.25
                q.z = self.ground(q.x, q.y) + q.size
                if self.code == POISON_CLOUD:
                    q.color = WHITE
                else:
                    q.color = (min(int(round(self.intensity * 160.0)), 255) << 24) | 0xffffff
                q.life = self.life
                q.frame = self.rng.randrange(self.frames)
                q.frame_ticks = self.frame_delay
                q.keep()
                self.particles.append(q)
        return False

    def move_cloud(self, p):
        p.frame_ticks -= 1
        if p.frame_ticks == 0:
            p.frame += 1
            if p.frame >= self.frames:
                p.frame = 0
            p.frame_ticks = self.frame_delay
        if not self.emitting:
            # the cloud fades when its effect is over
            alpha = (p.color >> 24) & 0xff
            if alpha < 10:
                return False
            fade = max(int(round(self.intensity * 4.5)), 1)
            alpha = (alpha - fade) & 0xff
            tint = 0xffaf7f if self.code == POISON_CLOUD else 0xffffff
            p.color = (alpha << 24) | tint
        return True

    #------------------------------------------------------------------------------
    # 0x200c AcidRay (488f90, 489190, 4892e0): the column of acid, the drops rise and splash

    def source_acid_column(self, source):
        source.ticks += 1
        if self.least > 8:
            self.least //= 2
        if source.ticks == 13:
            self.emitting = False

    def born_acid_column(self, p, index):
        ticks = self.sources[0].ticks
        radius = self.random() * self.size * 0.2
        angle = self.random() * 2.0 * math.pi
        cx, cy, cz = self.current
        p.x = radius * math.sin(angle) + cx
        p.y = radius * math.cos(angle) + cy
        p.z = self.random() * self.size * 2.0 + cz
        p.size = self.size * 0.2
        p.ex = math.sin(angle)
        p.ey = math.cos(angle)
        p.ez = (11 - ticks) * self.size * 0.1
        p.life = 41 - ticks
        p.frame = ticks // 3
        p.kind = 0
        p.color = (min(int(4.0 ** ticks), 255) << 24) | 0xffffff
        return True

    def move_acid_column(self, p):
        p.life -= 1
        if p.life == 0:
            return False
        if p.kind == 0:
            p.ez = 0.85 * p.ez - 0.055
            p.z += p.ez
            alpha = min(int(((p.color >> 24) & 0xff) * 4.0), 255)
            p.color = (alpha << 24) | 0xffffff
            if p.ez < 0.0 and p.z < self.size * 0.5 + self.current[2]:
                # the drop splashes on the ground
                p.kind = 1
                p.ex *= abs(p.ez)
                p.ey *= abs(p.ez)
            return True
        if p.frame >= 14:
            return False
        p.frame += 1
        p.x += p.ex
        p.y += p.ey
        p.ex *= 0.85
        p.ey *= 0.85
        p.size *= 1.13
        alpha = ((p.color >> 24) & 0xff) - 0x10
        if alpha < 0:
            alpha = 0
        p.color = (alpha << 24) | 0xffffff
        return True

    #------------------------------------------------------------------------------
    # 0x2002 Explosion (48c090, 48c590, 48c760; the fragments of 485540): the burst of the fireball

    def explosion_fragment(self):
        source = Source()
        for _ in range(10):
            x = self.uniform(-1.0, 1.0)
            y = self.uniform(-1.0, 1.0)
            z = self.uniform(-1.0, 1.0) + 0.7
            length2 = x * x + y * y + z * z
            if length2 < 1.0:
                break
        length = math.sqrt(length2) or 1.0
        speed = (self.random() * 0.4 + 0.8) * 0.8
        source.v = [x / length * speed, y / length * speed, z / length * speed]
        # +0x18 the size of the core, +0x1c its speed, +0x20 the drag; +0x24 the smoke, +0x28 the visibility
        source.f = [0.0, 0.4, 0.7, float(self.rng.randrange(6)), float(self.rng.randrange(2)), 0.0]
        source.counter = 0
        return source

    def source_explosion(self, source):
        source.counter += 1
        source.f[0] += source.f[1]
        source.f[1] *= source.f[2]
        if source.counter > 6:
            self.emitting = False
        source.p[0] += self.scale * source.v[0]
        source.p[1] += self.scale * source.v[1]
        source.p[2] += self.scale * source.v[2]
        source.v = [source.v[0] * 0.75, source.v[1] * 0.75, source.v[2] * 0.75 - 0.03]

    def born_explosion(self, p, index):
        first = self.sources[0]
        p.color = WHITE
        p.kind = self.rng.randrange(29) - 14
        cx, cy, cz = self.current
        if not self.particles and first.counter == 1:
            # the flash in the middle
            p.kind = -1
            p.x, p.y, p.z = cx, cy, cz
            p.frame = 15
            p.size = 0.0
            p.life = -1
            return True
        if p.kind >= 0:
            # the smoke behind a fragment
            source = self.sources[p.kind]
            p.frame = self.rng.randrange(3)
            if source.f[3] == 0.0:
                p.frame = 8
            t = self.random()
            sx, sy, sz = source.p[0] + cx, source.p[1] + cy, source.p[2] + cz
            p.x = sx - t * source.v[0] * self.scale
            p.y = sy - t * source.v[1] * self.scale
            p.z = sz - t * source.v[2] * self.scale
            size = (source.f[0] * 0.142857 + 0.3) * (0.3 - t * 0.15) * (self.random() * 0.8 + 0.7)
            p.size = size * source.f[4] * self.scale
            p.vx, p.vy, p.vz = (source.v[0] * self.scale * 0.5, source.v[1] * self.scale * 0.5,
                                source.v[2] * self.scale * 0.5)
            p.life = 0
            return True
        # the fire of the core: a point within the ball
        p.life = 0
        for _ in range(10):
            x, y, z = self.uniform(-1.0, 1.0), self.uniform(-1.0, 1.0), self.uniform(-1.0, 1.0)
            if x * x + y * y + z * z < 1.0:
                break
        rest = 1.0 - (x * x + y * y + z * z)
        k = self.scale * first.f[0]
        p.x, p.y, p.z = cx + x * k, cy + y * k, cz + z * k
        p.size = 0.0
        k = self.scale * first.f[1]
        p.vx, p.vy, p.vz = x * k, y * k, z * k
        edge = first.counter // 7 + 0.2
        p.b = first.f[0] * self.scale * (self.random() * 0.4 + 0.8) * 0.6
        if rest < edge:
            p.frame = 8
            p.c = float(p.frame)
            p.d = self.random() * 0.35 + 0.35
            return True
        p.frame = self.rng.randrange(3)
        p.c = float(p.frame)
        p.d = rest * 0.35 / edge + self.random() * 0.35 * 0.5 + 0.175
        return True

    def move_explosion(self, p):
        if p.kind < 0:
            if p.frame < 14:
                p.life += 1
                p.x += p.vx
                p.y += p.vy
                p.z += p.vz
                p.size = p.b * (1.01 if p.frame < 8 else 1.03)
                p.c += p.d
                p.d *= 1.02
                p.frame = int(round(p.c))
                p.b = p.size
                drag = self.sources[0].f[2] - 0.03
                p.vx *= drag
                p.vy *= drag
                p.vz *= drag
                p.color = (EXPLOSION_ALPHAS[min(p.frame, 15)] << 24) | 0xffffff
                return True
            if p.life == -1:
                p.size = self.size * 2.3 if p.size == 0.0 else 0.0
                return self.sources[0].counter < 3
            return False
        p.frame += 1
        if p.frame & 7 == 7:
            return False
        p.color = WHITE
        p.x += p.vx
        p.y += p.vy
        p.z += p.vz
        p.vx *= 0.7
        p.vy *= 0.7
        p.vz *= 0.7
        p.size *= 0.8
        return True

    #------------------------------------------------------------------------------
    # 0x2030 TendriLight (4923b0, 492b80, 492cc0): the tendrils of the invoked lightning

    def tendrils(self):
        cx, cy, cz = self.current
        for index in range(10):
            radius = self.uniform(0.0, 1.0)
            angle = self.uniform(0.0, 6.2831855)
            x, y = math.sin(angle) * radius, math.cos(angle) * radius
            if index == 0:
                x = y = 0.0
            x, y = x * self.size * 0.3, y * self.size * 0.3
            source = Source()
            px, py = x + cx, y + cy
            pz = self.ground(px, py) + 0.3
            source.p = [px, py, pz]
            source.f = [px, py, pz, 0.0, 0.0, 0.0]
            k = 1.0 / math.sqrt(x * x + y * y + 0.25)
            source.v = [x * k * 0.5, y * k * 0.5, k * 0.5 * 0.5]
            self.sources.append(source)

    def source_tendri_light(self, source):
        self.least = 50
        source.f[0:3] = source.p[:]
        source.ticks += 1
        source.p = [source.p[i] + source.v[i] for i in range(3)]
        source.v[2] -= 0.075
        if source.ticks > 9:
            self.emitting = False

    def born_tendri_light(self, p, index):
        if not self.sources:
            self.tendrils()
        first = self.sources[0]
        p.color = WHITE
        if first.counter < len(self.sources) * 2:
            # two particles for every tendril: its head and its flash at the ground
            n = first.counter // 2
            p.frame = 0
            p.life = 10
            p.frame_ticks = first.counter & 1
            p.kind = n + 1
            source = self.sources[n]
            p.x, p.y, p.z = source.p
            p.size = 0.0
            p.b = 0.1
            first.counter += 1
            if p.frame_ticks:
                p.b = (self.random() * 0.9 + 0.5) * self.size * 0.2
                p.frame = self.rng.randrange(4) + 12
                if n == 0:
                    p.b = self.size * 0.4
                p.size = p.b
            return True
        p.kind = 0
        if len(self.particles) < 200 and first.ticks < 5:
            # the sparks around the middle
            p.color = 0x07ffffff
            p.frame = 8
            k = 1.0 - self.random() * self.random()
            angle = self.random() * 6.2831855
            p.ex = math.cos(angle) * k * self.size * 0.3
            p.ey = math.sin(angle) * k * self.size * 0.3
            p.ez = 0.0
            p.x = first.p[0] + p.ex
            p.y = first.p[1] + p.ey
            p.z = (1.0 - k) * self.size * 0.1 + first.p[2]
            p.size = 0.0
            p.frame_ticks = 0
            p.life = int(round(9.0 - k * 3.0))
            p.b = (2.0 - k) * (self.random() * 0.04 + 0.1) * self.size
            return True
        # a spark along a tendril
        source = self.sources[self.rng.randrange(len(self.sources))]
        p.frame = 1
        p.life = 6
        p.frame_ticks = 1
        p.b = self.random() * 0.01 * 3.0 + 0.06
        t = self.random()
        p.x, p.y, p.z = source.p
        p.size = 0.0
        p.vx = t * source.p[0] + (1.0 - t) * source.f[0]
        p.vy = t * source.p[1] + (1.0 - t) * source.f[1]
        p.vz = t * source.p[2] + (1.0 - t) * source.f[2]
        return True

    def move_tendri_light(self, p):
        p.life -= 1
        if p.life < 0:
            return False
        if p.kind == 0:
            p.color = ((TENDRI_ALPHAS[min(p.life, 15)] << 22) & 0xff000000) | 0xffffff
            if p.frame_ticks == 0:
                p.frame = (p.frame & ~3) | ((p.frame + 1) & 3)
                p.x += p.ex
                p.y += p.ey
                p.z += p.ez
                p.size = p.b
                p.b *= 0.9
                p.ex *= 0.6
                p.ey *= 0.6
                p.ez *= 0.6
                return True
            p.frame += 1
            p.x, p.y, p.z = p.vx, p.vy, p.vz
            p.size = p.b
            p.b *= 0.9
            return True
        if p.frame_ticks == 0:
            source = self.sources[p.kind - 1]
            p.x, p.y, p.z = source.p
            p.size = p.b
            return True
        # the flash at the ground blinks
        if p.size == 0.0:
            self.least = 50
            p.size = p.b
            return True
        p.size = 0.0
        return True


    #------------------------------------------------------------------------------
    # 0x2006 Healing (48b630, 48bbe0, 48bce0): the sources circle the unit and rise, a ball goes with each
    # of them, the sparks around the balls; the points are of the holder (+0x98 is its point)

    def healing_sources(self):
        """
        48b630 on the first birth: round(size * 6 + 2) sources on the circle around the middle of the unit,
        from its middle minus the radius, each one 0.2 of the radius higher or lower at random.
        """
        middle, radius = self.holder
        count = int(round(self.size * 6.0 + 2.0))
        bottom = middle - radius
        span = radius * 2.0
        self.turn_growth = radius * 1.1 / HEALING_TICKS * 2.0
        self.size_growth = (self.size + 1.0) / (HEALING_TICKS * 4.0)
        for index in range(count):
            source = Source()
            shift = self.uniform(-0.1 * span, 0.1 * span)
            source.f = [0.0, 0.0, bottom + shift, 0.0, 0.0, 0.0]
            source.p = source.f[0:3]
            # v: the radius, the angle, the rise a tick
            source.v = [0.0, math.radians(360.0 / count) * index, (span - shift) / HEALING_TICKS]
            source.ticks = HEALING_TICKS
            self.sources.append(source)
        self.head = 0
        self.least = count * 3

    def source_healing(self, source):
        source.ticks -= 1
        if source.ticks == 0:
            self.emitting = False
            return
        if source.ticks < HEALING_TICKS // 2:
            source.v[0] -= self.turn_growth
        else:
            source.v[0] += self.turn_growth
        source.v[1] += HEALING_TURN
        source.p[0] = math.sin(source.v[1]) * source.v[0] + source.f[0]
        source.p[1] = math.cos(source.v[1]) * source.v[0] + source.f[1]
        source.p[2] += source.v[2]

    def born_healing(self, p, index):
        if not self.sources:
            if self.holder is None:
                # "Particle effect Healing without creature": nothing to circle, the emitter stops
                self.emitting = False
                self.least = 0
                return False
            self.healing_sources()
        cx, cy, cz = self.current
        count = len(self.sources)
        p.color = WHITE
        p.frame_ticks = 1
        if self.head <= count:
            if self.head == count:
                self.head += 1
                return False
            # the ball of a source
            source = self.sources[self.head]
            p.frame = 0
            p.life = HEALING_TICKS
            p.kind = self.head
            p.x, p.y, p.z = source.p[0] + cx, source.p[1] + cy, source.p[2] + cz
            p.size = (self.size + 1.0) * 0.1
            self.head += 1
            return True
        # a spark near a ball
        source = self.sources[self.rng.randrange(count)]
        p.frame = 8
        p.life = 8
        p.kind = -1
        spread = source.v[0] * 0.15
        p.x = self.uniform(-spread, spread) + source.p[0] + cx
        p.y = self.uniform(-spread, spread) + source.p[1] + cy
        p.z = source.p[2] + cz - self.random() * spread
        p.size = (self.size + 1.0) * 0.07
        return True

    def move_healing(self, p):
        p.life -= 1
        if p.life == 0:
            if p.kind < 0 or p.kind == HEALING_FADING:
                return False
            p.kind = HEALING_FADING
            p.life = HEALING_FADE_TICKS
            return True
        if p.kind == HEALING_FADING:
            p.color = (((p.color & 0xff000000) + 0xe1ffffff) | 0xffffff) & 0xffffffff
            return True
        if p.kind < 0:
            p.frame += 1
            return True
        p.frame = p.frame + 1 if p.frame < 7 else 0
        p.frame_ticks = 1
        if p.life < HEALING_TICKS // 2:
            p.size -= self.size_growth
        else:
            p.size += self.size_growth
        source = self.sources[p.kind]
        cx, cy, cz = self.current
        p.x, p.y, p.z = source.p[0] + cx, source.p[1] + cy, source.p[2] + cz
        return True


    #------------------------------------------------------------------------------
    # 0x2019 Teleport (48fdb0, 490060, 490110): the sparks in a circle of the size around the point

    def source_teleport(self, source):
        source.ticks += 1
        if source.ticks > TELEPORT_TICKS:
            self.emitting = False

    def born_teleport(self, p, index):
        cx, cy, cz = self.current
        size = abs(self.size)
        p.color = WHITE
        p.frame = 0
        p.life = self.life
        p.frame_ticks = self.frame_delay
        # the nearer to the middle, the more of them: (r + 1) * r / 2
        r = self.random()
        near = (r + 1.0) * r * 0.5
        angle = self.random() * 2.0 * math.pi
        p.x = size * near * math.sin(angle) + cx
        p.y = size * near * math.cos(angle) + cy
        p.z = cz + (self.random() * 0.2 - 0.4) * size
        p.size = 0.0
        p.b = (2.0 - near) * 0.2
        if self.random() <= near:
            p.frame = self.rng.randrange(4)
        else:
            p.frame = self.rng.randrange(12) + 4
        far = 1.0 - near
        if self.size <= 0.0:
            # it falls from above into the point, showing up
            p.color = 0x01000000
            p.c = TELEPORT_FALL
            p.kind = TELEPORT_SHOW
            p.vz = (far + 3.0) * size * TELEPORT_FALL_SPEED
        else:
            p.c = TELEPORT_RISE
            p.kind = -1
            p.vz = (far + 3.0) * size * TELEPORT_RISE_SPEED
        p.z += p.vz
        return True

    def move_teleport(self, p):
        p.life -= 1
        if p.life == 0:
            return False
        p.frame_ticks -= 1
        if p.frame_ticks == 0:
            p.frame += 1
            if p.frame == 4:
                p.frame = 0
            if p.frame == 16:
                p.frame = 4
            p.frame_ticks = self.frame_delay
        # the speed grows (or falls) by its factor, the point goes by the change
        p.z -= p.vz
        p.vz *= p.c
        p.z += p.vz
        p.size = 0.0 if p.life == 1 else p.b
        alpha = (p.color >> 24) + p.kind
        alpha = 255 if alpha > 254 or alpha < 0 else alpha
        p.color = (alpha << 24) | 0xffffff
        return True

    #------------------------------------------------------------------------------
    # 0x2051 StartTrans

    def source_start_trans(self, source):
        source.ticks -= 1
        if source.ticks + 1 < 0:
            # bits 1 and 4 of +8 go off and the source leaves its holder
            self.emitting = False
            self.attached = False
        source.f[0] += self.scale * START_TRANS_GROWTH
        source.f[1] += START_TRANS_MORE
        self.least = min(int(round(source.f[1])), START_TRANS_MOST_BIRTHS)

    def born_start_trans(self, p, index):
        cx, cy, cz = self.current
        if not self.particles:
            p.life = START_TRANS_GLOW_LIFE
            p.kind = 1
            p.frame = START_TRANS_GLOW_FRAME
            p.x, p.y, p.z = cx, cy, cz
            p.color = CLEAR
            p.size = 0.0
            return True
        p.life = START_TRANS_LIFE
        p.kind = 0
        p.frame = self.rng.getrandbits(32) % 12
        dx = dy = dz = 0.0
        for _ in range(START_TRANS_TRIES):
            dx = self.random() * 2.0 - 1.0
            dy = self.random() * 2.0 - 1.0
            dz = self.uniform(-1.0, 1.0)
            if dx * dx + dy * dy + dz * dz < 1.0:
                break
        length = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        p.vx, p.vy, p.vz = (dx / length * self.scale, dy / length * self.scale, dz / length * self.scale)
        p.x, p.y, p.z = cx + p.vx, cy + p.vy, cz + p.vz
        ticks = float(p.life + 3)
        p.ex, p.ey, p.ez = -p.vx / ticks, -p.vy / ticks, -p.vz / ticks
        p.b = self.scale * START_TRANS_SPARK_SIZE
        p.color = CLEAR
        p.size = 0.0
        return True

    def move_start_trans(self, p):
        if p.kind == 0:
            p.life -= 1
            if p.life < 0:
                return False
            if not self.attached:
                p.life = min(p.life, 4)
            p.frame = (p.frame & ~3) | ((p.frame + 1) & 3)
            p.x += p.ex
            p.y += p.ey
            p.z += p.ez
            p.size = p.b
            p.color = (LIFE_ALPHA[p.life] << 24) | 0xffffff
            return True
        if not self.attached:
            p.life -= 1
            if p.life < 0:
                return False
            p.size *= START_TRANS_GLOW_SHRINK
            p.color = (LIFE_ALPHA[p.life] << 24) | 0xffffff
            return True
        p.color = WHITE
        p.size = (self.random() * START_TRANS_FLICKER + START_TRANS_FLICKER_BASE) * self.sources[0].f[0]
        return True

    #------------------------------------------------------------------------------
    # 0x2052 Transform

    def source_transform(self, source):
        ticks = source.ticks
        source.ticks += 1
        source.p[0] = math.log(ticks + 2) * self.scale * TRANSFORM_WIDTH
        source.p[2] += self.scale * TRANSFORM_RISE
        if source.ticks == TRANSFORM_TICKS:
            self.emitting = False

    def born_transform(self, p, index):
        if not self.sources:
            # the first birth makes the source at the point, its width ln(1) = 0
            source = Source()
            source.f[0:3] = list(self.current)
            self.sources = [source]
        source = self.sources[0]
        cx, cy, cz = self.current
        radius = (self.random() * START_TRANS_FLICKER + START_TRANS_FLICKER_BASE) * source.p[0]
        dx = self.random() * 2.0 - 1.0
        dy = self.random() * 2.0 - 1.0
        dz = self.random() * TRANSFORM_UP_SPREAD + TRANSFORM_UP
        k = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
        ex, ey, ez = k * dx, k * dy, k * dz + TRANSFORM_RISE
        p.x = ex * radius + cx
        p.y = ey * radius + cy
        p.z = ez * radius + cz + source.p[2]
        p.size = (source.ticks * TRANSFORM_SIZE_GROWTH + TRANSFORM_SIZE) * self.scale
        speed = self.scale * TRANSFORM_SPEED
        p.ex, p.ey, p.ez = ex * speed, ey * speed, ez * speed
        p.life = (self.rng.getrandbits(32) & 3) + 16
        p.frame_ticks = 2
        p.color = TRANSFORM_COLOR
        p.frame = max(0, source.ticks // 10 + self.rng.getrandbits(32) % 3 - 1)
        return True

    def move_transform(self, p):
        p.life -= 1
        if p.life == 0:
            return False
        if p.frame < TRANSFORM_LAST_FRAME:
            p.frame_ticks -= 1
            if p.frame_ticks == 0:
                p.frame_ticks = 1
                p.frame += 1
            if p.frame < 8:
                p.size *= TRANSFORM_GROW
        alpha = p.color >> 24
        if p.life < 5:
            alpha = (alpha - 0x40) & 0xff
        else:
            alpha = min(alpha + 0x40, 0xff)
        p.color = (alpha << 24) | 0xffffff
        p.x += p.ex
        p.y += p.ey
        p.z += p.ez
        p.ex *= TRANSFORM_SLOW
        p.ey *= TRANSFORM_SLOW
        p.ez *= TRANSFORM_SLOW
        source = self.sources[0]
        dx = p.x - source.f[0]
        dy = p.y - source.f[1]
        distance = math.hypot(dx, dy)
        if p.z - source.f[2] < source.p[2] and distance > 0.0:
            p.ex -= dx * self.scale * TRANSFORM_PULL / distance
            p.ey -= dy * self.scale * TRANSFORM_PULL / distance
        return True


#------------------------------------------------------------------------------


def frame_uv(frame, columns=ATLAS_COLUMNS):
    """
    485540: the square of a frame in the atlas, (u0, v_top, u1, v_bottom) where v is the row of the pixels
    from the top of the image over its height. The frames go by the rows from the bottom of the image
    (v = 1 - row / columns in game.exe), so a particle of the fire starts bright and ends as smoke;
    the half texel in from the edges (_DAT_0073cc90) is left out.
    """
    step = 1.0 / columns
    column, row = frame % columns, frame // columns
    return (column * step, 1.0 - (row + 1) * step, (column + 1) * step, 1.0 - row * step)


# the particles of the visual of a spell (67f830): 18 Fireworks -> 0x2009, 20 Clairvoyance -> 0x2018,
# 27 Link -> 0x2014. The other spells of the visual have no particles of their own: their light and
# their sounds are made by engine.magic.
SPELL_PARTICLES = {
    18: FIREWORK,
    20: CLAY_RING,
    27: LINK,
}

#------------------------------------------------------------------------------


def spell_emitter(spell_index, position, origin=None, scale=1.0, rng=None, ground=None):
    """
    The emitter of the visual of a spell (67f830 case 0x12/0x14/0x1b), None for the spells without
    particles. origin is the second point of the link.
    """
    code = SPELL_PARTICLES.get(spell_index)
    if code is None:
        return None
    emitter = Emitter(code, position, scale=scale, rng=rng, ground=ground)
    if code == LINK:
        emitter.link_from = None if origin is None else tuple(origin)
    return emitter
