"""
The spells without a caster: the magic traps of the maps and the spells of the scripts.

The magic traps of the maps (MAGIC_TRAP of the .mob, CMagicTrapObject 529dd0, vtable 73e63c).

The record (46b680, 52a2c0) gives the trap:
    MT_DIPLOMACY        +0x238  the bits of the players whose units it casts at (1 << the player of the unit)
    MT_SPELL            +0x23c  the spell "name{modifier;modifier...}" built by 688ff0
    MT_AREAS            +0x244  the circles (x, y, radius) which it looks at
    MT_TARGETS          +0x258  the points (x, y) which a spell of the point goes to
    MT_CAST_INTERVAL    +0x284  the ticks between the casts (15 by default)
    0xbbac0005          +0x28c  only one target (LEVER_CAST_ONCE of EIrepack)
The flag 0x80000 of the object (+8) is on from the start: ActivateTrap( o, f ) (0x95) sets it for f != 0
and clears it for 0, a trap without it does not cast.

Every tick (vf06, 52ac20) a trap with a spell:
    - with the interval below 1 (52a680): forgets the units which died or left its circles; while it is on,
      the living units of its players which came into a circle are kept and the spell goes at them;
    - otherwise the counter +0x288 goes down to 0; then while it is on (52ab30) the living units of its players
      in the circles (52a920, d^2 < r^2 of CWorld::UnitsInCircle) are looked for and, when there are any,
      the spell goes (52a4b0) and the counter starts again from the interval.
The cast (52a4b0): a spell which is not of a unit (no 0x10000000 in +0x3c) goes to every point of the trap when
it has points; otherwise it goes at every unit found (with one target only at the last of them).
Each cast is CWorld::SpreadSpell (58e850) without a caster: the effect starts at the trap and nobody answers it.

CastSpellPoint( spell, x0, y0, x1, y1 ) (0xa8) and CastSpellUnit( spell, x0, y0, o ) (0xa9) of the scripts cast
the same way from the point x0, y0 (z 0 in game.exe, the land in the remake) at the point x1, y1 or the unit.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from engine import magic
from engine import unit as unit_module

#------------------------------------------------------------------------------

# the ticks between the casts by default (46b4a0: +0xc8 = 15)
DEFAULT_INTERVAL = 15

# all the players
ALL_PLAYERS = 0xffffffff

#------------------------------------------------------------------------------


class Caster(object):
    """
    A spell without a unit which casts it (CWorld::SpreadSpell with NULL, 58e850): a trap or a command of
    the scripts. It stands for the caster of engine.unit.make_spell_effect: the effect starts at its point,
    its blows always hit and nobody answers them. The client gives it on_spell_effect and on_shoot like
    to the units.
    """

    is_dead = False
    fighter = None
    on_blow = None
    # a kill by a trap gives no experience (553f40 shares it only for a unit of a party)
    on_kill = None

    def __init__(self, position, land_height=None, rng=None):
        self.x, self.y, self.z = position
        self.ground = land_height or (lambda x, y: 0.0)
        self.rng = rng
        self.on_spell_effect = None
        self.on_shoot = None

    def __repr__(self):
        return 'Caster(%.1f, %.1f)' % (self.x, self.y)

    def land_height(self, x, y):
        return self.ground(x, y)

    def height(self):
        return 0.0

    def is_hostile(self, other):
        return True

    def deliver_blow(self, foe, hit):
        """
        The blow of the spell without an attacker (vf20 of the unit with NULL): it always hits.
        """
        foe.take_spell_blow(hit, None, None)

    def spread(self, spell, main, units):
        """
        The spell at the main target (a unit or engine.magic.Spot) and the others which it spreads to.
        """
        living = [u for u in units if not u.is_dead]
        for target in magic.spread(spell, None, main, living):
            unit_module.make_spell_effect(self, spell, target)

#------------------------------------------------------------------------------


class MagicTrap(Caster):
    """
    A magic trap; player_of(unit) gives the player of a unit.
    """

    def __init__(self, spell, position, areas=(), targets=(), interval=DEFAULT_INTERVAL, players=ALL_PLAYERS,
                 one_target=False, land_height=None, rng=None, name=''):
        super(MagicTrap, self).__init__(position, land_height=land_height, rng=rng)
        self.spell = spell
        self.areas = [tuple(a) for a in areas]
        self.targets = [tuple(t) for t in targets]
        self.interval = interval
        self.players = players
        self.one_target = one_target
        self.name = name
        # the flag 0x80000
        self.active = True
        # +0x288
        self.counter = 0
        # +0x26c: the units in the circles which the trap has cast at (the interval below 1)
        self.inside = []
        self.player_of = lambda unit: 0

    def __repr__(self):
        return 'MagicTrap(%s %s at %.1f,%.1f%s)' % (self.name, self.spell.name if self.spell else None,
                                                   self.x, self.y, '' if self.active else ' off')

    def in_areas(self, unit):
        for x, y, r in self.areas:
            if (unit.x - x) ** 2 + (unit.y - y) ** 2 < r * r:
                return True
        return False

    def victims(self, units):
        """
        52a920: the living units of the players of the trap in its circles.
        """
        found = []
        for u in units:
            if u.is_dead or u in found or not self.in_areas(u):
                continue
            if self.players & (1 << (self.player_of(u) & 31)):
                found.append(u)
        return found

    def update(self, units):
        """
        One tick (vf06, 52ac20); units are all the units of the map.
        """
        if self.spell is None:
            return
        if self.interval < 1:
            self.inside = [u for u in self.inside if not u.is_dead and self.in_areas(u)]
            if not self.active:
                return
            new = [u for u in self.victims(units) if u not in self.inside]
            self.inside.extend(new)
            if new:
                self.cast(new, units)
            return
        if self.counter > 0:
            self.counter -= 1
        if self.counter < 1 and self.active:
            found = self.victims(units)
            if found:
                self.cast(found, units)

    def cast(self, found, units):
        """
        52a4b0: the spell goes to the points of the trap or at the units.
        """
        if not self.spell.flags & magic.TARGET_UNIT and self.targets:
            for x, y in self.targets:
                self.spread(self.spell, magic.Spot(x, y, self.ground(x, y)), units)
        else:
            if self.one_target:
                found = found[-1:]
            for u in found:
                self.spread(self.spell, u, units)
        self.counter = self.interval

#------------------------------------------------------------------------------


def parse_spell(spells, text):
    """
    688ff0: "lightning{e2;e2}" is the prototype with the code "lightning" and two modifiers e2;
    None when the prototype is not known.
    """
    name, _, rest = (text or '').partition('{')
    codes = [c.strip() for c in rest.rstrip('}').split(';') if c.strip()]
    return magic.make_spell(spells, name.strip(), codes)


def from_object(obj, spells, land_height=None, rng=None):
    """
    The trap of an engine.world.MapObject of MAGIC_TRAP; its z is above the land as for the other objects.
    """
    x, y, z = obj.position
    ground = land_height or (lambda px, py: 0.0)
    interval = obj.trap_interval
    if interval >= 0x80000000:
        interval -= 0x100000000
    return MagicTrap(parse_spell(spells, obj.trap_spell), (x, y, ground(x, y) + z), areas=obj.trap_areas,
                     targets=obj.trap_targets, interval=interval, players=obj.trap_players,
                     one_target=obj.trap_one_target, land_height=land_height, rng=rng, name=obj.name)
