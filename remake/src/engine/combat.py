"""
A blow of one unit at another, as the server of game.exe resolves it.

The numbers of a unit (477b00 copies UNIT_STATS into the unit of the server, see engine.stats): the attack
+0x254, the parry +0x25c, the weight of the weapon +0x260, the damage min +0x264 and spread +0x268, the parts
of the 7 types of the damage +0x270 and the absorption against every type (UNIT_STATS 28 times the tag 39
of the unit type).
CUnitServer::vf50 (555a00) gives them for a blow; for a unit which is not of the player (4613a0) 52e0d0
multiplies the attack and the parry by Attack and Defence of [DifficultyLevels] of ai.reg (7c549c, 7c54a8),
the numbers are then cut to integers. A bow or a crossbow in the hands leaves no parry.

A unit of the player with a weapon item (5318b0 of the slot +0x26c) takes the numbers of the item instead
(vf50 and 54f630): the damage min and spread, the parts of the types of the damage and the weight of the item,
which 6a5d20 builds from the shape and the material of items.idb (engine.equipment.Item); its roll is
min + spread * a random part of 1 (6a2da0), not in steps of 0.01. The attack and the parry of such a unit
come from its skills (526c70), which are not here: they stay those of the unit.

The parts of the body (the hit locations of the unit type, records of 0xf4 bytes at +0x110 of the stats):
a part has the share weight / sum of the weights of the maximum health (525a00) and loses the part `loss`
of the health of the unit (the tag 3) when it is gone: the health of the unit is the maximum minus, for every
part with a state above 0 (a cut off part counts too), loss * maximum * (1 - health of the part / its maximum), or loss * maximum
when the health of the part is below 0 (5259b0).

The blow (the packet of 54f630, 15 values):
- the roll of the damage: min + rand() % (round(spread * 100) + 1) * 0.01, at least 1 (523d30);
- the part of the body: +0xf0 of the attacker, the torso (1) unless the player aims (546480);
- the backstab (530a60): a unit of the party, not with a bow or a crossbow, hits a unit whose back
  (+y of the model, the models look at -y) is turned to it: back . (target - attacker) / distance < -0.5;
  then the damage is multiplied by skill * 0.01 + BackstabAdd of ai.reg (525d60, the skill is not known here).

Whether it hits (553df0, 555bd0): the roll of the attack is
    noise + attack - distance / (10 * radius of the target)
    [+ weight / (20 * weight of the target), the weights of the weapons, unless backstab]
not below 0, where the noise is uniform in [-(r + 1), r - 1] with r = +0x258 of the attacker (no writer
of it was found, so 0 and no noise here); an aimed blow loses 25 at the head, 10 at the arms and the legs.
It hits when the roll is not below the parry of the target (0 for a backstab).

The damage (5254b0, 525070): for every type i, damage_i = max(0, roll * part_i - absorption_i * k), k is 1
(and Absorption of [DifficultyLevels] for a target which is not of the party); then every layer of the armor
of the part takes absorption_i of the layer, from the outer one in, until nothing is left (only for a target of
the player); the rest, times the backstab
factor, comes off the health of the part. A part is cut off (state 1, its health -5 * maximum) when one type
alone takes at least its maximum and the type is in the flags of the part, or when its health falls below
-5 * maximum; a part whose health is 0 or less is broken (state 2). A cut off part takes no more blows.

The part 6 is the whole body (5254b0, the blows of the spells): the damage of every type, less the absorption
of the unit, is spread over the parts: a part takes the share max health of the part / (the sum of the losses
of the parts whose state is above 1 * the maximum health of the unit) of it, and the factor of its armor
in the same share, so that the health of the unit falls by exactly the damage. Every part then takes its
share as a blow of its own (525070), a cut off part takes nothing.

The wear of the items (5253e0): an item has the durability of its shape times that of its material (6a5d20,
engine.equipment.Item). The weapon of the attacker loses the damage which the absorption of the target
stops (5254b0) and the damage which the layers of its armor stop (525070). Every item of a layer (the table
73e09c, armor_kinds) loses the damage which goes through the layer (525720). A unit of the player hears that
an item is worn out when its durability falls below "Item Durability Critical" of [RPG] times the maximum, and
that it is broken while the rounded durability is below 1; a broken armor protects on until the armor is built
again (523300 on a change of the items).

The attack of a blow is +0x20 of the packet: when it is FLT_MAX (the spells) the roll 553df0 is skipped
and the blow always hits.

The spells on a unit (the list +0x194 of the unit, records of 20 bytes: the number of the prototype, the ticks
left, the effect...; engine.magic) change its numbers when 524560 counts them again; for every number of
a prototype the strongest effect counts:

    9, 10, 11   the absorption of the fire, the lightning, the acid (the types 3, 5, 4) grows by the effect,
    26          or by the effect of the antimagic when it is larger
    29, 30      the maximum health is times (1 + 0.005 * strengthen) / (1 + 0.005 * weaken), at least 1
                (the parts get their shares of it again, 525a00), the roll of the damage times
                (1 + 0.003 * strengthen) / (1 + 0.003 * weaken), at least 1 (523d30)

The healing (525800, a spell or the regeneration) is the blow of the whole body turned around: a part whose
state is above 1 and whose health is below its maximum takes amount * max health of the part / (the sum
of the losses of such parts * the maximum health), not above its maximum, and is whole again (state 3)
when its health is at least 0.001; a cut off part is not healed.

The armor of a unit of the player (523300, 5230c0): a part has 3 layers (+8 + layer * 0x48 of its record:
the flags, the 7 absorptions, the state at +0x3c, 3 when the layer is worn). Every worn armor item whose
durability is above 0 adds its absorption to the layer of its kind on up to 3 parts (the table 73e048):
the helm, the plate and the leggins (kinds 0..2) go to the outer layer 2, the shirt, the pants, the boots
and the gloves to the layer 1 (6a3160); the first part of the table takes the first absorption of the shape,
the others the second one:

    helm    head                    shirt   torso, arms     boots   legs
    plate   torso, arms             pants   legs            gloves  arms
    leggins legs

The radius of a unit (530570, +0x328): 0.9 * the larger half size of the figure (+0.2 for a unit of the party),
at most 2, plus 0.05.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math
import random

#------------------------------------------------------------------------------

from engine import flyinghp
from engine import stats

#------------------------------------------------------------------------------

DAMAGE_TYPES_COUNT = stats.DAMAGE_TYPES_COUNT

HEAD = 0
TORSO = 1
LEFT_ARM = 2
RIGHT_ARM = 3
LEFT_LEG = 4
RIGHT_LEG = 5
DEFAULT_AIM = TORSO
ALL_PARTS = 6

# the attack of a blow which always hits (FLT_MAX, the spells)
SURE_ATTACK = 3.4028234663852886e+38

# the numbers of the prototypes of spells.sdb whose effects 524560 counts in the numbers of a unit
PROTECTION_SPELLS = {3: 9, 5: 10, 4: 11}
ANTIMAGIC_SPELL = 26
STRENGTHEN_SPELL = 29
WEAKEN_SPELL = 30
# 73e130 for the health, 73e134 for the damage
HEALTH_STRENGTH_STEP = 0.005
DAMAGE_STRENGTH_STEP = 0.003
REGENERATION_SPELL = 31
SPEED_SPELL = 33
SLOW_SPELL = 34
# 524560: the speed of the strikes (+0x30 of the numbers): 15 for a unit out of the party (+0x2c), dexterity
# * 0.2 + 10 for a unit of the party; the delays of the strikes are counted for the speed 15 (73c518, 54d960)
PLAIN_STRIKE_SPEED = 15.0
DEXTERITY_STRIKE_SPEED = 0.2
BASE_STRIKE_SPEED = 10.0
QUICKNESS_STEP = 0.01
MIN_STRIKE_SPEED = 1.0
# the delay of a unit whose prototype is not known, that of the humans
DEFAULT_STRIKE_DELAY = 50
# 523740: the health and the mana come back every 15th tick, a perk adds 0.01 of its power (73c548)
REGENERATION_TICKS = 15
REGENERATION_PERK_STEP = 0.01
# 73ba88, 73ccf4: the smallest healing and the health of a part which is whole again
MIN_HEALING = 0.001

PART_LOST = 1
PART_BROKEN = 2
PART_WHOLE = 3

# 73ddc0
LOST_HEALTH = -5.0

AIMED_HEAD_PENALTY = 25.0
AIMED_LIMB_PENALTY = 10.0

DISTANCE_DIVISOR = 10.0
WEIGHT_DIVISOR = 20.0

# the distance of a blow beyond the radii of both units, the remake's choice: game.exe is not read here yet
STRIKE_REACH = 1.0

# 73e048: the parts of the body of the 7 kinds of the armor (idb.ARMOR_HELM...), 7 is none
NO_PART = 7
ARMOR_PARTS = (
    (HEAD, NO_PART, NO_PART),
    (TORSO, LEFT_ARM, RIGHT_ARM),
    (NO_PART, LEFT_LEG, RIGHT_LEG),
    (TORSO, LEFT_ARM, RIGHT_ARM),
    (NO_PART, LEFT_LEG, RIGHT_LEG),
    (NO_PART, LEFT_LEG, RIGHT_LEG),
    (NO_PART, LEFT_ARM, RIGHT_ARM),
)

ARMOR_LAYERS_COUNT = 3
OUTER_LAYER = 2
INNER_LAYER = 1
# 6a3160: the helm, the plate and the leggins are worn over the others
OUTER_ARMOR_KINDS = (0, 1, 2)

# vf50: the weapon types (adb.WEAPONS) which leave no parry, the bow and the crossbow
SHOOTING_WEAPON_TYPES = (5, 6)

DAMAGE_STEPS = 100
MIN_DAMAGE = 1.0

BACKSTAB_COSINE = -0.5
BACKSTAB_ADD = 3.0
SKILL_STEP = 0.01

RADIUS_FACTOR = 0.9
RADIUS_ADD = 0.05
PARTY_RADIUS_ADD = 0.2
MAX_RADIUS = 2.0

# 7c548c, "Item Durability Critical" of [RPG]: below this share of its durability an item is worn out
ITEM_DURABILITY_CRITICAL = 0.3

# the messages of a unit of the player about its items (5253e0 -> 552780)
MESSAGE_ARMOR_BROKEN = 0x23
MESSAGE_WEAPON_BROKEN = 0x24
MESSAGE_ARMOR_CRITICAL = 0x25
MESSAGE_WEAPON_CRITICAL = 0x26

RPG_SECTION = 'RPG'
DIFFICULTY_SECTION = 'DifficultyLevels'

#------------------------------------------------------------------------------


class Difficulty(object):
    """
    The multipliers of [DifficultyLevels] of ai.reg for the level of the game (0 is normal, 1 is easy).
    """

    def __init__(self, absorption=(1.0, ), attack=(1.0, ), defence=(1.0, ), level=0, backstab_add=BACKSTAB_ADD,
                 durability_critical=ITEM_DURABILITY_CRITICAL):
        self.absorption = tuple(absorption)
        self.attack = tuple(attack)
        self.defence = tuple(defence)
        self.level = level
        self.backstab_add = backstab_add
        self.durability_critical = durability_critical

    @classmethod
    def from_registry(cls, registry, level=0):
        registry = registry or {}
        levels = registry.get(DIFFICULTY_SECTION) or {}
        rpg = registry.get(RPG_SECTION) or {}
        return cls(absorption=_values(levels.get('Absorption')), attack=_values(levels.get('Attack')),
                   defence=_values(levels.get('Defence')), level=level,
                   backstab_add=float(rpg.get('BackstabAdd', BACKSTAB_ADD)),
                   durability_critical=float(rpg.get('Item Durability Critical', ITEM_DURABILITY_CRITICAL)))

    def value(self, values):
        return values[self.level] if self.level < len(values) else 1.0


class BodyPart(object):

    def __init__(self, name, flags, weight, loss, max_health):
        self.name = name
        self.flags = flags
        self.weight = weight
        self.loss = loss
        self.max_health = max_health
        self.health = max_health
        self.state = PART_WHOLE
        # the absorption of the 3 layers of the armor against the 7 types of the damage, None for a bare layer
        self.layers = [None] * ARMOR_LAYERS_COUNT

    def __repr__(self):
        return 'BodyPart(%s %.1f/%.1f state %d)' % (self.name, self.health, self.max_health, self.state)

    @property
    def health_part(self):
        return self.health / self.max_health if self.max_health > 0 else 0.0


class Hit(object):
    """
    The packet of a blow (54f630): the damage of every type per unit of the roll, the roll, the part of the body,
    the factor of the backstab, whether it is aimed.
    """

    def __init__(self, parts, roll, location=DEFAULT_AIM, backstab=1.0, aimed=False, factor=1.0, attack=None):
        self.parts = list(parts)
        # None for the attack of the attacker, SURE_ATTACK for a blow which always hits
        self.attack = attack
        # a blow of a spell which lasts (+0x30 of the packet)
        self.lasting = False
        self.roll = roll
        self.location = location
        self.backstab = backstab
        self.aimed = aimed
        self.factor = factor
        # the weapon item which strikes (the item of the attacker given to vf20 by 553f40) and the Fighter
        # which holds it: the weapon wears out by the damage which the absorption stops
        self.weapon = None
        self.wielder = None

    def __repr__(self):
        return 'Hit(roll %.2f, part %d, backstab %.2f)' % (self.roll, self.location, self.backstab)


class Enchantment(object):
    """
    A spell on a unit: a record of the list +0x194 of the unit (50f1e0).
    """

    def __init__(self, index, ticks, effect):
        self.index = index
        self.ticks = ticks
        self.effect = effect

    def __repr__(self):
        return 'Enchantment(%d, %d ticks, effect %g)' % (self.index, self.ticks, self.effect)


class Fighter(object):
    """
    The numbers of a unit in a fight: engine.stats.UnitStats and the unit type of units.udb (idb.UnitType).
    """

    def __init__(self, unit_stats, unit_type=None, in_party=False, difficulty=None, radius=0.5):
        self.difficulty = difficulty or Difficulty()
        self.in_party = in_party
        self.radius = radius
        self.reach = STRIKE_REACH
        self.mana = float(unit_stats.mana)
        self.max_mana = float(unit_stats.max_mana)
        self.weapon_weight = float(unit_stats.weapon_weight or 1.0)
        # True when the damage comes from a weapon item (a roll without steps)
        self.item_damage = False
        # idb.WeaponShape of the weapon of a hero, its attack and parry count with the skills (set_skill)
        self.weapon_shape = None
        attack, parry = float(unit_stats.attack), float(unit_stats.defense)
        if not in_party:
            attack *= self.difficulty.value(self.difficulty.attack)
            parry *= self.difficulty.value(self.difficulty.defence)
        self.attack = int(attack)
        self.parry = int(parry)
        self.damage_min = float(unit_stats.damage_min)
        self.damage_spread = float(unit_stats.damage_spread)
        if unit_stats.damage_types is not None:
            self.damage_types = tuple(unit_stats.damage_types)
        elif unit_type is not None:
            self.damage_types = tuple(unit_type.damage_types)
        else:
            self.damage_types = (0.0, 0.0, 1.0) + (0.0, ) * (DAMAGE_TYPES_COUNT - 3)
        factors = unit_type.armor_factors if unit_type is not None else (1.0, ) * DAMAGE_TYPES_COUNT
        self.base_absorption = tuple(float(unit_stats.armor) * f for f in factors)
        self.absorption = self.base_absorption
        self.base_max_health = float(unit_stats.max_health)
        self.max_health = self.base_max_health
        self.total_weight = 0.0
        # engine.combat.Enchantment, the spells on the unit
        self.enchantments = []
        locations = unit_type.hit_locations if unit_type is not None else []
        total_weight = sum(h.weight for h in locations)
        self.total_weight = total_weight
        self.parts = []
        for h in locations:
            share = self.max_health / total_weight * h.weight if total_weight > 0 else 0.0
            self.parts.append(BodyPart(h.name, h.flags, h.weight, h.health_part, share))
        self.start_health = float(unit_stats.health)
        # 5136d0: the parts of the maximum health and mana which come back (the tags 2 and 3 of the unit type)
        self.health_regen = unit_type.health_regen if unit_type is not None else 0.0
        self.mana_regen = unit_type.mana_regen if unit_type is not None else 0.0
        # the ticks since the last regeneration (+4 of the numbers)
        self.regen_ticks = 0.0
        # the ticks between two strikes at the speed 15: the weapon item or the prototype (54d960)
        self.strike_delay = DEFAULT_STRIKE_DELAY
        # the items of engine.equipment: the weapon in the hands and the worn armor by its kind (530830)
        self.weapon_item = None
        self.armor_items = {}
        # (item, message) of the wear of the items of a unit of the player, taken by engine.unit.Unit
        self.wear_messages = []
        # +0x6f8: the flags of the blows since the client part looked (engine.flyinghp), BACKSTAB and CRITICAL
        self.hit_flags = 0

    def __repr__(self):
        return 'Fighter(health %.1f/%.1f, attack %d, parry %d)' % (self.health, self.max_health, self.attack, self.parry)

    @property
    def health(self):
        """
        5259b0: the maximum minus the losses of the parts of the body.
        """
        if not self.parts:
            return self.start_health
        lost = 0.0
        for part in self.parts:
            if part.state <= 0:
                continue
            if part.health < 0.0:
                lost += self.max_health * part.loss
            elif part.max_health > 0:
                lost += (part.max_health - part.health) / part.max_health * part.loss * self.max_health
        return self.max_health - lost

    @property
    def is_dead(self):
        return self.health <= 0.0

    def equip(self, equipment, hero_stats=None):
        """
        The numbers of the items of engine.equipment.Equipment: the weapon and the armor count only for a unit
        of the player, a bow or a crossbow in the hands leaves no parry for anybody. hero_stats
        (engine.experience.HeroStats) gives a hero with a weapon the attack and the parry of vf50 (555a00):
        those of the weapon shape plus the skill of the weapon (526c70), cut to integers.
        """
        weapon = equipment.weapon
        self.weapon_item = weapon
        if weapon is not None and weapon.shape.type in SHOOTING_WEAPON_TYPES:
            self.parry = 0
        if weapon is not None and weapon.shape.strike_delay > 0:
            self.strike_delay = weapon.shape.strike_delay
        if not self.in_party:
            return
        if weapon is not None:
            self.damage_min, self.damage_spread = weapon.damage
            self.damage_types = weapon.damage_types
            self.weapon_weight = weapon.weight or 1.0
            self.item_damage = True
            if hero_stats is not None:
                self.weapon_shape = weapon.shape
                self.set_skill(hero_stats)
        for kind, item in sorted(equipment.armors.items()):
            if 0 <= kind < len(ARMOR_PARTS):
                self.armor_items[kind] = item
        self.rebuild_armor()

    def rebuild_armor(self):
        """
        523300: the layers of the parts of the body get the absorption of the worn armor items again; a broken
        item wears on but protects no more.
        """
        for part in self.parts:
            part.layers = [None] * ARMOR_LAYERS_COUNT
        for kind, item in sorted(self.armor_items.items()):
            if getattr(item, 'is_broken', False):
                continue
            layer = OUTER_LAYER if kind in OUTER_ARMOR_KINDS else INNER_LAYER
            for k, location in enumerate(ARMOR_PARTS[kind]):
                if location == NO_PART or location >= len(self.parts):
                    continue
                part = self.parts[location]
                values = item.absorption(limb=k != 0)
                if part.layers[layer] is None:
                    part.layers[layer] = [0.0] * DAMAGE_TYPES_COUNT
                for i, value in enumerate(values):
                    part.layers[layer][i] += value

    def set_skill(self, hero_stats):
        """
        The attack and the parry of a hero with a weapon again (vf50 555a00), after its skills changed.
        """
        shape = self.weapon_shape
        if shape is None or hero_stats is None:
            return
        skill = hero_stats.weapon_skill(shape.type, hero_stats.weapon_bonus(shape))
        self.attack = int(shape.attack + skill)
        self.parry = 0 if shape.type in SHOOTING_WEAPON_TYPES else int(shape.defense + skill)

    def enchantment(self, index):
        """
        50f690: the strongest effect of the spells of the prototype on the unit, 0 when there is none.
        """
        best = 0.0
        for e in self.enchantments:
            if e.index == index and e.effect > best:
                best = e.effect
        return best

    def strength_factor(self, step):
        return (1.0 + step * self.enchantment(STRENGTHEN_SPELL)) / (1.0 + step * self.enchantment(WEAKEN_SPELL))

    def enchant(self, index, ticks, effect):
        self.enchantments.append(Enchantment(index, ticks, effect))
        self.recount()

    def strike_speed(self, dexterity=None, quickness=0.0, arms_factor=1.0):
        """
        524560: the speed of the strikes (+0x30): (1 + 0.01 * quickness) * the factor of the arms (524110)
        * the base (15, or dexterity * 0.2 + 10 for a unit of the party, dexterity None for the others)
        + the speed spell - the slow spell, at least 1.
        """
        if dexterity is None:
            base = PLAIN_STRIKE_SPEED
        else:
            base = dexterity * DEXTERITY_STRIKE_SPEED + BASE_STRIKE_SPEED
        speed = (quickness * QUICKNESS_STEP + 1.0) * arms_factor * base
        speed += self.enchantment(SPEED_SPELL) - self.enchantment(SLOW_SPELL)
        return max(MIN_STRIKE_SPEED, speed)

    def strike_ticks(self, speed):
        """
        54d960: the ticks from the start of a strike until the next one may start, round(delay * 15 / speed).
        """
        return int(round(int(self.strike_delay) * PLAIN_STRIKE_SPEED / speed))

    def regenerate(self, ticks=1, idle=True, vitality=0.0, spirit=0.0):
        """
        523740 (CUnitServer::vf06): every 15th tick the mana grows by (1 + 0.01 * spirit) * the mana regeneration
        * the maximum, only while the unit has no order (+0x190 of the unit of the server is 0, here the idle
        unit), and the parts are healed (525800) by (1 + 0.01 * vitality) * the health regeneration * the maximum
        health, times the effect of the regeneration spell (31) when it is on the unit. vitality and spirit are
        the powers of the perks of the groups 18 and 19. True when something came back.
        """
        self.regen_ticks += ticks
        changed = False
        while self.regen_ticks >= REGENERATION_TICKS:
            self.regen_ticks -= REGENERATION_TICKS
            if idle and self.mana < self.max_mana:
                gain = (spirit * REGENERATION_PERK_STEP + 1.0) * self.mana_regen * self.max_mana
                self.mana = min(self.max_mana, self.mana + gain)
                changed = True
            if self.health < self.max_health:
                rate = (vitality * REGENERATION_PERK_STEP + 1.0) * self.health_regen
                spell = self.enchantment(REGENERATION_SPELL)
                if spell > 0.0:
                    rate *= spell
                heal(self, self.max_health * rate)
                changed = True
        return changed

    def tick_enchantments(self, ticks=1):
        """
        CMapObjectServer::vf06 (515820): every record loses a tick, the ones with none left go away.
        True when the numbers were counted again.
        """
        if not self.enchantments:
            return False
        for e in self.enchantments:
            e.ticks -= ticks
        left = [e for e in self.enchantments if e.ticks > 0]
        if len(left) == len(self.enchantments):
            return False
        self.enchantments = left
        self.recount()
        return True

    def recount(self):
        """
        524560: the absorption and the maximum health with the spells on the unit, the parts get their
        shares of the maximum again (525a00).
        """
        absorption = list(self.base_absorption)
        antimagic = self.enchantment(ANTIMAGIC_SPELL)
        for kind, index in PROTECTION_SPELLS.items():
            if kind < len(absorption):
                absorption[kind] += max(self.enchantment(index), antimagic)
        self.absorption = tuple(absorption)
        self.max_health = max(1.0, self.base_max_health * self.strength_factor(HEALTH_STRENGTH_STEP))
        if self.total_weight <= 0.0:
            return
        for part in self.parts:
            if part.state > 0:
                part.max_health = self.max_health / self.total_weight * part.weight
                part.health = min(part.health, part.max_health)

    def part_health(self, name):
        """
        The smallest part of the health of the parts with the name ("leg", "arm"), 1.0 when there are none.
        """
        values = [p.health_part for p in self.parts if p.name == name]
        return min(values) if values else 1.0

#------------------------------------------------------------------------------


def unit_radius(half_size, in_party=False):
    radius = half_size * RADIUS_FACTOR
    if in_party:
        radius += PARTY_RADIUS_ADD
    return min(radius, MAX_RADIUS) + RADIUS_ADD


def is_backstab(attacker_position, target_position, target_yaw):
    """
    530a60: the attacker stands behind the target (the back of a model is its +y turned by the yaw).
    """
    dx = target_position[0] - attacker_position[0]
    dy = target_position[1] - attacker_position[1]
    length = math.hypot(dx, dy)
    if length > 0:
        dx, dy = dx / length, dy / length
    back_x, back_y = -math.sin(target_yaw), math.cos(target_yaw)
    return back_x * dx + back_y * dy < BACKSTAB_COSINE


def damage_roll(attacker, rng=random):
    if attacker.item_damage:
        roll = attacker.damage_min + attacker.damage_spread * rng.random()
    else:
        steps = int(round(attacker.damage_spread * DAMAGE_STEPS))
        roll = attacker.damage_min + rng.randrange(steps + 1) / float(DAMAGE_STEPS)
    # 523d30: the strengthen and the weaken on the attacker
    return max(MIN_DAMAGE, roll * attacker.strength_factor(DAMAGE_STRENGTH_STEP))


def heal(target, amount):
    """
    525800: the healing spread over the parts which are not whole, the health which came back
    (CUnit::Heal 525900 then takes the health from the parts, 5259b0).
    """
    if abs(amount) <= MIN_HEALING or not target.parts or target.max_health <= 0.0:
        return 0.0
    before = target.health
    hurt = [p for p in target.parts if p.state >= PART_BROKEN and p.health < p.max_health]
    losses = sum(p.loss for p in hurt)
    if losses == 0.0:
        return 0.0
    for part in hurt:
        part.health = min(part.max_health, amount / (losses * target.max_health) * part.max_health + part.health)
        if part.health >= MIN_HEALING:
            part.state = PART_WHOLE
    return target.health - before


def make_hit(attacker, rng=random, location=DEFAULT_AIM, backstab=False, aimed=False, skill=0.0):
    factor = skill * SKILL_STEP + attacker.difficulty.backstab_add if backstab else 1.0
    hit = Hit(attacker.damage_types, damage_roll(attacker, rng), location=location, backstab=factor, aimed=aimed)
    hit.weapon = attacker.weapon_item
    hit.wielder = attacker
    return hit


def wear_item(item, amount, owner, broken_message, critical_message):
    """
    5253e0: the item loses amount of its durability (shared by the things of a stack). A unit of the player
    gets the message broken_message when the rounded durability is below 1, else critical_message when
    the durability falls below the critical share of its maximum. Returns the message or None.
    """
    if item is None or owner is None or amount <= 0.0:
        return None
    if getattr(item, 'max_durability', 0.0) <= 0.0:
        return None
    before = item.durability
    critical = item.max_durability * owner.difficulty.durability_critical
    count = getattr(item, 'count', 1)
    item.durability -= amount / count if count >= 1 else amount
    message = None
    if int(round(item.durability)) < 1:
        message = broken_message
    elif before >= critical and item.durability < critical:
        message = critical_message
    if message is not None and owner.in_party:
        owner.wear_messages.append((item, message))
    return message


def armor_kinds(location, layer):
    """
    The table 73e09c of 525720: the kinds of the armor worn on the layer of the part of the body.
    """
    if layer == 0:
        return ()
    result = []
    for kind, locations in enumerate(ARMOR_PARTS):
        if location in locations and (OUTER_LAYER if kind in OUTER_ARMOR_KINDS else INNER_LAYER) == layer:
            result.append(kind)
    return tuple(result)


def _wear_weapon(hit, amount):
    if hit.weapon is not None:
        wear_item(hit.weapon, amount, hit.wielder, MESSAGE_WEAPON_BROKEN, MESSAGE_WEAPON_CRITICAL)


def attack_roll(attacker, target, distance, hit):
    """
    555bd0 and 553df0: the roll of the attack against the parry of the target.
    """
    roll = attacker.attack - distance / (DISTANCE_DIVISOR * target.radius)
    if hit.backstab <= 1.0:
        roll += attacker.weapon_weight / (WEIGHT_DIVISOR * target.weapon_weight)
    roll = max(0.0, roll)
    if hit.aimed:
        if hit.location == HEAD:
            roll = max(0.0, roll - AIMED_HEAD_PENALTY)
        elif hit.location != TORSO:
            roll = max(0.0, roll - AIMED_LIMB_PENALTY)
    return roll


def hits(attacker, target, distance, hit):
    if hit.attack is not None and abs(hit.attack) >= SURE_ATTACK:
        return True
    parry = 0 if hit.backstab > 1.0 else target.parry
    return attack_roll(attacker, target, distance, hit) >= parry


def apply_hit(target, hit):
    """
    5254b0 and 525070: the damage of the blow on the target, the damage which came off the part of the body.
    """
    dealt = _apply_hit(target, hit)
    # 5254b0 marks a blow at the skull at once, 525070 a backstab which came to a part
    target.hit_flags |= flyinghp.hit_flags(hit) & (flyinghp.CRITICAL | (flyinghp.BACKSTAB if dealt else 0))
    return dealt


def _apply_hit(target, hit):
    factor = 1.0 if target.in_party else target.difficulty.value(target.difficulty.absorption)
    damages = []
    for i in range(DAMAGE_TYPES_COUNT):
        part = hit.parts[i] if i < len(hit.parts) else 0.0
        absorption = target.absorption[i] if i < len(target.absorption) else 0.0
        damage = max(0.0, hit.roll * part - hit.factor * absorption * factor)
        damages.append(damage)
        # 5254b0: the weapon wears by what the absorption of the unit stopped
        _wear_weapon(hit, hit.roll * part - damage)
    if sum(damages) <= 0.0:
        return 0.0
    if not target.parts or not 0 <= hit.location <= ALL_PARTS:
        dealt = sum(damages) * max(1.0, hit.backstab)
        target.start_health -= dealt
        return dealt
    if hit.location == ALL_PARTS:
        losses = sum(part.loss for part in target.parts if part.state >= PART_BROKEN)
        if losses <= 0.0 or target.max_health <= 0.0:
            return 0.0
        dealt = 0.0
        for location, part in enumerate(target.parts):
            share = part.max_health / (losses * target.max_health)
            dealt += _hit_part(target, location, [d * share for d in damages], hit.factor * share, hit)
        return dealt
    if hit.location >= len(target.parts):
        dealt = sum(damages) * max(1.0, hit.backstab)
        target.start_health -= dealt
        return dealt
    return _hit_part(target, hit.location, damages, hit.factor, hit)


def _hit_part(target, location, damages, factor, hit):
    """
    525070: the damages of the types come to one part of the body through the layers of its armor. The items
    of a layer wear by the damage which goes through the layer (525720), the weapon by the damage which
    the layers stop.
    """
    part = target.parts[location]
    backstab = hit.backstab
    if part.state < PART_BROKEN:
        return 0.0
    dealt = 0.0
    for i, damage in enumerate(damages):
        if damage <= 0.0:
            continue
        if target.in_party:
            start = damage
            for index in range(len(part.layers) - 1, -1, -1):
                layer = part.layers[index]
                if layer is None:
                    continue
                damage -= factor * layer[i]
                for kind in armor_kinds(location, index):
                    wear_item(target.armor_items.get(kind), max(0.0, damage), target, MESSAGE_ARMOR_BROKEN,
                              MESSAGE_ARMOR_CRITICAL)
                if damage <= 0.0:
                    break
            _wear_weapon(hit, start - max(0.0, damage))
            if damage <= 0.0:
                continue
        if backstab > 1.0:
            damage *= backstab
        dealt += damage
        part.health -= damage
        if (damage >= part.max_health and part.flags & (1 << i)) or part.health < part.max_health * LOST_HEALTH:
            part.state = PART_LOST
            part.health = part.max_health * LOST_HEALTH
        elif part.health <= 0.0:
            part.state = PART_BROKEN
    if _Debug:
        print('%s takes %.2f, health %.2f' % (part, dealt, target.health))
    return dealt

#------------------------------------------------------------------------------


def _values(value):
    if value is None:
        return (1.0, )
    if isinstance(value, (tuple, list)):
        return tuple(float(v) for v in value)
    return (float(value), )
