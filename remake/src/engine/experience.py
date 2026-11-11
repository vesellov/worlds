"""
The experience, the attributes, the skills and the perks of a hero or a mercenary: the stats object of a unit
of the class 0x39 (0x704 bytes, the object +0x1e0 of the unit of the server, kept by the records of the parties
from +0x20, see engine.roster).

    +4      the experience, all of it: the part put into the skills and the free part
    +8      the experience when the unit was made (5136d0 copies +4 there)
    +0xc    the strength, its base and +0x10 its value with the perks (524560 copies and adds)
    +0x14   the dexterity, +0x18 with the perks
    +0x1c   the intelligence, +0x20 with the perks
    +0x6c8  the science (Use/Steal) and two more bytes, +0x6cb the skills by their numbers of perks.pdb:
            0 melee, 1 archery, 2 backstab, 3 elemental, 4 sense, 5 astral, 6 stealth, 7 awareness
    +0x6d3  the level (0..3) of every group of the perks of perks.pdb (the group is the index of the perk // 3)
    +0x700  the experience which waits: a gain below zero is kept here until the gains make it positive

A unit of a prototype with a hero template in the section 4 of units.udb (5145d0) starts with its numbers
(5136d0): the attributes, the skills, the perks (526450) and the experience given by 523960.

The price of a skill (528ab0 and 528da0, the GiveSkill of the scripts): the skill of the level n stands for
Skill Val 1 * 1.61 ^ (n / Skill Val 2) of the experience ([RPG] of ai.reg, 50 and 5); raising it from a to b adds
round(price(b) - price(a)) to the experience, so the free part does not change. The experience of a template
is the price of its skills: 271 for the hero, whose skills 3, 10, 12 and 9 cost 16.5 + 79.6 + 106.9 + 67.9.

A gain of the experience (523960, 5239b0) is multiplied by 1.04 ^ (intelligence - 25), then the maximum health
and mana are counted again (5239b0, 524560):

    health = (perk of the group 16 * 0.01 + 1) * (strength / HP Val 1) * HP Val 2
             * HP Val 4 ^ (log(experience / HP Val 3) / log(HP Val 5))
    mana = the same with the group 17, the dexterity and MP Val 1..5

(the comment of ai.reg: "Health = (Str/v1)*v2*POWER(v4; LOG(EXP/v3; v5))"); the spells 0x1d and 0x1e scale
the health of a unit (engine.combat.Fighter.recount).

The skill of a weapon (526c70): the attribute of the skill in perks.pdb ("dex" for the melee and the archery)
- 25 + the skill (melee for the groups of the weapons 0..4, archery for 5 and 6) + the bonus of the weapon item
+ the power of the perk of the group of the weapon (the groups 0..6 are the kinds of the weapons of items.idb).
The skill of a school of magic (527eb0): the intelligence - 25 + the skill of the school + the perk of it.

QuestComplete (5911c0) shares the experience of the quest among the units of the party in the zone:
a unit takes sqrt(experience + waiting) of the sum of these weights of all (the sum is at least 0.01).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math

#------------------------------------------------------------------------------

RPG_SECTION = 'RPG'

# [RPG] of ai.reg, the values of the file of the game are the defaults
HP_VALUES = (25.0, 30.0, 50.0, 1.1, 1.61)
MP_VALUES = (25.0, 30.0, 50.0, 1.1, 1.61)
SKILL_VALUES = (50.0, 5.0)

# 73e140: the base of the price of a skill
SKILL_PRICE_BASE = 1.61

# 73e124 and 73e128: the plain attribute and the factor of a point of the intelligence for the experience
PLAIN_ATTRIBUTE = 25.0
INTELLIGENCE_EXPERIENCE_FACTOR = 1.04

# 73c548: a percent
PERCENT = 0.01

# 5911c0: the smallest sum of the weights of the units which share the experience of a quest
MIN_SHARE_WEIGHT = 0.01

# the skills by their numbers of perks.pdb, the bytes from +0x6cb
SKILL_MELEE = 0
SKILL_ARCHERY = 1
SKILL_BACKSTAB = 2
SKILL_ELEMENTAL = 3
SKILL_SENSE = 4
SKILL_ASTRAL = 5
SKILL_STEALTH = 6
SKILL_AWARENESS = 7
SKILLS_COUNT = 8
# the science is the first byte of +0x6c8 (528da0)
SKILL_SCIENCE = 'science'

# the names of the skills of GiveSkill( unit, "name", value ) (0xcd, the texts at 78b8b0..78b8fc)
SKILL_CODES = {
    'science': SKILL_SCIENCE,
    'melee': SKILL_MELEE,
    'archery': SKILL_ARCHERY,
    'backstab': SKILL_BACKSTAB,
    'elemental': SKILL_ELEMENTAL,
    'sense': SKILL_SENSE,
    'astral': SKILL_ASTRAL,
    'stealth': SKILL_STEALTH,
    'awareness': SKILL_AWARENESS,
}

# the attributes of the skills in perks.pdb (the tag 10 of the section 1), taken without the file
SKILL_ATTRIBUTES = {
    SKILL_MELEE: 'dex',
    SKILL_ARCHERY: 'dex',
    SKILL_BACKSTAB: 'dex',
    SKILL_ELEMENTAL: 'int',
    SKILL_SENSE: 'int',
    SKILL_ASTRAL: 'int',
}

# the groups of the perks (+0x6d3 + group)
PERK_GROUPS_COUNT = 0x24
GROUP_ARCHERY_LAST = 6
GROUP_SCHOOLS = 7
GROUP_NIGHT_VISION = 15
GROUP_HEALTH = 16
GROUP_MANA = 17
GROUP_VITALITY = 18
GROUP_SPIRIT = 19
GROUP_QUICKNESS = 20
GROUP_BACKSTAB = 22
GROUP_STRENGTH = 23
GROUP_DEXTERITY = 24
GROUP_INTELLIGENCE = 25

# the kinds of the weapons of items.idb (idb.WEAPON_*) are the groups of their perks; 0..4 use the melee
WEAPON_GROUP_BOW = 5

# 527eb0: the school of a spell (+0x14 of its prototype) -> the school whose perk and skill it uses
SCHOOL_REMAP = {4: 3, 6: 5, 7: 5}
# the skill of a school after SCHOOL_REMAP: fire, lightning, acid elemental, illusion sense, enchantments astral
SCHOOL_SKILLS = (SKILL_ELEMENTAL, SKILL_ELEMENTAL, SKILL_ELEMENTAL, SKILL_SENSE, SKILL_SENSE,
                 SKILL_ASTRAL, SKILL_ASTRAL, SKILL_ASTRAL)

ATTRIBUTE_STRENGTH = 's'
ATTRIBUTE_DEXTERITY = 'd'
ATTRIBUTE_INTELLIGENCE = 'i'

#------------------------------------------------------------------------------


class RpgSettings(object):
    """
    The numbers of [RPG] of ai.reg (7c5444..7c5470).
    """

    def __init__(self, hp_values=HP_VALUES, mp_values=MP_VALUES, skill_values=SKILL_VALUES):
        self.hp_values = tuple(hp_values)
        self.mp_values = tuple(mp_values)
        self.skill_values = tuple(skill_values)

    @classmethod
    def from_registry(cls, registry):
        keys = (registry or {}).get(RPG_SECTION) or {}
        hp_values = tuple(float(keys.get('HP Val %d' % (i + 1), v)) for i, v in enumerate(HP_VALUES))
        mp_values = tuple(float(keys.get('MP Val %d' % (i + 1), v)) for i, v in enumerate(MP_VALUES))
        skill_values = tuple(float(keys.get('Skill Val %d' % (i + 1), v)) for i, v in enumerate(SKILL_VALUES))
        return cls(hp_values, mp_values, skill_values)

    def skill_price(self, level):
        """
        The experience which a skill of the level stands for (528ab0).
        """
        return self.skill_values[0] * SKILL_PRICE_BASE ** (level / self.skill_values[1])

    def skill_raise_price(self, level, step=1):
        return int(self.skill_price(level + step) - self.skill_price(level))

#------------------------------------------------------------------------------


def growth(experience, values):
    """
    v4 ^ (log(experience / v3) / log(v5)), 0 without experience.
    """
    if experience <= 0.0 or values[2] <= 0.0 or values[4] <= 0.0 or values[4] == 1.0:
        return 0.0
    return values[3] ** (math.log(experience / values[2]) / math.log(values[4]))

#------------------------------------------------------------------------------


class HeroStats(object):

    def __init__(self, perks=None, settings=None):
        """
        perks: idb.PerkDatabase of perks.pdb (None: the perks give nothing).
        """
        self.perk_database = perks
        self.settings = settings or RpgSettings()
        self.experience = 0.0
        self.start_experience = 0.0
        self.waiting_experience = 0.0
        # +0x28: the experience paid for the perks in the camp (526450)
        self.perk_experience = 0.0
        self.base_strength = PLAIN_ATTRIBUTE
        self.base_dexterity = PLAIN_ATTRIBUTE
        self.base_intelligence = PLAIN_ATTRIBUTE
        self.science = 0
        self.skills = [0] * SKILLS_COUNT
        self.perk_levels = [0] * PERK_GROUPS_COUNT
        # the tag 12 of the hero template: the most weight the unit carries without a slowdown
        self.max_weight = 200.0
        # the tag 13: the percent of the speed at that weight, the tag 14 of the template
        self.weight_percent = 100
        self.template_flag = 1

    def __repr__(self):
        return 'HeroStats(exp %g, free %g, str %g dex %g int %g, skills %s, science %d)' % (
            self.experience, self.free_experience, self.strength, self.dexterity, self.intelligence,
            self.skills[:SKILL_ASTRAL + 1], self.science)

    @classmethod
    def from_template(cls, template, perks=None, settings=None):
        """
        5136d0: the numbers of a hero template (idb.HeroTemplate) of units.udb.
        """
        stats = cls(perks, settings)
        stats.base_strength = float(template.strength)
        stats.base_dexterity = float(template.dexterity)
        stats.base_intelligence = float(template.intelligence)
        stats.science = int(template.science[0])
        stats.skills = [int(v) for v in template.skills[:SKILLS_COUNT]]
        stats.skills += [0] * (SKILLS_COUNT - len(stats.skills))
        stats.max_weight = float(template.max_weight or stats.max_weight)
        stats.weight_percent = int(template.weight_percent or stats.weight_percent)
        stats.template_flag = int(template.template_flag)
        stats.add_experience(template.experience)
        for code in template.perk_codes:
            stats.add_perk(code)
        stats.start_experience = stats.experience
        return stats

    @classmethod
    def from_saved(cls, saved, perks=None, settings=None):
        """
        The numbers of a unit of a party in a save of the game (eilib.sav.SavedStats, the same layout of 0x704 bytes).
        """
        stats = cls(perks, settings)
        stats.experience = float(saved.experience)
        stats.start_experience = float(saved.start_experience)
        stats.waiting_experience = float(saved.waiting_experience)
        stats.base_strength = float(saved.strength)
        stats.base_dexterity = float(saved.dexterity)
        stats.base_intelligence = float(saved.intelligence)
        stats.science = int(saved.science)
        stats.skills = (list(saved.skills) + [0] * SKILLS_COUNT)[:SKILLS_COUNT]
        stats.perk_levels = (list(saved.perk_levels) + [0] * PERK_GROUPS_COUNT)[:PERK_GROUPS_COUNT]
        return stats

    def copy(self):
        """
        CopyStats (0x9d) copies the whole record.
        """
        other = HeroStats(self.perk_database, self.settings)
        other.experience = self.experience
        other.start_experience = self.start_experience
        other.waiting_experience = self.waiting_experience
        other.perk_experience = self.perk_experience
        other.base_strength = self.base_strength
        other.base_dexterity = self.base_dexterity
        other.base_intelligence = self.base_intelligence
        other.science = self.science
        other.skills = list(self.skills)
        other.perk_levels = list(self.perk_levels)
        other.max_weight = self.max_weight
        other.weight_percent = self.weight_percent
        other.template_flag = self.template_flag
        return other

    # the perks

    def perk_power(self, group):
        """
        528270 + the field +0x34: the power of the level of the group which the unit has, 0 without it.
        """
        if self.perk_database is None or not 0 <= group < len(self.perk_levels):
            return 0.0
        return self.perk_database.power(group, self.perk_levels[group])

    def add_perk(self, code):
        """
        526450: the perk by its code ("bs1"); a level which the unit has already does not go down.
        False for an unknown code.
        """
        perk = self.perk_database.perk(code) if self.perk_database is not None else None
        if perk is None:
            return False
        if perk.group < len(self.perk_levels) and self.perk_levels[perk.group] < perk.level:
            self.perk_levels[perk.group] = perk.level
        return True

    # the attributes with the perks (524560)

    @property
    def strength(self):
        return self.base_strength + self.perk_power(GROUP_STRENGTH)

    @property
    def dexterity(self):
        return self.base_dexterity + self.perk_power(GROUP_DEXTERITY)

    @property
    def intelligence(self):
        return self.base_intelligence + self.perk_power(GROUP_INTELLIGENCE)

    def attribute(self, code):
        """
        The attribute by the first letter of its code in perks.pdb ("dex", "int", "str"), 0 for "none".
        """
        letter = (code or '')[:1].lower()
        if letter == ATTRIBUTE_DEXTERITY:
            return self.dexterity
        if letter == ATTRIBUTE_INTELLIGENCE:
            return self.intelligence
        if letter == ATTRIBUTE_STRENGTH:
            return self.strength
        return None

    # the experience

    @property
    def spent_experience(self):
        """
        What the skills stand for (the prices of their levels over the price of the level 0) and the experience
        paid for the perks.
        """
        price = self.settings.skill_price
        levels = [self.science] + list(self.skills)
        return sum(price(level) - price(0) for level in levels if level > 0) + self.perk_experience

    @property
    def free_experience(self):
        return self.experience - self.spent_experience

    def add_experience(self, amount):
        """
        523960 and 5239b0 (called by CUnitServer::AddExperience 553380): the gain times 1.04 ^ (intelligence - 25);
        a sum below zero waits in +0x700.
        The experience really added.
        """
        gain = INTELLIGENCE_EXPERIENCE_FACTOR ** (self.intelligence - PLAIN_ATTRIBUTE) * amount
        return self.add_plain_experience(gain)

    def add_plain_experience(self, gain):
        total = gain + self.waiting_experience
        if total > 0.0:
            self.waiting_experience = 0.0
            self.experience += total
            return total
        self.waiting_experience = total
        return 0.0

    # the skills

    def skill(self, index):
        if index == SKILL_SCIENCE:
            return self.science
        return self.skills[index] if 0 <= index < len(self.skills) else 0

    def give_skill(self, index, amount):
        """
        GiveSkill (0xcd, 528ab0 and 528da0 for the science): the skill grows by amount (a byte) and
        the experience by the price of the levels, the free experience stays.
        """
        old = self.skill(index)
        new = max(0, min(0xff, old + int(amount)))
        if index == SKILL_SCIENCE:
            self.science = new
        elif 0 <= index < len(self.skills):
            self.skills[index] = new
        else:
            return False
        price = self.settings.skill_price
        self.add_plain_experience(float(int(price(new) - price(old))))
        return True

    def raise_skill(self, index):
        """
        The camp: a level of the skill bought with the free experience, False when there is not enough of it.
        """
        price = self.settings.skill_raise_price(self.skill(index))
        if price > self.free_experience:
            return False
        if index == SKILL_SCIENCE:
            self.science += 1
        else:
            self.skills[index] += 1
        return True

    def give_strength(self, amount):
        """
        GiveStrength (0xce, 529090), GiveDexterity (0xcf, 529220), GiveIntelligence (0xd0, 529400): the base grows.
        """
        self.base_strength += amount

    def give_dexterity(self, amount):
        self.base_dexterity += amount

    def give_intelligence(self, amount):
        self.base_intelligence += amount

    # the numbers of the unit

    @property
    def max_health(self):
        """
        5239b0, 524560: the maximum health without the spells, at least 1.
        """
        v = self.settings.hp_values
        value = (self.perk_power(GROUP_HEALTH) * PERCENT + 1.0) * (self.strength / v[0]) * v[1] * growth(self.experience, v)
        return max(1.0, value)

    @property
    def max_mana(self):
        v = self.settings.mp_values
        return (self.perk_power(GROUP_MANA) * PERCENT + 1.0) * (self.dexterity / v[0]) * v[1] * growth(self.experience, v)

    def weapon_bonus(self, shape):
        """
        526c70: the bonus of the weapon (idb.WeaponShape) to the skill of its group, +0x7b melee or +0x7c archery.
        """
        bonuses = getattr(shape, 'skill_bonuses', None) or ()
        index = SKILL_ARCHERY if shape.type >= WEAPON_GROUP_BOW else SKILL_MELEE
        return float(bonuses[index]) if index < len(bonuses) else 0.0

    def weapon_skill(self, group, item_bonus=0.0):
        """
        526c70: the skill of a weapon of the group (the kind of the weapon, 0..4 melee, 5 and 6 archery)
        with the bonus of the weapon item (+0x7b or +0x7c of its shape).
        """
        index = SKILL_ARCHERY if group >= WEAPON_GROUP_BOW else SKILL_MELEE
        value = self.skills[index] + item_bonus + self.perk_power(group)
        code = SKILL_ATTRIBUTES.get(index)
        if self.perk_database is not None:
            for s in self.perk_database.skills:
                if s.number == index:
                    code = s.attribute
        attribute = self.attribute(code)
        if attribute is not None:
            value += attribute - PLAIN_ATTRIBUTE
        return value

    def school_skill(self, school):
        """
        527eb0: the skill of a school of magic (the subtype of a spell, 0..7).
        """
        school = SCHOOL_REMAP.get(school, school)
        if not 0 <= school < len(SCHOOL_SKILLS):
            return 0.0
        perk = self.perk_power(GROUP_SCHOOLS + school)
        return self.intelligence - PLAIN_ATTRIBUTE + self.skills[SCHOOL_SKILLS[school]] + perk

#------------------------------------------------------------------------------


def update_fighter(fighter, stats):
    """
    5239b0, 529090: the maximum health and mana of the fighter (engine.combat.Fighter) of a unit counted again
    from its HeroStats; 525a70 keeps the part of the health of every part of the body.
    """
    if fighter is None or stats is None:
        return
    parts = [(p, p.health_part) for p in fighter.parts if p.state > 0 and p.max_health > 0]
    fighter.base_max_health = stats.max_health
    fighter.max_mana = stats.max_mana
    fighter.mana = min(fighter.mana, fighter.max_mana)
    fighter.set_skill(stats)
    fighter.recount()
    for part, health_part in parts:
        part.health = health_part * part.max_health


def share_weight(stats):
    """
    5911c0: the weight of a unit in the sharing of the experience of a quest.
    """
    return math.sqrt(max(0.0, stats.experience + stats.waiting_experience))


def share_quest_experience(stats_list, amount):
    """
    5911c0: the experience of a quest shared among the HeroStats of the units of the party in the zone,
    the list of the shares (before the factor of the intelligence).
    """
    weights = [share_weight(s) for s in stats_list]
    total = max(MIN_SHARE_WEIGHT, sum(weights))
    shares = []
    for stats, weight in zip(stats_list, weights):
        share = weight * amount / total
        stats.add_experience(share)
        shares.append(share)
    if _Debug:
        print('quest experience %g shared %s' % (amount, shares))
    return shares


def reward_units(units, amount):
    """
    5911c0: the experience shared among the units (engine.unit.Unit) of the party which have HeroStats,
    their health and mana counted again; the list of the shares.
    """
    units = [u for u in units if u.hero_stats is not None]
    if amount <= 0 or not units:
        return []
    shares = share_quest_experience([u.hero_stats for u in units], amount)
    for u, share in zip(units, shares):
        update_fighter(u.fighter, u.hero_stats)
        # 66a440: the client part of the unit shows the gain (engine.flyinghp)
        on_experience = getattr(u, 'on_experience', None)
        if on_experience is not None:
            on_experience(u, share)
    return shares


def is_rewarded_kill(killer, victim):
    """
    553f40: a kill gives the experience when the killer is of the player and the victim is not.
    """
    if killer.fighter is None or victim.fighter is None:
        return False
    return killer.fighter.in_party and not victim.fighter.in_party


def from_prototype(units, prototype_name, perks=None, settings=None):
    """
    HeroStats of the template of a prototype of units.udb (idb.UnitDatabase), None for a monster.
    """
    template = units.template(prototype_name) if units is not None and prototype_name else None
    if template is None:
        return None
    return HeroStats.from_template(template, perks, settings)
