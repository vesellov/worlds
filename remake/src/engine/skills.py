"""
The page of the hero in the camp: the skills bought with the free experience and the abilities which the
skills unlock (CCampWindow, 624ed0, 6268e0, 626160).

A skill costs the price of its next level (528ab0, 528da0): the skill of the level n stands for
Skill Val 1 * 1.61 ^ (n / Skill Val 2) of the experience, the camp adds the difference to the experience
of the skills, so the free part falls by that difference. A skill stops at 99.

An ability (perk) of perks.pdb is taken when (5260a0):
- its previous ability of the group is known at the level before it (the code "none" starts a group);
- the skill it needs (tag 11, the numbers of the skills of the sections 1) is at the level of the tag 18;
- the attributes of the tags 14..17, when they are not zero, are not above the attributes of the unit.
The abilities go by threes: the levels 1..3 of a group (index // 3). The game keeps the level of every
group in a byte, engine.experience.HeroStats.perk_levels.

Its price (528050) is the price of the level of its skill minus the price of the level of the previous
ability: cost = skill_price(required) - skill_price(previous required), 0 for the first level. The camp
adds the price to the experience spent on the perks (+0x28), so the free part falls by it; buying takes the
ability at once (526450), the health and the stamina of the unit are counted again (524560).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from engine import experience

#------------------------------------------------------------------------------

# the rows of the window in the order of the skills of perks.pdb and the science (s_Melee..s_Science, 793bc8)
SKILL_ROWS = (
    (experience.SKILL_MELEE, 'Melee'),
    (experience.SKILL_ARCHERY, 'Archery'),
    (experience.SKILL_BACKSTAB, 'Backstab'),
    (experience.SKILL_ELEMENTAL, 'Elemental Magic'),
    (experience.SKILL_SENSE, 'Sensory Magic'),
    (experience.SKILL_ASTRAL, 'Astral Magic'),
    (experience.SKILL_STEALTH, 'Stealth'),
    (experience.SKILL_AWARENESS, 'Awareness'),
    (experience.SKILL_SCIENCE, 'Use/Steal'),
)

# 624ed0: a skill stops at 99
MAX_SKILL = 99

#------------------------------------------------------------------------------


def perk_experience(stats):
    """
    The experience paid for the perks, the field +0x28 of the stats of game.exe (HeroStats.perk_experience).
    """
    return stats.perk_experience


def free_experience(stats):
    """
    The experience which the camp may spend: the free part of engine.experience.HeroStats, the perks already
    paid are in its spent part.
    """
    return stats.free_experience


def skill_cost(stats, index):
    """
    528ab0: the price of the next level of a skill; None when the skill is at its top.
    """
    level = stats.skill(index)
    if level >= MAX_SKILL:
        return None
    return stats.settings.skill_raise_price(level)


def raise_skill(stats, index):
    """
    The click on a skill of the camp: True when the level was bought, False when the free experience is
    not enough or the skill is at its top. The fighter is counted again by the caller (524560).
    """
    price = skill_cost(stats, index)
    if price is None or price > free_experience(stats):
        return False
    if not stats.raise_skill(index):
        return False
    if _Debug:
        print('skill %s -> %d for %d' % (index, stats.skill(index), price))
    return True


def skill_lines(stats):
    """
    (index, level, price) of the rows of the skills; price is None at the top of a skill.
    """
    if stats is None:
        return []
    return [(index, stats.skill(index), skill_cost(stats, index)) for index, _ in SKILL_ROWS]


def _skill_number(perk):
    """
    The number of the skill of a perk: its code (tag 11) through engine.experience.SKILL_CODES; None for
    the code "none".
    """
    code = (perk.skill_code or '').lower()
    if not code or code == 'none':
        return None
    return experience.SKILL_CODES.get(code)


def perk_cost(stats, perk):
    """
    528050: the price of the level of the skill of the perk minus the price of the level of the perk
    before it; None without the numbers of the skills.
    """
    if stats is None or perk is None or stats.perk_database is None:
        return None
    previous = stats.perk_database.perk(perk.previous_code)
    previous_required = previous.required if previous is not None else 0
    return stats.settings.skill_raise_price(previous_required, max(1, perk.required - previous_required))


def perk_available(stats, perk):
    """
    5260a0: the ability can be taken now.
    """
    if stats is None or perk is None or stats.perk_database is None:
        return False
    if not 0 <= perk.group < len(stats.perk_levels):
        return False
    flags = getattr(stats.perk_database, 'flags', None)
    if flags is not None and 0 <= perk.index < len(flags) and not flags[perk.index]:
        return False
    level = stats.perk_levels[perk.group]
    if level >= perk.level:
        return False
    # the level before it, or the first one of the group
    if perk.level > 1:
        previous = stats.perk_database.perk(perk.previous_code)
        if previous is None or stats.perk_levels[previous.group] < perk.level - 1:
            return False
    elif (perk.previous_code or '').lower() not in ('', 'none'):
        return False
    number = _skill_number(perk)
    if number is not None and stats.skill(number) < perk.required:
        return False
    # the attributes of the tags 14..17 (strength, dexterity, intelligence) are not kept by idb.Perk
    return True


def known_perks(stats):
    """
    The abilities of the groups the unit has, one perk of a group (the highest level).
    """
    if stats is None or stats.perk_database is None:
        return []
    known = []
    for group, level in enumerate(stats.perk_levels):
        perk = stats.perk_database.level_perk(group, level)
        if perk is not None:
            known.append(perk)
    return known


def available_perks(stats):
    """
    (perk, price) of every ability which the unit can take now, in the order of perks.pdb.
    """
    if stats is None or stats.perk_database is None:
        return []
    result = []
    for perk in stats.perk_database.perks:
        if perk_available(stats, perk):
            result.append((perk, perk_cost(stats, perk)))
    return result


def buy_perk(stats, perk):
    """
    The click on an ability of the camp: True when it was taken and its price went to the experience of the
    perks. The caller counts the fighter again.
    """
    if not perk_available(stats, perk):
        return False
    price = perk_cost(stats, perk)
    if price is None or price > free_experience(stats):
        return False
    if not stats.add_perk(perk.code):
        return False
    stats.perk_experience += price
    if _Debug:
        print('perk %s for %d' % (perk.code, price))
    return True
