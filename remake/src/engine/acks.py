"""
The answers of the units: the groups 10..44 of acks.db (6d8a00, 6d75e0) and the chance of a phrase.

A record of the section 1 of acks.db belongs to a prototype of units.udb and holds groups of phrases
(tag = the group, 0x1c4 bytes of a record); a phrase has the wav of sfx.res (tag 1), the chance of
saying it in percent (tag 2), the Russian text (tag 3) and its name (tag 4). The groups follow the
folders of the sounds: 10 Selected, 11 Move, 12 Attack, 13 Cast, 14 Loot, 15 UseObj, 16 Steal, 17 Follow,
18 UsePot, 19 ChPos, 20 NoWayAtt, 21 CntChPos, 22 NoPath, 23 CantCast, 24 CantTele, 25 SciFail, 26 NoTarget,
27 ComplSp, 28 DecToAtt, 29 OutStam, 30 ArmCrip, 31 LegCrip, 32 Bored, 34 Overload, 35 Injured, 36 BigAtt,
37 ArmorCri, 38 WeapCrip, 39 AttInDef, 40 WaitFoll, 41 Scenario, 42 StealEmp, 43 ShopYes, 44 ShopNo.

The remake takes a phrase of a group at random and says it with the chance of the phrase; the text of the
phrase goes to the messages when it is not empty. say and say_block differ in the script logic which waits
for the end of the phrase (not made in the remake; the answer is in NOTES_DEV2.md).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import random

#------------------------------------------------------------------------------

# the groups of the answers (the folders of Acks\\<voice>\\<folder>)
GROUP_SELECTED = 10
GROUP_MOVE = 11
GROUP_ATTACK = 12
GROUP_CAST = 13
GROUP_LOOT = 14
GROUP_USE_OBJECT = 15
GROUP_STEAL = 16
GROUP_FOLLOW = 17
GROUP_USE_POTION = 18
GROUP_CHANGE_POSITION = 19
GROUP_NO_WAY_ATTACK = 20
GROUP_CANT_CHANGE_POSITION = 21
GROUP_NO_PATH = 22
GROUP_CANT_CAST = 23
GROUP_CANT_TELEPORT = 24
GROUP_SCIENCE_FAIL = 25
GROUP_NO_TARGET = 26
GROUP_COMPLETE_SPELL = 27
GROUP_DECLINE_ATTACK = 28
GROUP_OUT_OF_STAMINA = 29
GROUP_ARM_CRIPPLED = 30
GROUP_LEG_CRIPPLED = 31
GROUP_BORED = 32
GROUP_OVERLOAD = 34
GROUP_INJURED = 35
GROUP_BIG_ATTACK = 36
GROUP_ARMOR_CRIPPLED = 37
GROUP_WEAPON_CRIPPLED = 38
GROUP_ATTACK_IN_DEFENSE = 39
GROUP_WAIT_FOLLOW = 40
GROUP_SCENARIO = 41
GROUP_STEAL_EMPTY = 42
GROUP_SHOP_YES = 43
GROUP_SHOP_NO = 44

# the messages of the wear of the items (engine.combat.MESSAGE_*, 552780) -> the groups of the answers;
# acks.db has one group for a crippled armor and one for a crippled weapon
ITEM_MESSAGE_GROUPS = {
    0x23: GROUP_ARMOR_CRIPPLED,
    0x24: GROUP_WEAPON_CRIPPLED,
    0x25: GROUP_ARMOR_CRIPPLED,
    0x26: GROUP_WEAPON_CRIPPLED,
}

# 548520 (partly): the reasons of engine.unit.Unit.run_limit (RUN_LEGS 1, RUN_OVERLOAD 2) -> the groups of the answers
RUN_LIMIT_GROUPS = {
    1: GROUP_LEG_CRIPPLED,
    2: GROUP_OVERLOAD,
}

# the mood signals (engine.signals) -> the nearest groups of acks.db: the game has other folders for them
SIGNAL_GROUPS = {
    5: GROUP_BORED,
    6: GROUP_BIG_ATTACK,
    7: GROUP_ATTACK,
    8: GROUP_ATTACK,
    9: GROUP_BIG_ATTACK,
}

#------------------------------------------------------------------------------


def item_message_group(message):
    """
    The group of the answer about a worn item, None for another message.
    """
    return ITEM_MESSAGE_GROUPS.get(message)


def run_limit_group(reason):
    """
    The group of the answer of a unit which may not run, None when it may.
    """
    return RUN_LIMIT_GROUPS.get(reason)


def signal_group(level):
    """
    The group of the voice of a mood signal, None for a signal without a voice.
    """
    return SIGNAL_GROUPS.get(level)


def player_for(game_data, unit, rng=None):
    """
    The AcksPlayer of a unit by the prototype name of the unit, None when acks.db knows no such prototype.
    """
    if unit is None or not getattr(unit, 'prototype_name', None):
        return None
    acks = game_data.acks().get(unit.prototype_name)
    if acks is None:
        return None
    return AcksPlayer(acks, rng=rng)


class AcksPlayer(object):
    """
    The phrases of one prototype; choose() returns a phrase of a group or None.
    """

    def __init__(self, acks, rng=None):
        self.acks = acks
        self.rng = rng or random.Random()

    def __repr__(self):
        return 'AcksPlayer(%s)' % self.acks.name

    def choose(self, group):
        """
        6d8a00: a phrase of the group at random, said with its chance in percent.
        """
        phrases = self.acks.groups.get(group) or ()
        if not phrases:
            return None
        phrase = phrases[self.rng.randrange(len(phrases))]
        if self.rng.randrange(100) >= phrase.chance:
            return None
        return phrase


class PhraseSound(object):
    """
    The answer to play: the wav of sfx.res and the place of the unit.
    """

    def __init__(self, phrase, position, block=False):
        self.phrase = phrase
        self.name = phrase.sound_name
        self.position = tuple(position)
        self.text = phrase.text
        self.block = block
        # the distances of the voice (510890 and the other sounds of the units)
        self.near = 10.0
        self.far = 50.0

    def __repr__(self):
        return 'PhraseSound(%s, %s)' % (self.phrase.name, self.position)
