"""
The mercenaries of the game (RecalcMercBriefings 591800 and the end of a briefing 591de0).

The ten mercenaries of the campaign live in the bases of the allods: every one has the name "merc<N>" of
the scripts, its prototype of units.udb and its base of map.txt. The scripts give it the topic
"b.merc<N>.n<k>_<N>" when it can be met, hired or left; RecalcMercBriefings picks the topic of the talk
from the party of the player and the base he is in:

    n1    take the mercenary on (the first time)
    n2    leave it in its base (the party is in the base of the mercenary)
    n3    take it back on (it was met or left before)
    n8    meet it
    n9    ask a fourth member (the party has the hero and two mercenaries, the mercenary refuses)
    n10   say goodbye (the scripts set it at the end of a part of the story)
    n11   the mercenary is busy
    amerc_<N>  the scripts set it to 1 when the mercenary comes into the story; the end of a briefing
               makes it 2 when the mercenary joins

591de0 after a briefing "b.merc<N>.n<k>_<N>":
    n1, n3  the mercenary joins the party (prototype of its object "merc<N>" of the map) and amerc_<N> = 2;
    n2      it leaves the party (it stays in the base);
    n8      the topic of the meeting goes away;
    n10     it leaves the party and the map.

The party holds the hero and two mercenaries: n9 is the offer of a fourth member, which every mercenary
refuses (the texts n9_1..n9_10). The mercenaries of the game ask no money for their service (the texts of
n1_5 and n1_6 say it themselves), the price of a mercenary is the place in the party.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import re

#------------------------------------------------------------------------------

# the mercenaries of the basecam, bz2g, bz6g, bz8k, bz9k and bz14h maps (the table of 591800 and their
# objects of the .mob); the names are those of the scripts and of the records of the parties
MERCENARIES = (
    (1, 'bz1g', 'Human Mercenary Warrior'),
    (2, 'bz1g', 'Human Mercenary Bodyguard'),
    (3, 'bz2g', 'Human Mercenary Thief'),
    (4, 'bz6g', 'Human Mercenary Shaivar'),
    (5, 'bz8k', 'Human Mercenary Kania Thief'),
    (6, 'bz8k', 'Human Mercenary Kania Archer'),
    (7, 'bz14h', 'Human Mercenary Hadagan Warrior'),
    (8, 'bz14h', 'Human Mercenary Hadagan Thief'),
    (9, 'bz9k', 'Human Mercenary Kania Paladin'),
    (10, 'bz14h', 'Human Mercenary Hadagan Rogue'),
)

# the mercenaries of a party without its hero (591740 and n9 of 591800)
PARTY_LIMIT = 2

AVAILABLE = 1.0
JOINED = 2.0

_MERC_NAME = re.compile(r'merc(\d+)', re.IGNORECASE)
_TOPIC_NAME = re.compile(r'n(\d+)_(\d+)', re.IGNORECASE)

#------------------------------------------------------------------------------


def mercenary(index):
    """
    (the base, the prototype of units.udb) of a mercenary, None when there is none.
    """
    for number, base, prototype_name in MERCENARIES:
        if number == index:
            return base, prototype_name
    return None


def name_of(index):
    """
    "merc<N>" of the scripts.
    """
    return 'merc%d' % index


def variable(index, topic):
    """
    "b.merc<N>.n<k>_<N>" of a topic.
    """
    return 'b.merc%d.n%d_%d' % (index, topic, index)


def amerc_variable(index):
    return 'amerc_%d' % index


def _get(game_vars, name):
    """
    The value of a variable by its name without the case; 0 when it is not set (FUN_006e8727 finds the
    names of the game without the case).
    """
    if name in game_vars:
        return float(game_vars[name])
    lower = name.lower()
    for key, value in game_vars.items():
        if key.lower() == lower:
            return float(value)
    return 0.0


def _set(game_vars, name, value):
    for key in list(game_vars):
        if key.lower() == name.lower():
            if key != name and name not in game_vars:
                game_vars[name] = value
                del game_vars[key]
            else:
                game_vars[key] = value
            return
    game_vars[name] = value


def recalc(game_vars, party_names=(), mercs_count=None, zone_name=''):
    """
    591800: the topics of the mercenaries by the records of the party, the number of the mercenaries in
    the zone and the base of the player. party_names are the names of the records of the current party
    ("Hero", "merc5"...), mercs_count the living mercenaries of the party in the zone without the hero
    (GetMercsNumber); without it the records less the hero count.
    """
    names = set(str(name).lower() for name in party_names)
    if mercs_count is None:
        mercs_count = max(len(names) - 1, 0)
    zone_name = (zone_name or '').lower()
    for index, base, prototype_name in MERCENARIES:
        in_party = name_of(index).lower() in names
        _set(game_vars, variable(index, 11), 0.0)
        n10 = _get(game_vars, variable(index, 10))
        if in_party:
            if n10 != 0.0:
                for topic in (8, 1, 2, 3, 9):
                    _set(game_vars, variable(index, topic), 0.0)
            else:
                for topic in (8, 1, 3, 9):
                    _set(game_vars, variable(index, topic), 0.0)
                _set(game_vars, variable(index, 2), 1.0 if zone_name == base else 0.0)
            continue
        if n10 != 0.0:
            for topic in (8, 1, 2, 3, 9, 10):
                _set(game_vars, variable(index, topic), 0.0)
            _set(game_vars, amerc_variable(index), 0.0)
            continue
        amerc = _get(game_vars, amerc_variable(index))
        if amerc != 0.0:
            if _get(game_vars, variable(index, 8)) == AVAILABLE:
                for topic in (1, 2, 3, 9):
                    _set(game_vars, variable(index, topic), 0.0)
                _set(game_vars, variable(index, 10), 0.0)
            elif mercs_count < PARTY_LIMIT:
                if amerc != AVAILABLE:
                    _set(game_vars, variable(index, 1), 0.0)
                    _set(game_vars, variable(index, 3), 1.0)
                    _set(game_vars, variable(index, 8), 0.0)
                    _set(game_vars, variable(index, 2), 0.0)
                    _set(game_vars, variable(index, 9), 0.0)
                else:
                    _set(game_vars, variable(index, 1), 1.0)
                    _set(game_vars, variable(index, 3), 0.0)
                    _set(game_vars, variable(index, 8), 0.0)
                    _set(game_vars, variable(index, 2), 0.0)
                    _set(game_vars, variable(index, 9), 0.0)
            else:
                for topic in (1, 3, 8, 2):
                    _set(game_vars, variable(index, topic), 0.0)
                _set(game_vars, variable(index, 9), 1.0)
        else:
            if _get(game_vars, variable(index, 8)) != AVAILABLE:
                _set(game_vars, variable(index, 11), 1.0)


def split_variable(variable_name):
    """
    "b.merc<N>.n<k>_<N>" -> (N, k), None for another variable.
    """
    parts = (variable_name or '').split('.')
    if len(parts) != 3 or parts[0].lower() != 'b':
        return None
    merc = _MERC_NAME.fullmatch(parts[1])
    topic = _TOPIC_NAME.fullmatch(parts[2])
    if merc is None or topic is None:
        return None
    index = int(merc.group(1))
    number = int(topic.group(2))
    if number != index:
        return None
    return index, int(topic.group(1))


def complete(game_vars, variable_name):
    """
    591de0: the end of the briefing of a mercenary. (N, the action) where the action is "hire", "leave",
    "leave_zone" or "met"; None when the variable is not of a mercenary.
    """
    parsed = split_variable(variable_name)
    if parsed is None:
        return None
    index, topic = parsed
    if mercenary(index) is None:
        return None
    if topic in (1, 3):
        _set(game_vars, amerc_variable(index), JOINED)
        return index, 'hire'
    if topic == 2:
        return index, 'leave'
    if topic == 8:
        _set(game_vars, variable(index, 8), 0.0)
        return index, 'met'
    if topic == 10:
        return index, 'leave_zone'
    return index, None
