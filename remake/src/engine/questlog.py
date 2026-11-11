"""
The messages and the sounds of the quests in the battle window (game.exe 636990 and the end of 638240 (partly)).

The scenario keeps a quest in the variable "q.<zone>.<quest>" and its subobjectives in "q.<zone>.<quest>.<n>"
(tools.scenario): 1 given, 2 done, 3 failed. When such a variable changes, 636990 (a virtual method of
the battle window) puts a line into the messages:
- a quest: "Quest received:", "Quest completed:" or "Quest failed:" (combat_get_quest, combat_complete_quest,
  combat_failed_quest of texts.res) and the title of "quest <quest>", the first line of the text;
- a subobjective, only while the variable of its quest is not 0: "Objective received:" and so on
  (combat_*_subobj) and the title of "#subobj <n>" of the text of the quest, the line after it.
game.exe paints the quest lines yellow and the subobjective lines orange (the color codes 07 'C' r g b);
the messages of the remake have no colors.

636990 also raises the sound of the window to the value of the variable (a failure counts as 1, a completed
quest as 3); the next tick of the window (638240, partly) plays it once and drops it: 1 "buttons\\quest.wav",
2 "buttons\\subcomplete.wav", 3 "buttons\\complete.wav". The briefings ("b.<owner>.<briefing>") have three
parts too, so the variables are taken by their prefix "q".
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import re

#------------------------------------------------------------------------------

QUEST_PREFIX = 'q'

DONE = 2.0
FAILED = 3.0

SOUND_COMPLETE = 3
SOUNDS = {
    1: 'buttons\\quest.wav',
    2: 'buttons\\subcomplete.wav',
    SOUND_COMPLETE: 'buttons\\complete.wav',
}

QUEST_TEXTS = {DONE: 'string combat_complete_quest', FAILED: 'string combat_failed_quest'}
QUEST_GET_TEXT = 'string combat_get_quest'
SUBOBJ_TEXTS = {DONE: 'string combat_complete_subobj', FAILED: 'string combat_failed_subobj'}
SUBOBJ_GET_TEXT = 'string combat_get_subobj'

SUBOBJ_RE = re.compile(r'^#subobj\s+(\d+)\s*$')

#------------------------------------------------------------------------------


def quest_vars(game_vars):
    """
    {lowercase name: value} of the variables of the quests and of their subobjectives.
    """
    result = {}
    for name, value in game_vars.items():
        parts = name.split('.')
        if len(parts) in (3, 4) and parts[0].lower() == QUEST_PREFIX:
            result[name.lower()] = float(value)
    return result


def quest_title(text):
    lines = [line.strip() for line in (text or '').replace('\r', '').split('\n') if line.strip()]
    return lines[0] if lines else ''


def subobjective_title(text, number):
    """
    The line after "#subobj <number>" of the text of a quest, '' when there is none.
    """
    lines = [line.strip() for line in (text or '').replace('\r', '').split('\n') if line.strip()]
    for index, line in enumerate(lines):
        match = SUBOBJ_RE.match(line)
        if match and int(match.group(1)) == number and index + 1 < len(lines):
            return lines[index + 1]
    return ''


def change_message(name, value, variables, text_of):
    """
    636990: (message, sound level) of the variable which became value, None when it makes none.
    variables are the lowercase quest_vars, text_of is GameData.text.
    """
    parts = name.split('.')
    quest_text = text_of('quest %s' % parts[2])
    if len(parts) == 3:
        label = text_of(QUEST_TEXTS.get(value, QUEST_GET_TEXT))
        title = quest_title(quest_text)
        level = SOUND_COMPLETE if value == DONE else 1 if value == FAILED else int(value)
    else:
        if variables.get('.'.join(parts[:3]).lower(), 0.0) == 0.0:
            return None
        label = text_of(SUBOBJ_TEXTS.get(value, SUBOBJ_GET_TEXT))
        try:
            title = subobjective_title(quest_text, int(parts[3]))
        except ValueError:
            title = ''
        level = 1 if value == FAILED else int(value)
    message = ' '.join(part for part in ((label or '').strip(), title) if part)
    return message, level


class QuestLog(object):
    """
    The variables of the quests seen by the battle window and the sound it is to play.
    """

    def __init__(self):
        # None until the first look: the variables of a new or a loaded game make no messages
        self.known = None
        self.sound_level = 0

    def __repr__(self):
        return 'QuestLog(%s)' % (None if self.known is None else len(self.known))

    def reset(self):
        self.known = None
        self.sound_level = 0

    def update(self, game_vars, text_of):
        """
        The messages of the variables which changed since the last call; the sound level rises.
        """
        variables = quest_vars(game_vars)
        if self.known is None:
            self.known = variables
            return []
        messages = []
        for name, value in variables.items():
            if self.known.get(name) == value:
                continue
            result = change_message(name, value, variables, text_of)
            if result is not None:
                messages.append(result[0])
                self.sound_level = max(self.sound_level, result[1])
        self.known = variables
        return messages

    def take_sound(self):
        """
        638240 (partly): the sound of the window, once.
        """
        name = SOUNDS.get(self.sound_level)
        self.sound_level = 0
        return name
