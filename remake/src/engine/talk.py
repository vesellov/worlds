"""
The conversation with a peaceful unit on a base: the topics which the scripts have prepared
(FUN_00607320 @ 607320, the window of FUN_006071a0 @ 6071a0, the rows of FUN_00608b20 @ 608b20).

A topic is a variable of the game "b.<owner>.<name>" whose value is 1; the scripts set it to 1 before the
conversation (GSSetVar( 0, "b.elder.s71_1", 1 ) on bz18h). The owner is the OBJNAME of the unit. The window
of the talk lists the topics of the owner (five rows of 24 pixels and "goodbye"), a click on a topic plays
the briefing "briefing <name>" and its end sets the variable to 2 and runs #OnBriefingComplete
(ScriptWorld.complete_briefing, 592090).

The remake keeps the topics of an owner, the chosen row and the pages of its five rows; the title of
a topic is the first line of its "briefing <name>" text, the name itself when there is no such text.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

BRIEFING_PREFIX = 'b.'
BRIEFING_AVAILABLE = 1.0

# 6071a0 and 608b20: the rows of the topics at y 34 + 24 i and five of them at once
ROWS_TOP = 34
ROW_HEIGHT = 24
VISIBLE_ROWS = 5
ROWS_RECT = (15, 18, 380, 154)

# 608b20: the row of goodbye under the topics
GOODBYE_ROW = 5

GOODBYE_TEXT = 'goodbye'
TITLE_FORMAT = 'Talk to %s'

#------------------------------------------------------------------------------


class TalkTopic(object):
    """
    A topic of the conversation: the variable "b.<owner>.<name>", the name and the title of its briefing.
    """

    def __init__(self, variable, name, title=''):
        self.variable = variable
        self.name = name
        self.title = title or name

    def __repr__(self):
        return 'TalkTopic(%s, %s)' % (self.name, self.variable)

#------------------------------------------------------------------------------


def topics(game_vars, owner_name):
    """
    607320: the variables "b.<owner>.<name>" with the value 1, sorted by their names.
    """
    owner = (owner_name or '').lower()
    result = []
    for variable, value in (game_vars or {}).items():
        parts = variable.split('.')
        if len(parts) != 3 or parts[0].lower() != 'b':
            continue
        if parts[1].lower() != owner or float(value) != BRIEFING_AVAILABLE:
            continue
        result.append(TalkTopic(variable, parts[2]))
    result.sort(key=lambda topic: topic.name.lower())
    return result


def title_of(topic, text_of):
    """
    The first line of "briefing <name>", the name of the topic when there is no such text.
    """
    text = text_of('briefing %s' % topic.name) if text_of is not None else None
    if text:
        first = text.replace('\r', '').split('\n', 1)[0].strip()
        if first:
            topic.title = first
    return topic.title

#------------------------------------------------------------------------------


class Talk(object):
    """
    The state of the window of a conversation: the owner, the topics and the chosen row.
    """

    def __init__(self, owner_name, topics_list=()):
        self.owner_name = owner_name
        self.topics = list(topics_list)
        self.selected = 0
        self.scroll = 0

    def __repr__(self):
        return 'Talk(%s, %d topics)' % (self.owner_name, len(self.topics))

    @property
    def is_empty(self):
        return not self.topics

    @property
    def selected_topic(self):
        if 0 <= self.selected < len(self.topics):
            return self.topics[self.selected]
        return None

    def select(self, index):
        if not self.topics:
            return
        index = min(max(index, 0), len(self.topics) - 1)
        self.selected = index
        if index < self.scroll:
            self.scroll = index
        elif index >= self.scroll + VISIBLE_ROWS:
            self.scroll = index - VISIBLE_ROWS + 1

    def row_rect(self, row):
        top = ROWS_TOP + row * ROW_HEIGHT
        return (ROWS_RECT[0], top, ROWS_RECT[2], top + ROW_HEIGHT)

    def goodbye_rect(self):
        return self.row_rect(GOODBYE_ROW)

    def click(self, x, y):
        """
        "topic", "goodbye" or None.
        """
        for row in range(VISIBLE_ROWS):
            if _inside(self.row_rect(row), x, y):
                index = self.scroll + row
                if index < len(self.topics):
                    self.select(index)
                    return 'topic'
        if _inside(self.goodbye_rect(), x, y):
            return 'goodbye'
        return None

    def on_key(self, name):
        if name == 'up':
            self.select(self.selected - 1)
        elif name == 'down':
            self.select(self.selected + 1)
        elif name in ('enter', 'spacebar'):
            return 'topic'
        elif name == 'escape':
            return 'goodbye'
        return None

#------------------------------------------------------------------------------


def _inside(rect, x, y):
    left, top, right, bottom = rect
    return left <= x < right and top <= y < bottom
