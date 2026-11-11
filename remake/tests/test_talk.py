"""
Tests of engine.talk: the topics of the owner of a conversation, their titles and the rows of the window.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from engine import talk

#------------------------------------------------------------------------------

VARS = {
    'b.elder.s71_1': 1.0,
    'b.elder.s72': 1.0,
    'b.elder.s73': 2.0,
    'b.smith.s71_1': 1.0,
    'b.elder': 1.0,
    'q.elder.s71': 1.0,
}

TEXTS = {'briefing s72': 'The meeting\r\n#phrase Elder 1\r\nHello.\r\n'}

#------------------------------------------------------------------------------


def text_of(name):
    return TEXTS.get(name)


class TalkTest(unittest.TestCase):

    def test_topics_of_the_owner(self):
        topics = talk.topics(VARS, 'Elder')
        self.assertEqual([topic.name for topic in topics], ['s71_1', 's72'])
        self.assertEqual(topics[0].variable, 'b.elder.s71_1')
        self.assertEqual(talk.topics(VARS, 'smith')[0].name, 's71_1')
        self.assertEqual(talk.topics(VARS, 'nobody'), [])

    def test_title_of_a_topic(self):
        topic = talk.topics(VARS, 'elder')[1]
        self.assertEqual(talk.title_of(topic, text_of), 'The meeting')
        other = talk.topics(VARS, 'elder')[0]
        self.assertEqual(talk.title_of(other, text_of), 's71_1')

    def test_rows_and_choice(self):
        topics = talk.topics(VARS, 'elder')
        value = talk.Talk('elder', topics)
        self.assertEqual(value.selected_topic.name, 's71_1')
        self.assertEqual(value.click(*_center(value.row_rect(1))), 'topic')
        self.assertEqual(value.selected_topic.name, 's72')
        self.assertEqual(value.click(*_center(value.goodbye_rect())), 'goodbye')
        self.assertEqual(value.on_key('down'), None)
        self.assertEqual(value.selected_topic.name, 's72')
        self.assertEqual(value.on_key('up'), None)
        self.assertEqual(value.selected_topic.name, 's71_1')
        self.assertEqual(value.on_key('enter'), 'topic')
        self.assertEqual(value.on_key('escape'), 'goodbye')


def _center(rect):
    left, top, right, bottom = rect
    return (left + right) // 2, (top + bottom) // 2
