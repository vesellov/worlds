"""
Tests of engine.questlog: the messages and the sounds of the variables of the quests (636990, 638240).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from engine import questlog

#------------------------------------------------------------------------------

TEXTS = {
    'string combat_get_quest': 'Quest received:\r\n',
    'string combat_complete_quest': 'Quest completed:\r\n',
    'string combat_failed_quest': 'Quest failed:\r\n',
    'string combat_get_subobj': 'Objective received:\r\n',
    'string combat_complete_subobj': 'Objective completed:\r\n',
    'string combat_failed_subobj': 'Objective failed:\r\n',
    'quest q72h': 'Portal\r\nThe portal is the only way.\r\n#subobj  1\r\nDestroy your enemies\r\nThe guards.\r\n'
                  '#subobj  2\r\nCross the western bridge\r\nGo.\r\n',
}

#------------------------------------------------------------------------------


class QuestLogTest(unittest.TestCase):

    def test_first_look_is_silent(self):
        log = questlog.QuestLog()
        self.assertEqual(log.update({'q.zone7.q72h': 1.0}, TEXTS.get), [])
        self.assertIsNone(log.take_sound())

    def test_quest_given_and_done(self):
        log = questlog.QuestLog()
        game_vars = {}
        log.update(game_vars, TEXTS.get)
        game_vars['q.zone7.q72h'] = 1.0
        self.assertEqual(log.update(game_vars, TEXTS.get), ['Quest received: Portal'])
        self.assertEqual(log.take_sound(), 'buttons\\quest.wav')
        self.assertIsNone(log.take_sound())
        game_vars['q.zone7.q72h'] = 2.0
        self.assertEqual(log.update(game_vars, TEXTS.get), ['Quest completed: Portal'])
        self.assertEqual(log.take_sound(), 'buttons\\complete.wav')
        # no change, no message
        self.assertEqual(log.update(game_vars, TEXTS.get), [])

    def test_subobjectives(self):
        log = questlog.QuestLog()
        game_vars = {'q.zone7.q72h': 0.0}
        log.update(game_vars, TEXTS.get)
        # the quest is not given yet: its subobjective is silent
        game_vars['q.zone7.q72h.1'] = 1.0
        self.assertEqual(log.update(game_vars, TEXTS.get), [])
        game_vars['q.zone7.q72h'] = 1.0
        game_vars['q.zone7.q72h.2'] = 2.0
        self.assertEqual(log.update(game_vars, TEXTS.get),
                         ['Quest received: Portal', 'Objective completed: Cross the western bridge'])
        # the louder of the two sounds of the tick
        self.assertEqual(log.take_sound(), 'buttons\\subcomplete.wav')
        game_vars['q.zone7.q72h.1'] = 3.0
        self.assertEqual(log.update(game_vars, TEXTS.get), ['Objective failed: Destroy your enemies'])
        self.assertEqual(log.take_sound(), 'buttons\\quest.wav')

    def test_briefings_are_not_quests(self):
        log = questlog.QuestLog()
        game_vars = {}
        log.update(game_vars, TEXTS.get)
        game_vars['b.Baron.brief_347'] = 1.0
        game_vars['z.zone7'] = 1.0
        self.assertEqual(log.update(game_vars, TEXTS.get), [])
        self.assertIsNone(log.take_sound())

    def test_reset_after_a_load(self):
        log = questlog.QuestLog()
        log.update({}, TEXTS.get)
        log.reset()
        self.assertEqual(log.update({'q.zone7.q72h': 2.0}, TEXTS.get), [])


if __name__ == '__main__':
    unittest.main()
