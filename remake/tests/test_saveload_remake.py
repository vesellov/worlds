"""
H1: the state of the remake in a save slot: remake.json, info.sav and the round trip of the party,
the roster and the fight.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import os
import tempfile
import unittest

#------------------------------------------------------------------------------

from eilib import gamedata
from eilib import sav as sav_module
from engine import combat
from engine import equipment
from engine import experience
from engine import party as party_module
from engine import roster as roster_module
from engine import saveload

#------------------------------------------------------------------------------


class SaveStateTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.game_data = gamedata.GameData()

    def make_roster(self):
        roster = roster_module.new_roster('Human Hero', ['saber.steel'], 'unhuma')
        record = roster.default.lead()
        settings = experience.RpgSettings()
        record.hero_stats = experience.from_prototype(self.game_data.units(), 'Human Hero',
                                                      self.game_data.perks(), settings)
        record.hero_stats.experience = 1234.0
        record.hero_stats.skills[experience.SKILL_MELEE] = 17
        record.hero_stats.perk_experience = 11.0
        item = self.game_data.items()
        measured = equipment.make_item(item, 'saber.steel')
        measured.durability = 3.5
        measured.count = 2
        record.worn_items = {'saber.steel': measured}
        record.quick_items = ['saber.steel']
        return roster

    def test_info_of_a_slot(self):
        data = sav_module.build_info(9.5, ['gipat', 'gz1g'])
        info = sav_module.read_info(data)
        self.assertIsNotNone(info)
        self.assertEqual(info.hour, 9.5)
        self.assertEqual(info.strings, ('gipat', 'gz1g'))

    def test_roster_round_trip(self):
        roster = self.make_roster()
        data = saveload.roster_state(roster)
        restored = saveload.restore_roster(roster_module.Roster(), data, self.game_data)
        record = restored.default.lead()
        self.assertEqual(record.name, 'Hero')
        self.assertEqual(record.prototype_name, 'Human Hero')
        self.assertEqual(record.item_names, ['saber.steel'])
        self.assertAlmostEqual(record.worn_items['saber.steel'].durability, 3.5)
        self.assertEqual(record.worn_items['saber.steel'].count, 2)
        self.assertAlmostEqual(record.hero_stats.experience, 1234.0)
        self.assertEqual(record.hero_stats.skills[experience.SKILL_MELEE], 17)
        self.assertAlmostEqual(record.hero_stats.perk_experience, 11.0)
        self.assertEqual(record.quick_items, ['saber.steel'])

    def test_party_round_trip(self):
        party = party_module.Party(self.game_data.items(), self.game_data.quests())
        party.give_money(456)
        party.give_item('long sword.iron')
        quest_item = next(iter(self.game_data.items().quest_items))
        party.quest_items.append(quest_item)
        party.completed_quests = ['The first quest']
        data = saveload.party_state(party)
        restored = saveload.restore_party(
            party_module.Party(self.game_data.items(), self.game_data.quests()), data, self.game_data)
        self.assertEqual(restored.money, 456)
        self.assertEqual(restored.items, ['long sword.iron'])
        self.assertEqual(restored.quest_items[0].number, quest_item.number)
        self.assertEqual(restored.completed_quests, ['The first quest'])

    def test_fighter_round_trip(self):
        class Fighter(object):
            def __init__(self):
                self.parts = [combat.BodyPart('torso', 0, 1.0, 1.0, 100.0),
                              combat.BodyPart('head', 1, 1.0, 1.0, 20.0)]
                self.mana = 30.0
                self.max_mana = 50.0
                self.start_health = 100.0
                self.enchantments = []

        class Unit(object):
            fighter = Fighter()
            strike_wait = 4.0
            is_dead = False

        unit = Unit()
        unit.fighter.parts[0].health = 40.0
        unit.fighter.parts[1].health = 5.0
        data = saveload.fighter_state(unit)
        unit.fighter.parts[0].health = 100.0
        unit.fighter.mana = 1.0
        saveload.restore_fighter(unit, data)
        self.assertEqual(unit.fighter.parts[0].health, 40.0)
        self.assertEqual(unit.fighter.parts[1].health, 5.0)
        self.assertEqual(unit.fighter.mana, 30.0)
        self.assertEqual(unit.strike_wait, 4.0)

    def test_dead_body_stays(self):
        class Clip(object):
            length = 30.0

        class Player(object):
            clip = None
            time = 0.0
            finished = False

        class Unit(object):
            def __init__(self):
                self.player = Player()
                self.mode = 'idle'

            def die(self):
                self.mode = 'dead'
                self.player.clip = Clip()

        unit = Unit()
        saveload.lie_dead(unit)
        self.assertEqual(unit.mode, 'dead')
        self.assertEqual(unit.player.time, 30.0)
        self.assertTrue(unit.player.finished)

    def test_slot_files(self):
        with tempfile.TemporaryDirectory() as dir_path:
            saves_path = os.path.join(dir_path, saveload.SAVES_DIR_NAME)
            slot_path = os.path.join(saves_path, 'slot1')
            state = {'version': saveload.REMAKE_VERSION, 'hour': 7.0, 'allod': 'gipat',
                     'zone_name': 'gz1g', 'map_name': 'zone1'}
            saveload.save_slot(slot_path, state)
            loaded = saveload.load_slot(slot_path)
            self.assertEqual(loaded['map_name'], 'zone1')
            found = saveload.slots(dir_path)
            self.assertEqual(len(found), 1)
            self.assertTrue(found[0].remake)
            self.assertEqual(found[0].zone, 'gz1g')

    def test_slot_of_the_game_is_kept(self):
        # savea of game.exe: the remake adds remake.json and leaves info.sav and mission.sav as they are
        with tempfile.TemporaryDirectory() as dir_path:
            slot_path = os.path.join(dir_path, saveload.SAVES_DIR_NAME, saveload.AUTO_ENTER_SLOT_NAME)
            os.makedirs(slot_path)
            for name in (saveload.INFO_NAME, saveload.MISSION_NAME):
                with open(os.path.join(slot_path, name), 'wb') as f:
                    f.write(b'game')
            saveload.save_slot(slot_path, {'version': saveload.REMAKE_VERSION, 'hour': 7.0, 'map_name': 'zone1'})
            with open(os.path.join(slot_path, saveload.INFO_NAME), 'rb') as f:
                self.assertEqual(f.read(), b'game')
            self.assertEqual(saveload.load_slot(slot_path)['map_name'], 'zone1')


if __name__ == '__main__':
    unittest.main()
