"""
Tests of engine.saveload: info.sav, the folders of saves/ and the list window of 633770 and 634750.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import os
import struct
import tempfile
import unittest

#------------------------------------------------------------------------------

from eilib import sav
from engine import saveload

#------------------------------------------------------------------------------

DATA_DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

#------------------------------------------------------------------------------


def _info(hour, strings):
    data = struct.pack('<IIf', saveload.INFO_SIGNATURE, len(strings), hour)
    return data + b''.join(s.encode('cp1251') + b'\0' for s in strings) + b'\1\2\3\4'


class SaveLoadTest(unittest.TestCase):

    def test_read_info(self):
        hour, strings = saveload.read_info(_info(6.5, ('gipat', 'gz1g')))
        self.assertAlmostEqual(hour, 6.5)
        self.assertEqual(strings, ('gipat', 'gz1g'))
        self.assertEqual(saveload.read_info(b'not a save'), (0.0, ()))

    def test_sample_info_sav(self):
        # the 28 byte info.sav of the save of the game: allod gipat, zone gz1g, hour 6, stamp 0x6af2561d
        with open(os.path.join(DATA_DIR_PATH, 'savea', sav.INFO_NAME), 'rb') as f:
            data = f.read()
        info = sav.read_info(data)
        self.assertEqual((info.allod, info.zone), ('gipat', 'gz1g'))
        self.assertAlmostEqual(info.hour, 6.0)
        self.assertEqual(info.stamp, b'\x1d\x56\xf2\x6a')
        self.assertEqual(saveload.read_info(data), (info.hour, info.strings))

    def test_slot_titles(self):
        # 653a10: saveq, savea and saveb are shown by their texts of texts.res, the other folders by name
        texts = {'string save_quick': 'QUICK SAVE\r\n', 'string save_auto_enter': 'Autosave (when entering)\r\n'}
        self.assertEqual(saveload.slot_title('saveq', texts.get), 'QUICK SAVE')
        self.assertEqual(saveload.slot_title('savea', texts.get), 'Autosave (when entering)')
        self.assertEqual(saveload.slot_title('saveb', texts.get), 'saveb')
        self.assertEqual(saveload.slot_title('save3', texts.get), 'save3')

    def test_autosave_on_the_tenth_tick(self):
        # 5641f0 sends "autosave" once, when the counter of the ticks of the zone becomes 10
        self.assertFalse(saveload.autosave_due(0, 9))
        self.assertTrue(saveload.autosave_due(9, 10))
        self.assertTrue(saveload.autosave_due(8, 12))
        self.assertFalse(saveload.autosave_due(10, 11))

    def test_headers_of_the_slot_files(self):
        scenario = struct.pack('<II', sav.SCENARIO_SIGNATURE, sav.VERSION) + b'body'
        mission = struct.pack('<II', sav.MISSION_SIGNATURE, sav.VERSION) + b'body'
        self.assertTrue(sav.is_save(scenario, sav.SCENARIO_NAME))
        self.assertTrue(sav.is_save(mission, sav.MISSION_NAME))
        self.assertFalse(sav.is_save(mission, sav.SCENARIO_NAME))
        header = sav.read_header(mission, sav.MISSION_NAME)
        self.assertEqual((header.signature, header.version), (sav.MISSION_SIGNATURE, sav.VERSION))
        self.assertTrue(header.recognized)
        self.assertEqual(header.size, len(mission))
        self.assertFalse(sav.read_header(b'??', sav.MISSION_NAME).recognized)
        self.assertIsNone(sav.read_info(scenario))

    def test_unpack(self):
        # 430960: three literals ('a', 'b', 'c' between the bytes of the bits 0x18 and 0x08) and a copy of
        # the length 3 from 3 bytes back
        self.assertEqual(sav.unpack(struct.pack('<I', 6) + b'\x18abc\x08'), b'abcabc')
        self.assertEqual(sav.unpack(struct.pack('<I', 0)), b'')
        with self.assertRaises(sav.SaveError):
            sav.unpack(struct.pack('<I', 6) + b'\x18ab')

    def test_body(self):
        # 579830: the checksum, the byte of the packing and the body; a version up to 0x64 is never packed
        body = b'zone1.mob\0zone1.mpr\0gz1g\0rest'
        raw = struct.pack('<IIIB', sav.MISSION_SIGNATURE, sav.VERSION, sav.checksum(body), 0) + body
        save = sav.read_body(raw)
        self.assertFalse(save.packed)
        self.assertEqual(save.body, body)
        self.assertEqual(sav.mission_files(save.body), ('zone1.mob', 'zone1.mpr', 'gz1g'))
        old = sav.read_body(struct.pack('<II', sav.MISSION_SIGNATURE, sav.PACKED_VERSION) + body)
        self.assertEqual(old.body, body)
        packed = struct.pack('<IIIB', sav.SCENARIO_SIGNATURE, sav.VERSION, 1, 1) + struct.pack('<I', 6)
        with self.assertRaises(sav.SaveError):
            sav.read_body(packed + b'\x18abc\x08')

    def test_nodes(self):
        # 409070: the id, the size * 2 in a byte or the size * 2 + 1 in a dword, the data
        long_data = b'x' * 200
        data = b'\x01\x04ab' + b'\x02' + struct.pack('<I', 200 * 2 + 1) + long_data + b'\x03\x00'
        self.assertEqual(sav.read_nodes(data), [(1, b'ab'), (2, long_data), (3, b'')])
        with self.assertRaises(sav.SaveError):
            sav.read_nodes(b'\x01\x08ab')
        blob, end = sav.read_blob(struct.pack('<H', 4) + b'ab!')
        self.assertEqual((blob, end), (b'ab', 4))

    def test_sample_scenario_sav(self):
        # scenario.sav of the save of the game: packed, the checksum matches, a blob of the nodes 1, 0, 2 where
        # the node 2 keeps the 5 traders and the scenario
        with open(os.path.join(DATA_DIR_PATH, 'savea', sav.SCENARIO_NAME), 'rb') as f:
            data = f.read()
        save = sav.read_body(data)
        self.assertTrue(save.packed)
        self.assertEqual(len(save.body), 8314)
        self.assertEqual(save.checksum, 0x30748c52)
        tree, end = sav.read_blob(save.body)
        nodes = sav.read_nodes(tree)
        self.assertEqual([node_id for node_id, _ in nodes], [1, 0, 2])
        traders = sav.read_nodes(nodes[2][1])
        self.assertEqual([node_id for node_id, _ in traders], [1] * 6 + [2])
        self.assertIn(b'gipat low helm', traders[0][1])

    def test_party_of_scenario_sav(self):
        # 661bd0, 65fda0: the party in gz1g with the hero, its 7 items and the 2 spells of its scrolls
        with open(os.path.join(DATA_DIR_PATH, 'savea', sav.SCENARIO_NAME), 'rb') as f:
            data = f.read()
        parties = sav.scenario_parties(sav.read_body(data).body)
        self.assertEqual(len(parties), 1)
        party = parties[0]
        self.assertEqual(party.zone, 'gz1g')
        self.assertEqual([(i.object_id, i.class_id, i.low, i.high) for i in party.items[:5]], [
            (2, sav.CLASS_ARMOR, 28, 97), (4, sav.CLASS_ARMOR, 28, 98), (6, sav.CLASS_ARMOR, 28, 99),
            (8, sav.CLASS_ARMOR, 28, 100), (10, sav.CLASS_WEAPON, 21, 9)])
        self.assertEqual([s.high for s in party.spells], [24, 12])
        hero = party.units[0]
        self.assertEqual((hero.name, hero.prototype_name, hero.model_name, hero.stats_name),
                         ('Hero', 'Human Hero', 'unhuma', 'Human Hero'))
        self.assertEqual(hero.item_lists, [[10], [], [2, 4, 6, 8], [12, 14]])
        stats = hero.stats
        self.assertAlmostEqual(stats.experience, 271.0)
        self.assertEqual((stats.strength, stats.dexterity, stats.intelligence), (23.0, 30.0, 25.0))
        self.assertEqual(stats.science, 3)
        self.assertEqual(stats.skills, [10, 0, 0, 0, 12, 9, 0, 0])

    def test_slots(self):
        with tempfile.TemporaryDirectory() as data_dir_path:
            saves_dir_path = os.path.join(data_dir_path, saveload.SAVES_DIR_NAME)
            os.mkdir(saves_dir_path)
            empty_dir_path = os.path.join(saves_dir_path, 'nofiles')
            os.mkdir(empty_dir_path)
            slot_dir_path = os.path.join(saves_dir_path, 'savea')
            os.mkdir(slot_dir_path)
            with open(os.path.join(slot_dir_path, saveload.MISSION_NAME), 'wb') as f:
                f.write(b'\x02\1\0\0')
            with open(os.path.join(slot_dir_path, saveload.INFO_NAME), 'wb') as f:
                f.write(_info(7.0, ('ingos', 'zone12')))
            found = saveload.slots(data_dir_path)
            self.assertEqual([slot.name for slot in found], ['savea'])
            self.assertEqual((found[0].allod, found[0].zone), ('ingos', 'zone12'))
            self.assertAlmostEqual(found[0].hour, 7.0)

    def test_dialog(self):
        entries = [saveload.Slot('a', ''), saveload.Slot('b', '')]
        dialog = saveload.Dialog(entries)
        self.assertEqual(dialog.selected_slot.name, 'a')
        self.assertEqual(dialog.click(*_center(dialog.row_rect(1))), 'select')
        self.assertEqual(dialog.selected_slot.name, 'b')
        self.assertEqual(dialog.click(*_center(saveload.OK_RECT)), 'confirm')
        self.assertEqual(dialog.click(*_center(saveload.CANCEL_RECT)), 'cancel')
        self.assertEqual(dialog.on_key('escape'), 'cancel')
        self.assertEqual(dialog.on_key('enter'), 'confirm')
        self.assertEqual(dialog.on_key('down'), None)
        self.assertEqual(dialog.selected_slot.name, 'b')

    def test_empty_dialog(self):
        dialog = saveload.Dialog([])
        self.assertIsNone(dialog.selected_slot)
        dialog.select(2)
        self.assertEqual(dialog.selected, 0)


def _center(rect):
    left, top, right, bottom = rect
    return (left + right) // 2, (top + bottom) // 2
