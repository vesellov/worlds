"""
Tests of the engine which repeat the behavior found in game.exe. The tests which read the original data
are skipped when the game folder (EI_DATA_PATH) is not there.

    PYTHONPATH=src venv/bin/python -m unittest discover -s tests
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import datetime
import math
import os
import random
import struct
import unittest

import numpy as np

#------------------------------------------------------------------------------

from eilib import adb
from eilib import cam
from eilib import reg
from eilib import sav
from eilib import anm
from eilib import gamedata
from eilib import idb
from eilib import script as script_lang
from eilib import zones
from engine import actions
from engine import ai
from engine import animation
from engine import combat
from engine import difficulty as difficulty_module
from engine import equipment
from engine import experience
from engine import figure
from engine import hud
from engine import inventory
from engine import lever
from engine import lighting
from engine import lightning
from engine import magic
from engine import menus
from engine import missiles
from engine import music
from engine import particles
from engine import party as party_module
from engine import pathing
from engine import prop
from engine import quat
from engine import roster as roster_module
from engine import saveload
from engine import script
from engine import signals
from engine import sounds
from engine import stats
from engine import terrain
from engine import timing
from engine import trap as trap_module
from engine import tutorial
from engine import unit as unit_module
from engine import water
from engine import world
from client import gl

#------------------------------------------------------------------------------


def _game_data():
    try:
        return gamedata.GameData()
    except IOError:
        return None

#------------------------------------------------------------------------------


class QuatTest(unittest.TestCase):

    def test_rotate_matches_matrix(self):
        q = quat.from_axis_angle((0.3, -0.2, 0.9), 1.1)
        v = np.array([0.4, 1.5, -0.7])
        np.testing.assert_allclose(quat.rotate(q, v), quat.to_matrix(q) @ v, atol=1e-12)

    def test_slerp_takes_the_shortest_arc(self):
        a = quat.from_axis_angle((0, 0, 1), 0.1)
        b = -quat.from_axis_angle((0, 0, 1), 0.3)
        middle = quat.slerp(a, b, 0.5)
        expected = quat.from_axis_angle((0, 0, 1), 0.2)
        np.testing.assert_allclose(middle, expected, atol=1e-9)

    def test_slerp_of_close_keys_is_lerp(self):
        a = quat.from_axis_angle((1, 0, 0), 0.001)
        b = quat.from_axis_angle((1, 0, 0), 0.002)
        np.testing.assert_allclose(quat.slerp(a, b, 0.25), a * 0.75 + b * 0.25, atol=1e-15)


class TrackTest(unittest.TestCase):

    def make_track(self):
        rotations = np.array([quat.from_axis_angle((0, 0, 1), a) for a in (0.0, 0.5, 1.0)])
        translations = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
        return anm.Track(rotations, translations, None)

    def test_fraction_of_the_frame(self):
        position, rotation = animation.sample_track(self.make_track(), 1.5)
        np.testing.assert_allclose(position, [1.0, 1.0, 0.0])
        np.testing.assert_allclose(rotation, quat.from_axis_angle((0, 0, 1), 0.75), atol=1e-9)

    def test_after_the_last_key_the_last_one_stays(self):
        position, rotation = animation.sample_track(self.make_track(), 7.25)
        np.testing.assert_allclose(position, [1.0, 2.0, 0.0])
        np.testing.assert_allclose(rotation, quat.from_axis_angle((0, 0, 1), 1.0), atol=1e-9)


class FakeAction(object):

    def __init__(self, name, length, cycle):
        self.name = name
        self.length = length
        self.is_cycle = cycle


class PlayerTest(unittest.TestCase):

    def test_cycle_loops(self):
        player = animation.AnimationPlayer()
        player.play(animation.Clip(FakeAction('walk', 10, True), {}))
        player.add_time(23.5)
        self.assertAlmostEqual(player.time, 3.5)

    def test_unique_holds_the_last_frame(self):
        ended = []
        player = animation.AnimationPlayer(on_clip_end=ended.append)
        player.play(animation.Clip(FakeAction('attack', 10, False), {}))
        player.add_time(12.0)
        player.add_time(5.0)
        self.assertEqual(player.time, 10.0)
        self.assertEqual(len(ended), 1)

    def test_queue_carries_the_rest_of_the_time(self):
        player = animation.AnimationPlayer()
        first = animation.Clip(FakeAction('start', 4, False), {})
        second = animation.Clip(FakeAction('cycle', 10, True), {})
        player.play(first)
        player.enqueue(second)
        player.add_time(6.0)
        self.assertIs(player.clip, second)
        self.assertAlmostEqual(player.time, 2.0)


class TimingTest(unittest.TestCase):

    def test_tick_and_fps(self):
        self.assertAlmostEqual(timing.ANIMATION_FPS, 18.1818, places=3)
        clock = timing.TickClock()
        self.assertEqual(clock.advance(120.0), 2)
        self.assertAlmostEqual(clock.fraction, 10.0 / 55.0)
        clock.set_fast(True)
        self.assertEqual(clock.tick_ms, 27)


class AdbKeyTest(unittest.TestCase):

    def test_attack_key_of_the_server(self):
        # sub_550100 asks for 0x90000 | weapon | modifier for an attack
        self.assertEqual(adb.make_key(state=actions.STATE_ATTACK, action_type=actions.TYPE_ATTACK), 0x90000)


class TerrainUvTest(unittest.TestCase):

    def test_rotation_turns_the_corners(self):
        per_row = 8
        corners = [terrain.tile_uv(0, r, 0.0, 0.0, per_row) for r in range(4)]
        self.assertEqual(len(set(corners)), 4)
        self.assertEqual(terrain.tile_uv(0, 0, 0.0, 1.0, per_row), (0.0, 7.0 / 8.0))


class FakeTree(object):

    def __init__(self, names):
        self.parents = {name: None for name in names}
        self.order = list(names)


def fake_item(cls, name, shape_type, number, code, layer=0):
    fields = {0: name.encode() + b'\0', 2: shape_type.to_bytes(4, 'little'), 5: number.to_bytes(4, 'little'),
              0x18: layer.to_bytes(4, 'little')}
    material = idb.Material(0, {0: b'metal\0', 2: code.encode() + b'\0'})
    return equipment.Item(name + '.metal', cls(0, fields), material)


class EquipmentTest(unittest.TestCase):

    BODY = ('hp', 'bd', 'hd', 'hr.00', 'rh1', 'rh2', 'rh3', 'lh1', 'lh2', 'lh3', 'rl1', 'rl2', 'rl3', 'll1', 'll2', 'll3')

    def test_helmet_stays_over_the_head_and_hides_the_hair(self):
        tree = FakeTree(self.BODY + ('hd.armor05', 'bd.armor05'))
        helm = fake_item(idb.ArmorShape, 'helm', idb.ARMOR_HELM, 5, 'st')
        plate = fake_item(idb.ArmorShape, 'plate', idb.ARMOR_PLATE, 5, 'st')
        shown, hidden = equipment.Equipment('unhuma', armors=[helm, plate], hair=0).parts(tree)
        self.assertIn('hd', shown)
        self.assertEqual(shown['hd.armor05'], equipment.TEXTURE_GEAR)
        self.assertNotIn('hr.00', shown)
        self.assertNotIn('bd', shown)
        self.assertEqual(shown['bd.armor05'], equipment.TEXTURE_BODY)
        # rh1.armor05 is not in the model, the bare arm stays
        self.assertIn('rh1', shown)

    def test_bow_parts_and_hidden_arrow(self):
        tree = FakeTree(self.BODY + ('lh3.bwpartb02', 'bwparta02', 'bwtetivaa02', 'bwtetivab02', 'quiver', 'arrows', 'rh3.arrow00'))
        bow = fake_item(idb.WeaponShape, 'bow', idb.WEAPON_BOW, 2, 'br')
        gear = equipment.Equipment('unorfe', weapon=bow)
        shown, hidden = gear.parts(tree)
        self.assertEqual(shown['bwparta02'], equipment.TEXTURE_GEAR)
        self.assertEqual(hidden, {'rh3.arrow00'})
        self.assertEqual(gear.weapon_name, 'BOW')
        self.assertEqual(gear.gear_layers(), ['unorfebw_02.br.0'])

    def test_layers_of_the_body(self):
        plate = fake_item(idb.ArmorShape, 'plate', idb.ARMOR_PLATE, 3, 'ir', layer=1)
        leggins = fake_item(idb.ArmorShape, 'leggins', idb.ARMOR_LEGGINS, 4, 'ir', layer=0)
        boots = fake_item(idb.ArmorShape, 'boots', idb.ARMOR_BOOTS, 1, 'le')
        gear = equipment.Equipment('unhuma', armors=[plate, leggins, boots], skin=7)
        self.assertEqual(gear.body_layers(), ['unhumaskin_07', 'unhumabt_01.le.0', 'unhumalg_04.ir.0', 'unhumapl_03.ir.0'])

    def test_compose_draws_over(self):
        bottom = np.zeros((4, 4, 4), dtype=np.uint8)
        bottom[...] = (10, 20, 30, 255)
        top = np.zeros((4, 4, 4), dtype=np.uint8)
        top[0, 0] = (200, 100, 50, 255)
        result = equipment.compose([bottom, top, np.zeros((2, 2, 4), dtype=np.uint8)], 4)
        self.assertEqual(tuple(result[0, 0]), (200, 100, 50, 255))
        self.assertEqual(tuple(result[1, 1]), (10, 20, 30, 255))


@unittest.skipIf(_game_data() is None, 'no game data')
class LightingTest(unittest.TestCase):

    def test_fill_hours(self):
        table = lighting.fill_hours({0: (0.0, 0.0, 0.0), 4: (1.0, 0.5, 0.0), 10: (0.0, 0.0, 1.0)})
        np.testing.assert_allclose(table[2], (0.5, 0.25, 0.0))
        np.testing.assert_allclose(table[7], (0.5, 0.25, 0.5))
        np.testing.assert_allclose(table[23], (0.0, 0.0, 1.0))

    def test_parse_lights(self):
        tables = lighting.parse_lights('[sunlight]\ntime00 = 255,0,0\ntime12 = 0,0,255 ; noon\n[ambient]\n')
        np.testing.assert_allclose(tables['sunlight'][6], (0.5, 0.0, 0.5))
        self.assertTrue(np.all(tables['ambient'] == 0.5))

    def test_sun(self):
        self.assertAlmostEqual(lighting.sun_elevation_degrees(12.0), 60.0)
        self.assertAlmostEqual(lighting.sun_elevation_degrees(4.0), 30.0)
        self.assertAlmostEqual(lighting.sun_elevation_degrees(0.0), 75.0)
        np.testing.assert_allclose(lighting.sun_direction(12.0), (-0.5, 0.0, -math.sqrt(3.0) / 2.0), atol=1e-9)

    def test_fog(self):
        self.assertEqual(lighting.fog_start(12.0), lighting.FOG_DAY_START)
        self.assertEqual(lighting.fog_start(0.0), lighting.FOG_NIGHT_START)
        self.assertAlmostEqual(lighting.fog_start(20.0), 70.0)
        self.assertAlmostEqual(lighting.fog_start(4.0), 70.0)

    def test_black_ambient_follows_the_sun(self):
        tables = lighting.parse_lights('[sunlight]\ntime00 = 255,255,255\n[ambient]\ntime00 = 0,0,0\n')
        state = lighting.WorldLight(tables).at(5.5)
        np.testing.assert_allclose(state.ambient_color, (0.8, 0.8, 0.8))

    def test_cave(self):
        tables = lighting.parse_lights('[sunlight]\ntime00 = 255,0,0\ntime12 = 0,0,255\n')
        state = lighting.WorldLight(tables, cave=True).at(12.0)
        np.testing.assert_allclose(state.sun_color, (1.0, 0.0, 0.0))
        self.assertEqual(state.fog_start, lighting.FOG_DAY_START)


class ZonesTest(unittest.TestCase):

    def test_parse(self):
        text = '## comment\n#zone zone3 ingos main\n#res\nzone3 zone3m\n#sky\ncave\n#zone zone4 gipat\n#res\nzone4\n'
        found = zones.parse(text.encode('cp1251'))
        self.assertEqual([z.name for z in found], ['zone3', 'zone4'])
        self.assertEqual((found[0].mpr_name, found[0].mob_name), ('zone3', 'zone3m'))
        self.assertEqual(found[0].lights_file_name(), 'lightscaveingos.ini')
        self.assertEqual(found[1].lights_file_name(), 'lightsgipat.ini')

    def test_exits(self):
        text = ('#zone gz1g gipat game\n#res\nzone1 zone1\n#maps\n8 6\nzone1map zone1quest\n'
                '## one-way entrance\n#exit 1\nnone 0\n#deploy\n49 133 59 143\n#view\n90\n'
                '#exit 2\nbz1g 1\n#deploy\n184 101 190 108\n#remove\n181 108 193 120\n#passtime\n13\n'
                '#deployangle\n270\n')
        zone = zones.parse(text.encode('cp1251'))[0]
        self.assertEqual([e.number for e in zone.exits], [1, 2])
        start = zone.exit(1)
        self.assertEqual((start.neighbour_name, start.neighbour_exit), ('none', 0))
        self.assertEqual(start.deploy_center, (54.0, 138.0))
        self.assertEqual(start.view, 90.0)
        # without #deployangle the party looks to the middle of the map
        yaw = start.deploy_yaw(256.0, 192.0)
        look = (math.sin(yaw), -math.cos(yaw))
        to_middle = (128.0 - 54.0, 96.0 - 138.0)
        self.assertAlmostEqual(look[0] * to_middle[1] - look[1] * to_middle[0], 0.0, places=6)
        self.assertGreater(look[0] * to_middle[0] + look[1] * to_middle[1], 0.0)
        base = zone.exit(2)
        self.assertEqual((base.neighbour_name, base.neighbour_exit, base.pass_time), ('bz1g', 1, 13.0))
        self.assertEqual(base.remove, (181.0, 108.0, 193.0, 120.0))
        self.assertAlmostEqual(base.deploy_yaw(256.0, 192.0), math.radians(270.0))
        self.assertIsNone(zone.exit(3))


class WaterTest(unittest.TestCase):

    def test_wind(self):
        calm = water.Wind((0.0, 1.0, 0.0), 0.0)
        self.assertEqual((calm.amplitude, calm.time_step), (0.0, 0.0))
        strong = water.Wind((3.0, 0.0, 0.0), 2.0)
        self.assertEqual(strong.strength, 1.0)
        self.assertAlmostEqual(strong.amplitude, 0.3)
        self.assertAlmostEqual(strong.step[0], math.pi / 7.0 * 0.50025016 * math.pi)
        self.assertEqual(strong.step[1], 0.0)

    def test_random_phases_repeat_on_the_edges(self):
        phases = water.random_phases(seed=5)
        self.assertEqual(phases.shape, (water.GRID, water.GRID))
        np.testing.assert_allclose(phases[0], phases[-1])
        np.testing.assert_allclose(phases[:, 0], phases[:, -1])
        self.assertTrue(np.all(phases >= 0.0))

    def test_clock(self):
        clock = water.WaveClock(water.Wind((0.0, 1.0, 0.0), 0.6))
        clock.update(3)
        self.assertAlmostEqual(clock.time_phase, -math.pi / 21.0 * 0.6 * 3)
        self.assertAlmostEqual(clock.sway_phase, 0.15)

    def test_world_settings_default(self):
        settings = world.world_settings(None)
        self.assertEqual((settings.hour, settings.wind_strength), (world.DEFAULT_HOUR, 0.0))


def _reg_bytes(sections):
    """
    A .reg file made the way game.exe reads them: section name -> [(type, name, packed value)].
    """
    body = b''
    offsets = []
    header_size = 6 + 6 * len(sections)
    for name, keys in sections:
        offsets.append(header_size + len(body))
        encoded = name.encode()
        section = struct.pack('<HH', len(keys), len(encoded)) + encoded
        table_size = 6 * len(keys)
        key_data = b''
        key_offsets = []
        for kind, key_name, value in keys:
            key_offsets.append(len(section) + table_size + len(key_data))
            encoded_key = key_name.encode()
            key_data += struct.pack('<BH', kind, len(encoded_key)) + encoded_key + value
        section += b''.join(struct.pack('<hI', -1, o) for o in key_offsets) + key_data
        body += section
    table = b''.join(struct.pack('<hI', -1, o) for o in offsets)
    return struct.pack('<IH', reg.SIGNATURE, len(sections)) + table + body


class RegTest(unittest.TestCase):

    def test_parse(self):
        data = _reg_bytes([
            ('MainMenu', [(reg.TYPE_INT | reg.ARRAY, 'NewGame', struct.pack('<H4i', 4, 367, 175, 507, 216)),
                          (reg.TYPE_STRING, 'Music', struct.pack('<H', 4) + b'main')]),
            ('Other', [(reg.TYPE_FLOAT, 'Speed', struct.pack('<f', 1.5)),
                       (reg.TYPE_STRING | reg.ARRAY, 'Combat', struct.pack('<HH', 2, 4) + b'rage' + struct.pack('<H', 3) + b'war')]),
        ])
        sections = reg.parse(data)
        self.assertEqual(sections['MainMenu']['NewGame'], (367, 175, 507, 216))
        self.assertEqual(sections['MainMenu']['Music'], 'main')
        self.assertEqual(sections['Other']['Speed'], 1.5)
        self.assertEqual(sections['Other']['Combat'], ('rage', 'war'))

    def test_signature(self):
        with self.assertRaises(reg.RegError):
            reg.parse(b'\0' * 16)


class CamTest(unittest.TestCase):

    def test_parse(self):
        data = struct.pack('<2I3f4f', 100, 7, 1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0) * 2
        keys = cam.parse(data)
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[1].position, (1.0, 2.0, 3.0))
        self.assertEqual(keys[0].rotation, (1.0, 0.0, 0.0, 0.0))

    def test_camera_view_looks_along_z(self):
        view = gl.camera_view((0.0, 0.0, 0.0), np.identity(3))
        # the +z of the camera is in front of the eye, its +y is down on the screen
        np.testing.assert_allclose(view @ (0.0, 0.0, 5.0, 1.0), (0.0, 0.0, -5.0, 1.0))
        np.testing.assert_allclose(view @ (0.0, 1.0, 0.0, 1.0), (0.0, -1.0, 0.0, 1.0))

    def test_interface_point(self):
        np.testing.assert_allclose(gl.interface_point(400.0, 300.0, 8.0), (0.0, 0.0, 8.0))
        x, y, z = gl.interface_point(800.0, 600.0, 2.0)
        self.assertAlmostEqual(x, gl.TAN_HALF_WIDTH * 2.0)
        self.assertAlmostEqual(y, gl.TAN_HALF_HEIGHT * 2.0)


class MenusTest(unittest.TestCase):

    def test_hover_plays_the_special_of_the_button(self):
        rects = {menus.NEW_GAME: (367, 175, 507, 216), menus.CREDITS: (354, 358, 473, 393)}
        column = menus.ButtonColumn(rects, menus.MAIN_MENU_BUTTONS, menus.MAIN_MENU_SPECIALS)
        self.assertEqual(column.hover(400, 200), (True, 1))
        self.assertEqual(column.hover(401, 201), (False, None))
        self.assertEqual(column.hover(400, 370), (True, 11))
        self.assertEqual(column.hover(10, 10), (True, None))
        self.assertIsNone(column.button_at(507, 200))

    def test_main_menu_hour(self):
        # game.exe 57c648: the hour of the local time plus its minutes
        self.assertEqual(menus.main_menu_hour(datetime.datetime(2026, 11, 11, 14, 30)), 14.5)
        self.assertEqual(menus.main_menu_hour(datetime.datetime(2026, 11, 11, 0, 15)), 0.25)


class HudTest(unittest.TestCase):

    def test_quad_texels_count_from_the_bottom_in_256th(self):
        element = hud.quad(10, (0, 100, 20, 130), (47, 62, 67, 92))
        vertices = element.vertices()
        self.assertEqual(len(vertices), 6)
        x, y, u, v = vertices[0]
        self.assertEqual((x, y), (0, 100))
        self.assertAlmostEqual(u, 47 / 256.0 + 0.5 / 256.0)
        self.assertAlmostEqual(v, 1.0 - 62 / 256.0 - 0.5 / 256.0)

    def test_quad_without_texels_has_no_texture(self):
        element = hud.quad(60, (20, 0, 180, 220), color=hud.SHADE)
        self.assertIsNone(element.texture_name)
        self.assertEqual(element.bounds(), (20, 0, 180, 220))

    def test_frame_is_four_trapezoids_around_the_rect(self):
        element = hud.frame(10, (-5, -5, 185, 225), 5, hud.PANEL_FRAME_TEXELS)
        self.assertEqual(len(element.triangles), 8)
        self.assertEqual(element.bounds(), (-5, -5, 185, 225))
        # the outer corners take v0, the inner ones v1
        outer = [p for t in element.triangles for p in t if p[0] in (-5, 185) and p[1] in (-5, 225)]
        self.assertTrue(outer and all(p[3] == 55 for p in outer))

    def test_disc_turns_around_its_center(self):
        flat = hud.disc(10, (800, 600), 65, 16, (174, 81), 65)
        turned = hud.disc(10, (800, 600), 65, 16, (174, 81), 65, angle=math.pi * 0.5)
        x, y, u, v = turned.triangles[0][0]
        self.assertAlmostEqual(x, 800.0)
        self.assertAlmostEqual(y, 665.0)
        self.assertEqual((u, v), flat.triangles[0][0][2:4])

    def test_dial_regions(self):
        pause = hud.pause_dial()
        self.assertEqual(hud.region_at([pause], 745, 585).tip, hud.TIP_PAUSE)
        self.assertEqual(hud.region_at([pause], 775, 535).tip, hud.TIP_FAST_SPEED)
        self.assertEqual(hud.region_at([pause], 790, 590).tip, hud.TIP_QUESTS)
        self.assertIsNone(hud.region_at([pause], 700, 500))
        mode = hud.mode_dial()
        self.assertEqual(hud.region_at([mode], 55, 590).tip, hud.TIP_CRAWL)
        self.assertEqual(hud.region_at([mode], 10, 540).tip, hud.TIP_RUN)
        self.assertEqual(hud.region_at([mode], 10, 590).tip, hud.TIP_AGGRESSIVE)

    def test_arrow_of_the_pause_dial_points_to_the_corner(self):
        arrow = [e for e in hud.pause_dial(speed_index=0).elements if e.name == 'arrow'][0]
        tip = arrow.triangles[0][0]
        self.assertAlmostEqual(tip[0], 800.0 - 73.0 * math.cos(math.pi / 12.0))
        self.assertAlmostEqual(tip[1], 600.0 - 73.0 * math.sin(math.pi / 12.0))
        base = [(p[0] + q[0]) * 0.5 for p, q in [arrow.triangles[0][1:3]]][0]
        self.assertLess(base, tip[0])

    def test_portraits_are_centered(self):
        self.assertEqual(hud.portrait_slots(1), [372])
        self.assertEqual(hud.portrait_slots(3), [316, 372, 428])
        bars = [e for e in hud.portrait(372, health=0.5).elements if e.name == 'health']
        self.assertEqual(bars[0].bounds(), (375, 587, 400, 592))

    def test_bar_shares(self):
        class Fighter(object):
            health, max_health, mana, max_mana = 15.0, 20.0, -1.0, 10.0

        class Unit(object):
            fighter = Fighter()
        self.assertEqual(hud.bar_shares(Unit()), (0.75, 0.0))
        Unit.fighter.max_mana = 0.0
        self.assertEqual(hud.bar_shares(Unit()), (0.75, 0.0))
        self.assertEqual(hud.bar_shares(object()), (1.0, 1.0))

    def test_names_of_the_portrait(self):
        self.assertEqual(hud.head_model_name('unhuma', 0.5, 1), 'infaunhumame1face')
        self.assertEqual(hud.head_model_name('unhuma', 0.1, 2), 'infaunhumath2face')
        self.assertEqual(hud.head_model_name('unhuma', 0.9, 3), 'infaunhumafa3face')
        self.assertEqual(hud.face_texture_name('unhuma', 0), 'facem00')
        self.assertEqual(hud.face_texture_name('unhufe', 3), 'facef03')

    def test_face_goes_to_the_lower_left_quarter(self):
        face = np.full((128, 128, 4), 7, dtype=np.uint8)
        texture = hud.portrait_face_texture(face)
        self.assertEqual(texture.shape, (256, 256, 4))
        self.assertTrue(np.all(texture[128:, 0:128] == 7))
        self.assertTrue(np.all(texture[0:128, :, 0:3] == 0))

    def test_minimap_disc_is_cut_by_the_map(self):
        minimap = hud.Minimap('zone1map', 8, 6)
        self.assertEqual((minimap.width, minimap.height, minimap.size), (256.0, 192.0, 256.0))
        self.assertAlmostEqual(minimap.max_zoom, 12.8)
        disc = minimap.disc((128.0, 96.0))
        x0, y0, x1, y1 = disc.bounds()
        self.assertAlmostEqual(x0, 636.0)
        self.assertAlmostEqual(x1, 764.0)
        # in the corner of the map only a quarter of the disc is left, from the center to the right and up
        corner = minimap.disc((0.0, 0.0))
        x0, y0, x1, y1 = corner.bounds()
        self.assertAlmostEqual(x0, 700.0)
        self.assertAlmostEqual(y1, 80.0)
        for x, y, u, v in corner.vertices():
            self.assertGreaterEqual(u, 0.0)
            self.assertGreaterEqual(v, 0.0)

    def test_minimap_zoom_is_limited(self):
        minimap = hud.Minimap('zone1map', 8, 6)
        minimap.zoom_by(1, 100.0)
        self.assertAlmostEqual(minimap.zoom, minimap.max_zoom)
        minimap.zoom_by(-1, 100.0)
        self.assertAlmostEqual(minimap.zoom, 1.0)

    def test_minimap_marks_far_units_are_not_shown(self):
        minimap = hud.Minimap('zone1map', 8, 6)
        marks = minimap.marks((128.0, 96.0), [(128.0, 96.0, 0.0, hud.MARK_OWN), (250.0, 96.0, 0.0, hud.MARK_ENEMY)])
        self.assertEqual(len(marks.triangles), 1)
        tip = marks.triangles[0][0]
        # a unit with the yaw 0 looks south (its models face -y), down on the screen
        self.assertAlmostEqual(tip[0], 700.0)
        self.assertAlmostEqual(tip[1], 80.0 + hud.MARK_TIP * hud.MINIMAP_UNIT_PIXELS)

    def test_info_modes_show_their_own_bars(self):
        general = hud.info_panel(hud.INFO_GENERAL, 1.0, 1.0, [1.0] * 6)
        names = [e.name for e in general.elements]
        self.assertIn('health', names)
        self.assertIn('mana', names)
        self.assertNotIn('hit location', names)
        hits = hud.info_panel(hud.INFO_HIT_LOCATIONS, 1.0, 1.0, [1.0] * 6)
        self.assertEqual([e.name for e in hits.elements].count('hit location'), 6)
        buttons = [e for e in hits.elements if e.name in hud.INFO_MODE_ACTIONS]
        self.assertEqual([b.color for b in buttons], [hud.DIM, hud.WHITE, hud.DIM, hud.DIM])

    def test_info_bar_is_cut_to_the_part_left(self):
        bar = hud.info_bar(4, (40, 190), 100, (64, 154, 164, 159), 0.25, 'health')[0]
        x0, y0, x1, y1 = bar.bounds()
        self.assertAlmostEqual(x1 - x0, 25.0)
        self.assertEqual(hud.info_bar(4, (40, 190), 100, (64, 154, 164, 159), 0.0, 'health'), [])

    def test_wounded_hit_location_turns_yellow_then_red(self):
        def first_v(part):
            panel = hud.info_panel(hud.INFO_HIT_LOCATIONS, 1.0, None, [part])
            bar = [e for e in panel.elements if e.name == 'hit location'][0]
            return bar.triangles[0][0][3]
        self.assertEqual(first_v(1.0), 126)
        self.assertEqual(first_v(0.5), 133)
        self.assertEqual(first_v(0.2), 140)

    def test_spell_icons_fill_the_slots_below_the_information(self):
        self.assertEqual(hud.spell_icon_rect(0), (28, 38, 52, 62))
        self.assertEqual(hud.spell_icon_rect(5), (28, 188, 52, 212))
        icons = hud.spell_icons(['spell0001', None, 'spell0004'])
        self.assertEqual([e.texture_name for e in icons], ['spell0001', 'spell0004'])
        self.assertEqual([e.z for e in icons], [hud.SPELL_ICON_Z, hud.SPELL_ICON_Z])
        # the gap keeps its own slot, the third spell goes to the third one
        self.assertEqual(icons[1].bounds(), (28, 98, 52, 122))
        # the whole texture goes into the quad, from its first row
        uvs = [vertex[2:] for vertex in icons[0].vertices()]
        self.assertIn((0.0, 0.0), uvs)
        self.assertIn((1.0, 1.0), uvs)
        self.assertIn((1.0, 0.0), uvs)
        self.assertIn((0.0, 1.0), uvs)

    def test_info_texts(self):
        labels = {hud.INFO_HEALTH_LABEL: 'Health:', 'string hl_skull': 'Head'}
        lines = hud.info_texts(hud.INFO_ATTRIBUTES, labels, (12, 20), None)
        self.assertEqual(lines[0][0], 'Health: 12/20')
        self.assertEqual(lines[0][1], (25, 35, 175, 50))
        labels_of_parts = hud.info_texts(hud.INFO_HIT_LOCATIONS, labels, hit_locations=[('skull', True)])
        self.assertEqual(labels_of_parts[0][0], 'Head')
        self.assertEqual(labels_of_parts[0][2], hud.INFO_BROKEN_COLOR)

    def test_compass_follows_the_camera(self):
        self.assertAlmostEqual(abs(hud.compass_angle(0.0)), math.pi)
        self.assertAlmostEqual(hud.compass_angle(90.0), -math.pi * 0.5)

#------------------------------------------------------------------------------


class InventoryTest(unittest.TestCase):

    def items(self, count):
        return [equipment.Item('item %d' % i, idb.WeaponShape.__new__(idb.WeaponShape), None) for i in range(count)]

    def test_strip_regions_and_tooltips(self):
        strip = inventory.InventoryStrip(self.items(3))
        regions = strip.regions()
        self.assertEqual(len(regions), 9)
        self.assertEqual([r.tip for r in regions[2:8]], list(range(21100, 21106)))
        self.assertEqual(regions[0].points[0], (30, 535))
        self.assertEqual(regions[8].points[2], (750, 600))

    def test_chosen_category_is_bright(self):
        strip = inventory.InventoryStrip(self.items(3))
        buttons = [e for e in strip.elements() if e.name.startswith('category')]
        self.assertEqual(len(buttons), 6)
        self.assertEqual(buttons[5].color, hud.WHITE)
        self.assertEqual(buttons[0].color, hud.DIM)
        # the mode of items moves the texels of the buttons by 168
        u = buttons[0].triangles[0][0][2]
        self.assertEqual(u, inventory.LEFT_BUTTON_U + 168)

    def test_held_arrow_scrolls_by_whole_slots(self):
        strip = inventory.InventoryStrip(self.items(12))
        self.assertEqual(strip.max_position(), 5)
        strip.press_arrow(1)
        for i in range(5):
            strip.update(0.05)
        strip.release()
        for i in range(100):
            strip.update(0.02)
        self.assertEqual(strip.position, 3.0)
        strip.press_arrow(1, jump=True)
        self.assertEqual(strip.position, 5.0)
        strip.press_arrow(-1, jump=True)
        self.assertEqual(strip.position, 0.0)

    def test_list_never_goes_past_the_last_items(self):
        strip = inventory.InventoryStrip(self.items(9))
        strip.press_arrow(1)
        for i in range(100):
            strip.update(0.05)
        self.assertEqual(strip.position, 2.0)
        self.assertEqual(len(strip.slots()), 7)

    def test_categories_filter_the_items(self):
        strip = inventory.InventoryStrip(self.items(4))
        strip.choose_category(inventory.CATEGORY_READY)
        self.assertEqual(len(strip.shown_items()), 4)
        strip.choose_category(inventory.CATEGORY_QUEST)
        self.assertEqual(strip.shown_items(), [])

    def test_names_of_the_figure_and_the_texture(self):
        axe = equipment.Item('axe', idb.WeaponShape.__new__(idb.WeaponShape), None)
        axe.shape.type, axe.shape.number, axe.shape.variant = 1, '01', 0
        self.assertEqual(inventory.item_model_name(axe), 'initweax1weapon')
        self.assertEqual(inventory.item_texture_name(axe), 'ax_01.0')
        boots = equipment.Item('boots', idb.ArmorShape.__new__(idb.ArmorShape), None)
        boots.shape.type, boots.shape.number, boots.shape.variant = idb.ARMOR_BOOTS, '00', 0
        self.assertEqual(inventory.item_model_name(boots), 'initarbt0armor')

    def test_item_texture_goes_to_the_bottom_rows(self):
        texture = inventory.item_texture(np.full((128, 128, 4), 7, dtype=np.uint8))
        self.assertEqual(texture.shape, (256, 256, 4))
        self.assertTrue(np.all(texture[128:, 0:128] == 7))
        self.assertTrue(np.all(inventory.item_texture(None) == 255))

    def test_hollow_frame_has_no_middle(self):
        element = hud.hollow_frame(4, (50, 500, 150, 600), 15, inventory.SLOT_TEXELS)
        self.assertEqual(len(element.triangles), 8)
        xs = sorted({p[0] for t in element.triangles for p in t})
        self.assertEqual(xs, [50, 65, 135, 150])

#------------------------------------------------------------------------------


class StatsTest(unittest.TestCase):

    def test_decode_types(self):
        dwords = [0] * stats.STATS_COUNT
        dwords[stats.HEALTH] = 25
        dwords[stats.TO_HIT] = struct.unpack('<I', struct.pack('<f', 12.6))[0]
        values = stats.decode(dwords)
        self.assertEqual(values[stats.HEALTH], 25)
        self.assertAlmostEqual(values[stats.TO_HIT], 12.6, places=5)
        dwords[stats.MAX_HEALTH] = 50
        self.assertEqual(stats.from_mob(dwords).attack, 13)
        self.assertAlmostEqual(stats.from_mob(dwords).health_part, 0.5)
        self.assertIsNone(stats.from_mob(dwords[:10]))

    def test_wounded_legs(self):
        wounds = stats.WoundLevels.from_registry({'RPG': {'DamageLevel1': 0.5, 'DamageLevel1Value': 0.5}})
        self.assertEqual(wounds.speed_factor(1.0), 1.0)
        self.assertEqual(wounds.speed_factor(0.5), 0.5)
        self.assertEqual(wounds.walk_modifier(1.0), 0)
        self.assertEqual(wounds.walk_modifier(0.5), 2)
        self.assertTrue(wounds.can_run(0.5))
        self.assertFalse(wounds.can_run(0.49))

#------------------------------------------------------------------------------


class ExperienceTest(unittest.TestCase):

    def test_skill_price_keeps_the_free_experience(self):
        hero = experience.HeroStats()
        self.assertEqual(hero.free_experience, 0.0)
        hero.give_skill(experience.SKILL_MELEE, 10)
        # 50 * 1.61 ^ 2 - 50
        self.assertEqual(hero.experience, 79.0)
        self.assertAlmostEqual(hero.free_experience, -0.605, places=3)
        hero.give_skill(experience.SKILL_SCIENCE, 3)
        self.assertEqual(hero.science, 3)
        self.assertEqual(hero.experience, 95.0)
        self.assertFalse(hero.give_skill(99, 1))

    def test_experience_gain(self):
        hero = experience.HeroStats()
        hero.base_intelligence = 26.0
        self.assertAlmostEqual(hero.add_experience(100.0), 104.0, places=4)
        # a loss waits until the gains cover it
        self.assertEqual(hero.add_plain_experience(-30.0), 0.0)
        self.assertEqual(hero.waiting_experience, -30.0)
        self.assertEqual(hero.add_plain_experience(20.0), 0.0)
        self.assertAlmostEqual(hero.add_plain_experience(15.0), 5.0)
        self.assertAlmostEqual(hero.experience, 109.0, places=4)
        # the health: (str / 25) * 30 * 1.1 ^ (log(exp / 50) / log(1.61)), the mana with the dexterity
        hero.experience = 50.0 * 1.61
        self.assertAlmostEqual(hero.max_health, 33.0, places=4)
        hero.give_dexterity(25)
        self.assertAlmostEqual(hero.max_mana, 66.0, places=4)

    def test_quest_share(self):
        heroes = [experience.HeroStats(), experience.HeroStats()]
        heroes[0].experience = 400.0
        heroes[1].experience = 100.0
        shares = experience.share_quest_experience(heroes, 300.0)
        # sqrt(400) : sqrt(100)
        self.assertEqual([round(v) for v in shares], [200, 100])
        self.assertEqual([h.experience for h in heroes], [600.0, 200.0])

    def test_kill_experience(self):
        class Unit(object):
            def __init__(self, in_party, hero_stats=None):
                self.fighter = combat.Fighter(stats.UnitStats(20, 20, 20, 20, 5, 5, 1, 2, 0), in_party=in_party)
                self.hero_stats = hero_stats

        hero = Unit(True, experience.HeroStats())
        hero.hero_stats.experience = 100.0
        wolf = Unit(False)
        # 553f40: a unit of the player kills a unit of nobody
        self.assertTrue(experience.is_rewarded_kill(hero, wolf))
        self.assertFalse(experience.is_rewarded_kill(wolf, hero))
        self.assertFalse(experience.is_rewarded_kill(hero, Unit(True)))
        self.assertEqual(experience.reward_units([hero, wolf], 14.0), [14.0])
        self.assertEqual(hero.hero_stats.experience, 114.0)
        self.assertEqual(experience.reward_units([wolf], 14.0), [])
        game_data = _game_data()
        if game_data is not None:
            prototypes = {p.name: p for p in game_data.units().prototypes}
            self.assertEqual(prototypes['WolfWhiteF6'].kill_experience, 14.0)
            self.assertEqual(prototypes['DragonBlueM14'].kill_experience, 300.0)

    def test_attack_and_parry_of_a_hero(self):
        class Shape(object):
            type = idb.WEAPON_SWORD
            attack = 0.0
            defense = 10.0
            skill_bonuses = (2, 0, 0, 0, 0, 0, 0, 0)

        hero = experience.HeroStats()
        hero.base_dexterity = 30.0
        hero.skills[experience.SKILL_MELEE] = 10
        fighter = combat.Fighter(stats.UnitStats(20, 20, 20, 20, 5, 5, 1, 2, 0), in_party=True)
        fighter.weapon_shape = Shape()
        fighter.set_skill(hero)
        # vf50: the dexterity 30 - 25 + the melee 10 + the bonus 2 of the weapon
        self.assertEqual((fighter.attack, fighter.parry), (17, 27))
        Shape.type = idb.WEAPON_BOW
        fighter.set_skill(hero)
        self.assertEqual((fighter.attack, fighter.parry), (5, 0))

    def test_difficulty_of_the_monsters(self):
        registry = {'DifficultyLevels': {'Attack': (1.0, 0.66), 'Defence': (1.0, 0.66), 'Absorption': (1.0, 0.33)}}
        unit_stats = stats.UnitStats(20, 20, 0, 0, 30, 20, 1, 2, 10)
        level = difficulty_module.GAME_INDEX[difficulty_module.NOVICE]
        wolf = combat.Fighter(unit_stats, difficulty=combat.Difficulty.from_registry(registry, level))
        # 52e0d0: the novice level weakens the units out of the party only
        self.assertEqual((wolf.attack, wolf.parry), (19, 13))
        hero = combat.Fighter(unit_stats, in_party=True, difficulty=combat.Difficulty.from_registry(registry, level))
        self.assertEqual((hero.attack, hero.parry), (30, 20))
        normal = combat.Difficulty.from_registry(registry, difficulty_module.GAME_INDEX[difficulty_module.NORMAL])
        self.assertEqual(combat.Fighter(unit_stats, difficulty=normal).attack, 30)

    def test_templates_of_the_heroes(self):
        game_data = _game_data()
        if game_data is None:
            self.skipTest('no game data')
        template = game_data.units().template('Human Hero')
        self.assertEqual((template.experience, template.strength, template.dexterity, template.intelligence),
                         (271.0, 23.0, 30.0, 25.0))
        self.assertEqual(template.perk_codes, ['bs1'])
        self.assertIsNone(game_data.units().template('GoblinA1'))
        perks = game_data.perks()
        self.assertEqual([s.code for s in perks.skills][:6], ['melee', 'archery', 'backstab', 'elemental', 'sense',
                                                               'astral'])
        self.assertEqual(perks.level_perk(experience.GROUP_HEALTH, 2).code, 'health2')
        hero = experience.from_prototype(game_data.units(), 'Human Hero', perks)
        # the experience of the template is the price of its skills
        self.assertEqual(hero.experience, 271.0)
        self.assertLess(abs(hero.free_experience), 1.0)
        self.assertEqual(hero.perk_levels[perks.perk('bs1').group], 1)
        self.assertAlmostEqual(hero.max_health, 38.7, places=1)
        self.assertAlmostEqual(hero.max_mana, 50.5, places=1)
        # the dexterity 30 adds 5 to the melee 10, the sense school takes the intelligence 25
        self.assertEqual(hero.weapon_skill(idb.WEAPON_SWORD), 15.0)
        self.assertEqual(hero.weapon_skill(idb.WEAPON_BOW), 5.0)
        self.assertEqual(hero.school_skill(4), 12.0)
        hero.add_perk('str2')
        self.assertEqual(hero.strength, 25.0)
        copy = hero.copy()
        copy.give_strength(1)
        self.assertEqual((hero.strength, copy.strength), (25.0, 26.0))

#------------------------------------------------------------------------------


class PathingTest(unittest.TestCase):

    def test_turns_cost_from_90_degrees(self):
        self.assertEqual(pathing.turn_cost(0, 5), 0)
        self.assertEqual(pathing.turn_cost(1, 1), 0)
        self.assertEqual(pathing.turn_cost(1, 2), 0)
        self.assertEqual(pathing.turn_cost(1, 3), pathing.TURN_COST)
        self.assertEqual(pathing.turn_cost(1, 5), 3 * pathing.TURN_COST)
        self.assertEqual(pathing.turn_cost(8, 2), pathing.TURN_COST)
        self.assertEqual(pathing.TURN_COST, 600)

    def test_direction_numbers(self):
        self.assertEqual(pathing.direction_number(-math.pi / 2), 1)
        self.assertEqual(pathing.direction_number(math.pi), 3)
        self.assertEqual(pathing.direction_number(0.0), 7)
        self.assertEqual(pathing.direction_number(math.pi / 4), 6)

    def test_push_out_to_the_nearest_open_point(self):
        grid = pathing.PassGrid.__new__(pathing.PassGrid)
        grid.width, grid.height = 10, 10
        grid.codes = np.ones((pathing.CLASSES_COUNT, 10, 10), dtype=np.uint8)
        # a statue closes the points 3..6 x 3..6
        grid.codes[:, 3:7, 3:7] = 0
        self.assertEqual(grid.nearest_open(pathing.HUMAN_CLASS, 1.0, 1.0), (1.25, 1.25))
        self.assertEqual(grid.nearest_open(pathing.HUMAN_CLASS, 1.6, 2.6), (1.25, 2.75))

        class Walker(object):
            path_grid = grid
            path_class = pathing.HUMAN_CLASS
            x, y = 2.6, 2.6
            stopped = False

            def teleport_to(self, x, y):
                self.x, self.y = x, y
                self.stopped = True

        walker = Walker()
        self.assertTrue(unit_module.push_out(walker))
        self.assertTrue(walker.stopped)
        self.assertTrue(grid.is_open(pathing.HUMAN_CLASS, *grid.point(walker.x, walker.y)))
        self.assertEqual((walker.x, walker.y), (3.75, 2.75))
        self.assertFalse(unit_module.push_out(walker))

    def test_smoothing_avoids_zigzags(self):
        grid = pathing.PassGrid.__new__(pathing.PassGrid)
        grid.width, grid.height = 12, 5
        # every step costs 1024, diagonals 1448
        tables = []
        for number, (dx, dy) in enumerate(pathing.DIRECTIONS):
            step = 1448 if number % 2 else 1024
            table = [-1] * (grid.width * grid.height)
            for y in range(grid.height):
                for x in range(grid.width):
                    if 0 <= x + dx < grid.width and 0 <= y + dy < grid.height:
                        table[y * grid.width + x] = step
            tables.append(table)
        zigzag = [(1, 2), (2, 1), (3, 2), (4, 1), (5, 2), (6, 1), (7, 2), (8, 2)]
        smooth = grid.smooth(tables, zigzag)
        self.assertEqual(smooth[0], zigzag[0])
        self.assertEqual(smooth[-1], zigzag[-1])
        self.assertEqual(len(smooth), 8)
        self.assertTrue(all(y == 2 for _, y in smooth))

    def test_points_lean_to_their_corners(self):
        z = np.array([[0.0, 4.0], [8.0, 12.0]])
        points = pathing.corner_points(z)
        self.assertEqual(points.shape, (2, 2))
        self.assertAlmostEqual(points[0, 0], 0.0 * 0.5 + (4.0 + 8.0) * 0.25)
        self.assertAlmostEqual(points[1, 1], 12.0 * 0.5 + (4.0 + 8.0) * 0.25)

    def test_slope_costs(self):
        scale = 20.0
        self.assertEqual(pathing.slope_cost(0, scale), 1024)
        self.assertGreater(pathing.slope_cost(3, scale), 1024)
        self.assertLess(pathing.slope_cost(-3, scale), 1024)
        steep = pathing.slope_limit(pathing.WALK_SLOPE_DEGREES, scale)
        self.assertEqual(pathing.slope_cost(steep, scale), -1)
        self.assertGreater(pathing.slope_cost(steep - 1, scale), 0)
        self.assertEqual(pathing.slope_cost(steep, scale, flying=True), 1024)

    def test_closed_points_spread(self):
        codes = np.full((5, 5), 14, dtype=np.uint8)
        codes[2, 2] = 0
        spread = pathing.spread_closed(codes, pathing.CROSS)
        self.assertEqual(int((spread == 0).sum()), 5)
        self.assertEqual(len(pathing.DIAMOND_2), 13)
        self.assertEqual(len(pathing.DIAMOND_3), 29)

    def test_kinds_of_parts(self):
        self.assertEqual(pathing.part_kind('trunk'), pathing.KIND_SOLID)
        self.assertEqual(pathing.part_kind('crownlea'), pathing.KIND_CROWN)
        self.assertEqual(pathing.part_kind('base0'), pathing.KIND_FLOOR)
        self.assertIsNone(pathing.part_kind('empty1'))

    def test_box_faces_cover_its_footprint(self):
        corners = np.array([(x, y, z) for x in (10.0, 12.0) for y in (20.0, 21.0) for z in (0.0, 3.0)])
        points, up = pathing.sample_box(corners)
        self.assertTrue(np.all(points[up, 2] == 3.0))
        self.assertTrue(np.all(points[:, 0] >= 10.0) and np.all(points[:, 0] <= 12.0))
        cells = {(int(x * 2), int(y * 2)) for x, y in points[:, :2]}
        # the faces at the far sides of the box fall on the next points
        self.assertTrue({(x, y) for x in range(20, 24) for y in range(40, 42)} <= cells)
        self.assertTrue(cells <= {(x, y) for x in range(20, 25) for y in range(40, 43)})

#------------------------------------------------------------------------------


class ScriptTest(unittest.TestCase):

    TEXT = """GlobalVars (
  NULL : object,
  x : object,
  F1 : float,
  Party : group
)

DeclareScript Wait#1 (  this : object )
DeclareScript Next#2 (  this : object )
DeclareScript Two#3 (  n : float,  s : string )

Script Wait#1
(
  if
  (
    IsEqual( GSGetVar( 0, "go" ) , 1 )
  )
  then
  (
    KillScript(  )
    Next#2( this )
  )
)

Script Next#2
(
  if
  (
  )
  then
  (
    KillScript(  )
    GSSetVarMax( 0, "done", 2 )
    Sleep( 2 )
    GSSetVar( 0, "slept", 1 )
  )
)

Script Two#3
(
  if
  (
    IsEqualString( s, "a" )
  )
  then
  (
    KillScript(  )
    F1 = Add( F1, n )
  )
  if
  (
  )
  then
  (
    F1 = Sub( F1, 100 )
  )
)

WorldScript
(
  Sleep( 2 )
  AddObject( Party, GetObject( 7 ) )
  AddObject( Party, GetObject( 8 ) )
  AddObject( Party, GetObject( 7 ) )
  AddObject( Party, GetObject( 99 ) )
  Wait#1( NULL )
)
"""

    class Host(script.Host):

        def __init__(self):
            self.objects = {7: 'seven', 8: 'eight'}

        def object_by_id(self, nid):
            return self.objects.get(nid)

    def world(self, text=None):
        world = script.ScriptWorld(script_lang.parse(text or self.TEXT), self.Host())
        world.start()
        return world

    def test_parse(self):
        program = script_lang.parse(self.TEXT)
        self.assertEqual(program.variables['Party'], script_lang.GROUP)
        self.assertEqual(program.declarations['Two#3'], [('n', 'float'), ('s', 'string')])
        self.assertEqual(len(program.scripts['Two#3'].branches), 2)
        command = program.world_commands[1]
        self.assertEqual(command.name, 'AddObject')
        self.assertEqual(command.function.number, 34)
        self.assertEqual(command.args[1].function.returns, script_lang.OBJECT)

    def test_parse_errors(self):
        for text in ('WorldScript ( Foo( 1 ) )', 'WorldScript ( Sleep( 1, 2 ) )', 'WorldScript ( y = 1 )',
                     'Script Nope ( )', 'WorldScript ( IsDead( 1 ) )', 'WorldScript ( Sleep( 1 )'):
            with self.assertRaises(script_lang.ScriptError):
                script_lang.parse(text)

    def test_sleep_waits_ticks(self):
        world = self.world()
        self.assertEqual(len(world.threads), 1)
        world.tick()
        self.assertEqual(world.globals['Party'], ())
        world.tick()
        # AddObject keeps every object once and no NULL
        self.assertEqual(world.globals['Party'], ('seven', 'eight'))
        self.assertEqual(len(world.instances), 1)
        world.tick()
        self.assertEqual(world.threads, [])

    def test_kill_and_call(self):
        world = self.world()
        for _ in range(4):
            world.tick()
        self.assertEqual([i.script.name for i in world.instances], ['Wait#1'])
        world.game_vars['go'] = 1.0
        world.tick()
        # called from a branch: the new instance waits for the next tick
        self.assertEqual([i.script.name for i in world.instances], ['Next#2'])
        self.assertNotIn('done', world.game_vars)
        world.tick()
        self.assertEqual(world.instances, [])
        self.assertEqual(world.game_vars['done'], 2.0)
        world.tick()
        self.assertNotIn('slept', world.game_vars)
        world.tick()
        self.assertEqual(world.game_vars['slept'], 1.0)

    def test_branches_stop_after_kill(self):
        world = self.world()
        world.call_script('Two#3', [5.0, 'a'])
        world.tick()
        self.assertEqual(world.globals['F1'], 5.0)
        self.assertEqual(world.instances, [])
        world.call_script('Two#3', [5.0, 'b'])
        world.tick()
        world.tick()
        # the second branch has no KillScript: it runs at every tick
        self.assertEqual(world.globals['F1'], -195.0)

    def test_for_every_any(self):
        text = """GlobalVars ( i : object, G : group, E : group, F1 : float, F2 : float, F3 : float )
WorldScript
(
  AddObject( G, GetObject( 7 ) )
  AddObject( G, GetObject( 8 ) )
  For( i, G )
  (
    F1 = Add( Mul( F1, 10 ) , GetObjectID( i ) )
  )
  F2 = Every( i, E, IsDead( i ) )
  F3 = Any( i, E, IsDead( i ) )
)
"""

        class Host(script.Host):

            def object_by_id(self, nid):
                return 'o%d' % nid

            def object_id(self, obj):
                return int(obj[1:])

        world = script.ScriptWorld(script_lang.parse(text), Host())
        world.start()
        # For goes from the last member
        self.assertEqual(world.globals['F1'], 87.0)
        self.assertEqual(world.globals['F2'], 1.0)
        self.assertEqual(world.globals['F3'], 0.0)

    def test_leader_and_units_of_players(self):
        text = """GlobalVars ( a : object, b : object, c : object )
WorldScript ( a = GetLeader( ) b = GetUnitOfPlayer( 3, 1 ) c = GetUnitOfPlayer( 3, 5 ) )
"""

        class Host(script.Host):

            def leader(self):
                return 'hero'

            def unit_of_player(self, player, index):
                units = {3: ['g0', 'g1']}.get(player, [])
                return units[index] if 0 <= index < len(units) else None

        world = script.ScriptWorld(script_lang.parse(text), Host())
        world.start()
        self.assertEqual((world.globals['a'], world.globals['b'], world.globals['c']), ('hero', 'g1', None))

    def test_missing_and_distance(self):
        text = """GlobalVars ( F1 : float, o : object )
WorldScript ( PlayMovie( "intro" ) F1 = DistanceUnitPoint( o, 1, 2 ) )
"""
        world = script.ScriptWorld(script_lang.parse(text))
        world.start()
        self.assertEqual(world.missing, {'PlayMovie intro': 1})
        self.assertEqual(world.globals['F1'], script.NO_DISTANCE)

    def test_script_sources(self):
        """
        The particle sources and the point lights of the scripts: by id, moved, resized, deleted; CreateFX.
        """
        text = """GlobalVars ( )
WorldScript (
    CreateParticleSource( 3, 10, 20, 5, 0.5, "PortalSTAR" )
    CreateParticleSource( -1, 0, 0, 0, -1, "PortalStar" )
    CreateParticleSource( 4, 0, 0, 0, 1, "NoSuchType" )
    MoveParticleSource( 3, 20, 20, 5, 10, 1 )
    SetParticleSourceSize( 3, 2 )
    CreatePointLight( 12, 1, 2, 3, -20, 255, 0, 51 )
    MovePointLight( 12, 1, 2, 13, 5, 1 )
    CreateFX( 1, 2, 3, 5, 50, "fx.wav" )
    CreateFXSource( -1, 456.6, 61, 25, 40, 70, "Tools\\Teleport1.wav" )
    CreateFXSource( 7, 1, 1, 1, 5, 9, "a.wav" )
    CreateFXSource( 7, 2, 2, 2, 5, 9, "b.wav" )
    CreateFXSource( -1, 0, 0, 0, 5, 9, "c.wav" )
    DeleteFXSource( -2 )
    CreateLightning( 30, 193.5, 77, 20.5, 190, 80, 1, 4 )
    CreateLightning( -1, 0, 0, 10, 0, 0, 0, 4 )
    DeleteLightning( -2 )
)
"""
        played = []

        class Host(script.Host):
            def play_fx(self, position, near, far, sound_name):
                played.append((position, near, far, sound_name))

        world = script.ScriptWorld(script_lang.parse(text), Host())
        world.start()
        sources = world.sources
        self.assertEqual(sorted(sources.particles), [-1, 3])
        self.assertEqual(world.missing, {'CreateParticleSource NoSuchType': 1})
        star = sources.particles[3]
        self.assertEqual(star.emitter.code, particles.PORTAL_STAR)
        self.assertAlmostEqual(star.emitter.scale, 2.0)
        self.assertAlmostEqual(sources.particles[-1].emitter.scale, -1.0)
        light = sources.lights[12].light
        self.assertAlmostEqual(light.radius, 20.0)
        self.assertEqual(light.color, (1.0, 0.0, 0.2))
        self.assertEqual(played, [((1.0, 2.0, 3.0), 5.0, 50.0, 'fx.wav')])
        # the sound sources play round until they are deleted; the id 7 took the second sound
        self.assertEqual(sorted(sources.sounds), [-1, 7])
        self.assertEqual(sources.sounds[7].name, 'b.wav')
        teleport = sources.sounds[-1]
        self.assertEqual((teleport.name, teleport.position, teleport.near, teleport.far),
                         ('Tools\\Teleport1.wav', (456.6, 61.0, 25.0), 40.0, 70.0))
        self.assertEqual(len(sources.sound_loops()), 2)
        for i in range(10):
            sources.update()
        self.assertAlmostEqual(star.position[0], 20.0)
        self.assertAlmostEqual(star.emitter.current[0], 20.0)
        self.assertAlmostEqual(light.position[2], 13.0)
        self.assertEqual(len(sources.emitters()), 2)
        self.assertEqual(len(sources.spell_lights()), 1)
        sources.delete_particles(3.0)
        sources.delete_light(12)
        self.assertEqual(list(sources.particles), [-1])
        self.assertEqual(sources.lights, {})
        # the lightnings take the free ids of the particles: -1 went to a particle source, -2 was deleted
        self.assertEqual(sorted(sources.lightnings), [30])
        bolt = sources.bolts()[0]
        self.assertEqual(tuple(bolt.start), (193.5, 77.0, 20.5))
        self.assertEqual(bolt.power, 4.0)
        self.assertTrue(bolt.bands())
        sources.delete_lightning(30)
        self.assertEqual(sources.bolts(), [])

    def test_portal_particles(self):
        """
        0x203e Portal: scale * 10 births a tick, the glow in the middle first, the sparks circle and rise to 8.
        """
        emitter = particles.Emitter(particles.PORTAL, (10.0, 20.0, 5.0), scale=0.75, rng=random.Random(1))
        self.assertEqual(emitter.least, 7)
        for i in range(60):
            emitter.update()
        glow = emitter.particles[0]
        self.assertEqual((glow.kind, glow.color, glow.size), (1, particles.PORTAL_GLOW_COLOR, 0.0))
        sparks = emitter.particles[1:]
        self.assertTrue(sparks)
        for p in sparks:
            self.assertLessEqual(math.hypot(p.x - 10.0, p.y - 20.0), particles.PORTAL_RADIUS + 1e-6)
            self.assertLess(p.frame, particles.PORTAL_FRAMES)
        self.assertGreater(max(p.z for p in sparks), 5.0 + particles.PORTAL_HEIGHT * 0.5)
        emitter.stop()
        emitter.update()
        self.assertNotIn(glow, emitter.particles)

    def test_lever(self):
        gate = lever.Lever(0, 3, cycled=False, coefs=(0.0, 0.5, 0.25))
        gate.switch()
        self.assertEqual(gate.state, 1)
        self.assertEqual(gate.logic_coefs(), (0.5, 0.5, 0.25))
        # the view morphs: the end is 10 + 1 ticks away, the fraction left is (end - now) / 10
        self.assertEqual(gate.view_coefs()[0], 0.0)
        gate.update(6.0)
        self.assertAlmostEqual(gate.view_coefs()[0], 0.5 - 0.5 * 0.5)
        for _ in range(6):
            gate.update(1.0)
        self.assertEqual(gate.view_coefs()[0], 0.5)
        gate.switch()
        gate.switch()
        # not cycled: it stays at the last state
        self.assertEqual(gate.state, 2)
        gate.switch(7)
        self.assertEqual(gate.state, 2)
        wheel = lever.Lever(2, 3, cycled=True)
        wheel.switch()
        self.assertEqual(wheel.state, 0)

    def test_lever_use(self):
        # 522a80 and vf41: a key lever wants its key, a science lever a level; the use forbids the lever
        key_lever = lever.Lever(0, 2, science=(8, 9, 0))
        self.assertFalse(key_lever.use((7, (), 100)))
        self.assertFalse(key_lever.use((8, (3, ), 0)))
        self.assertTrue(key_lever.use((8, (9, ), 0)))
        self.assertEqual(key_lever.state, 1)
        self.assertFalse(key_lever.usable)
        level_lever = lever.Lever(0, 2, science=(1, 0, 5))
        self.assertFalse(level_lever.allows((1, (), 4)))
        self.assertTrue(level_lever.allows((3, (), 5)))
        # the levers of (0, 0, 0) are moved by the scripts only
        self.assertFalse(lever.Lever(0, 2).usable)

    def test_lever_commands(self):
        text = """GlobalVars ( g : object, F1 : float, F2 : float )
WorldScript
(
  g = GetObject( 5 )
  F1 = GetLeverState( g )
  SwitchLeverState( g, 1 )
  F2 = GetLeverState( g )
  EnableLever( g, 0 )
  SetScience( g, 3, 2, 0, 1, 0, 1 )
)
"""
        gate = lever.Lever(0, 2)
        switched = []

        class Host(script.Host):

            def object_by_id(self, nid):
                return gate

            def lever_switched(self, obj):
                switched.append(obj)

        world = script.ScriptWorld(script_lang.parse(text), Host())
        world.start()
        self.assertEqual((world.globals['F1'], world.globals['F2']), (0.0, 1.0))
        self.assertEqual(switched, [gate])
        self.assertFalse(gate.enabled)
        # the flags 2 and 8, the key 3, the level 2
        self.assertEqual(gate.science, (10, 3, 2))
        self.assertEqual(world.missing, {})

    def test_party_of_a_save_of_the_game(self):
        """
        The hero of the party of savea of the game becomes a record of the roster: its worn items named like the maps
        name them, the scrolls apart, the numbers of its stats.
        """
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'savea', sav.SCENARIO_NAME)
        with open(path, 'rb') as f:
            party = sav.scenario_parties(sav.read_body(f.read()).body)[0]
        game_data = _game_data()
        if game_data is None:
            self.skipTest('no game data')
        record, other = roster_module.record_from_saved(party.units[0], party, game_data.items(),
                                                        game_data.quick_items())
        self.assertEqual((record.name, record.prototype_name, record.model_name), ('Hero', 'Human Hero', 'unhuma'))
        self.assertEqual(record.item_names, ['cheat dagger.bronze', 'hadagan low shirt.wool',
                                             'hadagan low pants.wool', 'hadagan low boots.wool',
                                             'hadagan low gloves.wool'])
        self.assertEqual(other, ['spell container', 'spell container'])
        stats = record.hero_stats
        self.assertEqual((stats.base_strength, stats.base_dexterity, stats.base_intelligence), (23.0, 30.0, 25.0))
        self.assertAlmostEqual(stats.experience, 271.0)
        self.assertEqual(stats.skills[:6], [10, 0, 0, 0, 12, 9])

    def test_read_a_save_of_the_game(self):
        """
        savea of game.exe: the map zone1 of gz1g at 6:00, the hero at 54, 138 (mission.sav) with its items and
        the two scrolls apart.
        """
        game_data = _game_data()
        if game_data is None:
            self.skipTest('no game data')
        dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'savea')
        game_save = saveload.read_game_save(dir_path, game_data)
        self.assertEqual((game_save.map_name, game_save.mob_name, game_save.zone), ('zone1', 'zone1', 'gz1g'))
        self.assertAlmostEqual(game_save.hour, 6.0)
        (x, y, _), yaw = game_save.hero_place
        self.assertEqual((x, y), (54.0, 138.0))
        self.assertAlmostEqual(yaw, 1.0546, places=3)
        hero = game_save.roster.default.lead()
        self.assertEqual(hero.name, 'Hero')
        self.assertEqual(len(hero.item_names), 5)
        self.assertEqual(game_save.other_item_names, ['spell container', 'spell container'])
        self.assertIsNone(saveload.read_game_save(os.path.dirname(dir_path), game_data))
        # without remake.json the save of the game is taken
        self.assertTrue(saveload.game_save_is_newer(dir_path))
        self.assertFalse(saveload.game_save_is_newer(os.path.dirname(dir_path)))

    def test_party_roster(self):
        """
        The parties of bz7g (the polymorph into JunBoy) and bz13h: made, filled, the stats and the items copied,
        the things of the parties moved by CopyLoot and SetCurrentParty.
        """
        text = """GlobalVars ( F1 : float )
WorldScript
(
  CreateParty( 0, "JunParty" )
  AddUnitToParty( 0, "JunParty::JunBoy", "Jun Male Hero" )
  CopyStats( 0, "Hero", "JunParty::JunBoy" )
  CopyLoot( 0, "", "JunParty" )
  CopyItems( 0, "Hero", "JunParty::JunBoy" )
  SetCurrentParty( 0, "JunParty" )
  RedeployParty( 0 )
  F1 = GetMercsNumber( 0 )
  AddUnitToParty( 0, "merc1", "Human Male Merc" )
  RemoveUnitFromParty( 0, "merc1" )
  RemoveUnitFromParty( 0, "nobody" )
)
"""
        roster = roster_module.new_roster('Human Hero', ['saber.bronze'], 'unhuma')
        things = party_module.Party()
        things.items = ['axe.iron']
        redeployed = []

        class Host(script.Host):

            def party(self, player):
                return things if player == 0 else None

            def roster(self, player):
                return roster if player == 0 else None

            def mercs_number(self, player):
                return 2

            def redeploy_party(self, player):
                redeployed.append(roster.current().lead())
                return True

        world = script.ScriptWorld(script_lang.parse(text), Host())
        world.start()
        self.assertEqual(world.missing, {})
        boy = roster.record('JunParty::JunBoy')
        self.assertEqual((boy.prototype_name, boy.stats_name, boy.item_names),
                         ('Jun Male Hero', 'Human Hero', ['saber.bronze']))
        self.assertEqual(roster.current_name, 'JunParty')
        self.assertEqual(redeployed, [boy])
        self.assertEqual(world.globals['F1'], 2.0)
        # the things went with the party: both have them, the current one's are the items of the player
        self.assertEqual(roster.default.loot, ['axe.iron'])
        self.assertEqual(things.items, ['axe.iron'])
        self.assertEqual([u.name for u in roster.default.units], ['Hero'])
        # back to the party by default, a name without "::" is of it, an unknown party gives None
        self.assertIsNotNone(roster.set_current(''))
        self.assertIs(roster.current(), roster.default)
        self.assertIsNone(roster.add_unit('Nowhere::X', 'Human Hero'))
        self.assertTrue(roster.copy_loot('JunParty', '', add=True))
        self.assertEqual(roster.default.loot, ['axe.iron', 'axe.iron'])

    def test_quest_items(self):
        # zone2 gives the key of the traps, zone1 completes q0g
        text = """GlobalVars ( F1 : float, F2 : float, F3 : float, F4 : float )
WorldScript
(
  GiveQuestItem( 0, "AmbushKey00" )
  F1 = HaveItem( 0, 9 )
  GiveMoney( 0, 150 )
  F2 = GetMoney( 0 )
  QuestComplete( 0, "q0g" )
  EraseQuestItem( 0, 9 )
  F3 = HaveItem( 0, 9 )
  GiveItem( 1, "AmbushKey00" )
  F4 = HaveItem( 1, 9 )
)
"""
        game_data = _game_data()
        party = party_module.Party(game_data.items(), game_data.quests())
        changes = []
        party.on_change = lambda kind, value: changes.append(kind)

        class Host(script.Host):

            def party(self, player):
                return party if player == 0 else None

        world = script.ScriptWorld(script_lang.parse(text), Host())
        world.start()
        self.assertEqual([world.globals[n] for n in ('F1', 'F2', 'F3', 'F4')], [1.0, 150.0, 0.0, 0.0])
        self.assertEqual(changes, ['quest_item', 'money', 'quest', 'quest_item_lost'])
        self.assertEqual(party.completed_quests, ['q0g'])
        self.assertEqual(world.missing, {})
        # the key of a lever is the number of a quest item
        party.give_item('ambushkey00')
        gate = lever.Lever(0, 2, science=(lever.KEY_FLAG, 9, 0))
        self.assertFalse(gate.allows((lever.LEVEL_FLAGS | lever.KEY_FLAG, (), 1000)))
        self.assertTrue(gate.allows((lever.LEVEL_FLAGS | lever.KEY_FLAG, party.keys(), 1000)))

    def test_add_mob(self):
        # the scripts of an added .mob share the globals by name and call their own scripts first
        main_text = """GlobalVars ( a1 : group, F1 : float )
DeclareScript VCheck#3#1 ( this : object )
Script VCheck#3#1
(
  if ( IsEqual( 1, 1 ) )
  then
  (
    F1 = 1
    KillScript( )
  )
)
WorldScript
(
  AddMob( "Extra.mob" )
)
"""
        extra_text = """GlobalVars ( NULL : object, F2 : float )
DeclareScript VCheck#3#1 ( this : object )
Script VCheck#3#1
(
  if ( IsEqual( 1, 1 ) )
  then
  (
    F2 = 2
    KillScript( )
  )
)
WorldScript
(
  AddObject( a1, GetObject( 7 ) )
  VCheck#3#1( NULL )
)
"""
        loaded = []

        class Host(script.Host):

            def object_by_id(self, nid):
                return 'stone %d' % nid

            def add_mob(self, name, variables):
                loaded.append(name)
                return script_lang.parse(extra_text, variables)

        world = script.ScriptWorld(script_lang.parse(main_text), Host())
        world.start()
        world.tick()
        self.assertEqual(loaded, ['Extra.mob'])
        self.assertEqual(world.globals['a1'], ('stone 7', ))
        self.assertEqual((world.globals['F1'], world.globals['F2']), (0.0, 2.0))
        self.assertEqual(world.missing, {})

    def test_set_place_of_an_object(self):
        text = """WorldScript
(
  SetCP( GetObject( 5 ), 10, 20, 1.5 )
  SetCPFast( GetObject( 5 ), 11, 21, -3 )
)
"""
        placed = []

        class Host(script.Host):

            def object_by_id(self, nid):
                return 'statue %d' % nid

            def place_object(self, obj, x, y, z, push):
                placed.append((obj, x, y, z, push))
                return True

        world = script.ScriptWorld(script_lang.parse(text), Host())
        world.start()
        # SetCP pushes the units out of the new place, SetCPFast does not
        self.assertEqual(placed, [('statue 5', 10.0, 20.0, 1.5, True), ('statue 5', 11.0, 21.0, -3.0, False)])
        self.assertEqual(world.missing, {})

    def test_remove_quest_item_of_a_unit(self):
        text = """WorldScript
(
  RemoveQuestItem( GetObject( 3 ), 11 )
)
"""

        class Item(object):
            def __init__(self, number):
                self.number = number

        class Guardian(object):
            brain = None
            quest_items = [Item(4), Item(11), Item(11)]

            def go_to(self, x, y):
                pass

        guardian = Guardian()

        class Host(script.Host):

            def object_by_id(self, nid):
                return guardian

        world = script.ScriptWorld(script_lang.parse(text), Host())
        world.start()
        # only the first item with the number goes away
        self.assertEqual([item.number for item in guardian.quest_items], [4, 11])
        self.assertEqual(world.missing, {})

    def test_unit_flags_of_the_scripts(self):
        text = """GlobalVars ( F1 : float, F2 : float, F3 : float, F4 : float )
WorldScript
(
  F1 = IsUnitBlocked( GetObject( 3 ) )
  BlockUnit( GetObject( 3 ), 1 )
  F2 = IsUnitBlocked( GetObject( 3 ) )
  F3 = WasLooted( GetObject( 3 ) )
  F4 = WasLooted( GetObject( 99 ) )
)
"""

        class Wolf(object):
            brain = None
            blocked = False
            looted = False
            is_dead = False

            def go_to(self, x, y):
                pass

        wolf = Wolf()

        class Host(script.Host):

            def object_by_id(self, nid):
                return wolf if nid == 3 else None

        world = script.ScriptWorld(script_lang.parse(text), Host())
        world.start()
        # a NULL object counts as looted
        self.assertEqual([world.globals[n] for n in ('F1', 'F2', 'F3', 'F4')], [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(world.missing, {})

    def test_skills_and_quest_experience(self):
        text = """WorldScript
(
  GiveSkill( GetObject( 3 ), "Melee", 5 )
  GiveSkill( GetObject( 3 ), "science", 2 )
  GiveSkill( GetObject( 4 ), "melee", 5 )
  GiveStrength( GetObject( 3 ), 5 )
  GiveIntelligence( GetObject( 3 ), 0 )
  QuestComplete( 0, "q0g" )
)
"""
        game_data = _game_data()
        if game_data is None:
            self.skipTest('no game data')
        party = party_module.Party(game_data.items(), game_data.quests())

        class Hero(object):
            brain = None
            is_dead = False
            fighter = None

            def go_to(self, x, y):
                pass

        hero = Hero()
        hero.hero_stats = experience.HeroStats()
        wolf = Hero()
        wolf.hero_stats = None

        class Host(script.Host):

            def object_by_id(self, nid):
                return {3: hero, 4: wolf}.get(nid)

            def party(self, player):
                return party if player == 0 else None

            def party_units(self, player):
                return [hero, wolf]

        world = script.ScriptWorld(script_lang.parse(text), Host())
        world.start()
        stats = hero.hero_stats
        self.assertEqual((stats.skills[experience.SKILL_MELEE], stats.science, stats.strength), (5, 2, 30.0))
        quest = game_data.quests()['q0g']
        self.assertGreater(quest.experience, 0.0)
        self.assertAlmostEqual(stats.experience, 30.0 + 10.0 + quest.experience, places=3)
        # a monster has no skills to give
        self.assertEqual(list(world.missing), ['GiveSkill melee'])

    def test_fix_items(self):
        text = """WorldScript
(
  FixItems(  )
)
"""
        fixed = []

        class Host(script.Host):

            def fix_items(self):
                fixed.append(True)

        world = script.ScriptWorld(script_lang.parse(text), Host())
        world.start()
        self.assertEqual(fixed, [True])
        self.assertEqual(world.missing, {})

    def test_briefing_complete(self):
        text = """GlobalVars ( F1 : float )
DeclareScript #OnBriefingComplete (  nPlayer : float,  szComplete : string )

Script #OnBriefingComplete
(
  if
  (
    IsEqualString( szComplete, "b.elder.s71_1" )
  )
  then
  (
    KillScript(  )
    F1 = GSGetVar( 0, "b.elder.s71_1" )
  )
)

WorldScript
(
  GSSetVar( 0, "b.elder.s71_1", 1 )
)
"""
        world = script.ScriptWorld(script_lang.parse(text), script.Host())
        world.start()
        self.assertEqual(world.briefing_variable('S71_1'), 'b.elder.s71_1')
        self.assertIsNone(world.briefing_variable('s72'))
        world.complete_briefing(0, 'b.elder.s71_1')
        world.tick()
        self.assertEqual(world.globals['F1'], 2.0)
        self.assertEqual(world.missing, {})

    def test_water_levels(self):
        text = """WorldScript
(
  SetWaterLevel( 5, -2.3, 100 )
  SetWaterLevel( 4, 2.3, 0 )
)
"""
        world = script.ScriptWorld(script_lang.parse(text), script.Host())
        world.start()
        levels = world.water_levels
        self.assertEqual(world.missing, {})
        # 0 ticks count as 1
        self.assertTrue(levels.tick())
        self.assertAlmostEqual(levels.offset(4), 2.3)
        self.assertAlmostEqual(levels.offset(5), -0.023)
        for _ in range(99):
            levels.tick()
        self.assertAlmostEqual(levels.offset(5), -2.3, places=5)
        self.assertFalse(levels.tick())
        # back to the level of the map from the current one; the record goes when it is at 0
        levels.set_level(4, 0.0, 2)
        levels.tick()
        self.assertAlmostEqual(levels.offset(4), 1.15)
        levels.tick()
        levels.tick()
        self.assertEqual(sorted(levels.offsets()), [5])

    def test_leave_to_zone(self):
        text = """WorldScript
(
  LeaveToZone( 0, "gz4g", 4 )
)
"""
        asked = []

        class Host(script.Host):

            def leave_to_zone(self, player, zone_name, exit_index):
                asked.append((player, zone_name, exit_index))

        world = script.ScriptWorld(script_lang.parse(text), Host())
        world.start()
        self.assertEqual(asked, [(0, 'gz4g', 4)])
        self.assertEqual(world.missing, {})
        # the index 4 is the fifth exit of the zone, an entrance "none 0" for the scripts
        zone = [z for z in _game_data().zones() if z.name == 'gz4g'][0]
        self.assertEqual((zone.exits[4].number, zone.exits[4].neighbour_name), (5, 'none'))

    def test_tutorial(self):
        text = """zt9_head\r
#screen\r
zt9_head_screen1\r
#position 20 20 200\r
#rectangle 403 102 600 499\r
#text\r
zt9_text_screen1\r
#screen\r
zt9_head_screen2\r
#position 155 34 645 22\r
#text\r
zt9_text_screen2\r
"""
        first = tutorial.parse('zt9', text)
        self.assertEqual(first.head_name, 'zt9_head')
        self.assertEqual([s.title_name for s in first.screens], ['zt9_head_screen1', 'zt9_head_screen2'])
        self.assertEqual(first.screens[0].rectangles, [(403, 102, 600, 499)])
        self.assertEqual(first.screens[1].text_name, 'zt9_text_screen2')
        # at least 300 wide; lines * 16 + 110 high, one line without the count
        self.assertEqual(first.screens[0].window_rect(), (20, 20, 320, 146))
        self.assertEqual(first.screens[1].window_rect(), (155, 34, 645, 496))
        self.assertEqual(tutorial.Screen().window_rect(), tutorial.DEFAULT_RECT)
        window = tutorial.TutorialWindow()
        second = tutorial.Tutorial('zt10', 'zt10_head', [tutorial.Screen()])
        window.add(first)
        window.add(second)
        self.assertIs(window.tutorial, first)
        self.assertFalse(window.has_back())
        window.forward()
        self.assertEqual(window.page, 1)
        self.assertFalse(window.has_next())
        self.assertTrue(window.go_to(0))
        self.assertEqual(window.page, 0)
        self.assertFalse(window.go_to(5))
        window.close()
        self.assertIs(window.tutorial, second)
        window.close()
        self.assertFalse(window.is_open)
        window.show_last()
        self.assertIs(window.tutorial, second)
        # H opens the last tutorial on the page it was closed on (658dd0 keeps the page)
        last_page = tutorial.TutorialWindow()
        last_page.add(first)
        last_page.forward()
        last_page.close()
        last_page.show_last()
        self.assertIs(last_page.tutorial, first)
        self.assertEqual(last_page.page, 1)
        self.assertFalse(last_page.has_next())

    def test_tutorials_load(self):
        game_data = _game_data()
        first = tutorial.load('zt1', game_data.text)
        self.assertEqual(len(first.screens), 3)
        self.assertEqual(game_data.text('tutor ' + first.head_name).strip(), 'Initial information')
        self.assertIsNone(tutorial.load('nothing', game_data.text))

    def test_acks(self):
        acks = _game_data().acks()
        phrase = acks['Human Hero'].phrase(idb.ACKS_SCENARIO, 'R1')
        self.assertEqual(phrase.sound_name, 'Acks\\HH\\Scenario\\169.wav')
        self.assertEqual(phrase.chance, 100)
        self.assertIsNotNone(_game_data().sound_name(phrase.sound_name))
        self.assertIsNone(acks['Human Hero'].phrase(idb.ACKS_SCENARIO, 'r1'))
        self.assertEqual(acks['Liz4'].phrase(idb.ACKS_SCENARIO, 's4').name, 's4')

    def test_maps_parse(self):
        game_data = _game_data()
        if game_data is None:
            self.skipTest('no game data')
        root = game_data.map_mob('zone1')
        text = [n.value for n in root.walk('SS_TEXT')][0]
        program = script_lang.parse(text)
        self.assertEqual(len(program.scripts), 141)
        self.assertEqual(program.world_commands[0].name, 'Sleep')

#------------------------------------------------------------------------------


class GameDataTest(unittest.TestCase):

    def setUp(self):
        self.game_data = _game_data()

    def test_attack_of_a_swordsman(self):
        database = self.game_data.animation_database('unhuma')
        action = actions.find(database, actions.STATE_ATTACK, actions.TYPE_ATTACK, modifier=14,
                              weapons=actions.weapon_bits('SWORD'), rng=random.Random(1))
        self.assertIsNotNone(action)
        self.assertEqual(action.type_name, 'ATTACK')
        self.assertIn('SWORD', action.weapon_names)

    def test_clip_has_one_key_more_than_frames(self):
        database = self.game_data.animation_database('unhuma')
        action = database.by_name('uattack04')
        tracks = self.game_data.clip('unhuma', 'uattack04')
        self.assertEqual(len(tracks['hp'].rotations), action.length + 1)

    def test_items_of_the_maps(self):
        shape, material = self.game_data.items().item('battle axe.adamantium')
        self.assertEqual((shape.type, shape.number, material.code), (idb.WEAPON_AXE, 4, 'ad'))
        gear = equipment.from_unit(self.game_data, 'unhuma', 'Human Kania Citizen F8',
                                   ['hadagan high2 helm.adamantium', 'hadagan high2 plate.adamantium'], [])
        self.assertEqual(gear.weapon.name, 'long sword.iron')
        for texture_name in gear.body_layers() + gear.gear_layers():
            self.assertIsNotNone(self.game_data.texture(texture_name), texture_name)
        shown, hidden = gear.parts(self.game_data.parts_tree('unhuma'))
        self.assertIn('hd.armor02', shown)
        self.assertIn('rh3.sword02', shown)

    def test_lights_of_the_maps(self):
        state = lighting.for_map(self.game_data, 'zone1').at(12.0)
        np.testing.assert_allclose(state.sun_color, (1.0, 1.0, 1.0))
        for map_name in self.game_data.map_names():
            settings = world.world_settings(self.game_data.map_mob(map_name))
            state = lighting.for_map(self.game_data, map_name).at(settings.hour)
            self.assertTrue(np.all(np.isfinite(state.sky_color)), map_name)
            self.assertTrue(0.0 <= settings.wind_strength <= 1.0, map_name)

    def test_water_of_zone1(self):
        land = terrain.Terrain(self.game_data, 'zone1')
        meshes = water.sector_meshes(land, 1, 1, water.Wind((1.0, 0.0, 0.0), 0.5), water.random_phases())
        tiles = sum(len(vertices) for vertices, _ in meshes.values()) // 9
        self.assertEqual(tiles, int((land.sectors[(1, 1)].water_materials != water.NO_LIQUID).sum()))

    def test_lakes_of_zone8_are_materials(self):
        """
        SetWaterLevel( 4 / 5, ... ) of zone8 moves the lakes which have their own materials of the liquid.
        """
        land = terrain.Terrain(self.game_data, 'zone8')
        used = set()
        for sector in land.sectors.values():
            if sector.has_water:
                used.update(int(v) for v in np.unique(sector.water_materials))
        self.assertTrue({4, 5} <= used)
        meshes = water.sector_meshes(land, 0, 0, water.Wind((1.0, 0.0, 0.0), 0.5), water.random_phases())
        self.assertIn(4, [material for texture, material in meshes])

    def test_tooltips_of_the_battle_screen(self):
        for panel in hud.battle_panels():
            for region in panel.regions:
                self.assertTrue(self.game_data.text(hud.TIP_PREFIX % region.tip), region)

    def test_head_of_a_portrait_is_a_loose_figure(self):
        name = hud.head_model_name('unhuma', 0.5, 1)
        self.assertTrue(self.game_data.has_model(name))
        model = figure.FigureModel(self.game_data, name)
        self.assertEqual(model.part_names, [name])
        self.assertIsNotNone(self.game_data.texture(hud.face_texture_name('unhuma', 0)))
        self.assertIsNotNone(self.game_data.texture(hud.BATTLE_TEXTURE))

    def test_minimap_of_a_zone(self):
        zone = self.game_data.zone_of_map('zone1')
        self.assertEqual(zone.sectors, (8, 6))
        self.assertEqual(zone.minimap_texture_name, 'zone1map')
        self.assertIsNotNone(self.game_data.texture(zone.minimap_texture_name))

    def test_compass_is_an_old_figure(self):
        part = self.game_data.part_figure(hud.COMPASS_MODEL, hud.COMPASS_MODEL)
        self.assertEqual(part.variants_count, 6)
        self.assertEqual(part.positions.shape, (6, 4, 3))
        self.assertEqual(len(part.triangles), 3)

    def test_hit_locations_of_a_human(self):
        units = self.game_data.units()
        unit_type = units.unit_type('Human Male')
        self.assertEqual([h.name for h in unit_type.hit_locations], ['skull', 'torso', 'arm', 'arm', 'leg', 'leg'])
        self.assertAlmostEqual(unit_type.hit_locations[1].weight, 3.0)
        hero = units.prototype('Human Hero')
        self.assertEqual((hero.health, hero.mana), (20.0, 20.0))

    def test_stats_of_zone1_units(self):
        """
        Most units of zone1.mob keep the numbers of their prototypes in UNIT_STATS.
        """
        units = self.game_data.units()
        same = total = 0
        for obj in world.map_objects(self.game_data.map_mob('zone1')):
            if not obj.is_unit or obj.stats is None:
                continue
            total += 1
            got = stats.from_mob(obj.stats)
            want = stats.from_prototype(units.prototype(obj.prototype))
            if (got.max_health, got.max_mana, got.attack, got.defense, got.damage_min, got.damage_max) == (
                    want.max_health, want.max_mana, want.attack, want.defense, want.damage_min, want.damage_max):
                same += 1
        self.assertGreater(total, 40)
        self.assertGreaterEqual(same, total - 3)

    def test_speeds_of_zone1_units(self):
        """
        UNIT_STATS 6..9 keep the speeds of the unit type of units.udb (tag 7), 4 is the multiplier.
        """
        units = self.game_data.units()
        types = {t.model_name: t for t in reversed(units.types)}
        checked = 0
        for obj in world.map_objects(self.game_data.map_mob('zone1')):
            if not obj.is_unit or obj.stats is None or obj.template not in ('unmoog1', 'unmogo'):
                continue
            got = stats.from_mob(obj.stats)
            for a, b in zip(got.speeds, types[obj.template].speeds):
                self.assertAlmostEqual(a, b, places=5)
            self.assertAlmostEqual(got.speed_scale, 1.0)
            checked += 1
        self.assertGreater(checked, 3)
        human = units.unit_type('Human Male')
        self.assertAlmostEqual(human.speeds[stats.SPEED_RUN], 0.467, places=3)

    def test_unit_moves_by_its_speeds(self):
        model = figure.FigureModel(self.game_data, 'unhuma')
        clock = timing.TickClock()
        runner = unit_module.Unit(model, rng=random.Random(1))
        runner.set_stats(stats.UnitStats(0, 0, 0, 0, 0, 0, 0, 0, 0, speed_scale=1.25, speeds=(0.4, 0.2, 0.1, 0.05)))
        runner.move_to(0, -100, run=True)
        self.assertAlmostEqual(runner.speed, 0.5)
        cycle = runner.player.queue[0].action
        self.assertAlmostEqual(runner.rate, 0.5 / (cycle.movement_speed * model.scale / unit_module.CLIP_SPEED_DIVISOR))
        runner.update(timing.TICK_MS, clock)
        self.assertAlmostEqual(runner.y, -0.5)

    def _human_fighter(self, health=20, attack=5, defense=5, armor=0, damage=(1, 2), in_party=False,
                       difficulty=None):
        human = self.game_data.units().unit_type('Human Male')
        unit_stats = stats.UnitStats(health, health, 0, 0, attack, defense, damage[0], damage[1], armor)
        return combat.Fighter(unit_stats, human, in_party=in_party, difficulty=difficulty)

    def test_health_comes_from_the_parts_of_the_body(self):
        """
        525a00 splits the health by the weights, 5259b0 takes the losses of the parts away from the maximum.
        """
        target = self._human_fighter()
        self.assertEqual([p.name for p in target.parts], ['skull', 'torso', 'arm', 'arm', 'leg', 'leg'])
        self.assertEqual([round(p.max_health, 3) for p in target.parts], [2.0, 6.0, 2.0, 2.0, 4.0, 4.0])
        self.assertAlmostEqual(target.health, 20.0)
        # human damage is all crush (type 2): 3 on the torso takes half of it, times its loss 1.01
        dealt = combat.apply_hit(target, combat.Hit(target.damage_types, 3.0))
        self.assertAlmostEqual(dealt, 3.0)
        self.assertAlmostEqual(target.health, 20.0 - 0.5 * 1.01 * 20.0, places=4)
        self.assertFalse(target.is_dead)
        # a broken torso (health below 0) costs 1.01 of the maximum
        combat.apply_hit(target, combat.Hit(target.damage_types, 3.5))
        self.assertEqual(target.parts[combat.TORSO].state, combat.PART_BROKEN)
        self.assertTrue(target.is_dead)

    def test_part_is_cut_off_by_its_damage_types(self):
        """
        525070: a blow not smaller than the part cuts it off when the flags (tag 1) have the bit of its type.
        """
        arm = 2
        slash = (1.0, ) + (0.0, ) * 6
        target = self._human_fighter()
        self.assertTrue(target.parts[arm].flags & 1)
        combat.apply_hit(target, combat.Hit(slash, 2.0, location=arm))
        self.assertEqual(target.parts[arm].state, combat.PART_LOST)
        self.assertAlmostEqual(target.parts[arm].health, 2.0 * combat.LOST_HEALTH)
        self.assertAlmostEqual(target.part_health('arm'), combat.LOST_HEALTH)
        self.assertAlmostEqual(combat.apply_hit(target, combat.Hit(slash, 5.0, location=arm)), 0.0)
        # the skull has no flags: it is only broken
        combat.apply_hit(target, combat.Hit(slash, 2.0, location=combat.HEAD))
        self.assertEqual(target.parts[combat.HEAD].state, combat.PART_BROKEN)

    def test_absorption_and_difficulty(self):
        """
        The armor takes the same amount off every type which the blow has, [DifficultyLevels] of ai.reg
        scales it, the attack and the parry of the units out of the party.
        """
        registry = self.game_data.registry('config/ai.reg')
        easy = combat.Difficulty.from_registry(registry, level=1)
        self.assertAlmostEqual(easy.value(easy.absorption), 0.33, places=2)
        mob = self._human_fighter(attack=10, defense=10, armor=3, difficulty=easy)
        self.assertEqual((mob.attack, mob.parry), (6, 6))
        self.assertAlmostEqual(combat.apply_hit(mob, combat.Hit(mob.damage_types, 5.0)), 5.0 - 3 * 0.33, places=2)
        hero = self._human_fighter(attack=10, defense=10, armor=3, in_party=True, difficulty=easy)
        self.assertEqual((hero.attack, hero.parry), (10, 10))
        self.assertAlmostEqual(combat.apply_hit(hero, combat.Hit(hero.damage_types, 5.0)), 2.0)
        self.assertAlmostEqual(combat.apply_hit(hero, combat.Hit(hero.damage_types, 2.5)), 0.0)

    def test_attack_against_parry(self):
        attacker = self._human_fighter(attack=12)
        target = self._human_fighter(defense=10)
        target.radius = 0.5
        hit = combat.Hit(attacker.damage_types, 1.0)
        # 12 - 3 / (10 * 0.5) + 1 / (20 * 1)
        self.assertAlmostEqual(combat.attack_roll(attacker, target, 3.0, hit), 11.45)
        self.assertTrue(combat.hits(attacker, target, 3.0, hit))
        aimed = combat.Hit(attacker.damage_types, 1.0, location=combat.HEAD, aimed=True)
        self.assertAlmostEqual(combat.attack_roll(attacker, target, 3.0, aimed), 0.0)
        self.assertFalse(combat.hits(attacker, target, 3.0, aimed))
        weak = self._human_fighter(attack=1)
        self.assertFalse(combat.hits(weak, target, 1.0, hit))
        backstab = combat.make_hit(weak, random.Random(1), backstab=True)
        self.assertAlmostEqual(backstab.backstab, combat.BACKSTAB_ADD)
        self.assertTrue(combat.hits(weak, target, 1.0, backstab))

    def test_backstab_from_behind(self):
        # with the yaw 0 the back of a model looks at +y
        self.assertTrue(combat.is_backstab((0.0, 2.0), (0.0, 0.0), 0.0))
        self.assertFalse(combat.is_backstab((0.0, -2.0), (0.0, 0.0), 0.0))
        self.assertFalse(combat.is_backstab((2.0, 0.0), (0.0, 0.0), 0.0))
        self.assertTrue(combat.is_backstab((-2.0, 0.0), (0.0, 0.0), math.pi / 2))

    def test_damage_roll_steps(self):
        fighter = self._human_fighter(damage=(2, 3))
        rolls = [combat.damage_roll(fighter, random.Random(i)) for i in range(200)]
        self.assertGreaterEqual(min(rolls), 2.0)
        self.assertLessEqual(max(rolls), 3.0)
        for roll in rolls:
            self.assertAlmostEqual(roll * 100, round(roll * 100))
        self.assertAlmostEqual(combat.damage_roll(self._human_fighter(damage=(0, 0))), combat.MIN_DAMAGE)

    def test_spells_of_the_database(self):
        """
        684510 reads spells.sdb: 42 prototypes (683b40), 38 modifiers (683ab0), the templates (683d00).
        """
        spells = self.game_data.spells()
        self.assertEqual((len(spells.spells), len(spells.modifiers)), (42, 38))
        self.assertEqual((len(spells.templates), len(spells.armor_templates), len(spells.weapon_templates)),
                         (59, 21, 19))
        arrow = spells.spell('Fire Arrow')
        self.assertIs(spells.spell('arrow'), arrow)
        self.assertEqual((arrow.index, arrow.subtype, arrow.subtype_id, arrow.target), (0, 'fire', 0, 'u'))
        self.assertEqual((arrow.mana, arrow.range, arrow.effect, arrow.duration, arrow.complex),
                         (10.0, 15.0, 7.0, 1, 7))
        self.assertEqual(arrow.modifier_flags, (True, True, False, True, False))
        self.assertEqual(spells.spell('Stun').special_modifiers, ['e0', 'e1', 'e2'])
        extra = spells.modifier('a1')
        self.assertEqual((extra.name, extra.kind, extra.value, extra.mana, extra.level),
                         ('Extra Area', idb.MODIFIER_AREA, 1.0, 15.0, 9))
        self.assertEqual(spells.modifier('im1').name, 'Item: Extra Mana')
        self.assertEqual(spells.weapon_templates[0].modifiers, ['it', 'fe'])

    def test_modifiers_of_a_spell(self):
        """
        68bf20: the modifiers add their value times the prototype, their mana, price and level;
        the area becomes a radius, the mana is not below the prototype.
        """
        spells = self.game_data.spells()
        fireball = magic.make_spell(spells, 'Fireball', ['a1', 'e1', 'fe'])
        self.assertEqual([m.code for m in fireball.modifiers], ['a1', 'e1', 'fe'])
        self.assertAlmostEqual(fireball.radius, math.sqrt(50.0 / math.pi))
        self.assertAlmostEqual(fireball.effect, 14.0)
        self.assertAlmostEqual(fireball.mana, 15.0 + 15.0 + 30.0 + 10.0)
        self.assertEqual((fireball.complexity, fireball.slots), (9 + 9 + 10 + 10, 2 + 3))
        self.assertEqual(fireball.flags, magic.TARGET_POINT | magic.FILTER_ENEMIES)
        # the fire arrow takes no area and no duration; the extra mana takes back the mana of other modifiers
        arrow = magic.make_spell(spells, 'Fire Arrow', ['a1', 'm1', 'r2', 'd2'])
        self.assertEqual([m.code for m in arrow.modifiers], ['m1', 'r2'])
        self.assertAlmostEqual(arrow.range, 45.0)
        self.assertAlmostEqual(arrow.mana, 10.0 - 20.0 + 25.0)
        # but not below the mana of the prototype
        self.assertAlmostEqual(magic.make_spell(spells, 'Fire Arrow', ['m2']).mana, 10.0)
        wall = magic.make_spell(spells, 'Wall of Fire', ['d1'])
        self.assertEqual(wall.duration, 300)
        self.assertAlmostEqual(wall.radius, 3.0 / math.sqrt(math.pi))

    def test_spell_hits_the_whole_body(self):
        """
        683910 and 5254b0: a spell always hits, its damage is spread over the parts of the body so that
        the health falls by the effect less the absorption.
        """
        spells = self.game_data.spells()
        arrow = magic.make_spell(spells, 'Fire Arrow')
        target = self._human_fighter(defense=1000)
        hit = arrow.hit()
        self.assertEqual((hit.location, hit.parts[magic.FIRE], sum(hit.parts)), (combat.ALL_PARTS, 1.0, 1.0))
        self.assertTrue(combat.hits(self._human_fighter(attack=0), target, 10.0, hit))
        # every part takes its max health / (the sum of the losses * 20) of the damage
        losses = sum(p.loss for p in target.parts)
        self.assertAlmostEqual(combat.apply_hit(target, hit), 7.0 * 20.0 / (losses * 20.0), places=4)
        self.assertAlmostEqual(target.health, 20.0 - 7.0, places=4)
        armored = self._human_fighter(armor=2)
        combat.apply_hit(armored, magic.make_spell(spells, 'Lightning Arc').hit())
        self.assertAlmostEqual(armored.health, 20.0 - (8.0 - 2.0), places=4)
        # a lasting spell gives 4 / duration of its effect a time
        wall = magic.make_spell(spells, 'Wall of Fire').hit()
        self.assertTrue(wall.lasting)
        self.assertAlmostEqual(wall.roll, 10.0 * 4.0 / 150.0)
        healing = magic.make_spell(spells, 'Healing')
        self.assertFalse(healing.hurts())
        self.assertAlmostEqual(combat.apply_hit(self._human_fighter(), healing.hit()), 0.0)

    def test_healing_and_spells_on_the_body(self):
        """
        525800 heals the hurt parts; 524560 counts the protections, the antimagic and the strengthen;
        the records go away when their ticks are over (515820).
        """
        spells = self.game_data.spells()
        target = self._human_fighter()
        combat.apply_hit(target, combat.Hit(target.damage_types, 3.0))
        self.assertAlmostEqual(target.health, 20.0 - 0.5 * 1.01 * 20.0, places=4)
        healing = magic.make_spell(spells, 'Healing')
        self.assertTrue(healing.heals())
        unit = unit_module.Unit(figure.FigureModel(self.game_data, 'unhuma'), rng=random.Random(1))
        unit.fighter = target
        unit.take_spell(healing, None)
        # the only hurt part is the torso: it takes all of the 20
        self.assertAlmostEqual(target.parts[combat.TORSO].health, target.parts[combat.TORSO].max_health)
        self.assertAlmostEqual(target.health, 20.0, places=4)
        # the protection from fire and the antimagic: the larger one counts
        unit.take_spell(magic.make_spell(spells, 'Protection from Fire'), None)
        self.assertAlmostEqual(target.absorption[magic.FIRE], 3.0)
        self.assertAlmostEqual(target.absorption[magic.ACID], 0.0)
        unit.take_spell(magic.make_spell(spells, 'Antimagic'), None)
        self.assertEqual([round(target.absorption[t], 3) for t in (magic.FIRE, magic.ACID, magic.LIGHTNING)],
                         [10.0, 10.0, 10.0])
        # the strengthen: the maximum health and the damage
        unit.take_spell(magic.make_spell(spells, 'Strengthen'), None)
        self.assertAlmostEqual(target.max_health, 30.0)
        self.assertAlmostEqual(target.parts[combat.TORSO].max_health, 9.0)
        base = combat.damage_roll(self._human_fighter(), rng=random.Random(5))
        self.assertAlmostEqual(combat.damage_roll(target, rng=random.Random(5)), base * 1.3)
        # the protection (150 ticks) goes away first, the strengthen (300) later
        self.assertTrue(target.tick_enchantments(150))
        self.assertAlmostEqual(target.absorption[magic.FIRE], 0.0)
        self.assertAlmostEqual(target.max_health, 30.0)
        target.tick_enchantments(150)
        self.assertEqual(target.enchantments, [])
        self.assertAlmostEqual(target.max_health, 20.0)
        self.assertAlmostEqual(target.parts[combat.TORSO].health, 6.0)

    def test_regeneration(self):
        """
        523740: every 15th tick a human gets back 0.005 of its maximum health and, while it stands, 0.1 of its
        maximum mana; the perks add 0.01 of their power, the regeneration spell multiplies the health.
        """
        target = self._human_fighter()
        self.assertAlmostEqual(target.health_regen, 0.005)
        self.assertAlmostEqual(target.mana_regen, 0.1)
        target.max_mana = 20.0
        target.mana = 0.0
        combat.apply_hit(target, combat.Hit(target.damage_types, 3.0))
        hurt = target.health
        self.assertFalse(target.regenerate(14))
        self.assertTrue(target.regenerate(1))
        self.assertAlmostEqual(target.health, hurt + 0.1, places=4)
        self.assertAlmostEqual(target.mana, 2.0, places=4)
        # an order stops the mana, not the health
        target.regenerate(15, idle=False)
        self.assertAlmostEqual(target.mana, 2.0, places=4)
        self.assertAlmostEqual(target.health, hurt + 0.2, places=4)
        target.regenerate(15, vitality=100.0, spirit=50.0)
        self.assertAlmostEqual(target.mana, 5.0, places=4)
        self.assertAlmostEqual(target.health, hurt + 0.4, places=4)
        target.enchant(combat.REGENERATION_SPELL, 150, 3.0)
        target.regenerate(15)
        self.assertAlmostEqual(target.health, hurt + 0.7, places=4)

    def test_spells_change_the_senses_and_freeze(self):
        """
        524560: the eagle sight and the invisibility change the sight of the AI; the stun freezes the unit (515820).
        """
        spells = self.game_data.spells()
        clock = timing.TickClock()
        controller = ai.Controller(ai.Diplomacy(), rng=random.Random(1))
        hero = self._ai_unit(0.0)
        controller.add_player_unit(hero)
        boar = self._ai_unit(7.0)
        brain = controller.add(boar, 3)
        self.assertTrue(brain.sees(hero))
        hero.take_spell(magic.make_spell(spells, 'Invisibility'), None)
        self.assertAlmostEqual(hero.detections[ai.SIGHT], 0.5)
        self.assertFalse(brain.sees(hero))
        boar.take_spell(magic.make_spell(spells, 'Eagle Sight'), None)
        self.assertAlmostEqual(boar.senses[ai.SIGHT], 9.5)
        self.assertFalse(brain.sees(hero))
        # the stun provokes the boar and freezes it for 5 ticks
        stun = magic.make_spell(spells, 'Stun')
        self.assertTrue(stun.provokes())
        boar.take_spell(stun, hero)
        self.assertEqual(brain.target, hero)
        self.assertTrue(boar.is_frozen)
        self.assertFalse(boar.attack(hero))
        x = boar.x
        for _ in range(4):
            boar.update(timing.TICK_MS, clock)
        self.assertTrue(boar.is_frozen)
        self.assertEqual(boar.x, x)
        boar.update(timing.TICK_MS, clock)
        self.assertFalse(boar.is_frozen)
        self.assertTrue(boar.attack(hero))
        # the filter of the friends keeps the healing away from an enemy
        healing = magic.make_spell(spells, 'Healing', ['ff'])
        self.assertTrue(controller.hostile(hero, boar))
        self.assertFalse(healing.affects(boar.figure.model_name, controller.hostile(hero, boar)))

    def test_filters_and_casting_of_a_spell(self):
        """
        68b480 skips the friends, the enemies or the races; 554f50 and 54ee10 check the skill and spend the mana.
        """
        spells = self.game_data.spells()
        plain = magic.make_spell(spells, 'Fire Arrow')
        self.assertTrue(plain.affects('unhuma', False))
        enemies = magic.make_spell(spells, 'Fire Arrow', ['fe', 'fh'])
        self.assertTrue(enemies.affects('unorma', True))
        self.assertFalse(enemies.affects('unorma', False))
        self.assertFalse(enemies.affects('unhufe', True))
        friends = magic.make_spell(spells, 'Healing', ['ff'])
        self.assertTrue(friends.affects('unhuma', False))
        self.assertFalse(friends.affects('unhuma', True))
        self.assertAlmostEqual(plain.mana_cost(), 10.0)
        self.assertAlmostEqual(plain.mana_cost(50.0), 5.0)
        self.assertTrue(plain.can_cast(6.6))
        self.assertFalse(plain.can_cast(6.4))
        self.assertEqual((plain.school_group, friends.school_group), (0, 2))

    def test_numbers_of_the_items(self):
        """
        6a5d20: the weight by the weight factor of the material, the damage by its damage factor,
        the absorption of an armor by its 7 factors and by the scale of the shape.
        """
        items = self.game_data.items()
        sword = equipment.make_item(items, 'long sword.steel')
        self.assertAlmostEqual(sword.weight, 3.0 * 5.0)
        self.assertAlmostEqual(sword.damage[0], 1.75 * 5.3, places=2)
        self.assertAlmostEqual(sword.damage[1], 3.55 * 5.3, places=2)
        self.assertEqual(sword.damage_types, (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        plate = equipment.make_item(items, 'gipat low plate.thick')
        self.assertAlmostEqual(plate.absorption()[0], 1.33 * 3.061, places=3)
        self.assertAlmostEqual(plate.absorption()[1], 1.33 * 3.826, places=3)
        # the plate has no scale for the arms
        self.assertEqual(plate.absorption(limb=True), (0.0, ) * 7)

    def test_items_of_a_hero(self):
        """
        A unit of the player takes the damage of its weapon item and wears the armor in the layers of the parts.
        """
        items = self.game_data.items()
        names = ('gipat low helm.thick', 'gipat medium shirt.thick', 'gipat high boots.thick', 'long sword.steel')
        gear = equipment.Equipment('unhuma', armors=[equipment.make_item(items, n) for n in names[:3]],
                                   weapon=equipment.make_item(items, names[3]))
        hero = self._human_fighter(in_party=True)
        hero.equip(gear)
        self.assertTrue(hero.item_damage)
        self.assertAlmostEqual(hero.weapon_weight, 15.0)
        self.assertAlmostEqual(hero.damage_min, 1.75 * 5.3, places=2)
        self.assertIsNotNone(hero.parts[combat.HEAD].layers[combat.OUTER_LAYER])
        self.assertIsNone(hero.parts[combat.HEAD].layers[combat.INNER_LAYER])
        # the shirt covers the torso by its first absorption, the arms by the second one (0 for this shirt)
        torso = hero.parts[combat.TORSO].layers[combat.INNER_LAYER]
        self.assertAlmostEqual(torso[2], 0.7 * 3.826, places=3)
        self.assertEqual(hero.parts[combat.LEFT_ARM].layers[combat.INNER_LAYER], [0.0] * 7)
        self.assertAlmostEqual(hero.parts[combat.LEFT_LEG].layers[combat.INNER_LAYER][2], 0.77 * 3.826, places=3)
        # the shirt takes its absorption off a blow on the torso
        crush = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(combat.apply_hit(hero, combat.Hit(crush, 5.0)), 5.0 - 0.7 * 3.826, places=3)
        self.assertAlmostEqual(combat.apply_hit(hero, combat.Hit(crush, 2.0)), 0.0)
        # the roll of an item has no steps
        rolls = [combat.damage_roll(hero, random.Random(i)) for i in range(50)]
        self.assertTrue(any(abs(r * 100 - round(r * 100)) > 1e-6 for r in rolls))
        self.assertTrue(all(hero.damage_min <= r <= hero.damage_min + hero.damage_spread for r in rolls))
        # the units out of the party keep their own numbers and wear no armor
        mob = self._human_fighter()
        mob.equip(gear)
        self.assertFalse(mob.item_damage)
        self.assertIsNone(mob.parts[combat.HEAD].layers[combat.OUTER_LAYER])
        self.assertEqual(mob.parry, 5)
        # a bow leaves no parry
        archer = self._human_fighter()
        archer.equip(equipment.Equipment('unhuma', weapon=equipment.make_item(items, 'short bow.steel')))
        self.assertEqual(archer.parry, 0)

    def test_wear_of_the_items(self):
        """
        6a5d20: the durability of the shape times that of the material. 525070 and 525720: the items of a layer
        wear by the damage which goes through it, the weapon by what the layers stop; 5253e0 tells a unit of
        the player that the item is worn out and broken, a broken item protects until the armor is built again.
        """
        items = self.game_data.items()
        sword = equipment.make_item(items, 'long sword.iron')
        self.assertAlmostEqual(sword.max_durability, 5556.0 * 4.9, places=1)
        self.assertEqual(sword.condition, 1.0)
        shirt = equipment.make_item(items, 'gipat medium shirt.thick')
        plate = equipment.make_item(items, 'gipat low plate.thick')
        hero = self._human_fighter(health=1000, in_party=True)
        hero.equip(equipment.Equipment('unhuma', armors=[shirt, plate]))
        attacker = self._human_fighter(in_party=True)
        attacker.equip(equipment.Equipment('unhuma', weapon=sword))
        self.assertEqual(combat.armor_kinds(combat.TORSO, combat.OUTER_LAYER), (1, ))
        self.assertEqual(combat.armor_kinds(combat.LEFT_ARM, combat.INNER_LAYER), (3, 6))
        self.assertEqual(combat.armor_kinds(combat.TORSO, 0), ())
        crush = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        hit = combat.Hit(crush, 20.0)
        hit.weapon, hit.wielder = sword, attacker
        outer = plate.absorption()[2]
        inner = shirt.absorption()[2]
        combat.apply_hit(hero, hit)
        # the plate wears by what goes through it, the shirt by what goes through both
        self.assertAlmostEqual(plate.max_durability - plate.durability, 20.0 - outer, places=3)
        self.assertAlmostEqual(shirt.max_durability - shirt.durability, 20.0 - outer - inner, places=3)
        self.assertAlmostEqual(sword.max_durability - sword.durability, outer + inner, places=3)
        self.assertEqual(hero.wear_messages, [])
        # below the critical share the item is worn out, at 0 it is broken but protects until the rebuild
        plate.durability = plate.max_durability * 0.3 + 0.5
        combat.apply_hit(hero, combat.Hit(crush, 20.0))
        self.assertEqual(hero.wear_messages, [(plate, combat.MESSAGE_ARMOR_CRITICAL)])
        plate.durability = 1.0
        combat.apply_hit(hero, combat.Hit(crush, 20.0))
        self.assertEqual(hero.wear_messages[-1], (plate, combat.MESSAGE_ARMOR_BROKEN))
        self.assertTrue(plate.is_broken)
        self.assertIsNotNone(hero.parts[combat.TORSO].layers[combat.OUTER_LAYER])
        hero.rebuild_armor()
        self.assertIsNone(hero.parts[combat.TORSO].layers[combat.OUTER_LAYER])
        # the weapon of a monster wears too, but only a unit of the player hears about it
        sword.durability = 1.0
        mob = self._human_fighter()
        mob.equip(equipment.Equipment('unhuma', weapon=sword))
        hit = combat.make_hit(mob, random.Random(1))
        self.assertIs(hit.weapon, sword)
        combat.wear_item(sword, 5.0, mob, combat.MESSAGE_WEAPON_BROKEN, combat.MESSAGE_WEAPON_CRITICAL)
        self.assertEqual(mob.wear_messages, [])
        self.assertLess(sword.durability, 0.0)

    def test_wear_goes_over_a_new_equipment(self):
        """
        The items of the same names keep their durability when the equipment of a hero is built again.
        """
        items = self.game_data.items()
        first = equipment.Equipment('unhuma', armors=[equipment.make_item(items, 'gipat low helm.thick')],
                                    weapon=equipment.make_item(items, 'long sword.iron'))
        worn = first.keep_wear({})
        first.weapon.durability = 100.0
        second = equipment.Equipment('unhuma', armors=[equipment.make_item(items, 'gipat low helm.thick')],
                                     weapon=equipment.make_item(items, 'long sword.iron'))
        worn = second.keep_wear(worn)
        self.assertEqual(second.weapon.durability, 100.0)
        self.assertEqual(second.armors[0].condition, 1.0)
        self.assertEqual(sorted(worn), ['gipat low helm.thick', 'long sword.iron'])

    def test_weight_of_the_weapon_in_the_attack(self):
        attacker = self._human_fighter(attack=12)
        target = self._human_fighter(defense=10)
        target.radius = 0.5
        attacker.weapon_weight = 15.0
        hit = combat.Hit(attacker.damage_types, 1.0)
        # 12 - 3 / (10 * 0.5) + 15 / (20 * 1)
        self.assertAlmostEqual(combat.attack_roll(attacker, target, 3.0, hit), 12.15)

    def test_units_fight_to_death(self):
        clock = timing.TickClock()
        human = self.game_data.units().unit_type('Human Male')
        fighters = []
        for x, attack in ((0.0, 40), (2.0, 0)):
            u = unit_module.Unit(figure.FigureModel(self.game_data, 'unhuma'), position=(x, 0.0),
                                 rng=random.Random(2))
            unit_stats = stats.UnitStats(20, 20, 0, 0, attack, 0, 5, 6, 0)
            u.set_stats(unit_stats)
            u.fighter = combat.Fighter(unit_stats, human, radius=combat.unit_radius(u.half_size()))
            fighters.append(u)
        a, b = fighters
        blows = []
        a.on_blow = lambda attacker, target, hit, dealt: blows.append(dealt)
        kills = []
        a.on_kill = lambda killer, victim: kills.append((killer, victim))
        self.assertTrue(a.attack(b))
        for _ in range(400):
            a.update(timing.TICK_MS, clock)
            b.update(timing.TICK_MS, clock)
            if b.is_dead:
                break
        self.assertTrue(b.is_dead)
        self.assertEqual(b.mode, unit_module.MODE_DEAD)
        self.assertTrue(blows)
        self.assertTrue(all(dealt for dealt in blows))
        self.assertFalse(a.is_dead)
        # only the killing blow reports the kill
        self.assertEqual(kills, [(a, b)])

    def _ai_unit(self, x, y=0.0, sight=8.5, attack=5, health=20):
        human = self.game_data.units().unit_type('Human Male')
        u = unit_module.Unit(figure.FigureModel(self.game_data, 'unhuma'), position=(x, y), rng=random.Random(2))
        unit_stats = stats.UnitStats(health, health, 0, 0, attack, 0, 1, 2, 0, speeds=(0.8, 0.4, 0.2, 0.1),
                                     senses=(sight, 0, 0, 0, 0, 0))
        u.set_stats(unit_stats)
        u.fighter = combat.Fighter(unit_stats, human, radius=combat.unit_radius(u.half_size()))
        return u

    def test_inflict_damage(self):
        """
        InflictDamage: a blow of the lightning to the torso which always hits; enough of it kills.
        """
        u = self._ai_unit(0.0, health=20)
        world = script.ScriptWorld(script_lang.parse("GlobalVars ( )\nWorldScript ( )\n"))
        before = u.fighter.health
        world.commands['InflictDamage'](u, 5.0)
        torso = u.fighter.parts[combat.TORSO]
        self.assertAlmostEqual(torso.max_health - torso.health, 5.0, places=3)
        self.assertLess(u.fighter.health, before)
        self.assertFalse(u.is_dead)
        world.commands['InflictDamage'](u, 1000.0)
        self.assertTrue(u.is_dead)

    def test_particles_of_the_spells_without_damage(self):
        """
        67f830 cases 0x12, 0x14, 0x1b: the fireworks, the clairvoyance and the link make their particles at the
        target (the link from the caster); they stop after the duration and go when the particles die.
        """
        spells = self.game_data.spells()
        caster = self._ai_unit(0.0, health=50)
        target = self._ai_unit(3.0, health=50)
        effects = []
        caster.on_spell_effect = effects.append
        link = magic.make_spell(spells, 'Link')
        unit_module.make_spell_effect(caster, link, target)
        self.assertEqual(len(effects), 1)
        visual = effects[0]
        self.assertIsInstance(visual, magic.SpellParticles)
        self.assertEqual(visual.emitters[0].code, particles.LINK)
        self.assertIsNotNone(visual.emitters[0].link_from)
        self.assertEqual(visual.left, link.duration)
        self.assertEqual(visual.lights, [])
        ticks = 0
        while not visual.finished and ticks < 1000:
            visual.update(1.0, [])
            ticks += 1
        self.assertTrue(visual.finished)
        self.assertGreaterEqual(ticks, link.duration)
        fireworks = magic.make_spell(spells, 'Fireworks')
        unit_module.make_spell_effect(caster, fireworks, target)
        self.assertEqual(effects[-1].emitters[0].code, particles.FIREWORK)
        self.assertEqual(len(effects[-1].lights), 1)

    def test_lasting_spells_without_damage(self):
        """
        682610 cases 0x12, 0x14, 0x1b: the fireworks alarm the units in its circle and sound their end, the
        clairvoyance sounds its end at 10 ticks before the end, the link breaks when its target dies.
        """
        spells = self.game_data.spells()
        caster = self._ai_unit(0.0, health=50)
        target = self._ai_unit(3.0, health=50)
        goblin = self._ai_unit(3.5)
        controller = ai.Controller(ai.Diplomacy(), rng=random.Random(1))
        controller.add(goblin, 3, ai.UnitLogic(model=0))
        effects = []
        caster.on_spell_effect = effects.append
        fireworks = magic.make_spell(spells, 'Fireworks')
        unit_module.make_spell_effect(caster, fireworks, target)
        effect = effects[-1]
        effect.sounds = []
        effect.update(1.0, [goblin])
        self.assertEqual(goblin.brain.threat_weight, int(fireworks.effect))
        self.assertEqual(goblin.brain.threat_step, magic.FIREWORKS_THREAT_STEP)
        effect.update(float(fireworks.duration), [goblin])
        self.assertEqual([s.name for s in effect.sounds], [magic.END_SOUNDS[magic.FIREWORKS_SPELL]])
        clairvoyance = magic.make_spell(spells, 'Clairvoyence')
        unit_module.make_spell_effect(caster, clairvoyance, target)
        effect = effects[-1]
        effect.sounds = []
        effect.update(float(clairvoyance.duration - magic.CLAIRVOYANCE_END_TICK - 1), [])
        self.assertEqual(effect.sounds, [])
        effect.update(1.0, [])
        self.assertEqual([s.name for s in effect.sounds], [magic.END_SOUNDS[magic.CLAIRVOYANCE_SPELL]])
        link = magic.make_spell(spells, 'Link')
        unit_module.make_spell_effect(caster, link, target)
        effect = effects[-1]
        effect.sounds = []
        effect.update(5.0, [])
        self.assertIsNotNone(effect.loop)
        target.die()
        effect.update(1.0, [])
        self.assertTrue(effect.broken)
        self.assertIsNone(effect.loop)
        self.assertEqual([s.name for s in effect.sounds], [magic.END_SOUNDS[magic.LINK_SPELL]])

    def test_trap_kills_a_unit(self):
        """
        A blow of a trap kills a unit without an error: the trap has no experience to share.
        """
        weak = self._ai_unit(1.0, health=5)
        trap = trap_module.MagicTrap(None, (0.0, 0.0, 2.0))
        weak.take_spell_blow(combat.Hit(weak.fighter.damage_types, 1000.0, location=combat.ALL_PARTS), trap, None)
        self.assertTrue(weak.is_dead)

    def test_magic_trap(self):
        """
        A magic trap: the lightning at the units of its circles every interval, off by ActivateTrap,
        at every unit which comes in with the interval 0, a wall at its points.
        """
        spells = self.game_data.spells()
        spell = trap_module.parse_spell(spells, 'lightning{e2;e2}')
        self.assertEqual(spell.index, 1)
        self.assertEqual(len(spell.modifiers), 2)
        inside = self._ai_unit(1.0, health=500)
        outside = self._ai_unit(9.0, health=500)
        effects = []
        trap = trap_module.MagicTrap(spell, (0.0, 0.0, 2.0), areas=[(0.0, 0.0, 3.0)], interval=5)
        trap.on_spell_effect = effects.append
        units = [inside, outside]
        trap.update(units)
        self.assertEqual(len(effects), 1)
        self.assertIs(effects[0].target, inside)
        self.assertLess(inside.fighter.health, 500.0)
        self.assertEqual(outside.fighter.health, 500.0)
        for i in range(4):
            trap.update(units)
        self.assertEqual(len(effects), 1)
        trap.update(units)
        self.assertEqual(len(effects), 2)
        world = script.ScriptWorld(script_lang.parse("GlobalVars ( )\nWorldScript ( )\n"))
        world.commands['ActivateTrap'](trap, 0.0)
        for i in range(10):
            trap.update(units)
        self.assertEqual(len(effects), 2)
        world.commands['ActivateTrap'](trap, 1.0)
        trap.update(units)
        self.assertEqual(len(effects), 3)
        # the players of the trap: the unit of the player 1 is not among them
        trap.players = 1 << 2
        trap.player_of = lambda u: 1
        trap.counter = 0
        trap.update(units)
        self.assertEqual(len(effects), 3)
        # the interval 0: once for a unit which comes in, again after it left and came back
        entry = trap_module.MagicTrap(spell, (0.0, 0.0, 2.0), areas=[(0.0, 0.0, 3.0)], interval=0)
        entry.on_spell_effect = effects.append
        entry.update(units)
        entry.update(units)
        self.assertEqual(len(effects), 4)
        inside.x = 5.0
        entry.update(units)
        inside.x = 1.0
        entry.update(units)
        self.assertEqual(len(effects), 5)
        # a spell of the point goes to the points of the trap
        wall = trap_module.MagicTrap(trap_module.parse_spell(spells, 'firewall'), (0.0, 0.0, 0.0),
                                     areas=[(0.0, 0.0, 3.0)], targets=[(4.0, 5.0)])
        walls = []
        wall.on_spell_effect = walls.append
        wall.update(units)
        self.assertEqual(len(walls), 1)
        self.assertEqual(walls[0].point, (4.0, 5.0))

    def test_script_spells(self):
        """
        CastSpellPoint / CastSpellUnit cast from a point without a caster; GetFutureX / GetFutureY go along the way.
        """
        spells = self.game_data.spells()
        u = self._ai_unit(0.0, health=50)
        u.target = (5.0, 0.0)
        u.waypoints = [(5.0, 5.0)]
        u.speed = 1.0
        self.assertEqual(u.future_position(3), (3.0, 0.0))
        self.assertEqual(u.future_position(7), (5.0, 2.0))
        self.assertEqual(u.future_position(100), (5.0, 5.0))
        cast = []

        class Host(script.Host):
            def cast_spell(self, text, origin, target, point):
                cast.append((text, origin, target, point))
                return True

        text = """GlobalVars ( u : object, F1 : float, F2 : float )
WorldScript (
    F1 = GetFutureX( u, 3 )
    F2 = GetFutureY( u, 7 )
    CastSpellPoint( "fireball", 1, 2, F1, F2 )
    CastSpellUnit( "healing{e2}", 1, 2, u )
)
"""
        world = script.ScriptWorld(script_lang.parse(text), Host())
        world.globals['u'] = u
        world.start()
        self.assertEqual((world.globals['F1'], world.globals['F2']), (3.0, 2.0))
        self.assertEqual(cast, [('fireball', (1.0, 2.0), None, (3.0, 2.0)), ('healing{e2}', (1.0, 2.0), u, None)])
        world = script.ScriptWorld(script_lang.parse(text))
        world.globals['u'] = u
        world.start()
        self.assertEqual(world.missing, {'CastSpellPoint fireball': 1, 'CastSpellUnit healing{e2}': 1})
        # the caster without a unit: a fireball from the point, the healing at the unit
        effects = []
        caster = trap_module.Caster((1.0, 2.0, 0.0))
        caster.on_spell_effect = effects.append
        caster.spread(trap_module.parse_spell(spells, 'fireball'), magic.Spot(3.0, 2.0), [u])
        self.assertEqual(len(effects), 1)
        self.assertEqual(effects[0].point, (3.0, 2.0))
        u.fighter.parts[combat.TORSO].health -= 5.0
        hurt = u.fighter.health
        caster.spread(trap_module.parse_spell(spells, 'healing{e2}'), u, [u])
        self.assertGreater(u.fighter.health, hurt)

    def test_music(self):
        """
        music.reg: the tracks of the allods; in a zone a calm track at once, the fight, the fading, the wait,
        the track of the scripts.
        """
        table = music.MusicTable.from_registry(self.game_data.registry(music.MUSIC_REG))
        self.assertEqual(table.main_menu, 'main')
        gipat = table.record('gipat', False)
        self.assertEqual((gipat.briefing, gipat.constructor, gipat.calm), ('village', 'shop', ['wild', 'gipath']))
        self.assertEqual(table.record('gipat', True).calm, ['horror'])
        self.assertIn('rage', gipat.combat)
        self.assertEqual(table.record('final', False).combat, ['evil'])
        self.assertEqual(music.zone_allod('gz20g', 'suslanger'), 'final')
        system = music.MusicSystem(table, rng=random.Random(3))
        self.assertTrue(system.set_zone('gipat', True))
        self.assertEqual(system.update(1000.0, False, ''), [(music.FADE, None), (music.PLAY, 'horror')])
        self.assertEqual(system.update(2000.0, False, 'horror'), [])
        requests = system.update(3000.0, True, 'horror')
        self.assertEqual(len(requests), 1)
        self.assertIn(requests[0][1], gipat.combat)
        self.assertEqual(system.update(4000.0, True, requests[0][1]), [])
        self.assertEqual(system.update(5000.0, False, requests[0][1]), [(music.FADE, None)])
        self.assertGreaterEqual(system.wait_until, 5000.0 + music.WAIT_MS)
        self.assertLess(system.wait_until, 5000.0 + music.WAIT_MS + music.WAIT_SPREAD_MS)
        self.assertEqual(system.update(6000.0, False, ''), [])
        self.assertEqual(system.update(system.wait_until, False, ''), [(music.PLAY, 'horror')])
        # the track of the scripts plays to its end, the fight does not change it
        system.play_script('Dungeon')
        self.assertEqual(system.update(10.0, True, 'dungeon'), [(music.PLAY, 'dungeon')])
        self.assertEqual(system.update(20.0, True, 'dungeon'), [])
        self.assertEqual(len(system.update(30.0, True, '')), 1)
        # PlayMusic of the scripts goes to the host for a player with a party
        played = []

        class Host(script.Host):
            def party(self, player):
                return object() if player == 0 else None

            def play_music(self, name):
                played.append(name)
                return True

        world = script.ScriptWorld(script_lang.parse("""GlobalVars ( )
WorldScript ( PlayMusic( 0, "Dungeon" ) PlayMusic( 1, "Nature" ) )
"""), Host())
        world.start()
        self.assertEqual(played, ['Dungeon'])

    def test_um_player(self):
        """
        UMPlayer: the unit does not go for a seen enemy until its player is in danger.
        """
        guard = self._ai_unit(0.0)
        friend = self._ai_unit(1.0)
        foe = self._ai_unit(4.0)
        relations = [ai.SAME_SIDE] * (ai.PLAYERS_COUNT * ai.PLAYERS_COUNT)
        relations[1 * ai.PLAYERS_COUNT + 2] = relations[2 * ai.PLAYERS_COUNT + 1] = ai.WAR
        controller = ai.Controller(ai.Diplomacy(relations))
        controller.add(guard, 1)
        controller.add(friend, 1)
        controller.add(foe, 2)
        controller.clear_behaviours(foe)
        controller.player_behaviour(guard)
        controller.think(guard.brain)
        self.assertIsNone(guard.brain.target)
        foe.brain.target = friend
        self.assertTrue(controller.in_danger(1))
        controller.think(guard.brain)
        self.assertIs(guard.brain.target, foe)

    def test_script_world(self):
        """
        SetDiplomacy and GetDiplomacy, InvokeAlarm and IsAlarm, UnitSee, UnitInSquare, the areas, the time.
        """
        near = self._ai_unit(0.0)
        far = self._ai_unit(30.0)
        controller = ai.Controller()
        controller.add(near, 1)
        controller.add(far, 2)
        text = """GlobalVars ( n : object, f : object, G : group, F1 : float, F2 : float, F3 : float, F4 : float,
  F5 : float, F6 : float, F7 : float, F8 : float, F9 : float, F10 : float )
WorldScript
(
  n = GetObject( 1 )
  f = GetObject( 2 )
  SetDiplomacy( 1, 2, -1 )
  F1 = GetDiplomacy( 1, 2 )
  F2 = GetDiplomacy( 2, 1 )
  InvokeAlarm( 3, 10, 12 )
  F3 = IsAlarm( 3 )
  F4 = AlarmPosY( 3 )
  G = UnitSee( n )
  F5 = UnitInSquare( f, 29, -1, 31, 1 )
  AddRectToArea( 7, 0, 0, 10, 10 )
  AddRoundToArea( 7, 20, 20, 2 )
  F6 = IsInArea( 7, 21, 21 )
  F7 = IsInArea( 7, 15, 15 )
  DeleteArea( 7 )
  F8 = IsInArea( 7, 5, 5 )
  FixWorldTime( 30 )
  F9 = GetWorldTime(  )
  F10 = IsPlayerInDanger( 0 )
)
"""
        hours = []

        class Host(script.Host):

            def object_by_id(self, nid):
                return {1: near, 2: far}.get(nid)

            def world_time(self):
                return hours[-1] if hours else 0.0

            def set_world_time(self, hour, runs):
                hours.append(hour)

        world = script.ScriptWorld(script_lang.parse(text), Host(), controller=controller)
        world.start()
        values = [world.globals['F%d' % i] for i in range(1, 11)]
        self.assertEqual(values, [-1.0, 1.0, 1.0, 12.0, 1.0, 1.0, 0.0, 0.0, 6.0, 0.0])
        self.assertEqual(world.globals['G'], ())
        self.assertEqual(world.missing, {})
        far.teleport_to(5.0, 0.0)
        self.assertEqual(controller.seen_by(near), [far])

    def test_script_orders(self):
        """
        MoveToPoint and RotateTo put the logic into the state 1 which SleepUntilIdle waits out; UMClear stops
        the fight, UMSentry sends the unit to its point, SetCP puts it there at once, KillUnit kills.
        """
        clock = timing.TickClock()
        walker = self._ai_unit(0.0)
        foe = self._ai_unit(3.0)
        controller = ai.Controller(ai.Diplomacy({(1, 2): ai.WAR, (2, 1): ai.WAR}))
        controller.add(walker, 1)
        controller.add(foe, 2)
        text = """GlobalVars ( w : object, f : object, F1 : float )
WorldScript
(
  UMClear( w )
  UMClear( f )
  Run( w )
  MoveToPoint( w, 0, 6 )
  SleepUntilIdle( w )
  RotateTo( w, 5, 6 )
  SleepUntilIdle( w )
  F1 = 1
  UMSentry( w, 0, 12 )
  Sleep( 200 )
  SetCP( f, 20, 20, 0 )
  KillUnit( f )
)
"""

        class Host(script.Host):

            def object_by_id(self, nid):
                return {1: walker, 2: foe}.get(nid)

        program = script_lang.parse(text.replace('UMClear( w )', 'w = GetObject( 1 ) f = GetObject( 2 ) UMClear( w )'))
        world = script.ScriptWorld(program, Host(), controller=controller)
        world.start()
        self.assertEqual(walker.brain.order, ai.ORDER_MOVE)
        self.assertEqual(walker.mode, unit_module.MODE_RUN)
        moved = turned = None
        for tick in range(400):
            controller.update(1)
            world.tick()
            for u in (walker, foe):
                u.update(timing.TICK_MS, clock)
            if moved is None and world.globals['F1'] == 0.0 and walker.mode == unit_module.MODE_IDLE:
                moved = tick
            if turned is None and world.globals['F1'] == 1.0:
                turned = tick
                self.assertLess(math.hypot(walker.x, walker.y - 6.0), 0.5)
                self.assertLess(abs(walker.yaw - ai.look_yaw(5.0, 0.0)), 0.1)
        self.assertIsNotNone(moved)
        self.assertIsNotNone(turned)
        # the enemies were cleared of their behaviours: no fight
        self.assertIsNone(walker.brain.target)
        self.assertLess(math.hypot(walker.x, walker.y - 12.0), ai.SENTRY_RADIUS + 0.5)
        self.assertEqual((foe.x, foe.y), (20.0, 20.0))
        self.assertTrue(foe.is_dead)
        self.assertEqual(world.missing, {})

    def test_transformation_particles(self):
        """
        bz7g turns the hero into JunBoy: StartTrans (0x2051) on the hero, the sparks come into the point and
        the source leaves the hero after 31 ticks; Transform (0x2052) rises from the point for 30 ticks.
        """
        hero = self._ai_unit(10.0, 20.0)
        world = script.ScriptWorld(script_lang.parse('GlobalVars ( )\nWorldScript ( )\n'))
        c = world.commands
        c['CreateParticleSource'](1.0, 0.0, 0.0, 0.0, 2.0, 'StartTrans')
        c['AttachParticles'](1.0, hero)
        source = world.sources.particles[1]
        self.assertEqual(source.emitter.texture_name, 'zoneexit')
        world.sources.update()
        middle = hero.z + hero.height() * 0.5
        self.assertEqual(source.emitter.current, (10.0, 20.0, middle))
        emitter = source.emitter
        for tick in range(10):
            world.sources.update()
        sparks = [p for p in emitter.particles if p.kind == 0]
        self.assertTrue(sparks)
        for p in sparks:
            # on the ball of the radius 2 and going in
            self.assertLessEqual(math.dist((p.x, p.y, p.z), (10.0, 20.0, middle)), 2.0 + 1e-6)
        hero.teleport_to(12.0, 20.0)
        for tick in range(25):
            world.sources.update()
        self.assertFalse(emitter.emitting)
        self.assertFalse(emitter.attached)
        # it stays where it left the hero
        hero.teleport_to(30.0, 20.0)
        world.sources.update()
        self.assertAlmostEqual(emitter.current[0], 12.0)
        c['CreateParticleSource'](2.0, 5.0, 5.0, 1.0, 2.0, 'Transform')
        transform = world.sources.particles[2].emitter
        self.assertEqual(transform.texture_name, 'changehero')
        # the source is made by the first birth and moves from the next tick on
        for tick in range(31):
            world.sources.update()
        self.assertFalse(transform.emitting)
        self.assertAlmostEqual(transform.sources[0].p[2], 2.0 * 0.12 * 30, places=4)
        self.assertGreater(max(p.z for p in transform.particles), 1.0 + 3.0)
        self.assertEqual(world.missing, {})
        for name in ('zoneexit', 'changehero'):
            self.assertIsNotNone(self.game_data.texture(name))

    def test_script_behaviours(self):
        """
        zone6 makes a dragon a friend: UMClear, UMFear( d, 0 ), UMAggression and UMFollow( d, hero ): it goes after
        the hero and attacks an enemy which it sees. UMRevenge leaves a unit alone until somebody hits it.
        """
        clock = timing.TickClock()
        hero = self._ai_unit(0.0)
        dragon = self._ai_unit(-3.0)
        avenger = self._ai_unit(-30.0, 20.0)
        enemy = self._ai_unit(40.0, sight=0.0)
        # the player 0 is at war with all the others
        controller = ai.Controller(ai.Diplomacy(), rng=random.Random(1))
        controller.add(hero, 0)
        controller.add(dragon, 0)
        controller.add(avenger, 0)
        controller.add(enemy, 2)
        world = script.ScriptWorld(script_lang.parse('GlobalVars ( )\nWorldScript ( )\n'), controller=controller)
        c = world.commands
        for u in (hero, dragon, avenger, enemy):
            c['UMClear'](u)
        c['UMFear'](dragon, 0.0)
        c['UMAggression'](dragon)
        c['UMFollow'](dragon, hero)
        c['UMRevenge'](avenger)
        self.assertTrue(dragon.brain.wary)
        self.assertFalse(dragon.brain.fears_all)
        self.assertIs(dragon.brain.companion, hero)
        hero.go_to(25.0, 0.0)
        # avenger and enemy are 6 apart, the avenger sees him but does not attack
        avenger.teleport_to(34.0, 0.0)

        def run(ticks):
            for _ in range(ticks):
                controller.update(1)
                for u in (hero, dragon, avenger, enemy):
                    u.update(timing.TICK_MS, clock)

        run(150)
        self.assertLess(dragon.distance_to(hero), ai.FOLLOW_DISTANCE * 2.0 + 1.0)
        self.assertIsNone(avenger.brain.target)
        # the enemy comes into the sight of the dragon
        enemy.teleport_to(dragon.x + 5.0, dragon.y)
        run(10)
        self.assertIs(dragon.brain.target, enemy)
        # the avenger answers a blow
        controller.attacked(avenger, enemy)
        self.assertIs(avenger.brain.target, enemy)
        self.assertEqual(world.missing, {})

    def test_unit_casts_a_spell(self):
        """
        The caster comes within the range, plays CAST and lets the spell go at its hit_frame (54ee10):
        the lightning hits at once, the fire arrow flies as a missile; the hero pays the mana.
        """
        spells = self.game_data.spells()
        clock = timing.TickClock()
        human = self.game_data.units().unit_type('Human Male')
        hero = unit_module.Unit(figure.FigureModel(self.game_data, 'unhuma'), position=(0.0, 0.0), rng=random.Random(2))
        hero_stats = stats.UnitStats(20, 20, 25, 25, 5, 5, 1, 2, 0, speeds=(0.8, 0.4, 0.2, 0.1))
        hero.set_stats(hero_stats)
        hero.fighter = combat.Fighter(hero_stats, human, in_party=True, radius=combat.unit_radius(hero.half_size()))
        foe = self._ai_unit(30.0)
        lightning = magic.make_spell(spells, 'Lightning Arc')
        self.assertTrue(hero.cast(foe, lightning))
        # the foe is out of the range: the hero goes to it
        self.assertEqual(hero.mode, unit_module.MODE_ATTACK)
        self.assertIsNotNone(hero.target)
        for _ in range(400):
            hero.update(timing.TICK_MS, clock)
            if hero.spell is None:
                break
        self.assertLessEqual(hero.distance_to(foe), lightning.range + 0.5)
        self.assertEqual(hero.player.clip.action.type_name, 'CAST')
        self.assertAlmostEqual(hero.fighter.mana, 25.0 - 15.0)
        self.assertAlmostEqual(foe.fighter.health, 20.0 - 8.0, places=4)
        # not enough mana for another one
        self.assertFalse(hero.cast(foe, lightning))
        missiles_sent = []
        hero.on_shoot = missiles_sent.append
        hero.fighter.mana = 25.0
        arrow = magic.make_spell(spells, 'Fire Arrow')
        self.assertTrue(hero.cast(foe, arrow))
        for _ in range(400):
            hero.update(timing.TICK_MS, clock)
            if missiles_sent:
                break
        missile = missiles_sent[0]
        self.assertEqual((missile.step, missile.model_name), (magic.MISSILE_STEP, None))
        self.assertAlmostEqual(foe.fighter.health, 12.0, places=4)
        while missile.update(1.0):
            pass
        hero.deliver_blow(missile.target, missile.hit)
        self.assertAlmostEqual(foe.fighter.health, 12.0 - 7.0, places=4)

    def test_area_spells_hit_around_their_point(self):
        """
        67f830/681b70: the fireball flies distance / speed * 15 ticks and bursts in the circle of its radius
        (6835c0); 682610: the wall of fire hits its strip across the way every 4th tick (6836f0), the victims
        answer only the first blow.
        """
        spells = self.game_data.spells()
        caster = self._ai_unit(0.0)
        near, side, far = self._ai_unit(10.0), self._ai_unit(11.5), self._ai_unit(14.0)
        units = [caster, near, side, far]
        fireball = magic.make_spell(spells, 'Fireball')
        self.assertAlmostEqual(fireball.radius, math.sqrt(25.0 / math.pi))
        effect = magic.AreaEffect(fireball, caster, (0.0, 0.0), (10.0, 0.0))
        effects = [effect]
        for tick in range(15):
            effects = unit_module.update_spell_effects(effects, units, 1.0)
        self.assertFalse(effect.done)
        self.assertEqual([e.code for e in effect.emitters], [particles.FIRE_SMOKE])
        self.assertAlmostEqual(effect.position[0], 14 * 10.0 / 15.0, places=4)
        self.assertAlmostEqual(near.fighter.health, 20.0)
        effects = unit_module.update_spell_effects(effects, units, 1.0)
        self.assertTrue(effect.done)
        # the smoke fades and the explosion flies on after the blow
        self.assertEqual([e.code for e in effect.emitters], [particles.FIRE_SMOKE, particles.EXPLOSION])
        self.assertFalse(effect.smoke.emitting)
        while effects:
            effects = unit_module.update_spell_effects(effects, units, 1.0)
        self.assertEqual(effect.emitters, [])
        self.assertEqual([round(u.fighter.health, 3) for u in units], [20.0, 13.0, 13.0, 20.0])
        # the wall stands across the way from the caster: 0.5 to both sides of its line, its radius long
        wall = magic.make_spell(spells, 'Wall of Fire')
        inside, behind, aside = self._ai_unit(10.0, 1.0), self._ai_unit(10.6), self._ai_unit(10.0, 2.0)
        answered = []
        for u in (inside, behind, aside):
            u.on_attacked = lambda u, attacker: answered.append(attacker)
        effect = magic.AreaEffect(wall, caster, (0.0, 0.0), (10.0, 0.0))
        effects = [effect]
        ticks = 0
        while not effect.done:
            effects = unit_module.update_spell_effects(effects, [inside, behind, aside], 1.0)
            ticks += 1
        self.assertEqual(ticks, 1 + wall.duration + 2)
        # the fire stands across the way: along y, as long as the radius to both sides
        self.assertEqual(effect.emitters[0].half, (-0.0, wall.radius, 0.0))
        # 38 blows of 10 * 4 / 150
        self.assertAlmostEqual(inside.fighter.health, 20.0 - 38 * 10.0 * 4.0 / 150.0, places=3)
        self.assertEqual((behind.fighter.health, aside.fighter.health), (20.0, 20.0))
        self.assertEqual(answered, [caster])

    def test_healing_particles(self):
        """
        48b630/48bbe0/48bce0: the balls circle the unit for 36 ticks from the bottom of its sphere to the top,
        the sparks follow them, then the balls fade for 8 ticks.
        """
        emitter = particles.Emitter(particles.HEALING, (10.0, 20.0, 0.0), rng=random.Random(1))
        emitter.holder = (1.0, 1.0)
        emitter.update()
        # round(0.5 * 6 + 2) = 5 sources, a ball for each one, the sparks up to 500
        self.assertEqual(len(emitter.sources), 5)
        self.assertEqual(len([p for p in emitter.particles if p.kind >= 0]), 5)
        self.assertEqual(len(emitter.particles), emitter.most)
        self.assertEqual(emitter.least, 15)
        lows = [s.p[2] for s in emitter.sources]
        self.assertTrue(all(-0.2 <= z <= 0.2 for z in lows))
        ticks = 1
        while emitter.emitting:
            emitter.update((11.0, 20.0, 0.0))
            ticks += 1
        # the sources come on the first birth, after the move of the sources of that tick
        self.assertEqual(ticks, particles.HEALING_TICKS + 1)
        # the sources came to the top of the sphere, near the middle again
        self.assertTrue(all(abs(s.p[2] - 2.0) < 0.25 for s in emitter.sources))
        # the radius grows for 18 ticks and shrinks for 17: one step is left
        self.assertTrue(all(abs(s.v[0] - emitter.turn_growth) < 1e-6 for s in emitter.sources))
        # the balls follow their sources on the holder, then fade
        balls = [p for p in emitter.particles if p.kind >= 0]
        self.assertEqual(len(balls), 5)
        for ball in balls:
            if ball.kind != particles.HEALING_FADING:
                self.assertAlmostEqual(ball.x, 11.0 + emitter.sources[ball.kind].p[0])
        while emitter.update():
            ticks += 1
        self.assertEqual(emitter.particles, [])
        self.assertLessEqual(ticks, particles.HEALING_TICKS + particles.HEALING_FADE_TICKS + 2)
        # without a unit it makes nothing
        lone = particles.Emitter(particles.HEALING, (0.0, 0.0, 0.0))
        self.assertFalse(lone.update())

    def test_teleport(self):
        """
        67f830/682610 case 28: the sparks rise at the caster and fall into the point, after 30 ticks the caster
        is at the point and frozen for the duration; it stays when it can not stand there.
        """
        spells = self.game_data.spells()
        teleport = magic.make_spell(spells, 'Teleport')
        caster = self._ai_unit(0.0)
        effect = magic.Teleport(teleport, caster, (8.0, 3.0), rng=random.Random(2))
        self.assertEqual([e.size > 0.0 for e in effect.emitters], [True, False])
        self.assertEqual([s.name for s in effect.sounds], [magic.START_SOUNDS[magic.TELEPORT_SPELL]])
        effects = [effect]
        for tick in range(magic.TELEPORT_DELAY - 1):
            effects = unit_module.update_spell_effects(effects, [caster], 1.0)
        self.assertEqual((caster.x, caster.y), (0.0, 0.0))
        # the sparks of the caster go up, the ones of the point come down to it
        rising, falling = effect.emitters
        self.assertTrue(all(p.z > 0.0 for p in rising.particles if p.life < 10))
        self.assertTrue(all(p.z > -0.2 for p in falling.particles))
        effects = unit_module.update_spell_effects(effects, [caster], 1.0)
        self.assertEqual((caster.x, caster.y), (8.0, 3.0))
        self.assertTrue(caster.is_frozen)
        self.assertFalse(effect.rising.emitting)
        while effects:
            effects = unit_module.update_spell_effects(effects, [caster], 1.0)
        self.assertTrue(effect.finished)
        # a point where it can not stand
        other = self._ai_unit(0.0)
        effect = magic.Teleport(teleport, other, (8.0, 3.0), can_stand=lambda u, x, y: False)
        effects = [effect]
        while effects:
            effects = unit_module.update_spell_effects(effects, [other], 1.0)
        self.assertEqual((other.x, other.y), (0.0, 0.0))
        self.assertFalse(other.is_frozen)

    def test_spell_sounds(self):
        """
        67f830/681b70/682610: the start, the loop and the end of the spells; 5112a0: the enchantments.
        """
        spells = self.game_data.spells()
        caster = self._ai_unit(0.0)
        fireball = magic.make_spell(spells, 'Fireball')
        speed = fireball.speed if fireball.speed > 0.0 else magic.DEFAULT_SPEED
        # a long flight loops the sound with the ball, the burst ends it
        effect = magic.AreaEffect(fireball, caster, (0.0, 0.0), (speed * 2.0, 0.0))
        effects = unit_module.update_spell_effects([effect], [caster], 1.0)
        self.assertEqual([s.name for s in effect.sounds], [magic.FLYING_FIREBALL_SOUND])
        self.assertEqual(effect.loop.name, magic.CIRCLE_SOUNDS[magic.FIREBALL_SPELL])
        effect.sounds = []
        effects = unit_module.update_spell_effects(effects, [caster], 2.0)
        self.assertGreater(effect.loop.position[0], 0.0)
        while not effect.done:
            effects = unit_module.update_spell_effects(effects, [caster], 1.0)
        self.assertIsNone(effect.loop)
        self.assertEqual([s.name for s in effect.sounds], [magic.END_SOUNDS[magic.FIREBALL_SPELL]])
        # a short one only starts
        effect = magic.AreaEffect(fireball, caster, (0.0, 0.0), (speed, 0.0))
        unit_module.update_spell_effects([effect], [caster], 1.0)
        self.assertEqual([s.name for s in effect.sounds], [magic.START_SOUNDS[magic.FIREBALL_SPELL]])
        self.assertIsNone(effect.loop)
        # the strike of Invoke Lightning is heard over the map
        sound = magic.start_sound(magic.make_spell(spells, 'Invoke Lightning'), (0.0, 0.0, 0.0))
        self.assertEqual((sound.near, sound.far), (magic.SOUND_EVERYWHERE, magic.SOUND_EVERYWHERE))
        self.assertIsNone(magic.start_sound(magic.make_spell(spells, 'Speed'), (0.0, 0.0, 0.0)))
        # the enchantments: the start when it comes, the warning of the party, the end when it goes
        fighter = caster.fighter
        fighter.in_party = True
        fighter.enchant(33, 100, 1.0)
        found, seen = magic.enchantment_sounds(set(), fighter, (0.0, 0.0, 0.0))
        self.assertEqual([s.name for s in found], ['magic\\Speed\\start.wav'])
        self.assertIsNone(magic.enchantment_warning(fighter, (0.0, 0.0, 0.0)))
        fighter.tick_enchantments(80)
        self.assertEqual(magic.enchantment_warning(fighter, (0.0, 0.0, 0.0)).name, magic.ENCHANTMENT_WARNING_SOUND)
        fighter.tick_enchantments(30)
        found, seen = magic.enchantment_sounds(seen, fighter, (0.0, 0.0, 0.0))
        self.assertEqual([s.name for s in found], ['magic\\Speed\\end.wav'])
        self.assertEqual(seen, set())
        # the volume of the distances of the spells
        self.assertEqual(sounds.volume(10.0, 10.0, 50.0), 1.0)
        self.assertAlmostEqual(sounds.volume(40.0, 10.0, 50.0), 0.25)
        self.assertEqual(sounds.volume(51.0, 10.0, 50.0), 0.0)

    def test_lightning_bolts(self):
        """
        CLightning (466820, 4680d0, 4672e0): the segments of a path, its axes across it, the branches of a bolt
        with power on its way, the copies of the wall, the balls; the strikes of the invoke lightning, the bolt
        of the lightning wall and the arc which follows its units.
        """
        self.assertEqual(lightning.segments_of(5.0), (48, 5.0))
        self.assertEqual(lightning.segments_of(12.3), (60, 12.3))
        self.assertEqual(lightning.segments_of(40.0), (72, 15.0))
        bolt = lightning.Bolt()
        bolt.aim((1.0, 2.0, 0.0), (6.0, 2.0, 3.0))
        d = np.array((5.0, 0.0, 3.0))
        self.assertAlmostEqual(float(np.dot(bolt.side, d)), 0.0, places=6)
        self.assertAlmostEqual(float(np.dot(bolt.up, d)), 0.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(bolt.side)), math.sqrt(34.0), places=6)
        self.assertTrue(np.allclose(bolt.end, (6.0, 2.0, 3.0)))
        rng = random.Random(5)
        table = lightning.noise_table()
        try:
            lightning.load_noise(np.array([rng.uniform(-0.05, 0.05) for _ in range(32 * 400)], dtype='<f4').tobytes())
            arc = lightning.Lightning((0.0, 0.0, 1.0), (5.0, 0.0, 1.0), 4.0, rng=rng)
            bands = arc.bands(0.5)
            self.assertEqual(len(bands), 7)
            points, width, along, band = bands[0]
            # the path starts at its start, the noise only bends it
            self.assertTrue(np.allclose(points[0], (0.0, 0.0, 1.0)))
            self.assertGreater(float(np.abs(points[:, 1:]).max()), 0.0)
            self.assertAlmostEqual(width, 0.4)
            self.assertAlmostEqual(along, 3.0)
            # the branches start on the main path, 0.2 of the way from its end
            index = int(round(0.8 * arc.main.count))
            self.assertTrue(np.allclose(bands[1][0][0], arc.main.points[index]))
            self.assertTrue(all(0.7 * 2.0 * 0.1 <= b[1] < 1.5 * 2.0 * 0.1 for b in bands[1:]))
            self.assertEqual(arc.balls(), [((5.0, 0.0, 1.0), 0.55)])
            # the storm kind: six copies, balls at both ends; the strike: no ball, the texture by its first column
            wall = lightning.Lightning((0.0, -3.0, 1.0), (0.0, 3.0, 1.0), 0.0, rng=rng)
            self.assertEqual(wall.main.width, 4.0)
            self.assertEqual(len(wall.balls()), 2)
            strike = lightning.Lightning((0.0, 0.0, 30.0), (0.0, 0.0, 0.0), -7.0, rng=rng)
            self.assertAlmostEqual(strike.main.width, 10.5)
            self.assertAlmostEqual(strike.sway, 0.05)
            self.assertEqual(strike.balls(), [])
            self.assertEqual({b[2] for b in strike.bands(0.0)}, {0.0})
            squares = lightning.ball_squares((0.0, 0.0, 0.0), 1.0, (0.0, 0.0, 1.0), 14)
            self.assertEqual([s[2] for s in squares[:2]], [14, 1])
            self.assertAlmostEqual(squares[4][1], math.sqrt(0.75))
        finally:
            lightning.load_noise(table.tobytes())
        # the invoke lightning strikes on its start and its third tick, one tick each
        spells = self.game_data.spells()
        caster = self._ai_unit(0.0)
        invoke = magic.AreaEffect(magic.make_spell(spells, 'Invoke Lightning'), caster, (0.0, 0.0), (5.0, 0.0),
                                  rng=random.Random(1))
        powers = []
        for tick in range(6):
            invoke.update(1.0, [])
            powers.append([round(b.power, 2) for b in invoke.bolts()])
        self.assertEqual(powers, [[-5.6], [], [], [-7.0], [], []])
        self.assertEqual(invoke.bolts(), [])
        # the lightning wall stands between the ends of the wall while it lasts
        spell = magic.make_spell(spells, 'Smtg Lightning')
        wall = magic.AreaEffect(spell, caster, (0.0, 0.0), (5.0, 0.0), rng=random.Random(1))
        wall.update(1.0, [])
        bolt = wall.bolts()[0]
        self.assertEqual(bolt.power, 0.0)
        self.assertTrue(np.allclose(bolt.start, (5.0, -spell.radius, 1.0)))
        self.assertTrue(np.allclose(bolt.end, (5.0, spell.radius, 1.0)))
        while not wall.done:
            wall.update(1.0, [])
        wall.update(1.0, [])
        self.assertEqual(wall.bolts(), [])
        # the arc follows the caster and the target for 7 ticks
        target = self._ai_unit(4.0)
        arc = magic.LightningArc(magic.make_spell(spells, 'Lightning Arc'), caster, target, rng=random.Random(1))
        self.assertTrue(np.allclose(arc.bolts()[0].end, (4.0, 0.0, target.z + 1.0)))
        self.assertEqual(arc.bolts()[0].texture_name, 'Lightning')
        effects = [arc]
        ticks = 0
        while arc.bolts():
            self.assertTrue(np.allclose(arc.bolts()[0].end, (target.x, 0.0, target.z + 1.0)))
            target.x = 5.0
            effects = unit_module.update_spell_effects(effects, [caster, target], 1.0)
            ticks += 1
        self.assertEqual(ticks, 7)
        demon = magic.LightningArc(magic.make_spell(spells, 'Curse Magic'), caster, target)
        self.assertEqual(demon.bolts()[0].ball_texture_name, 'DemonLight')

    def test_particles_of_the_spells(self):
        """
        CParticle::vf06 (496eb0) and the types of 485540: the frames go from the bottom row of the atlas, the
        smoke of a fireball is born along its way and lives 15 ticks, a cloud fills its circle at once and fades
        when it stops, an explosion emits 7 ticks and ends.
        """
        self.assertEqual(particles.frame_uv(0), (0.0, 0.75, 0.25, 1.0))
        self.assertEqual(particles.frame_uv(15), (0.75, 0.0, 1.0, 0.25))
        self.assertAlmostEqual(magic.visual_strength(7.0, 7.0), 0.8)
        self.assertAlmostEqual(magic.visual_strength(56.0, 7.0), 2.0 ** (0.33 * 3) - 0.2)
        self.assertAlmostEqual(magic.visual_strength(99.0, 7.0), magic.visual_strength(56.0, 7.0))
        smoke = particles.Emitter(particles.FIRE_SMOKE, (0.0, 0.0, 1.0), rng=random.Random(3))
        x = 0.0
        for _ in range(30):
            x += 0.6666667
            self.assertTrue(smoke.update((x, 0.0, 1.0)))
            self.assertTrue(all(0 < p.life <= 15 for p in smoke.particles))
        # round(0.667 / 0.09) = 7 tries a tick with the chance 88: 14 ticks of them live
        self.assertGreater(len(smoke.particles), 60)
        self.assertLess(len(smoke.particles), 7 * 15)
        smoke.stop()
        ticks = 0
        while smoke.update((x, 0.0, 1.0)):
            ticks += 1
        self.assertLessEqual(ticks, 15)
        cloud = particles.Emitter(particles.ACID_FOG, (10.0, 5.0, 0.0), scale=2.26, intensity=0.8,
                                  rng=random.Random(3))
        cloud.update()
        count = len(cloud.particles)
        self.assertTrue(all(math.hypot(p.x - 10.0, p.y - 5.0) < 2.26 for p in cloud.particles))
        self.assertGreater(count, 9)
        self.assertEqual({p.color >> 24 for p in cloud.particles}, {128})
        for _ in range(10):
            cloud.update()
        self.assertEqual(len(cloud.particles), count)
        cloud.stop()
        ticks = 0
        while cloud.update():
            ticks += 1
        # the alpha 128 falls by round(0.8 * 4.5) = 4 a tick below 10
        self.assertEqual(ticks, 30)
        explosion = particles.Emitter(particles.EXPLOSION, (0.0, 0.0, 1.0), scale=0.8, rng=random.Random(3))
        ticks = 0
        while explosion.emitting:
            explosion.update()
            ticks += 1
        self.assertEqual(ticks, 7)
        while explosion.update():
            ticks += 1
        self.assertLess(ticks, 40)

    def test_spell_spreads_over_its_targets(self):
        """
        58e850: a spell of several targets or with a filter goes to the main one and then to the nearest
        ones to it within the range of the caster which pass the filter.
        """
        spells = self.game_data.spells()
        caster = self._ai_unit(0.0)
        main, second, friend, third, away = (self._ai_unit(10.0), self._ai_unit(12.0), self._ai_unit(10.5),
                                             self._ai_unit(15.0), self._ai_unit(25.0))
        units = [caster, main, second, friend, third, away]
        caster.hostile_to = lambda unit, other: other is not friend and other is not unit
        lightning = magic.make_spell(spells, 'Lightning Arc')
        self.assertEqual(magic.spread(lightning, caster, main, units), [main])
        lightning.flags |= magic.FILTER_ENEMIES
        lightning.targets = 3
        self.assertEqual(magic.spread(lightning, caster, main, units), [main, second, third])
        caster.neighbours = lambda: units
        caster.spell = lightning
        caster.release_spell(main)
        self.assertEqual([round(u.fighter.health, 3) for u in (main, second, friend, third, away)],
                         [12.0, 12.0, 20.0, 12.0, 20.0])
        # a spell of the point makes its effect at the point of the main target
        made = []
        caster.on_spell_effect = made.append
        caster.spell = magic.make_spell(spells, 'Invoke Lightning')
        caster.release_spell(magic.Spot(10.0, 0.0))
        self.assertEqual([e.point for e in made], [(10.0, 0.0)])

    def test_diplomacy_and_logic_of_zone1(self):
        root = self.game_data.map_mob('zone1')
        diplomacy = ai.Diplomacy.from_mob(root)
        self.assertTrue(diplomacy.at_war(ai.PARTY_PLAYER, 3))
        self.assertTrue(diplomacy.at_war(6, ai.PARTY_PLAYER))
        self.assertEqual(diplomacy.relation(ai.PARTY_PLAYER, 2), ai.NEUTRAL)
        self.assertTrue(diplomacy.friends(6, 6))
        self.assertFalse(diplomacy.at_war(3, 6))
        settings = ai.Settings.from_registry(self.game_data.registry('config/ai.reg'))
        self.assertEqual((settings.guard_radius, settings.call_for_help_radius), (10.0, 7.0))
        goblins = [o for o in world.map_objects(root) if o.is_unit and o.prototype == 'GoblinTutor']
        logic = ai.UnitLogic.from_node(goblins[0].logic_node)
        self.assertEqual(logic.aggression_mode, 0)
        self.assertGreater(logic.guard_radius, 0)
        self.assertIsNotNone(logic.guard_point)
        goblin = self.game_data.units().prototype('GoblinTutor')
        self.assertAlmostEqual(stats.from_prototype(goblin).senses[ai.SIGHT], goblin.senses[0])

    def test_unit_attacks_the_enemy_it_sees(self):
        clock = timing.TickClock()
        controller = ai.Controller(ai.Diplomacy())
        hero = self._ai_unit(0.0)
        controller.add_player_unit(hero)
        goblin = self._ai_unit(12.0)
        brain = controller.add(goblin, 3)
        controller.update(ai.THINK_TICKS)
        # 12 is farther than the sight 8.5
        self.assertIsNone(brain.target)
        hero.x = 4.0
        controller.update(ai.THINK_TICKS)
        self.assertIs(brain.target, hero)
        self.assertIs(goblin.foe, hero)
        self.assertEqual(goblin.mode, unit_module.MODE_ATTACK)
        # a unit of the same side is no target
        friend = self._ai_unit(13.0)
        controller.add(friend, 3)
        controller.update(ai.THINK_TICKS)
        self.assertIs(goblin.foe, hero)
        for _ in range(100):
            goblin.update(timing.TICK_MS, clock)
        self.assertLess(goblin.distance_to(hero), goblin.strike_distance(hero) + 0.5)

    def test_hit_unit_fights_back_and_calls_for_help(self):
        controller = ai.Controller(ai.Diplomacy(), ai.Settings(call_for_help_radius=7.0))
        hero = self._ai_unit(0.0, attack=50)
        controller.add_player_unit(hero)
        # the hero shoots from afar: the goblins do not see it
        goblin = self._ai_unit(20.0)
        near = self._ai_unit(24.0)
        far = self._ai_unit(40.0)
        peaceful = self._ai_unit(21.0)
        brains = [controller.add(u, 3) for u in (goblin, near, far)]
        quiet = controller.add(peaceful, 3, ai.UnitLogic(model=0))
        controller.update(ai.THINK_TICKS)
        self.assertTrue(all(b.target is None for b in brains))
        hero.fighter.reach = 30.0
        hero.attack(goblin)
        hero.resolve_blow()
        self.assertIs(brains[0].target, hero)
        self.assertIs(goblin.foe, hero)
        self.assertIs(brains[1].target, hero)
        self.assertIsNone(brains[2].target)
        self.assertIsNone(quiet.target)
        self.assertIsNot(peaceful.foe, hero)

    def test_patrols_of_the_maps(self):
        """
        5d6630: the points at the middles of their cells, the yaw to the look point, the waits in ticks;
        a patrol which is not cyclic goes back by the same points.
        """
        root = self.game_data.map_mob('zone1')
        ogres = [o for o in world.map_objects(root) if o.is_unit and o.logic_node is not None and
                 len([c for c in o.logic_node.children if c.name == 'GUARD_PT' and c.children]) == 1 and
                 len(ai.UnitLogic.from_node(o.logic_node).patrol_points[0][2]) == 3]
        logic = ai.UnitLogic.from_node(ogres[0].logic_node)
        self.assertEqual(logic.model, ai.PATROL_MODEL)
        x, y, actions = logic.patrol_points[0]
        self.assertEqual((x, y), (154.75, 108.25))
        self.assertEqual([wait for _, wait in actions], [15, 15, 15])
        # the first look point is to the south-west: the model looks at -y, so the yaw is between -pi/2 and 0
        self.assertTrue(-math.pi / 2 < actions[0][0] < 0.0)
        there_and_back = ai.UnitLogic(model=ai.PATROL_MODEL, patrol_points=[(0, 0, []), (5, 0, [])], cyclic=False)
        self.assertEqual([p[0] for p in there_and_back.patrol_route()], [0, 5, 5, 0])
        self.assertEqual(ai.UnitLogic(model=1, patrol_points=[(0, 0, [])]).patrol_route(), [])

    def test_unit_patrols(self):
        clock = timing.TickClock()
        controller = ai.Controller(ai.Diplomacy())
        u = self._ai_unit(0.0)
        points = [(0.25, 0.25, [(1.0, 20)]), (6.25, 0.25, [])]
        brain = controller.add(u, 3, ai.UnitLogic(model=ai.PATROL_MODEL, patrol_points=points))
        visited = []
        for _ in range(400):
            controller.update(1)
            u.update(timing.TICK_MS, clock)
            if not visited or visited[-1] != brain.route_index:
                visited.append(brain.route_index)
            if brain.route_index == 0 and brain.wait_until is not None:
                self.assertAlmostEqual(u.target_yaw, 1.0)
        self.assertEqual(visited[:4], [0, 1, 0, 1])
        self.assertLess(abs(u.x - 0.25) + abs(u.x - 6.25), 6.0 + 1e-6)
        # an enemy in sight outweighs the patrol, after the fight the unit goes on
        hero = self._ai_unit(3.0, 2.0)
        controller.add_player_unit(hero)
        for _ in range(ai.THINK_TICKS):
            controller.update(1)
        self.assertIs(brain.target, hero)
        hero.die()
        for _ in range(ai.THINK_TICKS):
            controller.update(1)
        self.assertIsNone(brain.target)
        self.assertIsNotNone(brain.route_index)

    def test_guard_wanders_in_its_circle(self):
        """
        5d0be0, 5d7f80, 5d5b40: a free guard walks to random points of its circle and looks around there.
        """
        clock = timing.TickClock()
        controller = ai.Controller(ai.Diplomacy(), ai.Settings(guard_radius=4.0, guard_stay_time=1.0),
                                   rng=random.Random(3))
        u = self._ai_unit(0.0)
        brain = controller.add(u, 3, ai.UnitLogic(model=ai.GUARD_MODEL, guard_point=(0.0, 0.0)))
        farthest = 0.0
        looks = 0
        for _ in range(2000):
            controller.update(1)
            u.update(timing.TICK_MS, clock)
            farthest = max(farthest, math.hypot(u.x, u.y))
            if brain.wait_until is not None:
                looks += 1
        self.assertGreater(farthest, 0.5)
        self.assertLess(farthest, 4.0 + 0.5)
        self.assertGreater(looks, 0)
        # a unit of the model 0 stays
        still = self._ai_unit(10.0)
        controller.add(still, 3, ai.UnitLogic(model=0))
        for _ in range(200):
            controller.update(1)
            still.update(timing.TICK_MS, clock)
        self.assertEqual((still.x, still.y), (10.0, 0.0))

    def test_party_fights_on_its_own(self):
        """
        5d4170, 5d3f50: a free unit of the player attacks the enemy it notices or the foe of a fighting friend.
        """
        controller = ai.Controller(ai.Diplomacy(), ai.Settings(player_call_for_help=16.0))
        hero = self._ai_unit(0.0)
        friend = self._ai_unit(-12.0)
        controller.add_player_unit(hero)
        controller.add_player_unit(friend)
        goblin = self._ai_unit(5.0)
        controller.add(goblin, 3, ai.UnitLogic(model=0))
        hero.auto_fight = False
        controller.update(ai.THINK_TICKS)
        self.assertNotEqual(hero.mode, unit_module.MODE_ATTACK)
        # the friend is too far to see the goblin (8.5) and nobody fights yet
        self.assertNotEqual(friend.mode, unit_module.MODE_ATTACK)
        hero.auto_fight = True
        controller.update(ai.THINK_TICKS)
        self.assertIs(hero.foe, goblin)
        self.assertEqual(hero.mode, unit_module.MODE_ATTACK)
        # the friend 12 from the fighting hero helps it
        controller.update(ai.THINK_TICKS)
        self.assertIs(friend.foe, goblin)
        # a dead foe frees the units
        goblin.die()
        hero.idle()
        controller.update(ai.THINK_TICKS)
        self.assertNotEqual(hero.mode, unit_module.MODE_ATTACK)

    def test_guard_idle_signal(self):
        class One(random.Random):
            def randrange(self, *args):
                return 1
        controller = ai.Controller(ai.Diplomacy(), rng=One(1))
        u = self._ai_unit(0.0)
        u.mode = unit_module.MODE_WALK
        brain = controller.add(u, 3, ai.UnitLogic(model=ai.GUARD_MODEL, guard_point=(0.0, 0.0)))
        controller.wander(brain)
        self.assertEqual(brain.signal.pending, signals.IDLE_A)

    def test_look_actions(self):
        controller = ai.Controller(ai.Diplomacy(), ai.Settings(guard_stay_time=2.0), rng=random.Random(5))
        u = self._ai_unit(0.0)
        brain = controller.add(u, 3, ai.UnitLogic(model=ai.GUARD_MODEL, guard_point=(0.0, 0.0)))
        for _ in range(50):
            actions = controller.look_actions(brain, (1.0, 1.0))
            self.assertTrue(1 <= len(actions) <= 4)
            for yaw, wait in actions:
                # stay 2: 30 ticks at most, a quarter of it at least; a human has no long look
                self.assertTrue(7.5 <= wait <= 30.0)
                self.assertTrue(-math.pi <= yaw <= math.pi)

    def test_ground_slope(self):
        class Slope(object):
            def height_at(self, x, y):
                return x * 0.8
        self.assertAlmostEqual(ai.ground_slope(Slope(), 1.0, 1.0), 0.4)
        self.assertEqual(ai.ground_slope(None, 1.0, 1.0), 0.0)

    def test_shooter_keeps_away(self):
        """
        5d2960: a shooter in good health runs from an enemy nearer than RangedAttackDist.
        """
        toad = self.game_data.units().prototype('ToadTutor')
        self.assertAlmostEqual(stats.from_prototype(toad).attack_range, 8.0)
        controller = ai.Controller(ai.Diplomacy(), rng=random.Random(1))
        hero = self._ai_unit(0.0)
        controller.add_player_unit(hero)
        shooter = self._ai_unit(5.0)
        shooter.attack_range = 8.0
        self.assertTrue(shooter.shoots)
        brain = controller.add(shooter, 3)
        controller.update(ai.THINK_TICKS)
        self.assertTrue(brain.fleeing)
        self.assertEqual(shooter.mode, unit_module.MODE_RUN)
        end = shooter.waypoints[-1] if shooter.waypoints else shooter.target
        self.assertGreater(math.hypot(end[0] - hero.x, end[1] - hero.y), 5.0 + 5.0 - 3.0 * math.sqrt(2) - 1e-6)
        # a wounded shooter fights where it stands
        shooter.stop()
        shooter.idle()
        brain.fleeing = False
        shooter.fighter.parts[combat.TORSO].health = -1.0
        controller.update(ai.THINK_TICKS)
        self.assertFalse(brain.fleeing)
        # a melee unit does not run
        brawler = self._ai_unit(-5.0)
        self.assertFalse(brawler.shoots)
        other = controller.add(brawler, 3)
        controller.update(ai.THINK_TICKS)
        self.assertFalse(other.fleeing)
        self.assertIs(brawler.foe, hero)

    def test_alarms(self):
        """
        593fc0 and 5d5940: a unit raises its alarm by its condition, the units of the map take their records
        of the logic of the alarm when they are in use.
        """
        root = self.game_data.map_mob('zone12')
        raisers = [o for o in world.map_objects(root) if o.is_unit and o.logic_nodes and
                   ai.UnitLogic.from_node(o.logic_nodes[0]).alarm == 1]
        self.assertTrue(raisers)
        self.assertEqual(len(raisers[0].logic_nodes), ai.ALARMS_COUNT)
        controller = ai.Controller(ai.Diplomacy())
        hero = self._ai_unit(0.0)
        controller.add_player_unit(hero)
        sentry = self._ai_unit(20.0)
        sentry_brain = controller.add(sentry, 3, logics=[ai.UnitLogic(alarm=1, alarm_condition=ai.ALARM_ON_HIT)])
        guard = self._ai_unit(60.0)
        alarm_logic = ai.UnitLogic(guard_point=(25.0, 0.0, 0.0), guard_radius=2.0, use=True)
        guard_brain = controller.add(guard, 3, logics=[ai.UnitLogic(), alarm_logic])
        idle = self._ai_unit(-60.0)
        idle_brain = controller.add(idle, 3, logics=[ai.UnitLogic(), ai.UnitLogic(use=False)])
        controller.update(ai.THINK_TICKS)
        self.assertEqual(controller.alarms, {})
        # the sentry is hit: its alarm 1 is raised at the next thought
        hero.fighter.reach = 30.0
        hero.attack(sentry)
        hero.resolve_blow()
        self.assertTrue(sentry_brain.alarm_armed)
        controller.update(ai.THINK_TICKS)
        self.assertIn(1, controller.alarms)
        self.assertIs(guard_brain.logic, alarm_logic)
        self.assertEqual(guard_brain.home, (25.0, 0.0))
        self.assertIsNot(idle_brain.logic, idle_brain.logics[1])
        # the guard runs to its new point
        controller.update(ai.THINK_TICKS)
        self.assertIn(guard.mode, (unit_module.MODE_WALK, unit_module.MODE_RUN))
        # once
        self.assertFalse(controller.raise_alarm(1, (0.0, 0.0)))
        self.assertTrue(controller.raise_alarm(1, (0.0, 0.0), force=True))
        self.assertTrue(controller.raise_alarm(0, (0.0, 0.0), force=True))
        self.assertEqual(list(controller.alarms), [0])

    def test_unit_gives_up_and_goes_home(self):
        controller = ai.Controller(ai.Diplomacy())
        hero = self._ai_unit(5.0)
        controller.add_player_unit(hero)
        goblin = self._ai_unit(0.0)
        brain = controller.add(goblin, 3, ai.UnitLogic(guard_point=(0.0, 0.0, 0.0), guard_radius=5.0))
        controller.update(ai.THINK_TICKS)
        self.assertIs(brain.target, hero)
        # the hero runs away farther than the guard radius plus the sight from the point
        hero.x = 30.0
        goblin.x = 12.0
        controller.update(ai.THINK_TICKS)
        self.assertIsNone(brain.target)
        # 5d76a0, 5d21a0: it goes to look at the place where the hero got away
        self.assertTrue(brain.suspicious)
        self.assertEqual(brain.threat_point, (30.0, 0.0))
        self.assertEqual(goblin.mode, unit_module.MODE_WALK)
        self.assertEqual(goblin.waypoints[-1] if goblin.waypoints else goblin.target, (30.0, 0.0))
        # the suspicion fades, the goblin walks home
        hero.x = 60.0
        brain.threat_weight = -100
        controller.update(ai.THINK_TICKS)
        self.assertFalse(brain.suspicious)
        self.assertEqual(goblin.mode, unit_module.MODE_WALK)
        self.assertEqual(goblin.waypoints[-1] if goblin.waypoints else goblin.target, (0.0, 0.0))
        # a dead hero is no target
        hero.x = 3.0
        hero.die()
        controller.update(ai.THINK_TICKS)
        self.assertIsNone(brain.target)

    def test_suspicion(self):
        """
        5cbbe0, 5cb700, 5d1e90, 5d21a0, 5d2310: a unit which hears an enemy behind it remembers the place as a threat,
        raises the signal 5 and goes to look there; it attacks the enemy when it sees it. A threat without an enemy
        fades and the unit calms down with the signal 3.
        """
        controller = ai.Controller(ai.Diplomacy(), rng=random.Random(1))
        clock = timing.TickClock()
        hero = self._ai_unit(-5.0)
        hero.auto_fight = False
        controller.add_player_unit(hero)
        goblin = self._ai_unit(0.0)
        goblin.base_senses = (8.5, 0.0, 0.0, 6.0, 0.0, 0.0)
        goblin.refresh_senses()
        goblin.view_cos = 0.0
        goblin.yaw = goblin.target_yaw = ai.look_yaw(1.0, 0.0)
        brain = controller.add(goblin, 3, ai.UnitLogic(guard_point=(0.0, 0.0, 0.0), guard_radius=5.0))
        # the hero stands behind the goblin, quiet
        controller.update(ai.THINK_TICKS)
        self.assertIsNone(brain.target)
        self.assertFalse(brain.suspicious)
        self.assertEqual(brain.seen, [])
        # he walks: the goblin hears him
        hero.go_to(-5.0, 3.0)
        controller.update(ai.THINK_TICKS)
        self.assertIsNone(brain.target)
        self.assertTrue(brain.suspicious)
        self.assertEqual(brain.threat_weight, ai.THREAT_SOUND)
        self.assertEqual(brain.signal.pending, signals.TO_SUSPICION)
        self.assertEqual(goblin.mode, unit_module.MODE_WALK)
        # it turns to the place and sees him
        for _ in range(30):
            controller.update(1)
            for u in (hero, goblin):
                u.update(timing.TICK_MS, clock)
        self.assertIs(brain.target, hero)
        self.assertFalse(brain.suspicious)
        self.assertIn(hero, brain.seen)
        # 5caa00 (partly): once seen he is not lost while he lives; 5d2710: his death clears the threat
        hero.die()
        controller.update(ai.THINK_TICKS)
        self.assertIsNone(brain.target)
        self.assertEqual(brain.seen, [])
        self.assertEqual(brain.threat_weight, 0)
        self.assertFalse(brain.suspicious)
        # a threat with no enemy: the goblin looks around at the place until the weight fades
        self.assertFalse(brain.remember_threat((3.0, 0.0), 0, ai.THREAT_ENEMY_STEP))
        self.assertTrue(brain.remember_threat((3.0, 0.0), ai.THREAT_ENEMY, ai.THREAT_ENEMY_STEP))
        self.assertFalse(brain.remember_threat((9.0, 0.0), ai.THREAT_SOUND, ai.THREAT_SOUND_STEP))
        looked = False
        ticks = 0
        while brain.suspicious or ticks < ai.THINK_TICKS * 2:
            controller.update(1)
            goblin.update(timing.TICK_MS, clock)
            looked = looked or brain.wait_until is not None
            ticks += 1
            self.assertLess(ticks, 700)
        self.assertTrue(looked)
        # 1000 fades by 2 a tick down to the guard without the 100 of the running behaviour
        self.assertGreater(ticks, 500)
        self.assertLess(brain.suspicion_weight + ai.RUNNING_WEIGHT, ai.GUARD_WEIGHT)

    def test_what_the_player_sees(self):
        """
        668c10, 5ca5a0, 5caa00: the player sees the units its party saw; a living one is lost farther than the sight
        and the life sense, unless it attacks; nothing is hidden without a party.
        """
        controller = ai.Controller(ai.Diplomacy(), rng=random.Random(1))
        goblin = self._ai_unit(5.0)
        controller.add(goblin, 3, ai.UnitLogic(model=0))
        controller.update(ai.THINK_TICKS)
        self.assertIsNone(controller.visible)
        self.assertTrue(controller.is_visible(goblin))
        hero = self._ai_unit(0.0)
        hero.auto_fight = False
        controller.add_player_unit(hero)
        far = self._ai_unit(30.0)
        controller.add(far, 3, ai.UnitLogic(model=0))
        controller.update(ai.THINK_TICKS)
        self.assertTrue(controller.is_visible(hero))
        self.assertTrue(controller.is_visible(goblin))
        self.assertFalse(controller.is_visible(far))
        self.assertEqual(controller.party_seen[hero], [goblin])
        # out of the sight: lost
        goblin.x = 20.0
        controller.update(ai.THINK_TICKS)
        self.assertFalse(controller.is_visible(goblin))
        # it attacks the hero from afar: kept once seen
        goblin.x = 5.0
        controller.update(ai.THINK_TICKS)
        goblin.foe = hero
        goblin.mode = unit_module.MODE_ATTACK
        goblin.x = 20.0
        controller.update(ai.THINK_TICKS)
        self.assertTrue(controller.is_visible(goblin))
        # the dead are seen
        far.die()
        controller.update(ai.THINK_TICKS)
        self.assertTrue(controller.is_visible(far))

    def test_spells_show_units_to_the_player(self):
        """
        682610 cases 0x14, 0x15: the clairvoyance of the party shows the units in its circle while it lasts,
        the possession shows its target and the units the target sees.
        """
        spells = self.game_data.spells()
        controller = ai.Controller(ai.Diplomacy(), rng=random.Random(1))
        hero = self._ai_unit(0.0)
        hero.auto_fight = False
        controller.add_player_unit(hero)
        goblin = self._ai_unit(40.0)
        controller.add(goblin, 3, ai.UnitLogic(model=0))
        wolf = self._ai_unit(45.0)
        controller.add(wolf, 3, ai.UnitLogic(model=0))
        controller.effects = []
        hero.on_spell_effect = controller.effects.append
        controller.update(ai.THINK_TICKS)
        self.assertFalse(controller.is_visible(goblin))
        clairvoyance = magic.make_spell(spells, 'Clairvoyence')
        unit_module.make_spell_effect(hero, clairvoyance, goblin)
        controller.update(ai.THINK_TICKS)
        self.assertTrue(controller.is_visible(goblin))
        self.assertFalse(controller.is_visible(wolf))
        effect = controller.effects.pop()
        effect.update(float(clairvoyance.duration + 1), [])
        controller.update(ai.THINK_TICKS)
        self.assertFalse(controller.is_visible(goblin))
        possession = magic.make_spell(spells, 'Possession')
        unit_module.make_spell_effect(hero, possession, goblin)
        effect = controller.effects[-1]
        self.assertIsInstance(effect, magic.Possession)
        controller.update(ai.THINK_TICKS)
        self.assertTrue(controller.is_visible(goblin))
        self.assertTrue(controller.is_visible(wolf))
        effect.update(float(possession.duration + 1), [])
        self.assertTrue(effect.finished)
        controller.update(ai.THINK_TICKS)
        self.assertFalse(controller.is_visible(goblin))

    def test_a_blow_is_a_threat(self):
        """
        5d1e90: the place of the one who hit the unit is a threat of 1000, not the place of a friend.
        """
        controller = ai.Controller(ai.Diplomacy(), rng=random.Random(1))
        hero = self._ai_unit(20.0)
        controller.add_player_unit(hero)
        goblin = self._ai_unit(0.0)
        friend = self._ai_unit(2.0)
        brain = controller.add(goblin, 3, ai.UnitLogic(model=0))
        controller.add(friend, 3)
        controller.attacked(goblin, friend)
        self.assertIsNone(brain.threat_point)
        controller.attacked(goblin, hero)
        self.assertEqual(brain.threat_point, (20.0, 0.0))
        self.assertEqual((brain.threat_weight, brain.threat_step), (ai.THREAT_ENEMY, ai.THREAT_ENEMY_STEP))
        # a peaceful unit does not go to look
        controller.update(ai.THINK_TICKS)
        self.assertFalse(brain.suspicious)

    def test_called_shot(self):
        """
        The blow of a called shot goes to the chosen part and is aimed (the order keeps +0xf0 and +0xf4).
        """
        clock = timing.TickClock()
        a = self._ai_unit(0.0, attack=100)
        b = self._ai_unit(1.0, attack=0)
        hits = []
        a.on_blow = lambda attacker, target, hit, dealt: hits.append(hit)
        a.aim_part = combat.HEAD
        a.attack(b)
        for _ in range(60):
            a.update(timing.TICK_MS, clock)
            if hits:
                break
        self.assertEqual((hits[0].location, hits[0].aimed), (combat.HEAD, True))

    def test_delay_between_the_strikes(self):
        """
        54d960: the next strike starts round(delay * 15 / speed) ticks after the last one; 524560: the speed
        is 15 for a monster, dexterity * 0.2 + 10 for a hero, with the quickness, the arms and the spells 33, 34.
        """
        human = self.game_data.units().unit_type('Human Male')
        wolf = self.game_data.units().prototype('WolfGreyF0')
        self.assertEqual(wolf.strike_delay, 68.0)
        sword = [w for w in self.game_data.items().weapons if w.name == 'long sword'][0]
        self.assertEqual(sword.strike_delay, 30)
        fighter = combat.Fighter(stats.UnitStats(20, 20, 0, 0, 5, 5, 1, 2, 0), human)
        self.assertEqual(fighter.strike_ticks(fighter.strike_speed()), combat.DEFAULT_STRIKE_DELAY)
        fighter.strike_delay = sword.strike_delay
        # Zak: the dexterity 30 gives the speed 16
        self.assertAlmostEqual(fighter.strike_speed(30.0), 16.0)
        self.assertEqual(fighter.strike_ticks(16.0), 28)
        self.assertAlmostEqual(fighter.strike_speed(30.0, quickness=50.0, arms_factor=0.5), 12.0)
        fighter.enchant(combat.SLOW_SPELL, 150, 3.0)
        self.assertAlmostEqual(fighter.strike_speed(), 12.0)
        clock = timing.TickClock()
        a = self._ai_unit(0.0, attack=0)
        b = self._ai_unit(1.0, attack=0)
        starts = []
        for tick in range(120):
            if a.foe is None:
                a.attack(b)
            before = a.strike_action
            a.update(timing.TICK_MS, clock)
            b.update(timing.TICK_MS, clock)
            if a.strike_action is not None and a.strike_action is not before:
                starts.append(tick)
        # every strike waits the delay of the human (50 ticks) after the start of the last one
        self.assertGreaterEqual(len(starts), 2)
        self.assertTrue(all(t2 - t1 >= 50 for t1, t2 in zip(starts, starts[1:])))

    def test_long_frame_stops_the_chase_at_the_strike(self):
        clock = timing.TickClock()
        human = self.game_data.units().unit_type('Human Male')
        fighters = []
        for x in (0.0, 6.0):
            u = unit_module.Unit(figure.FigureModel(self.game_data, 'unhuma'), position=(x, 0.0),
                                 rng=random.Random(2))
            unit_stats = stats.UnitStats(20, 20, 0, 0, 5, 5, 1, 2, 0, speeds=(0.8, 0.4, 0.2, 0.1))
            u.set_stats(unit_stats)
            u.fighter = combat.Fighter(unit_stats, human, radius=combat.unit_radius(u.half_size()))
            fighters.append(u)
        a, b = fighters
        a.attack(b)
        a.update(timing.MAX_STEP_MS, clock)
        a.update(timing.TICK_MS, clock)
        self.assertTrue(a.can_strike(b))
        self.assertGreaterEqual(a.distance_to(b), a.strike_distance(b) - 1e-6)
        # the strike started: it plays or the unit waits for the delay of the next one (54d960)
        self.assertGreater(a.strike_wait, 0)
        self.assertTrue(a.player.clip.action is a.strike_action or a.waiting)

    def _voice(self, type_name, prototype_name, weapon_code=None):
        units = self.game_data.units()
        return sounds.Voice.from_unit_type(units.unit_type(type_name), units.prototype(prototype_name), weapon_code)

    def test_voices_of_units(self):
        """
        The folder of tag 34 of the type, replaced by tag 8 of the prototype; the code of the blow of tag 39.
        """
        ogre = self._voice('Ogre', 'OgreTutor')
        self.assertEqual((ogre.sfx_path, ogre.idle_chance, ogre.attack_chance), ('monsters\\ogre', 15, 100))
        self.assertEqual((ogre.material, ogre.weapon_code, ogre.humanoid), ('Leather', 12, False))
        hero = self._voice('Human Male', 'Human Hero')
        self.assertEqual((hero.sfx_path, hero.weapon_code, hero.humanoid), ('heroes\\Hero', 9, True))
        self.assertEqual(self._voice('Human Male', 'Human Hero', weapon_code=0).weapon_code, 0)
        self.assertIsNotNone(self.game_data.sound_name('heroes\\Hero\\attack\\1.wav'))

    def test_voice_at_the_frame_of_the_sound(self):
        """
        534760: the voice starts when the clip passes the frame of the sound of the action.
        """
        attack = [a for a in self.game_data.animation_database('unmoog1').actions if a.name == 'uattack01'][0]
        self.assertEqual(attack.special_sound_frame, 10)
        tracker = sounds.SoundTracker(self._voice('Ogre', 'OgreTutor'), rng=random.Random(1))
        started, stopped = tracker.update(attack, 5.0)
        self.assertEqual((started, stopped), ([], [sounds.CHANNEL_IDLE]))
        started, stopped = tracker.update(attack, 11.0)
        self.assertEqual(stopped, [])
        self.assertEqual(len(started), 1)
        sound = started[0]
        self.assertRegex(sound.name, r'^monsters\\ogre\\attack\\[1-4]\.wav$')
        self.assertEqual((sound.priority, sound.channel), (sounds.PRIORITY_HIGH, sounds.CHANNEL_ATTACK))
        self.assertIsNotNone(self.game_data.sound_name(sound.name))
        self.assertEqual(tracker.update(attack, 20.0), ([], []))
        # the attack voice does not start over one which still plays
        busy = sounds.SoundTracker(self._voice('Ogre', 'OgreTutor'), rng=random.Random(1))
        self.assertEqual(busy.update(attack, 11.0, busy_channels={sounds.CHANNEL_ATTACK})[0], [])
        # no chance, no voice
        silent = sounds.SoundTracker(sounds.Voice('monsters\\ogre', attack_chance=0), rng=random.Random(1))
        self.assertEqual(silent.update(attack, 11.0)[0], [])

    def test_steps_by_the_tiles(self):
        """
        534b60 and 534590: a step on the frames sound_frames - 1, the folder of tag 35 by the type of the tile.
        """
        units = self.game_data.units()
        human = units.unit_type('Human Male')
        self.assertEqual(len(human.step_paths), 16)
        voice = sounds.Voice.from_unit_type(human, units.prototype('Human Hero'), sound_count=self.game_data.sound_count)
        self.assertEqual(voice.step_counts[6], 6)
        self.assertEqual(voice.step_counts[8], 0)
        walk = [a for a in self.game_data.animation_database('unhuma').actions if a.name == 'cwalk01'][0]
        self.assertEqual(walk.sound_frames[:2], (7, 19))
        tracker = sounds.SoundTracker(voice, rng=random.Random(1))
        self.assertEqual(tracker.update(walk, 5.5, step_tile=lambda: 6)[0], [])
        started = tracker.update(walk, 6.0, step_tile=lambda: 6)[0]
        self.assertEqual(len(started), 1)
        self.assertRegex(started[0].name, r'^steps\\Human\\Water\\[1-6]\.wav$')
        self.assertIsNotNone(self.game_data.sound_name(started[0].name))
        self.assertEqual(tracker.update(walk, 17.0, step_tile=lambda: 6)[0], [])
        self.assertEqual(len(tracker.update(walk, 18.5, step_tile=lambda: 0)[0]), 1)
        # no files for the tile type 8
        self.assertEqual(tracker.update(walk, 6.5, step_tile=lambda: 8)[0], [])
        land = terrain.Terrain(self.game_data, 'zone1')
        self.assertEqual(land.tile_type_at(163.0, 112.0), 6)
        self.assertGreater(land.water_height_at(163.0, 112.0), land.height_at(163.0, 112.0))
        self.assertEqual(land.tile_type_at(-5.0, 10.0), terrain.OUTSIDE_TILE_TYPE)

    def test_sound_of_a_blow(self):
        """
        537500: the kind by the code of the weapon, the material by the skin of the target.
        """
        sound = sounds.blow_sound(12, 'Leather', 5.0)
        self.assertEqual(sound.name, 'weapons\\hands4\\leather\\1.wav')
        self.assertIsNotNone(self.game_data.sound_name(sound.name))
        self.assertEqual(sounds.blow_sound(0, 'Stone', 1.0).name, 'weapons\\slash\\stone\\1.wav')
        self.assertEqual(sounds.blow_sound(4, 'Metal', 1.0).name, 'weapons\\crash\\Metal\\1.wav')
        self.assertIsNotNone(self.game_data.sound_name('weapons\\crash\\Metal\\1.wav'))
        self.assertIsNone(sounds.blow_sound(16, 'Leather', 5.0))
        self.assertIsNone(sounds.blow_sound(12, 'Leather', 0.0))
        self.assertIsNone(sounds.miss_sound(self._voice('Ogre', 'OgreTutor')))
        self.assertEqual(sounds.miss_sound(self._voice('Human Male', 'Human Hero')).name, sounds.MISS_SOUND)
        self.assertAlmostEqual(sounds.volume(5.0), 1.0)
        # 6b7f80: the volume falls evenly from 8 to 40 by whole percents
        self.assertAlmostEqual(sounds.volume(16.0), 0.75)
        self.assertAlmostEqual(sounds.volume(24.0), 0.5)
        self.assertAlmostEqual(sounds.volume(40.0), 0.0)
        self.assertAlmostEqual(sounds.volume(39.9), 0.0)
        self.assertAlmostEqual(sounds.volume(39.8), 0.01)
        self.assertAlmostEqual(sounds.volume(41.0), 0.0)
        # 6b7cd0: a free channel, else the first of the lowest priority when the new sound is above it
        self.assertEqual(sounds.take_channel([None] * sounds.CHANNELS, sounds.PRIORITY_LOW), 0)
        self.assertEqual(sounds.take_channel([0, 1000, None, 0], 0), 2)
        self.assertEqual(sounds.take_channel([1000, 0, 500, 0], 0), -1)
        self.assertEqual(sounds.take_channel([1000, 0, 500, 0], 1000), 1)
        self.assertEqual(sounds.take_channel([1000, 1000], 1000), -1)
        # the height of the listener: all up to 30, nothing from 70
        self.assertAlmostEqual(sounds.height_factor(30.0), 1.0)
        self.assertAlmostEqual(sounds.height_factor(50.0), 0.5)
        self.assertAlmostEqual(sounds.height_factor(70.0), 0.0)
        self.assertAlmostEqual(sounds.volume(5.0, height=50.0), 0.5)
        self.assertAlmostEqual(sounds.volume(24.0, height=60.0), 0.125)
        self.assertAlmostEqual(sounds.volume(5.0, height=80.0), 0.0)

    def test_arrow_homes_on_its_target(self):
        """
        67aa70: the horizontal step of the arrow is 1.333 a tick, it turns to the target which moves.
        """
        model = figure.FigureModel(self.game_data, 'unhuma')
        target = unit_module.Unit(model, position=(20.0, 0.0), rng=random.Random(1))
        arrow = missiles.Arrow(None, target, None, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(math.hypot(arrow.velocity[0], arrow.velocity[1]), missiles.ARROW_STEP)
        ticks = 0
        while arrow.update(1.0):
            ticks += 1
            if ticks == 5:
                target.y = 6.0
        self.assertTrue(arrow.arrived)
        self.assertLess(math.hypot(arrow.x - target.x, arrow.y - target.y), missiles.ARROW_STEP)
        self.assertGreater(ticks, 13)
        self.assertLess(ticks, 20)
        # it lies along its flight
        m = arrow.world_matrix()
        self.assertEqual(m.shape, (4, 4))

    def test_arrow_model(self):
        self.assertTrue(self.game_data.has_model(missiles.ARROW_MODEL))
        arrow = figure.FigureModel(self.game_data, missiles.ARROW_MODEL)
        self.assertEqual(arrow.skeleton.visible, ['arrow'])
        self.assertIsNotNone(self.game_data.texture(missiles.ARROW_TEXTURE))
        items = self.game_data.items()
        self.assertEqual([items.shape(n).range for n in ('short bow', 'long bow', 'crossbow', 'long sword')],
                         [12.0, 16.0, 16.0, 0.0])

    def test_shooter_sends_an_arrow(self):
        clock = timing.TickClock()
        human = self.game_data.units().unit_type('Human Male')
        shooter = unit_module.Unit(figure.FigureModel(self.game_data, 'unhuma'), position=(0.0, 0.0), weapon='BOW',
                                   rng=random.Random(2))
        target = unit_module.Unit(figure.FigureModel(self.game_data, 'unhuma'), position=(10.0, 0.0),
                                  rng=random.Random(2))
        for u, attack in ((shooter, 40), (target, 0)):
            unit_stats = stats.UnitStats(20, 20, 0, 0, attack, 0, 5, 6, 0)
            u.fighter = combat.Fighter(unit_stats, human, in_party=u is shooter,
                                       radius=combat.unit_radius(u.half_size()))
        shooter.fighter.reach = 12.0
        self.assertEqual(shooter.weapon_code, 5)
        arrows = []
        shooter.on_shoot = arrows.append
        # the back of the target is turned to the shooter, but a bow does not backstab
        target.yaw = math.pi / 2
        self.assertTrue(shooter.attack(target))
        self.assertIs(shooter.player.clip.action, shooter.strike_action)
        for _ in range(60):
            shooter.update(timing.TICK_MS, clock)
            if arrows:
                break
        self.assertEqual(len(arrows), 1)
        arrow = arrows[0]
        self.assertAlmostEqual(arrow.hit.backstab, 1.0)
        self.assertAlmostEqual(target.fighter.health, 20.0)
        while arrow.update(1.0):
            pass
        shooter.deliver_blow(arrow.target, arrow.hit)
        self.assertLess(target.fighter.health, 20.0)

    def test_walk_starts_and_limps(self):
        model = figure.FigureModel(self.game_data, 'unhuma')
        clock = timing.TickClock()
        walker = unit_module.Unit(model, rng=random.Random(1))
        walker.move_to(0, -20)
        self.assertEqual(walker.player.clip.action.stage, actions.STAGE_START)
        self.assertEqual(walker.player.queue[0].action.stage, actions.STAGE_CYCLE)
        for _ in range(12):
            walker.update(timing.TICK_MS, clock)
        cycle = walker.player.clip.action
        self.assertEqual((cycle.type_name, cycle.stage), ('WALK', actions.STAGE_CYCLE))
        walker.move_to(0, -30)
        self.assertIs(walker.player.clip.action, cycle)
        lame = unit_module.Unit(model, rng=random.Random(1))
        lame.legs_part = 0.4
        lame.move_to(0, -20, run=True)
        self.assertEqual(lame.mode, unit_module.MODE_WALK)
        self.assertEqual(lame.player.queue[0].action.modifier, 2)
        self.assertAlmostEqual(lame.speed, walker.speed * 0.5)

    def test_path_around_the_river_of_zone1(self):
        tiles = pathing.tile_descriptions(self.game_data.registry('tileDesc.reg'))
        self.assertEqual(tiles[7].name, 'ROAD')
        grid = pathing.PassGrid(terrain.Terrain(self.game_data, 'zone1'), tiles)
        start, goal = (140.0, 100.0), (160.0, 120.0)
        human = pathing.HUMAN_CLASS
        self.assertFalse(grid.line_is_open(human, grid.point(*start), grid.point(*goal)))
        points = grid.find_path(human, start, goal)
        self.assertIsNotNone(points)
        # the goal is closed: the path goes to the open point of the region next to it on the way back (5bac10)
        self.assertLessEqual(math.hypot(points[-1][0] - goal[0], points[-1][1] - goal[1]), 3.0)
        # the sea cannot be reached: the path still goes to the shore toward it
        shore = grid.find_path(human, start, (2.0, 100.0))
        self.assertIsNotNone(shore)
        self.assertLess(shore[-1][0], start[0] - 50.0)
        self.assertTrue(grid.is_open(human, *grid.point(*shore[-1])))
        for a, b in zip([start] + points, points):
            self.assertTrue(grid.line_is_open(human, grid.point(*a), grid.point(*b)))
        self.assertGreater((grid.codes[pathing.FLYING_CLASS] != 0).mean(), (grid.codes[human] != 0).mean())
        self.assertEqual(self.game_data.units().unit_type('Human Male').path_class, human)

    def test_objects_close_points_and_bridges_open_them(self):
        land = terrain.Terrain(self.game_data, 'zone1')
        parts = []
        bridge = None
        for obj in world.map_objects(self.game_data.map_mob('zone1')):
            if obj.is_unit or not self.game_data.has_model(obj.template):
                continue
            x, y, z = obj.position
            matrix = np.identity(4)
            matrix[0:3, 0:3] = obj.rotation_matrix()
            matrix[0:3, 3] = (x, y, land.height_at(x, y) + z)
            model = figure.FigureModel(self.game_data, obj.template, coefs=obj.complection)
            parts.extend(pathing.object_parts(model, matrix))
            if obj.template == 'stbr1' and x > 165.0:
                bridge = (x, y)
        tiles = pathing.tile_descriptions(self.game_data.registry('tileDesc.reg'))
        plain = pathing.PassGrid(land, tiles)
        grid = pathing.PassGrid(land, tiles, parts)
        human = pathing.HUMAN_CLASS
        self.assertLess((grid.codes[human] != 0).mean(), (plain.codes[human] != 0).mean() - 0.03)
        self.assertFalse(plain.is_open(human, *plain.point(*bridge)))
        self.assertTrue(grid.is_open(human, *grid.point(*bridge)))
        # the sides of the boxes of the ruins at the start do not wall the hero in (5b2d10, 5b2e10)
        hero = 3
        path = grid.find_path(hero, (54.0, 138.0), (128.0, 96.0))
        self.assertIsNotNone(path)
        self.assertLess(math.hypot(path[-1][0] - 128.0, path[-1][1] - 96.0), 1.0)

    def test_items_of_the_strip_have_figures_and_textures(self):
        items = self.game_data.items()
        axe = equipment.make_item(items, 'stone axe')
        self.assertTrue(self.game_data.has_model(inventory.item_model_name(axe)))
        self.assertIsNotNone(self.game_data.texture(inventory.item_texture_name(axe)))
        self.assertIsNotNone(self.game_data.texture(inventory.LOWER_TEXTURE))

    def test_menus_reg(self):
        registry = self.game_data.registry(menus.MENUS_REG)
        self.assertEqual(registry[menus.MAIN_MENU][menus.NEW_GAME], (367, 175, 507, 216))
        self.assertEqual(set(registry[menus.ESC_MENU]), set(menus.ESC_MENU_BUTTONS))

    def test_stones_of_the_main_menu_under_their_rects(self):
        """
        The camera of MainMenu.cam, the angle of view of game.exe and the column of the .mob put every stone
        into its rect of menus.reg.
        """
        keys = self.game_data.camera(menus.MAIN_MENU_CAMERA)
        view = gl.camera_view(keys[0].position, quat.to_matrix(keys[0].rotation))
        projection = gl.perspective(gl.FIELD_OF_VIEW, 800.0 / 600.0, 0.1, 500.0)
        land = terrain.Terrain(self.game_data, menus.MAIN_MENU_MAP)
        column = [o for o in world.map_objects(self.game_data.map_mob(menus.MAIN_MENU_MAP))
                  if o.nid == menus.MAIN_MENU_COLUMN_NID][0]
        self.assertIn(column.template, menus.COLUMN_MODELS)
        x, y, z = column.position
        matrix = prop.placement_matrix((x, y, land.height_at(x, y) + z), column.rotation)
        model = figure.FigureModel(self.game_data, column.template)
        column_prop = prop.Prop(model, matrix)
        self.assertTrue(column_prop.idle())
        pose = column_prop.pose()
        rects = self.game_data.registry(menus.MENUS_REG)[menus.MAIN_MENU]
        stones = ('but01', 'but02', 'but08', 'but04', 'but06', 'but03')
        for button, stone, special in zip(menus.MAIN_MENU_BUTTONS, stones, menus.MAIN_MENU_SPECIALS):
            position, rotation = pose[stone]
            center = position + quat.to_matrix(rotation) @ model.figure(stone).center[0]
            clip = projection @ view @ matrix @ np.append(center, 1.0)
            screen_x = (clip[0] / clip[3] + 1.0) * 400.0
            screen_y = (1.0 - clip[1] / clip[3]) * 300.0
            left, top, right, bottom = rects[button]
            self.assertTrue(left <= screen_x < right and top <= screen_y < bottom, (button, screen_x, screen_y))
            self.assertEqual(model.texture_group(stone), figure.TEXTURE_SECOND)
            self.assertIsNotNone(column_prop.special_action(special), button)

    def test_every_map_header(self):
        for map_name in self.game_data.map_names():
            header = self.game_data.map_header(map_name)
            self.assertGreater(header.sectors_x, 0)
            self.assertTrue(math.isfinite(header.max_altitude))


if __name__ == '__main__':
    unittest.main()
