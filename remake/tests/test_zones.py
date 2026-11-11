"""
Tests of eilib.zones: the zones and the "#allod" blocks of map.txt.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from eilib import zones

#------------------------------------------------------------------------------

MAP_TXT = """#allod gipat
ingm2arrow
ingm2pointball
ingm2scull
ingm1down
ingm2armor

#allod ingos
ingm5arrow
ingm5pointball
// a comment

#zone gz1g gipat field
#res
zone1.mpr zone1.mob
#sky
normal
#weather
snow
#exit 1
none 0
#deploy
1 2 3 4
#view
90
"""

#------------------------------------------------------------------------------


class ZonesTest(unittest.TestCase):

    def test_allods(self):
        allods = zones.parse_allods(MAP_TXT)
        self.assertEqual([allod.name for allod in allods], ['gipat', 'ingos'])
        self.assertEqual(allods[0].model_names,
                         ['ingm2arrow', 'ingm2pointball', 'ingm2scull', 'ingm1down', 'ingm2armor'])
        self.assertEqual(allods[1].model_names, ['ingm5arrow', 'ingm5pointball'])

    def test_zones(self):
        parsed = zones.parse(MAP_TXT)
        self.assertEqual(len(parsed), 1)
        zone = parsed[0]
        self.assertEqual((zone.name, zone.allod, zone.type), ('gz1g', 'gipat', 'field'))
        self.assertEqual((zone.mpr_name, zone.mob_name), ('zone1.mpr', 'zone1.mob'))
        self.assertEqual(zone.weather, 'snow')
        self.assertEqual(len(zone.exits), 1)
        self.assertEqual(zone.exits[0].deploy, (1.0, 2.0, 3.0, 4.0))

    def test_weather_and_lights(self):
        zone = zones.Zone('gz1g', 'gipat', 'field')
        self.assertEqual(zone.weather, 'rain')
        self.assertFalse(zone.is_cave)
        self.assertEqual(zone.lights_file_name(), 'lightsgipat.ini')
        zone.sky = zones.SKY_CAVE
        self.assertEqual(zone.lights_file_name(), 'lightscavegipat.ini')

    def test_walk_exit(self):
        zone = zones.Zone('gz1g', 'gipat', 'field')
        exit_ = zones.Exit(2)
        exit_.remove = (10.0, 20.0, 30.0, 40.0)
        exit_.neighbour_name = 'bz1g'
        exit_.neighbour_exit = 1
        zone.exits.append(exit_)
        self.assertTrue(exit_.contains_remove(20.0, 30.0))
        self.assertFalse(exit_.contains_remove(5.0, 30.0))
        self.assertIs(zones.remove_exit(zone, 30.0, 40.0), exit_)
        self.assertIsNone(zones.remove_exit(zone, 31.0, 40.0))
        self.assertIsNone(zones.remove_exit(None, 20.0, 30.0))


if __name__ == '__main__':
    unittest.main()
