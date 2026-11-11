"""
The storm of the main menu (632ad0).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from client import weather

#------------------------------------------------------------------------------


class StormTest(unittest.TestCase):

    def test_storm_numbers(self):
        self.assertEqual(weather.STORM_HEIGHT, 30.0)
        self.assertEqual(weather.STORM_ROLL, 4)
        self.assertEqual(weather.STORM_CHANCE, 41)


if __name__ == '__main__':
    unittest.main()
