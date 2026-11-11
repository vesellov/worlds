"""
Reader of the camera paths of camera/*.cam: game.exe flies the camera along them in the cut scenes,
MainMenu.cam holds the camera of the main menu.

The file is a list of 36 byte keys: u32 time, u32 (an id, the same in all keys of a file),
3 floats of the position and the rotation quaternion (w, x, y, z). The camera looks along
the +z axis of its rotation and the +y axis goes down on the screen, as in Direct3D.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import struct

#------------------------------------------------------------------------------

KEY_SIZE = 36

#------------------------------------------------------------------------------


class CameraKey(object):

    def __init__(self, time, position, rotation):
        self.time = time
        self.position = position
        self.rotation = rotation

    def __repr__(self):
        return 'CameraKey(%d at %s)' % (self.time, tuple(round(v, 2) for v in self.position))

#------------------------------------------------------------------------------


def parse(data):
    data = bytes(data)
    keys = []
    for offset in range(0, len(data) - KEY_SIZE + 1, KEY_SIZE):
        values = struct.unpack_from('<2I3f4f', data, offset)
        keys.append(CameraKey(values[0], values[2:5], values[5:9]))
    if _Debug:
        print('.cam: %d keys' % len(keys))
    return keys
