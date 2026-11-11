"""
Reader of the .anm animation tracks of one part.

The layout is the same as CAnimationTrack::Load (game.exe 0x50aaf0) reads it:
rotation keys (w, x, y, z), translation keys (x, y, z) and the frames of the morphing of the vertices.
In figures.res "<model>.anm" is an archive of clips and every clip is an archive of tracks, one per part.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import struct

import numpy as np

#------------------------------------------------------------------------------


class Track(object):

    def __init__(self, rotations, translations, morph_frames):
        # (keys, 4) w, x, y, z
        self.rotations = rotations
        # (keys, 3)
        self.translations = translations
        # (frames, vertices, 3) or None
        self.morph_frames = morph_frames

    def __repr__(self):
        return 'Track(%d rotations, %d translations, %s morph frames)' % (
            len(self.rotations), len(self.translations),
            0 if self.morph_frames is None else len(self.morph_frames))

#------------------------------------------------------------------------------


def parse(data):
    data = bytes(data)
    offset = 0
    rotations_count = struct.unpack_from('<I', data, offset)[0]
    offset += 4
    rotations = np.frombuffer(data, dtype='<f4', count=rotations_count * 4, offset=offset).reshape(rotations_count, 4)
    offset += rotations_count * 16
    translations_count = struct.unpack_from('<I', data, offset)[0]
    offset += 4
    translations = np.frombuffer(data, dtype='<f4', count=translations_count * 3, offset=offset).reshape(translations_count, 3)
    offset += translations_count * 12
    morph_frames = None
    if offset + 8 <= len(data):
        frames_count, vertices_count = struct.unpack_from('<II', data, offset)
        offset += 8
        if frames_count and vertices_count:
            count = frames_count * vertices_count * 3
            morph_frames = np.frombuffer(data, dtype='<f4', count=count, offset=offset).reshape(
                frames_count, vertices_count, 3).astype(np.float64)
    return Track(rotations.astype(np.float64), translations.astype(np.float64), morph_frames)
