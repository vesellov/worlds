"""
Readers of the skeleton files: .lnk (the tree of the parts of a model) and .bon (the offset of a part
from its parent, one for every body variant). Both are entries of figures.res and of the .mod archives.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import struct

import numpy as np

#------------------------------------------------------------------------------

ENCODING = 'cp1251'

#------------------------------------------------------------------------------


class PartsTree(object):
    """
    Parts in the order of the file: a parent always comes before its children.
    """

    def __init__(self):
        self.order = []
        self.parents = {}
        self.children = {}
        self.root = None

    def __repr__(self):
        return 'PartsTree(root=%s, %d parts)' % (self.root, len(self.order))

    def add(self, name, parent_name):
        self.order.append(name)
        self.parents[name] = parent_name
        self.children.setdefault(name, [])
        if parent_name:
            self.children.setdefault(parent_name, []).append(name)
        else:
            self.root = name

#------------------------------------------------------------------------------


def _string(data, offset):
    length = struct.unpack_from('<I', data, offset)[0]
    raw = bytes(data[offset + 4:offset + 4 + length])
    return raw.split(b'\0', 1)[0].decode(ENCODING), offset + 4 + length


def parse_lnk(data):
    tree = PartsTree()
    count = struct.unpack_from('<I', data, 0)[0]
    offset = 4
    for _ in range(count):
        name, offset = _string(data, offset)
        parent_name, offset = _string(data, offset)
        tree.add(name, parent_name or None)
    return tree


def parse_bon(data):
    """
    (variants, 3) offsets of the part.
    """
    values = np.frombuffer(bytes(data), dtype='<f4')
    return values.reshape(len(values) // 3, 3).astype(np.float32)
