"""
Reader of the .res archives of Evil Islands.

A .res file is a flat table of named entries. Several game files are .res archives too:
figures.res holds .mod/.anm/.bon entries which are archives themselves, a .mpr map is an archive
of one .mp header and many .sec sectors. Names are looked up without regard to the case, as the game does.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import mmap
import struct

#------------------------------------------------------------------------------

MAGIC = b'\x3C\xE2\x9C\x01'

ENCODING = 'cp1251'

ENTRY_FORMAT = '<IIIIHI'

ENTRY_SIZE = struct.calcsize(ENTRY_FORMAT)

#------------------------------------------------------------------------------


class ResArchive(object):
    """
    Entries of one archive: name -> (offset, size). The data is a bytes-like object, memory mapped for big files,
    reading an entry returns a memoryview of it without a copy.
    """

    def __init__(self, data, archive_name=''):
        self.data = memoryview(data)
        self.archive_name = archive_name
        self.names = []
        self.entries = {}
        self.parse()

    def __repr__(self):
        return 'ResArchive(%s, %d entries)' % (self.archive_name, len(self.names))

    def __contains__(self, name):
        return name.lower() in self.entries

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)

    @classmethod
    def open(cls, file_path):
        with open(file_path, 'rb') as f:
            data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return cls(data, archive_name=file_path)

    @staticmethod
    def is_archive(data):
        return bytes(data[:4]) == MAGIC

    def parse(self):
        if not self.is_archive(self.data):
            raise ValueError('not a .res archive: %s' % self.archive_name)
        count, table_offset, _names_length = struct.unpack_from('<III', self.data, 4)
        names_offset = table_offset + count * ENTRY_SIZE
        for i in range(count):
            _next, size, offset, _timestamp, name_length, name_offset = struct.unpack_from(
                ENTRY_FORMAT, self.data, table_offset + i * ENTRY_SIZE)
            start = names_offset + name_offset
            name = bytes(self.data[start:start + name_length]).decode(ENCODING).replace('\\', '/')
            self.names.append(name)
            self.entries[name.lower()] = (offset, size)

    def read(self, name):
        offset, size = self.entries[name.lower()]
        return self.data[offset:offset + size]

    def get(self, name, default=None):
        if name.lower() not in self.entries:
            return default
        return self.read(name)

    def archive(self, name):
        return ResArchive(self.read(name), archive_name='%s/%s' % (self.archive_name, name))

    def names_with_suffix(self, suffix):
        suffix = suffix.lower()
        return [name for name in self.names if name.lower().endswith(suffix)]
