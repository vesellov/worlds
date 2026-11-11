"""
Reader of the .fig model parts (magic "FIG8").

Every part keeps 8 variants of its vertices: the corners of a cube of the body proportions,
see trilinear(). The vertices are stored in blocks of 4, the normals too (x, y, z, w of 4 normals per block).
A triangle corner is a "vertex component": a vertex, a normal and a texture coordinate.

A few loose figures (in25arrow, the arrow of the mini-map) keep an older layout without the magic,
read by game.exe 4f87d0: two numbers, 6 variants of the bounds, then every array after its own count:
vertex blocks, normal blocks, texture coordinates, 52 byte records (skipped by game.exe),
vertex components, morph components and the triangle indexes.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import struct

import numpy as np

#------------------------------------------------------------------------------

HEADER_SIZE = 40

OLD_VARIANTS = 6
OLD_HEADER_SIZE = 8
OLD_SKIPPED_RECORD_SIZE = 52

#------------------------------------------------------------------------------


class Figure(object):

    def __init__(self):
        self.variants_count = 0
        self.group = 0
        self.texture_number = 0
        # the center of the box of every variant and the offsets of its corners from the center
        self.center = None
        self.min = None
        self.max = None
        self.radius = None
        # (variants, vertices, 3)
        self.positions = None
        # (normals, 3)
        self.normals = None
        # (texcoords, 2)
        self.texcoords = None
        # (triangles, 3) indexes of the vertex components
        self.triangles = None
        # (components, ) indexes of the position, normal and texcoord of every vertex component
        self.component_position = None
        self.component_normal = None
        self.component_texcoord = None
        # (morph components, 2)
        self.morph_components = None

    def __repr__(self):
        return 'Figure(%d variants, %d vertices, %d triangles)' % (
            self.variants_count, self.positions.shape[1], len(self.triangles))

    def variant_positions(self, coefs):
        """
        Vertices of the body with proportions coefs = (c0, c1, c2), each from 0 to 1.
        """
        if self.variants_count == 8:
            return trilinear(self.positions, coefs)
        if self.variants_count == OLD_VARIANTS:
            return six_variants(self.positions, coefs)
        return self.positions[0]

    def variant_bounds(self, coefs):
        """
        (min, max) corners of the box of the body with the proportions coefs.
        The header keeps the center of the box and the offsets of its corners from the center.
        """
        if self.variants_count == 8:
            blend = trilinear
        elif self.variants_count == OLD_VARIANTS:
            blend = six_variants
        else:
            return self.center[0] + self.min[0], self.center[0] + self.max[0]
        center = blend(self.center, coefs)
        return center + blend(self.min, coefs), center + blend(self.max, coefs)

    def mesh_arrays(self, coefs=(0.0, 0.0, 0.0), scale=1.0):
        """
        Flat vertex array for the renderer: (components, 8) float32 with x, y, z, nx, ny, nz, u, v
        and the triangle indexes as uint16.
        """
        positions = self.variant_positions(coefs) * scale
        out = np.empty((len(self.component_position), 8), dtype=np.float32)
        out[:, 0:3] = positions[self.component_position]
        out[:, 3:6] = self.normals[self.component_normal]
        out[:, 6:8] = self.texcoords[self.component_texcoord]
        return out, self.triangles.astype(np.uint16).ravel()

#------------------------------------------------------------------------------


def trilinear(values, coefs):
    """
    Interpolation between the 8 variants along the first axis of values,
    the order of the variants is index = 4 * c2 + 2 * c0 + c1.
    """
    c0, c1, c2 = coefs
    a = values[0] + (values[1] - values[0]) * c1
    b = values[2] + (values[3] - values[2]) * c1
    low = a + (b - a) * c0
    a = values[4] + (values[5] - values[4]) * c1
    b = values[6] + (values[7] - values[6]) * c1
    high = a + (b - a) * c0
    return low + (high - low) * c2


def six_variants(values, coefs):
    """
    4f87d0: the variants 0, 1, 2 and 3, 4, 5 are mixed by c0 then c1, the two results by c2.
    """
    c0, c1, c2 = coefs
    low = values[0] + (values[1] - values[0]) * c0
    low = low + (values[2] - low) * c1
    high = values[3] + (values[4] - values[3]) * c0
    high = high + (values[5] - high) * c1
    return low + (high - low) * c2


def parse_old(data):
    """
    The layout without the magic, see the docstring of the module.
    """
    data = bytes(data)
    n = OLD_VARIANTS
    fig = Figure()
    fig.variants_count = n
    offset = OLD_HEADER_SIZE

    def _count(offset):
        return struct.unpack_from('<I', data, offset)[0], offset + 4

    def _floats(offset, shape):
        count = int(np.prod(shape))
        values = np.frombuffer(data, dtype='<f4', count=count, offset=offset).reshape(shape)
        return values.astype(np.float32), offset + 4 * count

    def _ushorts(offset, shape):
        count = int(np.prod(shape))
        values = np.frombuffer(data, dtype='<u2', count=count, offset=offset).reshape(shape)
        return values.astype(np.int64), offset + 2 * count

    fig.center, offset = _floats(offset, (n, 3))
    fig.min, offset = _floats(offset, (n, 3))
    fig.max, offset = _floats(offset, (n, 3))
    fig.radius, offset = _floats(offset, (n, ))
    vertex_blocks, offset = _count(offset)
    blocks, offset = _floats(offset, (vertex_blocks, 3, n, 4))
    fig.positions = np.ascontiguousarray(blocks.transpose(2, 0, 3, 1).reshape(n, vertex_blocks * 4, 3))
    normal_blocks, offset = _count(offset)
    normals, offset = _floats(offset, (normal_blocks, 4, 4))
    fig.normals = np.ascontiguousarray(normals.transpose(0, 2, 1).reshape(normal_blocks * 4, 4)[:, :3])
    texcoords_count, offset = _count(offset)
    fig.texcoords, offset = _floats(offset, (texcoords_count, 2))
    skipped, offset = _count(offset)
    offset += skipped * OLD_SKIPPED_RECORD_SIZE
    components_count, offset = _count(offset)
    components, offset = _ushorts(offset, (components_count, 3))
    fig.component_position = (components[:, 0] >> 2) * 4 + (components[:, 0] & 3)
    fig.component_normal = (components[:, 1] >> 2) * 4 + (components[:, 1] & 3)
    fig.component_texcoord = components[:, 2]
    morph_count, offset = _count(offset)
    fig.morph_components, offset = _ushorts(offset, (morph_count, 2))
    indexes_count, offset = _count(offset)
    fig.triangles, offset = _ushorts(offset, (indexes_count // 3, 3))
    return fig


def parse(data):
    data = bytes(data)
    if data[:3] != b'FIG':
        # 4f7e70: everything without the magic goes to the old reader
        return parse_old(data)
    fig = Figure()
    n = data[3] - ord('0')
    fig.variants_count = n
    (vertex_blocks, normal_blocks, texcoords_count, indexes_count,
     components_count, morph_count) = struct.unpack_from('<6I', data, 4)
    fig.group, fig.texture_number = struct.unpack_from('<2I', data, 32)
    offset = HEADER_SIZE

    def _floats(offset, shape):
        count = int(np.prod(shape))
        values = np.frombuffer(data, dtype='<f4', count=count, offset=offset).reshape(shape)
        return values.astype(np.float32), offset + 4 * count

    def _ushorts(offset, shape):
        count = int(np.prod(shape))
        values = np.frombuffer(data, dtype='<u2', count=count, offset=offset).reshape(shape)
        return values.astype(np.int64), offset + 2 * count

    fig.center, offset = _floats(offset, (n, 3))
    fig.min, offset = _floats(offset, (n, 3))
    fig.max, offset = _floats(offset, (n, 3))
    fig.radius, offset = _floats(offset, (n, ))
    # block, axis, variant, vertex of the block -> variant, vertex, axis
    blocks, offset = _floats(offset, (vertex_blocks, 3, n, 4))
    fig.positions = np.ascontiguousarray(blocks.transpose(2, 0, 3, 1).reshape(n, vertex_blocks * 4, 3))
    # block, axis (x, y, z, w), normal of the block -> normal, axis
    normals, offset = _floats(offset, (normal_blocks, 4, 4))
    fig.normals = np.ascontiguousarray(normals.transpose(0, 2, 1).reshape(normal_blocks * 4, 4)[:, :3])
    fig.texcoords, offset = _floats(offset, (texcoords_count, 2))
    fig.triangles, offset = _ushorts(offset, (indexes_count // 3, 3))
    components, offset = _ushorts(offset, (components_count, 3))
    fig.component_position = (components[:, 0] >> 2) * 4 + (components[:, 0] & 3)
    fig.component_normal = (components[:, 1] >> 2) * 4 + (components[:, 1] & 3)
    fig.component_texcoord = components[:, 2]
    fig.morph_components, offset = _ushorts(offset, (morph_count, 2))
    return fig
