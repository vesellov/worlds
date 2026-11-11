"""
Decoder of the .mmp textures: DXT1, DXT3, PNT3 (a simple run length packing of 32 bit pixels)
and raw 16/32 bit pixels described by channel masks. Only the biggest mip level is decoded.

The rows are returned in the order of the file: uploaded to OpenGL the same way, the texture coordinates
of the models work without flipping.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import struct

import numpy as np

#------------------------------------------------------------------------------

MAGIC = b'MMP\0'

DATA_OFFSET = 76

PNT3_MAX_RUN = 1000000

#------------------------------------------------------------------------------


def _expand_565(c):
    r = (c >> 11) & 31
    g = (c >> 5) & 63
    b = c & 31
    return np.stack([
        (r * 255 + 15) // 31,
        (g * 255 + 31) // 63,
        (b * 255 + 15) // 31,
    ], axis=-1).astype(np.int32)


def _decode_dxt(data, width, height, dxt3):
    bw, bh = max(1, width // 4), max(1, height // 4)
    block_size = 16 if dxt3 else 8
    raw = np.frombuffer(data, dtype=np.uint8, count=bw * bh * block_size, offset=DATA_OFFSET).reshape(bh * bw, block_size)
    alpha = None
    if dxt3:
        alpha_rows = raw[:, 0:8].copy().view('<u2').astype(np.int32)
        shifts = np.arange(4) * 4
        alpha = ((alpha_rows[:, :, None] >> shifts) & 15) * 17
        raw = raw[:, 8:]
    colors = raw[:, 0:4].copy().view('<u2').astype(np.int32)
    c0, c1 = colors[:, 0], colors[:, 1]
    p0, p1 = _expand_565(c0), _expand_565(c1)
    palette = np.zeros((len(raw), 4, 4), dtype=np.int32)
    palette[:, 0, :3] = p0
    palette[:, 1, :3] = p1
    palette[:, 0:2, 3] = 255
    four = (c0 > c1) | dxt3
    palette[:, 2, :3] = np.where(four[:, None], (2 * p0 + p1) // 3, (p0 + p1) // 2)
    palette[:, 2, 3] = 255
    palette[:, 3, :3] = np.where(four[:, None], (p0 + 2 * p1) // 3, 0)
    palette[:, 3, 3] = np.where(four, 255, 0)
    rows = raw[:, 4:8].astype(np.int32)
    shifts = np.arange(4) * 2
    indexes = (rows[:, :, None] >> shifts) & 3
    pixels = np.take_along_axis(palette[:, None, :, :], indexes[:, :, :, None], axis=2)
    pixels = pixels.reshape(len(raw), 4, 4, 4)
    if alpha is not None:
        pixels[:, :, :, 3] = alpha
    image = pixels.reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
    return image[:height, :width].astype(np.uint8)


def _decode_pnt3(data, width, height, packed_size):
    tokens = np.frombuffer(data, dtype='<u4', count=packed_size // 4, offset=DATA_OFFSET)
    is_run = (tokens != 0) & (tokens <= PNT3_MAX_RUN)
    lengths = np.where(is_run, tokens, 4).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    total = max(int(lengths.sum()), width * height * 4)
    out = np.zeros(total, dtype=np.uint8)
    literal = ~is_run
    literal_bytes = tokens[literal].view(np.uint8).reshape(-1, 4)
    positions = starts[literal][:, None] + np.arange(4)
    out[positions.ravel()] = literal_bytes.ravel()
    bgra = out[:width * height * 4].reshape(height, width, 4)
    return bgra[:, :, [2, 1, 0, 3]].copy()


def _channel(pixels, mask, shift):
    if mask == 0:
        return np.full(pixels.shape, 255, dtype=np.uint8)
    top = mask >> shift
    return (((pixels & mask) >> shift) * 255 + top // 2) // top


def _decode_raw(data, width, height):
    bits = struct.unpack_from('<I', data, 20)[0]
    masks = struct.unpack_from('<12I', data, 24)
    a_mask, a_shift, _ = masks[0:3]
    r_mask, r_shift, _ = masks[3:6]
    g_mask, g_shift, _ = masks[6:9]
    b_mask, b_shift, _ = masks[9:12]
    dtype = '<u2' if bits == 16 else '<u4'
    pixels = np.frombuffer(data, dtype=dtype, count=width * height, offset=DATA_OFFSET).astype(np.int64).reshape(height, width)
    return np.stack([
        _channel(pixels, r_mask, r_shift),
        _channel(pixels, g_mask, g_shift),
        _channel(pixels, b_mask, b_shift),
        _channel(pixels, a_mask, a_shift),
    ], axis=-1).astype(np.uint8)


def decode(data):
    """
    (height, width, 4) uint8 RGBA.
    """
    data = bytes(data)
    if data[:4] != MAGIC:
        raise ValueError('not an .mmp file')
    width, height, mips_or_size = struct.unpack_from('<III', data, 4)
    form = data[16:20]
    if form == b'DXT1':
        return _decode_dxt(data, width, height, dxt3=False)
    if form == b'DXT3':
        return _decode_dxt(data, width, height, dxt3=True)
    if form == b'PNT3':
        return _decode_pnt3(data, width, height, mips_or_size)
    return _decode_raw(data, width, height)
