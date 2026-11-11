"""
Grab an X window by its id into a PNG: xgrab.py <window id> <file>.
"""
import ctypes
import sys

from PIL import Image

x11 = ctypes.cdll.LoadLibrary('libX11.so.6')
x11.XOpenDisplay.restype = ctypes.c_void_p
x11.XGetImage.restype = ctypes.c_void_p
x11.XGetImage.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
                          ctypes.c_ulong, ctypes.c_int]


class XImage(ctypes.Structure):
    _fields_ = [('width', ctypes.c_int), ('height', ctypes.c_int), ('xoffset', ctypes.c_int), ('format', ctypes.c_int),
                ('data', ctypes.c_void_p), ('byte_order', ctypes.c_int), ('bitmap_unit', ctypes.c_int),
                ('bitmap_bit_order', ctypes.c_int), ('bitmap_pad', ctypes.c_int), ('depth', ctypes.c_int),
                ('bytes_per_line', ctypes.c_int), ('bits_per_pixel', ctypes.c_int)]


d = x11.XOpenDisplay(None)
w = int(sys.argv[1], 16)
w_size = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (800, 600)
p = x11.XGetImage(d, w, 0, 0, w_size[0], w_size[1], 0xffffffff, 2)
if not p:
    sys.exit('XGetImage failed')
img = XImage.from_address(p)
raw = ctypes.string_at(img.data, img.bytes_per_line * img.height)
Image.frombuffer('RGB', (img.width, img.height), raw, 'raw', 'BGRX', img.bytes_per_line, 1).save(sys.argv[2])
print('saved', img.width, img.height, img.depth)
