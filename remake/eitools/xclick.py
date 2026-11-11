"""
Click or press keys in the X server by XTest: xclick.py click <x> <y> [button] | key <keysym name> | move <x> <y> | rel <dx> <dy> | relclick <dx> <dy>.
The coordinates are in the root window.
"""
import ctypes
import sys
import time

x11 = ctypes.cdll.LoadLibrary('libX11.so.6')
xtst = ctypes.cdll.LoadLibrary('libXtst.so.6')
x11.XOpenDisplay.restype = ctypes.c_void_p
x11.XFlush.argtypes = [ctypes.c_void_p]
x11.XStringToKeysym.restype = ctypes.c_ulong
x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
xtst.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
xtst.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
xtst.XTestFakeRelativeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
xtst.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
d = x11.XOpenDisplay(None)
cmd = sys.argv[1]
if cmd in ('click', 'move'):
    x, y = int(sys.argv[2]), int(sys.argv[3])
    xtst.XTestFakeMotionEvent(d, -1, x, y, 0)
    x11.XFlush(d)
    time.sleep(0.15)
    if cmd == 'click':
        button = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        xtst.XTestFakeButtonEvent(d, button, 1, 0)
        x11.XFlush(d)
        time.sleep(0.1)
        xtst.XTestFakeButtonEvent(d, button, 0, 0)
elif cmd in ('rel', 'relclick'):
    # moves by small steps, the game reads the deltas
    dx, dy = int(sys.argv[2]), int(sys.argv[3])
    steps = max(1, max(abs(dx), abs(dy)) // 20)
    for i in range(steps):
        xtst.XTestFakeRelativeMotionEvent(d, dx * (i + 1) // steps - dx * i // steps, dy * (i + 1) // steps - dy * i // steps, 0)
        x11.XFlush(d)
        time.sleep(0.01)
    if cmd == 'relclick':
        time.sleep(0.2)
        xtst.XTestFakeButtonEvent(d, 1, 1, 0)
        x11.XFlush(d)
        time.sleep(0.1)
        xtst.XTestFakeButtonEvent(d, 1, 0, 0)
elif cmd == 'key':
    code = x11.XKeysymToKeycode(d, x11.XStringToKeysym(sys.argv[2].encode()))
    xtst.XTestFakeKeyEvent(d, code, 1, 0)
    x11.XFlush(d)
    time.sleep(0.08)
    xtst.XTestFakeKeyEvent(d, code, 0, 0)
x11.XFlush(d)
