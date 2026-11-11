#!/usr/bin/env python3
"""
Drive the original game and the remake demo from outside, on private Xvfb displays.

Two targets:

    original   EvilIslands1 in its own Wine prefix (start.sh), display :77
    demo       remake (make demo), display :78

Both processes run on their own virtual X server, so the mouse and the keyboard of the desktop are
never touched. The input goes to the target window by XTest on that virtual server; the original game
can also be driven by SendInput inside its Wine (the eiinput.py of C:\\eitools). The pictures are
XGetImage of the target window, so they do not need a compositor or a visible window.

"compare --screen" runs both targets on the display of the desktop instead: the original on the left,
the demo on the right, and the scenario on both at once, so the whole run is visible. The input then
goes by XWarpPointer and XSendEvent (backend xsend), because XWayland ignores XTest events of other
clients.

Usage:

    eidrive.py original start|stop|restart|status
    eidrive.py demo start|stop|restart|status
    eidrive.py <target> shot FILE
    eidrive.py <target> record DIR SECONDS [FPS]
    eidrive.py <target> input move X Y | click X Y [B] | rclick X Y | dclick X Y | drag X1 Y1 X2 Y2 [B]
                        key NAME [MS] | chord NAME NAME [MS] | keydown NAME | keyup NAME
                        rel DX DY [STEPS] | wheel up|down [N] | type TEXT | wait MS
    eidrive.py <target> run SCENARIO [--out DIR]
    eidrive.py compare SCENARIO [--out DIR] [--no-restart] [--screen [--display :0]]
                                [--keep] [--finish-timeout SECONDS]

A scenario is a text file, one command per line (the same commands as "input", plus):

    # comment
    timeout MS                                   the whole run; when it passes the targets are stopped
                                                 and the run fails (required in the top-level scenario)
    sync                                         wait here for the other target of the comparison
    original: ...     run the line only for the original game
    demo: ...         run the line only for the demo
    wait MS
    waitstable SETTLE_MS TIMEOUT_MS [POLL_MS]   wait until the picture stops changing
    waitpixel X Y R,G,B [TOL] [TIMEOUT_MS]       wait until the pixel has the color
    waitregion X Y W H R,G,B [TOL] [FRACTION] [TIMEOUT_MS]
                                                 wait until the region has the color
    shot NAME                                    save <out>/NAME.png
    expect_changed NAME [THRESHOLD]              the picture must differ from <out>/NAME.png
    expect_same NAME [THRESHOLD]                 the picture must match <out>/NAME.png
    note TEXT
    include FILE                                  run another scenario from this folder
    repeat N COMMAND...                           the command N times
    expect_closed [TIMEOUT_MS]                    the game window must close (the game quits)
    reset_settings                                compare resets the settings of both games first
    put_save FIXTURE SLOT                         the folder eitools/fixtures/saves/FIXTURE (a save of game.exe)
                                                  becomes data/saves/SLOT of the target, the old slot is removed

Settings: "eidrive.py original|demo reset-settings" (or "compare --reset-settings", or a
"reset_settings" line in the scenario) stops the game, copies its settings to
eitools/run/settings_backup/ and removes them, so the game starts with its defaults:
the original loses the Wine registry key Software\\Nival Interactive\\EvilIslands and gets
eitools/defaults/keyboard.ini back (without the file game.exe loses the action keys: H, Z, X...),
the demo loses ~/.cache/ei-remake/settings.json.

The coordinates of the mouse are the client coordinates of the game window (800 x 600).

Environment:

    EIDRIVE_HOME               the folder with EvilIslands1 and remake, default the parent of eitools
    EIDRIVE_ORIGINAL_DISPLAY   default :77 (EI_DISPLAY is read too)
    EIDRIVE_DEMO_DISPLAY       default :78
    EIDRIVE_ORIGINAL_BACKEND   xtest (default) or sendinput
    EIDRIVE_PHASE_FILE         the "phase NAME" lines of scenarios write NAME there (eitrace_phase.txt)
    EIDRIVE_ORIGINAL_SCRIPT    the launcher in EvilIslands1, default start.sh (trace.sh: under eitrace)
    EIDRIVE_DEMO_BACKEND       xtest (default)
    EIDRIVE_DATA_PATH          game data for the demo, default EvilIslands1/game_info/data of this copy
    EIDRIVE_DEMO_AUDIODRIVER   passed as SDL_AUDIODRIVER (dummy makes the demo silent)
"""

#------------------------------------------------------------------------------

import argparse
import ctypes
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

try:
    from PIL import Image
    from PIL import ImageChops
except ImportError:
    sys.exit('eidrive.py needs Pillow: sudo apt install python3-pil')

#------------------------------------------------------------------------------

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
EI_HOME = os.environ.get('EIDRIVE_HOME') or os.path.dirname(TOOLS_DIR)
RUN_DIR = os.path.join(TOOLS_DIR, 'run')
OUT_DIR = os.path.join(TOOLS_DIR, 'out')
# the saves of game.exe which the scenarios put into the slots (put_save)
FIXTURE_SAVES_DIR = os.path.join(TOOLS_DIR, 'fixtures', 'saves')
ORIGINAL_DIR = os.path.join(EI_HOME, 'EvilIslands1')
REMAKE_DIR = os.path.join(EI_HOME, 'remake')
SETTINGS_BACKUP_DIR = os.path.join(RUN_DIR, 'settings_backup')
# the settings of game.exe: a key of the Wine registry (every subkey of it) and the hotkeys file
ORIGINAL_REGISTRY_KEY = 'Software\\\\Nival Interactive\\\\EvilIslands'
ORIGINAL_KEYBOARD_PATH = os.path.join('game_info', 'data', 'config', 'keyboard.ini')
# the default hotkeys of game.exe, put back by reset-settings: without the file game.exe binds no actions
DEFAULT_KEYBOARD_PATH = os.path.join(TOOLS_DIR, 'defaults', 'keyboard.ini')
DEMO_SETTINGS_PATH = os.path.join(os.path.expanduser('~'), '.cache', 'ei-remake', 'settings.json')

#------------------------------------------------------------------------------

DISPLAYS = {
    'original': os.environ.get('EIDRIVE_ORIGINAL_DISPLAY') or os.environ.get('EI_DISPLAY') or ':77',
    'demo': os.environ.get('EIDRIVE_DEMO_DISPLAY') or ':78',
}
TITLES = {'original': 'Evil Islands', 'demo': 'Evil Islands remake'}
EXACT_TITLES = {'original': True, 'demo': False}

# the biggest window whose title has the part; xwininfo prints the absolute position at the end
WINDOW_RE = re.compile(r'^\s*(0x[0-9a-fA-F]+)\s+"([^"]*)".*?\s(\d+)x(\d+)\+(-?\d+)\+(-?\d+)'
                       r'\s+\+(-?\d+)\+(-?\d+)\s*$')

#------------------------------------------------------------------------------


class ToolError(Exception):
    pass


class ScenarioTimeout(Exception):
    """
    The scenario ran longer than its timeout, or the other target of the comparison stopped.
    """
    pass


class RunAbort(object):
    """
    The stop of all the runs of one comparison: the first timeout stops every target.
    """

    def __init__(self):
        self.event = threading.Event()
        self.reason = None

    def set(self, reason):
        if self.reason is None:
            self.reason = reason
        self.event.set()

    def check(self):
        if self.event.is_set():
            raise ScenarioTimeout(self.reason or 'the run was stopped')


class RunBarrier(object):
    """
    The "sync" line of a scenario: every run of the comparison waits for the others here.
    """

    def __init__(self, parties):
        self.parties = parties
        self.arrived = 0
        self.generation = 0
        self.condition = threading.Condition()

    def wait(self, abort, check_timeout):
        with self.condition:
            generation = self.generation
            self.arrived += 1
            if self.arrived >= self.parties:
                self.generation += 1
                self.arrived = 0
                self.condition.notify_all()
                return
            while self.generation == generation:
                abort.check()
                check_timeout()
                self.condition.wait(0.1)


def log(text):
    print(text, flush=True)


def sleep_ms(ms):
    if ms > 0:
        time.sleep(ms / 1000.0)

#------------------------------------------------------------------------------
# the input of two targets which share one display goes one command at a time
#------------------------------------------------------------------------------

_INPUT_LOCKS = {}
_INPUT_LOCKS_GUARD = threading.Lock()


def input_lock(display):
    with _INPUT_LOCKS_GUARD:
        lock = _INPUT_LOCKS.get(display)
        if lock is None:
            lock = threading.RLock()
            _INPUT_LOCKS[display] = lock
        return lock

#------------------------------------------------------------------------------
# Xlib: display, input by XTest, pictures by XGetImage
#------------------------------------------------------------------------------

X11 = ctypes.cdll.LoadLibrary('libX11.so.6')
XTST = ctypes.cdll.LoadLibrary('libXtst.so.6')



class XErrorEvent(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('display', ctypes.c_void_p), ('resourceid', ctypes.c_ulong),
                ('serial', ctypes.c_ulong), ('error_code', ctypes.c_ubyte),
                ('request_code', ctypes.c_ubyte), ('minor_code', ctypes.c_ubyte)]


X_ERROR_HANDLER_TYPE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(XErrorEvent))


def _x_error(dpy, event):
    """
    The default handler of Xlib ends the process: a window which closes under a step (the game
    after EXIT) must fail that step only. XGetImage returns NULL then and grab() raises ToolError.
    """
    error = event.contents
    log('X error %d of request %d on 0x%x' % (error.error_code, error.request_code, error.resourceid))
    return 0


# kept in a global: ctypes must not free the callback while Xlib holds it
X_ERROR_HANDLER = X_ERROR_HANDLER_TYPE(_x_error)
X11.XSetErrorHandler.restype = ctypes.c_void_p
X11.XSetErrorHandler.argtypes = [X_ERROR_HANDLER_TYPE]
X11.XSetErrorHandler(X_ERROR_HANDLER)

X11.XOpenDisplay.restype = ctypes.c_void_p
X11.XOpenDisplay.argtypes = [ctypes.c_char_p]
X11.XCloseDisplay.argtypes = [ctypes.c_void_p]
X11.XFlush.argtypes = [ctypes.c_void_p]
X11.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
X11.XStringToKeysym.restype = ctypes.c_ulong
X11.XStringToKeysym.argtypes = [ctypes.c_char_p]
X11.XKeysymToKeycode.restype = ctypes.c_ubyte
X11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
X11.XSetInputFocus.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
X11.XGetImage.restype = ctypes.c_void_p
X11.XGetImage.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_uint,
                          ctypes.c_uint, ctypes.c_ulong, ctypes.c_int]
X11.XMoveWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int]
X11.XRaiseWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
X11.XDefaultScreen.restype = ctypes.c_int
X11.XDefaultScreen.argtypes = [ctypes.c_void_p]
X11.XDisplayWidth.restype = ctypes.c_int
X11.XDisplayWidth.argtypes = [ctypes.c_void_p, ctypes.c_int]
X11.XDisplayHeight.restype = ctypes.c_int
X11.XDisplayHeight.argtypes = [ctypes.c_void_p, ctypes.c_int]
X11.XDefaultRootWindow.restype = ctypes.c_ulong
X11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
X11.XQueryTree.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
                           ctypes.POINTER(ctypes.c_ulong),
                           ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
                           ctypes.POINTER(ctypes.c_uint)]
X11.XFree.argtypes = [ctypes.c_void_p]
X11.XGetGeometry.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
                             ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                             ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint),
                             ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
X11.XTranslateCoordinates.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong,
                                      ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                      ctypes.POINTER(ctypes.c_int),
                                      ctypes.POINTER(ctypes.c_ulong)]
X11.XInternAtom.restype = ctypes.c_ulong
X11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
X11.XSendEvent.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_long,
                           ctypes.c_void_p]
X11.XWarpPointer.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int,
                             ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_int,
                             ctypes.c_int]
X11.XQueryPointer.restype = ctypes.c_int
X11.XQueryPointer.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
                              ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int),
                              ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                              ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint)]
try:
    X11.XDestroyImage.argtypes = [ctypes.c_void_p]
except AttributeError:
    X11.XDestroyImage = None

# the windows of the two targets may share one display, so the clients use Xlib from two threads
X11.XInitThreads()

XTST.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_ulong]
XTST.XTestFakeRelativeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                              ctypes.c_ulong]
XTST.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
XTST.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]

ALL_PLANES = 0xffffffff
Z_PIXMAP = 2

# the names of the keys of the scenarios and of keyboard.ini -> the names of X keysyms
KEYSYM_NAMES = {
    'esc': 'Escape', 'escape': 'Escape', 'enter': 'Return', 'return': 'Return',
    'space': 'space', 'spacebar': 'space', 'tab': 'Tab', 'backspace': 'BackSpace',
    'shift': 'Shift_L', 'lshift': 'Shift_L', 'rshift': 'Shift_R',
    'ctrl': 'Control_L', 'control': 'Control_L', 'lctrl': 'Control_L', 'rctrl': 'Control_R',
    'alt': 'Alt_L', 'lalt': 'Alt_L', 'ralt': 'Alt_R',
    'up': 'Up', 'down': 'Down', 'left': 'Left', 'right': 'Right',
    'home': 'Home', 'end': 'End', 'pageup': 'Prior', 'pgup': 'Prior',
    'pagedown': 'Next', 'pgdn': 'Next', 'insert': 'Insert', 'ins': 'Insert',
    'delete': 'Delete', 'del': 'Delete', 'minus': 'minus', 'plus': 'plus', 'equal': 'equal',
    'apostrophe': 'apostrophe', 'quote': 'apostrophe', 'comma': 'comma', 'period': 'period',
    'dot': 'period', 'slash': 'slash', 'backslash': 'backslash', 'semicolon': 'semicolon',
    'bracketleft': 'bracketleft', 'bracketright': 'bracketright', 'grave': 'grave',
    'kp_up': 'KP_Up', 'kp_down': 'KP_Down', 'kp_left': 'KP_Left', 'kp_right': 'KP_Right',
    'kp_home': 'KP_Home', 'kp_end': 'KP_End', 'kp_pgup': 'KP_Prior', 'kp_pgdn': 'KP_Next',
    'kp_center': 'KP_Begin', 'kp_enter': 'KP_Enter', 'kp_plus': 'KP_Add',
    'kp_minus': 'KP_Subtract', 'kp_multiply': 'KP_Multiply', 'kp_divide': 'KP_Divide',
}
for _i in range(1, 13):
    KEYSYM_NAMES['f%d' % _i] = 'F%d' % _i

BUTTONS = {'left': 1, 'middle': 2, 'right': 3, '1': 1, '2': 2, '3': 3}

# the state bits of the X key and button events
MODIFIER_MASKS = {
    'shift': 1, 'lshift': 1, 'rshift': 1,
    'ctrl': 4, 'control': 4, 'lctrl': 4, 'rctrl': 4,
    'alt': 8, 'lalt': 8, 'ralt': 8,
}
# Button1Mask .. Button5Mask of the state of the X events
BUTTON_MASKS = {1: 1 << 8, 2: 1 << 9, 3: 1 << 10, 4: 1 << 11, 5: 1 << 12}


class XImage(ctypes.Structure):
    _fields_ = [('width', ctypes.c_int), ('height', ctypes.c_int), ('xoffset', ctypes.c_int),
                ('format', ctypes.c_int), ('data', ctypes.c_void_p), ('byte_order', ctypes.c_int),
                ('bitmap_unit', ctypes.c_int), ('bitmap_bit_order', ctypes.c_int),
                ('bitmap_pad', ctypes.c_int), ('depth', ctypes.c_int),
                ('bytes_per_line', ctypes.c_int), ('bits_per_pixel', ctypes.c_int)]


class XClientMessage(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('serial', ctypes.c_ulong), ('send_event', ctypes.c_int),
                ('display', ctypes.c_void_p), ('window', ctypes.c_ulong),
                ('message_type', ctypes.c_ulong), ('format', ctypes.c_int),
                ('data', ctypes.c_long * 5)]


class XButtonEvent(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('serial', ctypes.c_ulong), ('send_event', ctypes.c_int),
                ('display', ctypes.c_void_p), ('window', ctypes.c_ulong), ('root', ctypes.c_ulong),
                ('subwindow', ctypes.c_ulong), ('time', ctypes.c_ulong), ('x', ctypes.c_int),
                ('y', ctypes.c_int), ('x_root', ctypes.c_int), ('y_root', ctypes.c_int),
                ('state', ctypes.c_uint), ('button', ctypes.c_uint), ('same_screen', ctypes.c_int)]


class XKeyEvent(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('serial', ctypes.c_ulong), ('send_event', ctypes.c_int),
                ('display', ctypes.c_void_p), ('window', ctypes.c_ulong), ('root', ctypes.c_ulong),
                ('subwindow', ctypes.c_ulong), ('time', ctypes.c_ulong), ('x', ctypes.c_int),
                ('y', ctypes.c_int), ('x_root', ctypes.c_int), ('y_root', ctypes.c_int),
                ('state', ctypes.c_uint), ('keycode', ctypes.c_uint), ('same_screen', ctypes.c_int)]


class XMotionEvent(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('serial', ctypes.c_ulong), ('send_event', ctypes.c_int),
                ('display', ctypes.c_void_p), ('window', ctypes.c_ulong), ('root', ctypes.c_ulong),
                ('subwindow', ctypes.c_ulong), ('time', ctypes.c_ulong), ('x', ctypes.c_int),
                ('y', ctypes.c_int), ('x_root', ctypes.c_int), ('y_root', ctypes.c_int),
                ('state', ctypes.c_uint), ('is_hint', ctypes.c_char), ('same_screen', ctypes.c_int)]

#------------------------------------------------------------------------------


def keysym_name(name):
    lower = name.lower()
    if lower in KEYSYM_NAMES:
        return KEYSYM_NAMES[lower]
    if len(name) == 1:
        return name
    return name


class XSession(object):
    """
    One connection to the X server of a target: the input of XTest and the pictures of XGetImage.
    """

    def __init__(self, display):
        self.display_name = display
        self.dpy = X11.XOpenDisplay(display.encode())
        if not self.dpy:
            raise ToolError('can not open display %s (start Xvfb with "eidrive.py <target> start")'
                            % display)

    def close(self):
        if self.dpy:
            X11.XCloseDisplay(self.dpy)
            self.dpy = None

    def sync(self):
        X11.XSync(self.dpy, False)

    def flush(self):
        X11.XFlush(self.dpy)

    def focus(self, window):
        X11.XSetInputFocus(self.dpy, window['id'], 1, 0)
        self.sync()

    def screen_size(self):
        screen = X11.XDefaultScreen(self.dpy)
        return X11.XDisplayWidth(self.dpy, screen), X11.XDisplayHeight(self.dpy, screen)

    def move_window(self, window, x, y):
        X11.XMoveWindow(self.dpy, window['id'], int(x), int(y))
        X11.XRaiseWindow(self.dpy, window['id'])
        self.sync()

    def root_window(self):
        return X11.XDefaultRootWindow(self.dpy)

    def query_tree(self, window_id):
        """
        The parent and the children of a window.
        """
        root = ctypes.c_ulong()
        parent = ctypes.c_ulong()
        children = ctypes.POINTER(ctypes.c_ulong)()
        count = ctypes.c_uint()
        X11.XQueryTree(self.dpy, window_id, ctypes.byref(root), ctypes.byref(parent),
                       ctypes.byref(children), ctypes.byref(count))
        result = [children[i] for i in range(count.value)] if children else []
        if children:
            X11.XFree(ctypes.cast(children, ctypes.c_void_p))
        return parent.value, result

    def top_level(self, window_id):
        """
        The window of the window manager frame over this window, or the window itself.
        """
        current = window_id
        root = self.root_window()
        while True:
            parent, _children = self.query_tree(current)
            if parent == 0 or parent == root:
                return current
            current = parent

    def geometry(self, window_id):
        """
        The window as {'id', 'name', 'w', 'h', 'x', 'y'} with the absolute position.
        """
        root = ctypes.c_ulong()
        x = ctypes.c_int()
        y = ctypes.c_int()
        width = ctypes.c_uint()
        height = ctypes.c_uint()
        border = ctypes.c_uint()
        depth = ctypes.c_uint()
        X11.XGetGeometry(self.dpy, window_id, ctypes.byref(root), ctypes.byref(x), ctypes.byref(y),
                         ctypes.byref(width), ctypes.byref(height), ctypes.byref(border),
                         ctypes.byref(depth))
        absolute_x = ctypes.c_int()
        absolute_y = ctypes.c_int()
        child = ctypes.c_ulong()
        X11.XTranslateCoordinates(self.dpy, window_id, root.value, 0, 0,
                                  ctypes.byref(absolute_x), ctypes.byref(absolute_y),
                                  ctypes.byref(child))
        return {'id': window_id, 'name': '', 'w': width.value, 'h': height.value,
                'x': absolute_x.value, 'y': absolute_y.value}

    def ewmh_move_window(self, window_id, x, y):
        """
        Ask the window manager to move the window: _NET_MOVERESIZE_WINDOW with StaticGravity.
        """
        event = XClientMessage()
        event.type = 33  # ClientMessage
        event.display = self.dpy
        event.window = window_id
        event.message_type = X11.XInternAtom(self.dpy, b'_NET_MOVERESIZE_WINDOW', 0)
        event.format = 32
        event.data[0] = 10 | (1 << 8) | (1 << 9)  # StaticGravity, X and Y are given
        event.data[1] = int(x)
        event.data[2] = int(y)
        X11.XSendEvent(self.dpy, self.root_window(), 0, 0x00180000, ctypes.byref(event))
        self.flush()
        self.sync()

    def ewmh_activate_window(self, window_id):
        """
        Ask the window manager to raise and activate the window: _NET_ACTIVE_WINDOW.
        """
        event = XClientMessage()
        event.type = 33  # ClientMessage
        event.display = self.dpy
        event.window = window_id
        event.message_type = X11.XInternAtom(self.dpy, b'_NET_ACTIVE_WINDOW', 0)
        event.format = 32
        event.data[0] = 1  # the source indication: a normal application
        event.data[1] = 0  # the timestamp: the current time
        X11.XSendEvent(self.dpy, self.root_window(), 0, 0x00180000, ctypes.byref(event))
        self.flush()
        self.sync()

    def motion(self, x, y):
        XTST.XTestFakeMotionEvent(self.dpy, -1, x, y, 0)

    def motion_rel(self, dx, dy):
        XTST.XTestFakeRelativeMotionEvent(self.dpy, dx, dy, 0)

    def button(self, number, down):
        XTST.XTestFakeButtonEvent(self.dpy, number, 1 if down else 0, 0)

    def keycode(self, name):
        keysym = X11.XStringToKeysym(keysym_name(name).encode())
        if not keysym:
            raise ToolError('no X keysym for key %r' % name)
        code = X11.XKeysymToKeycode(self.dpy, keysym)
        if not code:
            raise ToolError('the X server has no keycode for %r' % name)
        return code

    def key(self, name, down):
        XTST.XTestFakeKeyEvent(self.dpy, self.keycode(name), 1 if down else 0, 0)

    def grab(self, window):
        p = X11.XGetImage(self.dpy, window['id'], 0, 0, window['w'], window['h'], ALL_PLANES,
                          Z_PIXMAP)
        if not p:
            raise ToolError('XGetImage of window 0x%x failed' % window['id'])
        img = XImage.from_address(p)
        raw = ctypes.string_at(img.data, img.bytes_per_line * img.height)
        picture = Image.frombytes('RGB', (img.width, img.height), raw, 'raw', 'BGRX',
                                  img.bytes_per_line, 1)
        if X11.XDestroyImage is not None:
            X11.XDestroyImage(p)
        return picture

#------------------------------------------------------------------------------
# the windows of a display
#------------------------------------------------------------------------------


def list_windows(display):
    env = dict(os.environ, DISPLAY=display, LC_ALL='C', LANG='C')
    try:
        text = subprocess.run(['xwininfo', '-root', '-tree'], capture_output=True, text=True,
                              env=env, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        return []
    result = []
    for line in text.splitlines():
        match = WINDOW_RE.match(line)
        if match:
            class_match = re.search(r':\s+\("([^"]*)" "([^"]*)"\)', line)
            result.append({
                'id': int(match.group(1), 16),
                'name': match.group(2),
                'instance': class_match.group(1) if class_match else '',
                'class': class_match.group(2) if class_match else '',
                'w': int(match.group(3)),
                'h': int(match.group(4)),
                'x': int(match.group(7)),
                'y': int(match.group(8)),
            })
    return result


def find_window(display, title, exact=False):
    candidates = []
    for window in list_windows(display):
        name = window['name']
        if (name == title) if exact else (title.lower() in name.lower()):
            candidates.append(window)
    if not candidates:
        return None
    return max(candidates, key=lambda w: w['w'] * w['h'])


def find_screen_window(session, title, exact=False):
    """
    The window of the game as it is seen on the screen: the client of the window manager frame.
    A Wine game may be a child of its desktop window, a Kivy game is the client itself.
    """
    display = session.display_name
    window = find_window(display, title, exact)
    if window is None:
        return None
    windows = {item['id']: item for item in list_windows(display)}
    top = session.top_level(window['id'])
    top_info = windows.get(top)
    is_frame = top_info is not None and 'frame' in (top_info.get('class') or '').lower()
    if top == window['id'] and not is_frame:
        return window
    _parent, children = session.query_tree(top)
    best = None
    for child in children:
        item = session.geometry(child)
        if item['w'] < 100 or item['h'] < 100:
            continue
        if best is None or item['w'] * item['h'] > best['w'] * best['h']:
            best = item
    if best is not None:
        return best
    return window


def display_works(display):
    env = dict(os.environ, DISPLAY=display)
    try:
        return subprocess.run(['xwininfo', '-root'], capture_output=True, env=env,
                              timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def ensure_xvfb(display, timeout=10.0):
    if display_works(display):
        return
    os.makedirs(RUN_DIR, exist_ok=True)
    log_path = os.path.join(RUN_DIR, 'xvfb%s.log' % display.replace(':', '-'))
    with open(log_path, 'wb') as out:
        subprocess.Popen(['Xvfb', display, '-screen', '0', '1024x768x24', '-nolisten', 'tcp'],
                         stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if display_works(display):
            return
        time.sleep(0.2)
    raise ToolError('Xvfb %s did not start, see %s' % (display, log_path))

#------------------------------------------------------------------------------
# the processes
#------------------------------------------------------------------------------


def _read(path):
    try:
        with open(path, 'rb') as f:
            return f.read()
    except OSError:
        return b''


def _environ(pid):
    result = {}
    for item in _read('/proc/%d/environ' % pid).split(b'\0'):
        if b'=' in item:
            key, value = item.split(b'=', 1)
            result[key] = value
    return result


def _cmdline(pid):
    return _read('/proc/%d/cmdline' % pid).replace(b'\0', b' ').decode('utf-8', 'replace').strip()


def _cwd(pid):
    try:
        return os.readlink('/proc/%d/cwd' % pid)
    except OSError:
        return ''


def wine_processes(prefix, game_dir):
    result = []
    for name in os.listdir('/proc'):
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            exe = os.readlink('/proc/%d/exe' % pid)
        except OSError:
            exe = ''
        env = _environ(pid)
        if exe.startswith(game_dir + os.sep) or env.get(b'WINEPREFIX') == prefix.encode():
            result.append((pid, exe, _cmdline(pid)))
    return result


def demo_processes(display, remake_dir):
    result = []
    for name in os.listdir('/proc'):
        if not name.isdigit():
            continue
        pid = int(name)
        cmd = _cmdline(pid)
        # "remake" in the path of any command is not "make demo": match the demo commands exactly
        if 'client.main demo' not in cmd and 'make demo' not in cmd:
            continue
        if _cwd(pid) != remake_dir:
            continue
        env = _environ(pid)
        found = env.get(b'DISPLAY', b'').decode('utf-8', 'replace')
        if found and found != display:
            continue
        result.append((pid, cmd))
    return result

#------------------------------------------------------------------------------
# the input backends
#------------------------------------------------------------------------------


class XTestBackend(object):
    """
    Mouse and keys by XTest on the display of the target; works for Wine and for Kivy alike.
    """

    name = 'xtest'

    def __init__(self, target):
        self.target = target

    def _session(self):
        return self.target.x11()

    def _root(self, x, y):
        window = self.target.window()
        return window['x'] + int(x), window['y'] + int(y)

    def focus(self):
        self._session().focus(self.target.focus_window())
        self._session().sync()

    def move(self, x, y):
        self._session().motion(*self._root(x, y))
        self._session().flush()

    def click(self, x, y, button=1):
        session = self._session()
        session.motion(*self._root(x, y))
        session.flush()
        sleep_ms(80)
        session.button(BUTTONS[str(button).lower()], True)
        session.flush()
        sleep_ms(50)
        session.button(BUTTONS[str(button).lower()], False)
        session.flush()
        session.sync()

    def dclick(self, x, y, button=1):
        self.click(x, y, button)
        sleep_ms(120)
        self.click(x, y, button)

    def down(self, button=1):
        self._session().focus(self.target.focus_window())
        self._session().button(BUTTONS[str(button).lower()], True)
        self._session().flush()

    def up(self, button=1):
        self._session().button(BUTTONS[str(button).lower()], False)
        self._session().flush()
        self._session().sync()

    def drag(self, x1, y1, x2, y2, button=1, steps=12):
        session = self._session()
        session.motion(*self._root(x1, y1))
        session.flush()
        sleep_ms(80)
        session.button(BUTTONS[str(button).lower()], True)
        session.flush()
        sleep_ms(80)
        for i in range(1, steps + 1):
            x = int(x1) + (int(x2) - int(x1)) * i // steps
            y = int(y1) + (int(y2) - int(y1)) * i // steps
            session.motion(*self._root(x, y))
            session.flush()
            sleep_ms(15)
        session.button(BUTTONS[str(button).lower()], False)
        session.flush()
        session.sync()

    def rel(self, dx, dy, steps=None):
        session = self._session()
        dx, dy = int(dx), int(dy)
        steps = int(steps) if steps else max(1, max(abs(dx), abs(dy)) // 20)
        for i in range(steps):
            session.motion_rel(dx * (i + 1) // steps - dx * i // steps,
                               dy * (i + 1) // steps - dy * i // steps)
            session.flush()
            sleep_ms(10)
        session.sync()

    def key(self, name, hold_ms=60):
        session = self._session()
        session.focus(self.target.focus_window())
        code = session.keycode(name)
        XTST.XTestFakeKeyEvent(session.dpy, code, 1, 0)
        session.flush()
        sleep_ms(hold_ms)
        XTST.XTestFakeKeyEvent(session.dpy, code, 0, 0)
        session.flush()
        session.sync()

    def chord(self, names, hold_ms=60):
        session = self._session()
        session.focus(self.target.focus_window())
        codes = [session.keycode(name) for name in names]
        for code in codes:
            XTST.XTestFakeKeyEvent(session.dpy, code, 1, 0)
        session.flush()
        sleep_ms(hold_ms)
        for code in reversed(codes):
            XTST.XTestFakeKeyEvent(session.dpy, code, 0, 0)
        session.flush()
        session.sync()

    def keydown(self, name):
        session = self._session()
        session.focus(self.target.focus_window())
        session.key(name, True)
        session.flush()

    def keyup(self, name):
        session = self._session()
        session.key(name, False)
        session.flush()
        session.sync()

    def wheel(self, direction, count=1):
        session = self._session()
        button = 4 if direction.lower() in ('up', 'in') else 5
        for _ in range(int(count)):
            session.button(button, True)
            session.flush()
            sleep_ms(30)
            session.button(button, False)
            session.flush()
            sleep_ms(30)
        session.sync()

    def type_text(self, text):
        session = self._session()
        session.focus(self.target.focus_window())
        for char in text:
            if char == ' ':
                name = 'space'
            else:
                name = char
            shifted = char.isupper() or char in '!@#$%^&*()_+{}|:"<>?~'
            if shifted:
                shift = session.keycode('shift')
                XTST.XTestFakeKeyEvent(session.dpy, shift, 1, 0)
                session.flush()
            code = session.keycode(name)
            XTST.XTestFakeKeyEvent(session.dpy, code, 1, 0)
            session.flush()
            sleep_ms(40)
            XTST.XTestFakeKeyEvent(session.dpy, code, 0, 0)
            if shifted:
                XTST.XTestFakeKeyEvent(session.dpy, shift, 0, 0)
            session.flush()
            sleep_ms(50)
        session.sync()

    def flush(self):
        self._session().sync()

#------------------------------------------------------------------------------


class XSendBackend(object):
    """
    Mouse and keys on a display shared with the desktop, where XWayland ignores XTest:
    the pointer is moved by XWarpPointer, the buttons and the keys go to the game window by XSendEvent.
    The coordinates stay the client coordinates of the game window.
    """

    name = 'xsend'

    def __init__(self, target):
        self.target = target
        # the modifiers which are held by keydown and are not released yet
        self.held = set()
        # the mouse buttons which are pressed and not released yet: a drag needs them in the state
        # of its motion events, Wine ignores the motion of a button it does not see pressed
        self.buttons = set()

    def _session(self):
        return self.target.x11()

    def _screen_window(self):
        return self.target.window()

    def _focus_window(self):
        return self.target.focus_window()

    def _root(self, x, y):
        # the window manager may move the window after the arrangement
        window = self.target.window(refresh=True)
        return window['x'] + int(x), window['y'] + int(y)

    def _pointer(self):
        session = self._session()
        root_return = ctypes.c_ulong()
        child_return = ctypes.c_ulong()
        root_x = ctypes.c_int()
        root_y = ctypes.c_int()
        window_x = ctypes.c_int()
        window_y = ctypes.c_int()
        mask = ctypes.c_uint()
        X11.XQueryPointer(session.dpy, session.root_window(), ctypes.byref(root_return),
                          ctypes.byref(child_return), ctypes.byref(root_x), ctypes.byref(root_y),
                          ctypes.byref(window_x), ctypes.byref(window_y), ctypes.byref(mask))
        return root_x.value, root_y.value

    def _event_position(self):
        window = self._focus_window()
        root_x, root_y = self._pointer()
        return root_x - window['x'], root_y - window['y'], root_x, root_y

    def _key_state(self):
        state = 0
        for name in self.held:
            state |= MODIFIER_MASKS.get(name, 0)
        for number in self.buttons:
            state |= BUTTON_MASKS.get(number, 0)
        return state

    def _warp(self, x, y):
        session = self._session()
        absolute_x, absolute_y = self._root(x, y)
        X11.XWarpPointer(session.dpy, 0, session.root_window(), 0, 0, 0, 0,
                         absolute_x, absolute_y)
        session.flush()
        session.sync()

    def _send_motion(self, absolute_x, absolute_y):
        """
        A MotionNotify of the window: Kivy takes the position of its mouse from it, even when
        another window of the desktop covers the game at this point.
        """
        session = self._session()
        window = self._focus_window()
        event = XMotionEvent()
        event.type = 6  # MotionNotify
        event.display = session.dpy
        event.window = window['id']
        event.root = session.root_window()
        event.x = absolute_x - window['x']
        event.y = absolute_y - window['y']
        event.x_root = absolute_x
        event.y_root = absolute_y
        event.state = self._key_state()
        event.is_hint = 0
        event.same_screen = 1
        X11.XSendEvent(session.dpy, window['id'], 0, 0, ctypes.byref(event))
        session.flush()

    def _motion(self, x, y):
        absolute_x, absolute_y = self._root(x, y)
        self._send_motion(absolute_x, absolute_y)

    def _motion_current(self):
        absolute_x, absolute_y = self._pointer()
        self._send_motion(absolute_x, absolute_y)

    def _button(self, number, down):
        session = self._session()
        window = self._focus_window()
        x, y, root_x, root_y = self._event_position()
        event = XButtonEvent()
        event.type = 4 if down else 5
        event.display = session.dpy
        event.window = window['id']
        event.root = session.root_window()
        event.x = x
        event.y = y
        event.x_root = root_x
        event.y_root = root_y
        # the state is the one before the event: a release still has its button in it
        event.state = self._key_state()
        event.button = number
        event.same_screen = 1
        X11.XSendEvent(session.dpy, window['id'], 0, 0, ctypes.byref(event))
        session.flush()
        if down:
            self.buttons.add(number)
        else:
            self.buttons.discard(number)

    def _key(self, name, down, state=None):
        session = self._session()
        window = self._focus_window()
        x, y, root_x, root_y = self._event_position()
        event = XKeyEvent()
        event.type = 2 if down else 3
        event.display = session.dpy
        event.window = window['id']
        event.root = session.root_window()
        event.x = x
        event.y = y
        event.x_root = root_x
        event.y_root = root_y
        event.state = self._key_state() if state is None else state
        event.keycode = session.keycode(name)
        event.same_screen = 1
        X11.XSendEvent(session.dpy, window['id'], 0, 0, ctypes.byref(event))
        session.flush()

    def focus(self):
        session = self._session()
        # raise the frame of the target: another window of the desktop must not take its clicks
        session.ewmh_activate_window(self._screen_window()['id'])
        session.focus(self._focus_window())
        session.sync()
        sleep_ms(50)

    def move(self, x, y):
        self._warp(x, y)
        self._motion(x, y)

    def click(self, x, y, button=1):
        self._warp(x, y)
        self._motion(x, y)
        sleep_ms(80)
        number = BUTTONS[str(button).lower()]
        self._button(number, True)
        sleep_ms(50)
        self._button(number, False)
        # Kivy reads the position of the pointer while it handles the button event, so the pointer
        # stays here until the other target of the comparison can move it
        sleep_ms(500)
        self._session().sync()

    def dclick(self, x, y, button=1):
        self.click(x, y, button)
        sleep_ms(120)
        self.click(x, y, button)

    def down(self, button=1):
        self._motion_current()
        self._button(BUTTONS[str(button).lower()], True)
        self._session().sync()

    def up(self, button=1):
        self._motion_current()
        self._button(BUTTONS[str(button).lower()], False)
        self._session().sync()

    def drag(self, x1, y1, x2, y2, button=1, steps=12):
        self._warp(x1, y1)
        self._motion(x1, y1)
        sleep_ms(80)
        number = BUTTONS[str(button).lower()]
        self._button(number, True)
        sleep_ms(80)
        for i in range(1, int(steps) + 1):
            x = int(x1) + (int(x2) - int(x1)) * i // int(steps)
            y = int(y1) + (int(y2) - int(y1)) * i // int(steps)
            self._warp(x, y)
            self._motion(x, y)
            sleep_ms(15)
        self._button(number, False)
        sleep_ms(80)
        self._session().sync()

    def rel(self, dx, dy, steps=None):
        session = self._session()
        dx, dy = int(dx), int(dy)
        steps = int(steps) if steps else max(1, max(abs(dx), abs(dy)) // 20)
        start_x, start_y = self._pointer()
        for i in range(1, steps + 1):
            X11.XWarpPointer(session.dpy, 0, session.root_window(), 0, 0, 0, 0,
                             start_x + dx * i // steps, start_y + dy * i // steps)
            session.flush()
            self._motion_current()
            sleep_ms(10)
        session.sync()

    def wheel(self, direction, count=1):
        self._motion_current()
        number = 4 if direction.lower() in ('up', 'in') else 5
        for _ in range(int(count)):
            self._button(number, True)
            sleep_ms(30)
            self._button(number, False)
            sleep_ms(30)
        self._session().sync()

    def key(self, name, hold_ms=60):
        self.focus()
        self._key(name, True)
        sleep_ms(hold_ms)
        self._key(name, False)
        self._session().sync()

    def chord(self, names, hold_ms=60):
        self.focus()
        state = self._key_state()
        for name in names:
            self._key(name, True, state)
            state |= MODIFIER_MASKS.get(name.lower(), 0)
        sleep_ms(hold_ms)
        for name in reversed(names):
            state &= ~MODIFIER_MASKS.get(name.lower(), 0)
            self._key(name, False, state)
        self._session().sync()

    def keydown(self, name):
        self.focus()
        self._key(name, True)
        self.held.add(name.lower())
        self._session().sync()

    def keyup(self, name):
        lower = name.lower()
        self._key(name, False)
        self.held.discard(lower)
        self._session().sync()

    def type_text(self, text):
        self.focus()
        for char in text:
            shifted = char.isupper()
            if shifted:
                self._key('Shift_L', True)
            if char == ' ':
                name = 'space'
            elif len(char) == 1 and (char.isalnum() or char in '-.'):
                name = char.lower() if char.isalpha() else char
            else:
                raise ToolError('xsend can not type %r' % char)
            self._key(name, True, self._key_state() | (1 if shifted else 0))
            sleep_ms(20)
            self._key(name, False, self._key_state() | (1 if shifted else 0))
            if shifted:
                self._key('Shift_L', False)
            sleep_ms(20)
        self._session().sync()

    def flush(self):
        self._session().sync()
SENDINPUT_KEYS = {
    'esc': 'esc', 'escape': 'esc', 'enter': 'enter', 'return': 'enter',
    'space': 'space', 'spacebar': 'space', 'tab': 'tab', 'backspace': 'backspace',
    'shift': 'shift', 'lshift': 'shift', 'rshift': 'shift',
    'ctrl': 'ctrl', 'control': 'ctrl', 'lctrl': 'ctrl', 'rctrl': 'ctrl',
    'alt': 'alt', 'lalt': 'alt', 'ralt': 'alt',
    'up': 'up', 'down': 'down', 'left': 'left', 'right': 'right',
    'home': 'home', 'end': 'end', 'pageup': 'pageup', 'pgup': 'pageup',
    'pagedown': 'pagedown', 'pgdn': 'pagedown', 'insert': 'insert', 'ins': 'insert',
    'delete': 'delete', 'del': 'delete', 'minus': 'minus', 'plus': 'plus',
}
for _i in range(1, 13):
    SENDINPUT_KEYS['f%d' % _i] = 'f%d' % _i


class SendInputBackend(object):
    """
    Mouse and keys by SendInput of user32 inside the Wine of the original game (C:\\eitools\\eiinput.py).
    The commands are collected and sent by one run of the Windows Python before a wait or a picture.
    """

    name = 'sendinput'

    def __init__(self, target):
        self.target = target
        self.queue = []
        self.wine = target.wine
        self.prefix = target.prefix
        self.display = target.display

    def _key_name(self, name):
        lower = name.lower()
        if lower in SENDINPUT_KEYS:
            return SENDINPUT_KEYS[lower]
        if len(name) == 1:
            return name
        raise ToolError('sendinput has no key %r' % name)

    def move(self, x, y):
        self.queue.append(['abs', str(int(x)), str(int(y))])

    def click(self, x, y, button=1):
        name = {1: 'left', 2: 'middle', 3: 'right'}[BUTTONS[str(button).lower()]]
        self.queue.append(['abs', str(int(x)), str(int(y))])
        self.queue.append(['wait', '100'])
        self.queue.append(['down', name])
        self.queue.append(['wait', '60'])
        self.queue.append(['up', name])

    def dclick(self, x, y, button=1):
        self.click(x, y, button)
        self.queue.append(['wait', '120'])
        self.click(x, y, button)

    def down(self, button=1):
        self.queue.append(['down', {1: 'left', 2: 'middle', 3: 'right'}[BUTTONS[str(button).lower()]]])

    def up(self, button=1):
        self.queue.append(['up', {1: 'left', 2: 'middle', 3: 'right'}[BUTTONS[str(button).lower()]]])

    def drag(self, x1, y1, x2, y2, button=1, steps=12):
        self.queue.append(['abs', str(int(x1)), str(int(y1))])
        self.queue.append(['wait', '80'])
        self.down(button)
        self.queue.append(['wait', '80'])
        for i in range(1, int(steps) + 1):
            self.queue.append(['abs', str(int(x1) + (int(x2) - int(x1)) * i // int(steps)),
                               str(int(y1) + (int(y2) - int(y1)) * i // int(steps))])
            self.queue.append(['wait', '15'])
        self.up(button)

    def rel(self, dx, dy, steps=None):
        self.queue.append(['rel', str(int(dx)), str(int(dy))])

    def key(self, name, hold_ms=60):
        self.queue.append(['key', self._key_name(name), str(int(hold_ms))])

    def chord(self, names, hold_ms=60):
        for name in names:
            self.queue.append(['keydown', self._key_name(name)])
        self.queue.append(['wait', str(int(hold_ms))])
        for name in reversed(names):
            self.queue.append(['keyup', self._key_name(name)])

    def keydown(self, name):
        self.queue.append(['keydown', self._key_name(name)])

    def keyup(self, name):
        self.queue.append(['keyup', self._key_name(name)])

    def wheel(self, direction, count=1):
        raise ToolError('wheel is not supported by the sendinput backend, use EIDRIVE_ORIGINAL_BACKEND=xtest')

    def type_text(self, text):
        for char in text:
            if char == ' ':
                self.queue.append(['key', 'space'])
            elif len(char) == 1 and (char.isalnum() or char in '-.'):
                self.queue.append(['key', char])
            else:
                raise ToolError('sendinput can not type %r' % char)

    def flush(self):
        if not self.queue:
            return
        commands = ['focus'] + [item for command in self.queue for item in command]
        self.queue = []
        env = dict(os.environ, DISPLAY=self.display, WINEPREFIX=self.prefix, WINEDEBUG='-all')
        env.pop('WAYLAND_DISPLAY', None)
        args = [self.wine, 'C:\\eitools\\python\\python.exe', 'C:\\eitools\\eiinput.py'] + commands
        result = subprocess.run(args, cwd=self.target.dir, env=env, capture_output=True, text=True,
                                timeout=60)
        if result.returncode != 0:
            raise ToolError('eiinput failed: %s' % (result.stderr or result.stdout).strip())

#------------------------------------------------------------------------------
# a target: the original game or the demo
#------------------------------------------------------------------------------


class Target(object):

    def __init__(self, name, display=None, shared=False):
        self.name = name
        self.display = display or DISPLAYS[name]
        # the display may be the desktop and be shared with the other target
        self.shared = shared
        self.title = TITLES[name]
        self.exact = EXACT_TITLES[name]
        self.backend_name = ('xsend' if shared else
                             os.environ.get('EIDRIVE_%s_BACKEND' % name.upper(), 'xtest').lower())
        self._window = None
        self._x11 = None
        self._backend = None
        self.started_at = None
        if name == 'original':
            self.dir = ORIGINAL_DIR
            self.prefix = os.path.join(self.dir, 'prefix')
            self.wine = os.path.join(self.dir, 'wine', 'bin', 'wine')
            self.wine_python = os.path.join(self.dir, 'wine', 'bin', 'wine')
            self.log_path = os.path.join(RUN_DIR, 'original.log')
            self.pid_path = os.path.join(RUN_DIR, 'original.pid')
        else:
            self.dir = REMAKE_DIR
            self.prefix = None
            self.wine = None
            self.wine_python = None
            self.log_path = os.path.join(RUN_DIR, 'demo.log')
            self.pid_path = os.path.join(RUN_DIR, 'demo.pid')
        self.data_path = (os.environ.get('EIDRIVE_DATA_PATH')
                          or os.path.join(ORIGINAL_DIR, 'game_info', 'data'))

    def __repr__(self):
        return 'Target(%s, %s)' % (self.name, self.display)

    #-- the X server, the window, the input

    def x11(self):
        if self._x11 is None:
            self._x11 = XSession(self.display)
        return self._x11

    def window(self, refresh=False):
        if refresh or self._window is None:
            if self.shared:
                self._window = find_screen_window(self.x11(), self.title, self.exact)
            else:
                self._window = find_window(self.display, self.title, self.exact)
            if self._window is None:
                raise ToolError('no window %r on display %s (is %s running?)'
                                % (self.title, self.display, self.name))
        return self._window

    def forget_window(self):
        self._window = None

    def focus_window(self):
        """
        The window which takes the keyboard: the game window inside the Wine desktop window
        or the client of the window manager frame, not the frame itself.
        """
        window = self.window()
        if not self.shared:
            return window
        matched = find_window(self.display, self.title, self.exact)
        if matched is None or matched['id'] == window['id']:
            return window
        if self.x11().top_level(matched['id']) == matched['id']:
            return window
        return matched

    def backend(self):
        if self._backend is None:
            if self.shared:
                self._backend = XSendBackend(self)
            elif self.backend_name == 'sendinput':
                if self.name != 'original':
                    raise ToolError('sendinput is only for the original game')
                self._backend = SendInputBackend(self)
            else:
                self._backend = XTestBackend(self)
        return self._backend

    def focus_sendinput(self):
        env = dict(os.environ, DISPLAY=self.display, WINEPREFIX=self.prefix, WINEDEBUG='-all')
        env.pop('WAYLAND_DISPLAY', None)
        subprocess.run([self.wine, 'C:\\eitools\\python\\python.exe', 'C:\\eitools\\eiinput.py',
                        'focus'], cwd=self.dir, env=env, capture_output=True, timeout=60)

    def flush(self):
        self.backend().flush()

    #-- the pictures

    def shot(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        try:
            picture = self.x11().grab(self.window())
        except ToolError:
            self.forget_window()
            picture = self.x11().grab(self.window(refresh=True))
        picture.save(path)
        return path

    def record(self, dir_path, seconds, fps):
        env = dict(os.environ, DISPLAY=self.display)
        script = os.path.join(TOOLS_DIR, 'xrec.py')
        result = subprocess.run([sys.executable, script, self.title, dir_path, str(seconds),
                                 str(fps)], env=env, capture_output=True, text=True,
                                timeout=120 + seconds * 2)
        log((result.stdout or '').strip())
        if result.returncode != 0:
            raise ToolError('record failed: %s' % (result.stderr or result.stdout).strip())
        return dir_path + '.mp4'

    #-- the processes

    def running(self):
        return find_window(self.display, self.title, self.exact) is not None

    def processes(self):
        if self.name == 'original':
            return wine_processes(self.prefix, self.dir)
        return demo_processes(self.display, self.dir)

    def status(self):
        window = find_window(self.display, self.title, self.exact)
        procs = self.processes()
        log('target    %s' % self.name)
        log('display   %s (%s)' % (self.display, 'up' if display_works(self.display) else 'down'))
        log('backend   %s' % self.backend_name)
        if window:
            log('window    0x%x %r %dx%d at %d,%d' % (window['id'], window['name'], window['w'],
                                                      window['h'], window['x'], window['y']))
        else:
            log('window    none')
        if procs:
            for proc in procs:
                log('process   %s' % (proc,))
        else:
            log('process   none')
        return 0 if window else 1

    def start(self, timeout=None):
        ensure_xvfb(self.display)
        window = find_window(self.display, self.title, self.exact)
        if window:
            log('%s already runs: 0x%x %r' % (self.name, window['id'], window['name']))
            return 0
        procs = self.processes()
        if procs:
            # the window may still be opening
            try:
                window = self.wait_window(timeout or 30)
                log('%s already runs: 0x%x %r' % (self.name, window['id'], window['name']))
                return 0
            except ToolError:
                raise ToolError('%s has processes but no window; stop it first' % self.name)
        os.makedirs(RUN_DIR, exist_ok=True)
        self.started_at = time.time()
        if self.name == 'original':
            self._start_original()
            timeout = timeout or 90
        else:
            self._start_demo()
            timeout = timeout or 90
        window = self.wait_window(timeout)
        self._window = window
        log('%s started: 0x%x %r on %s' % (self.name, window['id'], window['name'], self.display))
        return 0

    def _spawn(self, args, env):
        out = open(self.log_path, 'wb')
        proc = subprocess.Popen(args, cwd=self.dir, env=env, stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=True)
        with open(self.pid_path, 'w') as f:
            f.write('%d\n' % proc.pid)
        return proc

    def _start_original(self):
        env = dict(os.environ, DISPLAY=self.display, WINEDEBUG='-all')
        env.pop('WAYLAND_DISPLAY', None)
        self._set_show_tutorial(env)
        # trace.sh runs game.exe under the coverage tracer of remake/re/eitrace
        self._spawn(['bash', os.environ.get('EIDRIVE_ORIGINAL_SCRIPT') or 'start.sh'], env)

    def _set_show_tutorial(self, env):
        """
        The option "show tutorial" of game.exe (ShowTutorial of Game Specific Settings): off for every start,
        because reset_settings removes the key and the hint windows pause the world; EIDRIVE_TUTORIAL=1 keeps
        them. The demo shows them only with its option --tutorial.
        """
        wine = os.path.join(self.dir, 'wine', 'bin', 'wine')
        if not os.path.exists(wine):
            return
        value = '1' if os.environ.get('EIDRIVE_TUTORIAL') == '1' else '0'
        reg_env = dict(env, WINEPREFIX=self.prefix)
        result = subprocess.run([wine, 'reg', 'add', 'HKCU\\' + ORIGINAL_REGISTRY_KEY.replace('\\\\', '\\') +
                                 '\\Game Specific Settings', '/v', 'ShowTutorial', '/t', 'REG_DWORD', '/d', value,
                                 '/f'], env=reg_env, capture_output=True, timeout=60)
        log('ShowTutorial=%s (%s)' % (value, 'ok' if result.returncode == 0 else 'reg failed'))

    def _start_demo(self):
        python = os.path.join(self.dir, 'venv', 'bin', 'python')
        if not os.path.exists(python):
            raise ToolError('the demo has no venv, run "make venv" in %s' % self.dir)
        env = dict(os.environ, DISPLAY=self.display, EI_DATA_PATH=self.data_path)
        env.pop('WAYLAND_DISPLAY', None)
        audio = os.environ.get('EIDRIVE_DEMO_AUDIODRIVER')
        if audio:
            env['SDL_AUDIODRIVER'] = audio
        self._spawn(['make', 'demo'], env)

    def wait_window(self, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            window = find_window(self.display, self.title, self.exact)
            if window:
                return window
            time.sleep(0.5)
        raise ToolError('no window %r on %s in %d s, see %s'
                        % (self.title, self.display, timeout, self.log_path))

    def stop(self, timeout=30):
        if self.name == 'original':
            return self._stop_original(timeout)
        return self._stop_demo(timeout)

    def reset_settings(self):
        """
        Stops the game and removes its settings, a copy goes to eitools/run/settings_backup/.
        The next start has the defaults built into the game; the hotkeys of the original come from
        eitools/defaults/keyboard.ini, because without the file game.exe binds no actions.
        """
        # wineserver writes the registry back when it ends, so the game must be stopped first
        self.stop()
        backup_dir_path = os.path.join(SETTINGS_BACKUP_DIR, '%s-%s' % (
            self.name, time.strftime('%Y%m%d-%H%M%S')))
        os.makedirs(backup_dir_path, exist_ok=True)
        removed = []
        if self.name == 'original':
            reg_file_path = os.path.join(self.prefix, 'user.reg')
            with open(reg_file_path, encoding='utf-8', errors='surrogateescape') as f:
                text = f.read()
            with open(os.path.join(backup_dir_path, 'user.reg'), 'w', encoding='utf-8',
                      errors='surrogateescape') as f:
                f.write(text)
            kept = []
            skip = False
            count = 0
            for line in text.split('\n'):
                if line.startswith('['):
                    skip = line.startswith('[' + ORIGINAL_REGISTRY_KEY)
                    if skip:
                        count += 1
                if not skip:
                    kept.append(line)
            with open(reg_file_path, 'w', encoding='utf-8', errors='surrogateescape') as f:
                f.write('\n'.join(kept))
            removed.append('%d registry keys removed' % count)
            settings_paths = [os.path.join(self.dir, ORIGINAL_KEYBOARD_PATH)]
        else:
            settings_paths = [DEMO_SETTINGS_PATH]
        for file_path in settings_paths:
            file_name = os.path.basename(file_path)
            existed = os.path.exists(file_path)
            if existed:
                with open(file_path, 'rb') as src, \
                        open(os.path.join(backup_dir_path, file_name), 'wb') as dst:
                    dst.write(src.read())
                os.unlink(file_path)
            if (self.name == 'original' and file_name == 'keyboard.ini' and
                    os.path.exists(DEFAULT_KEYBOARD_PATH)):
                with open(DEFAULT_KEYBOARD_PATH, 'rb') as src, open(file_path, 'wb') as dst:
                    dst.write(src.read())
                removed.append('%s %s with eitools/defaults' % (
                    file_name, 'replaced' if existed else 'written'))
            elif existed:
                removed.append('%s removed' % file_name)
        log('%s settings reset: %s, backup %s' % (
            self.name, ', '.join(removed) or 'nothing', backup_dir_path))
        return backup_dir_path

    def _stop_original(self, timeout):
        wineserver = os.path.join(self.dir, 'wine', 'bin', 'wineserver')
        env = dict(os.environ, WINEPREFIX=self.prefix, WINEDEBUG='-all')
        if os.path.exists(wineserver):
            subprocess.run([wineserver, '-k'], env=env, capture_output=True, timeout=30)
        else:
            raise ToolError('no wineserver in %s' % self.dir)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.processes():
                log('original stopped')
                return 0
            time.sleep(0.5)
        leftovers = self.processes()
        log('original: killing %d processes' % len(leftovers))
        for pid, _exe, _cmd in leftovers:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(1)
        if self.processes():
            raise ToolError('original still runs: %s' % (self.processes(),))
        log('original stopped')
        return 0

    def _stop_demo(self, timeout):
        pids = []
        if os.path.exists(self.pid_path):
            try:
                pid = int(open(self.pid_path).read().strip())
                pids.append(pid)
            except (ValueError, OSError):
                pass
        for pid, _cmd in demo_processes(self.display, self.dir):
            pids.append(pid)
        pids = sorted(set(pids))
        if not pids:
            log('demo stopped')
            return 0
        for pid in pids:
            try:
                os.killpg(pid, signal.SIGTERM)
            except OSError:
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not demo_processes(self.display, self.dir) and not find_window(
                    self.display, self.title, self.exact):
                log('demo stopped')
                return 0
            time.sleep(0.5)
        for pid, _cmd in demo_processes(self.display, self.dir):
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        time.sleep(1)
        log('demo stopped')
        return 0

    def restart(self, timeout=None):
        self.stop()
        self.forget_window()
        return self.start(timeout)

    #-- the input commands

    def focus_input(self):
        backend = self.backend()
        if hasattr(backend, 'focus'):
            backend.focus()
        else:
            self.focus_sendinput()

    def do_input(self, command, args):
        command = command.lower()
        # a wait is a pause of this target only: the other target keeps running
        if command in ('wait', 'flush') and self.backend_name != 'sendinput':
            if command == 'wait':
                self.flush()
                sleep_ms(int(args[0]))
            else:
                self.flush()
            return
        with input_lock(self.display):
            if self.shared:
                self.focus_input()
            self._do_input(command, args)

    def _do_input(self, command, args):
        backend = self.backend()
        command = command.lower()
        if command in ('move', 'abs'):
            backend.move(int(args[0]), int(args[1]))
        elif command == 'focus':
            if hasattr(backend, 'focus'):
                backend.focus()
            else:
                self.focus_sendinput()
        elif command == 'click':
            backend.click(int(args[0]), int(args[1]), args[2] if len(args) > 2 else 1)
        elif command == 'rclick':
            backend.click(int(args[0]), int(args[1]), 3)
        elif command == 'dclick':
            backend.dclick(int(args[0]), int(args[1]), args[2] if len(args) > 2 else 1)
        elif command == 'down':
            backend.down(args[0] if args else 1)
        elif command == 'up':
            backend.up(args[0] if args else 1)
        elif command == 'drag':
            if len(args) < 4:
                raise ToolError('drag needs X1 Y1 X2 Y2 [button] [steps]')
            backend.drag(int(args[0]), int(args[1]), int(args[2]), int(args[3]),
                         args[4] if len(args) > 4 else 1, int(args[5]) if len(args) > 5 else 12)
        elif command == 'rel':
            backend.rel(int(args[0]), int(args[1]), args[2] if len(args) > 2 else None)
        elif command == 'wheel':
            backend.wheel(args[0], args[1] if len(args) > 1 else 1)
        elif command == 'key':
            if not args:
                raise ToolError('key needs a name')
            hold = int(args[1]) if len(args) > 1 else 60
            if '+' in args[0]:
                backend.chord(args[0].split('+'), hold)
            else:
                backend.key(args[0], hold)
        elif command == 'chord':
            names = args[:-1] if len(args) > 1 and args[-1].isdigit() else args
            hold = int(args[-1]) if len(args) > 1 and args[-1].isdigit() else 60
            if not names:
                raise ToolError('chord needs key names')
            backend.chord(names, hold)
        elif command == 'keydown':
            backend.keydown(args[0])
        elif command == 'keyup':
            backend.keyup(args[0])
        elif command == 'type':
            backend.type_text(' '.join(args))
        elif command == 'wait':
            self.flush()
            sleep_ms(int(args[0]))
        elif command == 'flush':
            self.flush()
        else:
            raise ToolError('unknown input command %r' % command)

#------------------------------------------------------------------------------
# the scenarios
#------------------------------------------------------------------------------

INPUT_COMMANDS = ('move', 'abs', 'focus', 'click', 'rclick', 'dclick', 'down', 'up', 'drag',
                  'rel', 'wheel', 'key', 'chord', 'keydown', 'keyup', 'type', 'wait', 'flush')


def image_diff(path_a, path_b):
    """
    The mean absolute difference of two pictures (0..255), None when one of them is missing.
    """
    if not os.path.exists(path_a) or not os.path.exists(path_b):
        return None
    a = Image.open(path_a).convert('RGB')
    b = Image.open(path_b).convert('RGB')
    if a.size != b.size:
        b = b.resize(a.size)
    diff = ImageChops.difference(a, b).convert('L')
    histogram = diff.histogram()
    total = sum(histogram)
    return sum(value * count for value, count in enumerate(histogram)) / float(total)


def scenario_timeout(path):
    """
    The "timeout MS" line of the top-level scenario, required in every scenario.
    """
    with open(path, encoding='utf-8') as f:
        for number, raw in enumerate(f, 1):
            line = raw.split('#', 1)[0].strip()
            match = re.match(r'^timeout\s+(\d+)$', line, re.IGNORECASE)
            if match:
                value = int(match.group(1))
                if value <= 0:
                    raise ToolError('%s line %d: the timeout must be positive' % (path, number))
                return value
    raise ToolError('no "timeout MS" line in %s' % path)


def scenario_resets_settings(path):
    """
    True when the top-level scenario has a "reset_settings" line.
    """
    with open(path, encoding='utf-8') as f:
        for raw in f:
            if raw.split('#', 1)[0].strip().lower() == 'reset_settings':
                return True
    return False


def scenario_has_sync(path, depth=0):
    """
    True when the scenario (or one of its includes) has a "sync" line.
    """
    if depth > 8:
        return False
    base_dir = os.path.dirname(os.path.abspath(path))
    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            match = re.match(r'^(original|demo)\s*:\s*(.*)$', line)
            if match:
                line = match.group(2).strip()
            words = shlex.split(line)
            if not words:
                continue
            if words[0].lower() == 'sync':
                return True
            if words[0].lower() == 'include' and len(words) > 1:
                if scenario_has_sync(os.path.join(base_dir, words[1]), depth + 1):
                    return True
    return False


class Run(object):
    """
    One scenario over one target: sends the commands, saves the pictures and the checks.
    """

    def __init__(self, target, out_dir, abort=None, barrier=None):
        self.target = target
        self.out_dir = out_dir
        self.abort = abort or RunAbort()
        self.barrier = barrier
        self.finished = threading.Event()
        self.timeout_ms = None
        self.deadline = None
        self.shots = []
        self.checks = []
        self.errors = []
        self.started_at = time.time()
        os.makedirs(out_dir, exist_ok=True)
        self.log_path = os.path.join(out_dir, 'run.log')
        self._log_file = open(self.log_path, 'w', encoding='utf-8')

    def __repr__(self):
        return 'Run(%s, %s)' % (self.target.name, self.out_dir)

    def log(self, text):
        line = '%7.2f %s' % (time.time() - self.started_at, text)
        self._log_file.write(line + '\n')
        self._log_file.flush()
        log('[%s] %s' % (self.target.name, line))

    def error(self, text):
        self.errors.append(text)
        self.log('ERROR ' + text)

    def check_timeout(self):
        self.abort.check()
        if self.deadline is not None and time.time() >= self.deadline:
            reason = 'the scenario ran longer than %d ms' % self.timeout_ms
            self.abort.set(reason)
            raise ScenarioTimeout(reason)

    def wait_ms(self, ms):
        end = time.time() + ms / 1000.0
        while True:
            self.check_timeout()
            left = end - time.time()
            if left <= 0:
                return
            time.sleep(min(left, 0.1))

    def timeout_stop(self, exc):
        self.error('timeout: %s' % exc)
        self.abort.set(str(exc))
        try:
            self.target.stop()
        except Exception as stop_exc:
            self.error('stop after timeout: %s' % stop_exc)

    def run(self, path):
        self.timeout_ms = scenario_timeout(path)
        self.deadline = self.started_at + self.timeout_ms / 1000.0
        self.log('scenario %s' % path)
        self.log('timeout %d ms' % self.timeout_ms)
        try:
            try:
                self._run_file(path, 0)
            except ScenarioTimeout as exc:
                self.timeout_stop(exc)
            self.target.flush()
            return self.finish()
        finally:
            self.finished.set()

    def _run_file(self, path, depth):
        if depth > 8:
            raise ToolError('include is nested too deep at %r' % path)
        base_dir = os.path.dirname(os.path.abspath(path))
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
        for number, raw in enumerate(lines, 1):
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            self.check_timeout()
            prefix = None
            match = re.match(r'^(original|demo)\s*:\s*(.*)$', line)
            if match:
                prefix, line = match.group(1), match.group(2).strip()
            if prefix is not None and prefix != self.target.name:
                continue
            try:
                self.do(line, base_dir, depth)
            except ScenarioTimeout:
                raise
            except Exception as exc:
                self.error('line %d %r: %s' % (number, line, exc))

    def do(self, line, base_dir=None, depth=0):
        words = shlex.split(line)
        if not words:
            return
        command = words[0].lower()
        args = words[1:]
        if command in ('timeout', 'reset_settings'):
            # both are read from the top-level scenario by compare before the run starts
            return
        if command == 'sync':
            if self.barrier is not None:
                self.barrier.wait(self.abort, self.check_timeout)
            return
        if command == 'repeat':
            # repeat N COMMAND...: the command N times, "sync" and "repeat" are not allowed inside
            if len(args) < 2 or not args[0].isdigit():
                raise ToolError('repeat needs a count and a command')
            if args[1].lower() in ('sync', 'repeat', 'include', 'timeout'):
                raise ToolError('repeat can not run %r' % args[1])
            for _ in range(int(args[0])):
                self.check_timeout()
                self.do(shlex.join(args[1:]), base_dir, depth)
            return
        if command in ('note', 'echo'):
            self.log(' '.join(args))
        elif command == 'phase':
            self.phase(args[0])
        elif command == 'include':
            if not args:
                raise ToolError('include needs a file')
            self._run_file(os.path.join(base_dir or '.', args[0]), depth + 1)
        elif command in ('shot', 'screenshot'):
            self.shot(args[0])
        elif command == 'put_save':
            self.put_save(args)
        elif command == 'wait':
            self.target.flush()
            self.wait_ms(int(args[0]))
        elif command == 'waitstable':
            self.waitstable(int(args[0]), int(args[1]), int(args[2]) if len(args) > 2 else 300)
        elif command == 'waitpixel':
            self.waitpixel(args)
        elif command == 'waitregion':
            self.waitregion(args)
        elif command in ('expect_changed', 'expect_same'):
            self.expect(command, args)
        elif command == 'expect_closed':
            self.expect_closed(int(args[0]) if args else 5000)
        elif command in INPUT_COMMANDS:
            self.target.do_input(command, args)
        else:
            raise ToolError('unknown scenario command %r' % command)

    def put_save(self, args):
        """
        put_save FIXTURE SLOT: a copy of eitools/fixtures/saves/FIXTURE becomes the slot data/saves/SLOT, so the slot
        holds only the files of the save of game.exe (no remake.json of the remake).
        """
        if len(args) != 2:
            raise ToolError('put_save needs a fixture and a slot')
        fixture_name, slot_name = args
        if os.sep in fixture_name or os.sep in slot_name or slot_name in ('', '.', '..'):
            raise ToolError('put_save takes names, not paths')
        source_path = os.path.join(FIXTURE_SAVES_DIR, fixture_name)
        if not os.path.isdir(source_path):
            raise ToolError('no fixture %s' % source_path)
        dest_path = os.path.join(self.target.data_path, 'saves', slot_name)
        if os.path.isdir(dest_path):
            shutil.rmtree(dest_path)
        shutil.copytree(source_path, dest_path)
        self.log('put_save %s -> %s' % (fixture_name, dest_path))

    def phase(self, name):
        """
        A new phase of the coverage trace: its name goes to EIDRIVE_PHASE_FILE, which eitrace.exe
        (remake/re/eitrace) reads. Only for the original, a no-op without the variable.
        """
        self.target.flush()
        phase_file_path = os.environ.get('EIDRIVE_PHASE_FILE')
        if phase_file_path and self.target.name == 'original':
            with open(phase_file_path, 'w') as f:
                f.write(name + '\n')
        self.log('phase %s' % name)

    def shot(self, name):
        self.target.flush()
        path = os.path.join(self.out_dir, name + '.png')
        self.target.shot(path)
        self.shots.append({'name': name, 'file': os.path.basename(path),
                           't': round(time.time() - self.started_at, 2)})
        self.log('shot %s' % name)

    def grab(self):
        return self.target.x11().grab(self.target.window())

    def waitstable(self, settle_ms, timeout_ms, poll_ms):
        self.target.flush()
        deadline = time.time() + timeout_ms / 1000.0
        previous = self.grab()
        stable_since = time.time()
        while time.time() < deadline:
            self.check_timeout()
            time.sleep(poll_ms / 1000.0)
            current = self.grab()
            diff = ImageChops.difference(previous.convert('RGB'), current.convert('RGB'))
            histogram = diff.convert('L').histogram()
            mean = sum(value * count for value, count in enumerate(histogram)) / float(
                sum(histogram))
            if mean > 0.5:
                stable_since = time.time()
            previous = current
            if (time.time() - stable_since) * 1000.0 >= settle_ms:
                self.log('stable after %.1f s' % (time.time() - self.started_at))
                return
        self.error('the picture did not become stable in %d ms' % timeout_ms)

    @staticmethod
    def parse_color(text):
        parts = text.replace(',', ' ').split()
        if len(parts) != 3:
            raise ToolError('a color is R,G,B, got %r' % text)
        return tuple(int(part) for part in parts)

    @staticmethod
    def color_close(pixel, color, tolerance):
        return all(abs(int(a) - int(b)) <= tolerance for a, b in zip(pixel, color))

    def waitpixel(self, args):
        x, y, color = int(args[0]), int(args[1]), self.parse_color(args[2])
        tolerance = int(args[3]) if len(args) > 3 else 40
        timeout = int(args[4]) if len(args) > 4 else 60000
        self.target.flush()
        deadline = time.time() + timeout / 1000.0
        while time.time() < deadline:
            self.check_timeout()
            pixel = self.grab().getpixel((x, y))
            if self.color_close(pixel, color, tolerance):
                self.log('pixel %d,%d is %s' % (x, y, (pixel,)))
                return
            time.sleep(0.3)
        self.error('pixel %d,%d did not become %s in %d ms' % (x, y, color, timeout))

    def waitregion(self, args):
        x, y, w, h, color = (int(args[0]), int(args[1]), int(args[2]), int(args[3]),
                             self.parse_color(args[4]))
        tolerance = int(args[5]) if len(args) > 5 else 25
        fraction = float(args[6]) if len(args) > 6 else 0.5
        timeout = int(args[7]) if len(args) > 7 else 60000
        self.target.flush()
        deadline = time.time() + timeout / 1000.0
        while time.time() < deadline:
            self.check_timeout()
            data = self.grab().crop((x, y, x + w, y + h)).tobytes()
            total = len(data) // 3
            matched = 0
            for i in range(0, len(data), 3):
                if (abs(data[i] - color[0]) <= tolerance and
                        abs(data[i + 1] - color[1]) <= tolerance and
                        abs(data[i + 2] - color[2]) <= tolerance):
                    matched += 1
            if matched >= fraction * total:
                self.log('region %d,%d %dx%d matches %s at %.0f%%'
                         % (x, y, w, h, color, 100.0 * matched / total))
                return
            time.sleep(0.3)
        self.error('region %d,%d %dx%d did not become %s in %d ms'
                   % (x, y, w, h, color, timeout))

    def expect(self, command, args):
        name = args[0]
        threshold = float(args[1]) if len(args) > 1 else 0.5
        reference = os.path.join(self.out_dir, name + '.png')
        current = os.path.join(self.out_dir, '.current-%s.png' % self.target.name)
        self.target.flush()
        self.target.shot(current)
        diff = image_diff(reference, current)
        os.unlink(current)
        if diff is None:
            self.error('%s: no reference picture %r' % (command, name))
            return
        if command == 'expect_changed':
            ok = diff >= threshold
            text = 'changed from %s by %.3f (threshold %.3f)' % (name, diff, threshold)
        else:
            ok = diff <= threshold
            text = 'the same as %s within %.3f (threshold %.3f)' % (name, diff, threshold)
        self.checks.append({'check': command, 'reference': name, 'diff': round(diff, 4),
                            'threshold': threshold, 'ok': ok, 'text': text})
        self.log(('ok: ' if ok else 'FAIL: ') + text)
        if not ok:
            self.errors.append(text)

    def expect_closed(self, timeout_ms):
        """
        The window of the game must go away within the time: the game has closed itself.
        """
        self.target.flush()
        deadline = time.time() + timeout_ms / 1000.0
        ok = False
        while True:
            self.check_timeout()
            try:
                self.target.window(refresh=True)
            except ToolError:
                ok = True
                break
            if time.time() >= deadline:
                break
            time.sleep(0.2)
        text = ('the window is closed' if ok else
                'the window is still open after %d ms' % timeout_ms)
        self.checks.append({'check': 'expect_closed', 'ok': ok, 'text': text})
        self.log(('ok: ' if ok else 'FAIL: ') + text)
        if not ok:
            self.errors.append(text)

    def finish(self):
        self.target.flush()
        result = {
            'target': self.target.name,
            'display': self.target.display,
            'backend': self.target.backend_name,
            'scenario_started': time.strftime('%Y-%m-%d %H:%M:%S',
                                              time.localtime(self.started_at)),
            'duration': round(time.time() - self.started_at, 2),
            'shots': self.shots,
            'checks': self.checks,
            'errors': self.errors,
        }
        with open(os.path.join(self.out_dir, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        self.log('done, %d errors' % len(self.errors))
        self._log_file.write('errors: %d\n' % len(self.errors))
        for text in self.errors:
            self._log_file.write('ERROR %s\n' % text)
        self._log_file.close()
        return result

#------------------------------------------------------------------------------
# the report of the comparison
#------------------------------------------------------------------------------


def build_report(out_dir, scenario, results):
    path = os.path.join(out_dir, 'index.html')
    names = []
    for result in results.values():
        for shot in result['shots']:
            if shot['name'] not in names:
                names.append(shot['name'])
    rows = []
    for name in names:
        files = {}
        for target, result in results.items():
            for shot in result['shots']:
                if shot['name'] == name:
                    files[target] = os.path.join(target, shot['file'])
        original = files.get('original')
        demo = files.get('demo')
        diff = image_diff(os.path.join(out_dir, original), os.path.join(out_dir, demo)) \
            if original and demo else None
        diff_text = '—' if diff is None else '%.2f' % diff
        cells = []
        for target in ('original', 'demo'):
            file_path = files.get(target)
            if file_path:
                cells.append('<a href="%s"><img src="%s" alt="%s"></a>' % (
                    file_path, file_path, name))
            else:
                cells.append('<span class="missing">no picture</span>')
        rows.append('<tr><th>%s</th><td>%s</td><td>%s</td><td class="diff">%s</td></tr>'
                    % (name, cells[0], cells[1], diff_text))
    errors = []
    checks = []
    for target, result in results.items():
        for error in result['errors']:
            errors.append('<li><b>%s</b>: %s</li>' % (target, error))
        for check in result['checks']:
            checks.append('<li class="%s"><b>%s</b>: %s</li>' % (
                'ok' if check['ok'] else 'fail', target, check['text']))
    html = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>eidrive: %(scenario)s</title>
<style>
:root { color-scheme: dark; --bg: #16181d; --panel: #1f232b; --text: #e6e8ee;
        --muted: #9aa3b2; --line: #303746; --ok: #75c37a; --fail: #e07171; --accent: #6ea8fe; }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px; background: var(--bg); color: var(--text);
       font: 14px/1.45 system-ui, sans-serif; }
h1 { margin: 0 0 4px; font-size: 22px; }
h2 { margin: 28px 0 10px; font-size: 17px; }
p, li { color: var(--muted); }
a { color: var(--accent); }
table { width: 100%%; border-collapse: collapse; background: var(--panel); border-radius: 10px;
        overflow: hidden; }
th, td { border-bottom: 1px solid var(--line); padding: 8px; vertical-align: top; }
thead th { text-align: left; background: #262b35; }
tbody th { width: 130px; text-align: left; color: var(--muted); font-weight: 600; }
td img { width: 100%%; display: block; border-radius: 6px; background: #000; }
td.diff { width: 70px; text-align: center; color: var(--muted); font-variant-numeric: tabular-nums; }
.missing { color: var(--fail); font-size: 12px; }
.errors { padding: 12px 16px; background: var(--panel); border-left: 3px solid var(--fail);
          border-radius: 6px; }
.errors li { color: var(--fail); }
.checks { padding: 12px 16px; background: var(--panel); border-radius: 6px; }
.checks li.ok { color: var(--ok); }
.checks li.fail { color: var(--fail); }
.meta { display: flex; gap: 24px; flex-wrap: wrap; color: var(--muted); }
.meta b { color: var(--text); }
@media (max-width: 760px) {
  body { padding: 16px; }
  thead { display: none; }
  table, tbody, tr, th, td { display: block; width: 100%%; }
  tr { border-bottom: 2px solid var(--line); padding-bottom: 12px; margin-bottom: 12px; }
  tbody th::before { content: "step: "; }
  td.diff::before { content: "diff: "; }
}
</style>
</head>
<body>
<h1>eidrive: %(scenario)s</h1>
<div class="meta">%(meta)s</div>
<h2>Кадры</h2>
<p>Слева — оригинал, справа — демо. Число — средняя абсолютная разница пикселей (0 — одинаково,
255 — совсем разное); у оригинала и демо разные материалы и освещение, поэтому важно смотреть на
раскладку и поведение, а не на ноль.</p>
<table>
<thead><tr><th>шаг</th><th>оригинал</th><th>демо</th><th>diff</th></tr></thead>
<tbody>%(rows)s</tbody>
</table>
%(checks)s
%(errors)s
</body>
</html>
""" % {
        'scenario': scenario,
        'meta': ''.join('<span><b>%s</b>: %s</span>' % (target, result['display'])
                        for target, result in results.items()),
        'rows': '\n'.join(rows),
        'checks': ('<h2>Проверки</h2><ul class="checks">%s</ul>' % ''.join(checks))
                  if checks else '',
        'errors': ('<h2>Ошибки</h2><ul class="errors">%s</ul>' % ''.join(errors))
                  if errors else '<h2>Ошибки</h2><p>нет</p>',
    }
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path

#------------------------------------------------------------------------------
# the commands of eidrive.py
#------------------------------------------------------------------------------


def cmd_start(target, args):
    return target.start()


def cmd_stop(target, args):
    return target.stop()


def cmd_restart(target, args):
    return target.restart()


def cmd_status(target, args):
    return target.status()


def cmd_shot(target, args):
    if not args:
        raise ToolError('shot needs a file')
    target.shot(args[0])
    log('saved %s' % args[0])
    return 0


def cmd_record(target, args):
    if len(args) < 2:
        raise ToolError('record needs DIR SECONDS [FPS]')
    path = target.record(args[0], float(args[1]), float(args[2]) if len(args) > 2 else 10.0)
    log('video %s' % path)
    return 0


def cmd_input(target, args):
    if not args:
        raise ToolError('input needs a command')
    groups = []
    current = []
    for word in args:
        if word == ';':
            if current:
                groups.append(current)
            current = []
        else:
            current.append(word)
    if current:
        groups.append(current)
    for group in groups:
        target.do_input(group[0], group[1:])
    target.flush()
    return 0


def cmd_run(target, scenario, out=None):
    scenario = scenario or ''
    if not scenario:
        raise ToolError('run needs a scenario')
    out = out or os.path.join(OUT_DIR, os.path.splitext(os.path.basename(scenario))[0],
                              target.name)
    run = Run(target, out)
    result = run.run(scenario)
    log('results %s' % os.path.join(out, 'results.json'))
    return 0 if not result['errors'] else 1


def arrange_side_by_side(original, demo):
    """
    The window of the original on the left of the display, the window of the demo on the right.
    """
    session = original.x11()
    width, height = session.screen_size()
    original_window = original.window(refresh=True)
    demo_window = demo.window(refresh=True)
    top = max(0, (height - max(original_window['h'], demo_window['h'])) // 2)
    # both windows on one screen next to each other: the pointer does not cross a monitor
    demo_x = original_window['w'] + 80
    if demo_x + demo_window['w'] > width:
        demo_x = max(0, width - demo_window['w'])
    session.ewmh_move_window(original_window['id'], 0, top)
    demo.x11().ewmh_move_window(demo_window['id'], demo_x, top)
    session.ewmh_activate_window(original_window['id'])
    demo.x11().ewmh_activate_window(demo_window['id'])
    time.sleep(0.6)
    original.forget_window()
    demo.forget_window()
    original_window = original.window(refresh=True)
    demo_window = demo.window(refresh=True)
    log('screen %dx%d: original at %d,%d, demo at %d,%d'
        % (width, height, original_window['x'], original_window['y'],
           demo_window['x'], demo_window['y']))
    if (abs(original_window['x']) > 8 or abs(original_window['y'] - top) > 8 or
            abs(demo_window['x'] - demo_x) > 8 or abs(demo_window['y'] - top) > 8):
        log('warning: the window manager did not put the windows side by side')


def run_parallel(targets, out, scenario, abort, grace=None):
    """
    The same scenario on all the targets at once, each in its own thread.
    grace: seconds to wait for the other target after the first one has finished.
    """
    results = {}
    failures = []
    barrier = RunBarrier(len(targets))
    runs = [Run(target, os.path.join(out, target.name), abort=abort, barrier=barrier)
            for target in targets]

    def worker(run):
        try:
            results[run.target.name] = run.run(scenario)
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=worker, args=(run,), name=run.target.name, daemon=True)
               for run in runs]
    for thread in threads:
        thread.start()
    first_finished = None
    try:
        while any(thread.is_alive() for thread in threads):
            if any(run.finished.is_set() for run in runs):
                if first_finished is None:
                    first_finished = time.time()
                if grace is not None and time.time() - first_finished >= grace:
                    abort.set('the other target did not finish in %.1f s' % grace)
                    break
            time.sleep(0.05)
    except BaseException:
        abort.set('the comparison was interrupted')
        raise
    for thread in threads:
        thread.join()
    if failures:
        raise failures[0]
    return results


def aborted_result(target, reason):
    """
    The result of a target which was not run because the other target stopped the comparison.
    """
    return {
        'target': target.name,
        'display': target.display,
        'backend': target.backend_name,
        'scenario_started': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        'duration': 0.0,
        'shots': [],
        'checks': [],
        'errors': [reason],
    }


def cmd_compare(args):
    scenario = args.scenario
    if not os.path.exists(scenario):
        raise ToolError('no scenario %r' % scenario)
    scenario_timeout(scenario)
    if args.display and not args.screen:
        raise ToolError('--display is only used with --screen')
    out = args.out or os.path.join(OUT_DIR, os.path.splitext(os.path.basename(scenario))[0])
    os.makedirs(out, exist_ok=True)
    if args.screen:
        display = args.display or os.environ.get('DISPLAY')
        if not display:
            raise ToolError('--screen needs DISPLAY or --display')
        if not display_works(display):
            raise ToolError('display %s is not available' % display)
        targets = [Target('original', display=display, shared=True),
                   Target('demo', display=display, shared=True)]
    else:
        targets = [Target('original'), Target('demo')]
    if args.reset_settings or scenario_resets_settings(scenario):
        if args.no_restart:
            raise ToolError('the settings can be reset only with a restart of the games')
        for target in targets:
            target.reset_settings()
    abort = RunAbort()
    results = {}
    # the scenarios with "sync" keep the runs together; the first one which ends leaves
    # finish_timeout seconds to the others, then the rest is stopped
    grace = args.finish_timeout if args.screen and scenario_has_sync(scenario) else None
    try:
        if args.screen:
            for target in targets:
                if not args.no_restart:
                    target.restart()
                else:
                    target.start()
                target.forget_window()
            arrange_side_by_side(targets[0], targets[1])
            results = run_parallel(targets, out, scenario, abort, grace=grace)
        else:
            for target in targets:
                if abort.event.is_set():
                    try:
                        target.stop()
                    except Exception as exc:
                        log('stop %s: %s' % (target.name, exc))
                    results[target.name] = aborted_result(target, 'not run: %s' % abort.reason)
                    continue
                if not args.no_restart:
                    target.restart()
                else:
                    target.start()
                target.forget_window()
                results[target.name] = Run(target, os.path.join(out, target.name),
                                           abort=abort).run(scenario)
    finally:
        if args.keep:
            log('the targets are left running (--keep)')
        else:
            for target in targets:
                try:
                    target.stop()
                except Exception as exc:
                    log('stop %s: %s' % (target.name, exc))
    report = build_report(out, os.path.basename(scenario), results)
    log('report %s' % report)
    for name, result in results.items():
        log('%s: %d shots, %d errors' % (name, len(result['shots']), len(result['errors'])))
    return 0 if not any(result['errors'] for result in results.values()) else 1

#------------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        prog='eidrive.py', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='target', required=True)
    for name in ('original', 'demo', 'compare'):
        if name == 'compare':
            p = sub.add_parser('compare', help='run a scenario on both targets and build a report')
            p.add_argument('scenario')
            p.add_argument('--out')
            p.add_argument('--reset-settings', dest='reset_settings', action='store_true',
                           help='remove the settings of both games first (a backup stays in run/)')
            p.add_argument('--no-restart', action='store_true',
                           help='do not restart the targets, they must be at the main menu')
            p.add_argument('--screen', action='store_true',
                           help='run both targets on the desktop display: original left, demo right')
            p.add_argument('--display',
                           help='the display for --screen, default $DISPLAY')
            p.add_argument('--keep', action='store_true',
                           help='do not stop the targets after the scenario')
            p.add_argument('--finish-timeout', dest='finish_timeout', type=float, default=2.0,
                           metavar='SECONDS',
                           help='wait this long for the other target when one scenario ends, '
                                'then stop both (2 by default)')
            continue
        p = sub.add_parser(name, help='the %s target' % name)
        commands = p.add_subparsers(dest='command', required=True)
        for command in ('start', 'stop', 'restart', 'status', 'reset-settings'):
            commands.add_parser(command)
        shot = commands.add_parser('shot')
        shot.add_argument('file')
        record = commands.add_parser('record')
        record.add_argument('dir')
        record.add_argument('seconds', type=float)
        record.add_argument('fps', type=float, nargs='?', default=10.0)
        inp = commands.add_parser('input')
        inp.add_argument('words', nargs=argparse.REMAINDER)
        run = commands.add_parser('run')
        run.add_argument('scenario')
        run.add_argument('--out')
    return parser


def main(argv):
    if hasattr(signal, 'SIGPIPE'):
        # "eidrive.py ... | head" must end quietly, as any other command line tool
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.target == 'compare':
        try:
            return cmd_compare(args)
        except ToolError as exc:
            log('error: %s' % exc)
            return 2
        except KeyboardInterrupt:
            return 130
    target = Target(args.target)
    try:
        if args.command == 'start':
            return cmd_start(target, None)
        if args.command == 'stop':
            return cmd_stop(target, None)
        if args.command == 'restart':
            return cmd_restart(target, None)
        if args.command == 'status':
            return cmd_status(target, None)
        if args.command == 'reset-settings':
            target.reset_settings()
            return 0
        if args.command == 'shot':
            return cmd_shot(target, [args.file])
        if args.command == 'record':
            return cmd_record(target, [args.dir, str(args.seconds), str(args.fps)])
        if args.command == 'input':
            return cmd_input(target, args.words)
        if args.command == 'run':
            return cmd_run(target, args.scenario, args.out)
        raise ToolError('unknown command %r' % args.command)
    except ToolError as exc:
        log('error: %s' % exc)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
