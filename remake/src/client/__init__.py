"""
Kivy client of the remake. Kivy must not read the command line of the program, see client.main,
the variable is set here because it works only before the first import of kivy.

The swap of the buffers does not wait for the vertical sync: on Wayland (Xwayland) a window which the
compositor does not redraw makes the waiting swap block for up to a second, so the demo ran at one frame
a second; the clock of Kivy keeps the rate at MAX_FPS instead. KIVY_VSYNC=1 brings the sync back.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import os

#------------------------------------------------------------------------------

os.environ.setdefault('KIVY_NO_ARGS', '1')
os.environ.setdefault('KCFG_GRAPHICS_VSYNC', os.environ.get('KIVY_VSYNC', '0'))
os.environ.setdefault('KCFG_GRAPHICS_MAXFPS', '60')
