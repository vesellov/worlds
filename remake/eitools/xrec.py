"""
Record an X window into PNG frames by XGetImage, then an .mp4 by gstreamer, without touching the input:
xrec.py <part of the window name> <output dir> <seconds> [fps]. SIGTERM ends the recording early and still
makes the video of the frames taken.
"""

#------------------------------------------------------------------------------

import ctypes
import os
import re
import signal
import subprocess
import sys
import time

from PIL import Image

#------------------------------------------------------------------------------

x11 = ctypes.cdll.LoadLibrary('libX11.so.6')
x11.XOpenDisplay.restype = ctypes.c_void_p
x11.XGetImage.restype = ctypes.c_void_p
x11.XGetImage.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
                          ctypes.c_ulong, ctypes.c_int]
x11.XDestroyImage = None


class XImage(ctypes.Structure):
    _fields_ = [('width', ctypes.c_int), ('height', ctypes.c_int), ('xoffset', ctypes.c_int), ('format', ctypes.c_int),
                ('data', ctypes.c_void_p), ('byte_order', ctypes.c_int), ('bitmap_unit', ctypes.c_int),
                ('bitmap_bit_order', ctypes.c_int), ('bitmap_pad', ctypes.c_int), ('depth', ctypes.c_int),
                ('bytes_per_line', ctypes.c_int), ('bits_per_pixel', ctypes.c_int)]

#------------------------------------------------------------------------------


def find_window(name_part):
    """
    The id and the size of the biggest window whose name has name_part.
    """
    text = subprocess.run(['xwininfo', '-root', '-tree'], capture_output=True, text=True).stdout
    best = None
    for line in text.splitlines():
        m = re.match(r'\s*(0x[0-9a-f]+) "([^"]*)".*?\s(\d+)x(\d+)\+', line)
        if m and name_part.lower() in m.group(2).lower():
            w, h = int(m.group(3)), int(m.group(4))
            if best is None or w * h > best[1] * best[2]:
                best = (int(m.group(1), 16), w, h)
    return best


def grab(display, window, width, height):
    p = x11.XGetImage(display, window, 0, 0, width, height, 0xffffffff, 2)
    if not p:
        return None
    img = XImage.from_address(p)
    raw = ctypes.string_at(img.data, img.bytes_per_line * img.height)
    return Image.frombuffer('RGB', (img.width, img.height), raw, 'raw', 'BGRX', img.bytes_per_line, 1)


# set by SIGTERM: stop taking frames and make the video
stop_requested = []


def request_stop(signum, frame):
    stop_requested.append(signum)


# the default handler of Xlib ends the process when the window is closed during XGetImage; this one stops the
# recording instead, so the video of the frames taken is still made
X_ERROR_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)


def on_x_error(display, event):
    stop_requested.append('x error')
    return 0


x_error_handler = X_ERROR_HANDLER(on_x_error)


def main():
    signal.signal(signal.SIGTERM, request_stop)
    x11.XSetErrorHandler(x_error_handler)
    name_part, out_dir_path, seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
    fps = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
    found = None
    deadline = time.time() + 60.0
    while found is None and time.time() < deadline:
        found = find_window(name_part)
        if found is None:
            time.sleep(0.5)
    if found is None:
        sys.exit('no window %r' % name_part)
    window, width, height = found
    print('window', hex(window), width, height, flush=True)
    os.makedirs(out_dir_path, exist_ok=True)
    display = x11.XOpenDisplay(None)
    start = time.time()
    count = 0
    while time.time() - start < seconds and not stop_requested:
        frame = grab(display, window, width, height)
        if frame is not None:
            frame.save(os.path.join(out_dir_path, 'f%05d.png' % count), compress_level=1)
            count += 1
        if not stop_requested:
            time.sleep(max(0.0, start + count / fps - time.time()))
    print('frames', count, flush=True)
    video_file_path = out_dir_path.rstrip('/') + '.mp4'
    subprocess.run(['gst-launch-1.0', '-q', 'multifilesrc', 'location=%s/f%%05d.png' % out_dir_path,
                    'caps=image/png,framerate=%d/1' % int(fps), '!', 'pngdec', '!', 'videoconvert', '!',
                    'video/x-raw,format=I420', '!', 'x264enc', '!', 'mp4mux', '!', 'filesink',
                    'location=%s' % video_file_path], check=True)
    print('video', video_file_path, flush=True)


if __name__ == '__main__':
    main()
