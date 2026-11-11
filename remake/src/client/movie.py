"""
The movies of PlayMovie( name ) (0xc1): game.exe plays data/Movies/<name> (Bink) over the whole screen
by 5db3f0 and the game waits until it ends or Escape or Space stops it. Here the video provider of Kivy
plays it over the screen while the world stands; without a provider which can read the file nothing is
played. ffpyplayer brings the Bink demuxer and the binkvideo and binkaudio_rdft decoders; in the full
screen mode 5db3f0 stretches the movie to the width of the screen and centers it by height, which is the
"contain" fit mode of an Image.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import os

from kivy.core.video import Video as CoreVideo
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.video import Video

#------------------------------------------------------------------------------

MOVIES_DIR_NAME = 'movies'

NULL_PROVIDER = 'VideoNull'

# the keys which stop a movie: WM_KEYDOWN with VK_ESCAPE or VK_SPACE (5db3f0), Enter is not among them
STOP_KEYS = ('escape', 'spacebar')

#------------------------------------------------------------------------------


class MovieVideo(Video):
    """
    Kivy's Video is an Image too: setting the source to a .bik makes ImageLoader try to read the file as
    a picture and log "Error loading". The frames of the movie come from the provider into self.texture,
    so the preview of the source is not needed.
    """

    def texture_update(self, *largs):
        return


def movie_file_path(data_dir_path, name):
    """
    The file of the movie in data/movies, the name compared without the case of the letters; None when missing.
    """
    dir_path = os.path.join(data_dir_path, MOVIES_DIR_NAME)
    if not os.path.isdir(dir_path):
        return None
    wanted = os.path.basename(name.replace('\\', '/')).lower()
    for file_name in os.listdir(dir_path):
        if file_name.lower() == wanted:
            return os.path.join(dir_path, file_name)
    return None


class MoviePlayer(object):

    def __init__(self, layout, volume=1.0, windowed=False):
        self.layout = layout
        # 0.0 is the --no-sound of the client; 5db3f0 sets the volume of the options as a percentage
        self.volume = volume
        # game.exe mode 1: the movie at its own size in the middle of the window (mode 2 is full screen)
        self.windowed = windowed
        self.video = None
        # the fullscreen holder of the movie in the windowed mode
        self.container = None
        # called once when the movie ends or a key stops it, None when nobody waits for it
        self.on_end_callback = None

    @property
    def is_open(self):
        return self.video is not None

    @staticmethod
    def can_play():
        """
        Kivy gives VideoNull when it has no real provider (gstplayer, ffpyplayer): it plays nothing and never ends.
        """
        return CoreVideo is not None and CoreVideo.__name__ != NULL_PROVIDER

    def play(self, file_path, on_end=None):
        """
        False when no provider of Kivy can play videos; on_end is called once when the movie ends or
        a key stops it (5db3f0 returns to the caller in both cases).
        """
        if not self.can_play() or file_path is None:
            return False
        self.stop()
        self.on_end_callback = on_end
        if self.windowed:
            self.video = MovieVideo(source=file_path, state='play', allow_stretch=True, size_hint=(None, None),
                                    size=(0, 0), pos_hint={'center_x': 0.5, 'center_y': 0.5},
                                    volume=self.volume)
            self.video.bind(texture=self.fit_window)
            self.container = FloatLayout(size_hint=(1.0, 1.0), pos_hint={'x': 0.0, 'y': 0.0})
            self.container.add_widget(self.video)
            self.layout.add_widget(self.container)
        else:
            self.video = MovieVideo(source=file_path, state='play', allow_stretch=True, size_hint=(1.0, 1.0),
                                    pos_hint={'x': 0.0, 'y': 0.0}, volume=self.volume)
            self.layout.add_widget(self.video)
        self.video.bind(eos=self.on_end)
        if _Debug:
            print('movie %s' % file_path)
        return True

    def fit_window(self, video, texture):
        """
        The windowed mode of 5db3f0 draws the movie at its own size in the middle of the screen.
        """
        if texture is None:
            return
        video.size = texture.size

    def on_end(self, video, ended):
        if ended:
            self.stop()
            self.finish()

    def on_key(self, name):
        if self.video is None or name not in STOP_KEYS:
            return False
        self.stop()
        self.finish()
        return True

    def finish(self):
        callback = self.on_end_callback
        self.on_end_callback = None
        if callback is not None:
            callback()

    def stop(self):
        if self.video is None:
            return
        self.video.state = 'stop'
        if self.container is not None:
            self.layout.remove_widget(self.container)
            self.container = None
        else:
            self.layout.remove_widget(self.video)
        self.video.unload()
        self.video = None
