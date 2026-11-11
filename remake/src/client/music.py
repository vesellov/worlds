"""
The music in the client: engine.music.MusicSystem chooses the tracks, here Kivy (SDL2) plays the files
data/stream/<name>.mp3. A track plays once; a fading one goes down to silence in engine.music.FADE_MS.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import os

from kivy.core.audio import SoundLoader

#------------------------------------------------------------------------------

from engine import music

#------------------------------------------------------------------------------

STREAM_DIR_NAME = 'stream'

#------------------------------------------------------------------------------


class MusicPlayer(object):

    def __init__(self, game_data, enabled=True, volume=1.0):
        self.game_data = game_data
        self.enabled = enabled
        self.volume = volume
        self.system = music.MusicSystem(music.MusicTable.from_registry(game_data.registry(music.MUSIC_REG)))
        self.name = ''
        self.sound = None
        # the time when the fading started, None while the track does not fade
        self.fade_start = None

    def track_file_path(self, name):
        file_path = os.path.join(self.game_data.data_dir_path, STREAM_DIR_NAME, name + '.mp3')
        return file_path if os.path.isfile(file_path) else None

    def playing(self):
        """
        The name of the track which plays, '' when nothing plays or it fades.
        """
        if self.sound is None or self.fade_start is not None or self.sound.state != 'play':
            return ''
        return self.name

    def stop(self):
        if self.sound is not None:
            self.sound.stop()
        self.sound = None
        self.name = ''
        self.fade_start = None

    def set_volume(self, volume):
        """
        The volume_stream of the options (0..1); the track which plays takes it at once.
        """
        self.volume = volume
        if self.sound is not None and self.fade_start is None:
            self.sound.volume = volume

    def start(self, name):
        self.stop()
        self.name = name
        if not self.enabled:
            return
        file_path = self.track_file_path(name)
        if file_path is None:
            if _Debug:
                print('no music %s' % name)
            return
        self.sound = SoundLoader.load(file_path)
        if self.sound is not None:
            self.sound.volume = self.volume
            self.sound.play()

    def update(self, now_ms, fighting):
        for kind, name in self.system.update(now_ms, fighting, self.playing()):
            if kind == music.PLAY:
                self.start(name)
            elif kind == music.FADE and self.sound is not None and self.fade_start is None:
                self.fade_start = now_ms
        if self.fade_start is not None:
            left = 1.0 - (now_ms - self.fade_start) / music.FADE_MS
            if left <= 0.0 or self.sound is None:
                self.stop()
            else:
                self.sound.volume = self.volume * left
