"""
The sounds of sfx.res played by Kivy (SDL2): engine.sounds picks them, here they are unpacked into a cache
folder (the audio of Kivy reads only files), started at the volume of their distance to the listener
and kept by their channels. The sounds of the spells (engine.magic.SpellSound) play at their places, the loops
of them sound while the client gives them every frame.

SDL picks the pulseaudio driver first; on a system with PipeWire it hangs when the audio opens after
the window of Kivy, so the drivers are tried in the order of AUDIO_DRIVERS unless SDL_AUDIODRIVER is set
(SDL reads it when the first sound loads).

SDL_mixer of Kivy mixes 8 sounds at once; the voices, the steps, the loops of the spells and the land and
the single sounds of engine.ambient need more, so after the first sound the mixer gets MIXER_CHANNELS
(Mix_AllocateChannels of the SDL2_mixer which Kivy loaded).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import ctypes
import glob
import math
import os

import kivy
from kivy.core.audio import SoundLoader

#------------------------------------------------------------------------------

from engine import sounds

#------------------------------------------------------------------------------

CACHE_DIR_PATH = os.path.join(os.path.expanduser('~'), '.cache', 'ei-remake', 'sfx')

AUDIO_DRIVERS = 'pipewire,alsa,pulseaudio'

MIXER_CHANNELS = 32

#------------------------------------------------------------------------------


class AudioPlayer(object):

    def __init__(self, game_data, cache_dir_path=CACHE_DIR_PATH, enabled=True):
        os.environ.setdefault('SDL_AUDIODRIVER', AUDIO_DRIVERS)
        self.game_data = game_data
        self.cache_dir_path = cache_dir_path
        self.enabled = enabled
        # the volume_sfx and the volume_voice of the options, 0..1
        self.volume_sfx = 1.0
        self.volume_voice = 1.0
        # the loaded Kivy sounds by the name of sfx.res, None for a missing file
        self.loaded = {}
        # (unit id, channel) -> the Kivy sound which plays on it
        self.channels = {}
        # a key of the client -> the Kivy sound which plays for it (the warning of the enchantments)
        self.keyed = {}
        # the name -> the Kivy sound of a loop which plays
        self.loops = {}
        # the name -> the Kivy sound of a loop of engine.ambient
        self.ambient_loops = {}
        self.listener = (0.0, 0.0, 0.0)
        # 6b7cd0: the channels of the single sounds of the world, (the Kivy sound, its priority) or None
        self.world_channels = [None] * sounds.CHANNELS
        # the channels of the mixer are set after the first sound opened the audio
        self.mixer_ready = False

    def sound_file_path(self, name):
        """
        The .wav file of the sound in the cache, unpacked on the first use; None when sfx.res has no such sound.
        """
        real_name = self.game_data.sound_name(name)
        if real_name is None:
            return None
        file_path = os.path.join(self.cache_dir_path, *real_name.split('/'))
        if not os.path.isfile(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(self.game_data.sound(real_name))
        return file_path

    def load(self, name):
        if name not in self.loaded:
            file_path = self.sound_file_path(name)
            self.loaded[name] = SoundLoader.load(file_path) if file_path else None
            if self.loaded[name] is not None and not self.mixer_ready:
                self.mixer_ready = True
                allocate_channels(MIXER_CHANNELS)
            if file_path is None:
                if _Debug:
                    print('no sound %s' % name)
        return self.loaded[name]

    def take_world_channel(self, kivy_sound, priority):
        """
        6b7cd0: a channel for a single sound of the world; the sound of the lowest priority is cut for a sound
        above it. -1 when the sound must not play. A file has one Kivy sound, its old channel is freed.
        """
        channels = self.world_channels
        for index, busy in enumerate(channels):
            if busy is not None and (busy[0] is kivy_sound or busy[0].state != 'play'):
                channels[index] = None
        index = sounds.take_channel([None if busy is None else busy[1] for busy in channels], priority)
        if index >= 0 and channels[index] is not None:
            channels[index][0].stop()
            channels[index] = None
        return index

    def on_unit_sounds(self, unit, started, stopped):
        """
        engine.unit.Unit.on_sounds: stops the channels, starts the sounds at the position of the unit.
        """
        for channel in stopped:
            playing = self.channels.pop((id(unit), channel), None)
            if playing is not None:
                playing.stop()
            unit.busy_channels.discard(channel)
        if not self.enabled or getattr(unit, 'unseen', False):
            # 534b60: a unit which the player does not see is silent
            return
        distance = self.distance_to((unit.x, unit.y))
        volume = sounds.volume(distance, height=self.listener[2])
        for sound in started:
            if volume <= 0.0:
                continue
            kivy_sound = self.load(sound.name)
            if kivy_sound is None:
                continue
            index = self.take_world_channel(kivy_sound, sound.priority)
            if index < 0:
                continue
            kivy_sound.volume = volume * self.volume_voice
            kivy_sound.stop()
            kivy_sound.play()
            self.world_channels[index] = (kivy_sound, sound.priority)
            if _Debug:
                print('play %s at %.1f, volume %.2f' % (sound.name, distance, volume))
            if sound.channel is not None:
                self.channels[(id(unit), sound.channel)] = kivy_sound
                unit.busy_channels.add(sound.channel)

    def play_interface(self, name):
        """
        A sound of a button or of a window of the interface ("buttons/Camp/Put_on.wav") at volume_sfx.
        """
        if not self.enabled:
            return
        kivy_sound = self.load(name)
        if kivy_sound is None:
            return
        kivy_sound.volume = self.volume_sfx
        kivy_sound.stop()
        kivy_sound.play()

    def distance_to(self, position):
        """
        6b7f80: the distance on the map, the heights do not count.
        """
        return math.hypot(position[0] - self.listener[0], position[1] - self.listener[1])

    def play_sound(self, sound, key=None):
        """
        An engine.magic.SpellSound once at its place; with the key it does not start while the previous
        sound of the key plays.
        """
        if not self.enabled:
            return
        if key is not None:
            playing = self.keyed.get(key)
            if playing is not None and playing.state == 'play':
                return
        volume = sounds.volume(self.distance_to(sound.position), sound.near, sound.far, self.listener[2])
        if volume <= 0.0:
            return
        kivy_sound = self.load(sound.name)
        if kivy_sound is None:
            return
        priority = getattr(sound, 'priority', sounds.PRIORITY_LOW)
        index = self.take_world_channel(kivy_sound, priority)
        if index < 0:
            return
        kivy_sound.loop = False
        kivy_sound.volume = volume * self.volume_sfx
        kivy_sound.stop()
        kivy_sound.play()
        self.world_channels[index] = (kivy_sound, priority)
        if key is not None:
            self.keyed[key] = kivy_sound
        if _Debug:
            print('play %s volume %.2f' % (sound.name, volume))

    def set_loops(self, loops):
        """
        The engine.magic.SpellSound which loop now: the new ones start, the gone ones stop. One file
        plays once, at the volume of its nearest place.
        """
        volumes = {}
        if self.enabled:
            for sound in loops:
                volume = sounds.volume(self.distance_to(sound.position), sound.near, sound.far, self.listener[2])
                volumes[sound.name] = max(volumes.get(sound.name, 0.0), volume)
        for name, kivy_sound in list(self.loops.items()):
            if volumes.get(name, 0.0) <= 0.0:
                kivy_sound.stop()
                kivy_sound.loop = False
                del self.loops[name]
        for name, volume in volumes.items():
            if volume <= 0.0:
                continue
            kivy_sound = self.loops.get(name)
            if kivy_sound is None:
                kivy_sound = self.load(name)
                if kivy_sound is None:
                    continue
                kivy_sound.loop = True
                kivy_sound.volume = volume * self.volume_sfx
                kivy_sound.play()
                self.loops[name] = kivy_sound
            else:
                kivy_sound.volume = volume * self.volume_sfx

    def set_ambient(self, loops):
        """
        The loops of engine.ambient {name: volume 0..1}: the new ones start, the gone ones stop, the others take
        the volume (6b8210, 6b9460, 6b90c0).
        """
        if not self.enabled:
            loops = {}
        for name, kivy_sound in list(self.ambient_loops.items()):
            if name not in loops:
                kivy_sound.stop()
                kivy_sound.loop = False
                del self.ambient_loops[name]
        for name, volume in loops.items():
            kivy_sound = self.ambient_loops.get(name)
            if kivy_sound is None:
                kivy_sound = self.load(name)
                if kivy_sound is None:
                    continue
                kivy_sound.loop = True
                kivy_sound.volume = volume * self.volume_sfx
                kivy_sound.play()
                self.ambient_loops[name] = kivy_sound
            else:
                kivy_sound.volume = volume * self.volume_sfx

    def update(self, units, listener):
        """
        Frees the channels whose sounds ended, takes the place of the listener.
        """
        self.listener = listener
        units_by_id = {id(u): u for u in units}
        for key, kivy_sound in list(self.channels.items()):
            if kivy_sound.state != 'play':
                del self.channels[key]
                unit = units_by_id.get(key[0])
                if unit is not None:
                    unit.busy_channels.discard(key[1])

#------------------------------------------------------------------------------


def allocate_channels(count):
    """
    Mix_AllocateChannels of the SDL2_mixer of the Kivy wheel (Kivy.libs next to the package); False when
    there is no such library (a Kivy of the system).
    """
    libs_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(kivy.__file__))), 'Kivy.libs')
    for lib_path in glob.glob(os.path.join(libs_dir_path, 'libSDL2_mixer*.so*')):
        try:
            mixer = ctypes.CDLL(lib_path)
            allocated = mixer.Mix_AllocateChannels(count)
        except (OSError, AttributeError):
            continue
        if _Debug:
            print('the mixer has %d channels' % allocated)
        return True
    return False
