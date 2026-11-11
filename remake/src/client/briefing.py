"""
The window of a briefing: the title and the speech of the scene of the script in pages, the voices of the
phrases, the music "Briefing" of the allod (engine.music.set_briefing) and the callback of the end of the
briefing.

The voices are the entries "briefing/<name>/<number>.mp3" of res/speech.res (game.exe plays them with the
phrases of a briefing); they are unpacked into a cache folder because Kivy reads only files. A page of the
window holds whole utterances (engine.briefing.SpeechPage), its voices play one after another.

The keys: left and right turn the pages, Enter or a click on the button goes to the next page and closes
the last one, Escape closes the briefing.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math
import os

import numpy as np

from kivy.core.audio import SoundLoader

#------------------------------------------------------------------------------

from eilib import zones as zones_module
from engine import briefing as briefing_module
from engine import quat
from engine import trade as trade_module
from client import ui

#------------------------------------------------------------------------------

LAYER_NAME = 'briefing'

CACHE_DIR_PATH = os.path.join(os.path.expanduser('~'), '.cache', 'ei-remake', 'speech')

TEXT_COLOR = (228, 215, 167)
TITLE_COLOR = (255, 179, 49)
BUTTON_COLOR = (0.30, 0.26, 0.18, 1.0)
# the scene of the briefing is over the whole screen, the speech is in the box at its bottom
SCENE_SHADE = (0.0, 0.0, 0.0, 0.18)
PANEL_COLOR = (0.05, 0.05, 0.04, 0.82)

PANEL_RECT = (30, 300, 770, 590)
TITLE_BOX = (40, 8, 760, 44)
TEXT_BOX = (48, 340, 752, 556)
BUTTONS_TOP = 560
BACK_RECT = (300, 560, 380, 588)
NEXT_RECT = (420, 560, 500, 588)
BUTTON_LABELS = {'back': '<', 'next': '>'}

# where the actors stand when the map has no place for them: (right, forward) from the anchor
SLOT_OFFSETS = {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (1.4, 0.5), 3: (-1.4, 0.5), 4: (1.4, -0.8), 5: (-1.4, -0.8)}
# the actors of the map farther than this from the place of the scene stand on the slots
SCENE_RADIUS = 20.0
# the distance from the camera to the place of the actors when the map has no camera
FALLBACK_DISTANCE = 6.0

# the sounds of the window (buttons/Briefing/... of sfx.res)
SOUND_NEXT = 'buttons/Briefing/Next.wav'
SOUND_SCROLL = 'buttons/Briefing/Scroll.wav'

#------------------------------------------------------------------------------


class BriefingView(object):

    def __init__(self, app):
        self.app = app
        self.briefing = None
        self.pages = []
        self.page = 0
        self.shown_key = None
        self.voice_queue = []
        self.voice_sound = None
        self.utterances_by_voice = {}
        self.old_camera = None
        self.actors = []
        self.actor_views = []
        self.hidden_units = []
        # called when the briefing closes: the caller completes its variable (#OnBriefingComplete)
        self.on_close = None

    @property
    def is_open(self):
        return self.briefing is not None

    def __repr__(self):
        return 'BriefingView(%s)' % (self.briefing.name if self.briefing is not None else 'closed')

    def show(self, name):
        """
        The event "briefing <name>": the text of texts.res, the voices of res/speech.res and the music of
        the allod of the map.
        """
        app = self.app
        briefing = briefing_module.load(name, app.game_data.text)
        if briefing is None:
            if _Debug:
                print('no briefing %r' % name)
            return False
        zone = app.game_data.zone_of_map(app.map_name) if app.map_name else None
        allod = zone.allod if zone is not None else zones_module.DEFAULT_ALLOD
        if getattr(app, 'music', None) is not None:
            app.music.system.set_briefing(allod)
        self.briefing = briefing
        self.pages = briefing.speech_pages()
        self.page = 0
        self.shown_key = None
        self.voice_queue = []
        self.stop_voice()
        self.utterances_by_voice = {utterance.voice: utterance for utterance in briefing.utterances
                                    if utterance.voice is not None}
        self.start_scene()
        self.update()
        return True

    def close(self):
        if self.briefing is None:
            return
        self.stop_voice()
        self.end_scene()
        self.briefing = None
        self.pages = []
        self.shown_key = None
        self.app.interface.set_layer(LAYER_NAME, [])
        if getattr(self.app, 'map_name', None):
            self.app.start_zone_music(self.app.map_name)
        callback = self.on_close
        if callback is not None:
            callback()

    def next_page(self):
        if self.page + 1 < len(self.pages):
            self.app.ui_sound(SOUND_NEXT)
            self.page += 1
            self.shown_key = None
            self.update()
        else:
            self.close()

    def back_page(self):
        if self.page > 0:
            self.app.ui_sound(SOUND_SCROLL)
            self.page -= 1
            self.shown_key = None
            self.update()

    def update(self, dt_ms=0.0):
        if self.briefing is None:
            return
        for actor in self.actors:
            actor.update(dt_ms, self.app.clock)
        key = (self.briefing.name, self.page)
        if key != self.shown_key:
            self.shown_key = key
            self.start_page_voices()
            self.app.interface.set_layer(LAYER_NAME, self.elements())
        self.update_voice()

    def voice_file_path(self, number):
        """
        The mp3 of a phrase in the cache, unpacked from res/speech.res; None when the archive has none.
        """
        entry_name = briefing_module.voice_name(self.briefing.name, number)
        try:
            archive = self.app.game_data.archive(briefing_module.SPEECH_ARCHIVE)
        except OSError:
            return None
        if entry_name not in archive.names:
            return None
        file_path = os.path.join(CACHE_DIR_PATH, *entry_name.split('/'))
        if not os.path.isfile(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(bytes(archive.read(entry_name)))
        return file_path

    def voice_volume(self):
        audio = getattr(self.app, 'audio', None)
        return audio.volume_voice if audio is not None else 1.0

    def start_page_voices(self):
        self.stop_voice()
        page = self.pages[self.page] if 0 <= self.page < len(self.pages) else None
        self.voice_queue = list(page.voices) if page is not None else []

    def update_voice(self):
        if self.voice_sound is not None and self.voice_sound.state == 'play':
            return
        self.voice_sound = None
        if not getattr(self.app.args, 'sound', True):
            self.voice_queue = []
            page = self.pages[self.page] if 0 <= self.page < len(self.pages) else None
            for number in (page.voices if page is not None else ()):
                self.play_utterance(number)
            return
        while self.voice_queue:
            number = self.voice_queue.pop(0)
            file_path = self.voice_file_path(number)
            if file_path is None:
                continue
            sound = SoundLoader.load(file_path)
            if sound is None:
                continue
            sound.volume = self.voice_volume()
            sound.play()
            self.voice_sound = sound
            self.play_utterance(number)
            return

    def stop_voice(self):
        if self.voice_sound is not None:
            self.voice_sound.stop()
        self.voice_sound = None
        self.voice_queue = []


    def camera_name(self):
        """
        The .cam of the map of the briefing: the .mob of the map, else the zone of its .mpr.
        """
        app = self.app
        names = []
        if getattr(app, 'mob_name', None):
            names.append(app.mob_name)
        zone = app.game_data.zone_of_map(app.map_name) if app.map_name else None
        if zone is not None:
            names.append(zone.name)
        for name in names:
            keys = app.game_data.camera(name)
            if keys:
                return keys
        return None

    def find_unit(self, name):
        """
        The unit of the map with the name of "#show" (OBJNAME), the hero for "Hero".
        """
        app = self.app
        if name.lower() == 'hero' and app.unit is not None:
            return app.unit
        if app.script_host is not None:
            lower = name.lower()
            for object_name, obj in app.script_host.objects_by_name.items():
                if object_name.lower() == lower and hasattr(obj, 'figure'):
                    return obj
        return None

    def start_scene(self):
        """
        The 3D scene of the briefing: the camera of camera/<map>.cam (its first key; the map cameras have
        one view) and the figures of "#show" at their places of the map or on the slots of the scene.
        """
        app = self.app
        briefing = self.briefing
        self.old_camera = app.scene.camera
        self.actors = []
        self.actor_views = []
        self.hidden_units = []
        self.actors_by_name = {}
        keys = self.camera_name()
        key = keys[0] if keys else None
        if key is not None:
            eye = [float(v) for v in key.position]
            rotation = np.array(key.rotation, dtype=np.float64)
            forward = quat.rotate(rotation, np.array((0.0, 0.0, 1.0)))
            right = quat.rotate(rotation, np.array((1.0, 0.0, 0.0)))
            anchor = [eye[0] + forward[0] * FALLBACK_DISTANCE, eye[1] + forward[1] * FALLBACK_DISTANCE,
                      0.0]
            app.scene.camera = app.scene_camera_class(key.position, key.rotation)
        else:
            hero = app.unit
            anchor = [hero.x, hero.y, 0.0] if hero is not None else [0.0, 0.0, 0.0]
            forward, right = (0.0, 1.0), (1.0, 0.0)
            camera_class = type(self.old_camera)
            app.scene.camera = camera_class(target=(anchor[0], anchor[1], anchor[2] + 1.0),
                                            yaw=0.0, pitch=12.0, distance=FALLBACK_DISTANCE + 1.0)
        if app.terrain is not None:
            anchor[2] = app.terrain.height_at(anchor[0], anchor[1])
        seen = set()
        cast = list(briefing.cast)
        if not cast:
            cast = [(utterance.speaker, 0) for utterance in briefing.utterances if utterance.speaker]
        for order, (name, slot) in enumerate(cast, 1):
            lower = name.lower()
            if lower in seen:
                continue
            seen.add(lower)
            real = self.find_unit(name)
            if real is None:
                continue
            real_hidden = bool(getattr(real, 'hidden', False))
            self.hidden_units.append((real, real_hidden))
            real.hidden = True
            distance = math.hypot(real.x - anchor[0], real.y - anchor[1]) if hasattr(real, 'x') else SCENE_RADIUS + 1.0
            if distance <= SCENE_RADIUS:
                x, y = real.x, real.y
            else:
                offset = SLOT_OFFSETS.get(slot, SLOT_OFFSETS[0])
                x = anchor[0] + right[0] * offset[0] + forward[0] * offset[1]
                y = anchor[1] + right[1] * offset[0] + forward[1] * offset[1]
            z = app.terrain.height_at(x, y) if app.terrain is not None else getattr(real, 'z', 0.0)
            yaw = math.atan2(anchor[0] - x, -(anchor[1] - y))
            gear = None
            if hasattr(real, 'figure'):
                names = [item.name for item in trade_module.worn_items(real)]
                gear = app.make_equipment(real.figure.model_name, getattr(real, 'prototype_name', None), names)
            actor = app.make_unit(real.figure.model_name, position=(x, y), yaw=yaw, gear=gear,
                                  prototype_name=getattr(real, 'prototype_name', None), removable=True)
            actor.z = z
            actor.hidden = False
            actor.on_sounds = None
            actor.on_blow = None
            actor.on_kill = None
            actor.on_item_wear = None
            view = next((v for v in app.scene.unit_views if v.unit is actor), None)
            if view is not None:
                self.actor_views.append(view)
            self.actors.append(actor)
            self.actors_by_name[lower] = actor

    def end_scene(self):
        """
        The figures of the briefing leave, the units of the map are shown again and the camera of the player
        comes back.
        """
        app = self.app
        for view in self.actor_views:
            app.scene.remove_prop(view)
        self.actor_views = []
        for unit, hidden in self.hidden_units:
            unit.hidden = hidden
        self.hidden_units = []
        self.actors = []
        self.actors_by_name = {}
        if self.old_camera is not None:
            app.scene.camera = self.old_camera
            self.old_camera = None

    def play_utterance(self, number):
        """
        The #ANIMATION of a phrase: the action of its speaker (all the actors when the phrase has no unit).
        """
        utterance = self.utterances_by_voice.get(number)
        if utterance is None or utterance.animation is None:
            return
        if utterance.speaker is not None:
            actors = [self.actors_by_name.get(utterance.speaker.lower())]
        else:
            actors = list(self.actors)
        for actor in actors:
            if actor is not None:
                self.play_animation(actor, utterance.animation)

    def play_animation(self, actor, number):
        """
        The action of the number of "#ANIMATION" of the model of the unit.
        """
        database = actor.figure.database
        if database is None:
            return False
        action = next((a for a in database.actions if a.number == number), None)
        if action is None and 0 <= number < len(database.actions):
            action = database.actions[number]
        if action is None:
            return False
        actor.do_action(action)
        return True

    def elements(self):
        briefing = self.briefing
        page = self.pages[self.page] if 0 <= self.page < len(self.pages) else None
        items = [ui.box_item((0, 0, 800, 600), SCENE_SHADE)]
        items.extend(ui.panel_items(PANEL_RECT, 3, (90, 90, 66), (132, 130, 90),
                                    background=PANEL_COLOR))
        items.append(ui.text_item(briefing.title, TITLE_BOX, ui.FONT_LARGE, TITLE_COLOR, align='center'))
        text = '\n'.join(self.line_text(speaker, line) for speaker, line in (page.lines if page else []))
        items.append(ui.text_item(text, TEXT_BOX, ui.FONT_MEDIUM, TEXT_COLOR))
        counter = '%d/%d' % (self.page + 1, max(len(self.pages), 1))
        items.append(ui.text_item(counter, (120, BUTTONS_TOP, 280, 548), ui.FONT_SMALL, TEXT_COLOR,
                                  align='right'))
        items.append(ui.box_item(BACK_RECT, BUTTON_COLOR))
        items.append(ui.text_item(BUTTON_LABELS['back'], BACK_RECT, ui.FONT_MEDIUM, TEXT_COLOR, align='center'))
        items.append(ui.box_item(NEXT_RECT, BUTTON_COLOR))
        items.append(ui.text_item(BUTTON_LABELS['next'], NEXT_RECT, ui.FONT_MEDIUM, TEXT_COLOR, align='center'))
        return items

    def line_text(self, speaker, line):
        return '%s: %s' % (speaker, line) if speaker else line

    def on_click(self, x, y):
        if self.briefing is None:
            return False
        if _inside(BACK_RECT, x, y):
            self.back_page()
        else:
            self.next_page()
        return True

    def on_key(self, name):
        if self.briefing is None:
            return False
        if name == 'left':
            self.back_page()
        elif name in ('right', 'enter', 'spacebar'):
            self.next_page()
        elif name == 'escape':
            self.close()
        return True

#------------------------------------------------------------------------------


def _inside(rect, x, y):
    left, top, right, bottom = rect
    return left <= x < right and top <= y < bottom
