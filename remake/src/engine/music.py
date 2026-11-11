"""
The music of the game (CMusicSystem, vtable 744098): the tracks of res/music.reg and the choice of them.

music.reg (read by vf02, 6b46d0) has a section of every allod (gipat, ingos, suslanger, final) with the lists
CalmOpen, CalmDungeon and Combat and the tracks Briefing and Constructor, and the section common with
MainMenu and Credits. A track is data/stream/<name>.mp3. For every allod two records of 0x38 bytes are made,
one of the open zones (CalmOpen) and one of the caves (CalmDungeon), both with the same Combat.

The states +0x38: 1 a zone (SetZoneMode 6b4df0: the allod of map.txt, "final" for gz20g, and the sky cave
of the zone, 567ab0), 2 the briefing (6b4f50), 3 the constructor (6b5040), 4 the credits, 5 the main menu.
In a zone every frame (vf06, 6b5310):
    - a track of PlayMusic( player, name ) (6b5230) plays to its end; then the music waits 120..180 seconds
      (120000 + rand % 60001 ms of timeGetTime);
    - when the fight starts (the flag 7afc8c is 2, +0x40 keeps the last one) a random Combat track starts
      unless the last one of the fight plays still;
    - when the fight ends the track fades away (6bb200(2), 2 seconds) and the music waits 120..180 seconds;
    - when nothing plays and the wait is over a random calm track starts; the wait of a new zone is over at once.
The flag of the fight comes to the client from the server (668e59 reads it into +0x6e4 of the world); the remake
takes the danger of the player (the state 2 of +0x5120, 66e3f0: a unit of the party is chased).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import random

#------------------------------------------------------------------------------

MUSIC_REG = 'res/music.reg'

COMMON = 'common'
# gz20g plays the tracks of this section (567ab0)
FINAL = 'final'
FINAL_ZONE = 'gz20g'

ZONE = 1
BRIEFING = 2
CONSTRUCTOR = 3
CREDITS = 4
MAIN_MENU = 5

# 6b5310: the wait between the calm tracks, ms
WAIT_MS = 120000
WAIT_SPREAD_MS = 60001

# 6bb200: the fading of a track, ms
FADE_MS = 2000

PLAY = 'play'
FADE = 'fade'

#------------------------------------------------------------------------------


class Record(object):
    """
    A record of 0x38 bytes: the allod, whether it is of the caves, the tracks of the briefing and the constructor,
    the calm ones and the ones of the fight.
    """

    def __init__(self, allod, dungeon, briefing='', constructor='', calm=(), combat=()):
        self.allod = allod
        self.dungeon = dungeon
        self.briefing = briefing
        self.constructor = constructor
        self.calm = list(calm)
        self.combat = list(combat)

    def __repr__(self):
        return 'Record(%s%s, %d calm, %d combat)' % (self.allod, ' dungeon' if self.dungeon else '', len(self.calm),
                                                   len(self.combat))


class MusicTable(object):

    def __init__(self, records=(), main_menu='', credits=''):
        self.records = list(records)
        self.main_menu = main_menu
        self.credits = credits

    def record(self, allod, dungeon):
        allod = (allod or '').lower()
        for r in self.records:
            if r.allod == allod and r.dungeon == dungeon:
                return r
        return None

    @classmethod
    def from_registry(cls, sections):
        """
        The table of the dict of eilib.reg (section -> {key: value or tuple}).
        """
        records = []
        main_menu = credits = ''
        for name, values in (sections or {}).items():
            if name.lower() == COMMON:
                main_menu = _track(values.get('MainMenu'))
                credits = _track(values.get('Credits'))
                continue
            for dungeon, key in ((False, 'CalmOpen'), (True, 'CalmDungeon')):
                records.append(Record(name.lower(), dungeon, _track(values.get('Briefing')),
                                      _track(values.get('Constructor')), _tracks(values.get(key)),
                                      _tracks(values.get('Combat'))))
        return cls(records, main_menu, credits)

#------------------------------------------------------------------------------


class MusicSystem(object):
    """
    The choice of the tracks; update() gives the requests to the player: (PLAY, name) and (FADE, None).
    """

    def __init__(self, table, rng=None):
        self.table = table
        self.rng = rng or random.Random()
        self.state = 0
        self.record = None
        # +0x40, +0x48: the fight of the last frame and its track
        self.fighting = False
        self.combat_track = ''
        # +0x4c, +0x50: the track of the scripts
        self.script_track = ''
        # +0x54: the time when a calm track may start, 0 when it is not set
        self.wait_until = 0.0
        self.requests = []

    def __repr__(self):
        return 'MusicSystem(%d %s)' % (self.state, self.record)

    def random_index(self, count):
        return self.rng.getrandbits(32) % count

    def wait(self, now_ms):
        return now_ms + WAIT_MS + self.random_index(WAIT_SPREAD_MS)

    def play(self, name):
        self.requests.append((PLAY, name))

    def set_zone(self, allod, dungeon):
        """
        SetZoneMode (6b4df0); False when music.reg has no such allod.
        """
        record = self.table.record(allod, dungeon)
        if record is None:
            return False
        self.state = ZONE
        self.record = record
        self.fighting = False
        self.combat_track = ''
        self.script_track = ''
        self.wait_until = 1.0
        self.requests.append((FADE, None))
        return True

    def set_briefing(self, allod):
        """
        SetBriefing (6b4f50), the state 2.
        """
        self.state = BRIEFING
        record = self.table.record(allod, False)
        if record is not None and record.briefing:
            self.play(record.briefing)

    def set_constructor(self, allod):
        """
        SetConstructor (6b5040), the state 3.
        """
        self.state = CONSTRUCTOR
        record = self.table.record(allod, False)
        if record is not None and record.constructor:
            self.play(record.constructor)

    def set_main_menu(self):
        """
        SetMainMenu (6b5130), the state 5.
        """
        self.state = MAIN_MENU
        if self.table.main_menu:
            self.play(self.table.main_menu)

    def set_credits(self):
        """
        SetCredits (6b51b0), the state 4.
        """
        self.state = CREDITS
        if self.table.credits:
            self.play(self.table.credits)

    def play_script(self, name):
        """
        PlayMusic (6b5230): only in a zone.
        """
        if self.state != ZONE or not name:
            return
        self.script_track = name.lower()
        self.play(self.script_track)

    def update(self, now_ms, fighting, playing):
        """
        vf06 (6b5310) once a frame: now_ms is the real time, playing the name of the track which plays ('' for none).
        Gives the requests made since the last call.
        """
        if self.state == ZONE and self.record is not None:
            self.update_zone(now_ms, fighting, playing or '')
        requests, self.requests = self.requests, []
        return requests

    def update_zone(self, now_ms, fighting, playing):
        if self.script_track:
            if playing == self.script_track:
                return
            self.script_track = ''
            self.fighting = False
            self.wait_until = self.wait(now_ms)
        if not self.fighting:
            if fighting:
                if (playing != self.combat_track or not self.combat_track) and self.record.combat:
                    self.combat_track = self.record.combat[self.random_index(len(self.record.combat))]
                if self.combat_track and playing != self.combat_track:
                    self.play(self.combat_track)
            elif not playing:
                if not self.wait_until:
                    self.wait_until = self.wait(now_ms)
                if self.wait_until <= now_ms:
                    self.wait_until = 0.0
                    if self.record.calm:
                        self.play(self.record.calm[self.random_index(len(self.record.calm))])
        elif not fighting:
            self.requests.append((FADE, None))
            self.wait_until = self.wait(now_ms)
        self.fighting = fighting

#------------------------------------------------------------------------------


def _track(value):
    if isinstance(value, (tuple, list)):
        value = value[0] if value else ''
    return (value or '').lower()


def _tracks(value):
    if value is None:
        return []
    if isinstance(value, (tuple, list)):
        return [v.lower() for v in value if v]
    return [value.lower()] if value else []


def zone_allod(zone_name, allod):
    """
    567ab0: gz20g plays the music of "final", the others that of their allod.
    """
    if (zone_name or '').lower() == FINAL_ZONE:
        return FINAL
    return allod
