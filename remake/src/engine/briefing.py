"""
The briefings of the scripts: the text "briefing <name>" of texts.res (the event "briefing <name>" of
a .mob, game.exe 60xxxx), its speech and the pages of its window.

A briefing line is a title, then directives and the speech of the units:

    #show <unit> [<number>]   a unit of the scene
    #ANIMATION <n>            the action of the units
    #CAMERA <n>               the camera path
    #phrase <unit> <n>        the next lines are said by the unit
    #DESC <text>              the remark of the director, not shown
    #NOLIPS                   the lips of the units do not move

game.exe plays the briefing as a 3D scene (the figures on camera/*.cam paths) and the speech of every
#phrase: the voice "briefing/<name>/<number>.mp3" of res/speech.res, the number is the order of the phrase
(1, 2, ...). The remake shows the text of the scene in pages; a page holds whole utterances, its voices
play one after another. The scripts set #OnBriefingComplete after the briefing; the client calls its
callback when the window closes.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import re

#------------------------------------------------------------------------------

TEXT_PREFIX = 'briefing '
VOICE_FORMAT = 'briefing/%s/%d.mp3'
SPEECH_ARCHIVE = 'res/speech.res'

DIRECTIVE = re.compile(r'^#(\w+)\s*(.*)$')

# the number of the lines of a page of the window of the remake
PAGE_LINES = 18

#------------------------------------------------------------------------------


class Utterance(object):
    """
    One phrase of the briefing: the speaker, its lines and the number of its voice file (1, 2, ...).
    """

    def __init__(self, speaker, lines=(), voice=None, animation=None, camera=None, lips=True):
        self.speaker = speaker
        self.lines = list(lines)
        self.voice = voice
        # "#ANIMATION <n>" and "#CAMERA <n>" before the phrase, "#NOLIPS"
        self.animation = animation
        self.camera = camera
        self.lips = lips

    def __repr__(self):
        return 'Utterance(%s, %d lines, voice %s)' % (self.speaker, len(self.lines), self.voice)


class SpeechPage(object):
    """
    A page of the window: the lines and the voices of the utterances which start on it.
    """

    def __init__(self, lines=(), voices=()):
        self.lines = list(lines)
        self.voices = list(voices)

    def __repr__(self):
        return 'SpeechPage(%d lines, voices %s)' % (len(self.lines), self.voices)


class Briefing(object):
    """
    A briefing: its script name, the title, the speech lines as (speaker, text), the utterances and the
    pages which keep the phrases whole.
    """

    def __init__(self, name, title, lines=(), utterances=(), cast=(), hidden=()):
        self.name = name
        self.title = title
        self.lines = list(lines)
        self.utterances = list(utterances)
        # "#show <unit> [<slot>]" in the order of the text: (the name, the slot)
        self.cast = list(cast)
        # "#hide <unit>": the actor leaves the scene after the phrase, the remake keeps it shown
        self.hidden = list(hidden)

    def __repr__(self):
        return 'Briefing(%s, %d lines)' % (self.name, len(self.lines))

    def pages(self):
        """
        The lines split into pages of PAGE_LINES (the narration pages without voices).
        """
        if not self.lines:
            return [[]]
        return [self.lines[i:i + PAGE_LINES] for i in range(0, len(self.lines), PAGE_LINES)]

    def speech_pages(self):
        """
        The pages of whole utterances: a long utterance gets a page of its own, PAGE_LINES are the usual limit.
        """
        if not self.utterances:
            return [SpeechPage()]
        pages = []
        lines = []
        voices = []
        for utterance in self.utterances:
            if lines and len(lines) + len(utterance.lines) > PAGE_LINES:
                pages.append(SpeechPage(lines, voices))
                lines = []
                voices = []
            for line in utterance.lines:
                lines.append((utterance.speaker, line))
            if utterance.voice is not None:
                voices.append(utterance.voice)
        if lines:
            pages.append(SpeechPage(lines, voices))
        return pages or [SpeechPage()]

#------------------------------------------------------------------------------


def voice_name(name, number):
    """
    The entry of res/speech.res with the voice of a phrase, game.exe "briefing\\%s\\%d.mp3".
    """
    return VOICE_FORMAT % (name, number)


def parse(name, text):
    """
    Briefing of the text "briefing <name>" of texts.res, None when the text is empty.
    """
    raw = [line.strip() for line in (text or '').replace('\r', '').split('\n')]
    raw = [line for line in raw if line]
    if not raw:
        return None
    title = raw[0]
    lines = []
    utterances = []
    speaker = None
    voice = None
    voice_count = 0
    current = []
    cast = []
    hidden = []
    animation = None
    camera = None
    lips = True
    utterance_animation = None
    utterance_camera = None
    utterance_lips = True

    def _flush():
        if current:
            utterances.append(Utterance(speaker, current, voice, utterance_animation, utterance_camera,
                                        utterance_lips))
            del current[:]

    for line in raw[1:]:
        match = DIRECTIVE.match(line)
        if match is not None:
            _flush()
            directive, argument = match.group(1).upper(), match.group(2).strip()
            if directive == 'SHOW':
                words = argument.split()
                if words:
                    slot = int(words[1]) if len(words) > 1 and words[1].isdigit() else 0
                    cast.append((words[0], slot))
            elif directive == 'HIDE':
                words = argument.split()
                if words:
                    hidden.append(words[0])
            elif directive == 'ANIMATION':
                words = argument.split()
                animation = int(words[0]) if words and words[0].isdigit() else None
            elif directive == 'CAMERA':
                words = argument.split()
                camera = int(words[0]) if words and words[0].isdigit() else None
            elif directive == 'NOLIPS':
                lips = False
            elif directive == 'PHRASE':
                words = argument.split()
                speaker = words[0] if words else None
                voice_count += 1
                voice = voice_count
                utterance_animation = animation
                utterance_camera = camera
                utterance_lips = lips
                lips = True
            elif directive == 'DESC':
                speaker = None
                voice = None
            else:
                speaker = None
                voice = None
            continue
        if speaker is not None:
            lines.append((speaker if not current else None, line))
            current.append(line)
        else:
            lines.append((None, line))
            current.append(line)
            utterances.append(Utterance(None, current, None))
            del current[:]
    _flush()
    return Briefing(name, title, lines, utterances, cast, hidden)


def load(name, text_of):
    """
    Briefing of the name from a text database, text_of(name) -> text or None.
    """
    return parse(name, text_of(TEXT_PREFIX + name))
