"""
Animations of the figures, the same way the CAnimation* classes of game.exe do it.

sample_track()      CAnimationTrack::SetTime (0x50acd0): time in frames, modf() gives the key and the fraction,
                    translations are interpolated linearly, rotations by slerp, after the last key the last one stays.
Skeleton.pose()     the bone update (0x50c230): positions and rotations of the parts from the root down to the leaves,
                    without matrices: child = parent position + parent rotation applied to the offset of the child.
                    The remake does it for all the parts at once (pose_arrays): the keys of a clip are stacked into
                    tables per skeleton, the parts of one depth of the tree are turned together.
AnimationPlayer     the player of CAnimationSystem2 (0x50d600): the current clip and a queue of the next ones,
                    a clip loops only when its stage is CYCLE, any other clip stops on its last frame.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math
import collections

import numpy as np

#------------------------------------------------------------------------------

from eilib import fig
from engine import quat

#------------------------------------------------------------------------------

ZERO = np.zeros(3)
NO_TRACKS = {}

#------------------------------------------------------------------------------


class Clip(object):
    """
    One animation of a model: the action of the .adb and the tracks of the parts.
    """

    def __init__(self, action, tracks):
        self.action = action
        self.tracks = tracks
        self.name = action.name if action else ''

    def __repr__(self):
        return 'Clip(%s, %d frames, %d tracks)' % (self.name, self.length, len(self.tracks))

    @property
    def length(self):
        if self.action and self.action.length:
            return self.action.length
        keys = max((len(t.rotations) for t in self.tracks.values()), default=1)
        return max(keys - 1, 1)

    @property
    def is_cycle(self):
        return bool(self.action and self.action.is_cycle)


class Skeleton(object):
    """
    Tree of the visible parts of one figure with the offsets of the parts for the body proportions.
    """

    def __init__(self, parts_tree, bones, part_names, coefs=(0.0, 0.0, 0.0), scale=1.0):
        self.parts_tree = parts_tree
        self.coefs = coefs
        self.scale = scale
        wanted = set(part_names)
        # a part is posed when it is shown or when a shown part hangs on it
        needed = set()
        for name in part_names:
            while name and name not in needed:
                needed.add(name)
                name = parts_tree.parents.get(name)
        self.order = [name for name in parts_tree.order if name in needed]
        self.visible = [name for name in self.order if name in wanted]
        self.offsets = {}
        for name in self.order:
            bone = bones.get(name)
            if bone is None:
                self.offsets[name] = ZERO
            elif len(bone) == 8:
                self.offsets[name] = fig.trilinear(bone.astype(np.float64), coefs)
            else:
                self.offsets[name] = bone[0].astype(np.float64)
        self.index = {name: i for i, name in enumerate(self.order)}
        self.offset_array = np.array([self.offsets[name] for name in self.order], dtype=np.float64).reshape(-1, 3)
        # the parts by their depth in the tree: (the parts, their parents); the roots have no parent here
        depths = []
        parents = []
        for name in self.order:
            parent_name = parts_tree.parents.get(name)
            parent = self.index.get(parent_name) if parent_name is not None else None
            parents.append(parent)
            depths.append(0 if parent is None else depths[parent] + 1)
        self.roots = np.array([i for i, p in enumerate(parents) if p is None], dtype=np.int64)
        self.levels = []
        for depth in range(1, max(depths, default=0) + 1):
            members = [i for i, d in enumerate(depths) if d == depth]
            self.levels.append((np.array(members, dtype=np.int64),
                                np.array([parents[i] for i in members], dtype=np.int64)))
        # id of the tracks of a clip -> (the tracks, the tables of their keys for these parts)
        self.tables = {}

    def pose(self, tracks, time):
        """
        part name -> (position, rotation) in the space of the figure: CAnimationSystem2 Update (50d450) sets
        the time of every track, then the bone update (50c230) goes from the root down.
        """
        positions, rotations = self.pose_arrays(tracks, time)
        return {name: (positions[i], rotations[i]) for i, name in enumerate(self.order)}

    def clip_tables(self, tracks):
        """
        The keys of the tracks for the parts of the skeleton: rotations (keys, n, 4), translations (keys, n, 3) and
        the parts which have translations. A track with fewer keys repeats its last one, as sample_track stays on it.
        """
        if not tracks:
            # a new empty dictionary on every call: one entry for all of them
            tracks = NO_TRACKS
        cached = self.tables.get(id(tracks))
        if cached is not None and cached[0] is tracks:
            return cached[1]
        count = len(self.order)
        keys = max([len(t.rotations) for t in tracks.values()] + [len(t.translations) for t in tracks.values()] + [1])
        rotations = np.tile(quat.IDENTITY, (keys, count, 1))
        translations = np.tile(self.offset_array, (keys, 1, 1))
        moved = np.zeros(count, dtype=bool)
        for i, name in enumerate(self.order):
            track = tracks.get(name)
            if track is None:
                continue
            n = len(track.rotations)
            if n:
                rotations[:n, i] = track.rotations
                rotations[n:, i] = track.rotations[n - 1]
            n = len(track.translations)
            if n:
                translations[:n, i] = track.translations
                translations[n:, i] = track.translations[n - 1]
                moved[i] = True
        tables = (rotations, translations, moved)
        self.tables[id(tracks)] = (tracks, tables)
        return tables

    def pose_arrays(self, tracks, time):
        """
        The positions (n, 3) and rotations (n, 4) of the parts of self.order, the same as pose_slow.
        """
        rotation_keys, translation_keys, moved = self.clip_tables(tracks)
        fraction, whole = math.modf(max(time, 0.0))
        i = int(whole)
        last = len(rotation_keys) - 1
        if i < last:
            local_rotations = quat.slerp_many(rotation_keys[i], rotation_keys[i + 1], fraction)
            a = translation_keys[i]
            local_positions = a + (translation_keys[i + 1] - a) * fraction
        else:
            local_rotations = rotation_keys[last]
            local_positions = translation_keys[last]
        positions = np.empty((len(self.order), 3))
        rotations = np.empty((len(self.order), 4))
        roots = self.roots
        # the shifts of the root were recorded for one body, the size of the unit scales them
        positions[roots] = np.where(moved[roots, None], local_positions[roots], self.offset_array[roots]) * self.scale
        rotations[roots] = local_rotations[roots]
        for members, parents in self.levels:
            parent_rotations = rotations[parents]
            positions[members] = positions[parents] + quat.rotate_many(parent_rotations, self.offset_array[members])
            rotations[members] = quat.multiply_many(parent_rotations, local_rotations[members])
        return positions, rotations

    def pose_slow(self, tracks, time):
        """
        The bone update part by part (kept to check pose_arrays).
        """
        result = {}
        for name in self.order:
            parent_name = self.parts_tree.parents.get(name)
            track = tracks.get(name)
            if track is not None:
                position, rotation = sample_track(track, time)
            else:
                position, rotation = None, quat.IDENTITY
            if parent_name is None or parent_name not in result:
                # the shifts of the root were recorded for one body, the size of the unit scales them
                root_position = position if position is not None else self.offsets[name]
                result[name] = (root_position * self.scale, rotation)
                continue
            parent_position, parent_rotation = result[parent_name]
            world_position = parent_position + quat.rotate(parent_rotation, self.offsets[name])
            world_rotation = quat.multiply(parent_rotation, rotation)
            result[name] = (world_position, world_rotation)
        return result


class AnimationPlayer(object):

    def __init__(self, on_clip_end=None):
        self.clip = None
        self.time = 0.0
        self.queue = collections.deque()
        self.on_clip_end = on_clip_end
        self.finished = False

    def play(self, clip, start_time=0.0):
        """
        Starts the clip right away and forgets the queue, CAnimationSystem2 Play (0x50ca50).
        """
        self.clip = clip
        self.time = start_time
        self.queue.clear()
        self.finished = False

    def enqueue(self, clip):
        """
        The clip starts when the current one ends, CAnimationSystem2 Enqueue (0x50cba0).
        """
        if self.clip is None:
            self.play(clip)
        else:
            self.queue.append(clip)

    def add_time(self, frames):
        """
        CAnimationSystem2 AddTime (50d5e0): the time of the player plus the frames, then 50d600.
        """
        if self.clip is None:
            return
        self.time += frames
        while self.clip is not None and self.time >= self.clip.length:
            ended = self.clip
            if self.queue:
                self.time -= ended.length
                self.clip = self.queue.popleft()
                self.finished = False
            elif ended.is_cycle:
                self.time -= ended.length
            else:
                self.time = float(ended.length)
                if not self.finished:
                    self.finished = True
                    if self.on_clip_end:
                        self.on_clip_end(ended)
                break
            if self.on_clip_end:
                self.on_clip_end(ended)

    @property
    def tracks(self):
        return self.clip.tracks if self.clip else {}

#------------------------------------------------------------------------------


def sample_track(track, time):
    """
    Position (or None when the track has no translations) and rotation of the track at the time in frames.
    """
    fraction, whole = math.modf(max(time, 0.0))
    i = int(whole)
    position = None
    count = len(track.translations)
    if count:
        if i + 1 < count:
            a = track.translations[i]
            position = a + (track.translations[i + 1] - a) * fraction
        else:
            position = track.translations[count - 1]
    count = len(track.rotations)
    if count:
        if i + 1 < count:
            rotation = quat.slerp(track.rotations[i], track.rotations[i + 1], fraction)
        else:
            rotation = track.rotations[count - 1]
    else:
        rotation = quat.IDENTITY
    return position, rotation


def sample_morph(track, time):
    """
    Vertices of the morphing at the time in frames, None when the track has no morphing.
    """
    if track is None or track.morph_frames is None:
        return None
    frames = track.morph_frames
    fraction, whole = math.modf(max(time, 0.0))
    i = int(whole)
    if i + 1 < len(frames):
        return frames[i] + (frames[i + 1] - frames[i]) * fraction
    return frames[len(frames) - 1]
