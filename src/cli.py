import os
import math
import time
import random
import traceback
import collections

from kivy.clock import Clock

import web_sock
import hero

from server import models

_Debug = True

# if the drawn position is further than that (in segments) from the position received from the server
# the unit is moved there right away instead of smooth movement: spawn, teleport, etc.
SNAP_DISTANCE = 1.0

# how long (in ticks) the movement continues along the predicted path when the next update from the server is late
MAX_EXTRAPOLATION = 1.0

# server tick timing is matched to the local clock by the earliest arrived states during that period, seconds
TICK_CLOCK_WINDOW = 1.0

# simulate slow network on localhost to check the smooth movement, milliseconds
_FakeLag = float(os.environ.get('WORLDS_FAKE_LAG_MS', 0)) / 1000.0
_FakeJitter = float(os.environ.get('WORLDS_FAKE_JITTER_MS', 0)) / 1000.0


def angle_lerp(a, b, t):
    """
    Moves angle a towards angle b (degrees) along the shortest way, t=0 gives a, t=1 gives b.
    """
    diff = (b - a) % 360.0
    if diff > 180.0:
        diff -= 360.0
    result = a + diff * t
    return result % 360.0


class Motion(object):
    """
    Smooth movement of one object between the server ticks.
    Server sends position, direction and speed of the object every tick, so the client knows where the object
    is expected to be at any moment after that tick: position + speed * (ticks passed) along the direction.
    Every new state the client starts moving the object from where it is drawn now towards the position expected
    one tick later, so it moves a little every frame instead of jumping once per tick.
    While the server keeps the same speed and direction the drawn object stays exactly on the expected path,
    even if the network delivers the states at uneven moments.
    Positions are continuous map coordinates: x = w + shift_w, y = h + shift_h.
    """

    def __init__(self, x, y, direction, now):
        self.from_x = self.to_x = self.drawn_x = x
        self.from_y = self.to_y = self.drawn_y = y
        self.from_direction = self.to_direction = self.drawn_direction = direction
        self.t_start = now
        self.duration = 1.0

    def retarget(self, x, y, direction, speed, now, state_time, tick_interval):
        """
        New state from the server. state_time is the local time when that server tick is expected to happen.
        """
        # first bring the drawn position to this moment along the current path, the frame was not drawn yet
        self.advance(now)
        if (self.drawn_x - x) ** 2 + (self.drawn_y - y) ** 2 > SNAP_DISTANCE ** 2:
            self.drawn_x = x
            self.drawn_y = y
            self.drawn_direction = direction
        # same formula as the server uses in simulation.Simulation.run_unit(), it moves by "speed" every tick
        ticks_ahead = (now + tick_interval - state_time) / tick_interval
        self.to_x = x + speed * math.cos(math.radians(direction)) * ticks_ahead
        self.to_y = y + speed * math.sin(math.radians(direction)) * ticks_ahead
        self.to_direction = direction
        self.from_x = self.drawn_x
        self.from_y = self.drawn_y
        self.from_direction = self.drawn_direction
        self.t_start = now
        self.duration = tick_interval

    def advance(self, now):
        """
        Calculates position to be drawn at the given moment, returns True if it was changed.
        """
        t = (now - self.t_start) / self.duration if self.duration > 0 else 1.0
        if t < 0.0:
            t = 0.0
        # if next state is late keep moving along the same path for a while, but not too far
        t_position = min(t, 1.0 + MAX_EXTRAPOLATION)
        t_direction = min(t, 1.0)
        x = self.from_x + (self.to_x - self.from_x) * t_position
        y = self.from_y + (self.to_y - self.from_y) * t_position
        direction = angle_lerp(self.from_direction, self.to_direction, t_direction)
        if x == self.drawn_x and y == self.drawn_y and direction == self.drawn_direction:
            return False
        self.drawn_x = x
        self.drawn_y = y
        self.drawn_direction = direction
        return True


class GameClient(object):
    """
    Connects the scene to the game server.
    Server sends objects of the world and their state, client renders them as they are.
    User input is not applied locally, it is only sent to the server as requested controls.
    Between the server ticks movement of the objects is interpolated, see Motion.
    Server sends the list of all models of the world with the welcome message, nothing is rendered until
    the client has all of them: other messages are kept in the queue until the scene is attached.
    """

    def __init__(self, scene=None, renderer=None, url=None, on_welcome=None):
        self.scene = scene
        self.renderer = renderer
        # called with the list of templates from the welcome message, must call attach() when the models are ready
        self.on_welcome = on_welcome
        self.ready = scene is not None
        self.pending_messages = []
        self.url = url or f'ws://127.0.0.1:{models.DEFAULT_PORT}/'
        self.world = None
        self.hero_id = None
        self.tick_interval = 1.0 / models.TICK_RATE
        # (arrival time, arrival time - tick * tick_interval) of recently received states
        self.tick_clock_samples = collections.deque()
        # object id -> (model received from the server, hero.Hero which renders it, Motion)
        self.objects = {}
        self.controls_sent = None

    def start(self):
        web_sock.start(callbacks={
            'on_open': self.on_open,
            'on_close': self.on_close,
            'on_message': self.on_message_received if _FakeLag or _FakeJitter else self.on_message,
        }, url=self.url)

    def stop(self):
        if web_sock.is_started():
            web_sock.stop()

    def attach(self, scene, renderer):
        """
        All required models are ready and the scene was created, process messages received meanwhile.
        """
        self.scene = scene
        self.renderer = renderer
        self.ready = True
        pending = self.pending_messages
        self.pending_messages = []
        if _Debug:
            print(f'game client is ready, {len(pending)} messages were waiting')
        for json_data in pending:
            self.on_message(json_data)

    #--- CONNECTION ---

    def on_open(self, ws_inst):
        if _Debug:
            print(f'connected to game server {self.url}')
        self.controls_sent = None
        web_sock.send(models.message(models.MSG_JOIN))

    def on_close(self, ws_inst):
        if _Debug:
            print(f'disconnected from game server {self.url}')
        # server will create new hero after reconnect, all that client knows about the world is not valid anymore
        self.pending_messages = []
        for object_id in list(self.objects.keys()):
            self.despawn(object_id)
        self.hero_id = None
        if self.scene:
            self.scene.hero = None
        if self.renderer:
            self.renderer.camera_unit_lock = None
        # server may restart and count ticks from zero again
        self.tick_clock_samples.clear()

    def on_message_received(self, json_data):
        # debug only: delays incoming messages to simulate slow network, messages still keep their order
        now = time.perf_counter()
        delay = _FakeLag + random.uniform(0.0, _FakeJitter)
        due = max(now + delay, getattr(self, '_fake_last_due', 0.0))
        self._fake_last_due = due
        Clock.schedule_once(lambda dt: self.on_message(json_data), due - now)
        return True

    def on_message(self, json_data):
        msg_type = json_data.get('t')
        payload = json_data.get('p') or {}
        if not self.ready and msg_type != models.MSG_WELCOME:
            self.pending_messages.append(json_data)
            return True
        try:
            if msg_type == models.MSG_STATE:
                now = time.perf_counter()
                state_time = self.tick_time(payload.get('tick'), now)
                for d in payload['objects']:
                    self.update(d, now, state_time)
            elif msg_type == models.MSG_SPAWN:
                for d in payload['objects']:
                    self.spawn(models.object_from_dict(d))
            elif msg_type == models.MSG_DESPAWN:
                for object_id in payload['ids']:
                    self.despawn(object_id)
            elif msg_type == models.MSG_WELCOME:
                self.world = models.World.from_dict(payload['world'])
                self.hero_id = payload['hero_id']
                if self.world.tick_rate:
                    self.tick_interval = 1.0 / self.world.tick_rate
                if _Debug:
                    print(f'joined world {self.world}, my hero is {self.hero_id}')
                if self.on_welcome:
                    # wait until models are downloaded, see attach()
                    self.ready = False
                    self.on_welcome(payload.get('templates') or [])
            elif _Debug:
                print(f'unknown message from game server: {json_data}')
        except:
            traceback.print_exc()
        return True

    #--- OBJECTS ---

    def spawn(self, o):
        if o.id in self.objects:
            self.despawn(o.id)
        t = o.template
        view = hero.Hero(
            scene=self.scene,
            model_name=t.model_name,
            skin=t.skin,
            hair=t.hair,
            wears=t.wears,
            weapon=t.weapon,
            texture=t.texture,
        )
        view.create_unit(o.w, o.h, elevation_correction=t.elevation_correction, coefs=list(t.coefs), shift_w=o.shift_w, shift_h=o.shift_h, direction=o.direction)
        motion = Motion(o.w + o.shift_w, o.h + o.shift_h, o.direction, time.perf_counter())
        self.objects[o.id] = (o, view, motion)
        self.apply(o, view)
        if o.id == self.hero_id:
            self.scene.hero = view
            self.renderer.camera_unit_lock = view.unit_name
        if _Debug:
            print(f'spawned {o.kind} {o.id} as unit {view.unit_name}')

    def tick_time(self, tick, now):
        """
        Local time when the given server tick happened. Network delivers states at uneven moments, but the server
        ticks are even: the states which arrived the fastest show how server ticks are aligned to the local clock.
        """
        if tick is None:
            return now
        samples = self.tick_clock_samples
        samples.append((now, now - tick * self.tick_interval))
        while samples and samples[0][0] < now - TICK_CLOCK_WINDOW:
            samples.popleft()
        offset = min(s[1] for s in samples)
        return offset + tick * self.tick_interval

    def update(self, d, now=None, state_time=None):
        if d['id'] not in self.objects:
            return
        if now is None:
            now = time.perf_counter()
        if state_time is None:
            state_time = now
        o, view, motion = self.objects[d['id']]
        o.update(d)
        # the position is not applied right away, it is reached smoothly by interpolate() during the next frames
        motion.retarget(
            x=o.w + o.shift_w,
            y=o.h + o.shift_h,
            direction=o.direction,
            speed=getattr(o, 'speed', 0.0) or 0.0,
            now=now,
            state_time=state_time,
            tick_interval=self.tick_interval,
        )
        if isinstance(o, models.Unit):
            view.set_animation(o.animation)

    def apply(self, o, view):
        u = view.get_unit()
        if not u:
            return
        u.set_position(self.scene, o.w, o.h, o.shift_w, o.shift_h, o.direction)
        if isinstance(o, models.Unit):
            view.set_animation(o.animation)

    def interpolate(self, now):
        """
        Called every frame before drawing: moves all objects to their positions expected at this moment.
        """
        for o, view, motion in self.objects.values():
            if not motion.advance(now):
                continue
            u = view.get_unit()
            if not u:
                continue
            w = math.floor(motion.drawn_x)
            h = math.floor(motion.drawn_y)
            u.set_position(self.scene, w, h, motion.drawn_x - w, motion.drawn_y - h, motion.drawn_direction)

    def despawn(self, object_id):
        o, view, _ = self.objects.pop(object_id, (None, None, None))
        if not view:
            return
        if view.unit_name in self.scene.units:
            self.scene.remove_unit_from_stage(container=self.scene.container_animated_objects, unit_name=view.unit_name)
        if self.renderer.camera_unit_lock == view.unit_name:
            self.renderer.camera_unit_lock = None
        if self.scene.hero is view:
            self.scene.hero = None

    #--- INPUT ---

    def send_controls(self, move=0, turn=0, target_direction=None):
        if target_direction is not None:
            target_direction = round(target_direction, 1)
        controls = models.Controls(move=move, turn=turn, target_direction=target_direction)
        if controls.to_dict() == self.controls_sent:
            return
        if web_sock.send(models.message(models.MSG_INPUT, controls.to_dict())):
            self.controls_sent = controls.to_dict()
