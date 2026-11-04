"""
Game client: keeps the connection to the game server, receives the biomes and the state of the world,
predicts the movement of the objects between the server ticks and passes the updates to the scene.
"""

#------------------------------------------------------------------------------

_Debug = True

#------------------------------------------------------------------------------

import os
import math
import time
import base64
import random
import traceback
import collections

from kivy.clock import Clock

#------------------------------------------------------------------------------

from lib import models
from lib import physics
from client import ws
from client import hero
from client import loader

#------------------------------------------------------------------------------

# if the drawn position is further than that (in segments) from the position received from the server
# the unit is moved there right away instead of smooth movement: spawn, teleport, etc.
SNAP_DISTANCE = 1.0

# when the server state does not match the predicted one, the difference is removed during that time, seconds
CORRECTION_TIME = 0.1

# how many ticks after the last received state the movement is calculated, the unit stops there if the server is silent
MAX_PREDICTION = 10

# server tick timing is matched to the local clock by the earliest arrived states during that period, seconds
TICK_CLOCK_WINDOW = 5.0
# how fast the matched timing may move later (seconds per received state), the drawn movement never goes back in time
TICK_CLOCK_DRIFT = 0.001

# own controls are requested for the tick that many ticks after the current one, so they reach the server in time,
# it grows when the server reports the controls came late and slowly shrinks while they come early
INPUT_LEAD = 3.0
INPUT_LEAD_MIN = 1.0
INPUT_LEAD_MAX = 20.0
# spare ticks kept for uneven network
INPUT_LEAD_MARGIN = 1.0

#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------


class Prediction(object):
    """
    Position of one object at any moment, calculated with the same rules as the server uses, see lib.physics.
    Movement starts from the last state received from the server (the base) and the server ticks after it
    are repeated on the client with the controls of the unit: controls of the own hero are known in advance,
    because the client asks the server to apply them at the given tick, other units are expected to keep
    the controls they had in the last state. Between the ticks the position is interpolated.
    So the drawn unit is always where the server has it, including slowing down after the move button was released.
    When the server did something the client could not know (the unit was blocked, another player pressed a button)
    the new state does not match: the difference is removed smoothly in CORRECTION_TIME, or right away if it is large.
    Positions are continuous map coordinates: x = w + shift_w, y = h + shift_h.
    """

    def __init__(self, o, tick, now):
        self.o = o
        # (tick, models.Controls) sent by this client and not applied by the server yet, only for the own hero
        self.scheduled = []
        self.error_x = 0.0
        self.error_y = 0.0
        self.error_direction = 0.0
        self.error_time = now
        self.drawn = None
        self.set_base(tick)

    def set_base(self, tick):
        o = self.o
        self.base_tick = tick
        # states of the ticks starting from the base tick: (w, h, shift_w, shift_h, direction, speed)
        self.states = [(o.w, o.h, o.shift_w, o.shift_h, o.direction, getattr(o, 'speed', 0.0) or 0.0)]

    def schedule(self, controls):
        self.scheduled.append((controls.tick, controls))
        # ticks after that one must be calculated again with the new controls
        keep = max(1, controls.tick - self.base_tick)
        del self.states[keep:]

    def acknowledge(self, input_seq):
        self.scheduled = [(t, c) for t, c in self.scheduled if c.seq > input_seq]

    def controls_at(self, tick):
        o = self.o
        move, turn, target_direction = o.move, o.turn, o.target_direction
        for t, c in self.scheduled:
            if t <= tick:
                move, turn, target_direction = c.move, c.turn, c.target_direction
        return move, turn, target_direction

    def state_at(self, tick):
        o = self.o
        i = min(tick - self.base_tick, MAX_PREDICTION)
        if i <= 0 or not isinstance(o, models.Unit):
            return self.states[0]
        while len(self.states) <= i:
            w, h, shift_w, shift_h, direction, speed = self.states[-1]
            move, turn, target_direction = self.controls_at(self.base_tick + len(self.states))
            self.states.append(physics.step(
                w, h, shift_w, shift_h, direction, speed, move, turn, target_direction,
                o.max_speed, o.acceleration_value, o.deceleration_value, o.turn_speed,
            ))
        return self.states[i]

    def predicted(self, render_tick):
        """
        (x, y, direction) where the server has the object at that moment, render_tick is a fractional tick number.
        """
        if render_tick is None or render_tick <= self.base_tick:
            a = self.states[0]
            return a[0] + a[2], a[1] + a[3], a[4]
        k = int(math.floor(render_tick))
        f = render_tick - k
        a = self.state_at(k)
        b = self.state_at(k + 1)
        x = (a[0] + a[2]) + ((b[0] + b[2]) - (a[0] + a[2])) * f
        y = (a[1] + a[3]) + ((b[1] + b[3]) - (a[1] + a[3])) * f
        return x, y, angle_lerp(a[4], b[4], f)

    def rebase(self, tick, now, render_tick):
        """
        New state received from the server, the model was already updated.
        """
        if self.drawn is None:
            self.set_base(tick)
            return
        # where the object is drawn at this moment by the old prediction, the last frame was a bit earlier
        self.advance(now, render_tick)
        before = self.drawn
        self.set_base(tick)
        x, y, direction = self.predicted(render_tick)
        error_x = before[0] - x
        error_y = before[1] - y
        error_direction = ((before[2] - direction + 180.0) % 360.0) - 180.0
        if error_x ** 2 + error_y ** 2 > SNAP_DISTANCE ** 2:
            # teleport or the unit was moved a lot, jump right there
            error_x = error_y = error_direction = 0.0
        self.error_x, self.error_y, self.error_direction = error_x, error_y, error_direction
        self.error_time = now

    def advance(self, now, render_tick):
        """
        Calculates position to be drawn at the given moment, returns True if it was changed.
        """
        x, y, direction = self.predicted(render_tick)
        if self.error_x or self.error_y or self.error_direction:
            k = math.exp(-max(0.0, now - self.error_time) / CORRECTION_TIME)
            if k < 0.001:
                self.error_x = self.error_y = self.error_direction = 0.0
            else:
                x += self.error_x * k
                y += self.error_y * k
                direction = (direction + self.error_direction * k) % 360.0
        drawn = (x, y, direction)
        if drawn == self.drawn:
            return False
        self.drawn = drawn
        return True

#------------------------------------------------------------------------------


class GameClient(object):
    """
    Connects the scene to the game server.
    Server sends objects of the world and their state, client renders them as they are.
    User input is not applied locally, it is only sent to the server as requested controls, together with
    the tick they must be applied at. Movement of the objects is calculated with the same rules as on the server,
    see Prediction.
    After connecting the server sends its name and the list of its biomes (on_server_info), then the client enters
    one of them, see enter(). The server sends the archive of the biome unless the client has it already, it is
    extracted into biomes/<location_id>/ (on_biome) and the application loads the land from there and calls
    biome_loaded(). Then the server sends the list of all models of the biome with the welcome message (on_welcome),
    nothing is rendered until the client has all of them: other messages are kept in the queue until
    the scene is attached.
    """

    def __init__(self, scene=None, renderer=None, url=None, on_welcome=None, on_server_info=None, on_biome=None, on_biome_progress=None):
        self.scene = scene
        self.renderer = renderer
        # called with the list of templates from the welcome message, must call attach() when the models are ready
        self.on_welcome = on_welcome
        # called with the name of the server and the list of models.Biome, must call enter()
        self.on_server_info = on_server_info
        # called with (location_id, folder with the files of the biome, error), must call biome_loaded() when the land is loaded
        self.on_biome = on_biome
        # called with (received pieces, all pieces) while the archive of the biome is downloading
        self.on_biome_progress = on_biome_progress
        self.ready = scene is not None
        self.pending_messages = []
        self.server_name = None
        self.biomes = []
        # the biome the client asked to enter, messages of the previous biome are dropped until the server answers
        self.entering = None
        self.biome_id = None
        self.location_id = None
        self.biome_ready = scene is not None
        # archive of the biome being received: {sha256, size, chunks, parts}
        self.biome_receiving = None
        self.url = url or f'ws://127.0.0.1:{models.DEFAULT_PORT}/'
        self.world = None
        self.hero_id = None
        self.tick_interval = 1.0 / models.TICK_RATE
        # (arrival time, arrival time - tick * tick_interval) of recently received states
        self.tick_clock_samples = collections.deque()
        # local time of the server tick 0, see tick_time()
        self.tick_offset = None
        self.last_render_tick = None
        # object id -> (model received from the server, hero.Hero which renders it, Prediction)
        self.objects = {}
        # object id -> (model, hero.Hero, tick) of the objects waiting until their model and textures are downloaded
        self.loading = {}
        # move, turn, target_direction of the last sent controls
        self.controls_sent = None
        self.controls_seq = 0
        self.controls_last_tick = 0
        # controls seq -> lead used for it, see INPUT_LEAD
        self.controls_lead = {}
        self.input_lead = INPUT_LEAD
        self.input_acknowledged = 0

    def start(self):
        ws.start(callbacks={
            'on_open': self.on_open,
            'on_close': self.on_close,
            'on_message': self.on_message_received if _FakeLag or _FakeJitter else self.on_message,
        }, url=self.url)

    def stop(self):
        if ws.is_started():
            ws.stop()

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
        for json_data, arrived in pending:
            self.on_message(json_data, arrived=arrived)

    #--- CONNECTION ---

    def reset_world(self):
        """
        Forgets all about the current biome: its objects, ticks and the own hero.
        """
        self.pending_messages = []
        self.loading.clear()
        for object_id in list(self.objects.keys()):
            self.despawn(object_id)
        self.world = None
        self.hero_id = None
        if self.scene:
            self.scene.hero = None
        if self.renderer:
            self.renderer.camera_unit_lock = None
        # every biome counts its ticks
        self.tick_clock_samples.clear()
        self.tick_offset = None
        self.last_render_tick = None
        self.controls_sent = None
        self.controls_seq = 0
        self.controls_last_tick = 0
        self.controls_lead.clear()
        self.input_acknowledged = 0
        self.biome_receiving = None

    def on_open(self, ws_inst):
        if _Debug:
            print(f'connected to game server {self.url}')
        ws.send(models.message(models.MSG_JOIN))

    def on_close(self, ws_inst):
        if _Debug:
            print(f'disconnected from game server {self.url}')
        # server will create new hero after reconnect, all that client knows about the world is not valid anymore
        self.reset_world()
        self.entering = None
        self.biome_id = None
        self.biome_ready = False
        self.ready = False

    def on_message_received(self, json_data):
        # debug only: delays incoming messages to simulate slow network, messages still keep their order
        now = time.perf_counter()
        delay = _FakeLag + random.uniform(0.0, _FakeJitter)
        due = max(now + delay, getattr(self, '_fake_last_due', 0.0))
        self._fake_last_due = due
        Clock.schedule_once(lambda dt: self.on_message(json_data), due - now)
        return True

    def on_message(self, json_data, arrived=None):
        # messages waiting in the queue keep the time they arrived at, server ticks are matched to it
        if arrived is None:
            arrived = time.perf_counter()
        msg_type = json_data.get('t')
        payload = json_data.get('p') or {}
        try:
            if msg_type == models.MSG_SERVER_INFO:
                self.on_server_info_received(payload)
                return True
            if msg_type == models.MSG_BIOME:
                self.on_biome_received(payload)
                return True
            if msg_type == models.MSG_BIOME_DATA:
                self.on_biome_data_received(payload)
                return True
        except:
            traceback.print_exc()
            return True
        if self.entering or self.biome_id is None:
            # messages of the previous biome which were sent before the server got "enter"
            return True
        if not self.biome_ready or (not self.ready and msg_type != models.MSG_WELCOME):
            self.pending_messages.append((json_data, arrived))
            return True
        try:
            if msg_type == models.MSG_STATE:
                now = time.perf_counter()
                tick = payload.get('tick')
                self.tick_time(tick, arrived)
                updated = set()
                for d in payload['objects']:
                    self.update(d, tick, now)
                    updated.add(d['id'])
                if tick is not None:
                    # server sends only the objects which were changed, others are still the same at that tick
                    render_tick = self.render_tick(now)
                    for object_id, (_, _, prediction) in self.objects.items():
                        if object_id not in updated and tick > prediction.base_tick:
                            prediction.rebase(tick, now, render_tick)
            elif msg_type == models.MSG_SPAWN:
                tick = payload.get('tick')
                self.tick_time(tick, arrived)
                for d in payload['objects']:
                    self.spawn(models.object_from_dict(d), tick)
            elif msg_type == models.MSG_DESPAWN:
                for object_id in payload['ids']:
                    self.despawn(object_id)
            elif msg_type == models.MSG_WELCOME:
                self.world = models.World.from_data(payload['world'])
                self.hero_id = payload['hero_id']
                if self.world.tick_rate:
                    self.tick_interval = 1.0 / self.world.tick_rate
                if _Debug:
                    print(f'joined world {self.world} of biome {self.location_id}, my hero is {self.hero_id}')
                if self.on_welcome:
                    # wait until models are downloaded, see attach()
                    self.ready = False
                    self.on_welcome(payload.get('templates') or [])
            elif _Debug:
                print(f'unknown message from game server: {json_data}')
        except:
            traceback.print_exc()
        return True

    #--- BIOMES ---

    def on_server_info_received(self, payload):
        self.server_name = payload.get('name') or 'server'
        self.biomes = [models.Biome.from_data(b) for b in payload.get('biomes') or []]
        if _Debug:
            print(f'game server {self.server_name!r} has biomes {[b.id for b in self.biomes]}')
        if self.on_server_info:
            self.on_server_info(self.server_name, self.biomes)

    def enter(self, biome_id):
        """
        Asks the server to move the hero into that biome. The current biome is dropped right away:
        nothing is rendered until the new biome is downloaded and loaded.
        """
        self.reset_world()
        self.entering = biome_id
        self.biome_id = None
        self.biome_ready = False
        self.ready = False
        self.scene = None
        self.renderer = None
        self.location_id = models.location_id(self.server_name, biome_id)
        ws.send(models.message(models.MSG_ENTER, {
            'biome_id': biome_id,
            'sha256': loader.cached_biome_sha256(self.location_id),
        }))
        if _Debug:
            print(f'entering biome {biome_id} as {self.location_id}')

    def on_biome_received(self, payload):
        if payload.get('biome_id') != self.entering:
            return
        self.entering = None
        self.biome_id = payload['biome_id']
        if not payload.get('chunks'):
            # the same archive was received before
            if _Debug:
                print(f'biome {self.location_id} is already on the device')
            self.on_biome_files_ready(loader.biome_dir(self.location_id), None)
            return
        self.biome_receiving = {
            'sha256': payload['sha256'],
            'size': payload['size'],
            'chunks': payload['chunks'],
            'parts': [],
        }
        if _Debug:
            print(f'receiving biome {self.location_id}, {payload["size"]} bytes in {payload["chunks"]} pieces')
        if self.on_biome_progress:
            self.on_biome_progress(0, payload['chunks'])

    def on_biome_data_received(self, payload):
        receiving = self.biome_receiving
        if not receiving or payload.get('biome_id') != self.biome_id:
            return
        if payload.get('index') != len(receiving['parts']):
            self.biome_receiving = None
            self.on_biome_files_ready(None, f'piece {payload.get("index")} of biome {self.biome_id} came out of order')
            return
        receiving['parts'].append(base64.b64decode(payload['data']))
        if self.on_biome_progress:
            self.on_biome_progress(len(receiving['parts']), receiving['chunks'])
        if len(receiving['parts']) < receiving['chunks']:
            return
        self.biome_receiving = None
        archive_data = b''.join(receiving['parts'])
        location_id = self.location_id
        # checking and extracting takes a moment, the window must not freeze meanwhile
        loader.start_thread(self.extract_biome, archive_data, location_id, receiving['sha256'])

    def extract_biome(self, archive_data, location_id, sha256):
        try:
            dest_dir_path = loader.extract_biome(archive_data, location_id, sha256)
            error = None
        except loader.StoppedError:
            return
        except Exception as exc:
            traceback.print_exc()
            dest_dir_path, error = None, str(exc)
        Clock.schedule_once(lambda dt: self.on_biome_files_ready(dest_dir_path, error, location_id))

    def on_biome_files_ready(self, dest_dir_path, error, location_id=None):
        if location_id is not None and location_id != self.location_id:
            # the client went somewhere else meanwhile
            return
        if self.on_biome:
            self.on_biome(self.location_id, dest_dir_path, error)

    def biome_loaded(self):
        """
        The land of the biome was loaded by the application, messages received meanwhile are processed:
        the welcome message comes first, see on_welcome.
        """
        self.biome_ready = True
        pending = self.pending_messages
        self.pending_messages = []
        for json_data, arrived in pending:
            self.on_message(json_data, arrived=arrived)

    #--- OBJECTS ---

    def spawn(self, o, tick=None):
        if tick is None:
            tick = int(self.render_tick(time.perf_counter()) or 0)
        if o.id in self.objects or o.id in self.loading:
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
            store_id=t.store_id,
        )
        files = view.required_files()
        if not files:
            self.create_view(o, view, tick)
            return
        # the unit is shown when its model and textures are downloaded, updates of its state are kept meanwhile
        if _Debug:
            print(f'{o.kind} {o.id} waits for {len(files)} files from store {view.store_id}')
        self.loading[o.id] = (o, view, tick)
        loader.StoreLoader(files, on_done=lambda failed, o=o, view=view: self.on_files_ready(o, view, failed)).start()

    def on_files_ready(self, o, view, failed):
        if self.loading.get(o.id, (None, None, None))[1] is not view:
            # despawned or spawned again while the files were downloading
            return
        _, _, tick = self.loading.pop(o.id)
        if failed:
            print(f'ERROR, {o.kind} {o.id} is not shown, files not downloaded: {failed}')
            return
        try:
            self.create_view(o, view, tick)
        except:
            traceback.print_exc()

    def create_view(self, o, view, tick):
        t = o.template
        view.create_unit(o.w, o.h, elevation_correction=t.elevation_correction, coefs=list(t.coefs), scale=list(t.scale), shift_w=o.shift_w, shift_h=o.shift_h, direction=o.direction)
        prediction = Prediction(o, tick, time.perf_counter())
        self.objects[o.id] = (o, view, prediction)
        self.apply(o, view)
        if o.id == self.hero_id:
            self.scene.hero = view
            self.renderer.camera_unit_lock = view.unit_name
        if _Debug:
            print(f'spawned {o.kind} {o.id} as unit {view.unit_name}')

    def tick_time(self, tick, now):
        """
        Matches server ticks to the local clock. Network delivers states at uneven moments, but the server
        ticks are even: the states which arrived the fastest show how server ticks are aligned to the local clock.
        The timing may move earlier right away, but later only slowly, so the drawn movement never goes back in time.
        """
        if tick is None:
            return
        samples = self.tick_clock_samples
        samples.append((now, now - tick * self.tick_interval))
        while samples and samples[0][0] < now - TICK_CLOCK_WINDOW:
            samples.popleft()
        offset = min(s[1] for s in samples)
        if self.tick_offset is None or offset < self.tick_offset:
            self.tick_offset = offset
        else:
            self.tick_offset = min(offset, self.tick_offset + TICK_CLOCK_DRIFT)

    def render_tick(self, now):
        """
        Fractional number of the server tick which is drawn at that moment, None until the first state arrived.
        """
        if self.tick_offset is None:
            return None
        render_tick = (now - self.tick_offset) / self.tick_interval
        if self.last_render_tick is not None and render_tick < self.last_render_tick:
            render_tick = self.last_render_tick
        self.last_render_tick = render_tick
        return render_tick

    def update(self, d, tick, now=None):
        if d['id'] in self.loading:
            # applied when the unit is created
            o, view, _ = self.loading[d['id']]
            o.update(d)
            self.loading[d['id']] = (o, view, tick)
            return
        if d['id'] not in self.objects:
            return
        if now is None:
            now = time.perf_counter()
        o, view, prediction = self.objects[d['id']]
        o.update(d)
        if o.id == self.hero_id and isinstance(o, models.Unit):
            self.on_controls_acknowledged(o, prediction)
        # the position is not applied right away, the movement continues from this state, see interpolate()
        prediction.rebase(tick, now, self.render_tick(now))
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
        Called every frame before drawing: moves all objects to their positions at this moment.
        """
        render_tick = self.render_tick(now)
        for o, view, prediction in self.objects.values():
            if not prediction.advance(now, render_tick):
                continue
            u = view.get_unit()
            if not u:
                continue
            x, y, direction = prediction.drawn
            w = math.floor(x)
            h = math.floor(y)
            u.set_position(self.scene, w, h, x - w, y - h, direction)

    def despawn(self, object_id):
        self.loading.pop(object_id, None)
        o, view, _ = self.objects.pop(object_id, (None, None, None))
        if not view or not self.scene:
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
        if (move, turn, target_direction) == self.controls_sent:
            return
        # the controls are applied by the server at the requested tick, the hero is drawn moving with them
        # from that tick, see Prediction. Ticks of the sent controls never go back
        render_tick = self.render_tick(time.perf_counter())
        tick = 0
        if render_tick is not None:
            tick = max(int(math.floor(render_tick)) + 1 + int(math.ceil(self.input_lead)), self.controls_last_tick)
        self.controls_seq += 1
        controls = models.Controls(move=move, turn=turn, target_direction=target_direction, tick=tick, seq=self.controls_seq)
        if not ws.send(models.message(models.MSG_INPUT, controls.to_data())):
            self.controls_seq -= 1
            return
        self.controls_sent = (move, turn, target_direction)
        self.controls_last_tick = tick
        if tick:
            # controls without the tick are applied right away, they say nothing about the lead
            self.controls_lead[controls.seq] = self.input_lead
        _, _, prediction = self.objects.get(self.hero_id, (None, None, None))
        if prediction and tick:
            prediction.schedule(controls)

    def on_controls_acknowledged(self, o, prediction):
        """
        Server reported the last applied controls of the own hero and how early they arrived, the lead is corrected.
        """
        prediction.acknowledge(o.input_seq)
        if o.input_seq <= self.input_acknowledged:
            return
        self.input_acknowledged = o.input_seq
        lead_used = self.controls_lead.pop(o.input_seq, None)
        for seq in [s for s in self.controls_lead if s < o.input_seq]:
            self.controls_lead.pop(seq)
        if lead_used is None:
            return
        # with that lead the controls would come INPUT_LEAD_MARGIN ticks before their tick
        lead_needed = lead_used - o.input_slack + INPUT_LEAD_MARGIN
        if lead_needed > self.input_lead:
            self.input_lead = min(INPUT_LEAD_MAX, lead_needed)
        else:
            self.input_lead = max(INPUT_LEAD_MIN, self.input_lead * 0.9 + lead_needed * 0.1)
        if o.input_slack < 0:
            if _Debug:
                print(f'controls {o.input_seq} came {-o.input_slack} ticks late, input lead is {self.input_lead:.1f} ticks now')
