import traceback

import web_sock
import hero

from server import models

_Debug = True


class GameClient(object):
    """
    Connects the scene to the game server.
    Server sends objects of the world and their state, client renders them as they are.
    User input is not applied locally, it is only sent to the server as requested controls.
    """

    def __init__(self, scene, renderer, url=None):
        self.scene = scene
        self.renderer = renderer
        self.url = url or f'ws://127.0.0.1:{models.DEFAULT_PORT}/'
        self.world = None
        self.hero_id = None
        # object id -> (model received from the server, hero.Hero which renders it)
        self.objects = {}
        self.controls_sent = None

    def start(self):
        web_sock.start(callbacks={
            'on_open': self.on_open,
            'on_close': self.on_close,
            'on_message': self.on_message,
        }, url=self.url)

    def stop(self):
        if web_sock.is_started():
            web_sock.stop()

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
        for object_id in list(self.objects.keys()):
            self.despawn(object_id)
        self.hero_id = None
        self.scene.hero = None
        self.renderer.camera_unit_lock = None

    def on_message(self, json_data):
        msg_type = json_data.get('t')
        payload = json_data.get('p') or {}
        try:
            if msg_type == models.MSG_STATE:
                for d in payload['objects']:
                    self.update(d)
            elif msg_type == models.MSG_SPAWN:
                for d in payload['objects']:
                    self.spawn(models.object_from_dict(d))
            elif msg_type == models.MSG_DESPAWN:
                for object_id in payload['ids']:
                    self.despawn(object_id)
            elif msg_type == models.MSG_WELCOME:
                self.world = models.World.from_dict(payload['world'])
                self.hero_id = payload['hero_id']
                if _Debug:
                    print(f'joined world {self.world}, my hero is {self.hero_id}')
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
        self.objects[o.id] = (o, view)
        self.apply(o, view)
        if o.id == self.hero_id:
            self.scene.hero = view
            self.renderer.camera_unit_lock = view.unit_name
        if _Debug:
            print(f'spawned {o.kind} {o.id} as unit {view.unit_name}')

    def update(self, d):
        if d['id'] not in self.objects:
            return
        o, view = self.objects[d['id']]
        o.update(d)
        self.apply(o, view)

    def apply(self, o, view):
        u = view.get_unit()
        if not u:
            return
        u.set_position(self.scene, o.w, o.h, o.shift_w, o.shift_h, o.direction)
        if isinstance(o, models.Unit):
            view.set_animation(o.animation)

    def despawn(self, object_id):
        o, view = self.objects.pop(object_id, (None, None))
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
