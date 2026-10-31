"""
Game server process.

Runs in its own virtual environment (see "make venv_server"), listens for web socket connections on localhost
and runs the game loop: every tick all requests of the connected clients are processed,
the world state is changed according to the game rules and all changes are sent to the clients.

    venv_server/bin/python3 src/server/main.py [--host 127.0.0.1] [--port 8280]

Use "--host 0.0.0.0" to accept connections from other devices, for example from a phone in the same network.
"""

import os
import sys
import argparse

SERVER_PATH = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.dirname(SERVER_PATH)
ROOT_PATH = os.path.dirname(SRC_PATH)

# server modules are imported as top level modules, shared helpers like "serialization" and "const" are in src/
sys.path.insert(0, SRC_PATH)
sys.path.insert(0, SERVER_PATH)

from twisted.internet import reactor, task

import models
import simulation
import ws_listener

_Debug = True


class GameServer(object):

    def __init__(self, world):
        self.world = world
        # connection key -> id of the hero controlled by that client
        self.heroes = {}
        self.loop = None

    def start(self, port, host='127.0.0.1'):
        ws_listener.init(port=port, host=host, callbacks={
            'on_connected': self.on_connected,
            'on_disconnected': self.on_disconnected,
            'on_message': self.on_message,
        })
        self.loop = task.LoopingCall(self.on_tick)
        self.loop.start(1.0 / models.TICK_RATE, now=False)
        if _Debug:
            print(f'game server started on ws://{host}:{port}/ with {models.TICK_RATE} ticks per second')

    def stop(self):
        if self.loop and self.loop.running:
            self.loop.stop()
        ws_listener.shutdown()

    def on_connected(self, key):
        pass

    def on_disconnected(self, key):
        hero_id = self.heroes.pop(key, None)
        if hero_id is None:
            return
        self.world.remove_object(hero_id)
        ws_listener.push(models.message(models.MSG_DESPAWN, {'ids': [hero_id, ]}))

    def on_message(self, key, json_data):
        msg_type = json_data.get('t')
        payload = json_data.get('p') or {}
        if msg_type == models.MSG_JOIN:
            self.on_join(key)
        elif msg_type == models.MSG_INPUT:
            hero_id = self.heroes.get(key)
            if hero_id is not None:
                self.world.set_controls(hero_id, models.Controls.from_dict(payload))
        elif _Debug:
            print(f'unknown message from {key}: {json_data}')

    def on_join(self, key):
        if key in self.heroes:
            return
        hero = self.world.spawn_hero(owner='%s:%s' % (key[1], key[2]))
        self.heroes[key] = hero.id
        ws_listener.send(key, models.message(models.MSG_WELCOME, {
            'world': self.world.info.to_dict(),
            'hero_id': hero.id,
            # client must have all these models before it starts rendering the world
            'templates': self.world.templates,
        }))
        # new client receives the whole world, others only the new hero
        ws_listener.send(key, models.message(models.MSG_SPAWN, {
            'objects': [o.to_dict() for o in self.world.objects.values()],
        }))
        ws_listener.push(models.message(models.MSG_SPAWN, {'objects': [hero.to_dict(), ]}), exclude=key)

    def on_tick(self):
        changed = self.world.tick()
        if changed:
            ws_listener.push(models.message(models.MSG_STATE, {
                'tick': self.world.tick_number,
                'objects': [o.to_state() for o in changed],
            }))


def main():
    parser = argparse.ArgumentParser(description='worlds game server')
    parser.add_argument('--host', default='127.0.0.1', help='network interface to listen on, use 0.0.0.0 to accept connections from other devices')
    parser.add_argument('--port', type=int, default=models.DEFAULT_PORT)
    args = parser.parse_args()
    world = simulation.Simulation(root_path=ROOT_PATH)
    world.load()
    server = GameServer(world)
    server.start(port=args.port, host=args.host)
    reactor.addSystemEventTrigger('before', 'shutdown', server.stop)  # @UndefinedVariable
    reactor.run()  # @UndefinedVariable


if __name__ == '__main__':
    main()
