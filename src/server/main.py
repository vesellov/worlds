"""
Game server process.

Runs in its own virtual environment (see "make venv_server"), listens for web socket connections on localhost
and runs the game loop: every tick all requests of the connected clients are processed,
the world state is changed according to the game rules and all changes are sent to the clients.

The server has one or more biomes (worlds), see server.biomes and tools/pack_biome.py. A client which connects
gets the name of the server and the list of biomes, it enters the first biome of the list and can move to another
one later. Before entering a biome the client receives its archive over the web socket, unless it already has it.

    PYTHONPATH=src venv_server/bin/python3 -m server.main [--host 127.0.0.1] [--port 8280] [--biomes ./biomes/]

Use "--host 0.0.0.0" to accept connections from other devices, for example from a phone in the same network.
"""

#------------------------------------------------------------------------------

_Debug = True

#------------------------------------------------------------------------------

import os
import argparse

from twisted.internet import reactor, task

#------------------------------------------------------------------------------

from lib import models
from server import biomes
from server import simulation
from server import ws_listener

#------------------------------------------------------------------------------

# repository root folder, "catalog" and "biomes" folders are there
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#------------------------------------------------------------------------------


class GameServer(object):

    def __init__(self, name, worlds):
        self.name = name
        # biome id -> simulation.Simulation, in the order of the list of biomes
        self.worlds = worlds
        self.first_biome_id = next(iter(worlds))
        # connection key -> (biome id, hero id), the client which did not enter a biome yet is not there
        self.clients = {}
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
            print(f'game server {self.name!r} started on ws://{host}:{port}/ with {models.TICK_RATE} ticks per second, biomes: {list(self.worlds)}')

    def stop(self):
        if self.loop and self.loop.running:
            self.loop.stop()
        ws_listener.shutdown()

    def keys_in_biome(self, biome_id, exclude=None):
        return [key for key, (b, _) in self.clients.items() if b == biome_id and key != exclude]

    def on_connected(self, key):
        pass

    def on_disconnected(self, key):
        self.leave(key)

    def on_message(self, key, json_data):
        msg_type = json_data.get('t')
        payload = json_data.get('p') or {}
        if msg_type == models.MSG_JOIN:
            self.on_join(key)
        elif msg_type == models.MSG_ENTER:
            self.on_enter(key, payload.get('biome_id'), payload.get('sha256'))
        elif msg_type == models.MSG_INPUT:
            biome_id, hero_id = self.clients.get(key, (None, None))
            if hero_id is not None:
                self.worlds[biome_id].set_controls(hero_id, models.Controls.from_data(payload))
        elif _Debug:
            print(f'unknown message from {key}: {json_data}')

    def on_join(self, key):
        # the client chooses where to go, normally the first biome of the list
        ws_listener.send(key, models.message(models.MSG_SERVER_INFO, {
            'name': self.name,
            'protocol': models.PROTOCOL_VERSION,
            'biomes': [w.biome.info().to_data() for w in self.worlds.values()],
        }))

    def leave(self, key):
        """
        Removes the hero of that client from its biome, other clients there see it disappearing.
        """
        biome_id, hero_id = self.clients.pop(key, (None, None))
        if hero_id is None:
            return
        self.worlds[biome_id].remove_object(hero_id)
        ws_listener.send_many(self.keys_in_biome(biome_id), models.message(models.MSG_DESPAWN, {'ids': [hero_id, ]}))
        if _Debug:
            print(f'hero {hero_id} of {key} left biome {biome_id}')

    def on_enter(self, key, biome_id, client_sha256=None):
        world = self.worlds.get(biome_id)
        if not world:
            if _Debug:
                print(f'{key} asked for unknown biome {biome_id!r}')
            return
        self.leave(key)
        biome = world.biome
        # the archive is not sent again if the client has the same one
        send_archive = client_sha256 != biome.sha256
        ws_listener.send(key, models.message(models.MSG_BIOME, {
            'biome_id': biome.id,
            'sha256': biome.sha256,
            'size': biome.size,
            'chunks': biome.chunks_count() if send_archive else 0,
        }))
        if send_archive:
            for index, data in enumerate(biome.chunks()):
                ws_listener.send(key, models.message(models.MSG_BIOME_DATA, {'biome_id': biome.id, 'index': index, 'data': data}))
            if _Debug:
                print(f'sent biome {biome.id} archive to {key}, {biome.size} bytes in {biome.chunks_count()} pieces')
        hero = world.spawn_hero(owner='%s:%s' % (key[1], key[2]))
        self.clients[key] = (biome.id, hero.id)
        ws_listener.send(key, models.message(models.MSG_WELCOME, {
            'world': world.info.to_data(),
            'hero_id': hero.id,
            'biome_id': biome.id,
            # [template, store_id]: client must have all these models before it starts rendering the world
            'templates': world.templates,
        }))
        # new client receives the whole biome, others there only the new hero,
        # objects are at that tick, clients continue their movement from it
        ws_listener.send(key, models.message(models.MSG_SPAWN, {
            'tick': world.tick_number,
            'objects': [o.to_data() for o in world.objects.values()],
        }))
        ws_listener.send_many(self.keys_in_biome(biome.id, exclude=key), models.message(models.MSG_SPAWN, {
            'tick': world.tick_number,
            'objects': [hero.to_data(), ],
        }))

    def on_tick(self):
        for biome_id, world in self.worlds.items():
            changed = world.tick()
            keys = self.keys_in_biome(biome_id)
            if not keys:
                continue
            # sent every tick even if nothing was changed: clients match their clock to the server ticks
            # and know that all other objects are still the same at that tick
            ws_listener.send_many(keys, models.message(models.MSG_STATE, {
                'tick': world.tick_number,
                'objects': [o.to_state() for o in changed],
            }))

#------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description='worlds game server')
    parser.add_argument('--host', default='127.0.0.1', help='network interface to listen on, use 0.0.0.0 to accept connections from other devices')
    parser.add_argument('--port', type=int, default=models.DEFAULT_PORT)
    parser.add_argument('--biomes', dest='biomes_dir_path', metavar='BIOMES', default=os.path.join(ROOT_PATH, 'biomes'), help='folder with server.json and archives of the biomes, see tools/pack_biome.py')
    args = parser.parse_args()
    name, archives = biomes.load(args.biomes_dir_path)
    worlds = {}
    for archive in archives:
        world = simulation.Simulation(biome=archive, catalog_path=os.path.join(ROOT_PATH, 'catalog'))
        world.load()
        worlds[archive.id] = world
    server = GameServer(name, worlds)
    server.start(port=args.port, host=args.host)
    reactor.addSystemEventTrigger('before', 'shutdown', server.stop)  # @UndefinedVariable
    reactor.run()  # @UndefinedVariable


if __name__ == '__main__':
    main()
