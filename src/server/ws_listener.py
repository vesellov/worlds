#!/usr/bin/python
# ws_listener.py

"""
..

module:: ws_listener
"""

#------------------------------------------------------------------------------

from __future__ import absolute_import

#------------------------------------------------------------------------------

_Debug = True

#------------------------------------------------------------------------------

import traceback

from twisted.application.strports import listen
from twisted.internet.protocol import Protocol, Factory

#------------------------------------------------------------------------------

from lib import serialization
from server import txws

#------------------------------------------------------------------------------

_DebugLevel = 10

_DebugMessages = False


#------------------------------------------------------------------------------

_WebSocketListener = None
_WebSocketTransports = {}
_APISecret = None
_Callbacks = {}
_AllowRemote = False

#------------------------------------------------------------------------------


def init(port=None, callbacks=None, host='127.0.0.1'):
    """
    Starts listening for web socket connections on the given network interface, localhost by default.
    Use host "0.0.0.0" to accept connections from other machines.
    callbacks: dict with "on_connected(key)", "on_disconnected(key)" and "on_message(key, json_data)" functions,
    key is a unique identifier of the connection.
    """
    global _WebSocketListener
    global _Callbacks
    global _AllowRemote
    if _WebSocketListener is not None:
        if _Debug:
            print('_WebSocketListener already initialized')
        return
    _Callbacks = callbacks or {}
    if not port:
        port = 8280
    if not host:
        host = '127.0.0.1'
    # connections from other machines are refused unless the server was started on another interface on purpose
    _AllowRemote = host not in ('127.0.0.1', 'localhost')
    try:
        ws = WrappedWebSocketFactory(WebSocketFactory())
        _WebSocketListener = listen('tcp:%d:interface=%s' % (port, host), ws)
    except:
        traceback.print_exc()
        return
    if _Debug:
        print('ws_listener.init  _WebSocketListener=%r' % _WebSocketListener)
    # read_api_secret()


def shutdown():
    global _WebSocketListener
    global _Callbacks
    _Callbacks = {}
    if _WebSocketListener:
        if _Debug:
            print('ws_listener.shutdown calling _WebSocketListener.stopListening()')
        _WebSocketListener.stopListening()
        del _WebSocketListener
        _WebSocketListener = None
        if _Debug:
            print('    _WebSocketListener destroyed')
    else:
        if _Debug:
            print('_WebSocketListener is None')


#------------------------------------------------------------------------------


def read_api_secret():
    global _APISecret
    # _APISecret = local_fs.ReadTextFile(settings.APISecretFile())


#------------------------------------------------------------------------------


class WrappedWebSocketProtocol(txws.WebSocketProtocol):

    def validateHeaders(self):
        global _APISecret
        if _APISecret:
            access_granted = False
            _, _, api_secret_parameter = self.location.partition('?')
            param_name, _, api_secret_parameter = api_secret_parameter.partition('=')
            if param_name == 'a' or param_name == 'api_secret':
                if api_secret_parameter == _APISecret:
                    access_granted = True
            if not access_granted:
                # events.send('web-socket-access-denied', data=dict())
                # self.loseConnection()
                return False
        return txws.WebSocketProtocol.validateHeaders(self)

#------------------------------------------------------------------------------


class WrappedWebSocketFactory(txws.WebSocketFactory):

    protocol = WrappedWebSocketProtocol


#------------------------------------------------------------------------------


class WebSocketProtocol(Protocol):

    _key = None

    def dataReceived(self, data):
        try:
            json_data = serialization.BytesToDict(data, keys_to_text=True, values_to_text=True, encoding='utf-8')
        except:
            traceback.print_exc()
            return
        if _DebugMessages:
            print('received %d bytes from web socket %r: %r' % (len(data), self._key, json_data))
        cb = _Callbacks.get('on_message')
        if cb:
            try:
                cb(self._key, json_data)
            except:
                traceback.print_exc()

    def connectionMade(self):
        Protocol.connectionMade(self)
        peer = self.transport.getPeer()
        self._key = (peer.type, peer.host, peer.port)
        _WebSocketTransports[self._key] = self.transport
        if _Debug:
            print(f'connected key={self._key}, ws_connections={len(_WebSocketTransports)}')
        cb = _Callbacks.get('on_connected')
        if cb:
            cb(self._key)

    def connectionLost(self, *args, **kwargs):
        Protocol.connectionLost(self, *args, **kwargs)
        _WebSocketTransports.pop(self._key, None)
        if _Debug:
            print(f'disconnected key={self._key}, ws_connections={len(_WebSocketTransports)}')
        cb = _Callbacks.get('on_disconnected')
        if cb:
            cb(self._key)
        self._key = None


#------------------------------------------------------------------------------

class WebSocketFactory(Factory):

    protocol = WebSocketProtocol

    def buildProtocol(self, addr):
        """
        Only accepting connections from local machine, unless server listens on another interface.
        """
        if not _AllowRemote and addr.host != '127.0.0.1':
            if _Debug:
                print('refused connection from remote host: %r' % addr.host)
            return None
        proto = Factory.buildProtocol(self, addr)
        return proto

#------------------------------------------------------------------------------


def send(key, json_data):
    transp = _WebSocketTransports.get(key)
    if not transp:
        return False
    raw_bytes = serialization.DictToBytes(json_data, encoding='utf-8')
    try:
        transp.write(raw_bytes)
    except:
        traceback.print_exc()
        return False
    if _DebugMessages:
        print('sent %d bytes to web socket %r' % (len(raw_bytes), key))
    return True


def send_many(keys, json_data):
    """
    Sends same message to the given clients, the message is serialized only once.
    """
    raw_bytes = serialization.DictToBytes(json_data, encoding='utf-8')
    for key in keys:
        transp = _WebSocketTransports.get(key)
        if not transp:
            continue
        try:
            transp.write(raw_bytes)
        except:
            traceback.print_exc()
    return True


def push(json_data, exclude=None):
    """
    Sends same message to all connected clients.
    """
    if not _WebSocketTransports:
        return False
    raw_bytes = serialization.DictToBytes(json_data, encoding='utf-8')
    for _key, transp in list(_WebSocketTransports.items()):
        if _key == exclude:
            continue
        try:
            transp.write(raw_bytes)
        except:
            traceback.print_exc()
            continue
    if _DebugMessages:
        print('***   WS PUSH  %d bytes to %d connections' % (len(raw_bytes), len(_WebSocketTransports)))
    return True
