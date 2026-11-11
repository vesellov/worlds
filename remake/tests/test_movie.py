"""
Tests of client.movie: the names of the files of data/Movies and the player over the screen.

    PYTHONPATH=src venv/bin/python -m unittest discover -s tests
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import os
import tempfile
import unittest

#------------------------------------------------------------------------------

from client import movie

#------------------------------------------------------------------------------

MOVIE_NAME = 'Intro.BIK'

#------------------------------------------------------------------------------


class FakeVideo(object):

    def __init__(self):
        self.state = 'play'
        self.unloaded = False

    def unload(self):
        self.unloaded = True


class FakeLayout(object):

    def __init__(self):
        self.widgets = []

    def add_widget(self, widget):
        self.widgets.append(widget)

    def remove_widget(self, widget):
        self.widgets.remove(widget)

#------------------------------------------------------------------------------


class MovieTest(unittest.TestCase):

    def test_file_name_without_case(self):
        # game.exe asks for "Movies\Intro.bik", the folder has "intro.bik"
        with tempfile.TemporaryDirectory() as data_dir_path:
            movies_dir_path = os.path.join(data_dir_path, movie.MOVIES_DIR_NAME)
            os.mkdir(movies_dir_path)
            file_path = os.path.join(movies_dir_path, MOVIE_NAME)
            open(file_path, 'wb').close()
            self.assertEqual(movie.movie_file_path(data_dir_path, 'intro.bik'), file_path)
            self.assertEqual(movie.movie_file_path(data_dir_path, 'Movies\\Intro.bik'), file_path)

    def test_missing_file_and_folder(self):
        with tempfile.TemporaryDirectory() as data_dir_path:
            self.assertIsNone(movie.movie_file_path(data_dir_path, 'intro.bik'))
            os.mkdir(os.path.join(data_dir_path, movie.MOVIES_DIR_NAME))
            self.assertIsNone(movie.movie_file_path(data_dir_path, 'intro.bik'))

    def test_stop_keys(self):
        layout = FakeLayout()
        player = movie.MoviePlayer(layout)
        video = FakeVideo()
        layout.add_widget(video)
        player.video = video
        self.assertTrue(player.is_open)
        self.assertFalse(player.on_key('a'))
        self.assertTrue(player.is_open)
        self.assertTrue(player.on_key('escape'))
        self.assertFalse(player.is_open)
        self.assertEqual(video.state, 'stop')
        self.assertTrue(video.unloaded)
        self.assertEqual(layout.widgets, [])

    def test_play_without_a_provider(self):
        # VideoNull plays nothing and never ends, so the movie is not started at all
        provider = movie.CoreVideo
        called = []
        try:
            movie.CoreVideo = None
            self.assertFalse(movie.MoviePlayer(FakeLayout()).play('intro.bik', on_end=lambda: called.append(True)))
        finally:
            movie.CoreVideo = provider
        self.assertEqual(called, [])

    def test_play_without_a_file(self):
        self.assertFalse(movie.MoviePlayer(FakeLayout()).play(None))

    def test_on_end_callback_once(self):
        # the caller waits for the movie: the end of it and a key call the callback once
        layout = FakeLayout()
        player = movie.MoviePlayer(layout)
        video = FakeVideo()
        layout.add_widget(video)
        player.video = video
        called = []
        player.on_end_callback = lambda: called.append(True)
        player.on_end(None, True)
        self.assertEqual(called, [True])
        self.assertFalse(player.is_open)
        self.assertFalse(player.on_key('escape'))
        self.assertEqual(called, [True])

    def test_on_key_calls_the_callback(self):
        layout = FakeLayout()
        player = movie.MoviePlayer(layout)
        video = FakeVideo()
        layout.add_widget(video)
        player.video = video
        called = []
        player.on_end_callback = lambda: called.append(True)
        self.assertTrue(player.on_key('spacebar'))
        self.assertEqual(called, [True])
        self.assertFalse(player.is_open)
