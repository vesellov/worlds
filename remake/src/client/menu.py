"""
The main menu and the escape menu of the client, as engine.menus describes them.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import numpy as np

#------------------------------------------------------------------------------

from engine import figure
from engine import menus
from engine import messagebox
from engine import prop as prop_module
from client import credits as credits_view
from client import difficulty as difficulty_view
from client import gl
from client import options as options_view
from client import saveload as saveload_view
from client import ui

#------------------------------------------------------------------------------

NOT_READY_BOX = (40, 560, 760, 590)
NOT_READY_COLOR = (228, 215, 167)

QUIT_LAYER_NAME = 'quit_confirm'
QUIT_TITLE_FALLBACK = 'Quit game'
QUIT_MESSAGE_FALLBACK = 'Do you really want to quit this game?'
QUIT_SOUND_OK = 'buttons/MessBox/Ok.wav'
QUIT_SOUND_CANCEL = 'buttons/MessBox/Cancel.wav'

# the version of game.exe (6335a0) stands lower in its box with Kivy than with GDI and it has no
# shadow there; the text is raised by 14 pixels and the shadow of 1 pixel is drawn first
VERSION_RAISE = 14
VERSION_SHADOW_COLOR = (0, 0, 0)
VERSION_SHADOW_OFFSET = 1

#------------------------------------------------------------------------------


def _color(rgb):
    return tuple(c / 255.0 for c in rgb) + (1.0, )

#------------------------------------------------------------------------------


class MainMenu(object):
    """
    ZoneMainMenuNew seen by MainMenu.cam, the column of the map answers the mouse.
    """

    def __init__(self, app):
        self.app = app
        self.column = None
        self.buttons = menus.main_menu(app.game_data.registry(menus.MENUS_REG) or {})
        # client.difficulty.DifficultyWindow of the new game, None when it is closed
        self.difficulty = None
        # client.credits.CreditsView, None when the credits are closed
        self.credits = None
        # client.options.OptionsView, None when the options are closed
        self.options = None
        # client.saveload.SaveLoadView, None when the list of the saved games is closed
        self.saveload = None

    @property
    def has_modal(self):
        """
        True while a modal window of the main menu is open: the world goes gray and dark.
        """
        return (self.difficulty is not None or self.credits is not None or
                self.options is not None or self.saveload is not None)

    def setup(self):
        app = self.app
        hour = app.args.hour if app.args.hour is not None else menus.main_menu_hour()
        app.setup_world(menus.MAIN_MENU_MAP, hour=hour, with_unit=False)
        keys = app.game_data.camera(menus.MAIN_MENU_CAMERA)
        if keys:
            from_camera = keys[0]
            app.scene.camera = app.scene_camera_class(from_camera.position, from_camera.rotation)
        self.column = app.props_by_nid.get(menus.MAIN_MENU_COLUMN_NID)
        if self.column is not None:
            self.column.idle()
        version = app.game_data.text(menus.VERSION_TEXT)
        if version:
            left, top, right, bottom = menus.VERSION_BOX
            bottom -= VERSION_RAISE
            app.interface.add_text(version.strip(),
                                   (left, top, right + VERSION_SHADOW_OFFSET, bottom + VERSION_SHADOW_OFFSET),
                                   font=ui.FONT_LARGE, color=VERSION_SHADOW_COLOR, align='right', valign='bottom')
            app.interface.add_text(version.strip(), (left, top, right, bottom), font=ui.FONT_LARGE,
                                   color=menus.VERSION_COLOR, align='right', valign='bottom')
        app.interface.set_regions(self.buttons.rects)

    def on_mouse(self, x, y):
        if self.options is not None:
            self.options.on_mouse(x, y)
            return
        if self.difficulty is not None or self.credits is not None or self.saveload is not None:
            return
        changed, special = self.buttons.hover(x, y)
        if not changed or self.column is None:
            return
        self.app.ui_sound('buttons/Menu/Stone.wav')
        if special is None:
            self.column.idle()
        else:
            self.column.play_special(special)
        if _Debug:
            print('main menu: %s' % self.buttons.hovered)

    def on_click(self, x, y):
        if self.difficulty is not None:
            return self.difficulty.on_click(x, y)
        if self.credits is not None:
            return self.credits.on_click(x, y)
        if self.options is not None:
            result = self.options.on_click(x, y)
            if not self.options.is_open:
                self.options = None
            return result
        if self.saveload is not None:
            result = self.saveload.on_click(x, y)
            if not self.saveload.is_open:
                self.saveload = None
            return result
        name = self.buttons.button_at(x, y)
        if name is not None:
            self.app.ui_sound('buttons/Menu/Ok.wav')
        if name == menus.NEW_GAME:
            self.open_difficulty()
        elif name == menus.LOAD_GAME:
            self.open_saveload(saveload_view.LOAD)
        elif name == menus.CREDITS:
            self.open_credits()
        elif name == menus.OPTIONS:
            self.open_options()
        elif name == menus.EXIT_GAME:
            self.app.stop()
        elif name is not None:
            self.app.show_message('%s: not made yet' % name)
        return name is not None

    def on_key(self, name):
        """
        The keys of the main menu: the difficulty window and the credits take all of them while they are open.
        """
        if self.difficulty is not None:
            return self.difficulty.on_key(name)
        if self.credits is not None:
            return self.credits.on_key(name)
        if self.options is not None:
            result = self.options.on_key(name)
            if not self.options.is_open:
                self.options = None
            return result
        if self.saveload is not None:
            result = self.saveload.on_key(name)
            if not self.saveload.is_open:
                self.saveload = None
            return result
        return False

    def update(self, dt_ms):
        """
        The credits scroll while they are open, the tooltips of the options wait for their delay; the main
        menu itself needs no update.
        """
        if self.options is not None:
            self.options.tick(dt_ms)
        if self.credits is not None:
            self.credits.update(dt_ms)
            if not self.credits.is_open:
                self.credits = None
        return self.credits is not None

    def open_difficulty(self):
        """
        New game of the main menu: the window of the difficulty before the intro (633770, 632900).
        """
        self.difficulty = difficulty_view.DifficultyWindow(self.app, on_confirm=self.start_new_game,
                                                          on_cancel=self.close_difficulty,
                                                          selection=self.app.difficulty)
        self.difficulty.open()

    def close_difficulty(self):
        if self.difficulty is not None:
            self.difficulty.close()
            self.difficulty = None

    def open_credits(self):
        """
        Credits of the main menu: res/outro.res, #scroll 25 outro.rtf and the Credits track.
        """
        view = credits_view.CreditsView(self.app)
        if view.open():
            self.credits = view

    def open_options(self):
        """
        Options of the main menu (650380): the groups of the settings and their values.
        """
        view = options_view.OptionsView(self.app)
        view.open()
        self.options = view

    def open_saveload(self, mode):
        """
        Load game of the main menu: the list of the folders of saves/ (633770).
        """
        view = saveload_view.SaveLoadView(self.app, mode)
        view.open()
        self.saveload = view

    def start_new_game(self, level):
        """
        The chosen difficulty, then Movies\\Intro.bik; the game starts when it ends or Escape/Space
        stops it (CMainMenuWindow__OnMessage 632900).
        """
        self.close_difficulty()
        app = self.app
        app.difficulty = level
        if not app.play_movie('intro.bik', on_end=app.start_game):
            app.start_game()

#------------------------------------------------------------------------------


class QuitWindow(object):
    """
    The message box of "Exit to main menu" (engine.messagebox, game.exe 634750): the panel with the title
    "Quit game", the question, the check and the cross of the saveload atlas.
    """

    def __init__(self, app, on_confirm=None, on_cancel=None):
        self.app = app
        self.dialog = messagebox.Dialog()
        self.on_confirm = on_confirm
        self.on_cancel = on_cancel
        self.is_open = False

    def text(self, key, fallback):
        return (self.app.game_data.text(key) or '').replace('\r', '').strip() or fallback

    def open(self):
        self.is_open = True
        self.update()

    def close(self):
        self.is_open = False
        self.app.interface.set_layer(QUIT_LAYER_NAME, [])

    def update(self):
        items = [ui.box_item(messagebox.PANEL_RECT, messagebox.PANEL_COLOR)]
        items.extend(ui.bevel_items(messagebox.PANEL_RECT, messagebox.BORDER_WIDTH,
                                    messagebox.BORDER_HIGH_WIDTH, _color(messagebox.BORDER_COLOR),
                                    _color(messagebox.BORDER_HIGH_COLOR)))
        items.append(ui.text_item(self.text(messagebox.TITLE_KEY, QUIT_TITLE_FALLBACK),
                                  messagebox.TITLE_BOX, ui.FONT_LARGE, messagebox.TITLE_COLOR,
                                  align='center', valign='middle'))
        items.append(ui.text_item(self.text(messagebox.MESSAGE_KEY, QUIT_MESSAGE_FALLBACK),
                                  messagebox.MESSAGE_BOX, ui.FONT_LARGE, messagebox.TEXT_COLOR,
                                  align='center', valign='middle'))
        items.append(ui.sprite_item(messagebox.ATLAS_NAME, messagebox.OK_RECT, messagebox.OK_TEXELS))
        items.append(ui.sprite_item(messagebox.ATLAS_NAME, messagebox.CANCEL_RECT, messagebox.CANCEL_TEXELS))
        self.app.interface.set_layer(QUIT_LAYER_NAME, items)

    def confirm(self):
        self.close()
        if self.on_confirm is not None:
            self.on_confirm()

    def cancel(self):
        self.close()
        if self.on_cancel is not None:
            self.on_cancel()

    def on_click(self, x, y):
        """
        True while the window is open: it takes every click, away from the buttons nothing happens.
        """
        if not self.is_open:
            return False
        action = self.dialog.click(x, y)
        if action == 'confirm':
            self.app.ui_sound(QUIT_SOUND_OK)
            self.confirm()
        elif action == 'cancel':
            self.app.ui_sound(QUIT_SOUND_CANCEL)
            self.cancel()
        return True

    def on_key(self, name):
        """
        True while the window is open: it takes every key, Enter confirms and Escape drops it.
        """
        if not self.is_open:
            return False
        action = self.dialog.on_key(name)
        if action == 'confirm':
            self.app.ui_sound(QUIT_SOUND_OK)
            self.confirm()
        elif action == 'cancel':
            self.app.ui_sound(QUIT_SOUND_CANCEL)
            self.cancel()
        return True

#------------------------------------------------------------------------------


class EscMenu(object):
    """
    The column of the escape menu in the 3D interface over the map, it comes up from under the screen.
    """

    def __init__(self, app):
        self.app = app
        self.column = None
        self.column_view = None
        self.closing = False
        self.regions_before = []
        self.saveload = None
        # client.menu.QuitWindow of "Exit to main menu", None when the question is not shown
        self.confirm = None
        self.buttons = menus.esc_menu(app.game_data.registry(menus.MENUS_REG) or {})

    @property
    def is_open(self):
        return self.column is not None or self.confirm is not None

    @property
    def has_modal(self):
        """
        True while the column or the question of the exit is open: the world goes gray and dark.
        """
        return self.is_open

    def open(self):
        if self.is_open:
            return
        model = figure.FigureModel(self.app.game_data, menus.ESC_MENU_MODEL)
        x, y = menus.ESC_MENU_SCREEN_POINT
        position = gl.interface_point(x, y, menus.ESC_MENU_DEPTH)
        angle = menus.ESC_MENU_TURN
        rotation = (np.cos(angle * 0.5), np.sin(angle * 0.5), 0.0, 0.0)
        self.column = prop_module.Prop(model, prop_module.placement_matrix(position, rotation))
        self.column.play_special(menus.ESC_MENU_SHOW)
        self.closing = False
        self.buttons.hovered = None
        self.column_view = self.app.scene.add_prop(self.column, menus.ESC_MENU_TEXTURE, menus.ESC_MENU_LABELS, interface=True)
        self.regions_before = self.app.interface.regions
        self.app.interface.set_regions(self.buttons.rects)

    def close(self):
        if self.column is not None and not self.closing:
            self.closing = True
            self.column.play_special(menus.ESC_MENU_HIDE)

    def update(self, dt_ms, clock):
        if self.column is None:
            return
        self.column.update(dt_ms, clock)
        if self.closing and not self.column.is_playing:
            self.app.scene.remove_interface_prop(self.column_view)
            self.app.interface.set_regions(self.regions_before)
            self.column = None
            self.column_view = None
            self.closing = False

    def on_mouse(self, x, y):
        if self.confirm is not None or self.saveload is not None:
            return
        if self.column is None or self.closing or self.column.is_playing and self.buttons.hovered is None:
            return
        changed, special = self.buttons.hover(x, y)
        if changed:
            self.app.ui_sound('buttons/Menu/Stone.wav')
        if changed and special is not None:
            self.column.play_special(special)

    def on_click(self, x, y):
        if self.confirm is not None:
            result = self.confirm.on_click(x, y)
            if not self.confirm.is_open:
                self.confirm = None
            return result
        if self.saveload is not None:
            result = self.saveload.on_click(x, y)
            if not self.saveload.is_open:
                self.saveload = None
            return result
        if self.column is None or self.closing:
            return False
        name = self.buttons.button_at(x, y)
        if name is not None:
            self.app.ui_sound('buttons/Menu/Ok.wav')
        if name == menus.RESUME_GAME:
            self.close()
        elif name == menus.EXIT_TO_MAIN_MENU:
            self.ask_quit()
        elif name == menus.SAVE_GAME:
            self.open_saveload(saveload_view.SAVE)
        elif name == menus.LOAD_GAME:
            self.open_saveload(saveload_view.LOAD)
        elif name is not None:
            self.app.show_message('%s: not made yet' % name)
        return True

    def on_key(self, name):
        """
        The keys of the escape menu: the question of the exit and the list of the saved games take all of
        them while they are open.
        """
        if self.confirm is not None:
            result = self.confirm.on_key(name)
            if not self.confirm.is_open:
                self.confirm = None
            return result
        if self.saveload is not None:
            result = self.saveload.on_key(name)
            if not self.saveload.is_open:
                self.saveload = None
            return result
        return False

    def ask_quit(self):
        """
        Exit to main menu (62b00d): the column goes down and the message box "Quit game" asks the question;
        the check returns to the main menu, the cross goes back to the game.
        """
        self.confirm = QuitWindow(self.app, on_confirm=self.app.open_main_menu)
        self.confirm.open()
        self.close()

    def open_saveload(self, mode):
        view = saveload_view.SaveLoadView(self.app, mode)
        view.open()
        self.saveload = view
