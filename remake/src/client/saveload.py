"""
The window of the saved games of the main and the escape menus (engine.saveload, game.exe 633770 and
634750) as a layer of the interface: the list of the folders of saves/ with the allod, the zone and the
hour of info.sav. The remake only shows the list; loading and saving are not made.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from engine import difficulty as difficulty_module
from engine import saveload as saveload_module
from client import ui

#------------------------------------------------------------------------------

LAYER_NAME = 'saveload'

ATLAS_NAME = 'saveload'

OVERLAY_COLOR = (0.0, 0.0, 0.0, 0.72)
PANEL_COLOR = (0.0, 0.0, 0.0, 1.0)
TITLE_COLOR = (255, 179, 49)
BRIGHT_COLOR = (228, 215, 167)
DIM_COLOR = (150, 150, 150)
SELECTED_COLOR = (255, 255, 255)
HIGHLIGHT_COLOR = (1.0, 1.0, 1.0, 0.10)
EMPTY_COLOR = (150, 150, 150)

TITLE_FONT = ui.FONT_LARGE
ROW_FONT = ui.FONT_MEDIUM

# the details of the chosen slot under the list: what the remake can read and what it cannot
DETAILS_BOX = (180, 430, 620, 500)
# a slot of game.exe: engine.saveload.read_game_save takes the zone, the party and the place of the hero
GAME_SAVE = 'Game save: the zone, the party and the hero'

LOAD = 'load'
# the sounds of the window (buttons/Save/... and buttons/MessBox/... of sfx.res)
SOUND_SELECT = 'buttons/Save/Select.wav'
SOUND_OK = 'buttons/Menu/Ok.wav'
SOUND_CANCEL = 'buttons/MessBox/Cancel.wav'

SAVE = 'save'

TITLES = {
    LOAD: saveload_module.LOAD_TITLE,
    SAVE: saveload_module.SAVE_TITLE,
}

#------------------------------------------------------------------------------


def _color(rgb):
    return tuple(c / 255.0 for c in rgb) + (1.0, )


def slot_text(slot, text_of=None):
    """
    "name  allod, zone, hour:minute" of a slot; the quick save and the autosaves get their texts.
    """
    hour = int(slot.hour) % 24
    minute = int(round((slot.hour - int(slot.hour)) * 60.0))
    place = ', '.join(part for part in (slot.allod, slot.zone) if part)
    name = saveload_module.slot_title(slot.name, text_of) if text_of is not None else slot.name
    return '%s  %s  %02d:%02d' % (name, place, hour, minute)


class SaveLoadView(object):
    """
    The modal list of the saves; V and Enter answer with the chosen slot, X and Escape close the window.
    """

    def __init__(self, app, mode=LOAD):
        self.app = app
        self.mode = mode
        self.dialog = saveload_module.Dialog(saveload_module.slots(app.game_data.data_dir_path),
                                             TITLES[mode])
        self.is_open = False

    def open(self):
        self.is_open = True
        self.reload()
        self.update()
        return True

    def reload(self):
        self.dialog.entries = saveload_module.slots(self.app.game_data.data_dir_path)
        self.dialog.select(min(self.dialog.selected, max(len(self.dialog.entries) - 1, 0)))

    def close(self):
        self.is_open = False
        self.app.interface.set_layer(LAYER_NAME, [])

    def confirm(self):
        slot = self.dialog.selected_slot
        mode = self.mode
        self.close()
        if mode == SAVE:
            name = slot.name if slot is not None and slot.remake else None
            if name is None:
                name = saveload_module.next_name(self.app.game_data.data_dir_path)
            self.app.save_game(name)
            return
        if slot is None:
            self.app.show_message('No saved game')
            return
        self.app.load_game(slot.name)

    def update(self):
        dialog = self.dialog
        items = [ui.box_item((0, 0, ui.WIDTH, ui.HEIGHT), OVERLAY_COLOR)]
        items.extend(ui.panel_items(saveload_module.PANEL_RECT, 4, _color((90, 90, 66)),
                                    _color((132, 130, 90)), PANEL_COLOR))
        items.append(ui.text_item(dialog.title, saveload_module.TITLE_BOX, TITLE_FONT, TITLE_COLOR,
                                  align='center', valign='middle'))
        if not dialog.entries:
            items.append(ui.text_item('No saved games', saveload_module.TITLE_BOX, ROW_FONT, EMPTY_COLOR,
                                      align='center', valign='bottom'))
        for row in range(saveload_module.VISIBLE_ROWS):
            index = dialog.scroll + row
            if index >= len(dialog.entries):
                break
            rect = dialog.row_rect(row)
            if index == dialog.selected:
                items.append(ui.box_item(rect, HIGHLIGHT_COLOR))
            color = SELECTED_COLOR if index == dialog.selected else DIM_COLOR
            items.append(ui.text_item(slot_text(dialog.entries[index], self.app.game_data.text), rect, ROW_FONT, color,
                                      valign='middle'))
        if dialog.selected_slot is not None:
            details = 'Remake save: load it to continue' if dialog.selected_slot.remake else GAME_SAVE
            items.append(ui.text_item(details, DETAILS_BOX, ui.FONT_SMALL, DIM_COLOR))
        items.append(ui.sprite_item(ATLAS_NAME, saveload_module.OK_RECT, difficulty_module.OK_TEXELS))
        items.append(ui.sprite_item(ATLAS_NAME, saveload_module.CANCEL_RECT,
                                    difficulty_module.CANCEL_TEXELS))
        self.app.interface.set_layer(LAYER_NAME, items)

    def on_click(self, x, y):
        """
        True while the window is open: it takes every click.
        """
        if not self.is_open:
            return False
        action = self.dialog.click(x, y)
        if action == 'confirm':
            self.app.ui_sound(SOUND_OK)
            self.confirm()
        elif action == 'cancel':
            self.app.ui_sound(SOUND_CANCEL)
            self.close()
        else:
            if action is not None:
                self.app.ui_sound(SOUND_SELECT)
            self.update()
        return True

    def on_key(self, name):
        """
        True while the window is open: it takes every key.
        """
        if not self.is_open:
            return False
        action = self.dialog.on_key(name)
        if action == 'confirm':
            self.app.ui_sound(SOUND_OK)
            self.confirm()
        elif action == 'cancel':
            self.app.ui_sound(SOUND_CANCEL)
            self.close()
        else:
            self.update()
        return True
