"""
The things of a player which the scripts give and take (the player object of the server, 594180):
the money (+0xc0), the items of the common inventory (+0xe4) and the quests.

- GiveItem and GiveQuestItem (0xa6, 0x92) make an item by its name (594ab0 / 6ac930) and put it into
  the inventory (665430); a quest item is an item of the class 0x3009 (the section 5 of items.idb).
- EraseQuestItem( player, n ) (0x93, 665740) takes the first quest item whose number (+0x4c) is n.
- HaveItem( player, n ) (0x7c, 665600) tells whether the player has the quest item of the number.
- GiveMoney (0xa7) adds to +0xc0, GetMoney (0xe2) reads it.
- QuestComplete( player, name ) (0x9f) finds the quest of quests.qdb and gives its experience to the party
  (5911c0 shares it by the square roots of the experiences of the units); the remake keeps the name and
  has no experience yet.

The keys of the levers (engine.lever, the flag 8 of 522a80) are the numbers of the quest items.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------


class Party(object):

    def __init__(self, item_database=None, quests=None):
        # eilib.idb.ItemDatabase and the name -> eilib.idb.Quest of quests.qdb
        self.item_database = item_database
        self.quests = quests or {}
        self.money = 0
        # the names of the ordinary items
        self.items = []
        # eilib.idb.QuestItem
        self.quest_items = []
        # the names of the completed quests
        self.completed_quests = []
        # called with (kind, value): 'item', 'quest_item', 'quest_item_lost', 'money', 'quest'
        self.on_change = None

    def __repr__(self):
        return 'Party(money %d, %d items, %d quest items)' % (self.money, len(self.items), len(self.quest_items))

    def changed(self, kind, value):
        if self.on_change is not None:
            self.on_change(kind, value)

    def give_item(self, name):
        """
        GiveItem / GiveQuestItem: a quest item when items.idb has one of the name, an ordinary item otherwise.
        """
        quest_item = self.item_database.quest_item(name) if self.item_database is not None else None
        if quest_item is not None:
            self.quest_items.append(quest_item)
            self.changed('quest_item', quest_item)
            return quest_item
        self.items.append(name)
        self.changed('item', name)
        return name

    def erase_quest_item(self, number):
        for quest_item in self.quest_items:
            if quest_item.number == number:
                self.quest_items.remove(quest_item)
                self.changed('quest_item_lost', quest_item)
                return True
        return False

    def has_quest_item(self, number):
        return any(q.number == number for q in self.quest_items)

    def give_money(self, amount):
        self.money += int(amount)
        self.changed('money', int(amount))

    def complete_quest(self, name):
        """
        QuestComplete: the quest of quests.qdb, None when there is none.
        """
        quest = self.quests.get(name.lower())
        if quest is None:
            return None
        self.completed_quests.append(quest.name)
        self.changed('quest', quest)
        return quest

    def keys(self):
        """
        The numbers of the keys for the levers.
        """
        return tuple(q.number for q in self.quest_items)
