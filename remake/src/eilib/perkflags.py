"""
The flags of the perks (tag 22, +0x40 of the record of perks.pdb): game.exe offers in the camp only the
abilities whose flag is not zero (5260a0); Divination, Healing and Domination are taught by the scripts or
come with a hero template.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from eilib import idb

#------------------------------------------------------------------------------

SECTION_PERKS = 2

#------------------------------------------------------------------------------


def parse_flags(data):
    """
    The flag of every perk of the section 2 of perks.pdb, in the order of the records.
    """
    sections = {node.tag: node for node in idb.root(data).children()}
    section = sections.get(SECTION_PERKS)
    flags = []
    for record in (section.children() if section is not None else []):
        fields = record.fields()
        raw = bytes(fields.get(22) or b'\0')
        flags.append(bool(raw[0]) if raw else False)
    return flags
