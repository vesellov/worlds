"""
The visual scale of the spells on a unit: Enlarge (37) doubles the figure, Shrink (38) halves it, both
together leave it as it was. game.exe keeps the scale of a model of the client (CUnitClientSpecific) and
the particles of the change are 0x2051/0x2052 (engine.particles); the exact factor of the spells is not
found in the decompiled code, the remake takes the effect of the prototype (2.0) as the factor.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

SPELL_ENLARGE = 37
SPELL_SHRINK = 38

# the field of a figure which keeps the scale without the spells
BASE_SCALE = 'base_scale'

#------------------------------------------------------------------------------


def scale_of(unit):
    """
    The factor of the figure of a unit from its spells, 1.0 without them.
    """
    fighter = getattr(unit, 'fighter', None)
    if fighter is None:
        return 1.0
    factor = 1.0
    if fighter.enchantment(SPELL_ENLARGE) > 0.0:
        factor *= 2.0
    if fighter.enchantment(SPELL_SHRINK) > 0.0:
        factor *= 0.5
    return factor


def update(unit):
    """
    The figure of the unit gets its scale with the spells; True when it changed.
    """
    figure = getattr(unit, 'figure', None)
    if figure is None:
        return False
    base = getattr(figure, BASE_SCALE, None)
    if base is None:
        base = figure.scale
        setattr(figure, BASE_SCALE, base)
    scale = base * scale_of(unit)
    if abs(scale - figure.scale) <= 1e-6:
        return False
    figure.scale = scale
    return True


def update_units(units):
    changed = False
    for unit in units:
        changed = update(unit) or changed
    return changed
