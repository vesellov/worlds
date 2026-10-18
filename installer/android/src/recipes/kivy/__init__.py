from os.path import dirname, join

from pythonforandroid.recipes import kivy as stock_kivy


class KivyRecipe(stock_kivy.KivyRecipe):
    """
    Same as the stock p4a Kivy recipe, plus one patch: request a 24-bit depth buffer instead of 16-bit.
    With 16 bits and near/far planes at 2/720 the depth precision is too coarse on Android GPUs,
    which causes z-fighting (flickering land / water / objects). Desktop drivers already give 24 bits.
    """

    # stock patches live next to the stock recipe while apply_patch() resolves names against
    # this recipe dir, so refer to them by absolute path
    patches = []
    for _p in stock_kivy.KivyRecipe.patches:
        if isinstance(_p, tuple):
            patches.append((join(dirname(stock_kivy.__file__), _p[0]), _p[1]))
        else:
            patches.append(join(dirname(stock_kivy.__file__), _p))
    patches.append('gl-depth-24.patch')
    del _p


recipe = KivyRecipe()
