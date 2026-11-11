"""
Gives every function of the game code of game.exe a domain: the subsystem it belongs to (units, magic, render...).

The evidence, from the strongest:
    class        the class of re/names.json (CUnitServer -> units)
    guess_class  the class of the named neighbours of an unnamed function
    python       the module of the remake which implements the function (engine/magic.py -> magic)
    strings      the strings the function uses ("Can't lock surface" -> render, "saves\\%s" -> saveload)
    api          the DLL imports it calls (_AIL_* -> sound, Bink* -> video, Reg* -> platform)
Then the domains spread to the functions without evidence:
    block        the nearest functions with a domain before and after it have the same one (one .cpp file)
    callers      all its callers with a domain agree
    callees      most of its callees with a domain agree
    nearest      the nearest function with a domain by address (weak)

The tier tells whether a domain needs Python code at all:
    backend   replaced by Python, Kivy and SDL: nothing to port
    data      file formats: eilib reads them, check only what is not read yet
    game      the behaviour of the game: the place of the work
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import re
import collections

#------------------------------------------------------------------------------

# domain -> (tier, description)
DOMAINS = {
    'runtime': ('backend', 'containers, strings, memory, exceptions (STLport, CRT inlined into the game)'),
    'platform': ('backend', 'window, registry, files, threads, timers, the main loop of Windows'),
    'render': ('backend', 'Direct3D/DirectDraw, surfaces, the 2D library with MMX, upload of textures'),
    'input': ('backend', 'DirectInput, the mouse and the keyboard devices'),
    'video': ('backend', 'Bink movies'),
    'net': ('backend', 'multiplayer: sessions, packets of the network'),
    'data': ('data', 'readers of .res, databases, .reg, text files'),
    'math': ('game', 'vectors, quaternions, random, geometry'),
    'figures': ('game', 'models, animations, materials, textures as game objects'),
    'camera': ('game', 'the camera of the game'),
    'world': ('game', 'the world, map objects, terrain, water, sky, weather, fog, light'),
    'units': ('game', 'units: stats, movement, actions, blows, damage'),
    'ai': ('game', 'behaviours, logic, diplomacy, senses, squads'),
    'pathing': ('game', 'the path grid and the search of the way'),
    'items': ('game', 'items: weapons, armor, quick items, loot, inventory'),
    'magic': ('game', 'spells, effects, particles, lightnings, traps'),
    'script': ('game', 'scripts of the maps'),
    'player': ('game', 'players, parties, quests, experience, zones, global map'),
    'saveload': ('game', 'saving and loading the game'),
    'sound': ('game', 'sounds and music of the game'),
    'ui': ('game', 'windows, menus, the battle and camp screens, cursor, fonts, texts'),
}

CLASS_RULES = (
    (r'^(CEffectSound|CEffectSoundSeq|CEffectAmbientSound|CSoundManager|CSoundSystem|CMusicSystem)$', 'sound'),
    (r'^(CMagicTrap\w*|CSpell\w*|CEffect\w*|SpellEffect|CParticle\w*|Particle\w+|CLightning\w*|Lightning\w+)$', 'magic'),
    (r'^(CItem\w*|CWeaponShape|CArmorShape|CInventory|CTrapObject)$', 'items'),
    (r'^(CUnitLogic|CBehaviour\w*|CDiplomacy|CAiSettings|CSquad|CUnitSignal|CPlayerBehaviour)$', 'ai'),
    (r'^(ServerMap|PathWindow|PathPoint|PointHash|PointAction|MoveTrack)$', 'pathing'),
    (r'^(CScript\w*|Script)$', 'script'),
    (r'^(CPlayer|CPlayerClient|CPlayerManager|CParty|CZoneList|CHeroTemplate|CPerkDatabase|ScienceStats)$', 'player'),
    (r'^(CUnit\w*|UnitRpg|CFakeUnit)$', 'units'),
    (r'^(C\w*Window|CInterface\w*|CCursor\w*|CFont|CModeDial|CPauseDial|CTextManager|CI3DFigure|CInventoryStrip|CHoover)$',
     'ui'),
    (r'^(CAnimation\w*|CFigure\w*|CFakeFigure|CMaterial\w*|CTexture\w*)$', 'figures'),
    (r'^(CCamera)$', 'camera'),
    (r'^(CDbReader|CFileRegistry|Stream|CTextDatabase|CItemDatabase|CUnitDatabase|CMobLoader)$', 'data'),
    (r'^(Vec3|Quat|Random)$', 'math'),
    (r'^(CWorld\w*|CMapObject\w*|CTerrain|CMap|CWeather|CSkyObject|CVisibilityFog|CVisFogManager|CBridgeManager|'
     r'CTorch\w*|CTree|CBuilding|CStone|CStuffObject|CLeverObject|CLight|CLightManager|CObject3D\w*|CObject|'
     r'C\w*SpecificManager)$', 'world'),
)

FREE_NAMES = {
    'SplitAnimation': 'figures', 'LogError': 'platform', 'NameToObjectID': 'script', 'FindQuest': 'player',
    'SetFigureArrays': 'figures', 'LoadFigure': 'figures', 'LoadFigure8': 'figures', 'LoadFigureOld': 'figures',
    'ClampDamageMin': 'units', 'ShareQuestExperience': 'player', 'GetPlayer': 'player', 'MakeItemByName': 'items',
    'LoadLooseFigure': 'figures', 'LoadMenuRects': 'ui', 'LoadCursors': 'ui', 'PlayMovieFile': 'video',
    'SetTiles': 'world', 'SetHollowFrame': 'ui', 'GetItemFigure': 'items', 'MakeItemTexture': 'items',
    'DrawItemInSlot': 'ui', 'AddWindowTexture': 'ui', 'LoadAcksDatabase': 'sound', 'FindAcksOfPrototype': 'sound',
}

PYTHON_RULES = (
    (r'/(ai)\.py$', 'ai'),
    (r'/(pathing)\.py$', 'pathing'),
    (r'/(magic|particles|lightning|missiles|trap|spellvisual|sources)\.py$', 'magic'),
    (r'/(script|mapscript)\.py$', 'script'),
    (r'/(equipment|inventory|quickitems|loot|lootdb)\.py$', 'items'),
    (r'/(unit|stats|actions|combat)\.py$', 'units'),
    (r'/(figure|animation|fig|anm|skel|adb|mmp)\.py$', 'figures'),
    (r'/(camera|cam)\.py$', 'camera'),
    (r'/(terrain|water|lighting|weather|world|lever|prop|footprints|mapdata|zones|mob)\.py$', 'world'),
    (r'/(sounds|music|audio|acks|talk)\.py$', 'sound'),
    (r'/(party|roster|experience|skills|mercenaries|companions|trade|questmap|globalmap|difficulty)\.py$', 'player'),
    (r'/(saveload|sav)\.py$', 'saveload'),
    (r'/(hud|menus|menu|ui|tutorial|briefing|camp|messagebox|credits|options|cursor|keys|zoneexit|signals)\.py$', 'ui'),
    (r'/(movie)\.py$', 'video'),
    (r'/(quat|timing)\.py$', 'math'),
    (r'^src/eilib/', 'data'),
)

STRING_RULES = (
    (r'surface|D3D|DirectDraw|display_mode|vertex_buffer|pixel_format|MMX|graphics|DXDD|end_scene|texture_format|'
     r'HW_3D|DrawingTarget|ZBuffer|ColorDepth|Riched|bitmap_file', 'render'),
    (r'stlport|basic_string|length_error|bad_alloc', 'runtime'),
    (r'multiplayer|session|lobby|DirectPlay|_ipx|_tcp|net_|player_joined|server_', 'net'),
    (r'mouse_device|keyboard_device|DirectInput', 'input'),
    (r'Bink|_bik|movie', 'video'),
    (r'saves_|_sav$|_sav_|saveload|notify_saving|notify_loading|quick_save|autosave', 'saveload'),
    (r'_path$|Work_path|CD_ROM|Windows_|Software_|registry|Registry', 'platform'),
    (r'Unexpected_end_of_file|Wrong_lexeme|Can_t_open_database|_res$|_reg$|_db$|_idb$|_udb$|_sdb$|_pdb$|read_error',
     'data'),
    (r'^buttons_|_window|menu|^tip_|briefing|campinfo|button', 'ui'),
    (r'_wav$|sound|music|_mp3$', 'sound'),
    (r'^Particle_effect|spell|^school', 'magic'),
    (r'^quest|money|^zone$|^allod$|perk|party', 'player'),
    (r'^initwe|^initar|^initq|weapon|armor|item', 'items'),
)

API_RULES = (
    (r'^_AIL_|^mmio|DirectSound', 'sound'),
    (r'^_Bink', 'video'),
    (r'DirectInput|GetKeyState|GetAsyncKeyState|GetCursorPos|SetCursor|ScreenToClient|ClientToScreen', 'input'),
    (r'DirectDraw|Direct3D|^Ordinal_|SetDIBits|StretchDIBits|SelectPalette|CreateDIB|BitBlt', 'render'),
    (r'^Reg[A-Z]|Window|Message|CriticalSection|Thread|Heap|Virtual|Library|ProcAddress|File|Directory|Event|'
     r'Tls|Module|Profile|Locale|Interlocked|^lstr|^wsprintf|Sleep|GetVersion|SystemInfo|GetLastError|'
     r'OutputDebugString|CloseHandle|WaitFor|MultiByteToWideChar|WideCharToMultiByte|Global|Local', 'platform'),
    (r'^timeGetTime$|^GetTickCount$', 'platform'),
    (r'OffsetRect|CopyRect|SetRect|IntersectRect|PtInRect|IsRectEmpty|UnionRect|TextOut|GetTextExtent|'
     r'SelectObject|DeleteObject|CreateFont|SetTextColor|SetBkColor|MoveToEx|LineTo|Polyline|PolyBezier', 'ui'),
)

# domains which an api call proves only when nothing better is known: a game function may read the time too
WEAK_API_DOMAINS = ('platform', 'ui')

#------------------------------------------------------------------------------


def compile_rules(rules):
    return [(re.compile(pattern), domain) for pattern, domain in rules]


CLASS_RES = compile_rules(CLASS_RULES)
PYTHON_RES = compile_rules(PYTHON_RULES)
STRING_RES = compile_rules(STRING_RULES)
API_RES = compile_rules(API_RULES)

#------------------------------------------------------------------------------


def class_domain(class_name):
    for pattern, domain in CLASS_RES:
        if pattern.match(class_name):
            return domain
    return None


def vote(values, rules):
    counts = collections.Counter()
    order = {}
    for value in values:
        for index, (pattern, domain) in enumerate(rules):
            if pattern.search(value):
                counts[domain] += 1
                order.setdefault(domain, index)
                break
    if not counts:
        return None
    # a tie goes to the rule which stands first, the result must not depend on the order of a set
    return max(counts, key=lambda d: (counts[d], -order[d]))


def seed(func):
    if func.class_name:
        domain = class_domain(func.class_name)
        if domain:
            return domain, 'class'
    if func.name in FREE_NAMES:
        return FREE_NAMES[func.name], 'class'
    if func.guess_class:
        domain = class_domain(func.guess_class)
        if domain:
            return domain, 'guess_class'
    domain = vote([p['file'] for p in func.python], PYTHON_RES)
    if domain:
        return domain, 'python'
    domain = vote(func.all_strings, STRING_RES)
    if domain:
        return domain, 'strings'
    domain = vote(func.apis, API_RES)
    if domain and domain not in WEAK_API_DOMAINS:
        return domain, 'api'
    return None, None


def assign(functions):
    game = [f for f in functions if f.kind == 'game']
    by_address = {f.address: f for f in game}
    for func in game:
        func.domain, func.domain_source = seed(func)
    # one .cpp file is one block of code: a function between two of the same domain belongs to it
    known = [i for i, f in enumerate(game) if f.domain]
    for k in range(len(known) - 1):
        a, b = known[k], known[k + 1]
        if b - a > 1 and game[a].domain == game[b].domain:
            for i in range(a + 1, b):
                game[i].domain, game[i].domain_source = game[a].domain, 'block'
    # the callers, then the callees, until nothing changes
    for _ in range(20):
        changed = 0
        for func in game:
            if func.domain:
                continue
            domains = set(by_address[a].domain for a in func.callers if a in by_address and by_address[a].domain)
            if len(domains) == 1:
                func.domain, func.domain_source = domains.pop(), 'callers'
                changed += 1
        for func in game:
            if func.domain:
                continue
            counts = collections.Counter(by_address[a].domain for a in func.calls
                                         if a in by_address and by_address[a].domain)
            if counts:
                domain, n = counts.most_common(1)[0]
                if n * 3 >= sum(counts.values()) * 2:
                    func.domain, func.domain_source = domain, 'callees'
                    changed += 1
        if not changed:
            break
    # the weak api domains and the nearest neighbour for the rest
    for func in game:
        if not func.domain:
            domain = vote(func.apis, API_RES)
            if domain:
                func.domain, func.domain_source = domain, 'api'
    for i, func in enumerate(game):
        if func.domain:
            continue
        for step in range(1, len(game)):
            for j in (i - step, i + step):
                if 0 <= j < len(game) and game[j].domain and game[j].domain_source != 'nearest':
                    func.domain, func.domain_source = game[j].domain, 'nearest'
                    break
            if func.domain:
                break
    if _Debug:
        print(collections.Counter(f.domain for f in game))
