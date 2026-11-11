"""
Scripts of the maps: the SS_TEXT of a .mob in the language of game.exe, parsed into a tree which
engine.script runs.

    GlobalVars ( name : object, ... )             the variables of the map: float, string, object, group
    DeclareScript name ( this : object, ... )      the parameters of a script
    Script name (                                  checked every tick while it waits:
      if ( conditions... ) then ( commands... )   the first branch whose conditions are all nonzero
      ...                                          starts its commands
    )
    WorldScript ( commands... )                    runs once when the map starts

A command is a call "Name( args )", an assignment "x = expression" or "For( x, group ) ( commands )";
a call of a declared script "VCheck#0#1( this )" starts a new copy of it. The names of the variables
may hold "#", "-" and "." ("VSS#i#val", "Human_NPC_Male-699"), the strings are in double quotes.

FUNCTIONS is the table of game.exe (.data 78bd50-78caa0, records {name, id, 0, signature} of the
commands and {name, id, signature} of the functions): the letters of a signature are the types of the
arguments, f float, s string, o object, g group; a function starts with the type it returns
(F, S, O, G). An upper case letter among the arguments is a variable the command fills:
"For( x, group )" puts each member into x, "Every( x, group, condition )" too.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import re

#------------------------------------------------------------------------------

COMMAND = 'command'
FUNCTION = 'function'

FLOAT = 'float'
STRING = 'string'
OBJECT = 'object'
GROUP = 'group'

TYPES = (FLOAT, STRING, OBJECT, GROUP)

TYPE_LETTERS = {'f': FLOAT, 's': STRING, 'o': OBJECT, 'g': GROUP}

# (kind, id of game.exe, name, signature)
FUNCTIONS_TABLE = (
    (COMMAND, 1, 'SetSunLight', 'fffffffff'),
    (COMMAND, 2, 'MoveToPoint', 'off'),
    (COMMAND, 3, 'RotateTo', 'off'),
    (COMMAND, 4, 'MoveToObject', 'oo'),
    (COMMAND, 5, 'StartAnimation', 'os'),
    (COMMAND, 6, 'SetDirectionToObject', 'f'),
    (COMMAND, 7, 'Sleep', 'f'),
    (COMMAND, 9, 'WaitSegment', 'f'),
    (COMMAND, 10, 'WaitEndAnimation', ''),
    (COMMAND, 11, 'CreatePointLight', 'ffffffff'),
    (COMMAND, 12, 'MovePointLight', 'ffffff'),
    (COMMAND, 13, 'DeletePointLight', 'f'),
    (COMMAND, 14, 'CreateParticleSource', 'fffffs'),
    (COMMAND, 15, 'CreateLightning', 'ffffffff'),
    (COMMAND, 16, 'MoveParticleSource', 'ffffff'),
    (COMMAND, 17, 'DeleteParticleSource', 'f'),
    (COMMAND, 18, 'AttachParticleSource', 'ff'),
    (COMMAND, 19, 'SetCameraPosition', 'fff'),
    (COMMAND, 20, 'SetCameraOrientation', 'f'),
    (COMMAND, 22, 'CreateFXSource', 'ffffffs'),
    (COMMAND, 23, 'CreateFX', 'fffffs'),
    (COMMAND, 24, 'CreateRandomizedFXSource', 'fffffffffs'),
    (COMMAND, 25, 'DeleteFXSource', 'f'),
    (COMMAND, 26, 'DeleteLightning', 'f'),
    (COMMAND, 27, 'PlayAnimation', 'os'),
    (COMMAND, 28, 'PlayFX', 'ffffs'),
    (COMMAND, 29, 'PlayMusic', 'fs'),
    (COMMAND, 30, 'SetWind', 'ffff'),
    (COMMAND, 31, 'SetBackGroundColor', 'fff'),
    (COMMAND, 33, 'ShowBitmap', 'fs'),
    (COMMAND, 34, 'AddObject', 'go'),
    (COMMAND, 35, 'RemoveObject', 'go'),
    (COMMAND, 36, 'UMClear', 'o'),
    (COMMAND, 37, 'UMSentry', 'off'),
    (COMMAND, 38, 'UMGuard', 'offf'),
    (COMMAND, 39, 'UMGuardEx', 'offff'),
    (COMMAND, 40, 'UMPatrol', 'o'),
    (COMMAND, 41, 'UMPatrolClear', 'o'),
    (COMMAND, 42, 'UMPatrolAddPoint', 'off'),
    (COMMAND, 43, 'UMPatrolAddPointLook', 'offf'),
    (COMMAND, 44, 'UMStandard', 'o'),
    (COMMAND, 45, 'UMAggression', 'o'),
    (COMMAND, 46, 'UMRevenge', 'o'),
    (COMMAND, 47, 'UMSuspection', 'o'),
    (COMMAND, 48, 'UMCorpseWatcher', 'o'),
    (COMMAND, 49, 'UMPlayer', 'o'),
    (COMMAND, 50, 'BlockUnit', 'of'),
    (COMMAND, 51, 'Guard', 'offf'),
    (COMMAND, 52, 'Idle', 'o'),
    (COMMAND, 53, 'KillUnit', 'o'),
    (COMMAND, 54, 'SetPlayer', 'of'),
    (COMMAND, 55, 'SetDiplomacy', 'fff'),
    (COMMAND, 56, 'SetEnemy', 'of'),
    (COMMAND, 57, 'SendEvent', 'ff'),
    (COMMAND, 58, 'SendStringEvent', 'fs'),
    (COMMAND, 59, 'Follow', 'oo'),
    (COMMAND, 60, 'Attack', 'oo'),
    (COMMAND, 61, 'Cast', 'off'),
    (COMMAND, 62, 'Crawl', 'o'),
    (COMMAND, 63, 'Stand', 'o'),
    (COMMAND, 64, 'Walk', 'o'),
    (COMMAND, 65, 'Run', 'o'),
    (COMMAND, 66, 'SetSpellAggression', 'of'),
    (COMMAND, 67, 'InvokeAlarm', 'fff'),
    (COMMAND, 68, 'InflictDamage', 'of'),
    (FUNCTION, 69, 'IsEqual', 'Fff'),
    (FUNCTION, 70, 'IsLess', 'Fff'),
    (FUNCTION, 71, 'IsGreater', 'Fff'),
    (FUNCTION, 72, 'IsEqualString', 'Fss'),
    (FUNCTION, 73, 'Add', 'Fff'),
    (FUNCTION, 74, 'Sub', 'Fff'),
    (FUNCTION, 75, 'Mul', 'Fff'),
    (FUNCTION, 76, 'Div', 'Fff'),
    (FUNCTION, 77, 'Not', 'Ff'),
    (FUNCTION, 78, 'GetObject', 'Of'),
    (FUNCTION, 79, 'HP', 'Fo'),
    (FUNCTION, 80, 'MaxHP', 'Fo'),
    (FUNCTION, 81, 'Mana', 'Fo'),
    (FUNCTION, 82, 'MaxMana', 'Fo'),
    (FUNCTION, 83, 'DistanceUnitUnit', 'Foo'),
    (FUNCTION, 84, 'DistanceUnitPoint', 'Foff'),
    (FUNCTION, 85, 'GetX', 'Fo'),
    (FUNCTION, 86, 'GetY', 'Fo'),
    (FUNCTION, 87, 'GetZ', 'Fo'),
    (FUNCTION, 88, 'GroupSize', 'Fg'),
    (FUNCTION, 89, 'GroupAdd', 'Ggg'),
    (FUNCTION, 90, 'GroupSub', 'Ggg'),
    (FUNCTION, 91, 'GroupCross', 'Ggg'),
    (FUNCTION, 92, 'GroupHas', 'Fgo'),
    (FUNCTION, 93, 'UnitSee', 'Go'),
    (FUNCTION, 94, 'GroupSee', 'Gg'),
    (FUNCTION, 95, 'Random', 'Ff'),
    (FUNCTION, 96, 'GetDiplomacy', 'Fff'),
    (FUNCTION, 97, 'IsEnemy', 'Fof'),
    (FUNCTION, 98, 'GetPlayerUnits', 'Gf'),
    (FUNCTION, 99, 'IsAlive', 'Fo'),
    (FUNCTION, 100, 'IsDead', 'Fo'),
    (FUNCTION, 101, 'GetPlayer', 'Fo'),
    (FUNCTION, 102, 'GetObjectID', 'Fo'),
    (FUNCTION, 103, 'IsAlarm', 'Ff'),
    (FUNCTION, 104, 'Every', 'FOgf'),
    (FUNCTION, 105, 'Any', 'FOgf'),
    (FUNCTION, 106, 'Sum', 'FOgf'),
    (COMMAND, 107, 'For', 'Og'),
    (COMMAND, 108, 'ForIf', 'Ogf'),
    (COMMAND, 121, 'KillScript', ''),
    (FUNCTION, 124, 'HaveItem', 'Fff'),
    (COMMAND, 125, 'ConsoleFloat', 'f'),
    (COMMAND, 126, 'ConsoleString', 's'),
    (COMMAND, 127, 'GSSetVar', 'fsf'),
    (FUNCTION, 128, 'GSGetVar', 'Ffs'),
    (COMMAND, 130, 'GSDelVar', 'fs'),
    (COMMAND, 131, 'AddUnitUnderControl', 'fo'),
    (COMMAND, 132, 'RemoveUnitFromControl', 'fo'),
    (FUNCTION, 133, 'GetObjectByName', 'Os'),
    (COMMAND, 134, 'AddUnitToParty', 'fss'),
    (COMMAND, 135, 'RemoveUnitFromParty', 'fs'),
    (COMMAND, 136, 'AddUnitToServer', 'sssfff'),
    (COMMAND, 137, 'RemoveUnitFromServer', 'o'),
    (COMMAND, 138, 'LeaveToZone', 'fsf'),
    (FUNCTION, 139, 'AlarmTime', 'Ff'),
    (FUNCTION, 140, 'AlarmPosX', 'Ff'),
    (FUNCTION, 141, 'AlarmPosY', 'Ff'),
    (COMMAND, 142, 'SetCP', 'offf'),
    (FUNCTION, 143, 'GetAIClass', 'Fo'),
    (FUNCTION, 144, 'GetLeverState', 'Fo'),
    (COMMAND, 145, 'SwitchLeverState', 'of'),
    (COMMAND, 146, 'GiveQuestItem', 'fs'),
    (COMMAND, 147, 'EraseQuestItem', 'ff'),
    (COMMAND, 148, 'Sentry', 'off'),
    (COMMAND, 149, 'ActivateTrap', 'of'),
    (COMMAND, 150, 'RemoveObjectFromServer', 'o'),
    (COMMAND, 151, 'GodMode', 'ff'),
    (COMMAND, 152, 'GSSetVarMax', 'fsf'),
    (FUNCTION, 153, 'IsUnitInWater', 'Fo'),
    (COMMAND, 154, 'CreateParty', 'fs'),
    (COMMAND, 155, 'RemoveParty', 'fs'),
    (COMMAND, 156, 'SetCurrentParty', 'fs'),
    (COMMAND, 157, 'CopyStats', 'fss'),
    (COMMAND, 158, 'SwitchLeverStateEx', 'off'),
    (COMMAND, 159, 'QuestComplete', 'fs'),
    (COMMAND, 160, 'AddMob', 's'),
    (COMMAND, 161, 'PlayCamera', 's'),
    (FUNCTION, 162, 'IsCameraPlaying', 'F'),
    (COMMAND, 163, 'ShowCredits', 's'),
    (COMMAND, 164, 'SetPlayerAggression', 'ff'),
    (FUNCTION, 165, 'PlayerSee', 'Gf'),
    (COMMAND, 166, 'GiveItem', 'fs'),
    (COMMAND, 167, 'GiveMoney', 'ff'),
    (COMMAND, 168, 'CastSpellPoint', 'sffff'),
    (COMMAND, 169, 'CastSpellUnit', 'sffo'),
    (FUNCTION, 170, 'IsNight', 'F'),
    (FUNCTION, 171, 'GetObjectByID', 'Os'),
    (COMMAND, 172, 'SetScience', 'offffff'),
    (COMMAND, 173, 'SetCPFast', 'offf'),
    (COMMAND, 174, 'RemoveQuestItem', 'of'),
    (COMMAND, 175, 'GiveUnitQuestItem', 'os'),
    (COMMAND, 176, 'Rest', 'o'),
    (FUNCTION, 177, 'IsPlayerInDanger', 'Ff'),
    (FUNCTION, 178, 'IsPlayerInSafety', 'Ff'),
    (COMMAND, 179, 'GiveUnitSpell', 'os'),
    (FUNCTION, 180, 'GetUnitOfPlayer', 'Off'),
    (FUNCTION, 181, 'GetWorldTime', 'F'),
    (COMMAND, 182, 'RunWorldTime', 'f'),
    (COMMAND, 183, 'FixWorldTime', 'f'),
    (COMMAND, 184, 'SetWaterLevel', 'fff'),
    (COMMAND, 185, 'SleepUntilIdle', 'o'),
    (COMMAND, 186, 'SetParticleSourceSize', 'ff'),
    (FUNCTION, 187, 'IsInSquare', 'Fffffff'),
    (FUNCTION, 188, 'UnitInSquare', 'Foffff'),
    (FUNCTION, 189, 'GetZValue', 'Fff'),
    (FUNCTION, 190, 'GetBSZValue', 'Fff'),
    (FUNCTION, 191, 'GetFutureX', 'Fof'),
    (FUNCTION, 192, 'GetFutureY', 'Fof'),
    (COMMAND, 193, 'PlayMovie', 's'),
    (COMMAND, 194, 'HideObject', 'of'),
    (COMMAND, 195, 'AddRectToArea', 'fffff'),
    (COMMAND, 196, 'AddRoundToArea', 'ffff'),
    (FUNCTION, 197, 'IsInArea', 'Ffff'),
    (COMMAND, 198, 'DeleteArea', 'f'),
    (COMMAND, 199, 'UMFear', 'of'),
    (COMMAND, 200, 'UMAg', 'of'),
    (COMMAND, 201, 'Lie', 'o'),
    (COMMAND, 202, 'UMFollow', 'oo'),
    (COMMAND, 203, 'CopyLoot', 'fss'),
    (COMMAND, 204, 'EnableLever', 'of'),
    (COMMAND, 205, 'GiveSkill', 'osf'),
    (COMMAND, 206, 'GiveStrength', 'of'),
    (COMMAND, 207, 'GiveDexterity', 'of'),
    (COMMAND, 208, 'GiveIntelligence', 'of'),
    (FUNCTION, 209, 'GetMercsNumber', 'Ff'),
    (COMMAND, 210, 'RecalcMercBriefings', ''),
    (COMMAND, 211, 'RedeployParty', 'f'),
    (COMMAND, 212, 'CopyItems', 'fss'),
    (COMMAND, 213, 'QStart', 's'),
    (COMMAND, 214, 'QFinish', ''),
    (COMMAND, 215, 'QObjArea', 'f'),
    (COMMAND, 216, 'QObjSeeObject', 's'),
    (COMMAND, 217, 'QObjSeeUnit', 's'),
    (COMMAND, 218, 'QObjKillUnit', 's'),
    (COMMAND, 219, 'QObjKillGroup', 's'),
    (COMMAND, 220, 'QObjUse', 'sf'),
    (COMMAND, 221, 'QObjGetItem', 'f'),
    (COMMAND, 222, 'SleepUntil', 'f'),
    (COMMAND, 223, 'FixItems', ''),
    (COMMAND, 224, 'AddLoot', 'fss'),
    (FUNCTION, 225, 'GetLootItemsCount', 'Ff'),
    (FUNCTION, 226, 'GetMoney', 'Ff'),
    (FUNCTION, 227, 'WasLooted', 'Fo'),
    (FUNCTION, 228, 'GetLeader', 'O'),
    (FUNCTION, 229, 'IsUnitVisible', 'Fo'),
    (FUNCTION, 230, 'IsUnitBlocked', 'Fo'),
    (COMMAND, 231, 'AttachParticles', 'fo'),
    (COMMAND, 232, 'ResetTarget', 'o'),
)

TOKEN_RE = re.compile(r'\s*(?:("[^"]*")|([(),:=])|([^\s(),:="]+))')

KEYWORDS = ('GlobalVars', 'DeclareScript', 'Script', 'WorldScript', 'if', 'then')

#------------------------------------------------------------------------------


class ScriptError(Exception):
    pass

#------------------------------------------------------------------------------


class Function(object):

    def __init__(self, kind, number, name, signature):
        self.kind = kind
        self.number = number
        self.name = name
        self.signature = signature

    def __repr__(self):
        return 'Function(%s %d %s %s)' % (self.kind, self.number, self.name, self.signature)

    @property
    def returns(self):
        """
        The type a function returns, None for a command.
        """
        if self.kind == COMMAND:
            return None
        return TYPE_LETTERS[self.signature[0].lower()]

    @property
    def arguments(self):
        """
        The letters of the arguments, an upper case one is a variable the call fills.
        """
        return self.signature if self.kind == COMMAND else self.signature[1:]


FUNCTIONS = {name: Function(kind, number, name, signature) for kind, number, name, signature in FUNCTIONS_TABLE}

#------------------------------------------------------------------------------


class Number(object):

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return 'Number(%g)' % self.value


class String(object):

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return 'String(%r)' % self.value


class Variable(object):

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return 'Variable(%s)' % self.name


class Call(object):
    """
    A call of a function or a command of FUNCTIONS (function is the Function) or of a declared script
    (function is None); For has its commands in block.
    """

    def __init__(self, name, args, function=None, block=None, line=0):
        self.name = name
        self.args = args
        self.function = function
        self.block = block
        self.line = line

    def __repr__(self):
        return 'Call(%s, %r%s)' % (self.name, self.args, '' if self.block is None else ', %d commands' % len(self.block))


class Assign(object):

    def __init__(self, name, value, line=0):
        self.name = name
        self.value = value
        self.line = line

    def __repr__(self):
        return 'Assign(%s, %r)' % (self.name, self.value)

#------------------------------------------------------------------------------


class Branch(object):
    """
    "if ( conditions ) then ( commands )" of a script.
    """

    def __init__(self, conditions, commands):
        self.conditions = conditions
        self.commands = commands

    def __repr__(self):
        return 'Branch(%d conditions, %d commands)' % (len(self.conditions), len(self.commands))


class Script(object):

    def __init__(self, name, params, branches):
        self.name = name
        # [(name, type)]
        self.params = params
        self.branches = branches

    def __repr__(self):
        return 'Script(%s, %d branches)' % (self.name, len(self.branches))


class Program(object):

    def __init__(self):
        # name -> type
        self.variables = {}
        # name -> [(name, type)] of DeclareScript
        self.declarations = {}
        # name -> Script
        self.scripts = {}
        self.world_commands = []

    def __repr__(self):
        return 'Program(%d variables, %d scripts, %d world commands)' % (
            len(self.variables), len(self.scripts), len(self.world_commands))

#------------------------------------------------------------------------------


def tokenize(text):
    """
    [(token, line)]: a string keeps its quotes, the others are the punctuation and the words.
    """
    tokens = []
    for line_number, line in enumerate(text.splitlines(), 1):
        position = 0
        line = line.rstrip()
        while position < len(line):
            match = TOKEN_RE.match(line, position)
            if match is None or match.end() == position:
                raise ScriptError('line %d: unexpected %r' % (line_number, line[position:]))
            position = match.end()
            token = match.group(1) or match.group(2) or match.group(3)
            if token:
                tokens.append((token, line_number))
    return tokens


class Parser(object):

    def __init__(self, text, outer_variables=None):
        self.tokens = tokenize(text)
        self.position = 0
        self.program = Program()
        # name -> type of the globals of the map which the scripts of a .mob of AddMob also see
        self.outer_variables = outer_variables or {}

    def peek(self):
        if self.position < len(self.tokens):
            return self.tokens[self.position][0]
        return None

    def line(self):
        if self.position < len(self.tokens):
            return self.tokens[self.position][1]
        return self.tokens[-1][1] if self.tokens else 0

    def take(self, expected=None):
        if self.position >= len(self.tokens):
            raise ScriptError('unexpected end of the script, %r expected' % expected)
        token, line_number = self.tokens[self.position]
        if expected is not None and token != expected:
            raise ScriptError('line %d: %r expected, %r found' % (line_number, expected, token))
        self.position += 1
        return token

    def parse(self):
        while self.peek() is not None:
            line_number = self.line()
            keyword = self.take()
            if keyword == 'GlobalVars':
                self.program.variables.update(self.declarations())
            elif keyword == 'DeclareScript':
                name = self.take()
                self.program.declarations[name] = list(self.declarations().items())
            elif keyword == 'Script':
                self.script()
            elif keyword == 'WorldScript':
                self.program.world_commands.extend(self.block({}))
            else:
                raise ScriptError('line %d: %r is not a section' % (line_number, keyword))
        return self.program

    def declarations(self):
        """
        "( name : type, ... )" -> {name: type}.
        """
        result = {}
        self.take('(')
        while self.peek() != ')':
            if result:
                self.take(',')
            name = self.take()
            self.take(':')
            line_number = self.line()
            kind = self.take()
            if kind not in TYPES:
                raise ScriptError('line %d: unknown type %r' % (line_number, kind))
            result[name] = kind
        self.take(')')
        return result

    def script(self):
        line_number = self.line()
        name = self.take()
        if name not in self.program.declarations:
            raise ScriptError('line %d: script %s is not declared' % (line_number, name))
        params = self.program.declarations[name]
        local = dict(params)
        branches = []
        self.take('(')
        while self.peek() != ')':
            self.take('if')
            self.take('(')
            conditions = []
            while self.peek() != ')':
                conditions.append(self.expression(local))
            self.take(')')
            self.take('then')
            branches.append(Branch(conditions, self.block(local)))
        self.take(')')
        self.program.scripts[name] = Script(name, params, branches)

    def block(self, local):
        commands = []
        self.take('(')
        while self.peek() != ')':
            commands.append(self.command(local))
        self.take(')')
        return commands

    def command(self, local):
        line_number = self.line()
        name = self.take()
        if self.peek() == '=':
            self.take('=')
            self.check_variable(name, local, line_number)
            return Assign(name, self.expression(local), line_number)
        call = self.call(name, local, line_number)
        if call.function is not None and call.function.kind != COMMAND:
            raise ScriptError('line %d: %s is a function, not a command' % (line_number, name))
        if name == 'For':
            call.block = self.block(local)
        return call

    def expression(self, local):
        line_number = self.line()
        token = self.take()
        if token.startswith('"'):
            return String(token[1:-1])
        if token in '(),:=':
            raise ScriptError('line %d: unexpected %r' % (line_number, token))
        try:
            return Number(float(token))
        except ValueError:
            pass
        if self.peek() == '(':
            call = self.call(token, local, line_number)
            if call.function is None or call.function.kind != FUNCTION:
                raise ScriptError('line %d: %s does not return a value' % (line_number, token))
            return call
        self.check_variable(token, local, line_number)
        return Variable(token)

    def call(self, name, local, line_number):
        function = FUNCTIONS.get(name)
        if function is None and name not in self.program.declarations:
            raise ScriptError('line %d: unknown function %s' % (line_number, name))
        args = []
        self.take('(')
        while self.peek() != ')':
            if args:
                self.take(',')
            args.append(self.expression(local))
        self.take(')')
        expected = len(function.arguments) if function else len(self.program.declarations[name])
        if len(args) != expected:
            raise ScriptError('line %d: %s takes %d arguments, %d given' % (line_number, name, expected, len(args)))
        return Call(name, args, function, line=line_number)

    def check_variable(self, name, local, line_number):
        if name not in local and name not in self.program.variables and name not in self.outer_variables:
            raise ScriptError('line %d: unknown variable %s' % (line_number, name))


def parse(text, outer_variables=None):
    """
    Program of the text of SS_TEXT; ScriptError on a mistake. outer_variables: name -> type of the globals
    of the map for the scripts of a .mob of AddMob (zone19alarm1 fills the group a1 of zone19).
    """
    return Parser(text, outer_variables).parse()
