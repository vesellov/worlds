"""
Reader of the plain text of an RTF file: the credits of res/outro.res (outro.rtf).

Only what the credits need: the groups of the font table, the color table, the styles and the pictures are
skipped, the control words \\par and \\line become line breaks, \\tab a tab, \\'hh a byte of cp1251 and
\\uN a Unicode character. The other control words and their numeric arguments are dropped.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

ENCODING = 'cp1251'

# the destinations whose text is not the text of the document
SKIPPED_DESTINATIONS = (
    'fonttbl', 'colortbl', 'stylesheet', 'info', 'pict', 'header', 'footer',
    'headerl', 'headerr', 'footerl', 'footerr', 'listtable', 'listoverridetable',
    'rsidtbl', 'generator', 'filetbl', 'revtbl', 'themedata', 'datastore',
)

LINE_BREAKS = ('par', 'line', 'sect', 'page')

SPECIAL_CHARS = {
    'tab': '\t',
    '~': ' ',
    '_': '‑',
    '-': '',
}

#------------------------------------------------------------------------------


def plain_text(data):
    """
    The text of an RTF document, the groups of the skipped destinations and the control words removed.
    """
    text = bytes(data).decode(ENCODING, errors='replace')
    result = []
    # [(skipped, whether the group has text of the document)]
    stack = []
    skipped = False
    uc_skip = 1
    skip_chars = 0
    i = 0
    while i < len(text):
        char = text[i]
        if char == '{':
            stack.append((skipped, uc_skip))
            i += 1
            continue
        if char == '}':
            if stack:
                skipped, uc_skip = stack.pop()
            i += 1
            continue
        if char == '\\':
            i += 1
            if i >= len(text):
                break
            char = text[i]
            if char == "'":
                code = text[i + 1:i + 3]
                i += 3
                if skip_chars > 0:
                    skip_chars -= 1
                elif not skipped:
                    try:
                        result.append(bytes([int(code, 16)]).decode(ENCODING, errors='replace'))
                    except ValueError:
                        pass
                continue
            if char == '\\' or char == '{' or char == '}':
                if not skipped:
                    result.append(char)
                i += 1
                continue
            if char in SPECIAL_CHARS:
                if not skipped:
                    result.append(SPECIAL_CHARS[char])
                i += 1
                continue
            if char == '*':
                # a destination group like {\*\panose ...}
                skipped = True
                i += 1
                continue
            # a control word: letters, an optional minus and digits, then one space
            start = i
            while i < len(text) and text[i].isalpha():
                i += 1
            word = text[start:i]
            number = None
            if i < len(text) and (text[i] == '-' or text[i].isdigit()):
                number_start = i
                if text[i] == '-':
                    i += 1
                while i < len(text) and text[i].isdigit():
                    i += 1
                try:
                    number = int(text[number_start:i])
                except ValueError:
                    number = None
            if i < len(text) and text[i] == ' ':
                i += 1
            if word in SKIPPED_DESTINATIONS:
                skipped = True
                if stack:
                    stack[-1] = (stack[-1][0], stack[-1][1])
                continue
            if word == 'uc' and number is not None:
                uc_skip = number
                continue
            if word == 'u' and number is not None:
                if not skipped:
                    if number < 0:
                        number += 65536
                    result.append(chr(number))
                skip_chars = uc_skip
                continue
            if word in LINE_BREAKS:
                if not skipped:
                    result.append('\n')
                continue
            if word == 'emdash':
                if not skipped:
                    result.append('—')
                continue
            if word == 'endash':
                if not skipped:
                    result.append('–')
                continue
            if word == 'lquote':
                if not skipped:
                    result.append('‘')
                continue
            if word == 'rquote':
                if not skipped:
                    result.append('’')
                continue
            if word == 'ldblquote':
                if not skipped:
                    result.append('“')
                continue
            if word == 'rdblquote':
                if not skipped:
                    result.append('”')
                continue
            continue
        if skip_chars > 0:
            # the characters after \uN which pick another code page are dropped too
            skip_chars -= 1
            i += 1
            continue
        if char in '\r\n':
            i += 1
            continue
        if not skipped:
            result.append(char)
        i += 1
    lines = [' '.join(line.split()) for line in ''.join(result).splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return '\n'.join(lines)
