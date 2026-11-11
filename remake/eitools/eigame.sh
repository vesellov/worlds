#!/usr/bin/env bash
# The original game of this copy in its own Wine and on its own virtual X server:
#   eigame.sh start | stop | restart | status
#   eigame.sh input <commands of eidrive.py>   mouse and keys, for example:
#       eigame.sh input click 425 195 ; key Enter ; wait 1000
#   eigame.sh shot <file.png>
#   eigame.sh record <dir> <seconds> [fps]
#   eigame.sh run <scenario> [--out <dir>]
#   eigame.sh xvfb | xvfb-stop
# The display is :77, or EI_DISPLAY/EIDRIVE_ORIGINAL_DISPLAY. The mouse and the keyboard of the
# desktop are not touched: the game lives on a private Xvfb and the input goes there.
# The same for the demo: ../remake/eitools/eidrive.py demo ...
DIR=$(cd "$(dirname "$0")" && pwd)

case "$1" in
xvfb)
	exec python3 - "$DIR" <<'PY'
import os
import sys
sys.path.insert(0, sys.argv[1])
import eidrive
display = eidrive.DISPLAYS['original']
eidrive.ensure_xvfb(display)
print('Xvfb %s is up' % display)
PY
	;;
xvfb-stop)
	d="${EI_DISPLAY:-${EIDRIVE_ORIGINAL_DISPLAY:-:77}}"
	pkill -f "Xvfb ${d} " && echo "Xvfb $d stopped"
	;;
start|stop|restart|status)
	exec python3 "$DIR/eidrive.py" original "$1"
	;;
input)
	shift
	exec python3 "$DIR/eidrive.py" original input "$@"
	;;
shot)
	shift
	exec python3 "$DIR/eidrive.py" original shot "$@"
	;;
record)
	shift
	exec python3 "$DIR/eidrive.py" original record "$@"
	;;
run)
	shift
	exec python3 "$DIR/eidrive.py" original run "$@"
	;;
*)
	echo "usage: $0 start|stop|restart|status|input|shot|record|run|xvfb|xvfb-stop"
	exit 2
	;;
esac
