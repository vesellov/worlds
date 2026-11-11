# eitools — управление оригиналом и демо в фоне

Подробное описание воспроизведения и использования: [`verify.md`](verify.md).

`eidrive.py` запускает оригинальную игру (`EvilIslands1`, через Wine) и демо (`remake`, `make demo`),
управляет ими мышью и клавиатурой и снимает их окна в PNG/MP4. Оба процесса живут на своих виртуальных
X-серверах (`:77` и `:78`), поэтому курсор, клавиатура и фокус рабочего стола не затрагиваются.

```
EvilIslands1  ->  Wine  ->  Xvfb :77  ->  окно "Evil Islands"      (800x600)
remake        ->  Kivy  ->  Xvfb :78  ->  окно "Evil Islands remake" (800x600)
                    ^                       ^
                    +----- eidrive.py ------+   XTest / SendInput + XGetImage
```

Ввод идёт через XTest на том же виртуальном сервере. Для оригинала есть второй путь — `SendInput`
внутри его Wine (`C:\eitools\eiinput.py`, Windows Python из `EvilIslands1/prefix/drive_c/eitools`);
он включается переменной `EIDRIVE_ORIGINAL_BACKEND=sendinput`. В игровом мире Wine с DirectInput
перехватывает мышь, и абсолютные перемещения SendInput там ненадёжны; поэтому по умолчанию везде XTest.
SendInput удобен для меню и как проверка «изнутри Wine».

## Быстрый старт

```bash
cd /home/veselin/work/eipy

# один раз: окружение демо
cd remake && make venv && cd ..

# запустить оба (Xvfb поднимется сам)
python3 eitools/eidrive.py original start
python3 eitools/eidrive.py demo start

# посмотреть, что на экранах
python3 eitools/eidrive.py original shot /tmp/orig.png
python3 eitools/eidrive.py demo shot /tmp/demo.png

# нажать клавиши и кликнуть (координаты — внутри окна 800x600, ';' разделяет команды)
python3 eitools/eidrive.py original input click 425 195 ';' wait 1000 ';' key Enter
python3 eitools/eidrive.py demo input key Escape ';' wait 500

# остановить
python3 eitools/eidrive.py original stop
python3 eitools/eidrive.py demo stop
```

Старая обёртка для оригинала работает так же:

```bash
bash eitools/eigame.sh start|stop|restart|status|input|shot|record|run|xvfb|xvfb-stop
```

## Цели

| цель | папка | дисплей | заголовок окна | запуск |
|---|---|---|---|---|
| `original` | `EvilIslands1` | `:77` | `Evil Islands` | `bash start.sh` в своём Wine |
| `demo` | `remake` | `:78` | `Evil Islands remake - ...` | `make demo` в `venv` |

Переменные окружения:

| переменная | по умолчанию | смысл |
|---|---|---|
| `EIDRIVE_HOME` | родительская папка `eitools` | папка с `EvilIslands1` и `remake` |
| `EIDRIVE_ORIGINAL_DISPLAY` | `:77` (или `EI_DISPLAY`) | дисплей оригинала |
| `EIDRIVE_DEMO_DISPLAY` | `:78` | дисплей демо |
| `EIDRIVE_ORIGINAL_BACKEND` | `xtest` | `xtest` или `sendinput` |
| `EIDRIVE_DEMO_BACKEND` | `xtest` | только `xtest` |
| `EIDRIVE_DATA_PATH` | `EvilIslands1/game_info/data` | данные для демо (чтобы демо и оригинал читали один и тот же набор) |
| `EIDRIVE_DEMO_AUDIODRIVER` | — | например `dummy`, чтобы демо не играло звук |

`start` сам поднимает Xvfb, если дисплей ещё не работает. `stop` оригинала делает `wineserver -k`
и при необходимости добивает оставшиеся процессы; `stop` демо убивает группу процессов `make demo`.

Если запустить `make demo` вручную без `DISPLAY`, окно откроется на рабочем столе и будет читать
мышь и клавиатуру оттуда. Для изоляции запускайте демо через `eidrive.py demo start`: он делает
тот же `make demo`, но с `DISPLAY=:78` и `EI_DATA_PATH` этой копии игры. То же для оригинала:
`eidrive.py original start` запускает `EvilIslands1/start.sh` на `:77`.

## Ввод

Координаты мыши — клиентские координаты окна игры (800x600), начало в левом верхнем углу.
Команды `input` (и строки сценария):

```
move X Y              переместить курсор
click X Y [B]         клик; B = left (по умолчанию), middle, right, 1, 2, 3
rclick X Y            правый клик
dclick X Y [B]        двойной клик
down [B] / up [B]     нажать/отпустить кнопку (для перетаскиваний)
drag X1 Y1 X2 Y2 [B] [ШАГОВ]   перетаскивание, по умолчанию правая кнопка
rel DX DY [ШАГОВ]     относительное движение (поворот камеры)
wheel up|down [N]     колесо (зум)
key NAME [MS]         нажать и отпустить; MS — удержание, по умолчанию 60 мс
chord NAME NAME [MS]  несколько клавиш одновременно
keydown NAME / keyup NAME
type TEXT             напечатать текст (имена сохранений)
wait MS               пауза
flush                 отправить накопленный ввод (для sendinput)
```

Имена клавиш: `esc`, `enter`, `space`, `tab`, `backspace`, `shift`, `ctrl`, `alt`, `up`, `down`,
`left`, `right`, `home`, `end`, `pageup`, `pagedown`, `insert`, `delete`, `f1`..`f12`, `minus`, `plus`,
буквы и цифры, `kp_up`, `kp_plus` и т.п. Можно писать имена X11 (`Escape`, `Return`, `KP_Add`).
Аккорд — `key ctrl+s` или `chord ctrl s`.

Примеры:

```bash
# New game -> Normal
python3 eitools/eidrive.py original input click 425 195 ';' wait 1200 ';' click 297 482
# идти по клику, затем бегом с Shift
python3 eitools/eidrive.py demo input click 500 420 ';' wait 4000
python3 eitools/eidrive.py demo input keydown shift ';' click 450 200 ';' keyup shift
# повернуть камеру правой кнопкой и приблизить колесом
python3 eitools/eidrive.py original input drag 400 300 520 300 right 10 ';' wheel up 2
# выбрать заклинание в слоте 1 (оригинал) и отменить повторным нажатием
python3 eitools/eidrive.py original input key 1 ';' wait 500 ';' key 1
```

## Снимки и видео

```bash
python3 eitools/eidrive.py original shot /tmp/now.png
python3 eitools/eidrive.py demo record /tmp/demo_run 10 10   # 10 секунд, 10 кадров/с -> /tmp/demo_run.mp4
```

`shot` — XGetImage окна, без композитора и без видимого рабочего стола. `record` — тот же захват
в PNG-кадры и сборка mp4 через `gst-launch-1.0` (нужен `x264enc`).

## Сценарии

Сценарий — текстовый файл, одна команда в строке. Кроме команд ввода понимает:

```
# комментарий
timeout MS           потолок времени на весь прогон; обязателен в сценарии верхнего уровня,
                     при превышении игры останавливаются, прогон считается неудачным
sync                 ждать здесь другую игру (шаги обеих идут вместе)
original: ...        строку выполнять только у оригинала
demo: ...            строку выполнять только у демо
include FILE         выполнить другой сценарий (путь от папки текущего)
note TEXT            записать текст в лог
shot NAME            сохранить <out>/NAME.png
waitpixel X Y R,G,B [TOL] [TIMEOUT_MS]        ждать цвет пикселя (TOL по умолчанию 40)
waitregion X Y W H R,G,B [TOL] [FRACTION] [TIMEOUT_MS]
                                               ждать, пока доля пикселей области совпадёт с цветом
waitstable SETTLE_MS TIMEOUT_MS [POLL_MS]      ждать, пока картинка перестанет меняться
expect_changed NAME [THRESHOLD]                картинка должна отличаться от <out>/NAME.png
expect_same NAME [THRESHOLD]                   картинка должна совпадать с <out>/NAME.png
```

`waitpixel`/`waitregion` хороши для меню и загрузки: главное меню отличается от загрузочного экрана
цветом камня NEW GAME (430,200), а окно обучения — цветом пергамента в области 250,150 300x250.
Примеры лежат в `scenarios/`:

| файл | что делает |
|---|---|
| `common_intro.txt` | главное меню -> New game -> Normal -> Space (ролик) -> обучение -> зона1 |
| `smoke.txt` | ходьба по клику, стрелки камеры, пауза, Q/R оружие, Esc-меню, F10, обучение H |
| `actions.txt` | F5/F8 сохранение и загрузка, Q/E оружие, слот заклинания 1, поворот камеры правой кнопкой, F10 |
| `quests.txt` | TAB: экран заданий у оригинала против имён объектов у демо |
| `options.txt` | главное меню -> Options: страница обучения, разделы и значения настроек |
| `plan1_newgame_tutorial.txt` | New game -> первая подсказка -> все страницы с паузой 2 с -> закрыть -> выход через меню |

План проверки 1 с шагами и результатами: [`plans/plan1_newgame_tutorial.md`](plans/plan1_newgame_tutorial.md).

Запуск на одной цели:

```bash
python3 eitools/eidrive.py demo run eitools/scenarios/smoke.txt --out /tmp/demo_smoke
```

`run` не перезапускает игру: сценарий с `common_intro.txt` ждёт её в главном меню, сценарий без
общей части — уже в игре.

## Сравнение

```bash
python3 eitools/eidrive.py compare eitools/scenarios/smoke.txt
```

`compare` перезапускает обе цели (если не указан `--no-restart`), проигрывает один и тот же сценарий
сначала на оригинале, потом на демо и собирает отчёт:

```
eitools/out/<сценарий>/index.html        отчёт: кадры рядом, разница, проверки, ошибки
eitools/out/<сценарий>/original/*.png     кадры оригинала
eitools/out/<сценарий>/demo/*.png         кадры демо
eitools/out/<сценарий>/<цель>/run.log     лог шагов
eitools/out/<сценарий>/<цель>/results.json
```

В отчёте у каждого кадра слева оригинал, справа демо, справа число — средняя абсолютная разница
пикселей (0 — одинаково, 255 — совсем разное). Материалы, освещение и камера у демо и оригинала
разные, поэтому ноль не ожидается; смотреть надо на раскладку интерфейса и на поведение.

`--out DIR` задаёт другую папку отчёта. `--no-restart` оставляет запущенные процессы как есть
(тогда сценарий должен начинаться с главного меню).

`compare` сам останавливает обе игры сразу после сценария, в том числе при ошибке или `Ctrl+C`.
`--keep` оставляет их открытыми для осмотра. `--finish-timeout SECONDS` (по умолчанию 2) —
сколько ждать вторую игру после того, как первая закончила; затем обе останавливаются.
Сценарии со строками `sync` идут по шагам вместе, поэтому этот запас почти не расходуется.

### Показ на экране

```bash
python3 eitools/eidrive.py compare eitools/scenarios/plan1_newgame_tutorial.txt --screen
```

`--screen` запускает обе игры на дисплее рабочего стола (`$DISPLAY`, можно задать `--display`)
и ставит их рядом на одном мониторе: оригинал слева, демо справа. Сценарий идёт на обеих играх
одновременно, строки `sync` держат шаги вместе; строки лога помечены `[original]` и `[demo]`.

На Wayland (Xwayland) обычный XTest не доставляет ввод на `:0`, поэтому в этом режиме работает
backend `xsend`: указатель ставится `XWarpPointer`, а движение, кнопки и клавиши идут в окно игры
через `XSendEvent`. Синтетическое движение важно для Kivy: оно сообщает позицию указателя, даже
если окно демо перекрыто другим окном.

Таймаут сценария общий для прогона: когда он истекает, обе игры останавливаются, а в отчёте
появляется ошибка `timeout`.

Запуск на этом репозитории (папка `eitools` лежит внутри `remake`, а игры — уровнем выше):

```bash
EIDRIVE_HOME=/home/veselin/work/worlds python3 remake/eitools/eidrive.py compare \
    remake/eitools/scenarios/plan1_newgame_tutorial.txt --screen
```

## Что показал первый прогон

Это не приговор демо, а то, что видно на кадрах `out/smoke` и `out/actions`:

- В зоне1 у демо теперь 50 маны, как у оригинала: `DEMO_MANA` в `client/demo.py` больше не
  поднимает ману до 500. Стартовое сообщение о выбранном заклинании тоже убрано, а HUD
  округляет здоровье, поэтому панель показывает те же `39 / 50`.
- `TAB` у оригинала открывает экран заданий (Quests screen) с картой зоны и обучением;
  у демо `TAB` включает имена объектов (`Object names: on`). Экран заданий у демо, видимо,
  открывается центром диска паузы, но привязка TAB другая.
- Esc-меню у демо рисуется поверх незатемнённого HUD; у оригинала весь экран затемняется,
  и HUD под меню не виден.
- Демо первые 15 секунд в зоне показывает подсказку «DEMO KEYS»; в `common_intro.txt` она
  пережидается, чтобы кадры было проще сравнивать.
- Начальный вид камеры в зоне1 совпадает с оригиналом: `#view` входа — направление взгляда,
  а камера ставится за целью (`Exit.camera_yaw` = `view + 180`).
- Options: у оригинала сначала открывается страница обучения, затем окно с 11 разделами
  (Graphics, Sound, Sensitivity, Game, Selection, Controls, Camera, Objects, Spells, Combat,
  Windows); у демо страницы обучения нет, в списке видно 10 разделов (Windows отсутствует),
  хотя в заголовке «Graphics (1/11)». Значения тоже расходятся: у оригинала тени объектов и
  деревьев выключены (`Off`), у демо включены (`On`); у оригинала есть отдельные ползунки
  Brightness/Contrast/Gamma и пункты Show blood/footprints/path/hit result, у демо их нет.

## Ограничения

- Сценарии рассчитаны на окно 800x600 и на то, что игра не меняет разрешение.
- `record` требует `gst-launch-1.0` с `x264enc`; кадры при этом всё равно сохраняются.
- `sendinput` не умеет колесо и ненадёжен для абсолютных кликов в игровом мире; для игры
  используйте `xtest` (по умолчанию).
- Звук процессов не изолируется виртуальным X-сервером. Демо можно заглушить
  `EIDRIVE_DEMO_AUDIODRIVER=dummy`; у оригинала звук задаётся его настройками (`Audio` в
  `Software\Nival Interactive\EvilIslands\General Settings`).
- `run`/`compare` не трогают окно пользователя, но процессы демо и игры всё равно расходуют CPU.
- Сценарий `actions.txt` пишет быстрые сохранения в общую папку данных:
  `EvilIslands1/game_info/data/saves/saveq` у оригинала и `saves/quick` у демо, потому что демо
  получает `EI_DATA_PATH` этой копии. Если сохранения не нужны, удалите эти папки; если нужно
  изолировать их от сохранений игры — задайте демо отдельный `EIDRIVE_DATA_PATH` с копией данных.
- Оригинал при запуске новой игры может обновить автосохранение `saves/savea` — это его обычное
  поведение, не побочный эффект драйвера.
