# verify.md — как проверять демо против оригинала

Дата работы: 2026-11-10. Папка: `/home/veselin/work/eipy`.
Всё описанное лежит в `eitools/` и не требует правок в `remake/`.

Документ отвечает на два вопроса:

1. как воспроизвести эту работу с нуля (окружение, запуск, прогоны, отчёты);
2. как пользоваться утилитой `eitools/eidrive.py` для собственных проверок.

Короткая шпаргалка есть в [`README.md`](README.md); здесь — подробно, с командами,
ожидаемым выводом и разбором уже полученных результатов.

---

## 1. Что сделано

Оригинал и демо запускаются на **отдельных виртуальных X-серверах**:

```
EvilIslands1/start.sh (Wine)  ->  Xvfb :77  ->  окно "Evil Islands"        800x600
remake (make demo)            ->  Xvfb :78  ->  окно "Evil Islands remake" 800x600
                                      ^
                                      |
                           eidrive.py: XTest + XGetImage
```

Поэтому:

- мышь и клавиатура рабочего стола не захватываются: ввод идёт на `:77`/`:78`;
- окна можно снимать в PNG/MP4 без композитора и видимого рабочего стола;
- один и тот же сценарий проигрывается на обеих целях, кадры складываются в отчёт
  «оригинал | демо | разница».

Реализованы два пути ввода:

| путь | где работает | чем хорош | ограничения |
|---|---|---|---|
| `xtest` (по умолчанию) | и меню, и игровой мир | одинаково для Wine и Kivy, надёжные клики | события идут через X-сервер цели |
| `sendinput` | оригинал, внутри Wine | `SendInput` user32, как «изнутри игры» | абсолютные клики в игровом мире Wine с DirectInput ненадёжны; нет колеса; только оригинал |

`sendinput` включается так: `EIDRIVE_ORIGINAL_BACKEND=sendinput`. Для игрового мира
используйте `xtest` (он и стоит по умолчанию).

---

## 2. Состав

| путь | что это |
|---|---|
| `eitools/eidrive.py` | драйвер: запуск/остановка, ввод, снимки, видео, сценарии, сравнение |
| `eitools/eigame.sh` | обёртка для оригинала (`start/stop/status/input/shot/record/run`) |
| `eitools/README.md` | краткая шпаргалка |
| `eitools/verify.md` | этот документ |
| `eitools/scenarios/common_intro.txt` | общий вход: меню -> New game -> Normal -> ролик -> обучение -> зона1 |
| `eitools/scenarios/smoke.txt` | ходьба, камера, пауза, оружие, Esc-меню, F10, обучение H |
| `eitools/scenarios/actions.txt` | F5/F8, оружие, слот заклинания, поворот камеры, F10 |
| `eitools/scenarios/quests.txt` | TAB: экран заданий против имён объектов |
| `eitools/scenarios/options.txt` | главное меню -> Options |
| `eitools/scenarios/plan1_newgame_tutorial.txt` | New game -> Normal -> три страницы обучения -> выход через меню |
| `eitools/scenarios/plan2_options_gamma.txt` | Options -> Gamma на максимум, минимум, середину с OK -> EXIT |
| `eitools/scenarios/plan3_options_groups.txt` | Options: все 11 разделов, подсказка -> X -> EXIT |
| `eitools/scenarios/PLAN_FIX.md` | план сравнения и исправления с прогрессом; `planN_*.md` рядом со сценариями — отчёты по кускам |
| `eitools/out/<сценарий>/index.html` | готовый отчёт с кадрами рядом |
| `eitools/out/<сценарий>/<цель>/` | PNG-кадры, `results.json`, `run.log` |
| `eitools/run/` | логи запуска целей (`original.log`, `demo.log`), pid |
| `EvilIslands1/prefix/drive_c/eitools/` | Windows Python + `eiinput.py` для SendInput |

`eitools/run/` и `eitools/out/` — генерируемые, они в `.gitignore`. Если результаты нужно
сохранить в версии, копируйте папку отчёта целиком в другое место (см. `--out`).

---

## 3. Подготовка окружения

### 3.1. Проверить данные игры

```bash
cd /home/veselin/work/eipy
ls EvilIslands1/game_info/data/res EvilIslands1/game_info/data/maps
ls EvilIslands1/game_info/data/game.exe
```

Должны быть `res/`, `maps/`, `config/`, `game.exe`. Демо получает `EI_DATA_PATH`
этой копии автоматически (`EvilIslands1/game_info/data`), чтобы оригинал и демо
читали один и тот же набор данных.

### 3.2. Собрать venv демо

```bash
cd remake
make venv
cd ..
```

Нужен `python3.12`. Ставятся Kivy 2.3.1, numpy, Pillow, ffpyplayer (см.
`remake/requirements.txt`). Без venv `eidrive.py demo start` выдаст ошибку
`the demo has no venv, run "make venv"`.

### 3.3. Проверить Windows-утилиты Wine

```bash
ls EvilIslands1/prefix/drive_c/eitools/eiinput.py
ls EvilIslands1/prefix/drive_c/eitools/python/python.exe
```

Это нужно только для `sendinput`. В этой копии папка уже есть. Если её нет —
скопируйте из соседней копии:

```bash
cp -a /home/veselin/work/worlds/EvilIslands1/prefix/drive_c/eitools \
      /home/veselin/work/eipy/EvilIslands1/prefix/drive_c/
```

### 3.4. Проверить системные зависимости драйвера

Драйвер запускается системным `python3` (в этой среде 3.14.4) и требует Pillow:

```bash
python3 -c "import PIL; print(PIL.__version__)"   # 12.1.1 в этой среде
which Xvfb xwininfo gst-launch-1.0
```

`Xvfb` и `xwininfo` обязательны; `gst-launch-1.0` с `x264enc` нужен только для
`record` (mp4). Снимки PNG работают и без gstreamer.

Версии этой среды: Wine `wine-11.0.r0.g6cc805ea (TkG Staging NTsync)`,
Kivy 2.3.1, gstreamer 1.28.2, версия игры 1.0.6.

---

## 4. Быстрый запуск

```bash
cd /home/veselin/work/eipy

python3 eitools/eidrive.py original start
python3 eitools/eidrive.py demo start

python3 eitools/eidrive.py original status
python3 eitools/eidrive.py demo status

python3 eitools/eidrive.py original shot /tmp/orig.png
python3 eitools/eidrive.py demo shot /tmp/demo.png
```

Ожидаемый вывод `start` (номер окна может отличаться):

```
original started: 0xa0000b 'Evil Islands' on :77
demo started: 0x200007 'Evil Islands remake - main menu' on :78
```

`status` показывает дисплей, backend, окно и процессы. Если дисплея нет,
`start` сам поднимает `Xvfb :77`/`:78` с экраном 1024x768x24.

Остановка:

```bash
python3 eitools/eidrive.py original stop
python3 eitools/eidrive.py demo stop
```

`stop` оригинала делает `wineserver -k` и добивает оставшиеся процессы; `stop`
демо убивает группу `make demo`. Xvfb остаются работать — они дешёвые и
переиспользуются.

---

## 5. Полное воспроизведение уже сделанных прогонов

Четыре команды; каждая перезапускает обе цели и пишет отчёт в `eitools/out/`:

```bash
cd /home/veselin/work/eipy

python3 eitools/eidrive.py compare eitools/scenarios/smoke.txt
python3 eitools/eidrive.py compare eitools/scenarios/actions.txt
python3 eitools/eidrive.py compare eitools/scenarios/quests.txt
python3 eitools/eidrive.py compare eitools/scenarios/options.txt
```

Ожидаемый итог последних строк:

```
report /home/veselin/work/eipy/eitools/out/smoke/index.html
original: 13 shots, 0 errors
demo: 13 shots, 0 errors
```

и аналогично: `actions` — 12 кадров, `quests` — 6, `options` — 4; везде 0 errors.
Примерное время на этой машине (включая перезапуск оригинала и ожидание меню):

| сценарий | время | что проверяет |
|---|---|---|
| `smoke.txt` | ~70 с | меню, сложность, ролик, обучение, ходьба, камера, пауза, оружие, Esc-меню, F10, H |
| `actions.txt` | ~75 с | F5/F8, оружие, слот заклинания, правый drag камеры, F10 |
| `quests.txt` | ~45 с | TAB в зоне1 |
| `options.txt` | ~16 с | Options из главного меню |

Открыть отчёты (локальный браузер):

```bash
xdg-open eitools/out/smoke/index.html
# или просто открыть файл из проводника
```

Если нужно сохранить прогон, не затирая прошлый:

```bash
python3 eitools/eidrive.py compare eitools/scenarios/smoke.txt --out /tmp/smoke_2026_11_10
```

`--no-restart` не перезапускает цели; тогда обе должны уже стоять в главном меню
(для сценариев с `common_intro.txt`) или в нужном состоянии.

---

## 6. Как читать отчёт

```
eitools/out/smoke/
  index.html                 <- открывать этот файл
  original/
    01_menu.png ... 13_end.png
    results.json             <- список кадров, проверки, ошибки, длительность
    run.log                  <- пошаговый лог сценария
  demo/
    ... то же для демо
```

В `index.html` на каждый шаг одна строка:

- слева кадр оригинала, справа кадр демо (клик по картинке открывает её целиком);
- `diff` — средняя абсолютная разница пикселей 0..255. Материалы, освещение и
  камера у демо и оригинала разные, поэтому 0 не ожидается; смотреть надо на
  раскладку интерфейса и поведение;
- внизу — блок «Проверки» (`ok`/`FAIL`) и «Ошибки» (пусто, если всё прошло).

`results.json` удобно читать скриптом: там `shots`, `checks`, `errors`, `duration`,
`target`, `display`, `backend`.

---

## 7. Использование утилиты

### 7.1. Цели

| цель | папка | дисплей | окно | запуск |
|---|---|---|---|---|
| `original` | `EvilIslands1` | `:77` | `Evil Islands` | `bash start.sh` |
| `demo` | `remake` | `:78` | `Evil Islands remake - ...` | `make demo` |

Окно демо меняет заголовок (`... - main menu`, `... - map zone1`) — драйвер ищет
по префиксу `Evil Islands remake`. Окно оригинала ищется точно по `Evil Islands`.

### 7.2. Команды верхнего уровня

```text
eidrive.py original|demo start|stop|restart|status
eidrive.py original|demo shot FILE
eidrive.py original|demo record DIR SECONDS [FPS]
eidrive.py original|demo input <команда> [аргументы] [';' <команда> ...]
eidrive.py original|demo run SCENARIO [--out DIR]
eidrive.py compare SCENARIO [--out DIR] [--no-restart] [--screen [--display :0]]
                            [--keep] [--finish-timeout SECONDS]
```

`input` принимает несколько команд через `;` в одном запуске. Координаты —
клиентские координаты окна, `(0,0)` — левый верхний угол, окно 800x600.

### 7.3. Ввод

| команда | смысл |
|---|---|
| `move X Y` / `abs X Y` | переместить курсор |
| `click X Y [B]` | клик; `B`: `left` (по умолчанию), `middle`, `right`, `1/2/3` |
| `rclick X Y` | правый клик |
| `dclick X Y [B]` | двойной клик |
| `down [B]` / `up [B]` | нажать/отпустить кнопку |
| `drag X1 Y1 X2 Y2 [B] [ШАГОВ]` | перетаскивание (по умолчанию правой кнопкой) |
| `rel DX DY [ШАГОВ]` | относительное движение (поворот камеры) |
| `wheel up|down [N]` | колесо (зум); `sendinput` его не умеет |
| `key NAME [MS]` | нажать и отпустить, `MS` — удержание (60 мс по умолчанию) |
| `chord NAME NAME [MS]` | несколько клавиш одновременно |
| `keydown NAME` / `keyup NAME` | удержание клавиши |
| `type TEXT` | напечатать текст (например, имя сохранения) |
| `wait MS` | пауза |
| `flush` | отправить накопленное (для `sendinput`) |

Имена клавиш: `esc`, `enter`, `space`, `tab`, `backspace`, `shift`, `ctrl`, `alt`,
`up/down/left/right`, `home/end`, `pageup/pagedown`, `insert/delete`, `f1..f12`,
`minus/plus`, буквы, цифры, `kp_up`, `kp_plus` и т.п. Понимаются и имена X11
(`Escape`, `Return`, `KP_Add`). Аккорд: `key ctrl+s` или `chord ctrl s`.

Примеры:

```bash
# New game -> окно сложности -> Normal
python3 eitools/eidrive.py original input click 425 195 ';' wait 1200 ';' click 297 482

# идти по клику; Shift+клик — бегом
python3 eitools/eidrive.py demo input click 440 420 ';' wait 4000
python3 eitools/eidrive.py demo input keydown shift ';' click 450 200 ';' keyup shift

# повернуть камеру правой кнопкой и приблизить колесом
python3 eitools/eidrive.py original input drag 400 300 520 300 right 10 ';' wheel up 2

# выбрать заклинание в слоте 1; у оригинала повторное нажатие отменяет выбор
python3 eitools/eidrive.py original input key 1 ';' wait 500 ';' key 1

# сохранить/загрузить быстрое сохранение
python3 eitools/eidrive.py demo input key F5 ';' wait 1500
python3 eitools/eidrive.py demo input key F8 ';' wait 3000
```

Общие клавиши, которые одинаково работают в обоих (удобно для сравнения):
клик/Shift+клик, `Space` (пауза), стрелки и правый drag (камера), `H` (обучение),
`Enter` (страницы обучения), `Esc` (меню), `F5`/`F8` (быстрое сохранение/загрузка),
`Q`/`E`/`R` (наборы оружия), `F9`..`F12` (виды камеры).

Различаются: `TAB` (оригинал — экран заданий, демо — имена объектов), `Z/X/C/W/M`
в демо перехвачены под демо-действия (заклинания, оружие, мана, карта), в
оригинале это ходьба/бег/крадучись/оружие/миникарта. Для сценариев используйте
общие клавиши или строки `original:`/`demo:`.

### 7.4. Снимки и видео

```bash
python3 eitools/eidrive.py original shot /tmp/now.png
python3 eitools/eidrive.py demo record /tmp/demo_run 10 10
# /tmp/demo_run/f00000.png ... и /tmp/demo_run.mp4
```

`shot` — `XGetImage` окна. `record` — тот же захват в PNG-кадры и сборка mp4 через
`gst-launch-1.0` (`multifilesrc -> pngdec -> x264enc -> mp4mux`).

### 7.5. Сценарии

Сценарий — текстовый файл, одна команда в строке. Кроме команд ввода понимает:

| строка | смысл |
|---|---|
| `# ...` | комментарий (до конца строки) |
| `timeout MS` | потолок времени на весь прогон; обязателен в сценарии верхнего уровня, при превышении игры останавливаются и прогон считается неудачным |
| `sync` | ждать здесь другую игру сравнения: шаги обеих выполняются вместе |
| `original: ...` / `demo: ...` | выполнить строку только на одной цели |
| `include FILE` | выполнить другой сценарий; путь относительно папки текущего |
| `repeat N COMMAND...` | выполнить команду N раз, например `demo: repeat 55 key right`; `sync`, `include`, `timeout` и вложенный `repeat` внутри нельзя |
| `note TEXT` | записать текст в лог |
| `shot NAME` | сохранить `<out>/NAME.png` |
| `waitpixel X Y R,G,B [TOL] [TIMEOUT_MS]` | ждать цвет пикселя (TOL 40) |
| `waitregion X Y W H R,G,B [TOL] [FRACTION] [TIMEOUT_MS]` | ждать долю пикселей цвета в области |
| `waitstable SETTLE_MS TIMEOUT_MS [POLL_MS]` | ждать, пока картинка перестанет меняться |
| `expect_changed NAME [THRESHOLD]` | картинка должна отличаться от `<out>/NAME.png` |
| `expect_same NAME [THRESHOLD]` | картинка должна совпадать с `<out>/NAME.png` |
| `expect_closed [TIMEOUT_MS]` | окно игры должно закрыться за время (по умолчанию 5000): игра вышла сама |
| `reset_settings` | `compare` перед запуском сбрасывает настройки обеих игр (см. `scenarios/PLAN_FIX.md`, раздел 3) |

`waitpixel`/`waitregion` — надёжнее фиксированных пауз для меню и загрузки. В
`common_intro.txt` меню определяется по цвету камня `NEW GAME` в `(430,200)`, а
окно обучения — по цвету пергамента в области `250,150 300x250`.

Пример собственного сценария:

```text
# my_check.txt: начать в главном меню
waitpixel 430 200 128,123,92 45 90000
shot 01_menu

click 425 195
wait 1200
shot 02_difficulty
click 297 482

wait 3000
key Escape
wait 2000
waitregion 250 150 300 250 36,31,18 12 0.5 90000
demo: wait 16000
shot 03_tutorial
key Return
wait 600
key Return
wait 600
key Return
wait 1200
shot 04_zone

# мой шаг: открыть Options в игре через Esc-меню (оригинал)
original: key Escape
original: wait 1000
original: click 425 300
original: wait 1200
original: shot 05_options
```

Запуск на одной цели (без перезапуска; цель должна быть в нужном состоянии):

```bash
python3 eitools/eidrive.py demo run eitools/scenarios/smoke.txt --out /tmp/demo_smoke
```

`run` пишет кадры и `results.json` в `<out>` (по умолчанию
`eitools/out/<сценарий>/<цель>`) и возвращает код 1, если были ошибки.

### 7.6. Сравнение

```bash
python3 eitools/eidrive.py compare eitools/scenarios/smoke.txt
```

Порядок:

1. перезапускает оригинал (если не `--no-restart`), ждёт окно;
2. проигрывает сценарий на оригинале;
3. перезапускает демо, ждёт окно;
4. проигрывает тот же сценарий на демо;
5. собирает `eitools/out/<сценарий>/index.html`.

Ошибки одного шага не останавливают сценарий: они попадают в `run.log` и
`results.json`, а в отчёте появляется блок «Ошибки». Код возврата `compare` — 1,
если были ошибки хотя бы у одной цели.

`--out DIR` — другая папка отчёта (удобно хранить историю прогонов).
`--no-restart` — не перезапускать; обе цели должны быть в главном меню для
сценариев с `common_intro.txt`.

`compare` останавливает обе игры сразу после сценария, в том числе при ошибке
или `Ctrl+C`. `--keep` оставляет их открытыми для осмотра; `--finish-timeout
SECONDS` (по умолчанию 2) — сколько ждать вторую игру после того, как первая
закончила.

`--screen` — запустить обе игры на дисплее рабочего стола (`$DISPLAY`, или
`--display :0`) и поставить их рядом на одном мониторе: оригинал слева, демо
справа. Сценарий идёт на обеих играх одновременно; строки `sync` держат шаги
вместе, строки лога помечены `[original]`/`[demo]`.
На Wayland (Xwayland) XTest на `:0` не доставляет ввод, поэтому в этом режиме
работает backend `xsend`: `XWarpPointer` и синтетическое `MotionNotify` для
указателя, `XSendEvent` для кнопок и клавиш. Нажатые кнопки мыши идут в `state`
событий движения (`Button1Mask` и т.д.): без этого Wine не видит перетаскивания,
и ползунки оригинала не двигаются. Движение важно для Kivy: оно
сообщает позицию указателя, даже если окно демо перекрыто.

```bash
EIDRIVE_HOME=/home/veselin/work/worlds python3 remake/eitools/eidrive.py compare \
    remake/eitools/scenarios/plan1_newgame_tutorial.txt --screen
```

### 7.7. Переменные окружения

| переменная | по умолчанию | смысл |
|---|---|---|
| `EIDRIVE_HOME` | родительская папка `eitools` | папка с `EvilIslands1` и `remake` |
| `EIDRIVE_ORIGINAL_DISPLAY` | `:77` (или `EI_DISPLAY`) | дисплей оригинала |
| `EIDRIVE_DEMO_DISPLAY` | `:78` | дисплей демо |
| `EIDRIVE_ORIGINAL_BACKEND` | `xtest` | `xtest` или `sendinput` |
| `EIDRIVE_DEMO_BACKEND` | `xtest` | только `xtest` |
| `EIDRIVE_DATA_PATH` | `EvilIslands1/game_info/data` | данные для демо |
| `EIDRIVE_DEMO_AUDIODRIVER` | — | `dummy` — демо без звука |

### 7.8. Обёртка `eigame.sh`

```bash
bash eitools/eigame.sh start|stop|restart|status
bash eitools/eigame.sh input click 425 195 ';' wait 1000
bash eitools/eigame.sh shot /tmp/orig.png
bash eitools/eigame.sh record /tmp/rec 10 10
bash eitools/eigame.sh run eitools/scenarios/smoke.txt
bash eitools/eigame.sh xvfb|xvfb-stop
```

Это тонкая обёртка над `eidrive.py original`; `EI_DISPLAY` учитывается.

---

## 8. Что показали первые прогоны

Все четыре сценария завершились с `0 errors`. Наблюдения (это не приговор, а
различия, которые видно на кадрах):

1. **Options.**
   - Оригинал: сначала страница обучения, затем окно с 11 разделами
     (`Graphics, Sound, Sensitivity, Game, Selection, Controls, Camera, Objects,
     Spells, Combat, Windows`), тени объектов и деревьев `Off`, есть `Show blood`,
     `Show footprints`, `Show path`, `Show hit result` и ползунки
     `Brightness/Contrast/Gamma`.
   - Демо: страницы обучения нет, в списке видно 10 разделов (нет `Windows`), хотя
     заголовок говорит `Graphics (1/11)`; тени объектов/деревьев/персонажей `On`;
     части пунктов и ползунков нет, Gamma/Contrast/Brightness показаны числами 50.
   - Кадры: `eitools/out/options/original/06_options.png`,
     `eitools/out/options/demo/05_options_open.png`.
2. **TAB.** Оригинал открывает экран заданий (Quests screen) с картой зоны и
   обучением; демо включает `Object names: on`.
   Кадры: `eitools/out/quests/original/05_tab.png`,
   `eitools/out/quests/demo/05_tab.png`.
3. **Мана героя.** В первом прогоне демо показывало 500 маны, оригинал — 50.
   Теперь `DEMO_MANA = 50` в `client/demo.py`, демо показывает 50, как оригинал;
   стартовое сообщение о заклинании убрано, HUD округляет здоровье (`39 / 50`).
4. **Esc-меню.** У демо HUD остаётся видимым и не затемняется; у оригинала весь
   экран затемняется, HUD под меню скрыт.
   Кадры: `eitools/out/smoke/original/10_esc_menu.png`,
   `eitools/out/smoke/demo/10_esc_menu.png`.
5. **Камера.** Начальный вид в зоне1 теперь совпадает с оригиналом: `#view`
   входа — направление взгляда, камера ставится за целью (`Exit.camera_yaw` =
   `view + 180`). Ходьба по клику, стрелки, правый drag, пауза, F10 и обучение H
   отрабатывают одинаково.
6. **Подсказка демо.** Первые 15 секунд в зоне демо показывает `DEMO KEYS`; в
   `common_intro.txt` она пережидается (`demo: wait 16000`), чтобы кадры было
   проще сравнивать.

Проверки `expect_changed` в `smoke.txt`/`actions.txt` проходят (кадр после
ходьбы/загрузки отличается от опорного); порог `0.3` средней разницы.

---

## 9. Диагностика

| симптом | что смотреть |
|---|---|
| `no window 'Evil Islands' on display :77` | `eidrive.py original status`; лог `eitools/run/original.log`; `xvfb` жив? |
| `no window 'Evil Islands remake' ...` | лог `eitools/run/demo.log`; venv собран? |
| `the demo has no venv` | `cd remake && make venv` |
| `ModuleNotFoundError: PIL` | `sudo apt install python3-pil` или запускать драйвер `remake/venv/bin/python eitools/eidrive.py ...` |
| `record failed` / нет mp4 | нужен `gst-launch-1.0` с `x264enc`; PNG-кадры всё равно сохраняются |
| `eiinput failed` | `sendinput` требует `C:\eitools` в префиксе (см. 3.3) |
| клики/клавиши не доходят | проверьте, что окно на `:77`/`:78` и `status` его видит; для игры используйте `xtest` |
| `compare` долго ждёт | `waitpixel` ждёт меню до 90 с; смотрите `run.log` |
| `BrokenPipeError` при `| head` | не актуально: обрабатывается `SIGPIPE` |
| окно демо сменило заголовок | это нормально, драйвер ищет по префиксу |
| игра вручную запущена на `:0` | драйвер её не увидит; запускайте через `eidrive.py original start` |

Логи запуска: `eitools/run/original.log`, `eitools/run/demo.log`,
`eitools/run/xvfb-77.log`, `eitools/run/xvfb-78.log`.
Логи сценария: `eitools/out/<сценарий>/<цель>/run.log`.

---

## 10. Ограничения и предостережения

- Сценарии рассчитаны на окно 800x600 и на то, что игра не меняет разрешение.
  Координаты меню и кнопок проверены для этой копии; при другом разрешении
  правьте координаты.
- `waitpixel`/`waitregion` опираются на цвета конкретных экранов; при смене
  настроек освещения/скина их может понадобиться подстроить.
- `sendinput` не умеет колесо; абсолютные перемещения в игровом мире Wine с
  DirectInput могут не дойти. Для мира — `xtest`.
- На Wayland (Xwayland) XTest на дисплее рабочего стола не доставляет движение
  мыши и нажатия. Режим `--screen` поэтому использует backend `xsend`:
  `XWarpPointer` для указателя и `XSendEvent` для кнопок и клавиш.
- Звук виртуальным X-сервером не изолируется. Демо можно заглушить
  `EIDRIVE_DEMO_AUDIODRIVER=dummy`; у оригинала звук — в его настройках
  (`Software\Nival Interactive\EvilIslands\General Settings`, `Audio`).
- Сценарий `actions.txt` пишет быстрые сохранения в общую папку данных:
  `EvilIslands1/game_info/data/saves/saveq` (оригинал) и `saves/quick` (демо).
  Оригинал при запуске может обновить автосохранение `saves/savea` — это его
  обычное поведение.
- Процессы игры и демо всё равно расходуют CPU; после проверок делайте `stop`.
- Исходные `xclick.py`, `xgrab.py`, `xrec.py` оставлены как отдельные утилиты;
  `eidrive.py` их не требует (кроме `xrec.py` для `record`).

---

## 11. Чек-лист новой проверки

1. `python3 eitools/eidrive.py original status` и `demo status` — обе цели живы.
2. Если нужен чистый старт: `compare ...` (перезапустит сам) или
   `restart` каждой цели.
3. Новый сценарий кладите в `eitools/scenarios/`, начинайте с `common_intro.txt`,
   если проверка идёт после входа в зону.
4. Для меню/загрузки используйте `waitpixel`/`waitregion`, а не длинные `wait`.
5. Для действий героя — `click`/`Shift+click`/`key`; для камеры — стрелки,
   `drag ... right`, `wheel`.
6. Проверку результата оформляйте `expect_changed`/`expect_same` с опорным кадром.
7. Запускайте `compare` и смотрите `index.html`; расхождения фиксируйте в
   `verify.md` (раздел 8) со ссылками на кадры.
8. В конце — `stop` обеих целей.
