# P3: окно Options, все разделы

Сценарий: [plan3_options_groups.txt](plan3_options_groups.txt)
Статус: закрыт · исправлено 11 из 12 (D13 объяснено) · следующий шаг: —

## Назначение

Вид и состав окна Options (game.exe 650380): рамка, заголовок, кнопки 11 разделов, строки пунктов,
значения, ползунки, подсветка, всплывающие подсказки. Сценарий ничего не меняет и выходит кнопкой X.

## Шаги и кадры

| кадр | шаг |
|---|---|
| 01_menu | главное меню |
| 03_graphics | Options, раздел Graphics |
| 04_tooltip | курсор 2.5 с над первым пунктом |
| 05..14_group02..11 | клик по кнопкам разделов 2..11 (x 175, y 191..407 с шагом 24) |
| 15_menu_again | X, главное меню |
| (проверка) | EXIT закрывает игру |

## Отчёты о прогонах

### 2026-11-11 (виртуальные дисплеи)

- оригинал: 15 кадров, 0 ошибок, ~21 с; демо: 15 кадров, 0 ошибок, ~20 с; обе игры закрываются.
- Страницы обучения перед Options у оригинала в этот раз не было (см. D11).

### 2026-11-11, со сбросом настроек и без Enter

- оригинал: 14 кадров, 0 ошибок, ~19 с; демо: 14 кадров, 0 ошибок, ~19 с; обе игры закрываются.
- Страницы обучения перед Options нет и на чистых настройках: D11 не подтвердилось.
- Enter в окне Options у оригинала переключает выбранный пункт: в прошлом прогоне кадр 03 показывал
  Character shadows = Off из-за него. Enter убран, кадр 02 (дубль 03) убран.
- На чистых настройках значения раздела Graphics — умолчания game.exe: Character shadows On,
  Object shadows Off, Tree shadows Off, Show blood On, Show footprints On, Marker character
  selection Off, Show path On. D9 подтверждено.

### 2026-11-11, D9 и D10: умолчания и имена клавиш

- оригинал: 14 кадров, 0 ошибок, ~19 с; демо: 14 кадров, 0 ошибок, ~19 с; обе игры закрываются.
- D9: в `engine/options.py` умолчания приведены к game.exe — Object/Tree shadows Off,
  Marker character selection Off, Show hit result On. На кадре 03 демо: Tree shadows Off,
  Object shadows Off, Character shadows On. Marker и Show hit result в таблице демо появятся
  вместе с составом разделов (D8).
- D10: `eilib/keys.py` `display_name` пишет имена как окно game.exe — `KP_Begin`, `KP_Up`,
  `Prior`, `Next`, `space`, `Tab`; тест в `tests/test_keys.py`. На кадре 10 демо: `Tab` и `space`.
- Остальные различия раздела — вид окна (D1, D2, D4, D6), подсказки (D7), состав (D3, D5, D8).

### 2026-11-11, P3-T1 и P3-T2 в коде (без прогона)

- Разобраны таблицы game.exe: пункты 787430 и клавиши 787080 (раздел и строка у каждой записи),
  умолчания из реестра чистых настроек и 6521c0/652460. Раскладка строк из 651900.
- Демо: состав и порядок разделов как у game.exe, пустые строки, 13 строк без прокрутки,
  все 11 разделов кликаются, заголовок окна — имя раздела, значения столбцом с x=530.
- Прогона не было (правило плана: только код и тесты); D3, D5, D8 закрываются после прогона.

### 2026-11-11, P3-T3 в коде (без прогона)

- Вид окна по 650380, 651310, 651900, 65b870: табличка OPTIONS (части модели меню Esc), рамка из
  атласа, полупрозрачный фон, плашки разделов, оранжевая полоса выбора, ползунки из `Scrollbar`.
- Прогона не было; на кадре нужно проверить направление повернутых кусков (стрелки ползунка,
  боковые стороны рамки) и место таблички.

### 2026-11-11, прогон после P3-T1..T3

- оригинал: 14 кадров, 0 ошибок; демо: 14 кадров, 0 ошибок; обе игры закрываются.
- Первый прогон: состав, порядок, пустые строки, плашки, рамка, полоса выбора и ползунки совпали
  с оригиналом; повернутые куски стоят верно (стрелки наружу). Табличка и мир за окном у демо были
  темнее: поверх фильтра паузы лежал чёрный слой 0.72 на весь экран. Слой убран, табличка теперь
  яркая (как у оригинала), отношение яркости мира в окне к меню 0.41 против 0.36 у оригинала;
  сам мир у демо светлее и днём, и ночью — это свет сцены (P26).
- Ползунки Sensitivity на чистых настройках стоят как у оригинала (25, 50, 5, 2, 2).
- Найдено: имена клавиш KP_Add и KP_Subtract (у демо были KP_Plus, KP_Minus) — исправлено в
  `eilib/keys.py` без повторного прогона (проверено тестом); Show tutorial у оригинала Off — это
  eidrive ставит ShowTutorial=0 перед запуском (`_set_show_tutorial`), умолчание game.exe — 1
  (458870), как у демо (D13).
- После закрытия окна табличка убирается (кадр 15).

### 2026-11-11, P3-T4: подсказки

- оригинал: 14 кадров, 0 ошибок; демо: 14 кадров, 0 ошибок.
- Подсказки пунктов, разделов и клавиш появляются через задержку Pop-up tips delay (0.5 с) в том же
  месте и того же вида; рамки совпадают с оригиналом до пикселя (ширина ±1 по метрикам шрифта).
- Найдено и исправлено: положение мыши у демо было на пиксель ниже (Kivy пишет y мыши как
  `высота − 1 − y`, а касаний — как `высота − y`); `ui.mouse_window_pos` для наведения и курсора.

### 2026-11-11, D12

- оригинал: 14 кадров, 0 ошибок; демо: 14 кадров, 0 ошибок.
- Текст строк, значений и заголовка поднят на 1 пиксель (`TEXT_RAISE` в `client/options.py`):
  верх и низ букв совпадают с оригиналом на кадрах 03, 07, 13; подписи разделов и подсказки
  совпадали и раньше.
- Прозрачность фона окна: доля картинки за окном у демо 0.15 против 0.09–0.13, снаружи окна
  0.41 против 0.36 — та же разница, что у фильтра паузы, а сам мир у демо светлее и без окна
  (свет сцены и час, P26). Фон окна (альфа 0xa0) взят из game.exe, его не меняю.

## Различия

Кадры: `remake/eitools/out/plan3_options_groups/`.

| № | важн. | оригинал | демо | кадр | статус |
|---|---|---|---|---|---|
| D1 | B | над окном каменная табличка «OPTIONS»; заголовок окна — имя раздела («Graphics») | табличек нет; заголовок «Options» и строка «Graphics (1/11)» | все | исправлено 2026-11-11 |
| D2 | B | кнопки разделов — каменные плашки из атласа | строки текста, выбранная — серая полоса | все | исправлено 2026-11-11 |
| D3 | A | разделов 11, 6-й Controls, 7-й Camera, 11-й Windows | видно 10, Camera и Controls переставлены, Windows не попасть кликом (счётчик пишет /11) | 09, 10, 14 | исправлено 2026-11-11 |
| D4 | B | выбранный пункт — оранжево-коричневая полоса, значения столбцом у x≈530, слева | серая полоса, значения прижаты к правому краю (x≈750) | все | исправлено 2026-11-11 |
| D5 | B | между группами пунктов пустые строки | пунктов подряд, без пустых строк; справа полоса прокрутки | все | исправлено 2026-11-11 |
| D6 | B | числа — ползунки с ручкой и стрелками | числа текстом | 03, 05, 06 | исправлено 2026-11-11 |
| D7 | B | подсказка при наведении (жёлтая рамка, имя и пояснение), и у пунктов, и у кнопок разделов | подсказок нет; пояснение выбранного пункта внизу окна | 04..14 | исправлено 2026-11-11 |
| D8 | A | состав разделов — см. таблицу ниже | другой состав, порядок и подписи | 03..14 | исправлено 2026-11-11; лишняя строка демо Movies in a window в Sound — объяснено (своя настройка ремейка) |
| D9 | B | умолчания: Object/Tree shadows Off, Marker character selection Off, Show hit result On | Object/Tree shadows Off, Character shadows On; Marker и Show hit result появятся с составом D8 | 03, 07, 08 | исправлено 2026-11-11 |
| D10 | C | имена клавиш: KP_Begin, KP_Up, Prior, Next, space, Tab, KP_Add, KP_Subtract | так же | 07, 10, 13 | исправлено 2026-11-11 |
| D11 | — | при первом открытии показывается страница обучения Options (так записано в verify.md) | не показывается | 03 | не подтвердилось: страницы нет, потому что eidrive ставит оригиналу ShowTutorial=0; при ShowTutorial=1 game.exe показывает её при первом открытии (5f86f0) |

| D12 | C | фон окна пропускает ~9–13% картинки; текст строк на 1 пиксель выше | ~15%; текст на 1 пиксель ниже | 03..14 | исправлено 2026-11-11 (текст); прозрачность — объяснено, см. P26 |
| D13 | — | Show tutorial = Off | On | 14 | объяснено: eidrive ставит оригиналу ShowTutorial=0, умолчание game.exe — 1 |

### Состав разделов

Таблица ниже — что было до P3-T2. Теперь демо берёт состав из таблиц game.exe (`engine/options.py`
`GROUPS`); последний пункт Sound у оригинала — Reverse stereo sound, у демо после него ещё
Movies in a window.

`—` — пустая строка-разделитель у оригинала.

| раздел | оригинал | демо |
|---|---|---|
| Graphics | Character shadows, Object shadows, Tree shadows, —, Show blood, Show footprints, Marker character selection, Show path, Show hit result, —, Brightness, Contrast, Gamma | Gamma, Contrast, Brightness, Tree shadows, Object shadows, Character shadows, Inventory autofilters |
| Sound | Sound effects volume, Music volume, Speech volume, —, ещё один пункт со значением Off (подпись закрыта подсказкой) | Sound effects volume, Music volume, Speech volume, Reverse stereo sound, Movies in a window |
| Sensitivity | Keyboard, Mouse, Scrolling, Selection frame sensitivity и ещё один ползунок, —, Inverse camera left/right rotation, Inverse camera up/down movement | Keyboard sensitivity, Mouse sensitivity |
| Game | Difficulty, —, Accelerated speed, Normal speed, Pause on/off, View quests, —, Quick save, Quick load, —, Autosave, Inventory autofilters | Difficulty, Autosave, Show tutorial, Show hit result |
| Selection | Select character 1..3, —, Select all characters | Marker character selection, Selection frame sensitivity, Scrolling sensitivity |
| 6 | Controls: Follow character, Use/Steal, Aggressive/Defensive mode, —, Run, Walk, Sneak, Crawl | Camera: Inverse camera ×2, Pop-up tips delay, Camera: normal, Camera1..4, Camera: track, zoom in... |
| 7 | Camera: Tracking camera, Camera forward/back/left/right, Zoom in/out, —, (N), (F9), Camera position 2..4 | Controls: Obj, Run, Walk, Sneak, Crawl, Pause, Swarm, Use science, Follow, Tutorial script |
| Objects | Select weapon 1..4, —, Use quickitem 1..4 | Weapon1..4, Item1..4, Quicksave, Quickload |
| Spells | Cast spell 1..8 | Spell1..8 |
| Combat | Aim at head, body, right leg, left leg, right arm, left arm | Called shot: head, body, left hand, right hand, left leg, right leg |
| Windows | Text window: messages, collected items, —, Information window: general view, hit locations, attributes, spell effects, —, Show Mini-map, —, Tutorial, Show tutorial | нет |

## План исправления

### P3-T1: разобрать окно Options в game.exe — сделано 2026-11-11
- 650380 и соседние функции: координаты панели, таблички, кнопок разделов, строк, столбца значений,
  ползунков; спрайты (атлас, куски, ориентация — см. PLAN_FIX.md раздел 5, п. 2); цвета полосы выбора;
- таблица пунктов: записи game.exe (ключ, умолчание, максимум, вид, номер в разделе, номер пункта —
  см. docstring `engine/options.py`); выписать её целиком вместе с пустыми строками и подписями
  из `texts.res`;
- страница обучения при первом открытии и где хранится флаг.

Итог разбора:
- пункты 787430, записи (id, ключ, вид, максимум, раздел, строка): вид 0 — ползунок 0..максимум,
  вид 1 — переключатель из двух значений (Difficulty тоже: Novice/Normal); 26 записей, id 0..0x19;
- клавиши 787080, записи (id, действие, раздел, строка): 59 действий, id 1..0x3b, среди них
  accel (Accelerated speed) и decel (Normal speed) в разделе Game;
- 651900 строит строки: ползунок (5bbf0) в (530, 163 + 24n)-(680, 173 + 24n), значение
  переключателя (5bc90) в (530, 160 + 24n)-(690, 176 + 24n); подписи `string option_<ключ>`,
  у клавиш (6515c0) — `string action_<действие>`;
- 6521c0 (в окно) и 652460 (из окна): Keyboard/Mouse = CameraDefaultScrollPower* / 0.02,
  Scrolling = CameraBorderScrollArea, Selection frame = CameraFrameSelectionSensetiveArea / 10,
  Pop-up tips delay = ToolTip Time × 4, Brightness/Contrast/Gamma = (x + 1) / 0.02,
  Inverse camera = CameraFlipX/YAxis;
- умолчания чистого реестра: Keyboard 25, Mouse 50, Scrolling 5, Selection frame 2, Pop-up 2,
  Inverse camera Off ×2, Show blood/footprints/path On;
- страница обучения при открытии: на чистых настройках её нет (D11).

### P3-T2: таблица пунктов демо по game.exe (D3, D5, D8, D9, D10) — сделано 2026-11-11
- D9: умолчания приведены к game.exe (Object/Tree shadows Off, Marker Off, Show hit result On);
- D10: `keys.display_name` даёт имена клавиш окна game.exe (`KP_Begin`, `Prior`, `space`, `Tab`);
- осталось D3, D5, D8: таблица записей game.exe (787434, 26 записей) разобрана:
  поля — ключ, минимум, максимум, вид, номер группы (0 video .. 10 windows) и номер пункта.
  Пункты video: shadow_units 0, shadow_buildings 1, shadow_flora 2, пусто 3, marks 4,
  footprints 5, select_type 6, show_path 7, show_flying_hp 8, пусто 9, brightness 10,
  contrast 11, gamma 12; sound: 3 громкости, пусто, reverse_stereo; sens: power_kbd, power_mouse,
  scroll_border, rubber_select, tooltip_time, пусто, camera_reverse_x/y; game: difficulty, ...,
  autosave 11, switch_filters 12; windows: show_tutorial 11. Горячие клавиши идут в разделы
  game/selection/controls/camera/objects/spells/combat/windows, а не отдельными группами.
  Демо: `GROUPS` с `None`-разделителями, `Dialog` пропускает пустые строки, `client/options.py`
  рисует пустые строки; `tests/test_options.py` — состав и порядок.
- `engine/options.py`: `GROUPS`, `OPTIONS` и подписи из таблицы game.exe;
- `tests/test_options.py`: состав и порядок разделов, умолчания.

Сделано 2026-11-11:
- `engine/options.py`: `GROUPS` — 11 разделов game.exe, строки по номерам, `None` — пустая строка;
  новые пункты marks, footprints, show_path (On); scroll_border и rubber_select — ползунки 0..10
  (5 и 2), power_kbd 25; camera_reverse_x/y Off; подписи клавиш из `string action_<действие>`;
  `Dialog` не выбирает пустые строки (стрелки и клик их пропускают); 13 строк с y=156 через 24,
  кнопки разделов x 110..240, столбец значений x=530;
- файл настроек получил `version` 2: у старого файла сбрасываются пункты, сменившие смысл
  (camera_reverse_x/y, scroll_border, rubber_select, power_kbd, power_mouse);
- `client/main.py`: Inverse camera On = поворот камеры мышью наоборот (раньше было наоборот: 1 = обычный);
- `eilib/keys.py`: действия accel и decel в `ACTIONS` (59, как в 787080);
- `client/options.py`: заголовок — имя раздела (без «Options» и «(1/11)»), пустые строки, значения
  с x=530; пояснение внизу окна убрано (места нет, у оригинала его нет — подсказки в P3-T4);
- тесты: `test_groups`, `test_sensitivity_defaults`, `test_empty_rows_are_skipped`,
  `test_older_file_drops_the_changed_options`, подписи `texts.res`; `make test` — 453 OK;
- не подключены к сцене: Show blood, Show footprints, Show path, Marker character selection,
  чувствительности и Scrolling — это P9.

### P3-T3: вид окна (D1, D2, D4, D6) — сделано 2026-11-11
- `client/options.py`: табличка OPTIONS, плашки разделов, полоса выбора, столбец значений;
- ползунок: виджет в `client/ui.py` (дорожка, ручка, стрелки; перетаскивание и клики по стрелкам),
  нужен и для P2, Sound и Sensitivity.

Разбор game.exe:
- 650380: атлас saveload; рамка 5e7d60 (95,95)-(705,505) шириной 5 из полосы атласа
  (2,0xfe)-(0xb4,0xf9); фон (100,100)-(700,500) чёрный с альфой 0xa0; регионы разделов
  (100, 156+24i)-(250, 180+24i), плашки (110, 158+24i)-(240, 177+24i) из (0x7a,0x6c)-(0xfc,0x7f);
  полоса выбора цвета 0xc8a06800; табличка — модель unmoco1 (как колонна меню Esc) с частями
  `options` и `optionslabel` в точке экрана (400, 480), глубина 4.8; 6508e0 выдвигает её снизу
  (700 -> 480), когда окно открыто из игры;
- 651310: заголовок — имя раздела цветом (228,215,167); выбранный раздел — текст (228,215,167) и
  плашка полной яркости, прочие — (160,160,160) и плашка 0.5; полоса выбора по центру
  (471, 168+24n); подписи и значения (228,215,167), подпись клавиши в ожидании нажатия — красная;
- 65b870 (Ghidra её не разобрала, читал ассемблер) и 65b6f0: ползунок в (530, 163+24n)-(680, 173+24n),
  текстура `Scrollbar` 64x64, все куски повернуты на четверть вокруг z: стрелки 12x10 по краям,
  дорожка 142x6, ручка 28x10 с центром 556 + int(98 * value / max).

Сделано:
- `engine/options.py`: константы раскладки, `slider_rects`, `slider_value`; `Dialog.click_option` —
  клик по ползунку (стрелки ±1, дорожка ставит значение) и по переключателю срабатывает сразу,
  клавиша ждёт нажатия после клика по значению выбранной строки;
- `engine/figure.py`: `FigureModel(parts=...)` — показ только части деталей (CInterfaceModel::ShowPart);
- `client/ui.py`: `sprite_item(turn=True)` — кусок атласа, повернутый на четверть;
- `client/options.py`: табличка, рамка, фон, плашки, полоса, ползунки;
- тесты: `test_slider_geometry`, `test_slider_clicks`, `test_plaque_parts`; `make test` — 456 OK;
- после прогона: убран чёрный слой 0.72 на весь экран (затемнял табличку и мир сверх фильтра паузы);
- не сделано: перетаскивание ручки мышью, выдвижение таблички при открытии из игры (меню Esc
  пока не открывает Options).

### P3-T4: всплывающие подсказки (D7) — сделано 2026-11-11
- общий механизм подсказок в `client/ui.py`: задержка (`tooltip_time`), рамка, имя и пояснение
  из `texts.res`; используется также HUD (P13);
- убрать пояснение внизу окна, если у оригинала его нет.

Сделано:
- `engine/tooltip.py`: задержка (ToolTip Time = ползунок × 0.25 с, 0 выключает), место по 65e540
  (сдвиг от курсора 24, 24; у правого края влево, у нижнего — низ к курсору), вид по кадрам:
  рамка 1 пиксель 0xffb231, фон 0x291c00, заголовок цветом рамки, строки 0xe4d7a7 через 15 пикселей;
- `engine/options.py`: `Dialog.tip_at` — `tip option_group_<раздел>`, `tip option_<пункт>`,
  `tip action_<клавиша>` (651d30);
- `client/ui.py`: `tooltip_items`, `mouse_window_pos`; `client/options.py`: `on_mouse`, `tick`;
  `client/menu.py` передаёт окну движение мыши и такты;
- пояснения внизу окна нет с P3-T2;
- тесты: `tests/test_tooltip.py`; подсказки HUD пока свои (`client/hud.py`), перевести на
  `engine/tooltip.py` — P13.

### P3-T5: страница обучения Options при первом открытии (D11) — снято
На чистых настройках оригинал её не показывает; задача не нужна.

Проверка всех задач: P3 — кадры 03..14 должны совпадать по раскладке; P2 — ползунок;
сценарий тогда сможет тянуть ручку одинаково у обоих (убрать `demo:` с клавишами).

## История сценария

- 2026-11-11: создан на основе `options.txt` (там только открыть и закрыть): все разделы,
  подсказка, `expect_closed`.
- 2026-11-11: `reset_settings`; убран Enter оригиналу (переключал пункт) и кадр 02_options_open.
