# Реестр функций game.exe и их реализации в ремейке

Каждая функция декомпилированного `build/ghidra/game_exe.c` (Ghidra, первые «Проклятые Земли») сопоставлена с файлом и
функцией/классом в `remake/src`, которые её реализуют. Полный реестр (все 24916 функций) лежит в
[`func_registry.json`](func_registry.json), этот документ — его сводка.

Оба файла пишет генератор, руками их не правят:

```bash
make registry        # PYTHONPATH=src venv/bin/python -m tools.func_registry
```

## Как находится соответствие

В .py ремейка в комментариях и докстрингах стоят ссылки на код игры: адрес (`45ea20`, `0x50aaf0`,
`FUN_00461cb0`) или имя из `re/names.json` (`CUnitPrototype::Read`). Генератор читает каждый .py через
`tokenize` (только комментарии и строки, код не смотрится) и `ast`, и отдаёт ссылку той функции или классу,
внутри которых она стоит. Блок комментариев прямо над `def`, `class` или константой модуля относится к ним.
Адрес внутри тела функции (не её начало) относится к содержащей функции (`"inside": true`). Число без
шестнадцатеричных букв считается адресом, только если это точное начало функции.

Статусы:

| Статус | Значение |
|---|---|
| `implemented` | есть ссылка из функции или класса ремейка: реализация найдена |
| `partial` | ссылки только вида `5641f0 (partly)`: ремейк делает лишь часть функции |
| `module` | файл найден, но ссылка только в докстринге модуля или у константы (таблица, число из функции) |
| `builtin` | `re/func_decisions.json`: контейнер или доступ к полю, работу делает встроенный тип Python |
| `later` | `re/func_decisions.json`: нужно, но не для demo |
| `skip` | `re/func_decisions.json`: ремейку не нужно |
| `indirect` | ссылки нет, но функцию прямо вызывает реализованная (`covered_by`): возможно, она вошла в ту же реализацию |
| `missing` | игровой код, ссылок нет: **.py файл не найден** |
| `library` | CRT, MFC, импорты DLL (адреса от `6e60a2`) — реализовывать не нужно |
| `compiler` | обработчики исключений компилятора (`Unwind@`, `Catch@`) — не нужно |

Ограничения:

- соответствие найдено только там, где ремейк сам назвал адрес; код, написанный по поведению игры без ссылки,
  здесь числится `missing`;
- `indirect` — подсказка, а не проверка: вызванная функция могла и не понадобиться ремейку;
- `guess_class` у безымянной `FUN_` — класс её соседей по адресу (функции одного .cpp лежат подряд);
- имя (из `re/names.json` или Ghidra) есть только у 1533 игровых функций, остальные — `FUN_xxxxxxxx`.

## Итог

| Статус | Функций | Доля игрового кода по байтам |
|---|---:|---:|
| `implemented` | 450 | 11.8% |
| `partial` | 24 | 1.2% |
| `module` | 180 | 4.4% |
| `builtin` | 1414 | 2.0% |
| `indirect` | 1360 | 13.3% |
| `missing` | 7035 | 67.2% |
| `later` | 0 |  |
| `skip` | 92 | 0.1% |
| `library` | 1322 |  |
| `compiler` | 13039 |  |
| всего | 24916 | |

Игровых функций (без `library` и `compiler`): 10555. Python-файлов в `remake/src`: 119, из них ссылаются на
game.exe 92.

## Трасса оригинала (`re/coverage`)

`tools/func_coverage.py` запускает game.exe под отладчиком `re/eitrace` (int3 на первом байте каждой игровой
функции) и проходит сценарий `eitools/scenarios/<имя>.txt`. Сценарий делится на фазы; в каждой фазе все
функции заново ставятся на int3, а сработавшие — ещё раз в конце каждого окна в 2 с. Функция
**периодическая** в фазе, если работала хотя бы в половине окон фазы (фаза от 5 окон, 10 с): это код
каждого тика.
В JSON: `coverage` — {фаза: окон}, `periodic` — фазы, где функция периодическая.

| Сценарий | Фаза | Секунд | Функций | Периодических |
|---|---|---:|---:|---:|
| coverage | boot | 1.1 | 1242 | 0 |
| coverage | menu | 6.0 | 2150 | 0 |
| coverage | menu_options | 4.3 | 977 | 0 |
| coverage | menu_load | 4.3 | 975 | 0 |
| coverage | menu_idle | 6.0 | 679 | 0 |
| coverage | newgame | 7.5 | 2895 | 0 |
| coverage | tutorial | 7.3 | 1027 | 0 |
| coverage | zone_idle | 20.1 | 1005 | 669 |
| coverage | zone_walk | 25.0 | 1353 | 259 |
| coverage | zone_windows | 13.2 | 1511 | 729 |
| coverage | zone_actions | 6.1 | 1073 | 0 |
| coverage | quicksave | 4.0 | 1201 | 0 |
| coverage | quickload | 8.2 | 2531 | 0 |
| coverage | escmenu | 3.4 | 1051 | 0 |
| coverage | menu_back | 3.4 | 2647 | 0 |

| Статус | Сработали в трассе | Из них периодические |
|---|---:|---:|
| `implemented` | 272 | 91 |
| `partial` | 23 | 15 |
| `module` | 115 | 26 |
| `builtin` | 446 | 71 |
| `indirect` | 1006 | 264 |
| `missing` | 2907 | 295 |
| `skip` | 26 | 3 |
| всего | 4795 | 765 |

Игровых функций, которые в трассе ни разу не сработали: 5760.

### Сработали, но не реализованы, по подсистемам

| Подсистема | Уровень | missing | indirect | периодические | Строк C |
|---|---|---:|---:|---:|---:|
| world | game | 612 | 190 | 148 | 42586 |
| data | data | 337 | 104 | 10 | 26883 |
| ui | game | 250 | 112 | 49 | 17824 |
| units | game | 192 | 107 | 55 | 14145 |
| render | backend | 225 | 30 | 55 | 17949 |
| figures | game | 202 | 41 | 39 | 15233 |
| player | game | 163 | 62 | 13 | 12742 |
| magic | game | 160 | 57 | 18 | 10437 |
| script | game | 132 | 60 | 35 | 10398 |
| pathing | game | 135 | 56 | 37 | 10502 |
| runtime | backend | 133 | 36 | 32 | 8477 |
| sound | game | 95 | 33 | 22 | 7319 |
| items | game | 98 | 28 | 3 | 7520 |
| ai | game | 52 | 25 | 5 | 3852 |
| saveload | game | 49 | 26 | 20 | 4155 |
| platform | backend | 39 | 8 | 2 | 1372 |
| camera | game | 26 | 12 | 5 | 1837 |
| math | game | 2 | 14 | 10 | 442 |
| video | backend | 2 | 3 | 0 | 459 |
| input | backend | 3 | 1 | 0 | 151 |
| net | backend | 0 | 1 | 1 | 64 |

## Покрытие по подсистемам

Подсистема (`domain`) угадана `tools/func_domains.py`: по классу, строкам, вызовам DLL и .py реализации,
безымянным функциям она передаётся от соседей по адресу и по вызовам (`domain_source`). Уровень (`tier`):
`backend` заменён Python/Kivy/SDL и не переносится, `data` — форматы файлов (eilib), `game` — поведение игры.

Колонка «решено» — `builtin`, `skip` и `later`.

| Подсистема | Уровень | Всего | implemented | partial | module | решено | indirect | missing | Строк C в missing |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| world | game | 1874 | 43 | 4 | 29 | 349 | 251 | 1198 | 57643 |
| ui | game | 1378 | 62 | 4 | 22 | 163 | 166 | 961 | 51136 |
| units | game | 901 | 72 | 5 | 29 | 138 | 156 | 501 | 28357 |
| magic | game | 750 | 95 | 0 | 36 | 105 | 66 | 448 | 23986 |
| figures | game | 572 | 18 | 0 | 2 | 84 | 48 | 420 | 25365 |
| player | game | 533 | 18 | 0 | 13 | 56 | 88 | 358 | 17145 |
| script | game | 398 | 22 | 0 | 8 | 60 | 112 | 196 | 9921 |
| saveload | game | 335 | 9 | 3 | 4 | 41 | 29 | 249 | 11457 |
| pathing | game | 325 | 15 | 0 | 14 | 30 | 67 | 199 | 9980 |
| items | game | 300 | 12 | 0 | 5 | 50 | 41 | 192 | 9356 |
| sound | game | 290 | 15 | 0 | 3 | 36 | 45 | 191 | 8958 |
| ai | game | 257 | 50 | 7 | 4 | 21 | 57 | 118 | 5067 |
| camera | game | 97 | 4 | 0 | 5 | 8 | 21 | 59 | 2704 |
| math | game | 37 | 0 | 0 | 2 | 3 | 14 | 18 | 1344 |
| data | data | 891 | 14 | 1 | 4 | 100 | 111 | 661 | 36028 |
| runtime | backend | 874 | 0 | 0 | 0 | 182 | 40 | 652 | 30948 |
| render | backend | 615 | 0 | 0 | 0 | 63 | 33 | 519 | 31420 |
| platform | backend | 76 | 0 | 0 | 0 | 7 | 8 | 61 | 2382 |
| video | backend | 29 | 1 | 0 | 0 | 5 | 5 | 18 | 690 |
| net | backend | 17 | 0 | 0 | 0 | 4 | 1 | 12 | 721 |
| input | backend | 6 | 0 | 0 | 0 | 1 | 1 | 4 | 93 |

## Покрытие по классам

Классы из `re/names.json`; безымянные функции посчитаны по `guess_class`, остальные — в строке «без класса».

| Класс | Всего | implemented | partial | module | решено | indirect | missing |
|---|---:|---:|---:|---:|---:|---:|---:|
| CWorld | 314 | 7 | 0 | 2 | 55 | 31 | 219 |
| CUnitServer | 220 | 21 | 2 | 9 | 27 | 58 | 103 |
| CWorldServer | 100 | 8 | 0 | 6 | 15 | 30 | 41 |
| ServerMap | 83 | 13 | 0 | 9 | 2 | 29 | 30 |
| CUnitClientSpecific | 76 | 10 | 2 | 4 | 19 | 6 | 35 |
| CTerrain | 69 | 4 | 1 | 5 | 7 | 15 | 37 |
| CUnit | 66 | 11 | 0 | 9 | 3 | 21 | 22 |
| CUnitLogic | 65 | 24 | 2 | 0 | 0 | 20 | 19 |
| CScriptWorld | 63 | 8 | 0 | 7 | 0 | 16 | 32 |
| CSpell | 63 | 4 | 0 | 0 | 7 | 4 | 48 |
| CSpellManager | 63 | 6 | 0 | 1 | 7 | 11 | 38 |
| CParty | 61 | 5 | 0 | 5 | 13 | 9 | 29 |
| CObject3DClientSpecific | 41 | 0 | 1 | 1 | 4 | 7 | 28 |
| CUnitStats | 40 | 22 | 0 | 2 | 0 | 2 | 14 |
| CPlayer | 39 | 5 | 0 | 3 | 2 | 15 | 14 |
| CCamera | 34 | 2 | 0 | 2 | 4 | 8 | 18 |
| PathWindow | 30 | 2 | 0 | 1 | 0 | 6 | 21 |
| CItemWeapon | 29 | 4 | 0 | 0 | 3 | 7 | 15 |
| CTextureManager | 29 | 2 | 0 | 0 | 3 | 8 | 16 |
| CParticle | 27 | 6 | 0 | 2 | 3 | 7 | 9 |
| CPlayerClient | 26 | 0 | 0 | 0 | 11 | 1 | 14 |
| CFigureManager | 24 | 5 | 0 | 0 | 4 | 2 | 13 |
| CLeverObject | 24 | 4 | 0 | 2 | 5 | 0 | 13 |
| CWorldClient | 24 | 1 | 1 | 0 | 9 | 3 | 10 |
| CMapObjectClientSpecific | 21 | 4 | 0 | 0 | 3 | 0 | 14 |
| CObject3DServer | 21 | 0 | 0 | 0 | 18 | 0 | 3 |
| CInterfaceWindow | 20 | 3 | 0 | 0 | 0 | 14 | 3 |
| CMagicTrapObject | 20 | 3 | 0 | 4 | 2 | 2 | 9 |
| CMusicSystem | 19 | 7 | 0 | 1 | 2 | 2 | 7 |
| CEffectSpell | 18 | 7 | 0 | 1 | 0 | 3 | 7 |
| CFigure_1 | 18 | 0 | 0 | 0 | 2 | 1 | 15 |
| CInventory | 18 | 0 | 0 | 0 | 2 | 4 | 12 |
| CItemArmor | 18 | 0 | 0 | 2 | 3 | 3 | 10 |
| CMapObjectServer | 18 | 1 | 0 | 0 | 6 | 0 | 11 |
| CEffectArrow | 17 | 2 | 0 | 0 | 3 | 0 | 12 |
| CSoundManager | 16 | 0 | 0 | 0 | 3 | 1 | 12 |
| CEffectSound | 15 | 0 | 0 | 0 | 2 | 0 | 13 |
| CLightning | 15 | 5 | 0 | 0 | 3 | 2 | 5 |
| CEffectShowVisibility | 14 | 0 | 0 | 0 | 3 | 0 | 11 |
| CEffectSoundSeq | 14 | 0 | 0 | 0 | 2 | 0 | 12 |
| CLight | 14 | 0 | 0 | 1 | 5 | 1 | 7 |
| CTorchObject | 14 | 0 | 0 | 0 | 4 | 0 | 10 |
| CTorchObjectSpecific | 14 | 0 | 0 | 0 | 8 | 0 | 6 |
| CEffectArrowFigure | 13 | 1 | 0 | 2 | 2 | 0 | 8 |
| CEffectMoshka | 13 | 0 | 0 | 0 | 5 | 0 | 8 |
| CEffectParticle | 13 | 1 | 0 | 1 | 2 | 0 | 9 |
| CEffectTornado | 13 | 0 | 0 | 0 | 2 | 1 | 10 |
| CInventoryStrip | 13 | 8 | 0 | 1 | 0 | 2 | 2 |
| CItemLoot | 13 | 0 | 0 | 1 | 3 | 0 | 9 |
| CItemQuest | 13 | 0 | 0 | 0 | 3 | 0 | 10 |
| CAnimationTrack | 12 | 0 | 0 | 2 | 2 | 0 | 8 |
| CEffectAmbientSound | 12 | 0 | 0 | 0 | 3 | 0 | 9 |
| CEffectLightning | 12 | 1 | 0 | 3 | 2 | 0 | 6 |
| CItemManager | 12 | 2 | 0 | 1 | 3 | 0 | 6 |
| CTrapObject | 12 | 0 | 0 | 0 | 4 | 0 | 8 |
| SpellEffect | 12 | 10 | 0 | 1 | 0 | 0 | 1 |
| CEffectPointLight | 11 | 0 | 0 | 1 | 2 | 0 | 8 |
| CFileRegistry | 11 | 0 | 0 | 1 | 0 | 6 | 4 |
| CSpellConstructor | 11 | 3 | 0 | 0 | 2 | 1 | 5 |
| CInterfaceElement | 10 | 6 | 0 | 0 | 2 | 2 | 0 |
| CMapObjectManager | 10 | 0 | 0 | 0 | 3 | 0 | 7 |
| CMaterial | 10 | 1 | 0 | 0 | 3 | 0 | 6 |
| CTorchObjectClientSpecific | 10 | 0 | 0 | 0 | 3 | 0 | 7 |
| CVisibilityFog | 10 | 0 | 0 | 0 | 2 | 0 | 8 |
| CBehaviourPatrol | 9 | 1 | 0 | 2 | 1 | 2 | 3 |
| CDbReader | 9 | 1 | 0 | 0 | 0 | 5 | 3 |
| CFakeFigure | 9 | 0 | 0 | 0 | 8 | 0 | 1 |
| CFigureClientSpecific | 9 | 0 | 0 | 0 | 5 | 0 | 4 |
| CMainMenuWindow | 9 | 2 | 0 | 3 | 1 | 0 | 3 |
| CMaterialManager | 9 | 0 | 0 | 0 | 3 | 0 | 6 |
| CVisFogManager | 9 | 0 | 0 | 0 | 2 | 0 | 7 |
| Vec3 | 9 | 0 | 0 | 0 | 1 | 8 | 0 |
| CAnimationNode | 8 | 0 | 0 | 0 | 3 | 0 | 5 |
| CEffectManager | 8 | 0 | 0 | 0 | 2 | 0 | 6 |
| CFakeUnit | 8 | 0 | 0 | 0 | 2 | 0 | 6 |
| CFigure_2 | 8 | 0 | 0 | 0 | 2 | 0 | 6 |
| CItemQuick | 8 | 0 | 0 | 0 | 3 | 0 | 5 |
| CItemSpell | 8 | 0 | 0 | 0 | 3 | 0 | 5 |
| CLightningManager | 8 | 1 | 0 | 0 | 3 | 1 | 3 |
| CTexture | 8 | 0 | 0 | 0 | 3 | 0 | 5 |
| Quat | 8 | 0 | 0 | 1 | 0 | 4 | 3 |
| CAnimationSystem2 | 7 | 0 | 0 | 0 | 2 | 0 | 5 |
| CCursor | 7 | 1 | 0 | 0 | 0 | 2 | 4 |
| CHoover | 7 | 0 | 0 | 0 | 4 | 0 | 3 |
| CInfoWindow | 7 | 5 | 0 | 1 | 0 | 0 | 1 |
| CInterfaceModel | 7 | 2 | 0 | 1 | 0 | 3 | 1 |
| CItemConstructor | 7 | 0 | 0 | 0 | 2 | 0 | 5 |
| CLightManager | 7 | 0 | 0 | 0 | 2 | 0 | 5 |
| CPlayerManager | 7 | 0 | 0 | 0 | 2 | 0 | 5 |
| CWeather | 7 | 0 | 0 | 0 | 3 | 0 | 4 |
| CBridgeManager | 6 | 0 | 0 | 0 | 3 | 0 | 3 |
| CClientSpecificManager | 6 | 0 | 0 | 0 | 2 | 0 | 4 |
| CI3DFigure | 6 | 0 | 0 | 0 | 2 | 0 | 4 |
| CInterfaceRegion | 6 | 3 | 0 | 0 | 0 | 0 | 3 |
| CParticleManager | 6 | 0 | 0 | 0 | 2 | 0 | 4 |
| CTextManager | 6 | 0 | 0 | 0 | 2 | 0 | 4 |
| LightningBolt | 6 | 4 | 0 | 0 | 0 | 0 | 2 |
| CBehaviourAttack | 5 | 2 | 0 | 0 | 1 | 0 | 2 |
| CBuilding | 5 | 0 | 0 | 0 | 2 | 0 | 3 |
| CEffectWall | 5 | 0 | 0 | 0 | 2 | 0 | 3 |
| CMiniMapWindow | 5 | 4 | 0 | 0 | 0 | 1 | 0 |
| CServerSpecificManager | 5 | 0 | 0 | 0 | 2 | 0 | 3 |
| CStone | 5 | 0 | 0 | 0 | 2 | 0 | 3 |
| CStuffObject | 5 | 0 | 0 | 0 | 2 | 0 | 3 |
| CTree | 5 | 0 | 0 | 0 | 2 | 0 | 3 |
| Stream | 5 | 0 | 0 | 1 | 0 | 2 | 2 |
| CFigure | 4 | 1 | 0 | 0 | 1 | 0 | 2 |
| CSpellContainer | 4 | 0 | 0 | 0 | 4 | 0 | 0 |
| CSquad | 4 | 3 | 0 | 0 | 0 | 1 | 0 |
| Particle2010 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| CCampWindow | 3 | 1 | 0 | 1 | 0 | 0 | 1 |
| CDiplomacy | 3 | 2 | 0 | 0 | 0 | 0 | 1 |
| CEscMenuWindow | 3 | 1 | 0 | 1 | 0 | 1 | 0 |
| CPortraitsWindow | 3 | 2 | 0 | 0 | 0 | 1 | 0 |
| CSkyObject | 3 | 0 | 0 | 0 | 3 | 0 | 0 |
| CTutorialWindow | 3 | 1 | 0 | 1 | 0 | 0 | 1 |
| CUnitDatabase | 3 | 1 | 0 | 1 | 0 | 1 | 0 |
| Particle2002 | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| Particle2006 | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| Particle2009 | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| Particle200a | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| Particle200b | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| Particle200c | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| Particle200d | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| Particle2014 | 3 | 2 | 0 | 0 | 0 | 0 | 1 |
| Particle2018 | 3 | 2 | 0 | 0 | 0 | 0 | 1 |
| Particle2019 | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| Particle2026 | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| Particle202a | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| Particle202e | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| Particle202f | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| Particle2030 | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| Particle2033 | 3 | 0 | 0 | 3 | 0 | 0 | 0 |
| Particle2037 | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| Particle2039 | 3 | 0 | 0 | 0 | 1 | 0 | 2 |
| Particle203b | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| Particle203c | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| Particle203e | 3 | 1 | 0 | 2 | 0 | 0 | 0 |
| Particle2040 | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| Particle204b | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| Particle2052 | 3 | 0 | 0 | 3 | 0 | 0 | 0 |
| CBriefingWindow | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| CEffect | 2 | 0 | 0 | 1 | 1 | 0 | 0 |
| CInterface | 2 | 1 | 0 | 0 | 0 | 0 | 1 |
| CObject3D | 2 | 1 | 0 | 1 | 0 | 0 | 0 |
| CPauseDial | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| CScriptArea | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| CUnitSignal | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| CWorldLight | 2 | 1 | 0 | 1 | 0 | 0 | 0 |
| Particle2000 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| Particle2005 | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| Particle2008 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| Particle2012 | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| Particle2013 | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| Particle201a | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| Particle2038 | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| Particle203d | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| Particle2042 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| Particle2043 | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| Particle2047 | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| Particle2048 | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| Particle2049 | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| Particle204c | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| Particle204d | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| Particle204e | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| Particle204f | 2 | 0 | 0 | 1 | 0 | 0 | 1 |
| Particle2051 | 2 | 0 | 0 | 2 | 0 | 0 | 0 |
| Random | 2 | 0 | 0 | 1 | 0 | 1 | 0 |
| UnitRpg | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| CAiSettings | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| CArmorShape | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| CBattleWindow | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| CBehaviourGuard | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| CBehaviourSentry | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| CCampConstructorWindow | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| CCursorManager | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| CEffectSlotsWindow | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| CHeroTemplate | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| CInterface3D | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| CItem | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| CItemDatabase | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| CItemShape | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| CMagicTrapRecord | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| CMap | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| CMobLoader | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| CModeDial | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| CMultiplayerWindow | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| CObject | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| CPerkDatabase | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| CQuickItemsWindow | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| CScriptGroup | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| CScriptInstanceList | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| CSoundSystem | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| CSpellSlotsWindow | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| CTextDatabase | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| CTextWindow | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| CUnitPrototype | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| CUnitType | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| CWeaponShape | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| CZoneList | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| LightningNoise | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| MoveTrack | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| PathPoint | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| PointAction | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| PointHash | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| ScienceStats | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| Script | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| (без класса) | 7878 | 106 | 15 | 48 | 1092 | 962 | 5655 |

## Покрытие по блокам адресов (64 КБ)

| Блок | Всего | implemented | partial | module | решено | indirect | missing | Классы блока |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 400000 | 298 | 4 | 1 | 3 | 41 | 49 | 200 | CCamera, Vec3, CDbReader, Quat |
| 410000 | 291 | 2 | 0 | 0 | 46 | 24 | 219 | CCamera, Vec3 |
| 420000 | 443 | 0 | 0 | 0 | 103 | 29 | 311 | CClientSpecificManager |
| 430000 | 340 | 3 | 0 | 1 | 55 | 8 | 273 |  |
| 440000 | 197 | 0 | 0 | 0 | 28 | 7 | 162 | CFakeUnit, Vec3 |
| 450000 | 203 | 2 | 1 | 3 | 45 | 29 | 123 | CFakeFigure, CFigure, CDiplomacy, CFigure_2, CFigure_1 |
| 460000 | 235 | 12 | 0 | 6 | 39 | 43 | 135 | CLightning, CInventory, CHoover, LightningBolt, CLightningManager |
| 470000 | 234 | 0 | 0 | 1 | 52 | 8 | 173 | CFigure_1, CUnitServer |
| 480000 | 140 | 31 | 0 | 10 | 12 | 8 | 79 | CParticle, Particle2010, Particle2052, Particle200c, Particle200b |
| 490000 | 193 | 12 | 0 | 8 | 22 | 13 | 138 | CParticle, Particle2026, Particle202e, Particle202f, Particle202a |
| 4a0000 | 273 | 3 | 0 | 0 | 23 | 21 | 226 | CPlayer, CFigure_1 |
| 4b0000 | 216 | 22 | 0 | 8 | 34 | 49 | 103 | CScriptWorld, CScriptArea, CScriptGroup, CItem, PointAction |
| 4c0000 | 291 | 1 | 0 | 2 | 60 | 41 | 187 | CScriptInstanceList |
| 4d0000 | 207 | 0 | 0 | 0 | 25 | 22 | 160 | CTorchObjectSpecific, CServerSpecificManager, CSkyObject, CTextManager, CFigure_1 |
| 4e0000 | 173 | 4 | 0 | 6 | 18 | 10 | 135 | CVisibilityFog, CVisFogManager, CTextDatabase |
| 4f0000 | 213 | 2 | 0 | 0 | 30 | 9 | 172 |  |
| 500000 | 208 | 8 | 1 | 5 | 23 | 25 | 146 | CAnimationTrack, CAnimationNode, CAnimationSystem2, CUnit |
| 510000 | 279 | 11 | 1 | 5 | 52 | 48 | 162 | CObject3DClientSpecific, CObject3DServer, CMapObjectServer, CMapObjectClientSpecific, CMapObjectManager |
| 520000 | 319 | 46 | 1 | 16 | 39 | 44 | 173 | CUnitStats, CLeverObject, CMagicTrapObject, CUnit, CTorchObject |
| 530000 | 279 | 16 | 3 | 9 | 69 | 47 | 135 | CUnitClientSpecific, CUnitServer, CUnit, CUnitDatabase |
| 540000 | 168 | 9 | 1 | 1 | 9 | 46 | 102 | CUnitServer |
| 550000 | 306 | 11 | 1 | 7 | 60 | 40 | 187 | CUnitServer, CWorldServer, CEffectManager, CItemManager, CPlayerManager |
| 560000 | 280 | 6 | 1 | 3 | 71 | 38 | 161 | CWorldClient, CWeather, CBridgeManager, CLightningManager, CVisFogManager |
| 570000 | 290 | 4 | 0 | 4 | 30 | 50 | 202 | CWorld |
| 580000 | 290 | 5 | 0 | 2 | 43 | 23 | 217 | CWorldServer, CWorld |
| 590000 | 248 | 11 | 0 | 4 | 30 | 52 | 151 | CWorldServer, CFigure_1, CPlayer |
| 5a0000 | 128 | 8 | 0 | 1 | 24 | 7 | 88 | CFigureManager, CFigure_2, CFigureClientSpecific, CI3DFigure, CFigure_1 |
| 5b0000 | 197 | 14 | 1 | 11 | 10 | 54 | 107 | ServerMap, PointHash |
| 5c0000 | 192 | 11 | 3 | 3 | 12 | 48 | 115 | PathWindow, CSquad, CWorld, PathPoint, MoveTrack |
| 5d0000 | 157 | 32 | 3 | 4 | 11 | 34 | 73 | CUnitLogic, CBehaviourPatrol, CBehaviourAttack, CBehaviourGuard, CBehaviourSentry |
| 5e0000 | 153 | 13 | 0 | 0 | 20 | 33 | 87 | CInterfaceElement, CCursor, CI3DFigure, CCursorManager |
| 5f0000 | 299 | 10 | 1 | 4 | 33 | 51 | 200 | CInterfaceModel, CInterfaceWindow, CInterfaceRegion, CInterface3D, CInterface |
| 600000 | 186 | 3 | 0 | 1 | 20 | 20 | 142 | CBriefingWindow, CCampConstructorWindow |
| 610000 | 138 | 1 | 0 | 0 | 9 | 5 | 123 | CCampWindow, CInterface, CMultiplayerWindow |
| 620000 | 105 | 14 | 0 | 8 | 12 | 10 | 61 | CInventoryStrip, CCampWindow, CEscMenuWindow |
| 630000 | 141 | 8 | 1 | 4 | 17 | 12 | 99 | CMainMenuWindow, CBattleWindow, CSpellSlotsWindow, CQuickItemsWindow, CEffectSlotsWindow |
| 640000 | 88 | 17 | 1 | 2 | 3 | 20 | 45 | CInfoWindow, CMiniMapWindow, CPortraitsWindow, CPauseDial, CModeDial |
| 650000 | 189 | 10 | 1 | 3 | 33 | 34 | 108 | CParty, CTutorialWindow |
| 660000 | 265 | 9 | 0 | 9 | 25 | 51 | 171 | CPlayerClient, CParty, CPlayer, CPlayerManager, CItemManager |
| 670000 | 302 | 17 | 0 | 9 | 51 | 15 | 210 | CEffectSound, CEffectSoundSeq, CEffectArrowFigure, CEffectAmbientSound, CEffectLightning |
| 680000 | 238 | 19 | 0 | 3 | 43 | 24 | 149 | CSpellManager, CSpell, CSpellConstructor, CLight, CMaterialManager |
| 690000 | 269 | 6 | 0 | 1 | 25 | 44 | 193 | CTextureManager, CTexture, CFileRegistry, CInventory, CMaterial |
| 6a0000 | 244 | 9 | 0 | 5 | 34 | 37 | 159 | CItemWeapon, CItemArmor, CItemQuest, CItemLoot, CItemQuick |
| 6b0000 | 223 | 13 | 0 | 2 | 24 | 33 | 151 | CMusicSystem, CSoundManager, Stream, CSoundSystem |
| 6c0000 | 142 | 6 | 2 | 5 | 13 | 27 | 89 | CTerrain, CMap |
| 6d0000 | 225 | 4 | 0 | 1 | 23 | 11 | 186 | CUnit |
| 6e0000 | 60 | 1 | 0 | 0 | 5 | 7 | 47 |  |

## Python-файлы и функции game.exe

| Файл | Символов со ссылками | Функций game.exe |
|---|---:|---:|
| `client/audio.py` | 5 | 6 |
| `client/camp.py` | 10 | 11 |
| `client/cursor.py` | 2 | 1 |
| `client/demo.py` | 1 | 1 |
| `client/difficulty.py` | 3 | 1 |
| `client/flyinghp.py` | 4 | 2 |
| `client/gl.py` | 4 | 5 |
| `client/globalmap.py` | 1 | 1 |
| `client/hud.py` | 12 | 10 |
| `client/keys.py` | 1 | 1 |
| `client/loot.py` | 1 | 2 |
| `client/main.py` | 40 | 50 |
| `client/mapscript.py` | 11 | 14 |
| `client/menu.py` | 7 | 6 |
| `client/movie.py` | 5 | 1 |
| `client/options.py` | 10 | 8 |
| `client/picking.py` | 3 | 4 |
| `client/questmap.py` | 1 | 3 |
| `client/saveload.py` | 1 | 2 |
| `client/scene.py` | 10 | 11 |
| `client/talk.py` | 2 | 2 |
| `client/tutorial.py` | 2 | 1 |
| `client/ui.py` | 6 | 6 |
| `client/weather.py` | 5 | 5 |
| `client/zoneexit.py` | 1 | 2 |
| `eilib/adb.py` | 1 | 1 |
| `eilib/anm.py` | 1 | 1 |
| `eilib/fig.py` | 3 | 2 |
| `eilib/gamedata.py` | 6 | 7 |
| `eilib/idb.py` | 36 | 61 |
| `eilib/keys.py` | 3 | 1 |
| `eilib/lootdb.py` | 1 | 4 |
| `eilib/perkflags.py` | 1 | 1 |
| `eilib/reg.py` | 1 | 1 |
| `eilib/sav.py` | 24 | 19 |
| `eilib/zones.py` | 4 | 5 |
| `engine/acks.py` | 5 | 5 |
| `engine/ai.py` | 73 | 76 |
| `engine/ambient.py` | 9 | 9 |
| `engine/animation.py` | 5 | 7 |
| `engine/camera.py` | 13 | 10 |
| `engine/camp.py` | 9 | 9 |
| `engine/combat.py` | 29 | 39 |
| `engine/companions.py` | 1 | 1 |
| `engine/difficulty.py` | 6 | 3 |
| `engine/equipment.py` | 12 | 14 |
| `engine/experience.py` | 22 | 19 |
| `engine/figure.py` | 1 | 1 |
| `engine/flyinghp.py` | 8 | 10 |
| `engine/footprints.py` | 1 | 1 |
| `engine/globalmap.py` | 1 | 1 |
| `engine/hud.py` | 43 | 33 |
| `engine/inventory.py` | 18 | 14 |
| `engine/lever.py` | 6 | 12 |
| `engine/lighting.py` | 2 | 2 |
| `engine/lightning.py` | 22 | 16 |
| `engine/loot.py` | 3 | 2 |
| `engine/magic.py` | 82 | 67 |
| `engine/menus.py` | 4 | 9 |
| `engine/mercenaries.py` | 6 | 4 |
| `engine/messagebox.py` | 5 | 5 |
| `engine/missiles.py` | 3 | 6 |
| `engine/music.py` | 12 | 12 |
| `engine/options.py` | 13 | 12 |
| `engine/particles.py` | 24 | 60 |
| `engine/party.py` | 1 | 8 |
| `engine/pathing.py` | 28 | 25 |
| `engine/prop.py` | 3 | 3 |
| `engine/quat.py` | 1 | 1 |
| `engine/questlog.py` | 3 | 2 |
| `engine/questmap.py` | 3 | 2 |
| `engine/quickitems.py` | 1 | 3 |
| `engine/roster.py` | 6 | 12 |
| `engine/saveload.py` | 6 | 6 |
| `engine/script.py` | 36 | 54 |
| `engine/signals.py` | 5 | 2 |
| `engine/skills.py` | 6 | 9 |
| `engine/sounds.py` | 17 | 10 |
| `engine/sources.py` | 1 | 11 |
| `engine/stats.py` | 2 | 17 |
| `engine/talk.py` | 4 | 4 |
| `engine/terrain.py` | 2 | 1 |
| `engine/timing.py` | 1 | 1 |
| `engine/tooltip.py` | 3 | 2 |
| `engine/trade.py` | 3 | 3 |
| `engine/trap.py` | 7 | 13 |
| `engine/tutorial.py` | 2 | 4 |
| `engine/unit.py` | 37 | 43 |
| `engine/water.py` | 6 | 10 |
| `engine/weather.py` | 20 | 22 |
| `engine/woundskin.py` | 7 | 5 |
| `tools/scenario.py` | 1 | 1 |

Без единой ссылки на game.exe (27): `client/__init__.py`, `client/briefing.py`, `client/credits.py`, `client/footprints.py`, `client/music.py`, `eilib/__init__.py`, `eilib/cam.py`, `eilib/mapdata.py`, `eilib/mmp.py`, `eilib/mob.py`, `eilib/quickdb.py`, `eilib/res.py`, `eilib/rtf.py`, `eilib/script.py`, `eilib/skel.py`, `engine/__init__.py`, `engine/actions.py`, `engine/briefing.py`, `engine/credits.py`, `engine/spellvisual.py`, `engine/world.py`, `tools/__init__.py`, `tools/handover.py`, `tools/plot_report.py`, `tools/portraits.py`, `tools/snapshot.py`, `tools/story_pdf.py`.

## Реализованные функции (`implemented`) — 450

| Адрес | Функция C | Строк | Python |
|---|---|---:|---|
| 409070 | CDbReader::NextNode | 44 | `<module>` eilib/idb.py:6, `<module>` eilib/lootdb.py:14, `<module>` eilib/sav.py:18, `read_nodes` eilib/sav.py:368 |
| 40e4c0 | FUN_0040e4c0 | 119 | `RemakeApp.set_camera_floor` client/main.py:751, `<module>` engine/camera.py:16, `FLOOR_GROUND_REACH` engine/camera.py:97, `floor_height` engine/camera.py:143, +1 |
| 40f230 | FUN_0040f230 | 37 | `RemakeApp.move_camera` client/main.py:1573, `Camera.__init__` engine/camera.py:189, `Camera.scroll_keys` engine/camera.py:272 |
| 40fb30 | FUN_0040fb30 | 369 | `RemakeApp.listener_point` client/main.py:743, `RemakeApp.reset_camera` client/main.py:1380, `RemakeApp.move_camera` client/main.py:1573, `<module>` engine/camera.py:7, +7 |
| 4113a0 | CCamera::Shake | 23 | `SHAKE_STRENGTH` engine/camera.py:86, `Camera.__init__` engine/camera.py:184, `Camera.shake` engine/camera.py:206 |
| 41a020 | FUN_0041a020 | 19 | `ClientHost.leader` client/mapscript.py:186 |
| 430960 | FUN_00430960 | 312 | `<module>` eilib/sav.py:13, `WINDOW_SIZE` eilib/sav.py:56, `copy_length` eilib/sav.py:251, `copy_distance` eilib/sav.py:268, +1 |
| 430fe0 | FUN_00430fe0 | 28 | `_Bits` eilib/sav.py:213 |
| 431040 | FUN_00431040 | 19 | `_Bits` eilib/sav.py:213 |
| 4554e0 | CDiplomacy::Set | 62 | `Diplomacy.set_relation` engine/ai.py:241, `<module>` engine/script.py:57 |
| 455630 | CDiplomacy::Get | 15 | `Diplomacy.relation` engine/ai.py:229 |
| 461360 | FUN_00461360 | 21 | `ClientHost.speaker` client/mapscript.py:376 |
| 461cb0 | FUN_00461cb0 | 208 | `BattleScreen.open_loot` client/hud.py:1088, `<module>` client/loot.py:5, `RemakeApp.rebuild_corpse` client/main.py:1904, `<module>` eilib/lootdb.py:7, +3 |
| 466190 | LightningBolt::RandomRecord | 54 | `<module>` engine/lightning.py:19, `random_record` engine/lightning.py:153 |
| 4662c0 | LightningBolt::Noise | 232 | `<module>` engine/lightning.py:18, `NOISE_MOST_STEPS` engine/lightning.py:68, `Bolt.noise` engine/lightning.py:244 |
| 466820 | LightningBolt::Aim | 100 | `<module>` engine/lightning.py:9, `SHORT_LENGTH` engine/lightning.py:71, `segments_of` engine/lightning.py:168, `Bolt.aim` engine/lightning.py:220 |
| 466ce0 | LightningBolt::Shape | 62 | `<module>` engine/lightning.py:14, `Bolt.shape` engine/lightning.py:272 |
| 4672e0 | CLightning::Sway | 117 | `<module>` engine/lightning.py:29, `NOISE_TICKS` engine/lightning.py:66, `Lightning.shape` engine/lightning.py:381 |
| 467590 | CLightning::DrawBall | 173 | `<module>` engine/lightning.py:40, `BALL_SIZE_SCALE` engine/lightning.py:113, `Lightning.balls` engine/lightning.py:422, `ball_squares` engine/lightning.py:440 |
| 467ad0 | CLightning::Draw | 104 | `SceneWidget.set_particles` client/scene.py:440, `SceneWidget.band_quads` client/scene.py:589, `<module>` engine/lightning.py:38, `TEXTURE_LENGTH_SCALE` engine/lightning.py:108, +2 |
| 467e10 | CLightning::Follow | 97 | `<module>` engine/lightning.py:7, `Lightning.random_offset` engine/lightning.py:353, `Lightning.follow` engine/lightning.py:364 |
| 4680d0 | CLightning::Init | 195 | `<module>` engine/lightning.py:5, `BRANCH_FRACTIONS` engine/lightning.py:83, `POWER_SWAY` engine/lightning.py:89, `Bolt.seed` engine/lightning.py:236, +2 |
| 468a00 | CLightningManager::LoadNoise | 67 | `RemakeApp.__init__` client/main.py:174, `<module>` engine/lightning.py:2, `load_noise` engine/lightning.py:137 |
| 4847e0 | CParticle::Reset | 67 | `Emitter.setup` engine/particles.py:326 |
| 485280 | FUN_00485280 | 39 | `unit_particles` engine/magic.py:564, `Emitter.__init__` engine/particles.py:305 |
| 485480 | CParticle::Scale | 19 | `<module>` engine/particles.py:30, `Emitter.set_scale` engine/particles.py:522 |
| 485540 | CParticle::vf02 | 2172 | `<module>` engine/particles.py:4, `FIRE_SMOKE` engine/particles.py:47, `START_TRANS_TICKS` engine/particles.py:160, `Emitter.setup` engine/particles.py:326, +4 |
| 488f90 | Particle200c::Emit | 71 | `Emitter` engine/particles.py:1097 |
| 489190 | Particle200c::Move | 52 | `Emitter` engine/particles.py:1097 |
| 4892e0 | Particle200c::Source | 16 | `Emitter` engine/particles.py:1097 |
| 48a8f0 | Particle2009::Emit | 155 | `FIREWORK_LEAST` engine/particles.py:204, `Emitter` engine/particles.py:602 |
| 48ada0 | Particle2009::Move | 52 | `FIREWORK_LEAST` engine/particles.py:204, `Emitter` engine/particles.py:602 |
| 48aec0 | Particle2009::Source | 85 | `Emitter.source_firework` engine/particles.py:606 |
| 48b080 | Particle2008::Emit | 164 | `Emitter` engine/particles.py:1049 |
| 48b570 | Particle2008::Move | 39 | `Emitter` engine/particles.py:1049 |
| 48b630 | Particle2006::Emit | 222 | `HEALING_TICKS` engine/particles.py:152, `Emitter` engine/particles.py:1396, `Emitter.healing_sources` engine/particles.py:1401 |
| 48bbe0 | Particle2006::Move | 50 | `HEALING_TICKS` engine/particles.py:152, `Emitter` engine/particles.py:1396 |
| 48bce0 | Particle2006::Source | 35 | `HEALING_TICKS` engine/particles.py:152, `Emitter` engine/particles.py:1396 |
| 48c090 | Particle2002::Emit | 123 | `Emitter` engine/particles.py:1154 |
| 48c590 | Particle2002::Move | 68 | `Emitter` engine/particles.py:1154 |
| 48c760 | Particle2002::Source | 28 | `Emitter` engine/particles.py:1154 |
| 48c800 | Particle2000::Emit | 101 | `Emitter` engine/particles.py:805 |
| 48cad0 | Particle2000::Move | 45 | `Emitter` engine/particles.py:805 |
| 48cba0 | Particle204e::Emit | 115 | `Emitter` engine/particles.py:884 |
| 48ced0 | Particle204e::Move | 69 | `Emitter` engine/particles.py:884 |
| 48d8d0 | Particle2014::Emit | 171 | `LINK_LEAST` engine/particles.py:215, `Emitter` engine/particles.py:644 |
| 48de30 | Particle2014::Move | 80 | `LINK_LEAST` engine/particles.py:215, `Emitter` engine/particles.py:644 |
| 48e9e0 | Particle2010::Emit | 80 | `Emitter` engine/particles.py:980 |
| 48ec20 | Particle2010::Move | 66 | `Emitter` engine/particles.py:980 |
| 48ed60 | Particle2010::Emit | 33 | `Emitter` engine/particles.py:980, `Emitter.born_fire_wall` engine/particles.py:1010 |
| 48ee20 | Particle2010::Source | 25 | `Emitter` engine/particles.py:980 |
| 48f7f0 | Particle2018::Emit | 98 | `CLAY_RING_MOST` engine/particles.py:222, `Emitter` engine/particles.py:682 |
| 48fab0 | Particle2018::Move | 52 | `CLAY_RING_MOST` engine/particles.py:222, `Emitter` engine/particles.py:682 |
| 48fdb0 | Particle2019::Emit | 102 | `TELEPORT_TICKS` engine/particles.py:194, `Emitter` engine/particles.py:1500 |
| 490060 | Particle2019::Move | 47 | `TELEPORT_TICKS` engine/particles.py:194, `Emitter` engine/particles.py:1500 |
| 490110 | Particle2019::Source | 16 | `TELEPORT_TICKS` engine/particles.py:194, `Emitter` engine/particles.py:1500 |
| 4923b0 | Particle2030::Emit | 285 | `Emitter` engine/particles.py:1274 |
| 492b80 | Particle2030::Move | 58 | `Emitter` engine/particles.py:1274 |
| 492cc0 | Particle2030::Source | 26 | `Emitter` engine/particles.py:1274 |
| 493d10 | Particle203d::Emit | 16 | `PORTAL_STAR_SIZE` engine/particles.py:116, `Emitter` engine/particles.py:714 |
| 493d50 | Particle203d::Move | 28 | `PORTAL_STAR_SIZE` engine/particles.py:116, `Emitter` engine/particles.py:714 |
| 494930 | Particle203e::Source | 13 | `PORTAL_LEAST` engine/particles.py:120, `Emitter.set_scale` engine/particles.py:531 |
| 494950 | Particle2042::Emit | 18 | `Emitter` engine/particles.py:956 |
| 4949b0 | Particle2042::Move | 25 | `Emitter` engine/particles.py:956 |
| 496eb0 | CParticle::vf06 | 285 | `<module>` engine/particles.py:10, `Emitter.update` engine/particles.py:538 |
| 4976b0 | CParticle::Draw | 218 | `SceneWidget.set_particles` client/scene.py:438, `<module>` engine/particles.py:24, `SIZES` engine/weather.py:134 |
| 4a5250 | FUN_004a5250 | 32 | `<module>` eilib/sav.py:17, `SHORT_BLOB` eilib/sav.py:58, `read_blob` eilib/sav.py:350 |
| 4a5e10 | CPlayer::FindParty | 30 | `<module>` engine/roster.py:5, `Roster.party` engine/roster.py:121 |
| 4a5e90 | CPlayer::CreateParty | 67 | `<module>` engine/roster.py:4, `Roster.create_party` engine/roster.py:160 |
| 4b1ed0 | CScriptArea::AddRect | 115 | `ScriptWorld.__init__` engine/script.py:469 |
| 4b2070 | CScriptArea::AddRound | 114 | `ScriptWorld.__init__` engine/script.py:469 |
| 4b2530 | FUN_004b2530 | 18 | `ScriptWorld.unit_order` engine/script.py:805 |
| 4b2ef0 | CScriptGroup::Add | 130 | `SpellParticles` engine/magic.py:1148, `Possession` engine/magic.py:1259, `<module>` engine/script.py:30 |
| 4b31d0 | FUN_004b31d0 | 18 | `Possession` engine/magic.py:1259 |
| 4b3330 | FUN_004b3330 | 23 | `spread` engine/magic.py:1396 |
| 4b56c0 | FUN_004b56c0 | 114 | `ClientHost.unit_of_player` client/mapscript.py:173, `ClientHost.leader` client/mapscript.py:186, `Host.unit_of_player` engine/script.py:157, `Host.leader` engine/script.py:164 |
| 4b5b00 | CScriptWorld::EvaluateGroupFunction | 146 | `ClientHost.seen_by_player` client/mapscript.py:192 |
| 4b6760 | CScriptWorld::RunThread | 2081 | `<module>` engine/ai.py:61, `Controller` engine/ai.py:1102, `<module>` engine/script.py:6, `Thread` engine/script.py:341, +1 |
| 4ba030 | CItem::FixDurability | 10 | `repair_item` engine/trade.py:51 |
| 4ba360 | FUN_004ba360 | 20 | `<module>` engine/ai.py:37, `Controller.sentry` engine/ai.py:1254, `<module>` engine/script.py:38 |
| 4ba450 | FUN_004ba450 | 17 | `Controller.follow_behaviour` engine/ai.py:1232 |
| 4ba610 | FUN_004ba610 | 10 | `<module>` engine/ai.py:42, `Controller.standard_behaviours` engine/ai.py:1186 |
| 4ba710 | FUN_004ba710 | 14 | `<module>` engine/ai.py:42, `Controller.standard_behaviours` engine/ai.py:1186, `Controller.aggression` engine/ai.py:1201 |
| 4ba830 | FUN_004ba830 | 14 | `Controller.revenge` engine/ai.py:1212 |
| 4ba920 | FUN_004ba920 | 10 | `<module>` engine/ai.py:42, `Controller.standard_behaviours` engine/ai.py:1186 |
| 4baec0 | Script::PartyOfPath | 49 | `<module>` engine/roster.py:10, `SEPARATOR` engine/roster.py:38, `Roster.split` engine/roster.py:144 |
| 4bb520 | CScriptWorld::CheckBranch | 27 | `<module>` engine/script.py:8, `ScriptWorld.check` engine/script.py:572 |
| 4bb6a0 | CScriptWorld::SetScriptArgs | 72 | `ScriptWorld.call_script` engine/script.py:616 |
| 4bc420 | CScriptWorld::StartThread | 99 | `<module>` engine/script.py:9, `ScriptWorld.start_thread` engine/script.py:581 |
| 4bd260 | CScriptWorld::CreateFX | 55 | `Host.play_fx` engine/script.py:306 |
| 4bf000 | CScriptWorld::Tick | 277 | `<module>` engine/script.py:4, `ScriptWorld.tick` engine/script.py:556 |
| 4c9c90 | FUN_004c9c90 | 53 | `ScriptWorld.quest_complete` engine/script.py:913 |
| 4e40b0 | FUN_004e40b0 | 53 | `<module>` client/weather.py:12, `WeatherView.sound_volume` client/weather.py:191, `<module>` engine/weather.py:30, `LEVELS` engine/weather.py:75, +1 |
| 4e41f0 | FUN_004e41f0 | 17 | `<module>` engine/weather.py:9, `Precipitation.__init__` engine/weather.py:305 |
| 4e4240 | FUN_004e4240 | 423 | `SceneWidget.set_particles` client/scene.py:442, `<module>` client/weather.py:6, `<module>` engine/weather.py:8, `RAIN_HASH` engine/weather.py:94, +3 |
| 4e4c80 | FUN_004e4c80 | 401 | `SceneWidget.set_particles` client/scene.py:442, `<module>` client/weather.py:6, `<module>` engine/weather.py:10, `RAIN_HASH` engine/weather.py:94, +3 |
| 4f7e70 | LoadFigure | 22 | `parse` eilib/fig.py:183 |
| 4f87d0 | LoadFigureOld | 156 | `<module>` eilib/fig.py:9, `six_variants` eilib/fig.py:121 |
| 50c230 | FUN_0050c230 | 147 | `<module>` engine/animation.py:6, `Skeleton.pose` engine/animation.py:112 |
| 50ca50 | FUN_0050ca50 | 63 | `AnimationPlayer.play` engine/animation.py:212 |
| 50cba0 | FUN_0050cba0 | 42 | `AnimationPlayer.enqueue` engine/animation.py:221 |
| 50d450 | FUN_0050d450 | 32 | `Skeleton.pose` engine/animation.py:111 |
| 50d5e0 | FUN_0050d5e0 | 10 | `AnimationPlayer.add_time` engine/animation.py:230 |
| 50d600 | FUN_0050d600 | 113 | `Action.is_cycle` eilib/adb.py:122, `<module>` engine/animation.py:10, `AnimationPlayer.add_time` engine/animation.py:230 |
| 50f1e0 | CUnit::AddSpell | 47 | `Enchantment` engine/combat.py:290, `<module>` engine/magic.py:52 |
| 50f690 | CUnit::SpellEffect | 44 | `Fighter.enchantment` engine/combat.py:450 |
| 510890 | CMapObjectClientSpecific::vf06 | 124 | `RemakeApp.update_spell_sounds` client/main.py:2086, `PhraseSound.__init__` engine/acks.py:159, `<module>` engine/magic.py:133, `ENCHANTMENT_WARNING_SOUND` engine/magic.py:316, +1 |
| 5112a0 | FUN_005112a0 | 141 | `RemakeApp.__init__` client/main.py:235, `RemakeApp.new_scene` client/main.py:349, `RemakeApp.update_spell_sounds` client/main.py:2086, `<module>` engine/magic.py:131, +1 |
| 5114d0 | FUN_005114d0 | 127 | `<module>` engine/magic.py:131, `ENCHANTMENT_SOUNDS` engine/magic.py:307, `enchantment_loop` engine/magic.py:650 |
| 5117f0 | CMapObjectClientSpecific::vf16 | 18 | `RemakeApp.update_flying_numbers` client/main.py:1511, `<module>` engine/ai.py:77, `FlyingNumbers.add` engine/flyinghp.py:154 |
| 5136d0 | CUnitStats::InitFromPrototype | 211 | `RemakeApp.record_hero_stats` client/main.py:1032, `<module>` eilib/idb.py:32, `UnitType.__init__` eilib/idb.py:380, `UnitPrototype.__init__` eilib/idb.py:463, +6 |
| 5145d0 | CUnitDatabase::FindHeroTemplate | 52 | `<module>` eilib/idb.py:30, `UnitDatabase.template` eilib/idb.py:711, `<module>` engine/experience.py:16 |
| 515820 | CMapObjectServer::vf06 | 51 | `Fighter.tick_enchantments` engine/combat.py:512, `<module>` engine/magic.py:53, `FREEZING_SPELLS` engine/magic.py:351 |
| 5179e0 | CFigure::PartPassFlags | 233 | `<module>` engine/lever.py:12, `<module>` engine/pathing.py:41, `part_kind` engine/pathing.py:692 |
| 518fc0 | CObject3D::FindSpecialAction | 28 | `<module>` engine/prop.py:4, `Prop.special_action` engine/prop.py:49 |
| 51f3a0 | FUN_0051f3a0 | 45 | `Skill` eilib/idb.py:513 |
| 51f480 | FUN_0051f480 | 26 | `Perk` eilib/idb.py:531 |
| 5207d0 | FUN_005207d0 | 43 | `parse_perks` eilib/idb.py:790 |
| 5208c0 | FUN_005208c0 | 43 | `parse_perks` eilib/idb.py:790 |
| 522a80 | ScienceStats::Allows | 20 | `<module>` engine/lever.py:22, `KEY_FLAG` engine/lever.py:37, `Lever.allows` engine/lever.py:135, `<module>` engine/party.py:14 |
| 5230c0 | CUnitStats::WearArmorItem | 55 | `ArmorShape.__init__` eilib/idb.py:259, `<module>` engine/combat.py:78 |
| 523300 | CUnitStats::RebuildArmor | 36 | `<module>` engine/combat.py:58, `Fighter.rebuild_armor` engine/combat.py:418, `Item.is_broken` engine/equipment.py:95 |
| 523740 | CUnit::Regenerate | 61 | `UnitType.__init__` eilib/idb.py:380, `REGENERATION_TICKS` engine/combat.py:143, `Fighter.regenerate` engine/combat.py:487, `<module>` engine/magic.py:62, +1 |
| 523960 | CUnitStats::GiveExperience | 23 | `HeroTemplate.__init__` eilib/idb.py:483, `<module>` engine/experience.py:17, `HeroStats.add_experience` engine/experience.py:336 |
| 5239b0 | CUnitStats::AddExperience | 90 | `<module>` engine/experience.py:24, `HeroStats.add_experience` engine/experience.py:336, `HeroStats.max_health` engine/experience.py:406, `update_fighter` engine/experience.py:457 |
| 523cd0 | UnitRpg::ManaFactor | 11 | `Spell.mana_cost` engine/magic.py:508 |
| 523d30 | ClampDamageMin | 29 | `<module>` engine/combat.py:25, `damage_roll` engine/combat.py:581 |
| 524110 | CUnit::ArmsFactor | 33 | `Fighter.strike_speed` engine/combat.py:467, `Unit.strike_speed` engine/unit.py:426 |
| 5241c0 | CUnit::RunLimit | 33 | `<module>` engine/stats.py:49, `RUN_FREE` engine/unit.py:70, `Unit.run_limit` engine/unit.py:839 |
| 524560 | CUnit::RecountStats | 281 | `CampView.after_stats_change` client/camp.py:841, `<module>` engine/combat.py:64, `PROTECTION_SPELLS` engine/combat.py:123, `PLAIN_STRIKE_SPEED` engine/combat.py:134, +16 |
| 525070 | CUnitStats::DamagePart | 119 | `HitLocation` eilib/idb.py:411, `<module>` engine/combat.py:38, `apply_hit` engine/combat.py:679, `_hit_part` engine/combat.py:721, +2 |
| 5253e0 | CUnitStats::WearItem | 40 | `RemakeApp.on_item_wear` client/main.py:2179, `<module>` engine/combat.py:52, `MESSAGE_ARMOR_BROKEN` engine/combat.py:201, `wear_item` engine/combat.py:614, +2 |
| 5254b0 | CUnitStats::TakeBlow | 119 | `DAMAGE_TYPES_COUNT` eilib/idb.py:119, `<module>` engine/combat.py:38, `apply_hit` engine/combat.py:679, `_apply_hit` engine/combat.py:695, +3 |
| 525720 | CUnitStats::ArmorLayerHit | 41 | `<module>` engine/combat.py:55, `armor_kinds` engine/combat.py:638, `_hit_part` engine/combat.py:722 |
| 525800 | CUnit::HealParts | 75 | `<module>` engine/combat.py:73, `Fighter.regenerate` engine/combat.py:489, `heal` engine/combat.py:587, `<module>` engine/magic.py:49 |
| 525900 | CUnit::Heal | 14 | `heal` engine/combat.py:588 |
| 5259b0 | CUnitStats::HealthFromParts | 33 | `<module>` engine/combat.py:22, `Fighter.health` engine/combat.py:370, `heal` engine/combat.py:588 |
| 525a00 | CUnitStats::PartMaxHealth | 28 | `<module>` engine/combat.py:19, `Fighter.recount` engine/combat.py:529 |
| 525a70 | CUnitStats::ScalePartsHealth | 31 | `update_fighter` engine/experience.py:458 |
| 5260a0 | CUnitStats::PerkAvailable | 47 | `CampView.skills_items` client/camp.py:764, `<module>` eilib/perkflags.py:3, `<module>` engine/skills.py:9, `perk_available` engine/skills.py:125 |
| 526450 | CUnitStats::AddPerk | 62 | `CampView.on_perk_click` client/camp.py:829, `HeroTemplate.__init__` eilib/idb.py:494, `<module>` engine/experience.py:17, `HeroStats.__init__` engine/experience.py:192, +2 |
| 526c70 | CUnitStats::SkillAttackParry | 110 | `WeaponShape.__init__` eilib/idb.py:240, `<module>` engine/combat.py:16, `Fighter.equip` engine/combat.py:393, `<module>` engine/experience.py:34, +2 |
| 527eb0 | UnitRpg::SpellSkill | 48 | `<module>` engine/experience.py:37, `SCHOOL_REMAP` engine/experience.py:127, `HeroStats.school_skill` engine/experience.py:444, `<module>` engine/magic.py:38, +1 |
| 528050 | CUnitStats::PerkPrice | 50 | `CampView.on_perk_click` client/camp.py:829, `<module>` engine/skills.py:16, `perk_cost` engine/skills.py:113 |
| 528270 | CUnitStats::PerkOfGroup | 20 | `PerkDatabase.level_perk` eilib/idb.py:570, `HeroStats.perk_power` engine/experience.py:273 |
| 528ab0 | CUnitStats::GiveSkill | 99 | `CampView.on_skill_click` client/camp.py:817, `<module>` engine/experience.py:19, `RpgSettings.skill_price` engine/experience.py:160, `HeroStats.give_skill` engine/experience.py:361, +3 |
| 528da0 | CUnitStats::GiveScience | 99 | `<module>` engine/experience.py:19, `SKILL_SCIENCE` engine/experience.py:83, `HeroStats.give_skill` engine/experience.py:361, `ScriptWorld.give_skill` engine/script.py:929, +1 |
| 529090 | CUnitStats::GiveStrength | 68 | `HeroStats.give_strength` engine/experience.py:391, `update_fighter` engine/experience.py:457, `ScriptWorld.give_attribute` engine/script.py:942 |
| 529220 | CUnitStats::GiveDexterity | 63 | `HeroStats.give_strength` engine/experience.py:391, `ScriptWorld.give_attribute` engine/script.py:942 |
| 529400 | CUnitStats::GiveIntelligence | 85 | `HeroStats.give_strength` engine/experience.py:391, `ScriptWorld.give_attribute` engine/script.py:942 |
| 52a4b0 | CMagicTrapObject::Cast | 98 | `<module>` engine/trap.py:21, `MagicTrap.cast` engine/trap.py:169 |
| 52a920 | CMagicTrapObject::FindVictims | 112 | `<module>` engine/trap.py:20, `MagicTrap.victims` engine/trap.py:135 |
| 52ac20 | CMagicTrapObject::vf06 | 28 | `<module>` engine/trap.py:16, `MagicTrap.update` engine/trap.py:147 |
| 52b490 | CLeverObject::vf06 | 38 | `LeverView` client/scene.py:292, `<module>` engine/lever.py:5, `Lever.update` engine/lever.py:112 |
| 52b7a0 | CLeverObject::Switch | 158 | `<module>` engine/lever.py:7, `SWITCH_TICKS` engine/lever.py:34, `Lever.switch` engine/lever.py:95, `ScriptWorld.switch_lever` engine/script.py:821 |
| 52bad0 | CLeverObject::Enable | 16 | `<module>` engine/lever.py:17, `ScriptWorld.enable_lever` engine/script.py:831 |
| 52bb20 | CLeverObject::SetScience | 13 | `set_science` engine/script.py:1047 |
| 52d740 | CUnitPrototype::Read | 66 | `<module>` eilib/idb.py:28, `UnitPrototype` eilib/idb.py:430, `<module>` eilib/lootdb.py:2, `<module>` engine/stats.py:3 |
| 52da60 | CHeroTemplate::Read | 89 | `<module>` eilib/idb.py:30, `HeroTemplate` eilib/idb.py:476 |
| 52dc70 | CUnitType::Read | 54 | `UnitType` eilib/idb.py:373 |
| 52df40 | FUN_0052df40 | 40 | `_texts` eilib/idb.py:832 |
| 52e0d0 | CUnitServer::ApplyDifficulty | 17 | `RemakeApp.difficulty_level` client/main.py:654, `<module>` engine/combat.py:8, `<module>` engine/difficulty.py:11 |
| 52e5b0 | FUN_0052e5b0 | 69 | `Unit.__init__` engine/unit.py:167 |
| 530830 | FUN_00530830 | 9 | `Fighter.__init__` engine/combat.py:356 |
| 530a60 | CUnitServer::IsBackstab | 69 | `<module>` engine/combat.py:27, `is_backstab` engine/combat.py:564, `Unit.resolve_blow` engine/unit.py:448 |
| 5313b0 | FUN_005313b0 | 18 | `ScriptWorld.remove_quest_item` engine/script.py:976 |
| 534590 | CUnitClientSpecific::StepSoundName | 57 | `GameData.sound_count` eilib/gamedata.py:278, `UnitType.__init__` eilib/idb.py:390, `<module>` engine/sounds.py:22, `STEP_HEIGHT` engine/sounds.py:74, +2 |
| 534760 | CUnitClientSpecific::VoiceSoundName | 135 | `UnitType.__init__` eilib/idb.py:394, `UnitPrototype.__init__` eilib/idb.py:456, `<module>` engine/sounds.py:4, `SKILL_SOUNDS` engine/sounds.py:83, +2 |
| 534b60 | CUnitClientSpecific::UpdateSounds | 255 | `AudioPlayer.on_unit_sounds` client/audio.py:122, `<module>` engine/ai.py:77, `<module>` engine/sounds.py:17, `PRIORITY_LOW` engine/sounds.py:56, +2 |
| 535850 | CUnitClientSpecific::FollowTrack | 132 | `<module>` engine/unit.py:11, `CLIP_SPEED_DIVISOR` engine/unit.py:80, `Unit.natural_speed` engine/unit.py:698 |
| 536670 | CUnitClientSpecific::BuildTextures | 279 | `ArmorShape.__init__` eilib/idb.py:254, `<module>` engine/equipment.py:14, `UNDER_LAYER_SLOTS` engine/equipment.py:47 |
| 536ea0 | FUN_00536ea0 | 113 | `ArmorShape.__init__` eilib/idb.py:256, `<module>` engine/woundskin.py:2, `LIGHT_LIMIT` engine/woundskin.py:47, `ARMOR_COVERS` engine/woundskin.py:51, +1 |
| 5370f0 | FUN_005370f0 | 109 | `<module>` engine/woundskin.py:2, `LAYER_NAME` engine/woundskin.py:41, `layer_names` engine/woundskin.py:100, `wounded` engine/woundskin.py:130 |
| 537500 | CUnitClientSpecific::vf46 | 257 | `UnitType.__init__` eilib/idb.py:397, `UnitPrototype.__init__` eilib/idb.py:459, `<module>` engine/sounds.py:30, `MIN_DAMAGE` engine/sounds.py:71, +3 |
| 537cb0 | CUnitClientSpecific::ShowWeapon | 57 | `weapon_hidden_parts` engine/equipment.py:319 |
| 538c20 | CUnitClientSpecific::UpdateArrowInHand | 61 | `arrow_in_hand_visible` engine/equipment.py:337 |
| 53dc10 | FUN_0053dc10 | 43 | `parse_units` eilib/idb.py:777 |
| 53dd00 | FUN_0053dd00 | 43 | `parse_units` eilib/idb.py:777 |
| 53df50 | FUN_0053df50 | 43 | `parse_units` eilib/idb.py:777 |
| 546480 | CUnitServer::AimLocation | 115 | `<module>` engine/combat.py:26, `Unit.__init__` engine/unit.py:176 |
| 5499f0 | FUN_005499f0 | 98 | `Controller.think` engine/ai.py:831, `<module>` engine/magic.py:61 |
| 54b6c0 | CUnitServer::Shoots | 28 | `RemakeApp.make_unit` client/main.py:616, `<module>` engine/ai.py:53, `<module>` engine/stats.py:29, `Unit.shoots` engine/unit.py:311 |
| 54bbc0 | CUnitServer::CanTrace | 53 | `spread` engine/magic.py:1396 |
| 54d140 | CUnitServer::KeepAtFoe | 275 | `<module>` engine/unit.py:19, `Unit.strike` engine/unit.py:395 |
| 54d960 | CUnitServer::StartStrike | 171 | `WeaponShape.__init__` eilib/idb.py:242, `UnitPrototype.__init__` eilib/idb.py:464, `PLAIN_STRIKE_SPEED` engine/combat.py:135, `Fighter.__init__` engine/combat.py:354, +3 |
| 54ee10 | CUnitServer::ReleaseSpell | 92 | `<module>` engine/magic.py:36, `Spell.mana_cost` engine/magic.py:507, `<module>` engine/unit.py:22, `Unit.spell_mana` engine/unit.py:369, +1 |
| 54f450 | CUnitServer::DeliverBlow | 57 | `RemakeApp.make_unit` client/main.py:617, `<module>` engine/missiles.py:4, `SHOOTING_WEAPONS` engine/missiles.py:47, `Unit.resolve_blow` engine/unit.py:461 |
| 54f630 | CUnitServer::MakeBlowPacket | 98 | `UnitPrototype.__init__` eilib/idb.py:446, `<module>` engine/combat.py:13, `Hit` engine/combat.py:264, `<module>` engine/missiles.py:5, +1 |
| 552300 | CUnitServer::vf06 | 130 | `Fighter.regenerate` engine/combat.py:487, `Unit.regenerate` engine/unit.py:547 |
| 552780 | FUN_00552780 | 13 | `RemakeApp.on_item_wear` client/main.py:2179, `ITEM_MESSAGE_GROUPS` engine/acks.py:63, `MESSAGE_ARMOR_BROKEN` engine/combat.py:201, `SpellParticles` engine/magic.py:1150, +2 |
| 553380 | CUnitServer::AddExperience | 11 | `HeroStats.add_experience` engine/experience.py:336 |
| 553df0 | CUnitServer::AttackRoll | 57 | `<module>` engine/combat.py:31, `attack_roll` engine/combat.py:656, `<module>` engine/magic.py:30, `inflict_damage` engine/script.py:1071 |
| 553f40 | CUnitServer::vf20 | 149 | `RemakeApp.on_kill` client/main.py:2169, `UnitPrototype.__init__` eilib/idb.py:461, `Hit.__init__` engine/combat.py:279, `is_rewarded_kill` engine/experience.py:516, +3 |
| 554450 | CUnitServer::DropOrders | 102 | `RemakeApp.push_units_out` client/main.py:958, `<module>` engine/script.py:40, `ScriptWorld.set_place` engine/script.py:952 |
| 554f50 | CUnitServer::CastSpellUnit | 110 | `Controller.aggression` engine/ai.py:1203, `<module>` engine/magic.py:36, `Spell.can_cast` engine/magic.py:514, `Unit.cast` engine/unit.py:348 |
| 555270 | CUnitServer::SetGait | 26 | `Brain.set_logic` engine/ai.py:502, `Controller.set_gait` engine/ai.py:1136, `<module>` engine/script.py:37, `Unit.move_to` engine/unit.py:620 |
| 555a00 | CUnitServer::vf50 | 80 | `WeaponShape.__init__` eilib/idb.py:237, `<module>` engine/combat.py:8, `Fighter.equip` engine/combat.py:392, `Fighter.set_skill` engine/combat.py:439 |
| 555bd0 | CUnitServer::TryHit | 56 | `<module>` engine/combat.py:31, `attack_roll` engine/combat.py:656 |
| 55fc50 | FUN_0055fc50 | 20 | `DARK_MORNING` engine/unit.py:110, `Unit.__init__` engine/unit.py:243, `darkness` engine/unit.py:1018 |
| 563930 | CWorldLight::LoadLights | 220 | `<module>` engine/lighting.py:4, `fill_hours` engine/lighting.py:136 |
| 564070 | CWorldClient::vf28 | 10 | `Emitter.__init__` engine/particles.py:287 |
| 565e40 | FUN_00565e40 | 59 | `pick_part` client/picking.py:171 |
| 567ab0 | CWorld::LoadMap | 325 | `RemakeApp.start_zone_music` client/main.py:864, `<module>` eilib/zones.py:14, `<module>` engine/music.py:10, `FINAL` engine/music.py:35, +1 |
| 56a990 | FUN_0056a990 | 291 | `UnitView.update` client/scene.py:197, `<module>` engine/ai.py:77 |
| 56ce40 | FUN_0056ce40 | 22 | `FlyingNumbers.add` engine/flyinghp.py:154 |
| 5718b0 | FUN_005718b0 | 45 | `pick_part` client/picking.py:171, `UnitView.__init__` client/scene.py:123, `SceneWidget.units_at` client/scene.py:926 |
| 579830 | FUN_00579830 | 134 | `<module>` eilib/sav.py:10, `PACKED_VERSION` eilib/sav.py:51, `checksum` eilib/sav.py:313, `read_body` eilib/sav.py:325 |
| 579b80 | FUN_00579b80 | 135 | `<module>` eilib/sav.py:10, `checksum` eilib/sav.py:313 |
| 57c460 | FUN_0057c460 | 113 | `<module>` engine/menus.py:37, `main_menu_hour` engine/menus.py:119 |
| 58ccd0 | FUN_0058ccd0 | 221 | `RemakeApp.update_night_senses` client/main.py:1562, `<module>` client/weather.py:11, `WeatherView.strike` client/weather.py:239, `SIGHT_INFRA_SCALE` engine/unit.py:107, +7 |
| 58d3d0 | FUN_0058d3d0 | 83 | `SpellParticles` engine/magic.py:1151, `Possession` engine/magic.py:1261 |
| 58dcd0 | FUN_0058dcd0 | 190 | `<module>` engine/script.py:4, `ScriptWorld.__init__` engine/script.py:381, `<module>` engine/water.py:21, `<module>` engine/weather.py:27, +1 |
| 58e670 | CWorld::MakeSpellEffect | 92 | `<module>` engine/magic.py:64, `HOSTILE_SPELLS` engine/magic.py:348, `Unit.take_spell` engine/unit.py:864, `make_spell_effect` engine/unit.py:1030 |
| 58e850 | CWorld::SpreadSpell | 227 | `<module>` engine/magic.py:66, `UNSPREAD_SPELLS` engine/magic.py:236, `spread` engine/magic.py:1392, `<module>` engine/quickitems.py:6, +5 |
| 590700 | CWorldServer::DeployPartyAtExit | 136 | `DEMO_ZONE` client/demo.py:24, `RemakeApp.enter_zone` client/main.py:917, `<module>` client/zoneexit.py:4, `<module>` eilib/zones.py:12, +4 |
| 5911c0 | FUN_005911c0 | 170 | `RemakeApp.on_kill` client/main.py:2169, `Quest` eilib/idb.py:319, `UnitPrototype.__init__` eilib/idb.py:461, `<module>` engine/experience.py:39, +6 |
| 591740 | CPlayer::GetMercsNumber | 35 | `ClientHost.mercs_number` client/mapscript.py:330, `PARTY_LIMIT` engine/mercenaries.py:55, `<module>` engine/roster.py:18, `Host.mercs_number` engine/script.py:231 |
| 591800 | FUN_00591800 | 217 | `RemakeApp.recalc_mercenaries` client/main.py:1111, `<module>` engine/mercenaries.py:2, `MERCENARIES` engine/mercenaries.py:40, `PARTY_LIMIT` engine/mercenaries.py:55, +3 |
| 591de0 | FUN_00591de0 | 101 | `RemakeApp.hire_mercenary` client/main.py:1190, `RemakeApp.dismiss_mercenary` client/main.py:1226, `RemakeApp.complete_merc_briefing` client/main.py:1256, `<module>` engine/mercenaries.py:2, +3 |
| 592090 | FUN_00592090 | 211 | `BattleScreen.show_briefing` client/hud.py:1011, `RemakeApp.show_briefing` client/main.py:1637, `BRIEFING_COMPLETE_SCRIPT` engine/script.py:107, `ScriptWorld.complete_briefing` engine/script.py:603, +1 |
| 5935c0 | FUN_005935c0 | 141 | `RemakeApp.hide_mercenary_object` client/main.py:1145, `ClientHost.remove_object` client/mapscript.py:300, `Host.remove_object` engine/script.py:203 |
| 593fc0 | CWorldServer::RaiseAlarm | 84 | `<module>` engine/ai.py:57, `ALARMS_COUNT` engine/ai.py:144, `Controller.__init__` engine/ai.py:581, `Controller.raise_alarm` engine/ai.py:700 |
| 594180 | FUN_00594180 | 33 | `<module>` engine/party.py:2, `<module>` engine/roster.py:2, `Host.party` engine/script.py:219, `ScriptWorld.play_music` engine/script.py:987 |
| 5942d0 | FUN_005942d0 | 248 | `RemakeApp.corpse_at` client/main.py:1987, `<module>` engine/loot.py:2 |
| 595ab0 | CWorldServer::AddMob | 32 | `ClientHost.add_mob` client/mapscript.py:208, `<module>` engine/script.py:67 |
| 5aa5f0 | CFigureManager::LoadModelResources | 262 | `GameData.figures_archive` eilib/gamedata.py:95 |
| 5ac1a0 | LoadLooseFigure | 44 | `GameData.loose_figure_archive` eilib/gamedata.py:106 |
| 5ac2f0 | FUN_005ac2f0 | 45 | `GameData.part_figure` eilib/gamedata.py:156 |
| 5ac770 | CFigureManager::LoadSkeleton | 49 | `GameData.parts_tree` eilib/gamedata.py:141 |
| 5ac880 | FUN_005ac880 | 93 | `GameData.part_figure` eilib/gamedata.py:156 |
| 5accf0 | CFigureManager::LoadAnimation | 86 | `GameData.clip` eilib/gamedata.py:197 |
| 5aea60 | CUnitSignal::Raise | 14 | `Controller.wander` engine/ai.py:999, `Signal.raise_level` engine/signals.py:51 |
| 5aea80 | CUnitSignal::Update | 100 | `RemakeApp.on_unit_signal` client/main.py:2198, `<module>` engine/ai.py:63, `<module>` engine/signals.py:2, `IDLE_A` engine/signals.py:18, +2 |
| 5b2d10 | ServerMap::SampleTop | 69 | `<module>` engine/pathing.py:46, `NO_LOW` engine/pathing.py:187, `PassGrid.put_parts` engine/pathing.py:284, `sample_box` engine/pathing.py:734 |
| 5b2e10 | ServerMap::SampleBottom | 58 | `<module>` engine/pathing.py:46, `PassGrid.put_parts` engine/pathing.py:284, `sample_box` engine/pathing.py:735 |
| 5b43c0 | ServerMap::Load | 1293 | `<module>` engine/pathing.py:4, `UNLISTED_TILE_SPEED` engine/pathing.py:159, `SlopeCosts` engine/pathing.py:668, `corner_points` engine/pathing.py:780 |
| 5b6a80 | ServerMap::PutObject | 548 | `<module>` engine/pathing.py:41, `FACE_SAMPLES` engine/pathing.py:190, `PassGrid.put_parts` engine/pathing.py:272, `sample_box` engine/pathing.py:733 |
| 5b7a40 | FUN_005b7a40 | 41 | `Teleport` engine/magic.py:1311, `can_stand` engine/unit.py:1079 |
| 5b8710 | ServerMap::CloseSlopes | 128 | `<module>` engine/pathing.py:26, `PassGrid.close_slopes` engine/pathing.py:368 |
| 5b89f0 | ServerMap::SpreadClosed | 50 | `<module>` engine/pathing.py:24, `spread_closed` engine/pathing.py:834 |
| 5b8b30 | FUN_005b8b30 | 20 | `ClientHost.lever_switched` client/mapscript.py:283, `<module>` engine/lever.py:13 |
| 5b8b80 | FUN_005b8b80 | 13 | `ClientHost.lever_switched` client/mapscript.py:283, `<module>` engine/lever.py:13 |
| 5b8ba0 | ServerMap::PassCode | 162 | `<module>` engine/pathing.py:16, `PassGrid.class_codes` engine/pathing.py:309, `PassGrid.add_obstacles` engine/pathing.py:339 |
| 5ba9d0 | FUN_005ba9d0 | 69 | `RemakeApp.push_units_out` client/main.py:957, `ClientHost.lever_switched` client/mapscript.py:283, `<module>` engine/lever.py:13, `PUSH_OUT_POINTS` engine/pathing.py:166, +4 |
| 5bac10 | ServerMap::OpenTarget | 220 | `<module>` engine/pathing.py:36, `TURN_45` engine/pathing.py:163, `PassGrid.reachable_target` engine/pathing.py:537 |
| 5bb350 | ServerMap::FindPath | 196 | `<module>` engine/pathing.py:35, `target_factor` engine/pathing.py:847 |
| 5bc2a0 | FUN_005bc2a0 | 56 | `WANDER_MAX_SLOPE` engine/ai.py:176, `ground_slope` engine/ai.py:1344 |
| 5c0bd0 | CWorld::UnitsInCircle | 94 | `<module>` engine/magic.py:86, `AreaEffect.circle` engine/magic.py:1047, `<module>` engine/trap.py:20 |
| 5c11d0 | FUN_005c11d0 | 109 | `MapFigure.move` client/mapscript.py:86, `Teleport` engine/magic.py:1311, `<module>` engine/script.py:39, `Host.place_object` engine/script.py:191, +2 |
| 5c2420 | PathWindow::StraightLine | 48 | `<module>` engine/pathing.py:35, `PassGrid.line_is_open` engine/pathing.py:640 |
| 5c46c0 | PathWindow::Smooth | 255 | `<module>` engine/pathing.py:30, `DIAGONAL_END_COST` engine/pathing.py:150, `TURN_STEP_FACTOR` engine/pathing.py:153, `PassGrid.step_tables` engine/pathing.py:415, +3 |
| 5cb0f0 | FUN_005cb0f0 | 80 | `Controller.reset_target` engine/ai.py:1160 |
| 5cb260 | FUN_005cb260 | 74 | `Controller.aggression` engine/ai.py:1202 |
| 5cb670 | CSquad::NearestEnemy | 38 | `<module>` engine/ai.py:31, `Controller.nearest_enemy` engine/ai.py:662 |
| 5cb700 | FUN_005cb700 | 44 | `<module>` engine/ai.py:23, `Controller.think` engine/ai.py:834 |
| 5cb850 | CSquad::Detect | 137 | `<module>` client/mapscript.py:17, `<module>` engine/ai.py:20, `LIFE` engine/ai.py:111, `hears` engine/ai.py:415, +1 |
| 5cbbe0 | CUnitLogic::RememberThreat | 33 | `<module>` engine/ai.py:23, `Brain.__init__` engine/ai.py:445, `Brain.remember_threat` engine/ai.py:541, `Controller.fear` engine/ai.py:1223, +1 |
| 5ce0f0 | CAiSettings::Read | 242 | `GUARD_RADIUS` engine/ai.py:152, `Settings` engine/ai.py:258 |
| 5d0030 | FUN_005d0030 | 357 | `Controller.player_fight` engine/ai.py:798, `Controller.aggression` engine/ai.py:1202 |
| 5d0b70 | CBehaviourGuard::Init | 33 | `<module>` engine/ai.py:39, `GUARD_IDLE_SIGNAL_CHANCE` engine/ai.py:171, `WANDER_TRIES` engine/ai.py:173, `WANDER_FAILS` engine/ai.py:184, +2 |
| 5d1700 | CBehaviourPatrol::Run | 110 | `<module>` engine/ai.py:49, `Controller.patrol` engine/ai.py:1303 |
| 5d1b50 | CBehaviourSentry::Init | 12 | `SENTRY_RADIUS` engine/ai.py:199, `Controller.sentry` engine/ai.py:1254 |
| 5d1de0 | FUN_005d1de0 | 16 | `Controller.follow_behaviour` engine/ai.py:1233 |
| 5d1e90 | FUN_005d1e90 | 40 | `<module>` engine/ai.py:24, `THREAT_ENEMY` engine/ai.py:118, `SUSPICION_FAR` engine/ai.py:127, `Controller.attacked` engine/ai.py:682, +1 |
| 5d21a0 | CUnitLogic::ToSuspicion | 61 | `<module>` engine/ai.py:28, `Controller.suspect` engine/ai.py:898, `Controller.calm` engine/ai.py:945 |
| 5d2390 | FUN_005d2390 | 82 | `Controller.aggression` engine/ai.py:1201 |
| 5d26d0 | CUnitLogic::ToAggression | 16 | `Controller.aggression` engine/ai.py:1203 |
| 5d2710 | CUnitLogic::BackFromAggression | 24 | `<module>` engine/ai.py:26, `Brain.forget_threat` engine/ai.py:552, `Controller.aggression` engine/ai.py:1203 |
| 5d2850 | CBehaviourAttack::ctor | 15 | `<module>` engine/ai.py:6, `Brain.set_logic` engine/ai.py:489, `Controller.standard_behaviours` engine/ai.py:1187, `Controller.fear` engine/ai.py:1221, +1 |
| 5d2960 | CBehaviourAttack::Weigh | 219 | `<module>` engine/ai.py:6, `RANGED_HEALTH_PART` engine/ai.py:161, `Controller.fear` engine/ai.py:1222, `Controller.keep_away` engine/ai.py:1274 |
| 5d3760 | CUnitLogic::SetPlayerBehaviour | 17 | `Controller.player_behaviour` engine/ai.py:1241 |
| 5d3f50 | FUN_005d3f50 | 104 | `PLAYER_CALL_FOR_HELP` engine/ai.py:158, `Controller.player_fight` engine/ai.py:798 |
| 5d4860 | FUN_005d4860 | 91 | `Controller.player_behaviour` engine/ai.py:1241 |
| 5d4df0 | CUnitLogic::ClearBehaviours | 37 | `Controller.clear_behaviours` engine/ai.py:1143, `Controller.stop_orders` engine/ai.py:1175, `Controller.player_behaviour` engine/ai.py:1241, `<module>` engine/script.py:36 |
| 5d53b0 | CUnitLogic::UpdateState | 241 | `<module>` engine/ai.py:28, `GUARD_WEIGHT` engine/ai.py:124, `ALARM_ON_SIGHT` engine/ai.py:139, `Brain.arm_alarm` engine/ai.py:507, +4 |
| 5d5940 | CUnitLogic::OnAlarm | 18 | `<module>` engine/ai.py:57, `Controller.raise_alarm` engine/ai.py:701 |
| 5d5990 | CUnitLogic::CallForHelp | 81 | `<module>` engine/ai.py:32, `Controller.attacked` engine/ai.py:675, `Controller.aggression` engine/ai.py:1203 |
| 5d6230 | CUnitLogic::SetGuard | 47 | `Controller.guard` engine/ai.py:1260 |
| 5d6630 | CUnitLogic::Init | 301 | `<module>` engine/ai.py:4, `PEACEFUL_MODELS` engine/ai.py:133, `PATROL_CELL` engine/ai.py:147, `UnitLogic.patrol_route` engine/ai.py:311, +3 |
| 5d6f90 | FUN_005d6f90 | 190 | `Controller.follow` engine/ai.py:1086 |
| 5d7520 | CUnitLogic::DoPointActions | 75 | `<module>` engine/ai.py:50, `Controller.patrol` engine/ai.py:1303 |
| 5d76a0 | FUN_005d76a0 | 49 | `<module>` engine/ai.py:24, `THREAT_ENEMY` engine/ai.py:118, `Controller.think` engine/ai.py:865 |
| 5d7790 | CUnitLogic::ClearOrder | 14 | `Controller.stop_orders` engine/ai.py:1175, `Controller.player_behaviour` engine/ai.py:1241, `<module>` engine/script.py:36 |
| 5d7ae0 | CUnitLogic::OrderMove | 11 | `ORDER_NONE` engine/ai.py:194, `Controller.order_move` engine/ai.py:1106, `<module>` engine/script.py:35 |
| 5d7b30 | CUnitLogic::OrderTurn | 11 | `ORDER_NONE` engine/ai.py:194, `Controller.order_turn` engine/ai.py:1118, `<module>` engine/script.py:35 |
| 5d7b50 | FUN_005d7b50 | 59 | `Controller.aggression` engine/ai.py:1203 |
| 5d7c30 | FUN_005d7c30 | 12 | `Controller.reset_target` engine/ai.py:1160 |
| 5d7e40 | CUnitLogic::OrderFollow | 23 | `ORDER_NONE` engine/ai.py:194, `Controller.order_follow` engine/ai.py:1128, `<module>` engine/script.py:36 |
| 5d7e90 | CUnitLogic::OnAttacked | 41 | `<module>` engine/ai.py:59, `ALARM_ON_SIGHT` engine/ai.py:139, `Brain.arm_alarm` engine/ai.py:507 |
| 5db3f0 | FUN_005db3f0 | 251 | `<module>` client/movie.py:3, `STOP_KEYS` client/movie.py:28, `MoviePlayer.__init__` client/movie.py:63, `MoviePlayer.play` client/movie.py:87, +4 |
| 5e1870 | FUN_005e1870 | 39 | `<module>` engine/flyinghp.py:3, `FlyingNumbers.add` engine/flyinghp.py:154 |
| 5e18e0 | FUN_005e18e0 | 524 | `<module>` client/flyinghp.py:2, `FlyingNumbersView.draw` client/flyinghp.py:71, `RemakeApp.update_flying_numbers` client/main.py:1506, `HEAD_PARTS` client/scene.py:65, +5 |
| 5e52e0 | CCursor::Update | 83 | `pick_part` client/picking.py:173, `<module>` client/ui.py:10 |
| 5e6fa0 | CInterfaceElement::SetScreenPosition | 18 | `interface_point` client/gl.py:348 |
| 5e70e0 | CInterfaceElement::SetSprite | 241 | `CHECKBOX_TEXELS` engine/difficulty.py:74, `<module>` engine/hud.py:7, `quad` engine/hud.py:299 |
| 5e7680 | SetTiles | 305 | `tiles` engine/hud.py:387, `<module>` engine/inventory.py:10 |
| 5e7d60 | CInterfaceElement::SetFrame | 275 | `FRAME_TEXELS` client/options.py:52, `frame_items` client/options.py:297, `bevel_items` client/ui.py:325, `<module>` engine/hud.py:11, +4 |
| 5e8450 | SetHollowFrame | 288 | `hollow_frame` engine/hud.py:367, `<module>` engine/inventory.py:10 |
| 5e8c60 | CInterfaceElement::SetDisc | 273 | `<module>` engine/hud.py:13, `disc` engine/hud.py:403 |
| 5e91f0 | CInterfaceElement::SetRing | 328 | `<module>` engine/hud.py:14, `ring` engine/hud.py:421 |
| 5e99d0 | CInterfaceElement::SetTriangle | 227 | `<module>` engine/hud.py:15, `triangle` engine/hud.py:440 |
| 5ec020 | MakeItemTexture | 149 | `CampView.slot_items` client/camp.py:361, `<module>` engine/inventory.py:19, `item_texture` engine/inventory.py:367 |
| 5eca10 | DrawItemInSlot | 444 | `BattleScreen.add_item_model` client/hud.py:1208, `<module>` engine/inventory.py:11, `ITEM_PIXELS` engine/inventory.py:109, `item_rotation` engine/inventory.py:381 |
| 5f7e60 | FUN_005f7e60 | 10 | `SOUND_MODE` client/camp.py:84, `RemakeApp.ui_sound` client/main.py:540 |
| 5f8070 | CInterface::OnStringEvent | 117 | `RemakeApp.autosave` client/main.py:432, `<module>` client/mapscript.py:5, `<module>` engine/tutorial.py:4 |
| 5fc450 | CInterfaceModel::SetScreenPosition | 20 | `INTERFACE_VIEW` client/gl.py:30, `interface_point` client/gl.py:347, `hud_pixel` engine/inventory.py:396 |
| 5fc4c0 | CInterfaceModel::ShowPart | 19 | `FigureModel.__init__` engine/figure.py:53 |
| 5fc730 | CInterfaceRegion::SetTooltip | 63 | `Region` engine/hud.py:263 |
| 5fc9e0 | CInterfaceRegion::SetRect | 23 | `rect_region` engine/hud.py:447 |
| 5fca70 | CInterfaceRegion::AddPoint | 105 | `Region` engine/hud.py:262 |
| 5ff3c0 | CInterfaceWindow::NewElement | 36 | `Element` engine/hud.py:221 |
| 5ff830 | CInterfaceWindow::NewRegion | 93 | `Region` engine/hud.py:262 |
| 5fffb0 | CInterfaceWindow::CopyTextureToCell | 136 | `portrait_face_texture` engine/hud.py:734 |
| 607320 | FUN_00607320 | 157 | `RemakeApp.talk_to` client/main.py:1976, `<module>` engine/talk.py:3, `topics` engine/talk.py:57 |
| 608410 | FUN_00608410 | 213 | `ScriptWorld.briefing_variable` engine/script.py:591 |
| 608b20 | FUN_00608b20 | 49 | `RemakeApp.talk_to` client/main.py:1976, `<module>` client/talk.py:6, `<module>` engine/talk.py:3, `ROWS_TOP` engine/talk.py:24, +1 |
| 6111e0 | CCampWindow::DealButton | 218 | `Host.fix_items` engine/script.py:248, `<module>` engine/trade.py:18, `repair_item` engine/trade.py:51 |
| 620340 | CInventoryStrip::BuildElements | 152 | `<module>` engine/inventory.py:2, `END_DEPTH` engine/inventory.py:69, `InventoryStrip.elements` engine/inventory.py:234, `InventoryStrip.regions` engine/inventory.py:283 |
| 620a10 | CInventoryStrip::UpdateScroll | 117 | `<module>` engine/inventory.py:13, `SCROLL_REPEAT_S` engine/inventory.py:97, `InventoryStrip.update` engine/inventory.py:186 |
| 620dd0 | CInventoryStrip::OnMouseDown | 44 | `<module>` engine/inventory.py:15, `InventoryStrip.press_arrow` engine/inventory.py:171 |
| 621120 | CInventoryStrip::Create | 55 | `<module>` engine/camp.py:4, `<module>` engine/inventory.py:2, `LEFT_ARROW` engine/inventory.py:75, `InventoryStrip.regions` engine/inventory.py:283 |
| 6213e0 | CInventoryStrip::SetMode | 48 | `<module>` engine/inventory.py:9, `MODE_ITEMS` engine/inventory.py:81, `InventoryStrip.regions` engine/inventory.py:283 |
| 621530 | CInventoryStrip::FillList | 172 | `CATEGORY_MATERIALS` engine/inventory.py:88, `InventoryStrip.shown_items` engine/inventory.py:154 |
| 6218b0 | CInventoryStrip::ItemCategory | 34 | `CATEGORY_MATERIALS` engine/inventory.py:88, `item_category` engine/inventory.py:328 |
| 621b10 | CInventoryStrip::PlaceSlots | 137 | `<module>` engine/inventory.py:6, `END_ARROW_SHIFT` engine/inventory.py:57, `InventoryStrip.elements` engine/inventory.py:234, `InventoryStrip.slots` engine/inventory.py:267 |
| 622d70 | FUN_00622d70 | 178 | `<module>` engine/camp.py:3, `INFO_RECT` engine/camp.py:48, `INFO_TEXT_KEYS` engine/camp.py:88, `Camp.stat_lines` engine/camp.py:309 |
| 624ed0 | FUN_00624ed0 | 196 | `SKILL_ROW_HEIGHT` client/camp.py:93, `CampView.skills_items` client/camp.py:763, `CampView.on_skill_click` client/camp.py:817, `<module>` engine/camp.py:3, +4 |
| 6268e0 | FUN_006268e0 | 120 | `SKILL_ROW_HEIGHT` client/camp.py:93, `CampView.skills_items` client/camp.py:764, `CampView.on_perk_click` client/camp.py:829, `<module>` engine/skills.py:3 |
| 62ac70 | CEscMenuWindow::Deactivate | 10 | `EscMenu.ask_quit` client/menu.py:433, `<module>` engine/menus.py:13, `<module>` engine/messagebox.py:5 |
| 62d0c0 | FUN_0062d0c0 | 56 | `CampView.equip` client/camp.py:587, `CampView.unequip` client/camp.py:613, `SLOT_WEAPON` engine/camp.py:70 |
| 62d1d0 | FUN_0062d1d0 | 29 | `RemakeApp.autosave_exit` client/main.py:446, `QUICK_SLOT_NAME` engine/saveload.py:64 |
| 632900 | CMainMenuWindow::OnMessage | 54 | `MainMenu.open_difficulty` client/menu.py:184, `MainMenu.start_new_game` client/menu.py:223 |
| 632ad0 | CMainMenuWindow::Update | 92 | `<module>` client/weather.py:15, `STORM_PERIOD_MS` client/weather.py:45, `WeatherView.tick_storm` client/weather.py:265 |
| 633770 | FUN_00633770 | 249 | `<module>` client/difficulty.py:2, `PANEL_COLOR` client/difficulty.py:23, `DifficultyWindow.update` client/difficulty.py:69, `RemakeApp.__init__` client/main.py:194, +11 |
| 634750 | FUN_00634750 | 224 | `QuitWindow` client/menu.py:236, `<module>` client/saveload.py:3, `<module>` engine/messagebox.py:2, `OK_RECT` engine/messagebox.py:33, +3 |
| 635210 | CBattleWindow::Activate | 531 | `RemakeApp.new_scene` client/main.py:385, `<module>` engine/hud.py:2 |
| 6360b0 | FUN_006360b0 | 25 | `RemakeApp.autosave` client/main.py:433, `QUICK_SLOT_NAME` engine/saveload.py:63 |
| 636990 | FUN_00636990 | 197 | `RemakeApp.update_quest_log` client/main.py:1528, `<module>` engine/questlog.py:2, `change_message` engine/questlog.py:83 |
| 63ba70 | CSpellSlotsWindow::ctor | 69 | `QUICK_SLOT_RECTS` client/hud.py:74, `BattleScreen.update_quick_items` client/hud.py:602 |
| 640020 | CPortraitsWindow::AddPortrait | 175 | `<module>` engine/hud.py:25, `PORTRAIT_HEAD_Y` engine/hud.py:141, `portrait_slots` engine/hud.py:670, `portrait_face_texture` engine/hud.py:734 |
| 6407a0 | CPortraitsWindow::Rebuild | 336 | `BattleScreen.update_portraits` client/hud.py:422, `HEAD_MODEL_FORMAT` engine/hud.py:169 |
| 641ee0 | CMiniMapWindow::Create | 317 | `<module>` engine/hud.py:22, `MINIMAP_CENTER` engine/hud.py:183, `MARK_OWN` engine/hud.py:202, `minimap_panel` engine/hud.py:608 |
| 642b00 | CMiniMapWindow::Update | 102 | `MINIMAP_CENTER` engine/hud.py:183, `Minimap.zoom_by` engine/hud.py:771 |
| 642e40 | CMiniMapWindow::BuildMap | 599 | `MINIMAP_CENTER` engine/hud.py:183, `Minimap.disc` engine/hud.py:786, `clip_polygon` engine/hud.py:857, `compass_angle` engine/hud.py:879 |
| 643d80 | CMiniMapWindow::BuildMarkers | 479 | `BattleScreen.update_minimap` client/hud.py:204, `<module>` engine/ai.py:77, `MINIMAP_CENTER` engine/hud.py:183, `Minimap.marks` engine/hud.py:824 |
| 6456a0 | CInfoWindow::Create | 225 | `CampView.add_figure` client/camp.py:205, `BattleScreen.update_portraits` client/hud.py:423, `BattleScreen.add_info_figure` client/hud.py:493, `<module>` engine/hud.py:20, +4 |
| 645f40 | CInfoWindow::PlaceElements | 69 | `SPELL_ICON_RECT` engine/hud.py:160, `spell_icon_rect` engine/hud.py:324 |
| 646b50 | CInfoWindow::DrawSpellEffects | 216 | `INFO_GENERAL` engine/hud.py:65, `SPELL_ICON_RECT` engine/hud.py:160, `texture_quad` engine/hud.py:315, `spell_icons` engine/hud.py:333 |
| 647000 | CInfoWindow::DrawAttributes | 218 | `INFO_GENERAL` engine/hud.py:65, `INFO_ATTRIBUTES_LINE` engine/hud.py:78, `info_texts` engine/hud.py:578 |
| 6477a0 | CInfoWindow::DrawGeneral | 127 | `INFO_GENERAL` engine/hud.py:65, `INFO_HEALTH_NUMBER` engine/hud.py:75, `info_panel` engine/hud.py:498 |
| 6484e0 | CPauseDial::Create | 98 | `<module>` engine/hud.py:23, `DIAL_INNER_RADIUS` engine/hud.py:110, `pause_dial` engine/hud.py:630 |
| 6488a0 | CPauseDial::Update | 143 | `<module>` client/questmap.py:7, `DIAL_ARROW_RADIUS` engine/hud.py:115, `HOUR_TURN_OFFSET` engine/hud.py:124, `pause_dial` engine/hud.py:630 |
| 649150 | CModeDial::Create | 102 | `MODE_CRAWL` client/keys.py:34, `<module>` engine/hud.py:24, `DIAL_INNER_RADIUS` engine/hud.py:110, `mode_dial` engine/hud.py:649 |
| 6497d0 | FUN_006497d0 | 66 | `Unit.__init__` engine/unit.py:166 |
| 649ab0 | CTextWindow::Create | 81 | `BattleScreen.__init__` client/hud.py:115, `<module>` engine/hud.py:21 |
| 64cbd0 | FUN_0064cbd0 | 504 | `<module>` client/globalmap.py:2, `<module>` client/questmap.py:2, `Allod` eilib/zones.py:44, `<module>` engine/globalmap.py:2, +2 |
| 650380 | FUN_00650380 | 153 | `MainMenu.open_options` client/menu.py:206, `<module>` client/options.py:2, `PLAQUE_PARTS` client/options.py:63, `OptionsView.on_mouse` client/options.py:174, +6 |
| 651310 | FUN_00651310 | 107 | `<module>` client/options.py:4, `BRIGHT_COLOR` client/options.py:38, `OptionsView.group_items` client/options.py:196, `FRAME_RECT` engine/options.py:143, +1 |
| 6515c0 | FUN_006515c0 | 129 | `<module>` engine/options.py:11, `option_label` engine/options.py:196 |
| 651900 | FUN_00651900 | 151 | `<module>` client/options.py:5, `OptionsView.option_items` client/options.py:211, `<module>` engine/options.py:11, `FRAME_RECT` engine/options.py:140, +2 |
| 651d30 | FUN_00651d30 | 120 | `OptionsView.on_mouse` client/options.py:174, `Dialog.tip_at` engine/options.py:448 |
| 652460 | FUN_00652460 | 62 | `POWER_STEP` engine/options.py:62, `OPTIONS` engine/options.py:68, `keyboard_power` engine/options.py:229, `delay_ms` engine/tooltip.py:128 |
| 653a10 | FUN_00653a10 | 70 | `QUICK_SLOT_NAME` engine/saveload.py:62, `slot_title` engine/saveload.py:111 |
| 658dd0 | CTutorialWindow::Open | 104 | `<module>` engine/tutorial.py:2, `TutorialWindow.show_last` engine/tutorial.py:149 |
| 65b6f0 | FUN_0065b6f0 | 33 | `<module>` client/options.py:5, `slider_items` client/options.py:311, `sprite_item` client/ui.py:308, `SLIDER_X` engine/options.py:161, +2 |
| 65fda0 | FUN_0065fda0 | 243 | `SavedUnit` eilib/sav.py:473, `_read_unit` eilib/sav.py:574 |
| 6617b0 | FUN_006617b0 | 50 | `scenario_parties` eilib/sav.py:613 |
| 661880 | FUN_00661880 | 173 | `read_party` eilib/sav.py:593 |
| 661bd0 | FUN_00661bd0 | 90 | `SavedParty` eilib/sav.py:502, `read_party` eilib/sav.py:593 |
| 662170 | CParty::FindUnit | 23 | `UnitParty.unit` engine/roster.py:91 |
| 667e80 | FUN_00667e80 | 430 | `RemakeApp.spawn_objects` client/main.py:714, `COLUMN_MODELS` engine/menus.py:30 |
| 668b30 | FUN_00668b30 | 55 | `<module>` engine/ai.py:72, `Controller.update_player_sight` engine/ai.py:744, `<module>` engine/music.py:18 |
| 66a440 | FUN_0066a440 | 61 | `RemakeApp.on_experience` client/main.py:1518, `reward_units` engine/experience.py:507, `<module>` engine/flyinghp.py:8, `INTELLIGENCE_FACTOR` engine/flyinghp.py:63, +1 |
| 66e3f0 | CPlayer::UpdateDanger | 149 | `Controller.in_danger` engine/ai.py:628, `<module>` engine/music.py:19 |
| 66f2d0 | CPlayer::LeaveToZone | 59 | `RemakeApp.leave_to_zone` client/main.py:934, `<module>` client/zoneexit.py:3, `Host.leave_to_zone` engine/script.py:281 |
| 679240 | CEffectParticle::Attach | 12 | `unit_particles` engine/magic.py:563, `<module>` engine/sources.py:12 |
| 679bf0 | FUN_00679bf0 | 14 | `AreaEffect.start_visual` engine/magic.py:934 |
| 67a8b0 | CEffectLightning::vf06 | 46 | `<module>` engine/lightning.py:7, `ARC_TARGET_HEIGHT` engine/magic.py:263, `LightningArc` engine/magic.py:1080 |
| 67aa70 | CEffectArrow::Home | 72 | `MISSILE_STEP` engine/magic.py:212, `<module>` engine/missiles.py:9, `Arrow.update_velocity` engine/missiles.py:89 |
| 67ac30 | CEffectArrow::SetTarget | 11 | `make_spell_effect` engine/unit.py:1059 |
| 67bde0 | CEffectArrowFigure::vf06 | 34 | `RemakeApp.update_arrows` client/main.py:2036, `<module>` engine/missiles.py:13 |
| 67e690 | CEffectSpell::Update | 55 | `<module>` engine/magic.py:46, `AreaEffect` engine/magic.py:738 |
| 67ec70 | FUN_0067ec70 | 47 | `Teleport` engine/magic.py:1309 |
| 67edc0 | FUN_0067edc0 | 22 | `AreaEffect.set_light` engine/magic.py:892 |
| 67ef00 | FUN_0067ef00 | 23 | `AreaEffect.set_light` engine/magic.py:892 |
| 67ef80 | CEffectSpell::End | 53 | `<module>` engine/magic.py:47, `Teleport.tick` engine/magic.py:1367 |
| 67f130 | SpellEffect::MakeMissile | 49 | `make_spell_effect` engine/unit.py:1059 |
| 67f220 | CEffectSpell::MakeParticle | 53 | `AreaEffect.emitter` engine/magic.py:899 |
| 67f3d0 | SpellEffect::MakeLightningArc | 51 | `<module>` engine/magic.py:73, `ARC_POWER` engine/magic.py:260, `LightningArc` engine/magic.py:1079, `make_spell_effect` engine/unit.py:1072 |
| 67f620 | FUN_0067f620 | 66 | `<module>` engine/magic.py:109, `SOUND_NEAR` engine/magic.py:282, `circle_sound` engine/magic.py:622, `SpellParticles.__init__` engine/magic.py:1174 |
| 67f780 | SpellEffect::MakeLight | 39 | `<module>` engine/magic.py:95, `SpellLight` engine/magic.py:672, `AreaEffect.set_light` engine/magic.py:892 |
| 67f830 | SpellEffect::Start | 1147 | `<module>` engine/magic.py:46, `SPEED_TICKS` engine/magic.py:215, `VISUAL_POWER` engine/magic.py:239, `FIREBALL_HEIGHT` engine/magic.py:246, +30 |
| 681b70 | SpellEffect::Update | 346 | `RemakeApp.update_arrows` client/main.py:2067, `<module>` engine/magic.py:47, `FIREBALL_SMOKE_SCALE` engine/magic.py:243, `MISSILE_TRAILS` engine/magic.py:249, +7 |
| 682610 | CEffectSpell::Lasting | 558 | `Controller.update_player_sight` engine/ai.py:746, `Controller.spell_sight` engine/ai.py:769, `<module>` engine/magic.py:47, `ARC_POWER` engine/magic.py:260, +18 |
| 683570 | SpellEffect::HitUnit | 21 | `Unit.take_spell_blow` engine/unit.py:493 |
| 6835c0 | SpellEffect::HitCircle | 71 | `<module>` engine/magic.py:79, `FIREBALL_SPELL` engine/magic.py:219, `AreaEffect.circle` engine/magic.py:1047 |
| 6836f0 | SpellEffect::HitWall | 86 | `<module>` engine/magic.py:83, `FIREBALL_SPELL` engine/magic.py:219, `WALL_HALF_WIDTH` engine/magic.py:233, `AreaEffect.strip` engine/magic.py:1056 |
| 683910 | SpellEffect::MakeHit | 62 | `<module>` engine/magic.py:29, `FIRE` engine/magic.py:154, `Spell.hit` engine/magic.py:472, `AreaEffect` engine/magic.py:740 |
| 683ab0 | CSpellManager::ReadModifier | 18 | `<module>` eilib/idb.py:41, `SpellModifier` eilib/idb.py:631 |
| 683b40 | CSpellManager::ReadPrototype | 37 | `<module>` eilib/idb.py:40, `SpellPrototype` eilib/idb.py:585 |
| 683d00 | CSpellManager::ReadTemplate | 65 | `<module>` eilib/idb.py:43, `SpellTemplate` eilib/idb.py:653 |
| 6841a0 | CSpellConstructor::BeginConstruction | 31 | `make_spell` engine/magic.py:1474 |
| 684210 | CSpellConstructor::EndConstruction | 16 | `make_spell` engine/magic.py:1474 |
| 684260 | CSpellConstructor::AddModifier | 19 | `make_spell` engine/magic.py:1474 |
| 684c10 | FUN_00684c10 | 43 | `parse_spells` eilib/idb.py:802 |
| 684eb0 | FUN_00684eb0 | 43 | `parse_spells` eilib/idb.py:802 |
| 684fa0 | FUN_00684fa0 | 43 | `parse_spells` eilib/idb.py:802 |
| 688ff0 | CSpell::FromString | 125 | `ScriptWorld.cast_spell` engine/script.py:1007, `<module>` engine/trap.py:8, `parse_spell` engine/trap.py:186 |
| 68b480 | CSpell::PassesFilter | 176 | `<module>` engine/magic.py:23, `RACE_MODELS` engine/magic.py:198, `Spell.affects` engine/magic.py:456 |
| 68be80 | CSpell::AllowsModifier | 28 | `<module>` engine/magic.py:20, `MODIFIER_FLAGS` engine/magic.py:206, `Spell.allows` engine/magic.py:395 |
| 68bf20 | CSpell::Build | 278 | `<module>` engine/magic.py:4, `Spell` engine/magic.py:370 |
| 692fd0 | CTextureManager::DrawLayers | 59 | `<module>` engine/equipment.py:14, `compose` engine/equipment.py:364 |
| 693140 | CTextureManager::ComposeLayers | 320 | `compose` engine/equipment.py:363 |
| 69f5e0 | CMaterial::Serialize | 23 | `<module>` eilib/idb.py:10, `Material` eilib/idb.py:168 |
| 69f6c0 | CItemShape::Serialize | 25 | `<module>` eilib/idb.py:11, `Shape` eilib/idb.py:194, `WeaponShape` eilib/idb.py:222, `ArmorShape` eilib/idb.py:249 |
| 69f7c0 | CWeaponShape::Serialize | 20 | `<module>` eilib/idb.py:11, `WeaponShape` eilib/idb.py:222 |
| 69f880 | CArmorShape::Serialize | 16 | `<module>` eilib/idb.py:12, `ArmorShape` eilib/idb.py:249 |
| 6a0fb0 | CItemWeapon::PartNames | 1007 | `WEAPON_SWORD` eilib/idb.py:99, `<module>` engine/equipment.py:10, `weapon_parts` engine/equipment.py:291 |
| 6a2d00 | FUN_006a2d00 | 12 | `WeaponShape.__init__` eilib/idb.py:236 |
| 6a2d50 | FUN_006a2d50 | 12 | `WeaponShape.__init__` eilib/idb.py:236 |
| 6a2da0 | CItemWeapon::RollDamage | 32 | `WeaponShape.__init__` eilib/idb.py:230, `<module>` engine/combat.py:15 |
| 6a5d20 | CItemManager::BuildItem | 239 | `Material.__init__` eilib/idb.py:177, `Shape.__init__` eilib/idb.py:211, `CLASS_SPELL` eilib/sav.py:413, `<module>` engine/combat.py:14, +7 |
| 6a7900 | FUN_006a7900 | 43 | `parse_items` eilib/idb.py:746 |
| 6a79f0 | FUN_006a79f0 | 43 | `parse_items` eilib/idb.py:746 |
| 6a7af0 | FUN_006a7af0 | 43 | `parse_items` eilib/idb.py:746 |
| 6acfc0 | CItemManager::CreateItem | 72 | `make_item` engine/equipment.py:258 |
| 6b4df0 | CMusicSystem::SetZoneMode | 59 | `<module>` engine/music.py:9, `MusicSystem.set_zone` engine/music.py:145 |
| 6b4f50 | CMusicSystem::SetBriefing | 52 | `<module>` engine/music.py:10, `MusicSystem.set_briefing` engine/music.py:161 |
| 6b5040 | CMusicSystem::SetConstructor | 52 | `<module>` engine/music.py:10, `MusicSystem.set_constructor` engine/music.py:170 |
| 6b5130 | CMusicSystem::SetMainMenu | 28 | `MusicSystem.set_main_menu` engine/music.py:179 |
| 6b51b0 | CMusicSystem::SetCredits | 28 | `MusicSystem.set_credits` engine/music.py:187 |
| 6b5230 | CMusicSystem::PlayScriptTrack | 39 | `<module>` engine/music.py:12, `MusicSystem.play_script` engine/music.py:195 |
| 6b5310 | CMusicSystem::vf06 | 141 | `<module>` engine/music.py:11, `WAIT_MS` engine/music.py:45, `MusicSystem.update` engine/music.py:204 |
| 6b7cd0 | FUN_006b7cd0 | 131 | `AudioPlayer.__init__` client/audio.py:66, `AudioPlayer.take_world_channel` client/audio.py:99, `CHANNELS` engine/sounds.py:60, `take_channel` engine/sounds.py:312 |
| 6b7f80 | FUN_006b7f80 | 100 | `AudioPlayer.distance_to` client/audio.py:160, `<module>` engine/magic.py:112, `<module>` engine/sounds.py:37, `height_factor` engine/sounds.py:329, +1 |
| 6b8210 | FUN_006b8210 | 192 | `AudioPlayer.set_ambient` client/audio.py:228, `<module>` engine/ambient.py:9 |
| 6b85c0 | CSoundSystem::PlayAt | 165 | `<module>` engine/ambient.py:11, `SHOT_NEAR` engine/ambient.py:66, `<module>` engine/sounds.py:36, `PRIORITY_LOW` engine/sounds.py:56, +2 |
| 6b90c0 | FUN_006b90c0 | 98 | `AudioPlayer.set_ambient` client/audio.py:228, `<module>` engine/ambient.py:9 |
| 6b9460 | FUN_006b9460 | 59 | `AudioPlayer.set_ambient` client/audio.py:228, `<module>` engine/ambient.py:9 |
| 6c30a0 | FUN_006c30a0 | 51 | `SceneWidget.set_water_levels` client/scene.py:664 |
| 6c57a0 | CTerrain::MakeWavePhases | 135 | `<module>` engine/water.py:12, `random_phases` engine/water.py:145 |
| 6c8d00 | CMap::TileTypeAt | 47 | `<module>` engine/sounds.py:25, `OUTSIDE_TILE_TYPE` engine/terrain.py:26, `Terrain.tile_type_at` engine/terrain.py:77 |
| 6cb350 | FUN_006cb350 | 389 | `RemakeApp.update_ambient` client/main.py:1541, `<module>` engine/ambient.py:2, `material_of` engine/ambient.py:129 |
| 6cbbc0 | FUN_006cbbc0 | 29 | `<module>` engine/ambient.py:16, `day_time` engine/ambient.py:97 |
| 6cc7b0 | CTerrain::SetWaterLevel | 28 | `SceneWidget.set_water_levels` client/scene.py:663 |
| 6d43d0 | FUN_006d43d0 | 50 | `<module>` engine/water.py:18, `WaterLevels.set_level` engine/water.py:113 |
| 6d44a0 | FUN_006d44a0 | 53 | `<module>` engine/water.py:21, `WaterLevels.tick` engine/water.py:122 |
| 6d8a00 | CUnit::SayScenarioPhrase | 64 | `ClientHost.say` client/mapscript.py:386, `<module>` eilib/idb.py:47, `Acks.phrase` eilib/idb.py:363, `<module>` engine/acks.py:2, +1 |
| 6db3f0 | FUN_006db3f0 | 43 | `parse_acks` eilib/idb.py:768 |
| 6e3740 | FUN_006e3740 | 182 | `Acks` eilib/idb.py:346 |

## Частично перенесённые функции (`partial`) — 24

| Адрес | Функция C | Строк | Python |
|---|---|---:|---|
| 404860 | FUN_00404860 | 31 | `END_OF_OBJECTS` eilib/sav.py:410, `_skip_bodies` eilib/sav.py:560 |
| 456440 | FUN_00456440 | 658 | `RemakeApp.move_camera` client/main.py:1574 |
| 50f900 | FUN_0050f900 | 61 | `PLACE_SIZE` eilib/sav.py:637, `object_places` eilib/sav.py:645 |
| 51c3d0 | FUN_0051c3d0 | 171 | `<module>` engine/woundskin.py:2, `wounded` engine/woundskin.py:131 |
| 52ef90 | shared_vf_CFakeUnit49_CUnit44 | 283 | `FlyingNumbersView.__init__` client/flyinghp.py:36, `FlyingNumbersView.watch` client/flyinghp.py:50, `RemakeApp.update_flying_numbers` client/main.py:1506, `<module>` engine/flyinghp.py:5, +1 |
| 5321c0 | shared_vf_CFakeUnit10_CUnit5 | 294 | `find_unit_place` eilib/sav.py:663 |
| 5341a0 | CUnitClientSpecific::vf06 | 128 | `ACTION_WEAPON_BITS` engine/equipment.py:52, `weapon_hidden_parts` engine/equipment.py:319 |
| 537360 | CUnitClientSpecific::vf31 | 19 | `SceneWidget.update_wounds` client/scene.py:726, `<module>` engine/woundskin.py:2 |
| 548520 | FUN_00548520 | 87 | `RemakeApp.on_cant_run` client/main.py:2190, `RUN_LIMIT_GROUPS` engine/acks.py:72, `Unit.__init__` engine/unit.py:204, `Unit.move_to` engine/unit.py:620, +2 |
| 5536b0 | CUnitServer::vf45 | 102 | `RemakeApp.update_night_senses` client/main.py:1562, `sees` engine/ai.py:378, `hears` engine/ai.py:415, `NOISE_CRAWL` engine/unit.py:95, +5 |
| 5641f0 | CWorldClient::vf06 | 265 | `RemakeApp.on_frame` client/main.py:1436, `AUTOSAVE_TICK` engine/saveload.py:74, `autosave_due` engine/saveload.py:121 |
| 5bc3e0 | FUN_005bc3e0 | 101 | `<module>` engine/ai.py:17, `SIGHT_STEP` engine/ai.py:115, `line_of_sight` engine/ai.py:361, `Controller.look_actions` engine/ai.py:1059 |
| 5ca5a0 | FUN_005ca5a0 | 163 | `<module>` engine/ai.py:14, `sees` engine/ai.py:377, `Brain.sees` engine/ai.py:535, `Controller.look` engine/ai.py:645, +3 |
| 5caa00 | FUN_005caa00 | 167 | `<module>` engine/ai.py:19, `loses_sight` engine/ai.py:401, `Brain.__init__` engine/ai.py:443, `Controller.look` engine/ai.py:645, +1 |
| 5cc9d0 | FUN_005cc9d0 | 102 | `<module>` engine/ai.py:25, `THREAT_SOUND` engine/ai.py:121, `Controller.look` engine/ai.py:647 |
| 5d4170 | FUN_005d4170 | 228 | `Controller.player_fight` engine/ai.py:796, `Unit.__init__` engine/unit.py:166 |
| 5d5b40 | FUN_005d5b40 | 105 | `GUARD_STAY_TIME` engine/ai.py:169, `LOOK_TICKS_PER_STAY` engine/ai.py:186, `Controller.wander` engine/ai.py:1000, `Controller.look_actions` engine/ai.py:1057 |
| 5d7f80 | FUN_005d7f80 | 154 | `WANDER_TRIES` engine/ai.py:173, `WANDER_PAUSE_PER_DISTANCE` engine/ai.py:181, `Controller.wander` engine/ai.py:1000, `Controller.wander_to` engine/ai.py:1024 |
| 5f78c0 | FUN_005f78c0 | 116 | `SHOT_KEYS` client/main.py:121, `RemakeApp.__init__` client/main.py:198, `RemakeApp.save_shot` client/main.py:1659 |
| 638240 | FUN_00638240 | 467 | `RemakeApp.update_quest_log` client/main.py:1528, `<module>` engine/questlog.py:2, `QuestLog.take_sound` engine/questlog.py:143 |
| 641040 | FUN_00641040 | 464 | `BattleScreen.update_portrait_bars` client/hud.py:447, `bar_shares` engine/hud.py:695 |
| 65e540 | FUN_0065e540 | 126 | `TOOLTIP_STEP_MS` client/hud.py:48, `BattleScreen.tooltip_delay_ms` client/hud.py:818, `OptionsView.tick` client/options.py:184, `tooltip_items` client/ui.py:373, +2 |
| 6c1bc0 | FUN_006c1bc0 | 1108 | `<module>` engine/ambient.py:5, `LAST_TYPE` engine/ambient.py:55, `tile_counts` engine/ambient.py:109 |
| 6c6870 | CTerrain::LoadMapHeader | 685 | `<module>` engine/ambient.py:13, `SET_DUNGEON` engine/ambient.py:71, `sound_set` engine/ambient.py:85, `Ambient.__init__` engine/ambient.py:156, +2 |

## Функции, найденные на уровне модуля (`module`) — 180

| Адрес | Функция C | Строк | Python |
|---|---|---:|---|
| 40c8b0 | CCamera::ctor_dtor_40c8b0 | 93 | `SCROLL_SMOOTH` engine/camera.py:65 |
| 40d9b0 | FUN_0040d9b0 | 55 | `<module>` engine/camera.py:17, `TARGET_HEIGHT` engine/camera.py:48 |
| 40f8b0 | FUN_0040f8b0 | 112 | `<module>` engine/camera.py:7 |
| 431370 | FUN_00431370 | 182 | `<module>` tools/scenario.py:23 |
| 458da0 | FUN_00458da0 | 122 | `<module>` engine/camera.py:6 |
| 459100 | FUN_00459100 | 378 | `<module>` engine/camera.py:6, `<module>` engine/options.py:16 |
| 45ea20 | CZoneList::LoadMapTxt | 573 | `<module>` eilib/zones.py:2, `<module>` engine/weather.py:2 |
| 4613a0 | FUN_004613a0 | 12 | `<module>` engine/combat.py:8 |
| 464750 | FUN_00464750 | 81 | `<module>` engine/lever.py:10 |
| 465ee0 | FUN_00465ee0 | 64 | `<module>` engine/lightning.py:2 |
| 46b4a0 | CMagicTrapRecord::ctor | 25 | `DEFAULT_INTERVAL` engine/trap.py:41 |
| 46b570 | FUN_0046b570 | 47 | `<module>` engine/trap.py:6 |
| 46c010 | FUN_0046c010 | 46 | `<module>` engine/stats.py:3 |
| 477b00 | CUnitServer::SetUnitStats | 147 | `<module>` engine/combat.py:4, `<module>` engine/stats.py:25 |
| 481f00 | CMobLoader::ReadDiplomacy | 239 | `<module>` engine/ai.py:9 |
| 483fd0 | FUN_00483fd0 | 20 | `<module>` engine/particles.py:2 |
| 4849f0 | FUN_004849f0 | 50 | `START_TRANS_TICKS` engine/particles.py:162 |
| 484c00 | CParticle::HolderPoint | 225 | `<module>` engine/sources.py:13 |
| 4889d0 | Particle2052::Emit | 148 | `TRANSFORM_TICKS` engine/particles.py:178 |
| 488db0 | Particle2052::Move | 63 | `TRANSFORM_TICKS` engine/particles.py:178 |
| 488f00 | Particle2052::Source | 29 | `TRANSFORM_TICKS` engine/particles.py:178 |
| 48eea0 | Particle2033::Emit | 179 | `<module>` engine/weather.py:14 |
| 48f300 | Particle2033::Move | 30 | `<module>` engine/weather.py:14 |
| 48f380 | Particle2033::Source | 17 | `<module>` engine/weather.py:14 |
| 494470 | Particle203e::Emit | 112 | `PORTAL_LEAST` engine/particles.py:120 |
| 494770 | Particle203e::Move | 74 | `PORTAL_LEAST` engine/particles.py:120 |
| 4964b0 | Particle204f::Move | 50 | `START_TRANS_TICKS` engine/particles.py:160 |
| 496840 | Particle2051::Move | 59 | `START_TRANS_TICKS` engine/particles.py:160 |
| 496990 | Particle2051::Source | 29 | `START_TRANS_TICKS` engine/particles.py:160 |
| 498e80 | FUN_00498e80 | 13 | `<module>` engine/particles.py:2 |
| 49ed20 | Quat::Slerp | 49 | `<module>` engine/quat.py:5 |
| 49ee70 | Random::Refill | 102 | `<module>` engine/particles.py:32 |
| 4b41e0 | CScriptWorld::EvalFloat | 733 | `<module>` engine/script.py:8 |
| 4ba1c0 | FUN_004ba1c0 | 21 | `<module>` engine/ai.py:36 |
| 4bcff0 | CScriptWorld::DeletePointLight | 65 | `<module>` engine/sources.py:16 |
| 4bd310 | CScriptWorld::CreateFXSource | 132 | `<module>` engine/sources.py:19 |
| 4bd6a0 | CScriptWorld::CreatePointLight | 67 | `<module>` engine/sources.py:15 |
| 4bd9a0 | CScriptWorld::CreateParticleSource | 127 | `SCRIPT_TYPES` engine/particles.py:93, `<module>` engine/sources.py:5 |
| 4bdbb0 | CScriptWorld::MoveParticleSource | 45 | `<module>` engine/sources.py:10 |
| 4bdc90 | CScriptWorld::DeleteParticleSource | 75 | `<module>` engine/sources.py:11 |
| 4c6370 | FUN_004c6370 | 52 | `<module>` engine/sources.py:22 |
| 4c7c90 | CScriptInstanceList::PushFront | 75 | `<module>` engine/script.py:23 |
| 4e3630 | FUN_004e3630 | 106 | `GRID_OFFSETS` engine/weather.py:87, `FALL_MIDDLE` engine/weather.py:90 |
| 4e3980 | FUN_004e3980 | 91 | `FLAKE_SWAY` engine/weather.py:111 |
| 4e3c20 | FUN_004e3c20 | 192 | `<module>` engine/weather.py:4, `GRID_OFFSETS` engine/weather.py:87 |
| 4e3fc0 | FUN_004e3fc0 | 28 | `<module>` engine/weather.py:30 |
| 4e5660 | FUN_004e5660 | 10 | `FLAKE_SWAY` engine/weather.py:111 |
| 4e5680 | FUN_004e5680 | 30 | `<module>` engine/weather.py:4 |
| 500a00 | FUN_00500a00 | 30 | `TEXTURE_NAMES` engine/particles.py:69, `<module>` engine/weather.py:16, `TEXTURE_NAME` engine/weather.py:84 |
| 504a60 | FUN_00504a60 | 110 | `<module>` engine/options.py:16 |
| 50aaf0 | CAnimationTrack::Load | 114 | `<module>` eilib/anm.py:4 |
| 50acd0 | CAnimationTrack::SetTime | 129 | `<module>` engine/animation.py:4 |
| 50f630 | CUnit::HasEffect | 36 | `<module>` engine/sounds.py:28 |
| 512b20 | CObject3D::FindAction | 58 | `<module>` engine/prop.py:5 |
| 5141f0 | CUnit::CreatePrototypeItems | 72 | `UNIT_CLASS_HUMANOID` eilib/idb.py:96, `<module>` eilib/lootdb.py:9 |
| 517700 | CObject::SetComplection | 88 | `<module>` engine/lever.py:5 |
| 51c920 | FUN_0051c920 | 24 | `<module>` engine/woundskin.py:16 |
| 51f9b0 | FUN_0051f9b0 | 26 | `<module>` eilib/idb.py:36 |
| 520220 | CPerkDatabase::Load | 145 | `<module>` eilib/idb.py:33 |
| 522ce0 | CUnit::PackStats | 137 | `<module>` engine/stats.py:33 |
| 524060 | CUnit::LegsSpeedFactor | 33 | `<module>` engine/stats.py:44 |
| 524cd0 | FUN_00524cd0 | 120 | `<module>` engine/stats.py:46 |
| 525d60 | CUnitStats::BackstabFactor | 21 | `<module>` engine/combat.py:29 |
| 528800 | FUN_00528800 | 10 | `STATS_SIZE` eilib/sav.py:420 |
| 529dd0 | CMagicTrapObject::ctor_dtor_529dd0 | 76 | `<module>` engine/trap.py:4 |
| 52a2c0 | CMagicTrapObject::Load | 82 | `<module>` engine/trap.py:6 |
| 52a680 | CMagicTrapObject::CastAtEntering | 144 | `<module>` engine/trap.py:17 |
| 52ab30 | CMagicTrapObject::CheckAndCast | 52 | `<module>` engine/trap.py:19 |
| 52b3f0 | CLeverObject::SetStats | 25 | `<module>` engine/lever.py:5 |
| 52bb00 | CLeverObject::vf38 | 12 | `<module>` engine/lever.py:21 |
| 52e940 | CUnit::AddEnemyPlayer | 15 | `<module>` engine/magic.py:25 |
| 52f7f0 | shared_vf_CFakeUnit46_CUnit41 | 14 | `<module>` engine/magic.py:61 |
| 52f920 | CUnit::MoveSpeeds | 40 | `<module>` engine/stats.py:35 |
| 52fa20 | CUnit::MoveSpeed | 29 | `<module>` engine/stats.py:35 |
| 530570 | CUnit::UpdatePathClass | 61 | `<module>` engine/combat.py:89 |
| 530930 | CUnit::Prototype | 12 | `<module>` engine/magic.py:87 |
| 5318b0 | FUN_005318b0 | 28 | `<module>` engine/combat.py:12 |
| 532860 | FUN_00532860 | 80 | `<module>` engine/lightning.py:6, `ARC_HAND_HEIGHT` engine/magic.py:265 |
| 535770 | CUnitClientSpecific::GetWeaponBasePoint | 38 | `<module>` engine/equipment.py:11 |
| 535b30 | CUnitClientSpecific::vf40 | 68 | `MAX_STEP_MS` engine/timing.py:22, `<module>` engine/unit.py:3 |
| 535cb0 | CUnitClientSpecific::CollectBodyParts | 248 | `<module>` engine/equipment.py:5 |
| 5381b0 | CUnitClientSpecific::vf26 | 204 | `<module>` engine/equipment.py:5 |
| 53a200 | CUnitDatabase::Load | 217 | `<module>` eilib/idb.py:18 |
| 54e600 | FUN_0054e600 | 203 | `<module>` engine/unit.py:24 |
| 550100 | FUN_00550100 | 238 | `<module>` engine/unit.py:15 |
| 5511a0 | CUnitServer::MoveClips | 135 | `<module>` engine/stats.py:47 |
| 551400 | CUnitServer::MoveAnimationList | 141 | `<module>` engine/unit.py:32 |
| 552c60 | FUN_00552c60 | 27 | `<module>` engine/magic.py:44, `<module>` engine/quickitems.py:4 |
| 5543b0 | CUnitServer::vf21 | 29 | `<module>` engine/magic.py:49 |
| 5546b0 | FUN_005546b0 | 27 | `<module>` engine/ai.py:51 |
| 554d00 | CUnitServer::CastSpellPoint | 104 | `<module>` engine/magic.py:36 |
| 566360 | CWorldLight::UpdateTimeOfDay | 244 | `<module>` engine/lighting.py:8 |
| 56d4c0 | FUN_0056d4c0 | 89 | `<module>` engine/footprints.py:2 |
| 56daf0 | FUN_0056daf0 | 75 | `<module>` engine/camera.py:18 |
| 578e30 | FUN_00578e30 | 98 | `<module>` engine/weather.py:33 |
| 57a780 | FUN_0057a780 | 129 | `<module>` eilib/sav.py:24 |
| 57aba0 | FUN_0057aba0 | 267 | `<module>` eilib/sav.py:24 |
| 57b3b0 | FUN_0057b3b0 | 288 | `<module>` eilib/sav.py:24 |
| 58b750 | FUN_0058b750 | 570 | `<module>` engine/weather.py:29 |
| 58ef20 | CWorldServer::PlaceParty | 460 | `<module>` eilib/zones.py:12 |
| 593fa0 | FUN_00593fa0 | 10 | `<module>` engine/water.py:18 |
| 594ab0 | FUN_00594ab0 | 31 | `<module>` engine/party.py:5 |
| 595c70 | FUN_00595c70 | 158 | `<module>` engine/weather.py:27 |
| 596030 | CWorldServer::LoadMapObjects | 56 | `<module>` engine/script.py:67 |
| 5ace80 | LoadMenuRects | 91 | `<module>` engine/menus.py:6 |
| 5b0d90 | ServerMap::CoarsePath | 127 | `<module>` engine/pathing.py:37 |
| 5b2fb0 | ServerMap::InitPriceLog | 15 | `PRICE_BASE` engine/pathing.py:182 |
| 5b3160 | PointHash::Get | 26 | `<module>` engine/pathing.py:45 |
| 5b6070 | ServerMap::InitCodeCosts | 65 | `<module>` engine/pathing.py:30, `CODE_COSTS` engine/pathing.py:119 |
| 5b6740 | ServerMap::PutFloor | 123 | `<module>` engine/pathing.py:49 |
| 5b77f0 | ServerMap::PointHeight | 10 | `<module>` engine/pathing.py:44 |
| 5b8fa0 | ServerMap::UpdatePassGrid | 226 | `<module>` engine/pathing.py:28 |
| 5b9580 | ServerMap::PathCosts | 33 | `FREE_CODE_COSTS` engine/pathing.py:123, `TURN_STEP_FACTOR` engine/pathing.py:153 |
| 5b9c90 | ServerMap::SearchPath | 310 | `<module>` engine/pathing.py:37 |
| 5bb190 | ServerMap::BuildTrack | 107 | `<module>` engine/stats.py:37 |
| 5bfc40 | FUN_005bfc40 | 30 | `PUSH_OUT_POINTS` engine/pathing.py:166 |
| 5c25d0 | PathWindow::Dijkstra | 244 | `<module>` engine/pathing.py:38 |
| 5c5cc0 | PathPoint::Step | 52 | `DIRECTIONS` engine/pathing.py:146 |
| 5c7430 | MoveTrack::Direction | 175 | `<module>` engine/stats.py:37 |
| 5d0a50 | FUN_005d0a50 | 14 | `GUARD_WEIGHT` engine/ai.py:124 |
| 5d0f60 | CBehaviourPatrol::ctor | 16 | `<module>` engine/ai.py:36 |
| 5d16b0 | CBehaviourPatrol::Weigh | 15 | `<module>` engine/ai.py:48, `GUARD_WEIGHT` engine/ai.py:124 |
| 5daea0 | LoadCursors | 75 | `<module>` client/cursor.py:5, `EDGE_SIZE` client/cursor.py:34, `<module>` client/picking.py:5, `PART_CURSORS` client/picking.py:42, +1 |
| 5f4990 | CInterface3D::Create | 281 | `<module>` client/ui.py:8, `FONT_TENTHS` client/ui.py:43 |
| 5f74e0 | FUN_005f74e0 | 11 | `<module>` engine/tutorial.py:15 |
| 5fc3b0 | CInterfaceModel::PlaySpecial | 13 | `<module>` engine/menus.py:12 |
| 5fcc00 | FUN_005fcc00 | 91 | `INFO_TEXT_ORIGIN` engine/hud.py:71 |
| 6071a0 | FUN_006071a0 | 71 | `<module>` client/talk.py:5, `ORIGIN` client/talk.py:31, `<module>` engine/talk.py:3, `ROWS_TOP` engine/talk.py:24 |
| 620920 | CInventoryStrip::SelectCategory | 44 | `DIM` engine/hud.py:48, `<module>` engine/inventory.py:8 |
| 621fc0 | CCampWindow::Build | 111 | `<module>` engine/camp.py:2, `MEMBER_RECTS` engine/camp.py:38, `STRIP_BOTTOM_RECT` engine/camp.py:44 |
| 626160 | FUN_00626160 | 232 | `<module>` engine/skills.py:3 |
| 627060 | FUN_00627060 | 101 | `<module>` engine/camp.py:3, `MAIN_RECT` engine/camp.py:52, `CELL_SIZE` engine/camp.py:55 |
| 6279a0 | FUN_006279a0 | 95 | `<module>` engine/camp.py:3, `MAIN_RECT` engine/camp.py:52, `CELL_SIZE` engine/camp.py:55 |
| 628080 | FUN_00628080 | 131 | `<module>` engine/camp.py:4, `MAIN_RECT` engine/camp.py:52 |
| 628c80 | FUN_00628c80 | 88 | `<module>` engine/camp.py:4 |
| 62aae0 | CEscMenuWindow::Activate | 72 | `<module>` engine/menus.py:11 |
| 632670 | CMainMenuWindow::Activate | 114 | `<module>` engine/menus.py:4 |
| 6333c0 | CMainMenuWindow::OnMouseMove | 60 | `<module>` engine/menus.py:4, `IDLE_KEY` engine/prop.py:27 |
| 6335a0 | CMainMenuWindow::ShowVersion | 48 | `VERSION_RAISE` client/menu.py:37, `VERSION_TEXT` engine/menus.py:76 |
| 634cc0 | FUN_00634cc0 | 11 | `ENTER_KEYS` engine/messagebox.py:52 |
| 647b30 | CInfoWindow::DrawHitLocations | 144 | `INFO_GENERAL` engine/hud.py:65, `HIT_LOCATION_POINTS` engine/hud.py:87 |
| 64dc20 | FUN_0064dc20 | 252 | `<module>` client/questmap.py:2, `<module>` engine/questmap.py:2, `LIST_RECT` engine/questmap.py:47 |
| 6508e0 | FUN_006508e0 | 35 | `PLAQUE_PARTS` client/options.py:64 |
| 6521c0 | FUN_006521c0 | 70 | `OPTIONS` engine/options.py:66 |
| 6593e0 | CTutorialWindow::LayoutPage | 492 | `<module>` client/tutorial.py:4, `BUTTONS_SPACE` client/tutorial.py:36, `<module>` engine/tutorial.py:2 |
| 6604c0 | CParty::CopyItems | 138 | `<module>` engine/roster.py:15 |
| 661010 | CParty::ctor_dtor_661010 | 55 | `<module>` engine/roster.py:2 |
| 6620b0 | CParty::RemoveUnit | 37 | `<module>` engine/roster.py:13 |
| 6621b0 | CParty::InitUnitRecord | 164 | `<module>` engine/roster.py:12 |
| 662890 | CParty::AddUnit | 42 | `<module>` engine/roster.py:12 |
| 665430 | CPlayer::AddItems | 52 | `<module>` engine/party.py:6 |
| 665600 | CPlayer::HasQuestItem | 76 | `<module>` engine/party.py:8 |
| 665740 | CPlayer::EraseQuestItem | 53 | `<module>` engine/party.py:7 |
| 66a560 | FUN_0066a560 | 21 | `<module>` engine/flyinghp.py:9 |
| 678170 | CEffectPointLight::vf06 | 44 | `<module>` engine/magic.py:96 |
| 678e80 | CEffectParticle::MakeParticle | 117 | `<module>` engine/particles.py:29 |
| 679270 | CEffect::SetParticleSize | 12 | `<module>` engine/sources.py:11 |
| 67a670 | CEffectLightning::SetEnds | 29 | `<module>` engine/lightning.py:5, `<module>` engine/sources.py:26 |
| 67a850 | CEffectLightning::SetFollowed | 13 | `<module>` engine/lightning.py:6 |
| 67a880 | CEffectLightning::SetDemon | 12 | `<module>` engine/lightning.py:5 |
| 67beb0 | CEffectArrowFigure::vf24 | 13 | `<module>` engine/missiles.py:18 |
| 67bee0 | CEffectArrowFigure::vf23 | 62 | `<module>` engine/missiles.py:18 |
| 67f540 | FUN_0067f540 | 69 | `<module>` engine/magic.py:108, `SOUND_NEAR` engine/magic.py:282 |
| 681b40 | CEffectSpell::CanHit | 19 | `<module>` engine/magic.py:86 |
| 684510 | CSpellManager::LoadDatabase | 163 | `<module>` eilib/idb.py:39 |
| 68d560 | CLight::LightVertexWrapped | 42 | `LIGHT_SHADER` client/gl.py:58, `<module>` engine/magic.py:97 |
| 6945d0 | CFileRegistry::IsValid | 20 | `<module>` eilib/reg.py:3 |
| 6a3160 | CItemArmor::GetLayer | 24 | `<module>` engine/combat.py:82, `OUTER_ARMOR_KINDS` engine/combat.py:180 |
| 6a3200 | CItemArmor::vf17 | 170 | `ARMOR_HELM` eilib/idb.py:108 |
| 6a4960 | CItemLoot::ctor_dtor_6a4960 | 11 | `<module>` client/loot.py:7 |
| 6a6760 | CItemDatabase::Load | 164 | `<module>` eilib/idb.py:9 |
| 6ac930 | FUN_006ac930 | 244 | `<module>` engine/party.py:5 |
| 6b46d0 | CMusicSystem::vf02 | 280 | `<module>` engine/music.py:4 |
| 6bb200 | Stream::Fade | 62 | `<module>` engine/music.py:16, `FADE_MS` engine/music.py:49 |
| 6c3180 | CTerrain::LightVertex | 121 | `LIGHT_SHADER` client/gl.py:58, `<module>` engine/magic.py:97 |
| 6c3e40 | CTerrain::UpdateWaves | 161 | `WATER_VERTEX_FORMAT` client/gl.py:148, `<module>` engine/water.py:8 |
| 6c51f0 | FUN_006c51f0 | 47 | `AMPLITUDE` engine/water.py:49 |
| 6c7e70 | CTerrain::SetWindStrength | 39 | `<module>` engine/water.py:8 |
| 6cb1a0 | CTerrain::vf06 | 52 | `<module>` engine/ambient.py:2, `<module>` engine/water.py:11 |
| 6d75e0 | FUN_006d75e0 | 421 | `<module>` eilib/idb.py:44, `<module>` engine/acks.py:2 |

## Описаны в first_scan.md, но без реализации — 113

| Адрес | Функция C | Статус | Строк | Раздел first_scan.md |
|---|---|---|---:|---|
| 401000 | FUN_00401000 | missing | 66 | 2. Как восстанавливались классы |
| 401e00 | Vec3::Dot | indirect | 10 | 2. Как восстанавливались классы |
| 401e20 | Vec3::Length | indirect | 10 | 2. Как восстанавливались классы |
| 402220 | FUN_00402220 | indirect | 26 | 8. Движение на клиенте |
| 40a000 | FUN_0040a000 | indirect | 183 | 41. Отряд в scenario.sav (MISSING_PLAN, P2) |
| 40a440 | FUN_0040a440 | missing | 87 | 41. Отряд в scenario.sav (MISSING_PLAN, P2) |
| 40cdc0 | Vec3::Add | indirect | 21 | 2. Как восстанавливались классы |
| 40d0a0 | FUN_0040d0a0 | indirect | 36 | 28. Интерфейс: ролики, меню, камера, лагерь, квесты (второй разработчик) |
| 40eb60 | FUN_0040eb60 | indirect | 62 | 28. Интерфейс: ролики, меню, камера, лагерь, квесты (второй разработчик) |
| 410750 | Vec3::AddAssign | indirect | 12 | 2. Как восстанавливались классы |
| 410770 | Vec3::Scale | indirect | 17 | 2. Как восстанавливались классы |
| 42fca0 | FUN_0042fca0 | indirect | 29 | 40. Файлы сохранений игры: упаковка и дерево (MISSING_PLAN, P2) |
| 4489d0 | Vec3::Cross | indirect | 29 | 2. Как восстанавливались классы |
| 448a10 | Vec3::Scale2 | indirect | 17 | 2. Как восстанавливались классы |
| 44a7b0 | SplitAnimation | missing | 419 | 3. Загрузка моделей и анимаций |
| 455680 | FUN_00455680 | indirect | 64 | 39. Длящиеся заклинания без урона (MISSING_PLAN, P2) |
| 455940 | FUN_00455940 | indirect | 9 | 39. Длящиеся заклинания без урона (MISSING_PLAN, P2) |
| 461260 | FUN_00461260 | indirect | 70 | 41. Отряд в scenario.sav (MISSING_PLAN, P2) |
| 488970 | CParticle::RandomRange | indirect | 21 | 12. Магия и частицы |
| 49e800 | Quat::FromAxisAngle | indirect | 23 | 2. Как восстанавливались классы |
| 4a5340 | FUN_004a5340 | indirect | 15 | 40. Файлы сохранений игры: упаковка и дерево (MISSING_PLAN, P2) |
| 4a8560 | FUN_004a8560 | indirect | 90 | 40. Файлы сохранений игры: упаковка и дерево (MISSING_PLAN, P2); 41. Отряд в scenario.sav (MISSING_PLAN, P2) |
| 4d23a0 | FUN_004d23a0 | indirect | 528 | 29. Опыт, атрибуты, навыки и способности (по декомпилированному коду) |
| 4dbe40 | FUN_004dbe40 | indirect | 47 | 30. Обыск, разговоры, озвучка брифинга, лагерь, торговля, сохранения, курсоры (второй разработчик) |
| 4e0020 | CTextDatabase::GetString | indirect | 73 | 19. Интерфейс (по декомпилированному коду) |
| 4e5e50 | FUN_004e5e50 | missing | 437 | 13. Рендер |
| 4f0b90 | FUN_004f0b90 | missing | 164 | 13. Рендер |
| 500540 | FUN_00500540 | missing | 166 | 13. Рендер |
| 50af20 | CAnimationTrack::AddTime | missing | 10 | 4. Трек анимации: интерполяция |
| 512730 | CMapObjectManager::vf14 | missing | 138 | 27. Скрипты карт (по декомпилированному коду) |
| 518d30 | FUN_00518d30 | indirect | 26 | 5. Иерархия костей |
| 519380 | FUN_00519380 | missing | 83 | 9. Выбор анимации и момент удара (сервер); 15. Что не удалось выяснить |
| 51b7e0 | CObject3DClientSpecific::vf19 | missing | 431 | 13. Рендер |
| 525e90 | FUN_00525e90 | missing | 49 | 39. Длящиеся заклинания без урона (MISSING_PLAN, P2) |
| 52faf0 | CUnit::MoveSpeedOf | indirect | 29 | 22. Юниты: характеристики, ходьба, сетка пути (по декомпилированному коду) |
| 5300d0 | FUN_005300d0 | indirect | 25 | 39. Длящиеся заклинания без урона (MISSING_PLAN, P2) |
| 530120 | FUN_00530120 | indirect | 14 | 39. Длящиеся заклинания без урона (MISSING_PLAN, P2) |
| 530150 | FUN_00530150 | indirect | 10 | 35. Зрение ИИ, память об угрозе и подозрение (MISSING_PLAN, шаг 5) |
| 530180 | FUN_00530180 | missing | 11 | 39. Длящиеся заклинания без урона (MISSING_PLAN, P2) |
| 530280 | FUN_00530280 | indirect | 24 | 8. Движение на клиенте |
| 531920 | FUN_00531920 | indirect | 34 | 30. Обыск, разговоры, озвучка брифинга, лагерь, торговля, сохранения, курсоры (второй разработчик) |
| 5352b0 | CUnitClientSpecific::vf14 | missing | 70 | 7. Время: тик 55 мс и частота анимаций |
| 535480 | FUN_00535480 | indirect | 51 | 7. Время: тик 55 мс и частота анимаций |
| 535590 | CUnitClientSpecific::vf25 | missing | 102 | 17. Экипировка (по декомпилированному коду) |
| 537d80 | CUnitClientSpecific::vf51 | missing | 122 | 27. Скрипты карт (по декомпилированному коду) |
| 549160 | FUN_00549160 | indirect | 139 | 12. Магия и частицы |
| 54b760 | CUnitServer::ActionTicks | missing | 20 | 29. Опыт, атрибуты, навыки и способности (по декомпилированному коду) |
| 54b880 | FUN_0054b880 | missing | 24 | 29. Опыт, атрибуты, навыки и способности (по декомпилированному коду); 33. Сохранения, наёмники, глобальная карта, брифинг в 3D, слух, сигналы, заклинания (второй разработчик) |
| 54b9c0 | CUnitServer::StrikeTicks | missing | 29 | 29. Опыт, атрибуты, навыки и способности (по декомпилированному коду) |
| 54f000 | FUN_0054f000 | missing | 220 | 30. Обыск, разговоры, озвучка брифинга, лагерь, торговля, сохранения, курсоры (второй разработчик) |
| 551ae0 | FUN_00551ae0 | indirect | 147 | 39. Длящиеся заклинания без урона (MISSING_PLAN, P2) |
| 552280 | FUN_00552280 | missing | 18 | 39. Длящиеся заклинания без урона (MISSING_PLAN, P2) |
| 5533b0 | CUnitServer::vf19 | missing | 25 | 29. Опыт, атрибуты, навыки и способности (по декомпилированному коду) |
| 564190 | FUN_00564190 | missing | 21 | 34. Проверка кода второго разработчика (основной поток) |
| 565790 | FUN_00565790 | indirect | 123 | 34. Проверка кода второго разработчика (основной поток) |
| 5659d0 | FUN_005659d0 | indirect | 123 | 34. Проверка кода второго разработчика (основной поток) |
| 56d150 | FUN_0056d150 | indirect | 119 | 24. Звуки юнитов (по декомпилированному коду) |
| 5785d0 | CWorld::RunConsoleScript | missing | 36 | 29. Опыт, атрибуты, навыки и способности (по декомпилированному коду) |
| 578c40 | FUN_00578c40 | indirect | 41 | 30. Обыск, разговоры, озвучка брифинга, лагерь, торговля, сохранения, курсоры (второй разработчик) |
| 57bdf0 | FUN_0057bdf0 | missing | 247 | 38. Что видит игрок (MISSING_PLAN, P2) |
| 57f440 | FUN_0057f440 | indirect | 12 | 29. Опыт, атрибуты, навыки и способности (по декомпилированному коду) |
| 58d340 | FUN_0058d340 | indirect | 32 | 25. ИИ: ответный бой (по декомпилированному коду); 33. Сохранения, наёмники, глобальная карта, брифинг в 3D, слух, сигналы, заклинания (второй разработчик) |
| 58dff0 | CWorldServer::vf06 | indirect | 212 | 7. Время: тик 55 мс и частота анимаций |
| 58e5d0 | FUN_0058e5d0 | indirect | 31 | 12. Магия и частицы |
| 58ee00 | FUN_0058ee00 | missing | 12 | 19. Интерфейс (по декомпилированному коду) |
| 597bd0 | FUN_00597bd0 | indirect | 554 | 40. Файлы сохранений игры: упаковка и дерево (MISSING_PLAN, P2) |
| 5ac400 | CFigureManager::MakeAnimationNode | missing | 66 | 3. Загрузка моделей и анимаций |
| 5ac590 | FUN_005ac590 | missing | 42 | 3. Загрузка моделей и анимаций |
| 5aca60 | FUN_005aca60 | indirect | 149 | 3. Загрузка моделей и анимаций |
| 5b4150 | FUN_005b4150 | missing | 97 | 22. Юниты: характеристики, ходьба, сетка пути (по декомпилированному коду) |
| 5b61b0 | ServerMap::AddObstacle | indirect | 70 | 22. Юниты: характеристики, ходьба, сетка пути (по декомпилированному коду) |
| 5b8620 | FUN_005b8620 | indirect | 54 | 22. Юниты: характеристики, ходьба, сетка пути (по декомпилированному коду) |
| 5ba830 | FUN_005ba830 | indirect | 82 | 27. Скрипты карт (по декомпилированному коду) |
| 5bc1f0 | FUN_005bc1f0 | missing | 22 | 22. Юниты: характеристики, ходьба, сетка пути (по декомпилированному коду) |
| 5c42b0 | PathWindow::InitTurnTables | missing | 94 | 22. Юниты: характеристики, ходьба, сетка пути (по декомпилированному коду) |
| 5c6fc0 | FUN_005c6fc0 | indirect | 136 | 22. Юниты: характеристики, ходьба, сетка пути (по декомпилированному коду) |
| 5c98c0 | FUN_005c98c0 | indirect | 42 | 26. Магия: заклинания, модификаторы, урон (по декомпилированному коду) |
| 5ca1d0 | FUN_005ca1d0 | indirect | 122 | 35. Зрение ИИ, память об угрозе и подозрение (MISSING_PLAN, шаг 5) |
| 5d5f60 | CUnitLogic::AddBehaviour | indirect | 111 | 27. Скрипты карт (по декомпилированному коду); 39. Длящиеся заклинания без урона (MISSING_PLAN, P2) |
| 5edd80 | FUN_005edd80 | missing | 127 | 19. Интерфейс (по декомпилированному коду) |
| 5ee1b0 | FUN_005ee1b0 | indirect | 67 | 30. Обыск, разговоры, озвучка брифинга, лагерь, торговля, сохранения, курсоры (второй разработчик) |
| 5f85c0 | FUN_005f85c0 | indirect | 71 | 41. Отряд в scenario.sav (MISSING_PLAN, P2) |
| 5f86d0 | FUN_005f86d0 | indirect | 15 | 41. Отряд в scenario.sav (MISSING_PLAN, P2) |
| 5fc1f0 | CInterfaceModel::Load | indirect | 43 | 19. Интерфейс (по декомпилированному коду) |
| 5fc4f0 | CInterfaceModel::ShowAllParts | indirect | 31 | 19. Интерфейс (по декомпилированному коду) |
| 5fef20 | AddWindowTexture | indirect | 86 | 19. Интерфейс (по декомпилированному коду) |
| 604c50 | CBriefingWindow::ParseBriefing | missing | 585 | 32. Быстрые предметы, стрелы, гроза и следы, навыки в лагере, ответы юнитов (второй разработчик) |
| 605da0 | CBriefingWindow::ShowActor | missing | 246 | 32. Быстрые предметы, стрелы, гроза и следы, навыки в лагере, ответы юнитов (второй разработчик) |
| 607880 | FUN_00607880 | indirect | 54 | 30. Обыск, разговоры, озвучка брифинга, лагерь, торговля, сохранения, курсоры (второй разработчик) |
| 60c830 | CCampConstructorWindow::Activate | missing | 880 | 19. Интерфейс (по декомпилированному коду); 21. Полоса предметов (по декомпилированному коду) |
| 613e60 | CInterface::CreateWindows | missing | 915 | 19. Интерфейс (по декомпилированному коду) |
| 61b280 | CMultiplayerWindow::Build | missing | 473 | 19. Интерфейс (по декомпилированному коду) |
| 622920 | FUN_00622920 | missing | 23 | 28. Интерфейс: ролики, меню, камера, лагерь, квесты (второй разработчик) |
| 626ed0 | FUN_00626ed0 | indirect | 35 | 28. Интерфейс: ролики, меню, камера, лагерь, квесты (второй разработчик) |
| 627350 | FUN_00627350 | missing | 186 | 28. Интерфейс: ролики, меню, камера, лагерь, квесты (второй разработчик) |
| 627c50 | FUN_00627c50 | missing | 152 | 28. Интерфейс: ролики, меню, камера, лагерь, квесты (второй разработчик) |
| 629a00 | FUN_00629a00 | missing | 24 | 19. Интерфейс (по декомпилированному коду); 28. Интерфейс: ролики, меню, камера, лагерь, квесты (второй разработчик) |
| 62c210 | CCampWindow::Activate | missing | 195 | 19. Интерфейс (по декомпилированному коду); 21. Полоса предметов (по декомпилированному коду) |
| 630c80 | FUN_00630c80 | missing | 56 | 28. Интерфейс: ролики, меню, камера, лагерь, квесты (второй разработчик) |
| 630dc0 | FUN_00630dc0 | missing | 56 | 28. Интерфейс: ролики, меню, камера, лагерь, квесты (второй разработчик) |
| 639eb0 | FUN_00639eb0 | missing | 293 | 34. Проверка кода второго разработчика (основной поток) |
| 63c7c0 | CQuickItemsWindow::ctor | indirect | 102 | 20. Экран боя (панели в игре) |
| 63f2c0 | CEffectSlotsWindow::ctor | indirect | 66 | 20. Экран боя (панели в игре) |
| 6428f0 | CMiniMapWindow::PlaceElements | indirect | 46 | 20. Экран боя (панели в игре) |
| 648d30 | FUN_00648d30 | missing | 57 | 7. Время: тик 55 мс и частота анимаций |
| 648ed0 | FUN_00648ed0 | missing | 63 | 7. Время: тик 55 мс и частота анимаций |
| 666cd0 | CPlayerClient::ctor_dtor_666cd0 | missing | 80 | 38. Что видит игрок (MISSING_PLAN, P2) |
| 669710 | FUN_00669710 | missing | 13 | 7. Время: тик 55 мс и частота анимаций |
| 678290 | CEffectPointLight::UpdateClientLight | missing | 86 | 26. Магия: заклинания, модификаторы, урон (по декомпилированному коду) |
| 67c040 | CEffectArrowFigure::vf21 | missing | 44 | 11. Стрелы |
| 683fb0 | CSpellConstructor::ctor_dtor_683fb0 | missing | 33 | 2. Как восстанавливались классы |
| 68b330 | FUN_0068b330 | indirect | 48 | 39. Длящиеся заклинания без урона (MISSING_PLAN, P2) |
| 6b9710 | FUN_006b9710 | indirect | 86 | 37. Громкость звуков мира и каналы (MISSING_PLAN, шаг 5) |

## Именованные функции без реализации — 692

Функции с именем из `re/names.json`, на которые ремейк не ссылается. Безымянные `missing` (6343) — только в JSON.

| Класс | Методы (адрес) |
|---|---|
| (свободные функции) | shared_vf_CClientSpecificManager2_CLightManager2_CServerSpecificManager2 403020, shared_vf_CFakeUnit31_CUnit26 4467f0, shared_vf_CFakeUnit58_CUnit53 446810, shared_vf_CFakeUnit61_CUnit56 446830, shared_vf_CFakeUnit56_CUnit51 446850, shared_vf_CFakeUnit57_CUnit52 446870, shared_vf_CFakeUnit60_CUnit55 4468a0, shared_vf_CFakeUnit59_CUnit54 4468b0, SplitAnimation 44a7b0, thunk_FUN_00700930 4563a0, thunk_FUN_00700930 4614b0, thunk_FUN_0042bef0 46c7e0, thunk_FUN_0042bef0 46cbb0, thunk_FUN_0042bef0 46d070, thunk_FUN_007020ef 49f810, thunk_FUN_00700930 4cb7d0, thunk_FUN_0050e660 529b20, thunk_FUN_0050e660 529b50, shared_vf_CTorchObjectClientSpecific9_CTorchObjectSpecific9 52c490, shared_vf_CTorchObjectClientSpecific10_CTorchObjectSpecific10 52c500, shared_vf_CFakeUnit5_CUnit0 52e740, shared_vf_CFakeUnit19_CUnit14 52e870, shared_vf_CFakeUnit26_CUnit21 52e910, shared_vf_CFakeUnit33_CUnit28 52eb30, shared_vf_CFakeUnit32_CUnit27 52eb50, shared_vf_CFakeUnit34_CUnit29 52eb70, shared_vf_CFakeUnit50_CUnit45 52ec70, shared_vf_CFakeUnit51_CUnit46 52f740, shared_vf_CFakeUnit6_CUnit1 52f7d0, shared_vf_CFakeUnit25_CUnit20 52fc40, shared_vf_CFakeUnit52_CUnit47 52fcc0, shared_vf_CFakeUnit53_CUnit48 52fdf0, shared_vf_CFakeUnit48_CUnit43 5303a0, shared_vf_CFakeUnit35_CUnit30 530a20, shared_vf_CFakeUnit37_CUnit32 530a50, shared_vf_CFakeUnit9_CUnit4 531a80, shared_vf_CFakeUnit27_CUnit22 532a80, shared_vf_CFakeUnit30_CUnit25 532aa0, shared_vf_CFakeUnit28_CUnit23 532ab0, shared_vf_CFakeUnit18_CUnit13 532da0, thunk_FUN_00700930 53eff0, thunk_FUN_00700930 53f010, shared_vf_CParty24_CWorldClient24 55e940, thunk_FUN_00455930 55ed90, thunk_FUN_004554b0 58a370, thunk_FUN_0046aeb0 58edf0, thunk_FUN_0042d270 596130, shared_vf_CFigure17_CI3DFigure17_CSkyObject17 59f8f0, shared_vf_CFigure19_CI3DFigure19_CSkyObject19 5a2490, shared_vf_CFigure20_CI3DFigure20_CSkyObject20 5a2500, shared_vf_CFigure21_CI3DFigure21_CSkyObject21 5a3800, shared_vf_CFigure22_CI3DFigure22_CSkyObject22 5a4b30, shared_vf_CFigure23_CI3DFigure23_CSkyObject23 5a4bf0, shared_vf_CFigure2_CSkyObject2 5a6c70, shared_vf_CFigure3_CSkyObject3 5a6c80, thunk_FUN_00700930 5e0bf0, thunk_FUN_005fe930 5ee040, thunk_FUN_005fe930 5ee410, thunk_FUN_00600ba0 5f5f50, thunk_FUN_00600ba0 5f5f80, thunk_FUN_00700930 5f9680, thunk_FUN_00700930 5fbe20, thunk_FUN_00600ba0 603cf0, thunk_FUN_005fe930 60e240, thunk_FUN_005fe930 60e2c0, thunk_FUN_005fe930 60e2f0, thunk_FUN_005fe930 60e590, thunk_FUN_005fe930 60e630, thunk_FUN_005fe930 60e660, thunk_FUN_005fe930 6157b0, thunk_FUN_00700930 629d60, thunk_FUN_00700930 62bc50, thunk_FUN_00499e90 62ed80, thunk_FUN_005fe930 635d60, thunk_FUN_005fe930 635d90, shared_vf_CEffectParticle6_CEffectWall6 679580, thunk_FUN_00700930 695ba0, shared_vf_CItemQuick13_CItemSpell13 6a3ba0, shared_vf_CItemQuick14_CItemSpell14 6a3be0, shared_vf_CItemQuick9_CItemSpell9 6a3c40, shared_vf_CItemQuick10_CItemSpell10 6a3c60 |
| CAnimationNode | ctor_dtor_50af40 50af40, vf00 50b1b0, ctor_dtor_50b1d0 50b1d0, vf02 50b370, vf03 50b510 |
| CAnimationSystem2 | ctor_dtor_50c5f0 50c5f0, vf00 50c710, ctor_dtor_50c730 50c730, vf02 50c860, vf03 50c8e0 |
| CAnimationTrack | ctor_dtor_50a5d0 50a5d0, vf00 50a700, ctor_dtor_50a720 50a720, vf02 50a810, vf03 50a8c0, AddTime 50af20 |
| CBridgeManager | ctor_dtor_5612f0 5612f0, vf02 561800, vf03 561840 |
| CBriefingWindow | ParseBriefing 604c50, ShowActor 605da0 |
| CBuilding | ctor_dtor_5297c0 5297c0, vf00 529830, ctor_dtor_529850 529850 |
| CCamera | vf00 40ce10, ctor_dtor_40ce60 40ce60, vf02 40d190 |
| CCampConstructorWindow | Activate 60c830 |
| CCampWindow | Activate 62c210 |
| CClientSpecificManager | ctor_dtor_42d690 42d690, vf00 42d6d0, ctor_dtor_42d6f0 42d6f0, vf14 42d700 |
| CCursor | Load 5e4eb0 |
| CEffectAmbientSound | vf19 677650, vf00 677680, ctor_dtor_6776a0 6776a0, vf18 677830, vf23 677910, vf21 677990, vf22 677ae0, vf09 677bb0, vf10 677c20 |
| CEffectArrow | vf06 67ac50, ctor_dtor_67adf0 67adf0, vf00 67b160, ctor_dtor_67b180 67b180, vf09 67b8a0, vf10 67b960 |
| CEffectArrowFigure | vf18 67ba60, ctor_dtor_67bb50 67bb50, vf00 67bc70, ctor_dtor_67bc90 67bc90, vf21 67c040, vf22 67c130, vf09 67c1c0, vf10 67c250 |
| CEffectLightning | ctor_dtor_679f80 679f80, vf00 67a050, ctor_dtor_67a070 67a070, Reset 67a170, vf09 67a730, vf10 67a7b0 |
| CEffectManager | ctor_dtor_55ee40 55ee40, vf00 55f380, ctor_dtor_55f3a0 55f3a0, vf02 67e110, vf03 67e120, vf14 67e130 |
| CEffectMoshka | ctor_dtor_67c910 67c910, vf00 67ca90, ctor_dtor_67cab0 67cab0, vf18 67cc50, vf06 67cd30 |
| CEffectParticle | ctor_dtor_678770 678770, vf00 678930, ctor_dtor_678950 678950, vf09 6793a0, vf10 6794a0, ctor_dtor_6797a0 6797a0, ctor_dtor_6799a0 6799a0 |
| CEffectPointLight | ctor_dtor_677cf0 677cf0, vf00 677f60, ctor_dtor_677f80 677f80, UpdateClientLight 678290, vf09 678650, vf10 678700 |
| CEffectShowVisibility | ctor_dtor_67c300 67c300, vf00 67c430, ctor_dtor_67c450 67c450, vf18 67c5b0, vf23 67c620, vf21 67c670, vf22 67c760, vf06 67c800, vf09 67c8a0, vf10 67c8d0 |
| CEffectSound | ctor_dtor_675d90 675d90, vf19 676010, vf00 676040, ctor_dtor_676060 676060, vf18 6761c0, vf06 6762b0, vf23 676360, vf21 6763f0, vf22 676510, vf09 6765b0, vf10 676690 |
| CEffectSoundSeq | ctor_dtor_676720 676720, vf19 6768f0, vf00 676920, ctor_dtor_676940 676940, vf18 676ad0, vf23 676bc0, vf21 676f90, vf22 677140, vf06 677210, vf09 677290, vf10 6773f0 |
| CEffectTornado | ctor_dtor_67d630 67d630, vf00 67d7a0, ctor_dtor_67d7c0 67d7c0, vf18 67d930, vf06 67db90, vf21 67dd90, vf22 67de70, vf23 67ded0, vf09 67dfe0, vf10 67e030 |
| CEffectWall | vf00 679980, vf09 679e60, vf10 679e90 |
| CFakeFigure | ctor_dtor_4545c0 4545c0 |
| CFakeUnit | ctor_dtor_4466d0 4466d0, vf00 4468f0, ctor_dtor_446910 446910, vf02 446af0, vf03 446b20, vf62 446ba0 |
| CFigure | ctor_dtor_453240 453240, ctor_dtor_5aa8e0 5aa8e0 |
| CFigureClientSpecific | ctor_dtor_5a7010 5a7010, vf02 5a70e0 |
| CFigureManager | ctor_dtor_5aa070 5aa070, vf00 5aa1a0, vf16 5aa360, vf02 5aa510, vf23 5aae00, MakeAnimationNode 5ac400, ResizeAnimationNode 5ac670, vf03 674f00 |
| CFigure_1 | vf00 453470, ctor_dtor_476fb0 476fb0, ctor_dtor_4a0350 4a0350, ctor_dtor_4d9110 4d9110, ctor_dtor_59f030 59f030, vf02 59f210, vf03 59f2b0, ctor_dtor_5aabc0 5aabc0 |
| CFigure_2 | vf19 5a4cf0, vf21 5a4d60, vf20 5a5c70, vf02 5a6e00, vf03 5a6e10 |
| CHoover | ctor_dtor_461c10 461c10, vf00 461c60, ctor_dtor_461c80 461c80 |
| CI3DFigure | vf00 5aace0, ctor_dtor_5aad00 5aad00, vf02 5e6c60, vf03 5e6d70 |
| CInfoWindow | OnMouseDown 6451d0 |
| CInterface | CreateWindows 613e60 |
| CInventory | vf19 462360, vf20 462390, vf00 462400, ctor_dtor_462420 462420, vf02 462530, vf03 462540, vf14 69fe50, vf15 69ff80, vf09 6a0230, vf10 6a0320, vf13 6a0450 |
| CItemArmor | ctor_dtor_6a2ee0 6a2ee0, vf13 6a3030, vf14 6a3090, vf15 6a3100, vf18 6a3660, vf19 6a36e0, vf20 6a3770, vf09 6a3a30, vf10 6a3a60 |
| CItemConstructor | ctor_dtor_6a5a70 6a5a70, vf00 6a5ac0, ctor_dtor_6a5ae0 6a5ae0, vf02 6a5b50, vf03 6a5b90 |
| CItemLoot | vf13 6a4a70, vf14 6a4ac0, vf15 6a4b10, vf19 6a4b50, vf20 6a4ff0, vf16 6a5430, vf09 6a55f0, vf10 6a5640 |
| CItemManager | vf00 55f3e0, ctor_dtor_55f400 55f400, ctor_dtor_6614b0 6614b0, vf02 6ac740, vf03 6ac7c0, vf14 6ac7d0 |
| CItemQuest | ctor_dtor_6a4360 6a4360, vf21 6a4380, vf13 6a4490, vf14 6a44d0, vf15 6a4500, vf09 6a4530, vf10 6a4550, vf19 6a4580, vf20 6a4770 |
| CItemQuick | ctor_dtor_6a3a90 6a3a90, vf15 6a3c10, vf19 6a3c90, vf20 6a4000 |
| CItemSpell | ctor_dtor_6a56a0 6a56a0, vf15 6a57b0, vf19 6a57e0, vf20 6a58a0 |
| CItemWeapon | ctor_dtor_6a0500 6a0500, vf13 6a0620, vf14 6a0680, vf15 6a06f0, vf17 6a0800, TextureNameOfShape 6a0bb0, vf18 6a0f30, vf19 6a2710, vf20 6a27a0, vf09 6a2e20, vf10 6a2e70 |
| CLeverObject | ctor_dtor_52b1f0 52b1f0, vf00 52b300, ctor_dtor_52b320 52b320, vf02 52b370, vf60 52b3e0, vf50 52b560, vf49 52b650, vf41 52bb50, vf42 52bba0, vf40 52bbc0, vf44 52bbe0, vf09 52bdd0, vf10 52be40 |
| CLight | ctor_dtor_68ce90 68ce90, LightVertex 68d410 |
| CLightManager | ctor_dtor_68d810 68d810, vf00 68d930, ctor_dtor_68d950 68d950, vf14 68da00, vf23 68da90 |
| CLightning | vf02 466ea0, ctor_dtor_468680 468680, vf03 468790, vf00 4687f0, ctor_dtor_468810 468810 |
| CLightningManager | DrawAll 468940, vf02 4689d0, vf13 562d70 |
| CMagicTrapObject | vf00 529f70, ctor_dtor_529f90 529f90, vf02 52a100, vf03 52a200, vf56 52ac90, vf09 52adc0, vf10 52b070 |
| CMainMenuWindow | Deactivate 6328d0 |
| CMapObjectClientSpecific | ctor_dtor_510450 510450, vf17 5104d0, vf18 5104f0, vf00 510540, ctor_dtor_510560 510560, vf02 510770, vf03 5107b0, vf13 5117d0, vf19 511a00 |
| CMapObjectManager | ctor_dtor_5125e0 5125e0, vf00 512630, ctor_dtor_512650 512650, vf02 512660, vf03 512720, vf14 512730, vf23 512a30 |
| CMapObjectServer | ctor_dtor_515610 515610, vf17 5156a0, vf18 5156c0, vf00 515700, ctor_dtor_515720 515720, vf02 5157c0, vf03 5157e0, vf20 515980, vf09 5159d0, vf10 515ac0 |
| CMaterial | ctor_dtor_68db80 68db80, vf00 68dce0, ctor_dtor_68ddf0 68ddf0, vf02 68df50, vf03 68e000 |
| CMaterialManager | ctor_dtor_68e3b0 68e3b0, vf00 68e4e0, ctor_dtor_68e500 68e500, ctor_dtor_68e600 68e600, vf03 68e8b0, vf14 68e8c0 |
| CMultiplayerWindow | Build 61b280 |
| CMusicSystem | ctor_dtor_6b4310 6b4310, vf00 6b4450, ctor_dtor_6b4470 6b4470, vf03 6b4650 |
| CObject3DClientSpecific | ctor_dtor_51ab00 51ab00, vf18 51ad20, vf00 51ad70, ctor_dtor_51ad90 51ad90, vf02 51aec0, vf03 51af70, vf13 51b050, vf15 51b160, vf14 51b210, vf17 51b2d0, vf16 51b2e0, vf19 51b7e0, vf22 51c130, vf24 51c160, vf23 51c1d0, vf26 51c9e0, vf21 51cb30, vf20 51ce60 |
| CObject3DServer | vf18 51ac30, ctor_dtor_51f240 51f240 |
| CParticle | vf03 484710, ctor_dtor_4981b0 4981b0, vf00 4982f0, ctor_dtor_498310 498310 |
| CParticleManager | vf02 497fc0, vf03 497fe0, vf00 5627a0, ctor_dtor_5627c0 5627c0 |
| CParty | ctor_dtor_65f180 65f180, vf00 661150, ctor_dtor_661380 661380, vf03 661600, ctor_dtor_662f70 662f70, ctor_dtor_6632f0 6632f0 |
| CPlayer | ctor_dtor_664bb0 664bb0, vf00 664dc0, ctor_dtor_664de0 664de0, vf02 664fd0, vf03 6650a0 |
| CPlayerClient | ctor_dtor_666cd0 666cd0, vf00 666f70, ctor_dtor_666f90 666f90, vf02 6671f0, vf03 6674d0, vf15 6675c0, vf06 667640, vf23 6679f0, vf14 667b60 |
| CPlayerManager | vf00 55f330, ctor_dtor_55f350 55f350, vf02 66c0b0, vf03 66c0c0, vf14 66c0d0 |
| CScriptWorld | HasScript 4bc330 |
| CServerSpecificManager | ctor_dtor_4d6900 4d6900, vf00 4d6940, ctor_dtor_4d6960 4d6960 |
| CSoundManager | ctor_dtor_6b6000 6b6000, vf00 6b6270, ctor_dtor_6b6590 6b6590, vf02 6b6d30, vf03 6b6d40 |
| CSpell | ctor_dtor_68b280 68b280, FromPrototype 68b3f0, vf04 68b820, vf05 68b8e0, ctor_dtor_68bc10 68bc10 |
| CSpellConstructor | ctor_dtor_683fb0 683fb0, vf00 684040, ctor_dtor_684060 684060, vf02 6840e0, vf03 684110 |
| CSpellManager | ctor_dtor_688800 688800, vf00 688930, ctor_dtor_688950 688950, vf02 688af0, vf03 688b00, vf14 688c90 |
| CStone | ctor_dtor_5298a0 5298a0, vf00 529910, ctor_dtor_529930 529930 |
| CStuffObject | ctor_dtor_529980 529980, vf00 5299f0, ctor_dtor_529a10 529a10 |
| CTerrain | vf00 562b60, ctor_dtor_562b80 562b80, vf02 6c53f0, vf03 6c5490, vf14 6c6030 |
| CTextManager | vf02 4ded20, vf03 4ded40, vf00 562990, ctor_dtor_5629b0 5629b0 |
| CTexture | ctor_dtor_6917d0 6917d0, vf00 6918f0, ctor_dtor_691910 691910, vf02 6919c0, vf03 691a30 |
| CTextureManager | ctor_dtor_691100 691100, vf00 6911f0, ctor_dtor_691210 691210, vf02 6914c0, vf03 6915f0, vf23 691a60, vf24 691ac0 |
| CTorchObject | ctor_dtor_52bee0 52bee0, vf00 52bf40, ctor_dtor_52bf60 52bf60, vf02 52bfc0, vf03 52c150, vf50 52c180, vf49 52c220, vf09 52c280, vf10 52c2a0, vf19 52c2c0 |
| CTorchObjectClientSpecific | ctor_dtor_52d0d0 52d0d0, vf00 52d1b0, ctor_dtor_52d1d0 52d1d0, vf02 52d2a0, vf03 52d2e0, vf13 52d580 |
| CTorchObjectSpecific | ctor_dtor_4d6970 4d6970, vf00 4d6b80, ctor_dtor_4d6ba0 4d6ba0, vf18 52c600, vf02 52c610, vf03 52c680 |
| CTrapObject | ctor_dtor_529a80 529a80, vf00 529b30, vf02 529b60, vf06 529b80, vf41 529d30, vf42 529d60, vf09 529d70, vf10 529da0 |
| CTree | ctor_dtor_5296e0 5296e0, vf00 529750, ctor_dtor_529770 529770 |
| CUnit | ctor_dtor_52e310 52e310 |
| CUnitClientSpecific | ctor_dtor_533ad0 533ad0, vf00 533e80, ctor_dtor_533ea0 533ea0, vf02 533fe0, vf03 5340c0, vf18 534140, vf43 5344a0, vf13 5344e0, vf14 5352b0, vf25 535590, vf41 535b20, vf29 536100, vf30 536220, vf48 5373d0, vf47 537c00, vf51 537d80, vf52 538040, vf32 538140, vf27 5385a0, vf33 538860, vf34 5388c0, vf35 538920, vf50 538960, vf24 539160, vf36 5391b0, vf37 5391e0 |
| CUnitServer | ctor_dtor_545580 545580, vf05 546080, vf00 5460e0, ctor_dtor_546230 546230, vf02 546450, vf03 546470, ActionTicks 54b760, StrikeTicks 54b9c0, vf42 552670, vf38 552cd0, vf53 552f30, vf23 552fb0, vf49 553220, vf44 553260, vf19 5533b0, vf22 553500, vf26 553560, vf27 5535c0, vf28 5535e0, vf29 5535f0, vf25 553610, vf09 555d10, vf10 5561c0, vf24 556760, vf36 5567a0, vf37 5567e0 |
| CUnitStats | ClearArmor 5232b0, SkillPrice 527ff0 |
| CVisFogManager | vf06 4e2760, vf03 4e2850, vf02 4e2990, vf13 562da0, vf00 562dd0, ctor_dtor_562df0 562df0 |
| CVisibilityFog | vf06 4e1ab0, ctor_dtor_4e2290 4e2290, vf02 4e23d0, vf03 4e2440, vf00 4e24c0, ctor_dtor_4e24e0 4e24e0 |
| CWeather | ctor_dtor_5616c0 5616c0, vf03 5617b0, vf00 5617e0, ctor_dtor_561b00 561b00 |
| CWorld | RunConsoleScript 5785d0 |
| CWorldClient | vf40 55fae0, vf31 55fb30, vf32 55fb50, vf41 5618a0, vf42 5618c0, vf29 5618e0, vf30 561930, vf00 561a00 |
| CWorldServer | ctor_dtor_589ed0 589ed0, vf28 58a390, vf29 58a3d0, vf30 58a3f0, vf00 58a430, ctor_dtor_58aad0 58aad0, vf24 594ce0 |
| LightningBolt | PointAt 466270, AddBounds 466fd0 |
| LightningNoise | MakeRow 4660b0 |
| Particle2005 | Emit 48bd50, Move 48c010 |
| Particle200a | Emit 489f70, Move 48a310, Source 48a560 |
| Particle200b | Emit 489320, Move 489a50, Source 489b40 |
| Particle200d | Emit 489be0, Move 489ef0, Source 489f30 |
| Particle2012 | Emit 48d050, Move 48d330 |
| Particle2013 | Emit 48d420, Move 48d7f0 |
| Particle2014 | Source 48e040 |
| Particle2018 | Source 48fbe0 |
| Particle201a | Emit 48e070, Move 48e7c0 |
| Particle2026 | Emit 490130, Move 490990, Source 490b00 |
| Particle202a | Emit 4918e0, Move 492030, Source 4921d0 |
| Particle202e | Emit 490c10, Move 491230, Source 491390 |
| Particle202f | Emit 491420, Move 491830, Source 4918d0 |
| Particle2037 | Emit 48f3b0, Move 48f670, Source 48f6f0 |
| Particle2038 | Emit 492d30, Move 492f40 |
| Particle2039 | Emit 492fc0, Move 493280 |
| Particle203b | Emit 493300, Move 4933f0, Source 4934f0 |
| Particle203c | Emit 493560, Move 493770, Source 4938c0 |
| Particle2040 | Emit 493de0, Move 494170, Source 494340 |
| Particle2043 | Emit 494a10, Move 494cd0 |
| Particle2047 | Emit 4950b0, Move 495120 |
| Particle2048 | Emit 495320, Move 4955d0 |
| Particle2049 | Emit 4957e0, Move 495950 |
| Particle204b | Emit 495a30, Move 495c80, Source 495e00 |
| Particle204c | Emit 494e00, Move 494f30 |
| Particle204d | Emit 495fb0, Move 4961b0 |
| Particle204f | Emit 496270 |
| PathWindow | InitTurnTables 5c42b0 |
| SpellEffect | MakeLightning 67f4c0 |

## Формат JSON

```
{
  "source", "generator", "statuses", "summary",
  "unresolved_python_refs": [{"file", "line", "address"}],   адреса из .py, которых нет в game_exe.c
  "python_index": {"src/x.py": {"Symbol.method": ["45ea20", ...]}},
  "functions": [   по возрастанию адреса, одна функция на строку
    {"address": "45ea20", "name": "FUN_0045ea20", "status": "implemented",
     "lines": 120, "size": 1424,            строк C и байт кода до следующей функции
     "class", "method" | "guess_class",
     "domain", "domain_source", "tier",     подсистема, откуда она взята, backend/data/game
     "callers": 3, "calls": 12,             прямые вызовы внутри game.exe
     "python": [{"file", "symbol", "line", "via": "address"|"name", "constant"?, "inside"?}],
     "covered_by": ["..."],                 для indirect: реализованные вызывающие функции
     "docs": ["first_scan.md: раздел"],
     "strings": ["..."]}                    строки, которые использует функция (подсказка, что она делает)
  ]
}
```

У `library` и `compiler` записаны только адрес, имя, статус (и размер у `library`).
