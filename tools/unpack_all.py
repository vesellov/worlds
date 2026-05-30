import sys
import argparse
import os
import os.path
import json
import shutil
import res, mod, bon, adb, anm, cam, db, fig, lnk, mmp, mp, reg, sec, text, mob,\
       convert_map, compact, convert_model, textures_link, merge_collada

import time

funcs = []
not_copy = ["asi", "dll", "exe", "sav", "mp", "bmp"]
simple_copy = ["mp3", "rtf", "dat", "grp", "bik", "ini", "txt", "wav",
               "mat", "scr"]
archives = ["mq", "mpr", "res", "bon", "mod", "anm"]
convert = ["cam", "reg", "adb", "bon", "anm", "bon", "db", "idb", "mob",
           "fig", "lnk", "mmp", "mp", "pdb", "qdb", "sec", "sdb", "udb", "ldb"]

biomes = {
    'glacier': 'bz8k,bz10k,zone25,zone21',
    'hot_desert': 'zone1,zone2,zone3,zone4,zone5,zone19,zone18,zone17,zone17_1,bz19h,bz18h,bz16h,bz15h,bz13h,bz14h,bz2h,bz3h,bz5h,zone1dun1',
    'cold_desert': 'zone26,bz11k,zone23,zone24,bz21k,bz22k,bz23k,zone22,zone22dun1',
    'tundra': 'bz9k',
    'savanna': 'zonezero,zonemainmenunew,zone35,zone34,zone33,zone32,zone31,bz32j',
    'taiga': 'bz2g',
    'temperate': 'zone9,zone6_2,zone6_1,zone13,zone15',
    'lowland': 'bz13g,bz8g,bz5g,bz6g,bz4g,bz3g,zone71,zone6',
    'tropical': 'zone7,zone8,zone10,zone11,zone16',
    'grassland': 'bz12g,basegipat,zone3obr',
    'dungeon': 'bz14g,bz7g,zone7dun1,zone3dun1,zone9dun1,zone10dun1,zone11dun1,zone11dun2,zone11dun3,zone34dun1,zone20,zonefinal,zone5_1,zone1dun2,zone14',
}

plants_names = [
    'stst7041', 'stst7142',
    'naflbu13', 'naflbu14', 'naflbu17', 'naflbu18', 'naflbu20', 'naflbu21', 'naflbu6', 'naflbu7', 'naflbu8', 'naflbu9', 'nafltr20', 'nafltr21',
    'nafltr22', 'nafltr23', 'nafltr56', 'nafltr57', 'nafltr59', 'nafltr68', 'nafltr69', 'nafltr70', 'nafltr71', 'nafltr72', 'nafltr73', 'nafltr74',
    'nafltr75', 'nafltr76', 'nafltr77', 'nafltr78', 'nafltr82']
plants_winter = ['naflbu17', 'naflbu18', 'naflbu9']
plants_desert = ['nafltr71#tree04', 'nafltr72#tree04', 'nafltr73#tree04', 'nafltr82#tree04']
plants_deadwood = ['nafltr20', 'nafltr70']
plants_stump = ['nafltr21', 'nafltr22', 'nafltr23', 'nafltr74',]
plants_spruce = ['nafltr59']
plants_reed = ['naflbu7']
plants_seaweed = ['naflbu20', 'naflbu21']

bridges = ['stbr1', 'stbr10', 'stbr11', 'stbr12', 'stbr13', 'stbr14', 'stbr15', 'stbr17', 'stbr18', 'stbr19', 'stbr2', 'stbr3', 'stbr6',
           'stbr7', 'stbr8', 'stbr9']

buildings_names = [
    'stst100', 'stst5024', 'stst5125', 'stst5226', 'stst5427', 'stst5528', 'stst8052',
    'stbuho10', 'stbuho11', 'stbuho12', 'stbuho13', 'stbuho14', 'stbuho15', 'stbuho16', 'stbuho17', 'stbuho18', 'stbuho19', 'stbuho20',
    'stbuho21', 'stbuho22', 'stbuho23', 'stbuho24', 'stbuho25', 'stbuho26', 'stbuho27', 'stbuho28', 'stbuho29', 'stbuho3', 'stbuho30',
    'stbuho32', 'stbuho33', 'stbuho34', 'stbuho35', 'stbuho37', 'stbuho38', 'stbuho39', 'stbuho4', 'stbuho40', 'stbuho41', 'stbuho42',
    'stbuho43', 'stbuho44', 'stbuho45', 'stbuho46', 'stbuho47', 'stbuho48', 'stbuho49', 'stbuho5', 'stbuho50', 'stbuho51', 'stbuho52',
    'stbuho53', 'stbuho54', 'stbuho55', 'stbuho56', 'stbuho57', 'stbuho58', 'stbuho6', 'stbuho60', 'stbuho61', 'stbuho62', 'stbuho63',
    'stbuho64', 'stbuho65', 'stbuho66', 'stbuho67', 'stbuho68', 'stbuho7', 'stbuho8', 'stbuho9', 'stbuto1', 'stbuto2', ]

buildings_cultures = {
    'travel': 'stbuho67,stst21,stst146,stst147,stst148,stst149',
    'trade': 'stbuho17,stbuho53,stbuho54,stst125,stst14,stst141,stst30,stst4,stst42,stst44,stst59,stst82,stst83',
    'workshop': 'stbuho50,stst100,stst101,stst114,stst130,stst34,stst72,stst97,stst98,stst99,stto5',
    'undead': 'stbuho11,stbuho23,stbuho39,stbuho48,stbuho55,stst88,stst89,stst90,stst91,stto10,stto11',
    'north': 'stbuho22,stbuho24,stbuho27,stbuho28,stbuho29,stbuho30,stbuho64',
    'magic': 'stbuho49,stbuho5,stbuho56,stbuho6,stst116',
    'orc': 'stbuho7,stbuho8,stga1',
}

gates = ['stga1', 'stga2', 'stga3', 'stga4', 'stga5', 'stga6', 'stga7', 'stga8', 'stst5730']
walls = ['stwa1', 'stwa10', 'stwa11', 'stwa13', 'stwa14', 'stwa15', 'stwa2', 'stwa3', 'stwa4', 'stwa5', 'stwa7', 'stwa8', 'stwa9', 'stwali1',
         'stwame1', 'stwe1', 'stwe3']

humans = ['unhufe', 'unhuma']
orcs = ['unorfe', 'unorma']

broken_models = ['nafltr76', ] # ['unmoco2', 'efcu0', 'nafltr76', 'stbr6', 'stbr7', 'stbr8', 'stbr9']

models = {}
figures = {}
plants = {}
buildings = {}

registry = json.loads(open("../../catalog/figures_names.json", "rt").read())


def to_positive_zero(v):
    if v is -0.0 or v == 0.0:
        return 0.0
    return v


def quantize_coefs(coefs, quant_size=0.5):
    return [to_positive_zero(round(round(c / quant_size, 0) * quant_size, 1)) for c in coefs]


def unpack(args):
    update_progress("STARTED_AT_" + time.strftime("%H:%M:%S"))
    if args.verbose:
        print_log("Source dir: " + args.src_dir)
        print_log("Destination dir: " + args.dst_dir)
        print_log("\nCreate working copy\n")

    count = 0
    if not args.skip_copy:
        for d, dirs, files in os.walk(args.src_dir):
            for file in files:
                if os.path.splitext(file)[1][1:].lower() not in not_copy:
                    dest = os.path.join(args.dst_dir,
                                        os.path.relpath(d, args.src_dir))
                    if not os.path.exists(dest):
                        if args.verbose:
                            print_log("Create folder \"" + dest + "\"")
                        os.makedirs(dest)
                    shutil.copyfile(os.path.join(d, file),
                                    os.path.join(dest, file))
                    count += 1
                    update_progress()
                elif args.verbose:
                    print_log("Skip \"" + file + "\"")

    update_progress("FOLDERS_CONVERTED_" + time.strftime("%H:%M:%S"))
    if args.verbose:
        print_log("{} files copied; no need to read source \
folder anymore".format(count))
        print_log("\nUnpack archives recursively")

    count = 0
    flag = 0 if args.skip_extract else 1
    while flag:

        count += 1
        flag = 0
        if args.verbose:
            print_log("\n{} iteration of file unpacking".format(count))

        arr = []
        for d, dirs, files in os.walk(args.dst_dir):
            for file in files:
                if os.path.splitext(file)[1][1:].lower() in archives:
                    with open(os.path.join(d, file), "rb") as f_tst:
                        magic = f_tst.read(4)
                    if magic == b'\x3C\xE2\x9C\x01':
                        print_log(os.path.join(d, file))
                        update_progress()
                        flag = 1
                        if file[-3:] == "mod":
                            mod.read_info(os.path.join(d, file))
                        elif file[-3:] == "bon":
                            with open(os.path.join(d, file), "rb") as f:
                                try:
                                    tree = res.read_filetree(f)
                                except Exception as exc:
                                    print(os.path.join(d, file), exc)
                                    tree = []
                                for element in tree:
                                    element[0] += ".bon"
                                res.unpack_res(f, tree, os.path.join(d, file))
                        elif file[-3:] == "anm":
                            with open(os.path.join(d, file), "rb") as f:
                                tree = res.read_filetree(f)
                                for element in tree:
                                    element[0] += ".anm"
                                res.unpack_res(f, tree, os.path.join(d, file))
                        else:
                            with open(os.path.join(d, file), "rb") as f:
                                filetree = res.read_filetree(f)
                                res.unpack_res(f, filetree, os.path.join(d, file))
                        if os.path.exists(os.path.join(d, file)):
                            os.remove(os.path.join(d, file))

    update_progress("ARCHIVES_CONVERTED_" + time.strftime("%H:%M:%S"))
    if args.verbose:
        print_log("\nAfter {} iterations all archives unpacked".format(count))
        print_log("\nStart figures folder reorganisation\n")

    try:
        compact.compact_figs(os.path.join(args.dst_dir, "Res", "figures"))
    except:
        if args.verbose:
            print_log("Can't reorganise figures folder!")

    if args.verbose:
        print_log("\nFigures folder reorganised\n")

    if not args.skip_convert:
        if args.verbose:
            print_log("\nConvert files\n")
        static_objs = {} # name, objname, texture, complection, position, rotation, parts
        maps = []
        figs = []
        count = 0
        for d, dirs, files in os.walk(args.dst_dir):
            for file in files:
                file_e = os.path.splitext(file)[1][1:].lower()
                if file_e in convert:
                    count += 1
                    update_progress()
                    file_n = os.path.splitext(file)[0]
                    print_log(os.path.join(d, file))
                    if file_e == "adb":
                        try:
                            info = adb.read_info(os.path.join(d, file))
                        except:
                            print_log("ADB ERROR in file \"{}\"".format(file))
                            info = None
                        if info != None:
                            with open(os.path.join(d, file) + ".yaml", "w") as f:
                                f.write(adb.build_yaml(info))
                    elif file_e == "anm":
                        info = anm.read_info(os.path.join(d, file))
                        if info != None:
                            with open(os.path.join(d, file) + ".yaml", "w") as f:
                                f.write(anm.build_yaml(info))
                    elif file_e == "bon":
                        continue
                    elif file_e == "cam":
                        info = cam.read_info(os.path.join(d, file))
                        if info != None:
                            with open(os.path.join(d, file) + ".yaml", "w") as f:
                                f.write(cam.build_yaml(info))
                    elif file_e in ["idb", "ldb", "pdb", "db", "sdb", "udb", "qdb"]:
                        data = db.read_data(os.path.join(d, file))
                        with open(os.path.join(d, file) + ".csv", "w") as f:
                            f.write(db.build_data(data))
                    elif file_e == "fig":
                        continue
                    elif file_e == "lnk":
                        figs.append([d, file_n])
                        continue
                    elif file_e == "mmp":
                        try:
                            image = mmp.read_image(os.path.join(d, file))
                            image.save(os.path.join(d, file_n) + ".png")
                        except Exception as exc:
                            print(os.path.join(d, file), exc)
                    elif file_e == "mp":
                        maps.append([d, file_n])
                        continue
                    elif file_e == "reg":
                        try:
                            info = reg.read_info(os.path.join(d, file))
                        except UnicodeEncodeError:
                            reg.ENCODE = "cp1251"
                            info = reg.read_info(os.path.join(d, file))
                            reg.ENCODE = "cp866"
                        if info != None:
                            with open(os.path.join(d, file) + ".yaml", "w") as f:
                                try:
                                    f.write(reg.build_yaml(info))
                                except UnicodeEncodeError:
                                    reg.ENCODE = "cp1251"
                                    info = reg.read_info(os.path.join(d, file))
                                    f.write(reg.build_yaml(info))
                                    reg.ENCODE = "cp866"
                    elif file_e == "sec":
                        continue
                    elif file_e == "mob":
                        try:
                            info = mob.read_info(os.path.join(d, file))
                            if info != None:
                                with open(os.path.join(d, file) + ".yaml", "w") as f:
                                    f.write(mob.build_yaml(info))
                            buf_objs = []
                            with open(os.path.join(d, file) + ".yaml") as f:
                                cnt = 0
                                for line in f.readlines():
                                    if len(line) > 100:
                                        continue
                                    buf = line.replace("\n", "").replace("\r", "").\
                                               replace(" ", "").replace("\"", "").split(":")

                                    if cnt > 0:
                                        if buf_objs[-1][var_lbl][-cnt] is None:
                                            buf_objs[-1][var_lbl][-cnt] = float(buf[0][1:])
                                            cnt -= 1
                                        continue
                                    elif cnt < 0:
                                        if buf[0][0] == "-":
                                            buf_objs[-1][6].append(buf[0][1:])
                                            continue
                                        else:
                                            cnt = 0
                                    
                                    if buf[0] == "OBJTEMPLATE":
                                        count += 1
                                        buf_objs[-1][0] = buf[1]
                                        buf_objs[-1][1] = buf[1] + "_" + file_n + "_{}".format(count)
                                    if buf[0] == "OBJPRIMTXTR" and buf_objs[-1][2] is None:
                                        buf_objs[-1][2] = buf[1]
                                    if buf[0] == "OBJCOMPLECTION":
                                        var_lbl = 3
                                        cnt = 3
                                    if buf[0] == "OBJPOSITION":
                                        var_lbl = 4
                                        cnt = 3
                                    if buf[0] == "OBJROTATION":
                                        var_lbl = 5
                                        cnt = 4
                                    if buf[0] == "OBJBODYPARTS":
                                        buf_objs.append([None, None, None, [None, None, None],
                                                         [None, None, None], [None, None, None,
                                                         None], None])
                                        if buf[1] != "None":
                                            buf_objs[-1][6] = []
                                            var_lbl = 6
                                            cnt = -1
                            static_objs.update({file_n.lower(): buf_objs})
                        except Exception as exc:
                            print(file, exc)
                    if False:
                        os.remove(os.path.join(d, file))

        update_progress("COMMON_CONVERTED_" + time.strftime("%H:%M:%S"))
        if args.verbose:
            print_log("{} files converted".format(count))
            print_log("\nConvert game maps\n")

        count = 0
        for i in maps:
            if i[1].lower() not in static_objs:
                if os.path.isfile(os.path.join(args.dst_dir, "Maps",
                                               i[1] + ".mob.yaml")):
                    buf_objs = []
                    with open(os.path.join(args.dst_dir, "Maps",
                                           i[1] + ".mob.yaml")) as f:
                        cnt = 0
                        for line in f.readlines():
                            if len(line) > 100:
                                continue
                            buf = line.replace("\n", "").replace("\r", "").\
                                       replace(" ", "").replace("\"", "").split(":")

                            if cnt > 0:
                                if buf_objs[-1][var_lbl][-cnt] is None:
                                    buf_objs[-1][var_lbl][-cnt] = float(buf[0][1:])
                                    cnt -= 1
                                continue
                            elif cnt < 0:
                                if buf[0][0] == "-":
                                    buf_objs[-1][6].append(buf[0][1:])
                                    continue
                                else:
                                    cnt = 0
                            
                            if buf[0] == "OBJTEMPLATE":
                                count += 1
                                buf_objs[-1][0] = buf[1]
                                buf_objs[-1][1] = buf[1] + "_" + i[1] + "_{}".format(count)
                            if buf[0] == "OBJPRIMTXTR" and buf_objs[-1][2] is None:
                                buf_objs[-1][2] = buf[1]
                            if buf[0] == "OBJCOMPLECTION":
                                var_lbl = 3
                                cnt = 3
                            if buf[0] == "OBJPOSITION":
                                var_lbl = 4
                                cnt = 3
                            if buf[0] == "OBJROTATION":
                                var_lbl = 5
                                cnt = 4
                            if buf[0] == "OBJBODYPARTS":
                                buf_objs.append([None, None, None, [None, None, None],
                                                 [None, None, None], [None, None, None,
                                                 None], None])
                                if buf[1] != "None":
                                    buf_objs[-1][6] = []
                                    var_lbl = 6
                                    cnt = -1
            
                    static_objs.update({i[1].lower(): buf_objs})
                else:
                    static_objs.update({i[1].lower(): None})
            
            map_info = mp.read_info(os.path.join(i[0], i[1] + ".mp"))
            if args.verbose:
                print_log(os.path.join(i[0], i[1]) + 
                      "  + {} textures and {}x{} sectors".format(map_info[3],
                                                                 map_info[1],
                                                                 map_info[2]))
            count += map_info[3] + map_info[1] * map_info[2] + 1
            update_progress(map_info[3] + map_info[1] * map_info[2] + 1)
            for j in range(map_info[3]):
                shutil.copyfile(os.path.join(args.dst_dir, "Res", "textures",
                                             i[1] + "{:03}.png".format(j)),
                                os.path.join(i[0], i[1] + "{:03}.png".format(j)))

            if static_objs[i[1].lower()] is not None:
                unit_pos = [s_obj[4] for s_obj in static_objs[i[1].lower()]]
                if len(unit_pos) == 0:
                    unit_pos = None
            else:
                unit_pos = None
                
            unit_pos = convert_map.convert_map(os.path.join(i[0], i[1]), unit_pos)

            if static_objs[i[1].lower()] is not None:
                list_fpath_inputs = [os.path.join(args.dst_dir, "Res", "figures",
                                                  s_obj[0], s_obj[1] + ".dae") \
                                     for s_obj in static_objs[i[1].lower()]]

                if True:
                    for k, s_obj in enumerate(static_objs[i[1].lower()]):
                        convert_model.convert_model(os.path.join(args.dst_dir, "Res", "figures",
                                                                 s_obj[0], s_obj[0]),
                                                add_suf=s_obj[1][len(s_obj[0]):],
                                                coefs=s_obj[3],
                                                root_pos=unit_pos[k],
                                                root_rot=s_obj[5],
                                                tex_name=s_obj[2],
                                                need_parts=s_obj[6])

                        model_name = s_obj[0].lower()
                        map_name = i[1].lower()
                        if model_name in broken_models:
                            continue
                        # if model_name.lower().startswith('un'):
                            # skip animated models for now
                        #     continue
                        texture_name = s_obj[2].lower()
                        related_biomes = set()
                        for biome, maps_list in biomes.items():
                            if map_name in maps_list.split(','):
                                related_biomes.add(biome)
                        if f'{model_name}#{texture_name}' in plants_desert:
                            related_biomes = {'hot_desert', }
                        model_type = 'unknown'
                        if model_name in buildings_names:
                            model_type = 'building'
                        elif model_name in bridges:
                            model_type = 'bridge'
                        elif model_name in gates:
                            model_type = 'gate'
                        elif model_name in walls:
                            model_type = 'wall'
                        elif model_name in humans:
                            model_type = 'human'
                        elif model_name in orcs:
                            model_type = 'orc'
                        # elif model_name in plants_winter:
                        #     model_type = 'frozen_bush'
                        elif model_name in plants_deadwood:
                            model_type = 'deadwood'
                        if model_type == 'unknown':
                            if model_name.startswith('unan'):
                                model_type = 'animal'
                            elif model_name.startswith('unmo'):
                                model_type = 'monster'
                            elif model_name.startswith('stbuho'):
                                model_type = 'house'
                            elif model_name.startswith('stga'):
                                model_type = 'gate'
                            elif model_name.startswith('stwa'):
                                model_type = 'wall'
                            elif model_name.startswith('naflbu'):
                                # if model_name in ['naflbu6', ] and texture_name == 'tree03':
                                #     model_type = 'frozen_bush'
                                # else:
                                model_type = 'bush'
                            elif model_name.startswith('nafltr'):
                                if model_name in ['nafltr59', 'nafltr57', 'nafltr56', 'nafltr69', ] and texture_name == 'tree03':
                                    model_type = 'frozen_tree'
                                else:
                                    model_type = 'tree'
                            elif texture_name.count('stone'):
                                model_type = 'stone'
                            elif texture_name.count('skeleton'):
                                model_type = 'skeleton'
                            elif texture_name.count('goblin'):
                                model_type = 'goblin'
                            elif texture_name.count('ruins'):
                                model_type = 'ruins'
                            elif texture_name.count('house'):
                                model_type = 'house'
                            elif texture_name.count('mushroom'):
                                model_type = 'mushroom'
                        if model_type == 'unknown' and (texture_name.count('tree') or model_name in plants_names):
                            # if model_name in plants_winter:
                            #     model_type = 'frozen_bush'
                            if model_name in plants_deadwood:
                                model_type = 'deadwood'
                            elif model_name.startswith('naflbu'):
                                # if model_name in ['naflbu6', ] and texture_name == 'tree03':
                                #     model_type = 'frozen_bush'
                                # else:
                                model_type = 'bush'
                            else:
                                if model_name in ['nafltr59', 'nafltr57', 'nafltr56', 'nafltr69'] and texture_name == 'tree03':
                                    model_type = 'frozen_tree'
                                else:
                                    if model_name.startswith('stst'):
                                        model_type = 'object'
                                    else:
                                        model_type = 'tree'
                        culture = 'unknown'
                        best_culture = None
                        for culture_name, models_list in buildings_cultures.items():
                            models_list = models_list.split(',')
                            if model_name in models_list:
                                best_culture = culture_name
                                break
                        if best_culture:
                            model_type = 'building'
                            culture = best_culture
                        else:
                            if texture_name.count('kanian'):
                                culture = 'north'
                            elif texture_name.count('hadogan'):
                                culture = 'south'
                            elif texture_name.count('orc') and not texture_name.count('torch'):
                                culture = 'orc'
                            elif texture_name.count('goblin'):
                                culture = 'orc'
                            elif texture_name.count('ogr'):
                                culture = 'orc'
                            elif texture_name.count('masterhouse'):
                                culture = 'orc'
                            elif texture_name.count('necro'):
                                culture = 'undead'
                            elif texture_name.count('crypt'):
                                culture = 'undead'
                            elif texture_name.count('flyer'):
                                culture = 'undead'
                            elif texture_name.count('dgun'):
                                culture = 'magic'
                        if model_name not in models:
                            models[model_name] = []
                        coefs = s_obj[3]
                        coefs[2] = coefs[2] if coefs[2] >= 0 else -coefs[2]
                        parts = s_obj[6]
                        parts_str = ':'.join(sorted(parts)) if parts else 'null'
                        coefs_q = quantize_coefs(coefs)
                        coefs_q_short = quantize_coefs(coefs, quant_size=0.1)
                        coefs_str = ':'.join([str(c) for c in coefs_q])
                        coefs_short_str = ':'.join([str(c) for c in coefs_q_short])
                        model_template_key = f'{model_name}#{texture_name}#{parts_str}'
                        model_template_key_root = f'{model_name}#{texture_name}#null'
                        building_template_key = f'{model_name}#{texture_name}#{parts_str}#{culture}#{model_type}'
                        # model_template_short_key = f'{model_name}#{texture_name}#{parts_str}'
                        model_data = None
                        existing_index = None
                        existing_index_root = None
                        for j in range(len(models[model_name])):
                            m = models[model_name][j]
                            if m.get('i') == model_template_key_root:
                                existing_index_root = j
                                if model_template_key_root == model_template_key:
                                    model_data = m.copy()
                                    existing_index = j
                                    break
                                continue
                            if m.get('i') == model_template_key:
                                model_data = m.copy()
                                existing_index = j
                                break
                        if existing_index is not None:
                            if not model_data['c'].count(coefs_short_str):
                                model_data['c'] += f' {coefs_short_str}'
                            for b in related_biomes:
                                if b not in model_data['b']:
                                    model_data['b'].append(b)
                                    model_data['b'].sort()
                            if map_name not in model_data['l']:
                                model_data['l'].append(map_name)
                                model_data['l'].sort()
                        else:
                            model_data = {
                                'i': model_template_key,
                                'm': model_name,
                                't': texture_name,
                                'c': coefs_short_str,
                                'p': s_obj[6],
                                'k': model_type,
                                'b': list(related_biomes),
                                'l': [map_name, ],
                                'u': culture,
                            }
                        if model_type in ['building', 'house', ]:
                            model_data['s'] = ''
                            if texture_name.count('tower') or texture_name.count('pyramid'):
                                model_data['s'] = 'tower'
                        if model_type == 'deadwood':
                            if abs(coefs[0]) >= 1.5 or abs(coefs[1]) >= 1.5 or abs(coefs[2]) >= 1.5:
                                continue
                        if existing_index is not None:
                            models[model_name][existing_index] = model_data
                        else:
                            if model_data['i'] not in [v['i'] for v in models[model_name]]:
                                models[model_name].append(model_data)
                        if existing_index_root is None:
                            model_data_root = {
                                'i': model_template_key_root,
                                'm': model_name,
                                't': texture_name,
                                'c': '0.0:0.0:0.0',
                                'p': None,
                                'k': model_type,
                                'b': [],
                                'l': [],
                                'u': culture,
                            }
                            if model_data['i'] not in [v['i'] for v in models[model_name]]:
                                models[model_name].insert(0, model_data)
                        if model_type in ['building', 'house', 'bridge', 'gate', 'wall']:
                            if building_template_key not in buildings:
                                buildings[building_template_key] = ''
                            if not buildings[building_template_key].count(coefs_short_str):
                                if buildings[building_template_key]:
                                    buildings[building_template_key] += f' {coefs_short_str}'
                                else:
                                    buildings[building_template_key] = coefs_short_str

                ind = 0
                while ind < len(list_fpath_inputs):
                    if os.path.isfile(list_fpath_inputs[ind]):
                        ind += 1
                    else:
                        print_log("File {} not exist".format(list_fpath_inputs.pop(ind)))
                
                list_fpath_inputs.append(os.path.join(i[0], i[1]) + ".dae")
                merge_collada.merge_dae_files(list_fpath_inputs,
                                              os.path.join(i[0], i[1] + "_full.dae"),
                                              i[1] + "_full")
                
                for s_obj in static_objs[i[1].lower()]:
                    if not os.path.isfile(os.path.join(i[0], s_obj[2] + ".png")):
                        if os.path.exists(os.path.join(args.dst_dir, "Res", "textures", s_obj[2] + ".png")):
                            shutil.copyfile(os.path.join(args.dst_dir, "Res", "textures", s_obj[2] + ".png"), os.path.join(i[0], s_obj[2] + ".png"))

                for st_dae in list_fpath_inputs[:-1]:
                    if os.path.exists(st_dae):
                        os.remove(st_dae)

            for j in range(map_info[1]):
                for k in range(map_info[2]):
                    if os.path.exists(os.path.join(i[0], i[1] + "{:03}{:03}.sec".format(j, k))):
                        os.remove(os.path.join(i[0], i[1] + "{:03}{:03}.sec".format(j, k)))

            if False:
                for j in range(map_info[3]):
                    if os.path.exists(os.path.join(i[0], i[1] + "{:03}.png".format(j))):
                        os.remove(os.path.join(i[0], i[1] + "{:03}.png".format(j)))

            if os.path.exists(os.path.join(i[0], i[1] + ".mp")):
                os.remove(os.path.join(i[0], i[1] + ".mp"))

        update_progress("MAPS_CONVERTED_" + time.strftime("%H:%M:%S"))
        if args.verbose:
            print_log("{} files converted ({} maps)".format(count, len(maps)))
            print_log("\nConvert models\n")

        count = 0
        for i in figs:
            # print_log(i[0])
            try:
                if convert_model.convert_model(os.path.join(i[0], i[1])) is not None:
                    continue
            except Exception as e:
                print_log(str(e))
                continue
            for j in textures_link.textures.get(i[1], []):
                try:
                    shutil.copyfile(os.path.join(args.dst_dir, "Res",
                                                 "textures",
                                                 j + ".png"),
                                    os.path.join(i[0], j + ".png"))
                except Exception as e:
                    print_log(str(e))

            fig_tree = lnk.read_info(os.path.join(i[0], i[1] + ".lnk"))
            filelist = convert_model.flat_tree(fig_tree)
            count += 1 + len(filelist) * 2
            update_progress(1 + len(filelist) * 2)
            model_name = i[1].lower()
            if model_name in figures:
                raise Exception("Duplicate figure name: {}".format(model_name))
            figures[model_name] = fig_tree
            # print(model_name, fig_tree)

            if False:
                for j in filelist:
                    try:
                        os.remove(os.path.join(i[0], j + ".fig"))
                    except:
                        pass
                    try:
                        os.remove(os.path.join(i[0], j + ".bon"))
                    except:
                        pass
                try:
                    os.remove(os.path.join(i[0], i[1] + ".lnk"))
                except:
                    pass
        
        update_progress("FIGURES_CONVERTED_" + time.strftime("%H:%M:%S"))
        if args.verbose:
            print_log("{} files converted".format(count))

    count = 0
    if args.text_joint and os.path.isdir(os.path.join(args.dst_dir,
                               "Res", "texts")):
        if args.verbose:
            print_log("\nJoint game strings\n")
            
        with open(os.path.join(args.dst_dir,
                               "Res", "texts", "texts.yaml"), "w") as file:
            file.write(text.build_yaml(text.read_info(os.path.join(args.dst_dir,
                                                                   "Res", "texts"))))
        for file in os.listdir(os.path.join(args.dst_dir, "Res", "texts")):
            if "." in file or "(" in file:
                continue
            os.remove(os.path.join(args.dst_dir, "Res", "texts", file))
            count += 1
            update_progress()
            
        with open(os.path.join(args.dst_dir, "Res",
                               "textslmp", "textslmp.yaml"), "w") as file:
            file.write(text.build_yaml(text.read_info(os.path.join(args.dst_dir,
                                                                   "Res",
                                                                   "textslmp"))))
        for file in os.listdir(os.path.join(args.dst_dir, "Res", "textslmp")):
            if "." in file or "(" in file:
                continue
            os.remove(os.path.join(args.dst_dir, "Res/textslmp", file))
            count += 1
            update_progress()

        update_progress("TEXTS_CONVERTED_" + time.strftime("%H:%M:%S"))
        if args.verbose:
            print_log("{} files converted".format(count))

    for model_template_info, extra_info in registry.items():
        model_name, texture_name, parts_list = model_template_info.split('#')
        if model_name in broken_models:
            continue
        if texture_name.count('|'):
            texture_name = texture_name.split('|')[0]
        if model_name.startswith('un'):
            # skip animated models for now
            continue
        missing = False
        if model_name not in models:
            missing = True
        else:
            missing = True
            for m in models[model_name]:
                if m['l']:
                    missing = False
                    break
        if not missing:
            continue
        if model_name not in models:
            models[model_name] = []
        model_data = {
            'i': model_template_info,
            'm': model_name,
            't': texture_name,
            'c': '0.0:0.0:0.0',
            'p': parts_list.split(':') if parts_list != 'null' else None,
            'k': 'unknown',
            'b': [],
            'l': [],
            'u': 'unknown',
        }
        if model_data['i'] not in [v['i'] for v in models[model_name]]:
            models[model_name].append(model_data)

    for model_name, variants in models.items():
        found = False
        first_texture_name = None
        for v in variants:
            if not first_texture_name:
                first_texture_name = v['t']
            if v['i'].endswith('#null'):
                found = True
                break
        if found:
            continue
        if not first_texture_name:
            continue
        model_template_info_raw = f'{model_name}#{first_texture_name}#null'
        model_data_raw = {
            'i': model_template_info_raw,
            'm': model_name,
            't': first_texture_name,
            'c': '0.0:0.0:0.0',
            'p': None,
            'k': 'unknown',
            'b': [],
            'l': [],
            'u': 'unknown',
        }
        if model_data_raw['i'] not in [v['i'] for v in models[model_name]]:
            models[model_name].insert(0, model_data_raw)

    for model_name, variants in models.items():
        for i in range(len(variants)):
            is_plant = False
            if model_name in plants_deadwood:
                models[model_name][i]['k'] = 'deadwood'
                is_plant = True
            elif model_name in plants_stump:
                models[model_name][i]['k'] = 'stump'
                is_plant = True
            elif model_name in plants_spruce:
                models[model_name][i]['k'] = 'spruce'
                is_plant = True
            elif model_name in plants_reed:
                models[model_name][i]['k'] = 'reed'
                is_plant = True
            elif model_name in plants_seaweed:
                models[model_name][i]['k'] = 'seaweed'
                is_plant = True
            elif model_name.startswith('naflbu'):
                models[model_name][i]['k'] = 'bush'
                is_plant = True
            elif model_name.startswith('nafltr'):
                models[model_name][i]['k'] = 'tree'
                is_plant = True
            if models[model_name][i]['t'] == 'tree03':
                models[model_name][i]['w'] = 'winter'
                if model_name in ['stst70', 'stst116']:
                    models[model_name][i]['k'] = 'stump'
                    is_plant = True
            elif models[model_name][i]['t'] == 'tree04':
                models[model_name][i]['w'] = 'desert'
            elif models[model_name][i]['t'] in ['tree01', 'tree05', 'tree06']:
                models[model_name][i]['w'] = 'autumn'
            elif models[model_name][i]['t'] == 'tree02':
                models[model_name][i]['w'] = 'temperate'
            elif models[model_name][i]['t'].count('mushroom'):
                models[model_name][i]['k'] = 'mushroom'
                models[model_name][i]['w'] = 'generic'
                if models[model_name][i]['t'] in ['mushroom01_2', 'mushroom02_2']:
                    is_plant = False
                else:
                    is_plant = True
            if is_plant:
                model_type = models[model_name][i]['k']
                model_weather = models[model_name][i].get('w')
                if not model_weather:
                    model_weather = 'generic'
                    models[model_name][i]['w'] = model_weather
                plant_key = f'{model_weather}_{model_type}'
                model_template_key = models[model_name][i]['i']
                coefs_short_str = models[model_name][i]['c']
                if plant_key not in plants:
                    plants[plant_key] = {}
                if model_template_key not in plants[plant_key]:
                    plants[plant_key][model_template_key] = ''
                if not (plants[plant_key][model_template_key] or '').count(coefs_short_str):
                    if plants[plant_key][model_template_key]:
                        plants[plant_key][model_template_key] += f' {coefs_short_str}'
                    else:
                        plants[plant_key][model_template_key] = coefs_short_str

    for model_name in list(models.keys()):
        model_variants = models[model_name]
        first_variant = model_variants[0]
        model_template_key_root = f'{first_variant["m"]}#{first_variant["t"]}#null'
        model_template_key_index = None
        for i in range(len(model_variants)):
            if model_variants[i]['i'] == model_template_key_root:
                model_template_key_index = i
                break
        if model_template_key_index is None:
            model_data_raw = {
                'i': model_template_key_root,
                'm': first_variant['m'],
                't': first_variant['t'],
                'c': first_variant['c'],
                'p': None,
                'k': first_variant['k'],
                'b': [],
                'l': [],
                'u': first_variant['u'],
            }
            if model_template_key_root not in [v['i'] for v in models[model_name]]:
                models[model_name].insert(0, model_data_raw)
            continue
        if model_template_key_index != 0:
            model_data_raw = models[model_name].pop(model_template_key_index)
            if model_data_raw['i'] not in [v['i'] for v in models[model_name]]:
                models[model_name].insert(0, model_data_raw)
            continue

    for model_name in models.keys():
        if model_name in figures:
            continue
        fig_tree = lnk.read_info(os.path.join(args.dst_dir, 'Res', 'figures', model_name, model_name + ".lnk"))
        figures[model_name] = fig_tree

    figures_parts = {}
    for model_name, fig_tree in figures.items():
        figures_parts[model_name] = ':'.join(convert_model.flat_tree(fig_tree))

    import json
    open('figures_samples.json', 'wt').write(json.dumps(models, indent=2))
    open('figures.json', 'wt').write(json.dumps(figures, indent=2))
    open('figures_parts.json', 'wt').write(json.dumps(figures_parts, indent=2))
    open('plants.json', 'wt').write(json.dumps(plants, indent=2))
    open('buildings.json', 'wt').write(json.dumps(buildings, indent=2))


def phldr(val=1):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", add_help=True)
    parser.add_argument("src_dir", type=str,
                    help="game folder")
    parser.add_argument("dst_dir", type=str,
                    help="output folder")
    parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
    parser.add_argument("-c", "--skip_copy", action="store_true",
                    help="skip file copy")
    parser.add_argument("-s", "--skip_extract", action="store_true",
                    help="skip archive extraction")
    parser.add_argument("-r", "--skip_convert", action="store_true",
                    help="skip files convertion")
    parser.add_argument("-t", "--text_joint", action="store_true",
                    help="joint game strings")

    if len(sys.argv) > 1:
        print_log = print
        update_progress = phldr
        args = parser.parse_args()
        unpack(args)
