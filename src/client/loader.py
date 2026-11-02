import io
import os
import json
import hashlib
import tarfile
import time
import shutil
import threading
import traceback

import requests

from kivy.clock import Clock

from lib import res

_Debug = True

REQUEST_TIMEOUT = (10, 60)  # seconds to connect, seconds to wait for the next piece of data

# network errors and temporary errors of the hosting are repeated, the pause doubles every time: 2, 4, 8 seconds
MAX_RETRIES = 3
RETRY_DELAY = 2.0
RETRY_STATUSES = {500, 502, 503, 504}


class DownloadError(Exception):
    pass


def download_file(dest_file_path, url, validate=None, chunk_size=64 * 1024):
    """
    Downloads the file as it is into <dest>.part, the file is moved in place only when it arrived completely
    and validate(path) did not raise, so an interrupted download never leaves a broken file which looks complete
    on the next start. Blocks for the whole download, so it must run in a background thread.
    """
    if os.path.isfile(dest_file_path):
        return dest_file_path
    os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
    part_file_path = dest_file_path + '.part'
    try:
        download_url(url, part_file_path, chunk_size)
        if validate:
            validate(part_file_path)
        os.replace(part_file_path, dest_file_path)
    finally:
        if os.path.isfile(part_file_path):
            os.remove(part_file_path)
    if _Debug:
        print(f'loader: downloaded {dest_file_path} from {url}, {os.path.getsize(dest_file_path)} bytes')
    return dest_file_path


def download_url(url, file_path, chunk_size):
    reason = None
    for attempt in range(MAX_RETRIES + 1):
        if attempt:
            wait = RETRY_DELAY * 2 ** (attempt - 1)
            if _Debug:
                print(f'loader: {url} {reason}, retry {attempt}/{MAX_RETRIES} in {wait:.0f} seconds')
            time.sleep(wait)
        try:
            with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
                if r.status_code == 200:
                    # opened again on every attempt, so what a failed attempt wrote is dropped
                    with open(file_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size):
                            f.write(chunk)
                    return
                if r.status_code not in RETRY_STATUSES:
                    raise DownloadError(f'{url} rejected with HTTP {r.status_code}')
                reason = f'HTTP {r.status_code}'
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
            reason = exc.__class__.__name__
    raise DownloadError(f'{url} failed after {MAX_RETRIES} retries, last error: {reason}')


def validate_res(file_path):
    with open(file_path, 'rb') as f:
        res.read_res_filetree(f)


#--- STORES ---

# store <store_id> keeps <kind>/<name>.res files: figures/<template>.res, textures/<group>.res, redress/<group>.res,
# see tools/split_figures.py and tools/split_textures.py
STORE_URL_PREFIX = os.environ.get('WORLDS_STORE_URL', 'http://p2p-alice.ai/store')

# catalog/textures_store.json says which .res file of the store holds the texture:
# {store_id: {texture: "<kind>/<group>"}}, for example {"2": {"unhumaskin_41": "redress/unhuma_skin"}}
TEXTURES_STORE_FILE_NAME = 'textures_store.json'

# flags downloaded from <STORE_URL_PREFIX>flags/ into textures/flags/, those are used instead of the textures of the stores
SHARED_TEXTURES = ('flag_east', 'flag_north', 'flag_south', 'flag_undead', 'gipat2sys', 'gipatsys', 'ingossys', 'jigransys', 'quater01', 'suslangersys', )

_StoreIndex = None
_StoreLock = threading.Lock()

# number of the land tiles textures: textures/land/0.png ... 28.png
LAND_TILES_COUNT = 29


def inventory_list():
    """
    Files the client must have before the game starts: list of (folder, file name, url prefix),
    every file is downloaded as it is from <url prefix><file name>, see download_file().
    Models and textures are not here: they are downloaded from the stores when the game server tells
    which are needed, see StoreLoader. Files of the catalog/ folder are part of the client.
    Files of the land (map, buildings, plants) belong to the biome, those come from the game server, see extract_biome().
    """
    files = [
        ('textures/nature', 'water1.png', f'{STORE_URL_PREFIX}/nature/'),
        ('textures/nature', 'sky1.png', f'{STORE_URL_PREFIX}/nature/'),
    ]
    for i in range(LAND_TILES_COUNT):
        files.append(('textures/land', f'{i}.png', f'{STORE_URL_PREFIX}/land/'))
    for texture in SHARED_TEXTURES:
        files.append(('textures/flags', f'{texture}.png', f'{STORE_URL_PREFIX}/flags/'))
    return files


def store_res_path(store_id, kind, name):
    return res.data_path(os.path.join('data', 'store', str(store_id), kind, name + '.res'))


def store_res_url(store_id, kind, name):
    store_name = f'ei{store_id}'
    return f'{STORE_URL_PREFIX}/{store_name}/{kind}/{name}.res'


def store_index(store_id):
    global _StoreIndex
    if _StoreIndex is None:
        with open(res.catalog_path(TEXTURES_STORE_FILE_NAME), 'rt') as f:
            _StoreIndex = json.loads(f.read())
    return _StoreIndex.get(str(store_id), {})


def texture_name(texture):
    # textures of the templates can have the layer number at the end: unhumaskin_41:0
    return texture.split(':')[0].lower()


def texture_location(texture, store_id):
    """
    Returns (kind, group) of the .res file in the store which holds the texture, None when the store has no such texture.
    """
    location = store_index(store_id).get(texture_name(texture))
    if not location:
        return None
    kind, group = location.split('/', 1)
    return kind, group


def texture_png_path(texture, store_id):
    # textures of different stores can have the same name, so every store has its own folder
    return res.data_path(os.path.join('textures', f'store{store_id}', texture_name(texture) + '.png'))


def shared_texture_png_path(texture):
    # textures which are not part of the stores: flags, see SHARED_TEXTURES
    return res.data_path(os.path.join('textures', 'flags', texture_name(texture) + '.png'))


def template_res_path(template, store_id):
    return store_res_path(store_id, 'figures', template.lower())


def template_models_dir(store_id):
    # models are unpacked into models/store<store_id>/<template>/, see dat.ModelData.unpack_figure_data()
    return res.data_path(os.path.join('models', f'store{store_id}'))


def is_template_available(template, store_id):
    # models which were already unpacked earlier do not need the .res file anymore
    return os.path.isfile(template_res_path(template, store_id)) or os.path.isdir(os.path.join(template_models_dir(store_id), template.lower()))


def missing_files(templates=(), textures=()):
    """
    Files which must be downloaded from the stores before the models can be rendered: sorted list of (store_id, kind, name).
    templates and textures are lists of (name, store_id).
    """
    files = set()
    for template, store_id in templates:
        if not is_template_available(template, store_id):
            files.add((store_id, 'figures', template.lower()))
    for texture, store_id in textures:
        if texture_name(texture) in SHARED_TEXTURES or os.path.isfile(texture_png_path(texture, store_id)):
            continue
        location = texture_location(texture, store_id)
        if not location:
            # not part of the store, it is taken from the shared textures or reported when the model is built
            continue
        kind, group = location
        if not os.path.isfile(store_res_path(store_id, kind, group)):
            files.add((store_id, kind, group))
    return sorted(files)


def download_store_file(store_id, kind, name):
    """
    Downloads one .res file of the store, blocks for the whole download so must run in a background thread.
    The lock makes sure the same file is not downloaded by two threads at the same moment.
    """
    with _StoreLock:
        return download_file(store_res_path(store_id, kind, name), store_res_url(store_id, kind, name), validate=validate_res)


def ensure_texture(texture, store_id):
    """
    Returns the path to the .png file of the texture, unpacks it from the .res file of the store when needed.
    Normally the .res file is already downloaded by StoreLoader, otherwise it is downloaded right here,
    which freezes the window for the time of the download. Returns None when the texture is not found.
    """
    if texture_name(texture) in SHARED_TEXTURES:
        return shared_texture_png_path(texture)
    png_file_path = texture_png_path(texture, store_id)
    if os.path.isfile(png_file_path):
        return png_file_path
    location = texture_location(texture, store_id)
    if location:
        kind, group = location
        res_file_path = store_res_path(store_id, kind, group)
        if not os.path.isfile(res_file_path):
            if _Debug:
                print(f'loader: WARNING, {kind}/{group}.res of store {store_id} was not downloaded before, downloading now for texture {texture}')
            try:
                download_store_file(store_id, kind, group)
            except Exception:
                traceback.print_exc()
                return None
        if res.unpack_texture(res_file_path, os.path.dirname(png_file_path), texture_name(texture)):
            return png_file_path
        return None
    shared_file_path = shared_texture_png_path(texture)
    if os.path.isfile(shared_file_path):
        return shared_file_path
    if _Debug:
        print(f'loader: WARNING, texture {texture} not found in store {store_id}')
    return None


class StoreLoader(object):
    """
    Downloads files from the stores one by one in a background thread: models and textures needed to render something.
    Files which were downloaded before are skipped, see missing_files().
    """

    def __init__(self, files, on_progress=None, on_done=None):
        self.files = list(files)
        self.on_progress = on_progress
        self.on_done = on_done
        self.failed = []
        self.thread = None

    def start(self):
        if _Debug:
            print(f'store loader: {len(self.files)} files need to be downloaded')
        if not self.files:
            Clock.schedule_once(lambda dt: self.finish())
            return
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        for i, (store_id, kind, name) in enumerate(self.files):
            try:
                download_store_file(store_id, kind, name)
            except Exception as exc:
                if _Debug:
                    print(f'store loader: failed to download {store_res_url(store_id, kind, name)}: {exc}')
                self.failed.append(f'{kind}/{name}')
            Clock.schedule_once(lambda dt, i=i, name=name: self.progress(i + 1, name))
        Clock.schedule_once(lambda dt: self.finish())

    def progress(self, done, name):
        if self.on_progress:
            self.on_progress(done, len(self.files), name)

    def finish(self):
        if self.failed:
            print(f'store loader: ERROR, files not available: {self.failed}')
        if self.on_done:
            self.on_done(self.failed)


#--- BIOMES ---

# keeps sha256 of the archive the files of the biome were extracted from
BIOME_SHA256_FILE_NAME = '.sha256'


def biome_dir(location_id):
    # files of the biome received from the game server: biomes/<server name>_<biome id>/
    return res.data_path(os.path.join('biomes', location_id))


def cached_biome_sha256(location_id):
    sha256_file_path = os.path.join(biome_dir(location_id), BIOME_SHA256_FILE_NAME)
    if not os.path.isfile(sha256_file_path):
        return None
    with open(sha256_file_path, 'rt') as f:
        return f.read().strip() or None


def extract_biome(archive_data, location_id, sha256):
    """
    Checks the archive received from the game server and extracts it into biomes/<location_id>/.
    Files are extracted into a new folder first, the old files are replaced only when all was extracted,
    so a broken download never leaves a half written biome. Blocks, so better runs in a background thread.
    """
    if hashlib.sha256(archive_data).hexdigest() != sha256:
        raise ValueError(f'biome {location_id} archive is damaged, sha256 does not match')
    dest_dir = biome_dir(location_id)
    part_dir = dest_dir + '.part'
    shutil.rmtree(part_dir, ignore_errors=True)
    os.makedirs(part_dir)
    with tarfile.open(fileobj=io.BytesIO(archive_data), mode='r:gz') as tar:
        # "data" filter refuses absolute paths, links and anything pointing outside of the folder
        tar.extractall(part_dir, filter='data')
    with open(os.path.join(part_dir, BIOME_SHA256_FILE_NAME), 'wt') as f:
        f.write(sha256)
    shutil.rmtree(dest_dir, ignore_errors=True)
    os.replace(part_dir, dest_dir)
    if _Debug:
        print(f'loader: biome {location_id} extracted into {dest_dir}, {len(archive_data)} bytes archive')
    return dest_dir
