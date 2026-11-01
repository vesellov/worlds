import os
import gzip
import zlib
import time
import shutil
import random
import threading
import traceback
import email.utils

import requests

try:
    import lzma
except ImportError:
    # lzma is an optional module, python-for-android, kivy-ios and Pyodide builds can be without it
    lzma = None

from kivy.clock import Clock

from lib import res

_Debug = True

# where separate <template>.res files are downloaded from, see tools/split_figures_res.py
# while empty the models are taken out of the whole data/figures.res which is downloaded at the start
FIGURES_URL_PREFIX = os.environ.get('WORLDS_FIGURES_URL', '')

# every file is stored on GitHub as .xz and as .gz, see tools/compress_files.py, this sets which one is downloaded:
# "xz" is smaller, "gz" decompresses faster and only needs zlib. When lzma is not available .gz is downloaded anyway
DOWNLOAD_FORMAT = os.environ.get('WORLDS_DOWNLOAD_FORMAT', 'xz').strip().lower()

CLIENT_VERSION = '0.1'
# GitHub asks every client to send a User-Agent naming the application, so the owner can be reached if it misbehaves
USER_AGENT = f'WorldsGameClient/{CLIENT_VERSION} (+https://github.com/bitdust-io/worlds) python-requests/{requests.__version__}'

# random pause before every request, so many clients started at the same moment do not hit GitHub all together
JITTER_MIN = 0.5
JITTER_MAX = 2.0

# when GitHub rejects or drops a request it is repeated, the pause doubles every time: BACKOFF_BASE * 2 ** attempt,
# never more than BACKOFF_MAX, and a random part of it is added so the clients do not retry in sync
MAX_RETRIES = 6
BACKOFF_BASE = 2.0
BACKOFF_MAX = 120.0
# GitHub asks to wait at least a minute after a rate limit response which does not say how long to wait
RATE_LIMIT_MIN_WAIT = 60.0
# give up instead of waiting longer than this, for example when the hourly rate limit is used up
MAX_WAIT = 15 * 60.0

REQUEST_TIMEOUT = (10, 60)  # seconds to connect, seconds to wait for the next piece of data
RETRY_STATUSES = {500, 502, 503, 504}


class DownloadError(Exception):
    pass


class NotFoundError(DownloadError):
    pass


def download_formats():
    """
    Formats to try in order: the configured one first when it can be decompressed here, .gz is always the last resort.
    """
    formats = []
    if DOWNLOAD_FORMAT == 'xz' and lzma is not None:
        formats.append('xz')
    elif DOWNLOAD_FORMAT not in ('xz', 'gz') and _Debug:
        print(f'loader: unknown download format {DOWNLOAD_FORMAT!r}, using gz')
    formats.append('gz')
    return formats


def compressed_url(url_prefix, file_name, fmt='gz'):
    # every file is stored on GitHub as <file_name>.gz and <file_name>.xz, see tools/compress_files.py
    return f'{url_prefix}{file_name}.{fmt}'


def download_compressed(dest_file_path, url_prefix, file_name, validate=None):
    """
    Downloads the file in the configured format (see DOWNLOAD_FORMAT). When the .xz file is not published
    or can not be decompressed, the .gz file is downloaded instead.
    Blocks for the whole download including the pauses, so it must run in a background thread.
    """
    formats = download_formats()
    for fmt in formats:
        try:
            return download_file(dest_file_path, compressed_url(url_prefix, file_name, fmt), fmt=fmt, validate=validate)
        except (NotFoundError, DecompressError) as exc:
            # rate limits and network errors are not retried with another format, it would only make things worse
            if fmt == formats[-1]:
                raise
            if _Debug:
                print(f'loader: {exc}, trying the next format')


def download_file(dest_file_path, url, fmt='gz', validate=None, chunk_size=64 * 1024):
    """
    Downloads one compressed file (fmt is "gz" or "xz") into <dest>.<fmt>.part and decompresses it into <dest>.part.
    The file is moved in place only when it arrived completely, decompressed fine and validate(path) did not raise,
    so an interrupted download never leaves a broken file which looks complete on the next start.
    Blocks for the whole download including the pauses, so it must run in a background thread.
    """
    if os.path.isfile(dest_file_path):
        return dest_file_path
    os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
    compressed_file_path = f'{dest_file_path}.{fmt}.part'
    part_file_path = dest_file_path + '.part'
    try:
        with requests.Session() as session, open(compressed_file_path, 'wb') as f:
            session.headers['User-Agent'] = USER_AGENT
            download_url(session, url, f, chunk_size)
        compressed_size = os.path.getsize(compressed_file_path)
        decompress(compressed_file_path, part_file_path, fmt)
        os.remove(compressed_file_path)
        if validate:
            validate(part_file_path)
        os.replace(part_file_path, dest_file_path)
    finally:
        for path in (compressed_file_path, part_file_path):
            if os.path.isfile(path):
                os.remove(path)
    if _Debug:
        print(f'loader: downloaded {dest_file_path} as .{fmt}, {compressed_size} bytes, {os.path.getsize(dest_file_path)} after decompression')
    return dest_file_path


class DecompressError(DownloadError):
    pass


def decompress(src_file_path, dest_file_path, fmt='gz', chunk_size=1024 * 1024):
    # gzip only needs zlib which is part of every Python build, also on Android, iOS and in the browser (Pyodide).
    # gzip checks the CRC32 and the length stored at the end of the data, xz checks CRC64,
    # so a damaged or cut download fails here
    if fmt == 'xz':
        if lzma is None:
            raise DecompressError(f'can not decompress {src_file_path}, the lzma module is not available')
        opener, errors = lzma.open, (EOFError, lzma.LZMAError)
    else:
        opener, errors = gzip.open, (EOFError, zlib.error, gzip.BadGzipFile)
    try:
        with opener(src_file_path, 'rb') as src, open(dest_file_path, 'wb') as dst:
            shutil.copyfileobj(src, dst, chunk_size)
    except errors as exc:
        raise DecompressError(f'failed to decompress {src_file_path}: {exc}')


def download_url(session, url, f, chunk_size):
    start = f.tell()
    attempt = 0
    while True:
        time.sleep(random.uniform(JITTER_MIN, JITTER_MAX))
        try:
            with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
                if r.status_code == 200:
                    for chunk in r.iter_content(chunk_size):
                        f.write(chunk)
                    return
                wait = retry_wait(r, attempt)
                if wait is None:
                    if r.status_code == 404:
                        raise NotFoundError(f'{url} not found')
                    raise DownloadError(f'{url} rejected with HTTP {r.status_code}')
                reason = f'HTTP {r.status_code}'
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
            wait = backoff(attempt)
            reason = exc.__class__.__name__
        attempt += 1
        if attempt > MAX_RETRIES:
            raise DownloadError(f'{url} failed after {MAX_RETRIES} retries, last error: {reason}')
        if wait > MAX_WAIT:
            raise DownloadError(f'{url}: {reason}, GitHub asks to wait {wait:.0f} seconds, giving up')
        # the file is downloaded again from its start, drop what the failed attempt wrote
        f.seek(start)
        f.truncate()
        if _Debug:
            print(f'loader: {url} {reason}, retry {attempt}/{MAX_RETRIES} in {wait:.1f} seconds')
        time.sleep(wait)


def backoff(attempt):
    delay = min(BACKOFF_MAX, BACKOFF_BASE * 2 ** attempt)
    return random.uniform(delay / 2, delay)


def retry_wait(r, attempt):
    """
    Seconds to wait before the request is repeated, None when repeating it makes no sense (404 and so on).
    """
    rate_limited = r.status_code == 429 or (r.status_code == 403 and (
        r.headers.get('X-RateLimit-Remaining') == '0' or 'Retry-After' in r.headers))
    if not rate_limited and r.status_code not in RETRY_STATUSES:
        return None
    # when GitHub says how long to wait, it is followed exactly, a bit of jitter is added on top
    seconds = parse_retry_after(r.headers.get('Retry-After'))
    if seconds is None and r.headers.get('X-RateLimit-Remaining') == '0':
        try:
            seconds = max(0.0, int(r.headers['X-RateLimit-Reset']) - time.time())
        except (KeyError, ValueError):
            seconds = None
    if seconds is not None:
        return seconds + random.uniform(JITTER_MIN, JITTER_MAX)
    if rate_limited:
        return max(RATE_LIMIT_MIN_WAIT, backoff(attempt))
    return backoff(attempt)


def parse_retry_after(value):
    # Retry-After is either a number of seconds or an HTTP date
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, email.utils.parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError):
        return None


def validate_res(file_path):
    with open(file_path, 'rb') as f:
        res.read_res_filetree(f)


def template_res_path(template):
    return res.data_path(os.path.join('data', 'figures', template + '.res'))


def is_template_available(template):
    # models which were already unpacked earlier do not need the .res file anymore
    return os.path.isfile(template_res_path(template)) or os.path.isdir(res.data_path(os.path.join('models', template)))


class ModelsLoader(object):
    """
    Makes sure that all models (templates) needed for the world are on the device before the scene is rendered.
    Models which were downloaded before are skipped, others are downloaded one by one in a background thread.
    """

    def __init__(self, templates, on_progress=None, on_done=None):
        self.templates = list(templates)
        self.on_progress = on_progress
        self.on_done = on_done
        self.missing = [t for t in self.templates if not is_template_available(t)]
        self.failed = []
        self.thread = None

    def start(self):
        if _Debug:
            print(f'models loader: {len(self.templates)} templates required, {len(self.missing)} need to be downloaded')
        if not self.missing:
            Clock.schedule_once(lambda dt: self.finish())
            return
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        figures_res = None
        for i, template in enumerate(self.missing):
            ok = False
            if FIGURES_URL_PREFIX:
                ok = self.download(template)
            if not ok:
                if figures_res is None:
                    figures_res = self.open_figures_res()
                if figures_res:
                    ok = self.extract(figures_res, template)
            if not ok:
                self.failed.append(template)
            Clock.schedule_once(lambda dt, i=i, template=template: self.progress(i + 1, template))
        if figures_res:
            figures_res[0].close()
        Clock.schedule_once(lambda dt: self.finish())

    def download(self, template):
        try:
            # the file is checked to be a valid .res file before it is used
            download_compressed(template_res_path(template), FIGURES_URL_PREFIX, template + '.res', validate=validate_res)
            return True
        except Exception as exc:
            if _Debug:
                print(f'models loader: failed to download {template}.res from {FIGURES_URL_PREFIX}: {exc}')
        return False

    def open_figures_res(self):
        # the whole figures.res can still be used as the source, models are taken out of it the same way
        # as tools/split_figures_res.py does
        figures_res_path = res.data_path(os.path.join('data', 'figures.res'))
        if not os.path.isfile(figures_res_path):
            return False
        try:
            figures_file = open(figures_res_path, 'rb')
            filetree_dict = res.read_res_filetree(figures_file, return_dict=True)
            groups, _ = res.figures_templates(figures_file, filetree_dict)
        except:
            traceback.print_exc()
            return False
        return figures_file, filetree_dict, groups

    def extract(self, figures_res, template):
        figures_file, filetree_dict, groups = figures_res
        if template not in groups:
            if _Debug:
                print(f'models loader: template {template} not found in figures.res')
            return False
        dest_file_path = template_res_path(template)
        if not os.path.isdir(os.path.dirname(dest_file_path)):
            os.makedirs(os.path.dirname(dest_file_path))
        try:
            res.extract_figures_template(figures_file, filetree_dict, groups[template], dest_file_path)
        except:
            traceback.print_exc()
            return False
        if _Debug:
            print(f'models loader: extracted {template}.res from figures.res')
        return True

    def progress(self, done, template):
        if self.on_progress:
            self.on_progress(done, len(self.missing), template)

    def finish(self):
        if self.failed:
            print(f'models loader: ERROR, models not available: {self.failed}')
        if self.on_done:
            self.on_done(self.failed)
