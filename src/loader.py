import os
import time
import random
import threading
import traceback
import email.utils

import requests

from kivy.clock import Clock

import res

_Debug = True

# where separate <template>.res files are downloaded from, see tools/split_figures_res.py
# while empty the models are taken out of the whole data/figures.res which is downloaded at the start
FIGURES_URL_PREFIX = os.environ.get('WORLDS_FIGURES_URL', '')

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


def download_file(dest_file_path, urls, validate=None, chunk_size=64 * 1024):
    """
    Downloads one file which can be stored as several parts on GitHub, the parts are joined in the given order.
    Data goes into <dest>.part first and is moved in place only when all parts arrived and validate(path) did not raise,
    so an interrupted download never leaves a broken file which looks complete on the next start.
    Blocks for the whole download including the pauses, so it must run in a background thread.
    """
    if os.path.isfile(dest_file_path):
        return dest_file_path
    os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
    part_file_path = dest_file_path + '.part'
    try:
        with requests.Session() as session, open(part_file_path, 'wb') as f:
            session.headers['User-Agent'] = USER_AGENT
            for url in urls:
                download_part(session, url, f, chunk_size)
        if validate:
            validate(part_file_path)
        os.replace(part_file_path, dest_file_path)
    finally:
        if os.path.isfile(part_file_path):
            os.remove(part_file_path)
    if _Debug:
        print(f'loader: downloaded {dest_file_path}, {os.path.getsize(dest_file_path)} bytes')
    return dest_file_path


def download_part(session, url, f, chunk_size):
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
        # the part is downloaded again from its start, drop what the failed attempt wrote
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
            download_file(template_res_path(template), [FIGURES_URL_PREFIX + template + '.res'], validate=validate_res)
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
