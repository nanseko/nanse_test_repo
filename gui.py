""" Gradio GUI for training the (attention-augmented) CUT model.

Works on a normal PC and on Google Colab. On Colab a public share link is
created automatically. Requirements 1-8 from the project request are covered:

  1. Configure Input / Output folder paths.
  2. Folders are scanned and their image files are listed / counted.
  3. Basic training parameters are editable and persisted with a Save button.
  4. CUT-specific parameters are editable and persisted with a Save button.
  5. Attention modules can be toggled (all off / all on / individual) and saved.
  6. Live learning rate, epoch, step and current file name are shown.
  7. A scrolling log is shown and also written to <out_dir>/logs/.
  8. Runs on PC and Colab (Gradio web UI).

Launch:
    python gui.py                 # local, http://127.0.0.1:7860
    python gui.py --share         # force a public share link
On Colab just run `!python gui.py` (share link is automatic).
"""

import os
import sys

# This codebase targets the Keras 2 API. Modern Colab ships TensorFlow 2.16+
# with Keras 3, which breaks Keras-2 idioms (Model subclassing, save_weights
# naming, etc.). When the `tf-keras` compatibility package is available, force
# the legacy Keras 2 backend BEFORE TensorFlow is imported (TF is only imported
# later, inside the training worker). We only set this when `tf_keras` exists,
# because on TF 2.15 (Keras 2 is built-in) setting it without the package would
# break `tf.keras`.
import importlib.util as _ilu
if _ilu.find_spec('tf_keras') is not None:
    os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

import json
import glob
import time
import argparse
import datetime
import threading
import traceback

import gradio as gr


# Colab is the only environment allowed to reach the external network for
# dataset download. On a corporate intranet (non-Colab) the feature stays off.
def _detect_colab():
    # Works even when launched as a subprocess (`!python gui.py`), where
    # 'google.colab' is not yet imported into sys.modules.
    if 'google.colab' in sys.modules:
        return True
    if os.environ.get('COLAB_RELEASE_TAG') or os.environ.get('COLAB_GPU'):
        return True
    try:
        import importlib.util
        if importlib.util.find_spec('google.colab') is not None:
            return True
    except Exception:
        pass
    return False


IN_COLAB = _detect_colab()


# --------------------------------------------------------------------------- #
# Configuration handling
# --------------------------------------------------------------------------- #

DEFAULT_CONFIG_PATH = './gui_config.json'

# Stable key order. The Gradio input list is assembled in exactly this order so
# that every Save button can collect the whole config consistently.
CONFIG_KEYS = [
    # 1-2. Data folders
    'train_src_dir', 'train_tar_dir', 'test_src_dir', 'test_tar_dir',
    'out_dir', 'image_size',
    # 3. Basic training params
    'mode', 'epochs', 'batch_size', 'lr', 'beta_1', 'beta_2',
    'lr_decay_rate', 'lr_decay_step', 'save_n_epoch',
    # 4. CUT params
    'gan_mode', 'norm_layer', 'resnet_blocks', 'netF_units',
    'netF_num_patches', 'nce_temp', 'impl', 'use_antialias',
    'lambda_grad', 'lambda_color',
    # 5. Attention params
    'attention_type', 'attention_reduction',
    'attention_encoder', 'attention_resblocks', 'attention_decoder',
]

DEFAULTS = {
    'train_src_dir': './datasets/SAR/trainA',
    'train_tar_dir': './datasets/Optical/trainB',
    'test_src_dir': './datasets/SAR/testA',
    'test_tar_dir': './datasets/Optical/testB',
    'out_dir': './output',
    'image_size': 256,
    'mode': 'cut',
    'epochs': 400,
    'batch_size': 1,
    'lr': 0.0002,
    'beta_1': 0.5,
    'beta_2': 0.999,
    'lr_decay_rate': 0.9,
    'lr_decay_step': 100000,
    'save_n_epoch': 5,
    'gan_mode': 'lsgan',
    'norm_layer': 'instance',
    'resnet_blocks': 9,
    'netF_units': 256,
    'netF_num_patches': 256,
    'nce_temp': 0.07,
    'impl': 'ref',
    'use_antialias': True,
    'lambda_grad': 0.0,
    'lambda_color': 0.0,
    'attention_type': 'none',
    'attention_reduction': 16,
    'attention_encoder': False,
    'attention_resblocks': False,
    'attention_decoder': False,
}

IMAGE_EXTS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
IMAGE_EXTS_FLAT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def load_config(path=DEFAULT_CONFIG_PATH):
    cfg = dict(DEFAULTS)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                cfg.update(json.load(f))
        except Exception as exc:  # pragma: no cover - defensive
            print(f'[gui] Failed to read config {path}: {exc}')
    return cfg


def save_config(cfg, path=DEFAULT_CONFIG_PATH):
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
    return path


def list_images(folder):
    if not folder or not os.path.isdir(folder):
        return []
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


# --------------------------------------------------------------------------- #
# Training state shared between the worker thread and the UI
# --------------------------------------------------------------------------- #

class TrainingState:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.running = False
        self.stop_requested = False
        self.epoch = 0
        self.total_epochs = 0
        self.step = 0
        self.steps_per_epoch = 0
        self.lr = 0.0
        self.current_file = '-'
        self.speed = 0.0          # iterations / second
        self.losses = {}
        self.message = '대기 중 (Idle)'
        self.logs = []
        self.log_file = None

    def log(self, text):
        stamp = datetime.datetime.now().strftime('%H:%M:%S')
        line = f'[{stamp}] {text}'
        with self.lock:
            self.logs.append(line)
            self.logs = self.logs[-500:]   # keep last 500 lines
            if self.log_file:
                try:
                    with open(self.log_file, 'a') as f:
                        f.write(line + '\n')
                except Exception:
                    pass
        print(line)

    def snapshot(self):
        with self.lock:
            return {
                'running': self.running,
                'epoch': self.epoch,
                'total_epochs': self.total_epochs,
                'step': self.step,
                'steps_per_epoch': self.steps_per_epoch,
                'lr': self.lr,
                'current_file': self.current_file,
                'speed': self.speed,
                'losses': dict(self.losses),
                'message': self.message,
                'logs': '\n'.join(self.logs[-300:]),
            }


STATE = TrainingState()


# --------------------------------------------------------------------------- #
# Training worker (manual loop so we can report epoch / lr / current file)
# --------------------------------------------------------------------------- #

def _build_dataset(src_files, tar_files, image_size, batch_size):
    import tensorflow as tf

    size = int(image_size)

    def load_src(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = (tf.cast(img, tf.float32) / 127.5) - 1.0
        img = tf.image.resize(img, (size, size))
        return img, path

    def load_tar(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = (tf.cast(img, tf.float32) / 127.5) - 1.0
        img = tf.image.resize(img, (size, size))
        return img

    autotune = tf.data.experimental.AUTOTUNE
    src = (tf.data.Dataset.from_tensor_slices(src_files)
           .shuffle(len(src_files)).map(load_src, num_parallel_calls=autotune))
    tar = (tf.data.Dataset.from_tensor_slices(tar_files)
           .shuffle(len(tar_files)).map(load_tar, num_parallel_calls=autotune))
    ds = (tf.data.Dataset.zip((src, tar))
          .batch(int(batch_size), drop_remainder=True).prefetch(autotune))
    return ds


def training_worker(cfg, state):
    try:
        import tensorflow as tf

        keras_ver = str(getattr(tf.keras, '__version__', '?'))
        state.log(f'TensorFlow {tf.__version__}, Keras {keras_ver}')
        if keras_ver.startswith('3'):
            state.log('오류: Keras 3가 감지되었습니다. 이 코드는 Keras 2가 필요합니다. '
                      'Colab에서 `pip install tf-keras` 실행 후 런타임을 재시작하고 '
                      '다시 시도하세요. (gui.py가 TF_USE_LEGACY_KERAS=1 을 설정합니다.)')
            with state.lock:
                state.message = '오류: Keras 3 (tf-keras 설치 필요)'
                state.running = False
            return

        from modules.cut_model import CUT_model

        state.log('데이터셋 준비 중...')
        src_files = list_images(cfg['train_src_dir'])
        tar_files = list_images(cfg['train_tar_dir'])
        if not src_files or not tar_files:
            state.log('오류: 학습용 source 또는 target 폴더에 이미지가 없습니다.')
            with state.lock:
                state.message = '오류: 학습 이미지 없음'
                state.running = False
            return

        n_pairs = min(len(src_files), len(tar_files))
        steps_per_epoch = max(n_pairs // int(cfg['batch_size']), 1)
        with state.lock:
            state.steps_per_epoch = steps_per_epoch
        state.log(f'source {len(src_files)}장, target {len(tar_files)}장 '
                  f'-> epoch당 {steps_per_epoch} step')

        ds = _build_dataset(src_files, tar_files, cfg['image_size'], cfg['batch_size'])

        size = int(cfg['image_size'])
        shape = (size, size, 3)
        cut = CUT_model(
            shape, shape,
            cut_mode=cfg['mode'],
            gan_mode=cfg['gan_mode'],
            use_antialias=bool(cfg['use_antialias']),
            norm_layer=cfg['norm_layer'],
            resnet_blocks=int(cfg['resnet_blocks']),
            netF_units=int(cfg['netF_units']),
            netF_num_patches=int(cfg['netF_num_patches']),
            nce_temp=float(cfg['nce_temp']),
            impl=cfg['impl'],
            attention_type=cfg['attention_type'],
            attention_reduction=int(cfg['attention_reduction']),
            attention_encoder=bool(cfg['attention_encoder']),
            attention_resblocks=bool(cfg['attention_resblocks']),
            attention_decoder=bool(cfg['attention_decoder']),
            lambda_grad=float(cfg['lambda_grad']),
            lambda_color=float(cfg['lambda_color']),
        )

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=float(cfg['lr']),
            decay_steps=int(cfg['lr_decay_step']),
            decay_rate=float(cfg['lr_decay_rate']),
            staircase=True)

        def opt():
            return tf.keras.optimizers.Adam(
                learning_rate=lr_schedule,
                beta_1=float(cfg['beta_1']), beta_2=float(cfg['beta_2']))

        cut.compile(G_optimizer=opt(), F_optimizer=opt(), D_optimizer=opt())

        ckpt_dir = os.path.join(cfg['out_dir'], 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)

        state.log('학습 시작')
        total_epochs = int(cfg['epochs'])
        with state.lock:
            state.total_epochs = total_epochs

        for epoch in range(total_epochs):
            if state.stop_requested:
                break
            with state.lock:
                state.epoch = epoch + 1
            for step, ((src_img, src_path), tar_img) in enumerate(ds):
                if state.stop_requested:
                    break
                t0 = time.time()
                logs = cut.train_step((src_img, tar_img))
                dt = max(time.time() - t0, 1e-6)

                cur_file = os.path.basename(src_path[0].numpy().decode('utf-8'))
                cur_lr = float(lr_schedule(cut.G_optimizer.iterations))
                with state.lock:
                    state.step = step + 1
                    state.lr = cur_lr
                    state.current_file = cur_file
                    state.speed = 1.0 / dt
                    state.losses = {k: float(v) for k, v in logs.items()}

                if (step + 1) % 10 == 0 or step == 0:
                    loss_str = ', '.join(f'{k}={float(v):.4f}' for k, v in logs.items())
                    state.log(f'epoch {epoch+1}/{total_epochs} '
                              f'step {step+1}/{steps_per_epoch} '
                              f'lr={cur_lr:.6f} file={cur_file} | {loss_str}')

            # checkpoint every save_n_epoch
            if (epoch + 1) % int(cfg['save_n_epoch']) == 0:
                ckpt_path = os.path.join(ckpt_dir, f'{epoch+1:03d}')
                try:
                    cut.save_weights(ckpt_path)
                    state.log(f'체크포인트 저장: {ckpt_path}')
                except Exception as exc:
                    state.log(f'체크포인트 저장 실패: {exc}')

        if state.stop_requested:
            state.log('사용자 요청으로 학습 중단됨')
            with state.lock:
                state.message = '중단됨 (Stopped)'
        else:
            state.log('학습 완료')
            with state.lock:
                state.message = '완료 (Done)'

    except Exception:
        state.log('학습 중 예외 발생:\n' + traceback.format_exc())
        with state.lock:
            state.message = '오류 (Error)'
    finally:
        with state.lock:
            state.running = False


# --------------------------------------------------------------------------- #
# M4-SAR dataset download (Colab only)
# --------------------------------------------------------------------------- #

M4SAR_REPO = 'wchao0601/m4-sar'
M4SAR_ZIP = 'M4-SAR.zip'


def summarize_extracted(target_dir, max_depth=2):
    """ Return a short folder tree of the extracted dataset with image counts. """
    if not os.path.isdir(target_dir):
        return '(추출 폴더가 없습니다.)'
    lines = []
    base = target_dir.rstrip(os.sep)
    for root, dirs, files in os.walk(base):
        depth = root[len(base):].count(os.sep)
        if depth > max_depth:
            dirs[:] = []
            continue
        dirs.sort()
        imgs = [f for f in files if f.lower().endswith(IMAGE_EXTS_FLAT)]
        indent = '  ' * depth
        name = os.path.basename(root) or root
        lines.append(f'{indent}{name}/  (이미지 {len(imgs)}개, 전체 {len(files)}개)')
        if len(lines) > 200:
            lines.append('  ...(생략)')
            break
    return '\n'.join(lines)


def download_and_extract(repo_id, filename, target_dir, token, allow_non_colab):
    """ Stream-download the dataset zip from HuggingFace and extract it.

    Enabled only on Colab (external network). On a non-Colab intranet it is
    refused unless the user explicitly ticks the override checkbox.
    """
    if not (IN_COLAB or allow_non_colab):
        yield ('⛔ 비활성화됨: Colab 환경이 아닙니다.\n'
               '사내망에서는 외부망 다운로드가 차단됩니다. '
               'Colab에서 실행하거나, 외부망이 가능한 환경이라면 '
               '"외부망 다운로드 강제 허용"을 체크하세요.')
        return

    import zipfile
    try:
        import requests
    except Exception:
        yield '오류: requests 패키지가 필요합니다. `pip install requests` 후 다시 시도하세요.'
        return

    repo_id = (repo_id or M4SAR_REPO).strip()
    filename = (filename or M4SAR_ZIP).strip()
    target_dir = (target_dir or './datasets/M4-SAR').strip()
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, filename)

    url = f'https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}'
    headers = {'Authorization': f'Bearer {token.strip()}'} if token and token.strip() else {}

    yield f'다운로드 시작\n  repo : {repo_id}\n  file : {filename}\n  url  : {url}\n  대상 : {target_dir}'
    try:
        with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
            if r.status_code == 401 or r.status_code == 403:
                yield (f'접근 거부(HTTP {r.status_code}). gated/비공개 데이터셋이면 '
                       'HF 토큰을 입력하세요. (huggingface.co/settings/tokens)')
                return
            r.raise_for_status()
            total = int(r.headers.get('Content-Length', 0))
            done = 0
            t0 = time.time()
            last = 0.0
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1 << 20):   # 1 MB
                    if not chunk:
                        continue
                    f.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if now - last > 1.0:
                        last = now
                        spd = done / max(now - t0, 1e-6) / 1e6
                        if total:
                            pct = done / total * 100
                            yield (f'다운로드 중... {done/1e9:.2f} / {total/1e9:.2f} GB '
                                   f'({pct:.1f}%)  {spd:.1f} MB/s')
                        else:
                            yield f'다운로드 중... {done/1e9:.2f} GB  {spd:.1f} MB/s'
    except Exception as exc:
        yield f'다운로드 실패: {exc}'
        return

    yield f'다운로드 완료 ({done/1e9:.2f} GB). 압축 해제 중...'
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            n = len(names)
            for i, member in enumerate(names):
                z.extract(member, target_dir)
                if i % 1000 == 0:
                    yield f'압축 해제 중... {i}/{n}'
    except zipfile.BadZipFile:
        yield f'오류: 잘못된 zip 파일입니다 ({zip_path}). 다시 다운로드하세요.'
        return
    except Exception as exc:
        yield f'압축 해제 실패: {exc}'
        return

    tree = summarize_extracted(target_dir)
    yield ('✅ 완료. 아래 폴더 구조를 참고해 "탭 1"에서 Source/Target 경로를 지정하세요.\n'
           f'추출 위치: {target_dir}\n\n{tree}')


def organize_m4sar_to_cut(source_root, out_dir, sar_kw, opt_kw, link_mode, test_ratio):
    """ Reorganize an extracted dataset into CUT layout.

    Walks ``source_root``, classifies each image as SAR (domain A / source) or
    optical (domain B / target) by path keywords, and as train/test, then
    builds <out_dir>/{trainA,trainB,testA,testB} via symlink (default) or copy.

    CUT is unpaired (A and B are shuffled independently), so exact pair
    alignment is not required; files are given unique names to avoid clashes.

    Yields progress strings plus gr.update() values that auto-fill the four
    training folder paths on completion.
    """
    import shutil
    import random

    blank = [gr.update(), gr.update(), gr.update(), gr.update()]

    if not source_root or not os.path.isdir(source_root):
        yield (f'오류: 소스 폴더가 없습니다: {source_root}', *blank)
        return

    sar_keys = [k.strip().lower() for k in (sar_kw or '').split(',') if k.strip()]
    opt_keys = [k.strip().lower() for k in (opt_kw or '').split(',') if k.strip()]
    if not sar_keys or not opt_keys:
        yield ('SAR / Optical 키워드를 모두 입력하세요.', *blank)
        return

    out_dir = (out_dir or './datasets/M4-SAR-cut').strip()

    yield ('소스 폴더 스캔 중...', *blank)
    items = []  # (abs_path, domain 'A'/'B', split 'train'/'test')
    for root, _, files in os.walk(source_root):
        rel = os.path.relpath(root, source_root).lower()
        for f in files:
            if not f.lower().endswith(IMAGE_EXTS_FLAT):
                continue
            hay = rel + '/' + f.lower()
            if any(k in hay for k in sar_keys):
                domain = 'A'
            elif any(k in hay for k in opt_keys):
                domain = 'B'
            else:
                continue
            if 'test' in hay or 'val' in hay or 'valid' in hay:
                split = 'test'
            else:
                split = 'train'
            items.append((os.path.join(root, f), domain, split))

    if not items:
        yield ('분류된 이미지가 없습니다. SAR/Optical 키워드 또는 소스 경로를 확인하세요.', *blank)
        return

    # If no explicit test split exists, optionally carve one out per domain.
    has_test = any(s == 'test' for _, _, s in items)
    try:
        test_ratio = float(test_ratio)
    except (TypeError, ValueError):
        test_ratio = 0.0
    if not has_test and test_ratio > 0:
        for dom in ('A', 'B'):
            idxs = [i for i, (_, d, _) in enumerate(items) if d == dom]
            random.shuffle(idxs)
            k = int(len(idxs) * test_ratio)
            for i in idxs[:k]:
                p, d, _ = items[i]
                items[i] = (p, d, 'test')

    dests = {key: os.path.join(out_dir, key)
             for key in ('trainA', 'trainB', 'testA', 'testB')}
    for d in dests.values():
        os.makedirs(d, exist_ok=True)

    use_copy = (link_mode == 'copy')
    counters = {}
    counts = {'trainA': 0, 'trainB': 0, 'testA': 0, 'testB': 0}
    total = len(items)
    for n, (src, domain, split) in enumerate(items):
        key = f'{split}{domain}'
        idx = counters.get(key, 0)
        counters[key] = idx + 1
        dst = os.path.join(dests[key], f'{idx:06d}_{os.path.basename(src)}')
        try:
            if os.path.exists(dst) or os.path.islink(dst):
                pass
            elif use_copy:
                shutil.copy2(src, dst)
            else:
                os.symlink(os.path.abspath(src), dst)
            counts[key] += 1
        except OSError:
            # symlink may be unsupported (e.g. Windows without privilege) -> copy
            try:
                shutil.copy2(src, dst)
                counts[key] += 1
            except Exception:
                pass
        if (n + 1) % 5000 == 0:
            yield (f'정리 중... {n+1}/{total}  {counts}', *blank)

    summary = (f'✅ CUT 형식 정리 완료 ({"복사" if use_copy else "심볼릭 링크"})\n'
               f'출력 폴더: {out_dir}\n'
               f'  trainA (SAR)     : {counts["trainA"]}장\n'
               f'  trainB (Optical) : {counts["trainB"]}장\n'
               f'  testA  (SAR)     : {counts["testA"]}장\n'
               f'  testB  (Optical) : {counts["testB"]}장\n'
               '아래 탭 1 경로가 자동으로 채워졌습니다.')
    yield (summary,
           gr.update(value=dests['trainA']),
           gr.update(value=dests['trainB']),
           gr.update(value=dests['testA']),
           gr.update(value=dests['testB']))


# --------------------------------------------------------------------------- #
# UI callbacks
# --------------------------------------------------------------------------- #

def _cfg_from_values(values):
    return dict(zip(CONFIG_KEYS, values))


def do_save(cfg_path, *values):
    cfg = _cfg_from_values(values)
    path = save_config(cfg, cfg_path or DEFAULT_CONFIG_PATH)
    return f'✅ 저장됨: {path}  ({datetime.datetime.now().strftime("%H:%M:%S")})'


def do_scan(*folders):
    msgs = []
    labels = ['Train-Source', 'Train-Target', 'Test-Source', 'Test-Target']
    for label, folder in zip(labels, folders):
        files = list_images(folder)
        sample = ', '.join(os.path.basename(p) for p in files[:5])
        more = ' ...' if len(files) > 5 else ''
        status = f'{len(files)}개' if files else '없음/경로확인'
        msgs.append(f'• {label} [{folder}] : {status}  {sample}{more}')
    return '\n'.join(msgs)


def attention_all_on():
    return True, True, True


def attention_all_off():
    return False, False, False


def _format_status(snap):
    ep = f"{snap['epoch']}/{snap['total_epochs']}"
    st = f"{snap['step']}/{snap['steps_per_epoch']}"
    lr = f"{snap['lr']:.6f}"
    spd = f"{snap['speed']:.2f} it/s"
    loss_str = ', '.join(f'{k}={v:.4f}' for k, v in snap['losses'].items()) or '-'
    return ep, st, lr, snap['current_file'], spd, snap['message'], loss_str, snap['logs']


def start_training(cfg_path, *values):
    cfg = _cfg_from_values(values)
    save_config(cfg, cfg_path or DEFAULT_CONFIG_PATH)

    if STATE.running:
        snap = STATE.snapshot()
        yield _format_status(snap)
        return

    # prepare log file
    log_dir = os.path.join(cfg['out_dir'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    STATE.reset()
    STATE.log_file = os.path.join(
        log_dir, f'gui_train_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.log')
    STATE.running = True
    STATE.message = '학습 준비 중...'

    thread = threading.Thread(target=training_worker, args=(cfg, STATE), daemon=True)
    thread.start()

    # stream status until the worker finishes
    while True:
        snap = STATE.snapshot()
        yield _format_status(snap)
        if not snap['running']:
            break
        time.sleep(1.0)
    yield _format_status(STATE.snapshot())


def stop_training():
    if STATE.running:
        STATE.stop_requested = True
        return '⏹️ 중단 요청됨. 현재 step 완료 후 멈춥니다.'
    return 'ℹ️ 실행 중인 학습이 없습니다.'


# --------------------------------------------------------------------------- #
# Build the Gradio UI
# --------------------------------------------------------------------------- #

def build_ui():
    cfg = load_config()
    comp = {}

    with gr.Blocks(title='CUT + Attention 학습 GUI', theme=gr.themes.Soft()) as demo:
        gr.Markdown('# CUT + Attention 학습 GUI\n'
                    'SAR-to-Optical CUT 모델을 폴더 지정만으로 학습합니다. '
                    '각 탭에서 값을 수정하고 **저장** 버튼을 누르면 `gui_config.json`에 보존됩니다.')

        cfg_path = gr.Textbox(value=DEFAULT_CONFIG_PATH, label='설정 파일 경로 (config json)')

        env_txt = ('🟢 Colab 환경 감지됨 — 데이터셋 다운로드 사용 가능'
                   if IN_COLAB else
                   '🔒 비-Colab 환경 — 외부망 차단 가정으로 데이터셋 다운로드 기본 비활성화')
        gr.Markdown(f'**실행 환경:** {env_txt}')

        # ---- Tab 0 : M4-SAR dataset download (Colab only) -------------- #
        with gr.Tab('0. 데이터셋 다운로드 (M4-SAR · Colab 전용)'):
            gr.Markdown(
                'HuggingFace `wchao0601/m4-sar` 에서 **M4-SAR.zip** 을 받아 압축을 풉니다.\n\n'
                '- **Colab 환경에서만** 기본 활성화됩니다.\n'
                '- 사내망(비-Colab)에서는 외부망 차단을 가정해 비활성화됩니다. '
                '외부망이 가능한 환경이라면 아래 "강제 허용"을 체크하세요.')
            ds_override = gr.Checkbox(
                value=False,
                label='외부망 다운로드 강제 허용 (사내망에서는 체크하지 마세요)',
                visible=not IN_COLAB)
            with gr.Row():
                ds_repo = gr.Textbox(M4SAR_REPO, label='HF dataset repo_id')
                ds_file = gr.Textbox(M4SAR_ZIP, label='zip 파일명')
            ds_target = gr.Textbox('./datasets/M4-SAR', label='압축 해제 대상 폴더')
            ds_token = gr.Textbox('', label='HF 토큰 (gated/비공개일 때만)', type='password')
            ds_btn = gr.Button('⬇️ 다운로드 + 압축 해제', variant='primary',
                               interactive=IN_COLAB)
            ds_out = gr.Textbox(label='진행 상황 / 결과', lines=14, interactive=False)

            # Non-Colab: enable the button only when the override is ticked.
            ds_override.change(
                lambda v: gr.update(interactive=(IN_COLAB or bool(v))),
                inputs=ds_override, outputs=ds_btn)
            ds_btn.click(
                download_and_extract,
                inputs=[ds_repo, ds_file, ds_target, ds_token, ds_override],
                outputs=ds_out)

            gr.Markdown('---\n### CUT 형식으로 정리 (trainA/trainB/testA/testB)\n'
                        '추출된 폴더를 SAR=Source(A), Optical=Target(B)로 자동 분류해 '
                        'CUT 학습 폴더 구조로 만듭니다. 경로 키워드로 도메인/split을 판별합니다.')
            with gr.Row():
                org_src = gr.Textbox('./datasets/M4-SAR', label='정리할 소스(추출) 폴더')
                org_out = gr.Textbox('./datasets/M4-SAR-cut', label='CUT 출력 폴더')
            with gr.Row():
                org_sar_kw = gr.Textbox('sar,vh,vv', label='SAR(Source/A) 키워드')
                org_opt_kw = gr.Textbox('optical,opt,rgb,vis,visible', label='Optical(Target/B) 키워드')
            with gr.Row():
                org_mode = gr.Radio(['symlink', 'copy'], value='symlink',
                                    label='파일 처리 (대용량은 symlink 권장)')
                org_ratio = gr.Number(0.1, label='test 폴더 없을 때 분리 비율 (0=안함)')
            org_btn = gr.Button('🗂️ CUT 형식으로 정리', variant='primary')
            org_out_box = gr.Textbox(label='정리 결과', lines=8, interactive=False)

        # ---- Tab 1 : Data folders -------------------------------------- #
        with gr.Tab('1. 데이터 폴더 (Input / Output)'):
            comp['train_src_dir'] = gr.Textbox(cfg['train_src_dir'], label='입력 Train Source 폴더 (예: SAR/trainA)')
            comp['train_tar_dir'] = gr.Textbox(cfg['train_tar_dir'], label='입력 Train Target 폴더 (예: Optical/trainB)')
            comp['test_src_dir'] = gr.Textbox(cfg['test_src_dir'], label='입력 Test Source 폴더')
            comp['test_tar_dir'] = gr.Textbox(cfg['test_tar_dir'], label='입력 Test Target 폴더')
            comp['out_dir'] = gr.Textbox(cfg['out_dir'], label='출력(Output) 폴더 — 체크포인트/로그/결과 저장')
            comp['image_size'] = gr.Number(cfg['image_size'], label='이미지 크기 (정사각 resize)', precision=0)
            scan_btn = gr.Button('📂 폴더 스캔 (내부 이미지 파일 확인)')
            scan_out = gr.Textbox(label='스캔 결과', lines=5, interactive=False)
            scan_btn.click(do_scan,
                           inputs=[comp['train_src_dir'], comp['train_tar_dir'],
                                   comp['test_src_dir'], comp['test_tar_dir']],
                           outputs=scan_out)

        # ---- Tab 2 : Basic training params ----------------------------- #
        with gr.Tab('2. 기본 학습 파라미터'):
            with gr.Row():
                comp['mode'] = gr.Dropdown(['cut', 'fastcut'], value=cfg['mode'], label='mode')
                comp['epochs'] = gr.Number(cfg['epochs'], label='epochs', precision=0)
                comp['batch_size'] = gr.Number(cfg['batch_size'], label='batch_size', precision=0)
            with gr.Row():
                comp['lr'] = gr.Number(cfg['lr'], label='learning rate')
                comp['beta_1'] = gr.Number(cfg['beta_1'], label='beta_1')
                comp['beta_2'] = gr.Number(cfg['beta_2'], label='beta_2')
            with gr.Row():
                comp['lr_decay_rate'] = gr.Number(cfg['lr_decay_rate'], label='lr_decay_rate')
                comp['lr_decay_step'] = gr.Number(cfg['lr_decay_step'], label='lr_decay_step', precision=0)
                comp['save_n_epoch'] = gr.Number(cfg['save_n_epoch'], label='save_n_epoch', precision=0)
            save_basic = gr.Button('💾 기본 파라미터 저장', variant='primary')
            save_basic_out = gr.Textbox(label='', interactive=False)

        # ---- Tab 3 : CUT params ---------------------------------------- #
        with gr.Tab('3. CUT 파라미터'):
            with gr.Row():
                comp['gan_mode'] = gr.Dropdown(['lsgan', 'nonsaturating'], value=cfg['gan_mode'], label='gan_mode')
                comp['norm_layer'] = gr.Dropdown(['instance', 'batch'], value=cfg['norm_layer'], label='norm_layer')
                comp['impl'] = gr.Dropdown(['ref', 'cuda'], value=cfg['impl'], label='impl (antialias op)')
            with gr.Row():
                comp['resnet_blocks'] = gr.Number(cfg['resnet_blocks'], label='resnet_blocks', precision=0)
                comp['netF_units'] = gr.Number(cfg['netF_units'], label='netF_units', precision=0)
                comp['netF_num_patches'] = gr.Number(cfg['netF_num_patches'], label='netF_num_patches', precision=0)
            with gr.Row():
                comp['nce_temp'] = gr.Number(cfg['nce_temp'], label='nce_temp')
                comp['use_antialias'] = gr.Checkbox(bool(cfg['use_antialias']), label='use_antialias')
            with gr.Row():
                comp['lambda_grad'] = gr.Number(cfg['lambda_grad'], label='lambda_grad (구조 보존)')
                comp['lambda_color'] = gr.Number(cfg['lambda_color'], label='lambda_color (색 일관성)')
            save_cut = gr.Button('💾 CUT 파라미터 저장', variant='primary')
            save_cut_out = gr.Textbox(label='', interactive=False)

        # ---- Tab 4 : Attention ----------------------------------------- #
        with gr.Tab('4. Attention 설정'):
            comp['attention_type'] = gr.Radio(['none', 'cbam', 'coord'],
                                              value=cfg['attention_type'],
                                              label='Attention 종류 (none = 완전 OFF)')
            comp['attention_reduction'] = gr.Number(cfg['attention_reduction'],
                                                    label='attention_reduction (bottleneck 축소비)', precision=0)
            gr.Markdown('**적용 위치 On/Off** — 개별 토글하거나 아래 버튼으로 모두 켜고 끌 수 있습니다.')
            with gr.Row():
                comp['attention_encoder'] = gr.Checkbox(bool(cfg['attention_encoder']), label='Encoder')
                comp['attention_resblocks'] = gr.Checkbox(bool(cfg['attention_resblocks']), label='ResBlocks')
                comp['attention_decoder'] = gr.Checkbox(bool(cfg['attention_decoder']), label='Decoder')
            with gr.Row():
                all_on = gr.Button('모두 ON')
                all_off = gr.Button('모두 OFF')
            save_att = gr.Button('💾 Attention 설정 저장', variant='primary')
            save_att_out = gr.Textbox(label='', interactive=False)

            all_on.click(attention_all_on, outputs=[comp['attention_encoder'],
                                                    comp['attention_resblocks'],
                                                    comp['attention_decoder']])
            all_off.click(attention_all_off, outputs=[comp['attention_encoder'],
                                                      comp['attention_resblocks'],
                                                      comp['attention_decoder']])

        # Ordered input list shared by every Save / Start button
        ordered_inputs = [comp[k] for k in CONFIG_KEYS]

        save_basic.click(do_save, inputs=[cfg_path] + ordered_inputs, outputs=save_basic_out)
        save_cut.click(do_save, inputs=[cfg_path] + ordered_inputs, outputs=save_cut_out)
        save_att.click(do_save, inputs=[cfg_path] + ordered_inputs, outputs=save_att_out)

        # Organize button (Tab 0) auto-fills the Tab 1 folder paths on completion.
        org_btn.click(
            organize_m4sar_to_cut,
            inputs=[org_src, org_out, org_sar_kw, org_opt_kw, org_mode, org_ratio],
            outputs=[org_out_box, comp['train_src_dir'], comp['train_tar_dir'],
                     comp['test_src_dir'], comp['test_tar_dir']])

        # ---- Tab 5 : Train & Monitor ----------------------------------- #
        with gr.Tab('5. 학습 실행 / 모니터링'):
            with gr.Row():
                start_btn = gr.Button('▶ 학습 시작', variant='primary')
                stop_btn = gr.Button('⏹ 중단', variant='stop')
            with gr.Row():
                st_epoch = gr.Textbox(label='Epoch', interactive=False)
                st_step = gr.Textbox(label='Step', interactive=False)
                st_lr = gr.Textbox(label='현재 학습률 (lr)', interactive=False)
                st_speed = gr.Textbox(label='처리 속도', interactive=False)
            with gr.Row():
                st_file = gr.Textbox(label='현재 처리 파일', interactive=False)
                st_msg = gr.Textbox(label='상태', interactive=False)
            st_loss = gr.Textbox(label='현재 손실 (D/G/NCE)', interactive=False)
            st_log = gr.Textbox(label='로그 (Log)', lines=16, interactive=False, max_lines=16)

            monitor_outputs = [st_epoch, st_step, st_lr, st_file, st_speed,
                               st_msg, st_loss, st_log]
            start_btn.click(start_training,
                            inputs=[cfg_path] + ordered_inputs,
                            outputs=monitor_outputs)
            stop_btn.click(stop_training, outputs=st_msg)

    return demo


def main():
    parser = argparse.ArgumentParser(description='CUT + Attention training GUI')
    parser.add_argument('--share', action='store_true', help='Force a public share link')
    parser.add_argument('--no-share', action='store_true', help='Disable share link')
    parser.add_argument('--port', type=int, default=7860, help='Server port')
    args = parser.parse_args()

    # Colab cannot reach the VM's 127.0.0.1 from the browser, so a public
    # share link is required there.
    share = (args.share or IN_COLAB) and not args.no_share

    if IN_COLAB:
        print('\n[gui] Colab 감지됨. 아래 출력의 공개 URL '
              '(https://XXXX.gradio.live) 을 클릭하세요. '
              '127.0.0.1 / localhost 는 Colab에서 접속되지 않습니다.\n')

    demo = build_ui()
    demo.queue().launch(share=share, server_port=args.port)


if __name__ == '__main__':
    main()
