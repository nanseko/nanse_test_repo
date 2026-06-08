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
    'out_dir', 'image_size', 'max_pairs',
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
    'max_pairs': 0,
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


def matched_pairs(src_dir, tar_dir, limit=0):
    """ Pair SAR(source) and optical(target) images by filename stem.

    The two domains share the same file names (e.g. 00001 in both folders), so
    pairs are matched by the filename without extension. Returns parallel
    (src_files, tar_files) lists for the common stems, optionally truncated to
    the first ``limit`` pairs (0 = use all). Returns ([], []) if no stems match.
    """
    src = {}
    for p in list_images(src_dir):
        src.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    tar = {}
    for p in list_images(tar_dir):
        tar.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    common = sorted(set(src) & set(tar))
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 0
    if limit > 0:
        common = common[:limit]
    return [src[k] for k in common], [tar[k] for k in common]


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
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                state.log(f'GPU 사용: {len(gpus)}개 감지됨 ({", ".join(g.name for g in gpus)})')
            else:
                state.log('⚠ GPU 미감지 — CPU로 학습합니다(느림). '
                          'tensorflow-cpu가 깔렸거나, 드라이버/CUDA가 GPU와 안 맞을 수 있습니다. '
                          'RTX 50xx(Blackwell)는 CUDA 12.8+와 최신 TF 빌드가 필요합니다.')
        except Exception:
            pass
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
        limit = int(cfg.get('max_pairs', 0) or 0)
        src_files, tar_files = matched_pairs(
            cfg['train_src_dir'], cfg['train_tar_dir'], limit)
        if src_files:
            state.log(f'파일명 매칭 쌍 {len(src_files)}개 사용'
                      + (f' (수량 제한 {limit})' if limit > 0 else ' (전체)'))
        else:
            # Filenames do not match across folders -> fall back to independent
            # lists, still honouring the quantity limit.
            src_files = list_images(cfg['train_src_dir'])
            tar_files = list_images(cfg['train_tar_dir'])
            if limit > 0:
                src_files = src_files[:limit]
                tar_files = tar_files[:limit]
            if src_files and tar_files:
                state.log('경고: 파일명이 매칭되지 않아 독립 목록으로 사용합니다.')

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

        def make_model(impl_choice):
            return CUT_model(
                shape, shape,
                cut_mode=cfg['mode'],
                gan_mode=cfg['gan_mode'],
                use_antialias=bool(cfg['use_antialias']),
                norm_layer=cfg['norm_layer'],
                resnet_blocks=int(cfg['resnet_blocks']),
                netF_units=int(cfg['netF_units']),
                netF_num_patches=int(cfg['netF_num_patches']),
                nce_temp=float(cfg['nce_temp']),
                impl=impl_choice,
                attention_type=cfg['attention_type'],
                attention_reduction=int(cfg['attention_reduction']),
                attention_encoder=bool(cfg['attention_encoder']),
                attention_resblocks=bool(cfg['attention_resblocks']),
                attention_decoder=bool(cfg['attention_decoder']),
                lambda_grad=float(cfg['lambda_grad']),
                lambda_color=float(cfg['lambda_color']),
            )

        try:
            cut = make_model(cfg['impl'])
        except Exception as exc:
            # The StyleGAN2 'cuda' custom op needs an exact nvcc/TF-header match
            # and fails on Colab (ABI mismatch). Fall back to pure-TF 'ref' ops.
            if cfg['impl'] == 'cuda':
                state.log(f"'cuda' 커스텀 연산 로드 실패 ({type(exc).__name__}). "
                          "순수 TensorFlow 연산('ref')으로 자동 전환합니다. "
                          "Colab에서는 'ref' 사용을 권장합니다.")
                cut = make_model('ref')
            else:
                raise

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
    # Filename-matched pair counts
    tr_src, _ = matched_pairs(folders[0], folders[1], 0)
    te_src, _ = matched_pairs(folders[2], folders[3], 0)
    msgs.append(f'• 매칭된 학습 쌍(Train) : {len(tr_src)}개')
    msgs.append(f'• 매칭된 검증 쌍(Test)  : {len(te_src)}개')
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


def run_inference(weights_dir, input_dir, output_dir, *cfg_values):
    """ Load a trained checkpoint (pretrained weights) and translate test images.

    The model architecture is rebuilt from the current config (Tabs 3/4) so it
    MUST match the architecture used during training, then the latest checkpoint
    in ``weights_dir`` is loaded and netG is run on each input image.
    Yields (status_text, [result_image_paths]) for live updates + a gallery.
    """
    cfg = _cfg_from_values(cfg_values)
    gallery = []

    if not input_dir or not os.path.isdir(input_dir):
        yield (f'오류: 입력 폴더가 없습니다: {input_dir}', gallery)
        return
    if not weights_dir or not os.path.isdir(weights_dir):
        yield (f'오류: 가중치(체크포인트) 폴더가 없습니다: {weights_dir}', gallery)
        return

    files = list_images(input_dir)
    if not files:
        yield (f'오류: 입력 폴더에 이미지가 없습니다: {input_dir}', gallery)
        return

    output_dir = (output_dir or './output/test').strip()
    os.makedirs(output_dir, exist_ok=True)

    try:
        import numpy as np
        import tensorflow as tf
        from PIL import Image

        keras_ver = str(getattr(tf.keras, '__version__', '?'))
        if keras_ver.startswith('3'):
            yield ('오류: Keras 3 감지. `pip install tf-keras` 후 런타임 재시작이 필요합니다.', gallery)
            return

        from modules.cut_model import CUT_model

        size = int(cfg['image_size'])
        shape = (size, size, 3)
        yield (f'모델 생성 중 (아키텍처는 탭 3/4 설정과 일치해야 합니다)...', gallery)

        def make_model(impl_choice):
            return CUT_model(
                shape, shape,
                cut_mode=cfg['mode'], gan_mode=cfg['gan_mode'],
                use_antialias=bool(cfg['use_antialias']), norm_layer=cfg['norm_layer'],
                resnet_blocks=int(cfg['resnet_blocks']), netF_units=int(cfg['netF_units']),
                netF_num_patches=int(cfg['netF_num_patches']), nce_temp=float(cfg['nce_temp']),
                impl=impl_choice,
                attention_type=cfg['attention_type'], attention_reduction=int(cfg['attention_reduction']),
                attention_encoder=bool(cfg['attention_encoder']),
                attention_resblocks=bool(cfg['attention_resblocks']),
                attention_decoder=bool(cfg['attention_decoder']))

        try:
            cut = make_model(cfg['impl'])
        except Exception as exc:
            if cfg['impl'] == 'cuda':
                cut = make_model('ref')
            else:
                raise

        latest = tf.train.latest_checkpoint(weights_dir)
        if latest is None:
            yield (f'오류: {weights_dir} 에서 체크포인트를 찾지 못했습니다.', gallery)
            return
        cut.load_weights(latest).expect_partial()
        yield (f'가중치 로드 완료: {latest}\n추론 시작 ({len(files)}장)...', gallery)

        def load_one(path):
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = (tf.cast(img, tf.float32) / 127.5) - 1.0
            img = tf.image.resize(img, (size, size))
            return tf.expand_dims(img, 0)

        for i, path in enumerate(files):
            src = load_one(path)
            pred = cut.netG(src, training=False)[0].numpy()
            pred = np.clip(pred * 127.5 + 127.5, 0, 255).astype(np.uint8)
            name = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(output_dir, f'{name}_translated.png')
            Image.fromarray(pred).save(out_path)
            gallery.append(out_path)
            if (i + 1) % 5 == 0 or i == 0 or i == len(files) - 1:
                yield (f'추론 중... {i+1}/{len(files)}  현재: {os.path.basename(path)}',
                       gallery[-12:])

        yield (f'✅ 완료: {len(files)}장 변환 → {output_dir}', gallery[-12:])

    except Exception:
        yield ('추론 중 예외 발생:\n' + traceback.format_exc(), gallery[-12:])


# --------------------------------------------------------------------------- #
# SAR preprocessing callbacks (see docs/README_pipeline.md)
# Pipeline is an ORDERED, editable list of steps held in a gr.State.
# --------------------------------------------------------------------------- #

def pp_default_steps():
    """Recommended pipeline as an ordered list of step dicts (with labels)."""
    return [
        {'name': 'validate_image', 'enabled': True,
         'params': {'drop_empty': True, 'handle_nan': 'zero'}, 'label': 'validate'},
        {'name': 'sar_intensity_transform', 'enabled': True,
         'params': {'mode': 'log1p', 'eps': 1e-6}, 'label': 'intensity: log1p'},
        {'name': 'speckle_filter', 'enabled': True,
         'params': {'method': 'refined_lee', 'window_size': 7, 'enl': 'auto'},
         'label': 'speckle: refined_lee'},
        {'name': 'outlier_clipping', 'enabled': True,
         'params': {'min_percentile': 0.2, 'max_percentile': 99.8, 'ignore_zero': True},
         'label': 'clipping 0.2-99.8'},
        {'name': 'histogram_mapping', 'enabled': True,
         'params': {'mode': 'sar_only', 'bins': 1024, 'optical_reference_dir': None,
                    'clahe': {'enabled': False, 'clip_limit': 2.0, 'tile_grid_size': [8, 8]}},
         'label': 'histogram: sar_only'},
        {'name': 'resize_or_tile', 'enabled': True,
         'params': {'mode': 'resize', 'image_size': 256}, 'label': 'resize 256'},
        {'name': 'channel_adapter', 'enabled': True,
         'params': {'output_channels': 3}, 'label': 'channel 3ch'},
        {'name': 'normalize_for_cut', 'enabled': True,
         'params': {'output_range': 'uint8'}, 'label': 'normalize uint8'},
    ]


def _pp_short(params):
    keys = ('method', 'mode', 'window_size', 'enl', 'damping_factor', 'bm3d_sigma',
            'min_percentile', 'max_percentile', 'bins', 'image_size', 'output_channels')
    return ', '.join(f'{k}={params[k]}' for k in keys if k in params) or '-'


def _pp_rows(steps):
    return [[i + 1, s.get('label', s['name']), _pp_short(s['params'])]
            for i, s in enumerate(steps)]


def _speckle_params(method, window, enl_auto, enl_val, damping, sig_auto, sig_val):
    p = {'method': method}
    if method in ('lee', 'frost', 'refined_lee', 'gamma_map'):
        p['window_size'] = int(window)
        p['enl'] = 'auto' if enl_auto else float(enl_val)
    if method == 'frost':
        p['damping_factor'] = float(damping)
    if method == 'bm3d':
        p['bm3d_sigma'] = 'auto' if sig_auto else float(sig_val)
    return p


def _build_step(op, intmode, method, window, enl_auto, enl_val, damping, sig_auto, sig_val,
                cmin, cmax, ign, histmode, bins, optref, clahe, size, ch):
    if op.startswith('validate'):
        return {'name': 'validate_image', 'enabled': True,
                'params': {'drop_empty': True, 'handle_nan': 'zero'}, 'label': 'validate'}
    if op.startswith('intensity'):
        return {'name': 'sar_intensity_transform', 'enabled': True,
                'params': {'mode': intmode, 'eps': 1e-6}, 'label': f'intensity: {intmode}'}
    if op.startswith('speckle'):
        return {'name': 'speckle_filter', 'enabled': True,
                'params': _speckle_params(method, window, enl_auto, enl_val, damping, sig_auto, sig_val),
                'label': f'speckle: {method}'}
    if op.startswith('clipping') or op.startswith('outlier'):
        return {'name': 'outlier_clipping', 'enabled': True,
                'params': {'min_percentile': float(cmin), 'max_percentile': float(cmax),
                           'ignore_zero': bool(ign)}, 'label': f'clipping {cmin}-{cmax}'}
    if op.startswith('histogram'):
        return {'name': 'histogram_mapping', 'enabled': True,
                'params': {'mode': histmode, 'bins': int(bins),
                           'optical_reference_dir': (optref or None),
                           'clahe': {'enabled': bool(clahe), 'clip_limit': 2.0,
                                     'tile_grid_size': [8, 8]}},
                'label': f'histogram: {histmode}'}
    if op.startswith('resize'):
        return {'name': 'resize_or_tile', 'enabled': True,
                'params': {'mode': 'resize', 'image_size': int(size)}, 'label': f'resize {int(size)}'}
    if op.startswith('channel'):
        return {'name': 'channel_adapter', 'enabled': True,
                'params': {'output_channels': int(ch)}, 'label': f'channel {int(ch)}ch'}
    if op.startswith('normalize'):
        return {'name': 'normalize_for_cut', 'enabled': True,
                'params': {'output_range': 'uint8'}, 'label': 'normalize uint8'}
    raise ValueError(op)


def _default_step(category):
    """Create a step of the chosen top-level category with default params."""
    if category == 'speckle':
        return {'name': 'speckle_filter', 'enabled': True,
                'params': {'method': 'lee', 'window_size': 7, 'enl': 'auto'},
                'label': 'speckle: lee'}
    if category == 'intensity':
        return {'name': 'sar_intensity_transform', 'enabled': True,
                'params': {'mode': 'log1p', 'eps': 1e-6}, 'label': 'intensity: log1p'}
    if category == 'clipping':
        return {'name': 'outlier_clipping', 'enabled': True,
                'params': {'min_percentile': 0.2, 'max_percentile': 99.8, 'ignore_zero': True},
                'label': 'clipping 0.2-99.8'}
    if category == 'histogram':
        return {'name': 'histogram_mapping', 'enabled': True,
                'params': {'mode': 'sar_only', 'bins': 1024, 'optical_reference_dir': None,
                           'clahe': {'enabled': False, 'clip_limit': 2.0, 'tile_grid_size': [8, 8]}},
                'label': 'histogram: sar_only'}
    if category == 'resize':
        return {'name': 'resize_or_tile', 'enabled': True,
                'params': {'mode': 'resize', 'image_size': 256}, 'label': 'resize 256'}
    if category == 'channel':
        return {'name': 'channel_adapter', 'enabled': True,
                'params': {'output_channels': 3}, 'label': 'channel 3ch'}
    if category == 'validate':
        return {'name': 'validate_image', 'enabled': True,
                'params': {'drop_empty': True, 'handle_nan': 'zero'}, 'label': 'validate'}
    if category == 'normalize':
        return {'name': 'normalize_for_cut', 'enabled': True,
                'params': {'output_range': 'uint8'}, 'label': 'normalize uint8'}
    raise ValueError(category)


def pp_add_category(steps, category, sel):
    """Append a new step of the given category; select it."""
    steps = list(steps) + [_default_step(category)]
    return steps, _pp_rows(steps), len(steps) - 1


def pp_move_up(steps, sel):
    steps = list(steps)
    i = int(sel)
    if 0 < i < len(steps):
        steps[i - 1], steps[i] = steps[i], steps[i - 1]
        i -= 1
    return steps, _pp_rows(steps), i


def pp_move_down(steps, sel):
    steps = list(steps)
    i = int(sel)
    if 0 <= i < len(steps) - 1:
        steps[i + 1], steps[i] = steps[i], steps[i + 1]
        i += 1
    return steps, _pp_rows(steps), i


def pp_remove_sel(steps, sel):
    steps = list(steps)
    i = int(sel)
    if 0 <= i < len(steps):
        del steps[i]
    i = max(0, min(i, len(steps) - 1)) if steps else 0
    return steps, _pp_rows(steps), i


def pp_reset_steps():
    steps = pp_default_steps()
    return steps, _pp_rows(steps), 0


def pp_speckle_vis(method):
    win = method in ('lee', 'frost', 'refined_lee', 'gamma_map')
    damp = (method == 'frost')
    bm = (method == 'bm3d')
    import gradio as gr
    return (gr.update(visible=win), gr.update(visible=win), gr.update(visible=win),
            gr.update(visible=damp), gr.update(visible=bm), gr.update(visible=bm))


# Edit-panel widgets are returned in this fixed order by pp_on_select / wired
# to pp_apply:  method, window, enl_auto, enl_val, damp, sig_auto, sig_val,
#               intmode, cmin, cmax, ign, histmode, bins, optref, clahe, size, ch
def pp_on_select(steps, evt: gr.SelectData):
    """Row clicked -> open the edit panel pre-filled for that step."""
    import gradio as gr
    row = 0
    try:
        row = int(evt.index[0]) if evt and evt.index is not None else 0
    except Exception:
        row = 0
    if not steps or row >= len(steps):
        return [gr.update()] * 26
    s = steps[row]
    name = s['name']
    p = s.get('params', {})

    method = p.get('method', 'lee')
    window = int(p.get('window_size', 7))
    enl = p.get('enl', 'auto')
    enl_auto = (enl == 'auto')
    enl_val = 10.0 if enl_auto else float(enl)
    damp = float(p.get('damping_factor', 2.0))
    sig = p.get('bm3d_sigma', 'auto')
    sig_auto = (sig == 'auto')
    sig_val = 0.1 if sig_auto else float(sig)
    intmode = p.get('mode', 'log1p') if name == 'sar_intensity_transform' else 'log1p'
    cmin = float(p.get('min_percentile', 0.2))
    cmax = float(p.get('max_percentile', 99.8))
    ign = bool(p.get('ignore_zero', True))
    histmode = p.get('mode', 'sar_only') if name == 'histogram_mapping' else 'sar_only'
    bins = int(p.get('bins', 1024))
    optref = p.get('optical_reference_dir') or ''
    clahe = bool((p.get('clahe', {}) or {}).get('enabled', False))
    size = int(p.get('image_size', 256))
    ch = int(p.get('output_channels', 3))

    is_spk = (name == 'speckle_filter')
    is_int = (name == 'sar_intensity_transform')
    is_clip = (name == 'outlier_clipping')
    is_hist = (name == 'histogram_mapping')
    is_resize = (name == 'resize_or_tile')
    is_chan = (name == 'channel_adapter')
    # speckle sub-widget visibility (only meaningful when the speckle group shows)
    win_v = method in ('lee', 'frost', 'refined_lee', 'gamma_map')
    damp_v = (method == 'frost')
    bm_v = (method == 'bm3d')

    title = f'편집 중: #{row + 1}  ·  {s.get("label", name)}'
    if name in ('validate_image', 'normalize_for_cut'):
        title += '  (이 스텝은 조절할 파라미터가 없습니다)'

    # Outputs: pp_sel, panel, title, [6 category groups], [17 edit widgets]
    return [
        row,                                   # pp_sel
        gr.update(visible=True),               # edit_panel
        title,                                  # edit_title
        gr.update(visible=is_spk),             # g_spk
        gr.update(visible=is_int),             # g_int
        gr.update(visible=is_clip),            # g_clip
        gr.update(visible=is_hist),            # g_hist
        gr.update(visible=is_resize),          # g_resize
        gr.update(visible=is_chan),            # g_chan
        gr.update(value=method),               # e_method
        gr.update(value=window, visible=win_v),
        gr.update(value=enl_auto, visible=win_v),
        gr.update(value=enl_val, visible=win_v),
        gr.update(value=damp, visible=damp_v),
        gr.update(value=sig_auto, visible=bm_v),
        gr.update(value=sig_val, visible=bm_v),
        gr.update(value=intmode),              # e_intmode
        gr.update(value=cmin),                 # e_cmin
        gr.update(value=cmax),                 # e_cmax
        gr.update(value=ign),                  # e_ign
        gr.update(value=histmode),             # e_histmode
        gr.update(value=bins),                 # e_bins
        gr.update(value=optref),               # e_optref
        gr.update(value=clahe),                # e_clahe
        gr.update(value=size),                 # e_size
        gr.update(value=ch),                   # e_ch
    ]


def pp_apply(steps, sel, method, window, enl_auto, enl_val, damp, sig_auto, sig_val,
             intmode, cmin, cmax, ign, histmode, bins, optref, clahe, size, ch):
    """Apply edit-panel params to the selected step."""
    steps = list(steps)
    i = int(sel)
    if not (0 <= i < len(steps)):
        return steps, _pp_rows(steps)
    name = steps[i]['name']
    if name == 'speckle_filter':
        steps[i]['params'] = _speckle_params(method, window, enl_auto, enl_val,
                                             damp, sig_auto, sig_val)
        steps[i]['label'] = f'speckle: {method}'
    elif name == 'sar_intensity_transform':
        steps[i]['params'] = {'mode': intmode, 'eps': 1e-6}
        steps[i]['label'] = f'intensity: {intmode}'
    elif name == 'outlier_clipping':
        steps[i]['params'] = {'min_percentile': float(cmin), 'max_percentile': float(cmax),
                              'ignore_zero': bool(ign)}
        steps[i]['label'] = f'clipping {cmin}-{cmax}'
    elif name == 'histogram_mapping':
        steps[i]['params'] = {'mode': histmode, 'bins': int(bins),
                              'optical_reference_dir': (optref or None),
                              'clahe': {'enabled': bool(clahe), 'clip_limit': 2.0,
                                        'tile_grid_size': [8, 8]}}
        steps[i]['label'] = f'histogram: {histmode}'
    elif name == 'resize_or_tile':
        steps[i]['params'] = {'mode': 'resize', 'image_size': int(size)}
        steps[i]['label'] = f'resize {int(size)}'
    elif name == 'channel_adapter':
        steps[i]['params'] = {'output_channels': int(ch)}
        steps[i]['label'] = f'channel {int(ch)}ch'
    return steps, _pp_rows(steps)


def _pp_config_from_steps(input_dir, output_dir, max_items, recursive, shuffle, steps):
    return {
        'io': {'input_dir': input_dir, 'output_dir': output_dir,
               'max_items': int(max_items or 0), 'recursive': bool(recursive),
               'shuffle': bool(shuffle), 'seed': 42, 'save_format': 'png'},
        'pipeline': {'steps': [{'name': s['name'], 'enabled': s.get('enabled', True),
                                'params': s['params']} for s in steps]},
    }


# --- preprocessing settings persistence (folders + options + pipeline) ----- #
PP_CONFIG_PATH = './preproc_config.json'


def pp_load_settings():
    if os.path.exists(PP_CONFIG_PATH):
        try:
            with open(PP_CONFIG_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def pp_save_settings(steps, input_dir, output_dir, max_items, recursive, shuffle):
    data = {'input_dir': input_dir, 'output_dir': output_dir,
            'max_items': int(max_items or 0), 'recursive': bool(recursive),
            'shuffle': bool(shuffle), 'steps': steps}
    try:
        with open(PP_CONFIG_PATH, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    except Exception:
        pass


def pp_save_btn_fn(steps, input_dir, output_dir, max_items, recursive, shuffle):
    pp_save_settings(steps, input_dir, output_dir, max_items, recursive, shuffle)
    return f'✅ 전처리 설정 저장됨: {PP_CONFIG_PATH} ({datetime.datetime.now().strftime("%H:%M:%S")})'


def pp_preview(steps, input_dir, output_dir, max_items, recursive, shuffle):
    import os
    import preprocessing as PP
    pp_save_settings(steps, input_dir, output_dir, max_items, recursive, shuffle)
    if not steps:
        return None, None, '파이프라인에 스텝이 없습니다. 스텝을 추가하세요.'
    cfg = _pp_config_from_steps(input_dir, output_dir, max_items, recursive, shuffle, steps)
    files = PP.scan_images(input_dir, bool(recursive), False, 42, 1)
    if not files:
        return None, None, '입력 폴더에 이미지가 없습니다.'
    try:
        before, after = PP.preprocess_single(cfg, files[0])
        return before, after, f'미리보기: {os.path.basename(files[0])}'
    except Exception:
        return None, None, '미리보기 오류:\n' + traceback.format_exc()


def pp_run(steps, input_dir, output_dir, max_items, recursive, shuffle):
    import preprocessing as PP
    pp_save_settings(steps, input_dir, output_dir, max_items, recursive, shuffle)
    if not steps:
        yield '파이프라인에 스텝이 없습니다.', []
        return
    cfg = _pp_config_from_steps(input_dir, output_dir, max_items, recursive, shuffle, steps)
    try:
        for log, prev in PP.run_pipeline(cfg):
            yield log, prev
    except Exception:
        yield ('전처리 중 예외:\n' + traceback.format_exc(), [])


def pp_export(output_dir, optical_dir, test_ratio, link_mode):
    import os
    import preprocessing as PP
    sar_dir = os.path.join(output_dir, 'images')
    try:
        return PP.export_cut_layout(sar_dir, './datasets/M4-SAR-cut',
                                    optical_dir or None, float(test_ratio), link_mode)
    except Exception:
        return 'export 오류:\n' + traceback.format_exc()


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
            with gr.Row():
                comp['image_size'] = gr.Number(cfg['image_size'], label='이미지 크기 (정사각 resize)', precision=0)
                comp['max_pairs'] = gr.Number(cfg['max_pairs'],
                                              label='사용할 데이터 쌍 수 (0 = 전체)', precision=0)
            scan_btn = gr.Button('📂 폴더 스캔 (내부 이미지 파일 확인)')
            scan_out = gr.Textbox(label='스캔 결과', lines=5, interactive=False)
            scan_btn.click(do_scan,
                           inputs=[comp['train_src_dir'], comp['train_tar_dir'],
                                   comp['test_src_dir'], comp['test_tar_dir']],
                           outputs=scan_out)

        # ---- Tab 2 : SAR Preprocessing (before training) -------------- #
        with gr.Tab('2. SAR 전처리 (학습 전)'):
            import preprocessing as PP
            gr.Markdown(
                'CUT 학습 **전에** SAR 이미지를 전처리합니다. 아래에서 전처리 스텝을 '
                '**원하는 순서로 추가/이동/삭제**하고, 미리보기로 확인한 뒤 실행하세요. '
                '같은 speckle 필터를 Lee·Frost로 여러 번 넣을 수도 있습니다. '
                '설계: `docs/README_pipeline.md`')

            _pps = pp_load_settings()
            _pp_steps0 = _pps.get('steps') or pp_default_steps()

            with gr.Accordion('① 폴더 / 데이터', open=True):
                pp_in = gr.Textbox(_pps.get('input_dir', './datasets/M4-SAR/raw_sar'), label='입력 SAR 폴더')
                pp_out = gr.Textbox(_pps.get('output_dir', './datasets/M4-SAR-preprocessed'), label='출력 폴더')
                with gr.Row():
                    pp_max = gr.Number(_pps.get('max_items', 20), label='처리 개수 (0=전체)', precision=0)
                    pp_recursive = gr.Checkbox(_pps.get('recursive', True), label='하위 폴더 포함')
                    pp_shuffle = gr.Checkbox(_pps.get('shuffle', False), label='섞기(shuffle)')
                with gr.Row():
                    pp_save_btn = gr.Button('💾 전처리 설정 저장 (폴더/순서 보존)')
                    pp_save_msg = gr.Textbox(label='', interactive=False)

            with gr.Accordion('② 전처리 순서 만들기', open=True):
                gr.Markdown(
                    '1) **추가할 전처리** 종류를 고르고 `➕ 추가` → 맨 아래 #으로 생성됩니다.\n'
                    '2) 표에서 **행(#)을 클릭**하면 선택되고, 아래 **편집 패널**이 열립니다 '
                    '(선택한 스텝만 표시 · speckle은 기본 Lee, 필터 변경 가능).\n'
                    '3) 선택한 #을 `⬆/⬇` 로 위/아래 이동, `🗑` 로 삭제합니다.')
                pp_steps = gr.State(_pp_steps0)
                pp_sel = gr.State(0)
                pp_table = gr.Dataframe(
                    headers=['#', '스텝', '파라미터'], datatype=['number', 'str', 'str'],
                    value=_pp_rows(_pp_steps0), interactive=False, wrap=True,
                    label='현재 파이프라인 (위→아래 순서로 실행 · 행을 클릭해 선택/편집)')
                with gr.Row():
                    pp_addcat = gr.Dropdown(
                        ['speckle', 'intensity', 'clipping', 'histogram', 'resize',
                         'channel', 'validate', 'normalize'],
                        value='speckle', label='추가할 전처리 (상위 메뉴)')
                    pp_add_btn = gr.Button('➕ 추가', variant='primary')
                with gr.Row():
                    pp_up_btn = gr.Button('⬆ 위로')
                    pp_down_btn = gr.Button('⬇ 아래로')
                    pp_rm_btn = gr.Button('🗑 선택 삭제')
                    pp_reset_btn = gr.Button('↺ 기본 순서로')

            # ----- Edit panel: only the selected step's params show ------ #
            with gr.Group(visible=False) as pp_edit_panel:
                pp_edit_title = gr.Markdown('편집')
                with gr.Group(visible=False) as g_spk:
                    e_method = gr.Dropdown(PP.SPECKLE_METHODS, value='lee', label='speckle 필터 종류')
                    with gr.Row():
                        e_window = gr.Number(7, label='window_size', precision=0)
                        e_enlauto = gr.Checkbox(True, label='ENL auto')
                        e_enlval = gr.Number(10, label='ENL 값')
                    with gr.Row():
                        e_damp = gr.Number(2.0, label='Frost damping_factor', visible=False)
                        e_sigauto = gr.Checkbox(True, label='BM3D sigma auto', visible=False)
                        e_sigval = gr.Number(0.1, label='BM3D sigma 값', visible=False)
                with gr.Group(visible=False) as g_int:
                    e_intmode = gr.Dropdown(PP.INTENSITY_MODES, value='log1p', label='intensity mode')
                with gr.Group(visible=False) as g_clip:
                    with gr.Row():
                        e_cmin = gr.Number(0.2, label='clip min %')
                        e_cmax = gr.Number(99.8, label='clip max %')
                        e_ign = gr.Checkbox(True, label='0값 제외')
                with gr.Group(visible=False) as g_hist:
                    with gr.Row():
                        e_histmode = gr.Dropdown(PP.HISTOGRAM_MODES, value='sar_only', label='histogram 모드')
                        e_bins = gr.Number(1024, label='bins', precision=0)
                        e_clahe = gr.Checkbox(False, label='CLAHE')
                    e_optref = gr.Textbox('', label='Optical 참조 폴더 (unpaired 모드)')
                with gr.Group(visible=False) as g_resize:
                    e_size = gr.Number(256, label='resize image_size', precision=0)
                with gr.Group(visible=False) as g_chan:
                    e_ch = gr.Number(3, label='출력 채널', precision=0)
                pp_apply_btn = gr.Button('✔ 적용', variant='primary')

            # edit widget order (matches pp_on_select outputs tail & pp_apply args)
            edit_widgets = [e_method, e_window, e_enlauto, e_enlval, e_damp, e_sigauto,
                            e_sigval, e_intmode, e_cmin, e_cmax, e_ign, e_histmode,
                            e_bins, e_optref, e_clahe, e_size, e_ch]
            edit_groups = [g_spk, g_int, g_clip, g_hist, g_resize, g_chan]

            # wiring
            pp_add_btn.click(pp_add_category, inputs=[pp_steps, pp_addcat, pp_sel],
                             outputs=[pp_steps, pp_table, pp_sel])
            pp_up_btn.click(pp_move_up, inputs=[pp_steps, pp_sel],
                            outputs=[pp_steps, pp_table, pp_sel])
            pp_down_btn.click(pp_move_down, inputs=[pp_steps, pp_sel],
                              outputs=[pp_steps, pp_table, pp_sel])
            pp_rm_btn.click(pp_remove_sel, inputs=[pp_steps, pp_sel],
                            outputs=[pp_steps, pp_table, pp_sel])
            pp_reset_btn.click(pp_reset_steps, outputs=[pp_steps, pp_table, pp_sel])
            pp_table.select(pp_on_select, inputs=[pp_steps],
                            outputs=[pp_sel, pp_edit_panel, pp_edit_title] + edit_groups + edit_widgets)
            e_method.change(pp_speckle_vis, inputs=e_method,
                            outputs=[e_window, e_enlauto, e_enlval, e_damp, e_sigauto, e_sigval])
            pp_apply_btn.click(pp_apply, inputs=[pp_steps, pp_sel] + edit_widgets,
                               outputs=[pp_steps, pp_table])

            pp_io_inputs = [pp_steps, pp_in, pp_out, pp_max, pp_recursive, pp_shuffle]
            pp_save_btn.click(pp_save_btn_fn, inputs=pp_io_inputs, outputs=pp_save_msg)

            with gr.Accordion('⑤ 미리보기 (Before / After)', open=True):
                pp_prev_btn = gr.Button('🔍 첫 이미지 미리보기')
                with gr.Row():
                    pp_before = gr.Image(label='Before (원본 SAR)', type='numpy')
                    pp_after = gr.Image(label='After (전처리)', type='numpy')
                pp_prev_msg = gr.Textbox(label='', interactive=False)
                pp_prev_btn.click(pp_preview, inputs=pp_io_inputs,
                                  outputs=[pp_before, pp_after, pp_prev_msg])

            with gr.Accordion('⑥ 실행 / Export', open=True):
                pp_run_btn = gr.Button('▶ 전처리 실행', variant='primary')
                pp_log = gr.Textbox(label='로그', lines=10, interactive=False, max_lines=10)
                pp_gallery = gr.Gallery(label='Before|After 미리보기', columns=3, height='auto')
                pp_run_btn.click(pp_run, inputs=pp_io_inputs, outputs=[pp_log, pp_gallery])

                gr.Markdown('---\n**CUT 폴더 구조로 export** (전처리 결과 → trainA/testA, optical → trainB/testB)')
                with gr.Row():
                    pp_exp_opt = gr.Textbox('', label='Optical 폴더 (trainB/testB용, 선택)')
                    pp_exp_ratio = gr.Number(0.1, label='test 비율')
                    pp_exp_link = gr.Radio(['symlink', 'copy'], value='symlink', label='파일 처리')
                pp_exp_btn = gr.Button('🗂️ CUT layout export')
                pp_exp_msg = gr.Textbox(label='export 결과', lines=4, interactive=False)
                pp_exp_btn.click(pp_export,
                                 inputs=[pp_out, pp_exp_opt, pp_exp_ratio, pp_exp_link],
                                 outputs=pp_exp_msg)

        # ---- Tab 3 : Basic training params ----------------------------- #
        with gr.Tab('3. 기본 학습 파라미터'):
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

        # ---- Tab 4 : CUT params ---------------------------------------- #
        with gr.Tab('4. CUT 파라미터'):
            with gr.Row():
                comp['gan_mode'] = gr.Dropdown(['lsgan', 'nonsaturating'], value=cfg['gan_mode'], label='gan_mode')
                comp['norm_layer'] = gr.Dropdown(['instance', 'batch'], value=cfg['norm_layer'], label='norm_layer')
                comp['impl'] = gr.Dropdown(['ref', 'cuda'], value=cfg['impl'],
                                          label='impl (antialias op) — Colab은 ref 권장 (cuda는 커스텀 빌드 필요)')
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

        # ---- Tab 5 : Attention ----------------------------------------- #
        with gr.Tab('5. Attention 설정'):
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

        # ---- Tab 6 : Train & Monitor ----------------------------------- #
        with gr.Tab('6. 학습 실행 / 모니터링'):
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

        # ---- Tab 7 : Inference / Test (pretrained) --------------------- #
        with gr.Tab('7. 추론 / 테스트 (pretrained)'):
            gr.Markdown(
                '학습으로 저장된 **체크포인트(pretrained 가중치)** 를 불러와 테스트 이미지를 바로 변환합니다.\n\n'
                '- 체크포인트는 학습 시 `출력폴더/checkpoints` 에 `save_n_epoch` 마다 저장됩니다.\n'
                '- ⚠️ **탭 4/5의 CUT·Attention 설정이 학습 때와 동일**해야 가중치가 올바르게 로드됩니다.')
            inf_weights = gr.Textbox('./output/checkpoints', label='가중치(체크포인트) 폴더')
            inf_input = gr.Textbox('./datasets/SAR/testA', label='입력 이미지 폴더 (Source)')
            inf_output = gr.Textbox('./output/test', label='변환 결과 저장 폴더')
            inf_btn = gr.Button('▶ 추론 실행', variant='primary')
            inf_status = gr.Textbox(label='진행 상황', lines=4, interactive=False)
            inf_gallery = gr.Gallery(label='변환 결과 (미리보기)', columns=4, height='auto')

            inf_btn.click(run_inference,
                          inputs=[inf_weights, inf_input, inf_output] + ordered_inputs,
                          outputs=[inf_status, inf_gallery])

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
