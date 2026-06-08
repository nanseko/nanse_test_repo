# CUT + CBAM/Coordinate Attention — PyTorch 포팅 가이드

이 폴더는 기존 **TensorFlow** CUT 포크(`modules/`)에 추가했던
**CBAM · Coordinate Attention**(+ 구조/색 손실)을, 공식 **PyTorch** CUT 레포
[`taesungp/contrastive-unpaired-translation`](https://github.com/taesungp/contrastive-unpaired-translation)
에 적용할 수 있도록 PyTorch로 다시 구현한 것입니다.

> TensorFlow 코드는 그대로 두고, 아래 PyTorch 파일을 공식 레포에 넣어
> "PyTorch 환경에서 동작"하게 만드는 방식입니다.

## 1. 포함 파일

| 파일 | 내용 | 공식 레포에서의 위치(권장) |
|---|---|---|
| `attention.py` | `CBAM`, `CoordinateAttention`, `make_attention()` (NCHW) | `models/attention.py` |
| `networks.py` | attention 적용 `ResnetAttnGenerator` + antialias up/down + attention `ResnetBlock` | `models/networks.py` 에 병합 또는 별도 추가 |
| `losses_extra.py` | `gradient_loss`, `color_moment_loss` (선택) | `models/` 아무 곳 |
| `test_pytorch_port.py` | torch 스모크 테스트 | — |

TF 원본과의 대응:

| TensorFlow (`modules/`) | PyTorch (`pytorch/`) |
|---|---|
| `attention.py` ChannelAttention/SpatialAttention/CBAM/CoordinateAttention (NHWC) | `attention.py` (NCHW) |
| `cut_model.py` Generator의 attention 삽입(encoder/resblock/decoder) | `networks.py` `ResnetAttnGenerator` |
| `layers.py` ResBlock(attention_position) | `networks.py` `ResnetBlock` |
| `losses.py` gradient_loss/color_moment_loss | `losses_extra.py` |

## 2. 단독 사용 (검증)

```bash
pip install torch
cd pytorch
python test_pytorch_port.py
```

```python
from networks import ResnetAttnGenerator
G = ResnetAttnGenerator(3, 3, n_blocks=9,
        attention_type='coord', attention_encoder=True, attention_resblocks=True)
print(G.nce_default)          # 이 설정에서의 PatchNCE tap 인덱스
fake = G(real_A)              # 일반 생성
feats = G(real_A, layers=G.nce_default, encode_only=True)   # PatchNCE용 특징
```

## 3. 공식 CUT 레포에 통합하기

### 3-1. 파일 복사
- `attention.py` → `models/attention.py`
- `losses_extra.py` → `models/losses_extra.py`
- `networks.py` 의 `ResnetAttnGenerator`/`ResnetBlock`/`Downsample`/`Upsample` 를
  `models/networks.py` 에 추가(또는 파일째 두고 import).
- **import 경로 수정**: `networks.py` 상단의 `from attention import make_attention` 를
  공식 레포에선 `from models.attention import make_attention` 로 바꾸세요.

### 3-2. 옵션 추가 (`options/base_options.py` 또는 `train_options.py`)
```python
parser.add_argument('--attention_type', type=str, default='none',
                    choices=['none', 'cbam', 'coord'])
parser.add_argument('--attention_reduction', type=int, default=16)
parser.add_argument('--attention_encoder', action='store_true')
parser.add_argument('--attention_resblocks', action='store_true')
parser.add_argument('--attention_decoder', action='store_true')
parser.add_argument('--lambda_grad', type=float, default=0.0)
parser.add_argument('--lambda_color', type=float, default=0.0)
```

### 3-3. Generator 연결 (`models/networks.py` 의 `define_G`)
`netG == 'resnet_9blocks'` 분기에서 attention이 요청되면 새 generator를 쓰도록:
```python
from models.networks import ResnetAttnGenerator   # 같은 파일이면 직접 사용
if getattr(opt, 'attention_type', 'none') != 'none' or opt.attention_encoder \
        or opt.attention_resblocks or opt.attention_decoder:
    net = ResnetAttnGenerator(
        input_nc, output_nc, ngf, norm_layer=norm_layer, n_blocks=9,
        use_antialias=(not opt.no_antialias),
        attention_type=opt.attention_type,
        attention_reduction=opt.attention_reduction,
        attention_encoder=opt.attention_encoder,
        attention_resblocks=opt.attention_resblocks,
        attention_decoder=opt.attention_decoder)
else:
    net = ResnetGenerator(...)   # 기존 그대로
```
(그 뒤 공식 `init_net(net, ...)` 로 초기화/그대로 통과)

### 3-4. ⚠️ nce_layers 보정 (가장 중요)
공식 CUT 기본값은 `--nce_layers 0,4,8,12,16` 인데, **attention 모듈이 끼면
`nn.Sequential` 인덱스가 밀립니다.** 이 포팅 generator는 현재 설정에 맞는
올바른 tap 인덱스를 `G.nce_default` 로 노출합니다.

가장 안전한 연결: `models/cut_model.py` 의 `data_dependent_initialize`/`__init__`
근처에서, 문자열 대신 generator의 기본값을 쓰도록:
```python
# self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
self.nce_layers = list(self.netG.module.nce_default) \
    if hasattr(self.netG, 'module') else list(self.netG.nce_default)
```
또는 스모크 테스트/`print(G.nce_default)` 로 값을 확인해 `--nce_layers` 에 그대로
넣으세요. (예: `coord+encoder+resblocks` → `0,8,13,15,19`)

attention을 끄면(`none`) tap은 의미상 공식과 동일한 위치
`[input, conv128, conv256, resblock0, resblock4]` 를 가리킵니다.

### 3-5. (선택) 구조/색 손실 추가 (`models/cut_model.py`)
`compute_G_loss()` 끝부분에 추가:
```python
from models.losses_extra import gradient_loss, color_moment_loss
...
if self.opt.lambda_grad > 0:
    self.loss_G += self.opt.lambda_grad * gradient_loss(self.real_A, self.fake_B)
if self.opt.lambda_color > 0 and self.opt.nce_idt:   # idt_B 가 있을 때
    self.loss_G += self.opt.lambda_color * color_moment_loss(self.idt_B, self.real_B)
return self.loss_G
```
(CUT의 `nce_idt`가 True면 `self.idt_B`, `self.real_B` 가 이미 존재합니다.)

## 4. 학습 명령 예시 (공식 레포에서)

```bash
# baseline (기존 CUT)
python train.py --dataroot ./datasets/sar2opt --name sar_cut --CUT_mode CUT

# coordinate attention (encoder + resblocks) + 구조/색 손실
python train.py --dataroot ./datasets/sar2opt --name sar_cut_coord --CUT_mode CUT \
  --attention_type coord --attention_encoder --attention_resblocks \
  --attention_reduction 16 --lambda_grad 1.0 --lambda_color 1.0 \
  --nce_layers 0,8,13,15,19
```

## 5. 주의/차이점

1. **nce_layers 인덱스**: attention 삽입으로 공식 기본값과 달라집니다 → 반드시
   `G.nce_default` 사용(3-4). 안 맞추면 PatchNCE가 엉뚱한 층을 읽습니다.
2. **antialias up/down**: 공식의 `Downsample/Upsample`(Adobe antialiased-cnns 기반)
   대신, 여기서는 동작이 같은(정확한 2x) **이항 블러 기반**으로 깔끔히 재구현했습니다.
   공식 모듈을 그대로 쓰고 싶으면 `networks.py`의 `Downsample/Upsample`만 공식 것으로
   교체하면 됩니다(나머지 attention 로직은 그대로 호환).
3. **InstanceNorm/bias**: 공식과 동일 규칙(`use_bias = norm == InstanceNorm`).
4. **batch size**: CUT PatchNCE는 작은 batch에 맞춰져 있습니다(공식 기본 1).
5. attention 종류/위치/리덕션의 의미는 TF 버전 문서(`README_2.md`,
   `docs/attention_explained.html`)와 동일합니다.

## 6. 검증 결과 (torch 2.x, CPU)

```
attention shape: OK
generator[none]                              nce=[0, 6, 10, 12, 16]
generator[coord, encoder+resblocks]          nce=[0, 8, 13, 15, 19]
generator[cbam, encoder+resblocks+decoder]   nce=[0, 8, 13, 15, 19]
generator[coord, ..., no_antialias]          OK
losses: grad/color + backward OK
→ All PyTorch port smoke tests passed.
```
