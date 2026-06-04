# CUT Attention 확장 사용 가이드

이 문서는 SAR-to-Optical CUT 학습을 위해 추가된 Generator Attention 옵션의 사용 방법, 수정 위치, 주요 파라미터와 하이퍼파라미터 조정 방법을 정리합니다.

기본 실행은 기존 CUT와 동일합니다. Attention 관련 플래그를 주지 않으면 `attention_type='none'`으로 동작하며, Generator/Discriminator/PatchNCE 흐름은 기존 구현과 동일하게 유지됩니다.

## 1. 추가된 Attention 구조

Attention 모듈은 `modules/attention.py`에 구현되어 있습니다.

| 모듈 | 설명 | 입력/출력 |
| --- | --- | --- |
| `ChannelAttention` | 채널별 중요도를 계산합니다. Global Average Pooling과 Global Max Pooling을 사용하고, 공유 MLP를 거친 뒤 sigmoid weight를 만듭니다. | NHWC -> channel weight |
| `SpatialAttention` | 위치별 중요도를 계산합니다. 채널 평균/최댓값을 concat한 뒤 `7x7 Conv2D`와 sigmoid를 적용합니다. | NHWC -> spatial weight |
| `CBAM` | Channel Attention 후 Spatial Attention을 순차 적용합니다. | NHWC -> NHWC |
| `CoordinateAttention` | height 방향과 width 방향을 따로 pooling한 뒤, 1x1 bottleneck conv를 거쳐 좌표 방향 attention map을 만듭니다. | NHWC -> NHWC |

두 attention 모듈 모두 입력 feature map과 같은 shape을 반환합니다.

## 2. Generator에 Attention이 들어가는 위치

Generator는 `modules/cut_model.py`의 `Generator(...)` 함수에서 정의됩니다.

추가된 옵션은 다음과 같습니다.

```python
Generator(
    input_shape,
    output_shape,
    norm_layer,
    use_antialias,
    resnet_blocks,
    impl,
    attention_type='none',
    attention_reduction=16,
    attention_encoder=False,
    attention_resblocks=False,
    attention_decoder=False,
)
```

Attention은 선택적으로 세 구간에 적용할 수 있습니다.

| 옵션 | 적용 위치 |
| --- | --- |
| `attention_encoder=True` | 초기 `7x7 ConvBlock`, 첫 번째 downsampling block, 두 번째 downsampling block 뒤 |
| `attention_resblocks=True` | 각 `ResBlock`의 residual branch |
| `attention_decoder=True` | decoder upsampling block 뒤 |

PatchNCE는 `Encoder(generator, nce_layers)`를 통해 Generator 내부 feature를 읽습니다. 따라서 Generator 중간 feature에 Attention을 넣으면 PatchNCE가 attention-refined feature를 자동으로 사용합니다. `modules/losses.py`의 `PatchNCELoss`는 수정하지 않아도 됩니다.

## 3. 학습 명령어

### 기본 CUT 학습

Attention을 사용하지 않는 기존 동작입니다.

```bash
python train.py \
  --mode cut \
  --train_src_dir ./datasets/SAR/trainA \
  --train_tar_dir ./datasets/Optical/trainB \
  --test_src_dir ./datasets/SAR/testA \
  --test_tar_dir ./datasets/Optical/testB
```

### Coordinate Attention 실험

SAR-to-Optical 변환에서는 위치 구조와 방향성 정보가 중요할 수 있으므로, 첫 실험으로는 `coord + encoder + resblocks` 조합을 권장합니다.

```bash
python train.py \
  --mode cut \
  --train_src_dir ./datasets/SAR/trainA \
  --train_tar_dir ./datasets/Optical/trainB \
  --test_src_dir ./datasets/SAR/testA \
  --test_tar_dir ./datasets/Optical/testB \
  --attention_type coord \
  --attention_encoder \
  --attention_resblocks \
  --attention_reduction 16
```

### CBAM 실험

CBAM은 채널 중요도와 공간 중요도를 모두 명시적으로 조정합니다.

```bash
python train.py \
  --mode cut \
  --train_src_dir ./datasets/SAR/trainA \
  --train_tar_dir ./datasets/Optical/trainB \
  --test_src_dir ./datasets/SAR/testA \
  --test_tar_dir ./datasets/Optical/testB \
  --attention_type cbam \
  --attention_encoder \
  --attention_resblocks \
  --attention_reduction 16
```

### Decoder까지 Attention 적용

출력 영상의 세부 질감이나 색 변환 품질을 더 강하게 조정하고 싶을 때 decoder attention도 켤 수 있습니다.

```bash
python train.py \
  --mode cut \
  --train_src_dir ./datasets/SAR/trainA \
  --train_tar_dir ./datasets/Optical/trainB \
  --test_src_dir ./datasets/SAR/testA \
  --test_tar_dir ./datasets/Optical/testB \
  --attention_type coord \
  --attention_encoder \
  --attention_resblocks \
  --attention_decoder \
  --attention_reduction 16
```

Decoder attention은 생성 이미지의 외형에 직접 영향을 줄 수 있으므로, 먼저 `attention_encoder`와 `attention_resblocks`만 켠 결과를 확인한 뒤 추가하는 것을 권장합니다.

## 4. Attention 관련 CLI 파라미터

`train.py`에 추가된 attention 옵션입니다.

| 파라미터 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `--attention_type` | choice | `none` | `none`, `cbam`, `coord` 중 선택 |
| `--attention_reduction` | int | `16` | attention bottleneck 축소 비율 |
| `--attention_encoder` | flag | off | encoder ConvBlock 뒤 attention 적용 |
| `--attention_resblocks` | flag | off | ResBlock 내부 residual branch에 attention 적용 |
| `--attention_decoder` | flag | off | decoder upsampling block 뒤 attention 적용 |

### `attention_type`

- `none`: Attention 비활성화. 기존 CUT와 동일합니다.
- `cbam`: Channel + Spatial Attention을 순차 적용합니다.
- `coord`: Coordinate Attention을 적용합니다.

### `attention_reduction`

Attention 내부 bottleneck 채널 수를 결정합니다.

```text
bottleneck_channels = input_channels // attention_reduction
```

값이 작을수록 attention 용량이 커지고, 값이 클수록 가벼워집니다.

| 값 | 경향 |
| --- | --- |
| `8` | 더 강한 attention, 파라미터 증가, 과적합 가능성 증가 |
| `16` | 기본 권장값 |
| `32` | 더 가벼운 attention, 효과는 약해질 수 있음 |

SAR-to-Optical 첫 실험은 `16`으로 시작하고, 결과가 불안정하거나 과적합이면 `32`, attention 효과가 약하면 `8`을 시도합니다.

## 5. 주요 학습 하이퍼파라미터

기존 `train.py` 학습 옵션도 함께 조정할 수 있습니다.

| 파라미터 | 기본값 | 설명 | 조정 팁 |
| --- | --- | --- | --- |
| `--mode` | `cut` | `cut` 또는 `fastcut` | 품질 우선이면 `cut`, 빠른 실험이면 `fastcut` |
| `--epochs` | `400` | 전체 epoch 수 | 작은 데이터셋은 조기 중단 결과를 자주 확인 |
| `--batch_size` | `1` | batch size | 현재 PatchNCE 구현은 batch size 1 기준으로 작성됨 |
| `--lr` | `0.0002` | Adam 초기 learning rate | attention 추가 후 불안정하면 `0.0001` 시도 |
| `--beta_1` | `0.5` | Adam beta1 | GAN 학습 기본값 유지 권장 |
| `--beta_2` | `0.999` | Adam beta2 | 기본값 유지 권장 |
| `--lr_decay_rate` | `0.9` | learning rate decay 비율 | 느린 decay가 필요하면 `0.95` |
| `--lr_decay_step` | `100000` | decay step | 데이터 크기와 epoch당 step 수 기준으로 조정 |
| `--save_n_epoch` | `5` | checkpoint 저장 주기 | 실험 비교 시 `5` 또는 `10` 권장 |
| `--impl` | `ref` | antialias op 구현 | CUDA custom op 환경이면 `cuda` 가능 |

## 6. 실험 순서 추천

SAR-to-Optical 작업에서는 한 번에 모든 옵션을 켜기보다, attention 위치와 종류를 나누어 비교하는 편이 좋습니다.

1. Baseline

```bash
python train.py --mode cut \
  --train_src_dir ./datasets/SAR/trainA \
  --train_tar_dir ./datasets/Optical/trainB \
  --test_src_dir ./datasets/SAR/testA \
  --test_tar_dir ./datasets/Optical/testB
```

2. Coordinate Attention, encoder + resblocks

```bash
python train.py --mode cut \
  --train_src_dir ./datasets/SAR/trainA \
  --train_tar_dir ./datasets/Optical/trainB \
  --test_src_dir ./datasets/SAR/testA \
  --test_tar_dir ./datasets/Optical/testB \
  --attention_type coord \
  --attention_encoder \
  --attention_resblocks \
  --attention_reduction 16
```

3. CBAM, encoder + resblocks

```bash
python train.py --mode cut \
  --train_src_dir ./datasets/SAR/trainA \
  --train_tar_dir ./datasets/Optical/trainB \
  --test_src_dir ./datasets/SAR/testA \
  --test_tar_dir ./datasets/Optical/testB \
  --attention_type cbam \
  --attention_encoder \
  --attention_resblocks \
  --attention_reduction 16
```

4. Best attention type + decoder attention

```bash
python train.py --mode cut \
  --train_src_dir ./datasets/SAR/trainA \
  --train_tar_dir ./datasets/Optical/trainB \
  --test_src_dir ./datasets/SAR/testA \
  --test_tar_dir ./datasets/Optical/testB \
  --attention_type coord \
  --attention_encoder \
  --attention_resblocks \
  --attention_decoder \
  --attention_reduction 16
```

각 실험은 `--out_dir`을 다르게 지정하면 결과 이미지와 checkpoint를 구분하기 쉽습니다.

```bash
python train.py \
  --out_dir ./output/coord_enc_res \
  --mode cut \
  --train_src_dir ./datasets/SAR/trainA \
  --train_tar_dir ./datasets/Optical/trainB \
  --test_src_dir ./datasets/SAR/testA \
  --test_tar_dir ./datasets/Optical/testB \
  --attention_type coord \
  --attention_encoder \
  --attention_resblocks
```

## 7. Attention 메커니즘 수정 방법

### CBAM 수정

CBAM은 `modules/attention.py`의 `CBAM`, `ChannelAttention`, `SpatialAttention`에서 수정합니다.

공간 attention kernel size를 바꾸고 싶다면 `CBAM.__init__`의 아래 부분을 수정합니다.

```python
self.spatial_attention = SpatialAttention(kernel_size=7)
```

예를 들어 더 작은 receptive field를 쓰려면 `kernel_size=3`으로 변경할 수 있습니다.

Channel MLP 구조를 바꾸고 싶다면 `ChannelAttention.build()`의 `self.shared_mlp`를 수정합니다.

```python
self.shared_mlp = tf.keras.Sequential([
    Dense(hidden_units, activation='relu'),
    Dense(channels),
])
```

### Coordinate Attention 수정

Coordinate Attention은 `modules/attention.py`의 `CoordinateAttention`에서 수정합니다.

현재 구조는 다음 흐름입니다.

```text
input
 -> height pooling, width pooling
 -> concat
 -> 1x1 bottleneck conv
 -> split height/width
 -> 1x1 conv_h, conv_w
 -> sigmoid
 -> input * attn_h * attn_w
```

Bottleneck 활성화 함수를 바꾸고 싶다면 `build()`의 아래 부분을 수정합니다.

```python
self.bottleneck = Conv2D(bottleneck, 1, padding='same', activation='relu')
```

예를 들어 `swish`를 쓰려면 다음처럼 바꿀 수 있습니다.

```python
self.bottleneck = Conv2D(bottleneck, 1, padding='same', activation='swish')
```

### ResBlock Attention 위치 수정

`modules/layers.py`의 `ResBlock`은 `attention_position`을 지원합니다.

기본값은 `residual`입니다.

```python
ResBlock(
    filters,
    kernel_size,
    use_bias,
    norm_layer,
    attention_type='coord',
    attention_reduction=16,
    attention_position='residual',
)
```

| 값 | 동작 |
| --- | --- |
| `residual` | residual branch에 attention 적용 후 skip connection |
| `post_add` | skip connection 이후 전체 output에 attention 적용 |

현재 `Generator`에서는 기본값인 `residual`을 사용합니다. 전체 ResBlock 출력에 attention을 적용하고 싶다면 `modules/cut_model.py`의 ResBlock 생성부에서 `attention_position='post_add'`를 넘기면 됩니다.

```python
x = ResBlock(
    256,
    3,
    use_bias,
    norm_layer,
    attention_type=attention_type if attention_resblocks else 'none',
    attention_reduction=attention_reduction,
    attention_position='post_add',
)(x)
```

## 8. 모델 코드에서 직접 사용하는 방법

`train.py`를 거치지 않고 직접 모델을 만들 때는 다음처럼 사용합니다.

```python
from modules.cut_model import CUT_model

cut = CUT_model(
    source_shape=(256, 256, 3),
    target_shape=(256, 256, 3),
    cut_mode='cut',
    attention_type='coord',
    attention_reduction=16,
    attention_encoder=True,
    attention_resblocks=True,
    attention_decoder=False,
)
```

Generator만 직접 만들 수도 있습니다.

```python
from modules.cut_model import Generator

netG = Generator(
    input_shape=(256, 256, 3),
    output_shape=(256, 256, 3),
    norm_layer='instance',
    use_antialias=True,
    resnet_blocks=9,
    impl='ref',
    attention_type='cbam',
    attention_reduction=16,
    attention_encoder=True,
    attention_resblocks=True,
    attention_decoder=False,
)
```

## 9. Smoke Test

Attention 모듈이 정상적으로 build되고 forward pass가 되는지 확인하는 간단한 스크립트가 있습니다.

```bash
python tests/smoke_attention.py
```

이 스크립트는 다음을 확인합니다.

- `attention_type='cbam'` 모델 생성
- `attention_type='coord'` 모델 생성
- `[1, 256, 256, 3]` random tensor forward pass
- Generator 출력 shape이 `[1, 256, 256, 3]`인지 확인
- `netE`가 feature map list를 반환하는지 확인

TensorFlow가 설치되어 있어야 실행됩니다.

## 10. 결과 확인

학습 중 생성 이미지는 기본적으로 다음 위치에 저장됩니다.

```text
./output/images
```

Checkpoint는 다음 위치에 저장됩니다.

```text
./output/checkpoints
```

실험별 비교를 위해 `--out_dir`을 다르게 지정하는 것을 권장합니다.

```bash
python train.py \
  --out_dir ./output/cbam_enc_res_r16 \
  --mode cut \
  --train_src_dir ./datasets/SAR/trainA \
  --train_tar_dir ./datasets/Optical/trainB \
  --test_src_dir ./datasets/SAR/testA \
  --test_tar_dir ./datasets/Optical/testB \
  --attention_type cbam \
  --attention_encoder \
  --attention_resblocks \
  --attention_reduction 16
```

## 11. 권장 비교 지표

정성 평가는 `output/images`의 epoch별 결과를 비교합니다.

SAR-to-Optical에서는 다음 항목을 함께 보는 것이 좋습니다.

- 구조 보존: 도로, 건물 경계, 해안선, 농경지 패턴이 유지되는지
- 색 변환 품질: optical domain의 색감이 자연스러운지
- texture hallucination: SAR speckle이 optical texture로 과하게 번역되지 않는지
- mode collapse: 다양한 입력이 비슷한 optical 결과로 수렴하지 않는지
- NCE 안정성: attention 추가 후 학습 초반 loss가 급격히 불안정해지지 않는지

정량 평가를 추가한다면 paired ground truth가 없는 설정에서는 FID/KID, downstream segmentation 성능, 또는 사람이 정의한 지형 클래스별 평가를 별도로 구성해야 합니다.
