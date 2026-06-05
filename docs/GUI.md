# CUT + Attention 학습 GUI 사용법

`gui.py` 는 Gradio 기반 웹 UI로, SAR-to-Optical CUT 모델을 폴더 지정만으로 학습/모니터링합니다.
일반 PC와 Google Colab 모두에서 동작합니다.

## 포함 기능

| # | 요구사항 | 구현 위치 |
| --- | --- | --- |
| 1 | Input/Output 폴더 경로 설정 | 탭 1 "데이터 폴더" |
| 2 | 폴더 지정 시 내부 이미지 파일 읽기/목록 | 탭 1 "📂 폴더 스캔" |
| 2' | 사용할 데이터 쌍 수 선택 (0=전체) | 탭 1 "사용할 데이터 쌍 수" |
| 3 | 기본 학습 파라미터 수정 + 저장 | 탭 2 + "💾 기본 파라미터 저장" |
| 4 | CUT 파라미터 수정 + 저장 | 탭 3 + "💾 CUT 파라미터 저장" |
| 5 | Attention 종류/위치 On·Off (전체/개별) + 저장 | 탭 4 + "모두 ON/OFF", "💾 저장" |
| 6 | 현재 학습률·Epoch·Step·처리 파일명 표시 | 탭 5 모니터링 |
| 7 | 로그 표시 및 파일 저장 | 탭 5 로그창 + `<out_dir>/logs/gui_train_*.log` |
| 8 | PC + Colab 지원 | Gradio 웹 UI |
| + | M4-SAR 데이터셋 다운로드 (Colab 전용) | 탭 0 "데이터셋 다운로드" |
| + | 학습된 체크포인트로 추론/테스트 | 탭 6 "추론 / 테스트" |
| + | SAR 전처리 파이프라인 (모듈형) | 탭 7 "SAR 전처리" |

설정은 모두 `gui_config.json` 에 저장되어 다음 실행 시 자동 복원됩니다.

### 데이터 수량 선택 / 파일명 매칭

SAR(Source)와 Optical(Target)은 **같은 파일명**(예: `00001` ↔ `00001`, 폴더만 다름)으로
대응됩니다. GUI는 두 폴더에 공통으로 존재하는 파일명(확장자 무시)을 기준으로 쌍을
구성하며, "사용할 데이터 쌍 수"(탭 1)로 **앞에서부터 N쌍만** 사용할 수 있습니다(0=전체).
"폴더 스캔"을 누르면 매칭된 학습/검증 쌍 수가 함께 표시됩니다.

> CUT은 unpaired 학습이라 실제 학습 단계에서는 A/B를 독립적으로 셔플합니다. 파일명
> 매칭은 (1) 수량 제한 시 양쪽에서 대응되는 장면을 고르고, (2) 두 폴더가 같은 집합을
> 담고 있는지 확인하는 데 쓰입니다. 파일명이 매칭되지 않으면 독립 목록으로 자동 대체됩니다.

## M4-SAR 데이터셋 다운로드 (Colab 전용)

탭 0에서 HuggingFace [`wchao0601/m4-sar`](https://huggingface.co/datasets/wchao0601/m4-sar)
의 `M4-SAR.zip` 을 받아 지정 폴더에 바로 압축 해제합니다.

- **Colab 환경에서만 기본 활성화**됩니다 (`google.colab` 모듈 감지로 판별).
- 사내망 등 **비-Colab 환경에서는 외부망 차단을 가정해 비활성화**됩니다.
  외부망이 가능한 환경이라면 "외부망 다운로드 강제 허용" 체크박스로 켤 수 있습니다.
- 다운로드 진행률(GB/%, 속도)과 압축 해제 진행이 실시간 표시되며, 완료 후
  추출 폴더의 디렉토리 트리(폴더별 이미지 수)를 보여줍니다. 이 트리를 참고해
  탭 1의 Source/Target 폴더 경로를 지정하면 됩니다.
- gated/비공개 데이터셋이라 401/403이 나오면 HF 토큰 입력란을 사용하세요
  (huggingface.co/settings/tokens). M4-SAR는 공개 데이터셋입니다.

> 참고: M4-SAR는 512×512 optical(10m/60m)·SAR(VH/VV) 이미지로 구성된 대용량
> 데이터셋입니다. 다운로드에 시간이 걸리며 Colab 디스크 용량을 확인하세요.

### CUT 형식으로 자동 정리

탭 0 하단의 "CUT 형식으로 정리"는 추출된 폴더를 CUT 학습 구조
(`trainA`/`trainB`/`testA`/`testB`)로 만들어 줍니다.

- 경로 키워드로 도메인을 분류합니다. 기본값: SAR(Source/A) = `sar,vh,vv`,
  Optical(Target/B) = `optical,opt,rgb,vis,visible`. 실제 폴더명에 맞게 수정 가능.
- 경로에 `test`/`val` 이 있으면 test, 없으면 train 으로 분류합니다. test 폴더가
  전혀 없으면 "분리 비율"(기본 0.1)만큼 도메인별로 무작위로 test 를 떼어냅니다.
- 대용량을 고려해 기본은 **symlink**(원본을 가리키는 링크, 용량 추가 거의 없음)
  이며, 필요 시 **copy** 선택. symlink가 불가한 환경에서는 자동으로 copy 로 대체.
- CUT은 unpaired 학습이라 A/B를 독립적으로 셔플하므로 파일 1:1 정렬은 필요 없으며,
  파일명 충돌 방지를 위해 일련번호 접두어가 붙습니다.
- 완료 시 **탭 1의 Source/Target 경로가 자동으로 채워집니다.**

## 설치

이 코드베이스는 **Keras 2 API** 기준입니다. TensorFlow 버전에 따라 두 가지 방법:

```bash
# (A) Python <= 3.11 : Keras 2 가 포함된 TF 2.15 사용
pip install gradio "tensorflow==2.15.1"     # CPU 전용은 tensorflow-cpu==2.15.1

# (B) 최신 Colab (Python 3.12, TF 2.16+/Keras 3) : tf-keras 호환 레이어 사용
pip install gradio tf-keras
```

> `gui.py` 는 `TF_USE_LEGACY_KERAS=1` 을 자동 설정합니다. 따라서 (B) 환경에서는
> **`tf-keras` 만 설치**하면 기존 Keras 2 코드가 그대로 동작합니다. 설치하지 않으면
> `Layer.__init__() takes 1 positional argument but 2 were given` 같은 Keras 3
> 오류가 납니다.

## 실행 (PC)

```bash
python gui.py                 # http://127.0.0.1:7860
python gui.py --share         # 외부 공유 링크 생성
python gui.py --port 8080     # 포트 변경
```

## 실행 (Google Colab)

새 셀에서 저장소를 클론하고 그대로 실행하면 공유 링크가 자동 생성됩니다.

```python
!git clone https://github.com/nanseko/nanse_test_repo.git
%cd nanse_test_repo
!pip install -q gradio tf-keras          # 최신 Colab(Keras 3) 대비 tf-keras 필수
!python gui.py                           # Colab 감지 시 share 링크 자동 생성
```

> 출력의 `https://XXXX.gradio.live` 공개 URL을 클릭하세요. `127.0.0.1` 은 Colab에서 접속되지 않습니다.

Colab에서 데이터셋은 Google Drive를 마운트해 사용하는 것을 권장합니다.

```python
from google.colab import drive
drive.mount('/content/drive')
# 탭 1에서 폴더 경로를 /content/drive/MyDrive/... 로 지정
```

## 사용 순서

1. **탭 1**에서 Train/Test 의 Source·Target 폴더와 Output 폴더를 입력하고 "폴더 스캔"으로 파일 수 확인
2. **탭 2~4**에서 파라미터를 조정하고 각 탭의 **저장** 버튼 클릭
3. **탭 5**에서 "▶ 학습 시작" — Epoch/Step/학습률/현재 파일/로그가 실시간 갱신
4. 필요 시 "⏹ 중단" (현재 step 완료 후 멈춤). 체크포인트는 `save_n_epoch` 마다 `<out_dir>/checkpoints/` 에 저장

## 추론 / 테스트 (pretrained 체크포인트 사용)

탭 6에서 학습으로 저장된 체크포인트(= pretrained 가중치)를 불러와 테스트 이미지를 바로 변환합니다.

- **가중치 폴더**: 학습 출력의 `<out_dir>/checkpoints` (가장 최신 체크포인트를 자동 로드)
- **입력 폴더**: 변환할 Source(SAR) 이미지 폴더
- **결과 폴더**: `<name>_translated.png` 로 변환 결과 저장 + 미리보기 갤러리 표시
- ⚠️ **탭 3/4의 CUT·Attention 설정이 학습 때와 동일**해야 가중치가 올바르게 로드됩니다.
  (설정은 `gui_config.json` 에 저장되므로, 같은 설정으로 GUI를 켜면 자동 복원됩니다.)
- 별도 학습 없이 추론만 하려면, 학습 때 쓰던 설정 그대로 두고 탭 6만 실행하면 됩니다.

## SAR 전처리 (탭 2, 학습 전)

`docs/README_pipeline.md` 설계를 구현한 모듈형 SAR 전처리 파이프라인입니다.
전처리는 학습 **전** 단계라 데이터 폴더(탭 1) 바로 옆 **탭 2**에 배치되어 있습니다.
(코드: `preprocessing/`, CLI: `scripts/preprocess_pipeline.py`)

- **순서 직접 구성**:
  1. **추가할 전처리(상위 메뉴)** = speckle / intensity / clipping / histogram / resize /
     channel / validate / normalize 를 고르고 `➕ 추가` → 맨 아래 #으로 생성.
  2. 표에서 **행(#)을 클릭**하면 선택되고 **편집 패널**이 열려 파라미터를 설정합니다.
  3. 선택한 #을 `⬆/⬇` 로 이동, `🗑` 로 삭제(선택된 행만), `↺` 로 기본 순서.
  - 같은 speckle을 **Lee → Frost** 처럼 여러 번 넣을 수 있고, 표(위→아래)가 실행 순서입니다.
- **Speckle 필터별 파라미터**: 편집 패널에서 필터 종류 변경 가능(기본 Lee). 필터에 맞는
  파라미터만 표시 — `lee/refined_lee/gamma_map`=window+ENL, `frost`=+damping, `bm3d`=sigma.
  순수 NumPy 구현(`bm3d` 없으면 refined_lee로 자동 대체).
- **Clipping/Histogram**: percentile clipping, `sar_only / unpaired_optical_reference / preset`
  histogram 매핑, CLAHE(opencv 있으면).
- **미리보기**: 첫 이미지 Before/After 즉시 확인.
- **실행**: 진행 로그 + Before|After 갤러리, `output/images`, `manifest.csv`, `logs/` 저장.
- **CUT layout export**: `datasets/M4-SAR-cut/{trainA,testA}`(+optical 지정 시 `trainB,testB`)로 export.

> CLI 예시: `python scripts/preprocess_pipeline.py --input_dir ./raw_sar --output_dir ./pre --max_items 100 --speckle_method refined_lee`
