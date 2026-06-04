# CUT + Attention 학습 GUI 사용법

`gui.py` 는 Gradio 기반 웹 UI로, SAR-to-Optical CUT 모델을 폴더 지정만으로 학습/모니터링합니다.
일반 PC와 Google Colab 모두에서 동작합니다.

## 포함 기능

| # | 요구사항 | 구현 위치 |
| --- | --- | --- |
| 1 | Input/Output 폴더 경로 설정 | 탭 1 "데이터 폴더" |
| 2 | 폴더 지정 시 내부 이미지 파일 읽기/목록 | 탭 1 "📂 폴더 스캔" |
| 3 | 기본 학습 파라미터 수정 + 저장 | 탭 2 + "💾 기본 파라미터 저장" |
| 4 | CUT 파라미터 수정 + 저장 | 탭 3 + "💾 CUT 파라미터 저장" |
| 5 | Attention 종류/위치 On·Off (전체/개별) + 저장 | 탭 4 + "모두 ON/OFF", "💾 저장" |
| 6 | 현재 학습률·Epoch·Step·처리 파일명 표시 | 탭 5 모니터링 |
| 7 | 로그 표시 및 파일 저장 | 탭 5 로그창 + `<out_dir>/logs/gui_train_*.log` |
| 8 | PC + Colab 지원 | Gradio 웹 UI |
| + | M4-SAR 데이터셋 다운로드 (Colab 전용) | 탭 0 "데이터셋 다운로드" |

설정은 모두 `gui_config.json` 에 저장되어 다음 실행 시 자동 복원됩니다.

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

```bash
pip install -r requirements_gui.txt
# 또는
pip install gradio "tensorflow-cpu==2.15.1"   # GPU 환경은 tensorflow==2.15.1
```

> 이 저장소 코드는 TensorFlow 2.15(Keras 2) 기준입니다. Keras 3(TF 2.16+)에서는 호환되지 않습니다.

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
!pip install -q gradio "tensorflow==2.15.1"
!python gui.py          # Colab 감지 시 share 링크 자동 생성
```

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
