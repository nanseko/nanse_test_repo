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

설정은 모두 `gui_config.json` 에 저장되어 다음 실행 시 자동 복원됩니다.

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
