# Al₂O₃ Thermal ALE 레시피 최적화 — 웹 데모

`ale_ml 2`의 보정식, 대리모델, 모델 A/B 최적화를 Streamlit 화면으로 만든 데모입니다.
원본 코드(`generator.py`, `train.py`, `optimize.py`)는 수정하지 않고 복사해서 사용합니다.

## 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```
브라우저에서 http://localhost:8501 이 열립니다. (Python 3.13에서 확인)

## 구성
| 파일 | 역할 |
|---|---|
| `app.py` | 화면 3개: ① 레시피 최적화 ② Trade-off 분석 ③ 모델 근거와 한계 |
| `core.py` | `run()`(모델 A/B + 제약 체크), 물리식 기반 빠른 최적화와 민감도 계산 |
| `generator.py`, `train.py`, `optimize.py`, `params.json` | `ale_ml 2` 원본 그대로 |
| `models/` | 학습된 대리모델 (GP: cycle time, HistGB: EPC) |
| `assets/` | 보정·학습 결과 표와 그림 |

## 계산 방식
- ① 화면: ML 대리모델로 최적점을 찾고 물리 생성식으로 다시 계산해 제약을 확인합니다 (`optimize.py`와 동일).
- ② 화면: 물리 생성식만으로 모델 B와 같은 문제를 빠르게 풉니다. 대리모델은 허용 손실 5%로만 학습돼 있어서,
  허용 손실을 바꾸는 분석은 이 방식으로만 가능합니다. 허용 손실 5%에서는 ①과 결과가 일치합니다.

## 배포 (Streamlit Community Cloud)
1. 이 폴더를 GitHub 저장소 루트에 올립니다 (`app.py`가 루트에 있어야 함).
2. https://share.streamlit.io 에서 GitHub 계정으로 로그인하고 저장소와 `app.py`를 선택합니다.
3. 고급 설정에서 Python 버전을 3.13으로 맞춥니다. (대리모델 파일이 sklearn 버전에 민감해 `requirements.txt`의 버전을 바꾸지 마세요)

## 주의
합성 데이터(논문 anchor 보정) 기반 예측이며 실제 장비 검증 전의 결과입니다. 장비 퍼지 τ는 시나리오(×1 ~ ×100)로 비교하세요.
