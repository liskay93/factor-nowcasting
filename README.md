# factor-nowcasting — 매크로 팩터로 대체투자 분기수익률 nowcast

`macro_factor` 저장소의 4개 매크로 팩터(growth / term / inflation / credit, 일간)로 Burgiss·MSCI·Giliberto-Levy·HFRI·FTSE 등
대체투자 지수의 **분기 수익률**을 nowcast 한다. 대체투자 지수는 분기 빈도이고 1~2분기 늦게 발표되며 감정평가 스무딩이 있어,
(1) 빈도 불일치, (2) 보고 지연, (3) 스무딩 세 가지를 같이 다뤄야 한다.

- 결과 리포트: [`reports/nowcast_report.md`](reports/nowcast_report.md) · 노트북: [`notebooks/report.ipynb`](notebooks/report.ipynb)
- 최신 nowcast: `data/processed/nowcast_latest.csv` (시리즈별) · `nowcast_by_label_ensemble.csv` (엑셀 21 라벨)

## 데이터

| 자료 | 위치 | 비고 |
|---|---|---|
| 대체투자 분기 수익률 (21 라벨, 2000Q1~2026Q1) | `data/raw/alt_returns_bm.xlsx` (`값` 시트) | 원지수가 같은 라벨은 한 시리즈로 추정 → 15 시리즈. 매핑은 `config/nowcast.yaml` `series` |
| 매크로 팩터 일간 수익률 | `../macro_factor/data/processed/factors.csv` | **읽기 전용.** 환경변수 `MACRO_FACTOR_ROOT` 로 경로 지정 가능. 부호·정의는 그 저장소 `SPEC.md` |

팩터 일간 → 월·분기는 복리 `Π(1+r)−1`. 부분 기간(진행 중 분기, 2006Q1)은 훈련에서 빼고, nowcast 에서는 `QTD + (1−경과비율)×표본평균` 으로 채운다.

## 모델

| | 식 | 성격 |
|---|---|---|
| **A · Ridge 분포시차** (`src/altnow/models/ridge_dl.py`) | y_q = c + Σ_{l=0..4} β_l'F_{q−l} + φ y_{q−1} + e | reduced form. 표준화 후 ridge, λ 는 훈련창 안 시간순 CV. λ=0 은 OLS |
| **B · 월간 상태공간** (`src/altnow/models/statespace.py`) | r*_m = α + β'F_m + η_m ;  y_q = θ y_{q−1} + (1−θ) Σ_{m∈q} r*_m | structural. Geltner(1991) 스무딩 θ 를 칼만 필터 MLE 로 추정. 월간 잠재 수익률·부분 분기·반복 예측이 자연스럽다. 관측오차는 분기 자료만으로 식별되지 않아 0 고정 |
| Ensemble | (A+B)/2 | |
| 벤치마크 | 확장창 평균 · AR(1) · **OLS** y_q = c + φ y_{q−1} + b'F_q | OLS 는 B 의 완전분기 점예측과 같은 식(재매개화)이라 OOS RMSE 가 거의 같다 |

## 실행

```bash
pip install -r requirements.txt
python scripts/run_backtest.py      # 확장창 OOS 백테스트 → data/processed/backtest_*.csv   (~40초)
python scripts/run_nowcast.py       # 최신 nowcast → data/processed/nowcast_*.csv, model_*_full.csv
python scripts/make_report.py       # 그림 + reports/nowcast_report.md
python scripts/make_notebook.py && jupyter nbconvert --to notebook --execute --inplace notebooks/report.ipynb
python -m pytest -q                 # 집계 정합성 · ridge=OLS · 상태공간 항등식/모수 복원
```

## 결과 요약 (팩터 기준일 2026-09-15, OOS 2012Q1~2026Q1)

Ensemble 의 R²_OOS (대 확장창 평균), 괄호는 AR(1):

| 그룹 | 잘 되는 것 | 안 되는 것 |
|---|---|---|
| PE | Buyout 0.68 (0.04), FoF Primary 0.65 (0.38), Mezzanine 0.74 (−0.08), VC 0.53 (0.41) | Senior Loan 0.26 (−0.01) — 작은 변동성 |
| HF | HFRI FoF Cons 0.77 (−0.12) | |
| RE | Listed RE 0.63 (−0.07), G-L 1 0.68 (−0.05) | Burgiss RE 0.14 (0.39), MSCI GPFI 0.30 (0.29), G-L 2 −0.40 (−0.12) |
| INFRA | FTSE FI Infra 0.91 (−0.10), Burgiss Infra 0.27 (−0.03) | |

그룹 평균 R²_OOS: HF 0.77 · INFRA 0.59 · PE 0.52 · RE 0.27 (Ensemble); 동분기 OLS 는 0.73 · 0.57 · 0.45 · 0.14.

- **완전 분기 점예측은 동분기 OLS(팩터 + AR(1))로 거의 다 설명된다.** B 의 값어치는 부분 분기(QTD) 월 단위 반영, θ/β 분해, 예측 표준편차이고, A 의 팩터 lag 1~4 는 OOS 에서 도움이 안 된다.
- 감정평가 부동산은 θ 가 0.6~0.7 이라 AR(1) 이 강하고 4개 팩터가 더해 주는 것이 적다. Ensemble 이 RE 에서 0.27 로 올라가는 것은 두 모델 오차가 덜 상관돼서다.
- 스무딩 θ 추정치: Burgiss RE 0.63, MSCI GPFI 0.71, VC 0.50, FoF 0.40~0.49, Buyout 0.30, 상장·채권형은 0~0.09.

## 주의

- **의사 실시간**이다. 지수 최종 빈티지를 쓰고 y_{q−1} 을 q 분기말에 안다고 가정한다. 실전에서는 발표 지연만큼 정확도가 떨어진다.
- OOS 34~57분기로 짧다. DM 검정 p 값은 참고용.
- 팩터 4개(주식·금리·기대인플레·크레딧)만 쓴다. 부동산·인프라에는 실질금리·유동성·섹터 팩터 추가가 자연스러운 다음 단계다.

## 구조

```
config/nowcast.yaml        경로·시리즈 매핑·모델·백테스트 설정
src/altnow/data.py         엑셀 로더(dedupe), 팩터 로더(읽기 전용), 복리 집계, 부분기간 판별
src/altnow/models/         ridge_dl.py (모델 A) · statespace.py (모델 B)
src/altnow/backtest.py     확장창 OOS
src/altnow/nowcast.py      최신 nowcast (완전 분기 + QTD, 반복 예측)
src/altnow/metrics.py      RMSE·MAE·적중률·R²_OOS·Diebold-Mariano
src/altnow/plots.py        그림
scripts/                   run_backtest · run_nowcast · make_report
tests/                     정합성 검사
```
