"""모델 A — Ridge 분포시차 (reduced form).

    y_q = c + Σ_{l=0..L} β_l' F_{q-l} + φ y_{q-1} + e_q

보고 지연·평가 스무딩은 팩터 lag 계수와 AR 항이 흡수한다. 회귀변수는 훈련 표본으로 표준화하고
절편은 벌점하지 않는다. λ 는 훈련창 안에서 시간순(확장창) CV 로 고른다 — 미래 정보 누출 없음.
λ=0 이면 OLS 와 같다 (tests 에서 확인).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """표준화된 X, 중심화된 y 에 대한 ridge. 반환 (coef, intercept). 절편 = mean(y) (X 가 중심화돼 있으므로)."""
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    p = Xc.shape[1]
    A = Xc.T @ Xc + lam * np.eye(p)
    coef = np.linalg.solve(A, Xc.T @ yc)
    return coef, ym - xm @ coef


@dataclass
class RidgeDL:
    alphas: list[float] = field(default_factory=lambda: [0.01, 0.1, 1, 10, 100])
    cv_min_train: int = 20
    standardize: bool = True
    # 적합 결과
    alpha_: float | None = None
    coef_: pd.Series | None = None
    intercept_: float | None = None
    mu_: np.ndarray | None = None
    sd_: np.ndarray | None = None
    cv_sse_: dict | None = None
    resid_sd_: float | None = None

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu_) / self.sd_ if self.standardize else X

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeDL":
        Xv, yv = X.values.astype(float), y.values.astype(float)
        self.mu_ = Xv.mean(0)
        self.sd_ = Xv.std(0, ddof=0)
        self.sd_[self.sd_ == 0] = 1.0
        Xs = self._scale(Xv)
        n = len(yv)
        # 시간순 확장창 CV: t = cv_min_train .. n-1 을 한 점씩 예측
        sse = {}
        for lam in self.alphas:
            e = 0.0
            for t in range(self.cv_min_train, n):
                c, b = _ridge_fit(Xs[:t], yv[:t], lam)
                e += (yv[t] - (Xs[t] @ c + b)) ** 2
            sse[lam] = e
        self.cv_sse_ = sse
        self.alpha_ = min(sse, key=sse.get) if n > self.cv_min_train else self.alphas[len(self.alphas) // 2]
        c, b = _ridge_fit(Xs, yv, self.alpha_)
        self.coef_ = pd.Series(c, index=X.columns)
        self.intercept_ = float(b)
        self.resid_sd_ = float(np.std(yv - (Xs @ c + b), ddof=1))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self._scale(X.values.astype(float))
        return Xs @ self.coef_.values + self.intercept_

    def coef_unscaled(self) -> pd.Series:
        """원 단위(수익률/수익률) 계수."""
        return self.coef_ / self.sd_ if self.standardize else self.coef_


def build_design(fq: pd.DataFrame, y: pd.Series, lags: int, include_ar: bool) -> tuple[pd.DataFrame, pd.Series]:
    """분기 팩터 lag 패널 + AR 항. y 와 X 가 모두 있는 분기만 남긴다."""
    from ..data import quarterly_lag_panel
    X = quarterly_lag_panel(fq, lags)
    if include_ar:
        X["y_l1"] = y.shift(1)
    df = pd.concat([X, y.rename("y")], axis=1).dropna()
    return df.drop(columns="y"), df["y"]
