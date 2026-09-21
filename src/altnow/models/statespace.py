"""모델 B — 월간 잠재 수익률 상태공간 + Geltner 스무딩 관측식 (structural).

월 m 의 '진짜' 로그수익률:      r*_m = α + β'F_m + η_m,  η_m ~ N(0, σ_η²)
분기 q 누적기(상태):            C_m  = δ_m C_{m-1} + r*_m,   δ_m = 0 (분기 첫 달) / 1 (그 외)
분기말에만 관측되는 보고 수익률: y_q  = θ y_{q-1} + (1-θ) C_{m(q)} + ε_q

θ 는 평가 스무딩 계수(Geltner 1991 의 AR(1) 형). (1-θ)β 가 '보고된' 팩터 베타, β 가 '경제적' 베타.
식별: 분기 관측만으로는 σ_η 와 σ_ε 이 따로 식별되지 않아 σ_ε 은 0 으로 고정한다 (모든 잡음을 잠재 수익률에 둔다).
분기 안의 달이 일부만 관측된 경우(진행 중 분기) 남은 달의 F 는 훈련 표본 월평균으로 채운다 — nowcast 는
관측된 달까지의 정보 + 남은 달의 무조건 기대치가 된다.

수익률은 로그로 변환해 합산이 정확히 성립하게 한다. 출력은 다시 단순수익률로 되돌린다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.mlemodel import MLEModel


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


class SmoothedFactorSS(MLEModel):
    """endog: 월간 인덱스, 분기말 달에만 값이 있고 나머지는 NaN (로그 보고수익률).
    exog: 월간 팩터 로그수익률 (n × k). ylag: 분기말 달의 y_{q-1} (그 외 0).

    statsmodels 의 시점 규약은 α_{t+1} = T_t α_t + c_t + R_t η_t 이므로 (index t 의 전이·절편이 t+1 을 만든다)
    맨 앞에 더미 달 하나를 붙여 상태 0 = '첫 분기 시작 전 누적기 = 0' 으로 두고,
    index t 의 transition = δ_{t+1}, state_intercept = α + β'F_{t+1} 로 채운다.
    """

    def __init__(self, endog: pd.Series, exog: pd.DataFrame, ylag: pd.Series,
                 theta_bounds=(0.0, 0.95), obs_var: float = 0.0):
        self.k_fac = exog.shape[1]
        self.fac_names = list(exog.columns)
        self.theta_lo, self.theta_hi = theta_bounds
        self.obs_var = float(obs_var)
        idx = endog.index
        if idx[0].month % 3 != 1:
            raise ValueError("월간 표본은 분기 첫 달(1·4·7·10월)에서 시작해야 누적기 초기값 0 이 맞는다")
        # 더미 달(index 0) 을 앞에 붙인다. 이후 self.months[1:] 가 실제 달.
        dummy = idx[0] - pd.offsets.MonthEnd(1)
        self.months = pd.DatetimeIndex([dummy]).append(idx)
        endog_p = pd.concat([pd.Series([np.nan], index=[dummy]), endog.astype(float)])
        super().__init__(endog=endog_p, k_states=1, k_posdef=1, initialization="known",
                         initial_state=np.zeros(1), initial_state_cov=np.zeros((1, 1)))
        n = self.nobs                                                        # = len(idx) + 1
        F = np.asarray(exog.values, dtype=float)
        # index t 의 전이/절편은 t+1 을 만든다 → F_{t+1}, δ_{t+1}. 마지막 index 는 쓰이지 않는다(0 으로 둔다).
        self._F_next = np.vstack([F, np.zeros((1, self.k_fac))])            # n × k
        delta_next = np.array([0.0 if t.month % 3 == 1 else 1.0 for t in idx] + [0.0])
        self._ylag = np.r_[0.0, np.asarray(ylag.values, dtype=float)]
        self.ssm["transition"] = delta_next.reshape(1, 1, n)
        self.ssm["selection"] = np.ones((1, 1))
        self.ssm["obs_cov"] = np.array([[self.obs_var]])
        self.ssm["state_intercept"] = np.zeros((1, n))
        self.ssm["obs_intercept"] = np.zeros((1, n))
        self.ssm["design"] = np.zeros((1, 1, n))

    @property
    def param_names(self):
        return ["alpha"] + [f"beta_{c}" for c in self.fac_names] + ["log_sigma_eta", "theta_raw"]

    @property
    def start_params(self):
        return np.r_[0.0, np.zeros(self.k_fac), np.log(0.02), _logit(0.5)]

    def unpack(self, params):
        params = np.asarray(params, dtype=float)
        alpha = params[0]
        beta = params[1:1 + self.k_fac]
        sig = np.exp(params[1 + self.k_fac])
        theta = self.theta_lo + (self.theta_hi - self.theta_lo) * _sigmoid(params[2 + self.k_fac])
        return alpha, beta, sig, theta

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        alpha, beta, sig, theta = self.unpack(params)
        n = self.nobs
        c = alpha + self._F_next @ beta
        c[-1] = 0.0
        self.ssm["state_intercept"] = c.reshape(1, n)
        self.ssm["state_cov"] = np.array([[sig ** 2]])
        self.ssm["design"] = np.full((1, 1, n), 1.0 - theta)
        self.ssm["obs_intercept"] = (theta * self._ylag).reshape(1, n)


def monthly_frame(fm_log: pd.DataFrame, yq_log: pd.Series, fill_mean: pd.Series | None = None) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """월간 팩터 + 분기 관측을 상태공간 입력으로 정렬한다.
    - 시작: 팩터·타깃 모두 있는 첫 분기의 첫 달
    - 끝:  fm_log 의 마지막 달이 속한 분기의 분기말 (없는 달의 F 는 fill_mean 으로 채움)
    - endog: 분기말 달에 y_q (y_{q-1} 도 있어야 관측으로 인정; 아니면 NaN), ylag: 그 달의 y_{q-1}
    """
    yq = yq_log.dropna()
    first_q = max(yq.index[0], fm_log.index[0] + pd.offsets.QuarterEnd(0))
    # 첫 분기는 그 분기 세 달의 팩터가 모두 있어야 한다
    while (fm_log.index[0] > pd.Timestamp(first_q.year, first_q.month - 2, 1)):
        first_q = first_q + pd.offsets.QuarterEnd(1)
    start_m = pd.Timestamp(first_q.year, first_q.month - 2, 1) + pd.offsets.MonthEnd(0)
    end_m = fm_log.index[-1] + pd.offsets.QuarterEnd(0)
    months = pd.date_range(start_m, end_m, freq="ME")
    F = fm_log.reindex(months)
    if F.isna().any().any():
        if fill_mean is None:
            raise ValueError("팩터 결측 달이 있는데 fill_mean 이 없다")
        F = F.fillna(fill_mean)
    endog = pd.Series(np.nan, index=months)
    ylag = pd.Series(0.0, index=months)
    for t in months:
        if t.month % 3 == 0 and t in yq.index:
            prev = t - pd.offsets.QuarterEnd(1)
            if prev in yq.index:
                endog[t] = yq[t]
                ylag[t] = yq[prev]
    return endog, F, ylag


class SSResult:
    def __init__(self, model: SmoothedFactorSS, res):
        self.model, self.res, self.months = model, res, model.months     # months[0] 은 더미 달
        self.alpha, self.beta, self.sigma_eta, self.theta = model.unpack(res.params)

    def params_table(self) -> pd.Series:
        s = {"alpha_m": self.alpha, "theta": self.theta, "sigma_eta_m": self.sigma_eta, "llf": float(self.res.llf)}
        for k, b in zip(self.model.fac_names, self.beta):
            s[f"beta_{k}"] = b                       # 경제적(unsmoothed) 베타
            s[f"beta_reported_{k}"] = (1 - self.theta) * b
        return pd.Series(s)

    def nowcast(self, quarter_end: pd.Timestamp) -> tuple[float, float]:
        """분기말 t 의 보고 로그수익률 예측치와 표준편차: θ y_{q-1} + (1-θ) E[C_t | 그때까지의 F]."""
        i = self.months.get_loc(quarter_end)
        # predicted_state[:, i] = 관측 반영 전 (t 의 y 가 NaN 이면 filtered = predicted)
        c_hat = float(self.res.predicted_state[0, i])
        c_var = float(self.res.predicted_state_cov[0, 0, i])
        ylag = float(self.model._ylag[i])
        mu = self.theta * ylag + (1 - self.theta) * c_hat
        sd = float(np.sqrt((1 - self.theta) ** 2 * c_var + self.model.obs_var))
        return mu, sd

    def unsmoothed_monthly(self) -> pd.Series:
        """스무딩된 잠재 월간 로그수익률 E[r*_m | 전체 표본]: C_m − δ_m C_{m−1}."""
        C = pd.Series(self.res.smoothed_state[0], index=self.months)
        delta = pd.Series([0.0 if t.month % 3 == 1 else 1.0 for t in self.months], index=self.months)
        return (C - delta * C.shift(1).fillna(0.0)).iloc[1:]


def fit_ss(endog: pd.Series, F: pd.DataFrame, ylag: pd.Series, theta_bounds=(0.0, 0.95),
           start_params=None, maxiter: int = 500) -> SSResult:
    mod = SmoothedFactorSS(endog, F, ylag, theta_bounds=theta_bounds)
    sp = start_params if start_params is not None else _ols_start(mod, endog, F, ylag)
    res = mod.fit(start_params=sp, method="lbfgs", maxiter=maxiter, disp=False)
    return SSResult(mod, res)


def _ols_start(mod: SmoothedFactorSS, endog: pd.Series, F: pd.DataFrame, ylag: pd.Series) -> np.ndarray:
    """분기 OLS y = c + φ y_{q-1} + b'ΣF 에서 시작값을 만든다."""
    obs = endog.dropna()
    Fq = F.rolling(3).sum().reindex(obs.index)
    X = np.column_stack([np.ones(len(obs)), ylag.reindex(obs.index).values, Fq.values])
    coef, *_ = np.linalg.lstsq(X, obs.values, rcond=None)
    theta0 = float(np.clip(coef[1], mod.theta_lo + 0.02, mod.theta_hi - 0.02))
    b0 = coef[2:] / (1 - theta0)
    a0 = coef[0] / (3 * (1 - theta0))
    resid = obs.values - X @ coef
    sig0 = max(np.std(resid) / (np.sqrt(3) * (1 - theta0)), 1e-4)
    return np.r_[a0, b0, np.log(sig0), _logit((theta0 - mod.theta_lo) / (mod.theta_hi - mod.theta_lo))]
