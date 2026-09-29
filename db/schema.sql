-- 대시보드 테이블 (Postgres/MySQL/SQLite 공통 문법). load_db.py 가 같은 구조로 만든다.

CREATE TABLE IF NOT EXISTS beta_timeseries (
    date          DATE         NOT NULL,   -- 분기말 (롤링 창 끝)
    program       VARCHAR(64)  NOT NULL,   -- 엑셀 21 라벨 (예: 'Buyout', 'Core RE Equity')
    program_code  VARCHAR(32)  NOT NULL,   -- 원지수 코드 (같은 원지수 라벨은 같은 값)
    "group"       VARCHAR(16),             -- PE / HF / RE / INFRA
    index_name    VARCHAR(128),            -- 원지수 정식명
    factor        VARCHAR(16)  NOT NULL,   -- growth / term / inflation / credit
    basis         VARCHAR(16)  NOT NULL,   -- reported / unsmoothed
    method        VARCHAR(16)  NOT NULL,   -- ols / priority
    window_q      SMALLINT     NOT NULL,   -- 롤링 창(분기)
    beta          DOUBLE PRECISION NOT NULL,
    r2            DOUBLE PRECISION,        -- ols 만. priority 는 NULL
    n_obs         SMALLINT,
    run_id        VARCHAR(32)  NOT NULL,   -- 산출 시각 (UTC)
    PRIMARY KEY (date, program, factor, basis, method, window_q)
);
CREATE INDEX IF NOT EXISTS ix_bts_program_factor ON beta_timeseries (program, factor, basis, method, date);

CREATE TABLE IF NOT EXISTS beta_latest (
    program       VARCHAR(64)  NOT NULL,
    program_code  VARCHAR(32)  NOT NULL,
    "group"       VARCHAR(16),
    index_name    VARCHAR(128),
    asof_quarter  DATE         NOT NULL,
    growth        DOUBLE PRECISION,
    term          DOUBLE PRECISION,
    inflation     DOUBLE PRECISION,
    credit        DOUBLE PRECISION,
    r2            DOUBLE PRECISION,
    n_obs         SMALLINT,
    basis         VARCHAR(16)  NOT NULL,
    method        VARCHAR(16)  NOT NULL,
    window_q      SMALLINT     NOT NULL,
    PRIMARY KEY (program, basis, method, window_q)
);

CREATE TABLE IF NOT EXISTS factor_latest (
    factor            VARCHAR(16) PRIMARY KEY,
    last_full_quarter DATE,
    ret_last_quarter  DOUBLE PRECISION,
    ret_4q            DOUBLE PRECISION,
    qtd               DOUBLE PRECISION,
    qtd_asof          DATE,
    qtd_quarter       DATE,
    qtd_days          SMALLINT,
    ann_vol_5y        DOUBLE PRECISION,
    ann_vol_full      DOUBLE PRECISION
);
