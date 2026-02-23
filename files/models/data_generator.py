"""
Synthetic Data Generator
Produces realistic banking (credit default) and insurance (high-claim) datasets
for educational and research purposes. No real user data is used.
"""

import numpy as np
import pandas as pd

_banking_df = None
_insurance_df = None


def generate_banking_data(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    """
    Simulate a retail-banking credit portfolio.
    Target: loan_default (1 = default, 0 = no default)
    """
    rng = np.random.default_rng(random_state)

    age               = rng.integers(22, 70, n_samples)
    annual_income     = np.clip(rng.lognormal(10.5, 0.5, n_samples), 20_000, 500_000).astype(int)
    employment_years  = rng.integers(0, 30, n_samples)
    loan_amount       = rng.integers(5_000, 150_000, n_samples)
    credit_score      = rng.integers(300, 850, n_samples)
    debt_to_income    = rng.uniform(0.05, 0.65, n_samples).round(3)
    num_open_accounts = rng.integers(1, 20, n_samples)
    delinquencies_2yr = rng.poisson(0.4, n_samples)
    home_ownership    = rng.choice(['OWN', 'RENT', 'MORTGAGE'], n_samples, p=[0.20, 0.38, 0.42])
    loan_purpose      = rng.choice(
        ['debt_consolidation', 'home_improvement', 'major_purchase', 'medical', 'other'],
        n_samples, p=[0.35, 0.20, 0.18, 0.12, 0.15]
    )
    int_rate          = rng.uniform(5.0, 28.0, n_samples).round(2)

    log_odds = (
        -2.2
        + 0.012 * (age - 40)
        - 2e-6  * annual_income
        - 0.03  * employment_years
        + 3e-6  * loan_amount
        - 0.001 * credit_score        # reduced to avoid dominating
        + 2.2   * debt_to_income
        + 0.50  * delinquencies_2yr
        - 0.04  * num_open_accounts
        + 0.04  * int_rate
        + rng.normal(0, 0.5, n_samples)
    )
    prob    = 1 / (1 + np.exp(-log_odds))
    default = (rng.random(n_samples) < prob).astype(int)

    return pd.DataFrame({
        'age':               age,
        'annual_income':     annual_income,
        'employment_years':  employment_years,
        'loan_amount':       loan_amount,
        'credit_score':      credit_score,
        'debt_to_income':    debt_to_income,
        'num_open_accounts': num_open_accounts,
        'delinquencies_2yr': delinquencies_2yr,
        'int_rate':          int_rate,
        'home_ownership':    home_ownership,
        'loan_purpose':      loan_purpose,
        'default':           default,
    })


def generate_insurance_data(n_samples: int = 2000, random_state: int = 99) -> pd.DataFrame:
    """
    Simulate a personal-lines insurance portfolio.
    Target: high_claim (1 = claim cost above threshold, 0 = below)
    """
    rng = np.random.default_rng(random_state)

    age           = rng.integers(18, 75, n_samples)
    bmi           = np.clip(rng.normal(27.5, 5.5, n_samples), 15, 55).round(1)
    smoker        = rng.choice([0, 1], n_samples, p=[0.76, 0.24])
    num_dependents= rng.integers(0, 5, n_samples)
    region        = rng.choice(['northeast', 'northwest', 'southeast', 'southwest'], n_samples)
    years_insured = rng.integers(1, 30, n_samples)
    prior_claims  = rng.poisson(0.3, n_samples)
    annual_premium= np.clip(
        1_200 + 28 * age + 80 * bmi + 12_000 * smoker + 400 * num_dependents
        + rng.normal(0, 600, n_samples), 400, 60_000
    ).round(0).astype(int)

    log_odds = (
        -3.5
        + 0.020 * (age - 40)
        + 0.045 * (bmi - 25)
        + 1.60  * smoker
        + 0.18  * num_dependents
        + 0.35  * prior_claims
        - 0.01  * years_insured
        + rng.normal(0, 0.35, n_samples)
    )
    prob       = 1 / (1 + np.exp(-log_odds))
    high_claim = (rng.random(n_samples) < prob).astype(int)

    return pd.DataFrame({
        'age':            age,
        'bmi':            bmi,
        'smoker':         smoker,
        'num_dependents': num_dependents,
        'years_insured':  years_insured,
        'prior_claims':   prior_claims,
        'annual_premium': annual_premium,
        'region':         region,
        'high_claim':     high_claim,
    })


# ── Cached singletons ────────────────────────────────────────────────────────

def get_banking_data() -> pd.DataFrame:
    global _banking_df
    if _banking_df is None:
        _banking_df = generate_banking_data()
    return _banking_df.copy()


def get_insurance_data() -> pd.DataFrame:
    global _insurance_df
    if _insurance_df is None:
        _insurance_df = generate_insurance_data()
    return _insurance_df.copy()
