import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import json

from flask import Flask, render_template_string, jsonify

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    RandomForestRegressor, GradientBoostingRegressor, IsolationForest,
)
from sklearn.linear_model import LogisticRegression, TweedieRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, roc_curve, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error, f1_score
)
import statsmodels.api as sm
from plotly.utils import PlotlyJSONEncoder

app = Flask(__name__)

# ──────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATORS
# ──────────────────────────────────────────────────────────────

def gen_credit_risk(n=2500, seed=42):
    np.random.seed(seed)
    age              = np.random.normal(40, 12, n).clip(18, 80).astype(int)
    income           = np.exp(np.random.normal(10.6, 0.5, n)).clip(20_000, 600_000)
    loan_amount      = (income * np.random.uniform(0.5, 5, n)).clip(1_000, 1_000_000)
    credit_score     = np.random.normal(680, 80, n).clip(300, 850).astype(int)
    employment_years = np.random.gamma(3, 3, n).clip(0, 40).round(1)
    debt_to_income   = np.random.beta(2, 5, n).round(3)
    num_accounts     = np.random.poisson(4, n).clip(1, 15).astype(int)
    late_payments    = np.random.poisson(0.5, n).clip(0, 10).astype(int)
    ltv              = (loan_amount / (loan_amount + income * 2) + np.random.normal(0, 0.05, n)).clip(0.1, 1.5).round(3)
    logit   = -0.012 * credit_score + 2.5 * debt_to_income + 0.3 * late_payments \
              - 0.05 * employment_years + 0.4 * ltv - 0.5
    default = np.random.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame({
        "age": age, "income": income.astype(int), "loan_amount": loan_amount.astype(int),
        "credit_score": credit_score, "employment_years": employment_years,
        "debt_to_income": debt_to_income, "num_accounts": num_accounts,
        "late_payments": late_payments, "ltv": ltv, "default": default
    })


def gen_insurance(n=2500, seed=42):
    np.random.seed(seed)
    age            = np.random.normal(42, 15, n).clip(18, 80).astype(int)
    vehicle_age    = np.random.exponential(5, n).clip(0, 20).astype(int)
    annual_mileage = np.random.normal(12_000, 4_000, n).clip(1_000, 40_000).astype(int)
    accident_hist  = np.random.poisson(0.3, n).clip(0, 5).astype(int)
    vehicle_value  = np.random.lognormal(10, 0.5, n).clip(5_000, 100_000).astype(int)
    credit_score   = np.random.normal(690, 75, n).clip(300, 850).astype(int)
    urban          = np.random.binomial(1, 0.6, n)
    coverage_type  = np.random.choice([0, 1, 2], n, p=[0.3, 0.4, 0.3])
    premium        = (500 + 200 * accident_hist + 0.01 * annual_mileage
                      + 20 * vehicle_age - 0.3 * (credit_score - 600)
                      + 200 * urban + 300 * coverage_type
                      + np.random.normal(0, 80, n)).clip(300, 5_000).round(2)
    return pd.DataFrame({
        "age": age, "vehicle_age": vehicle_age, "annual_mileage": annual_mileage,
        "accident_history": accident_hist, "vehicle_value": vehicle_value,
        "credit_score": credit_score, "urban": urban, "coverage_type": coverage_type,
        "premium": premium
    })


def gen_churn(n=2500, seed=42):
    np.random.seed(seed)
    tenure       = np.random.exponential(24, n).clip(1, 120).astype(int)
    num_products = np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.35, 0.2, 0.05])
    balance      = np.random.lognormal(8, 1.5, n).clip(0, 200_000).round(2)
    is_active    = np.random.binomial(1, 0.7, n)
    credit_score = np.random.normal(670, 90, n).clip(300, 850).astype(int)
    age          = np.random.normal(40, 14, n).clip(18, 80).astype(int)
    salary       = np.exp(np.random.normal(10.7, 0.5, n)).astype(int)
    complaints   = np.random.poisson(0.4, n).clip(0, 5).astype(int)
    satisfaction = np.random.normal(7, 2, n).clip(1, 10).round(1)
    logit   = -0.02 * tenure - 0.3 * num_products + 0.5 * complaints \
              - 0.3 * is_active - 0.3 * (satisfaction - 5) + 0.002 * (700 - credit_score)
    churned = np.random.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame({
        "tenure_months": tenure, "num_products": num_products, "balance": balance,
        "is_active": is_active, "credit_score": credit_score, "age": age,
        "salary": salary, "complaints": complaints, "satisfaction": satisfaction,
        "churned": churned
    })


def gen_fraud(n=3000, fraud_rate=0.05, seed=42):
    np.random.seed(seed)
    nf, nl = int(n * fraud_rate), int(n * (1 - fraud_rate))
    legit_w = np.array([0.5]*8 + [3.0]*8 + [2.0]*8); legit_w /= legit_w.sum()
    fraud_w = np.array([3.0]*8 + [0.5]*8 + [1.0]*8); fraud_w /= fraud_w.sum()
    return pd.DataFrame({
        "amount":        np.concatenate([np.random.lognormal(4,1.2,nl).clip(1,5000),
                                         np.random.lognormal(6,1.5,nf).clip(100,50000)]).round(2),
        "hour":          np.concatenate([np.random.choice(24,nl,p=legit_w),
                                         np.random.choice(24,nf,p=fraud_w)]),
        "velocity":      np.concatenate([np.random.poisson(2,nl).clip(0,10),
                                         np.random.poisson(8,nf).clip(0,30)]),
        "distance_km":   np.concatenate([np.random.exponential(20,nl).clip(0,200),
                                         np.random.exponential(100,nf).clip(0,1000)]).round(1),
        "merchant_risk": np.concatenate([np.random.beta(1,5,nl),
                                         np.random.beta(5,2,nf)]).round(3),
        "is_fraud":      np.concatenate([np.zeros(nl), np.ones(nf)]).astype(int)
    }).sample(frac=1, random_state=seed).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# MODEL TRAINING
# ──────────────────────────────────────────────────────────────

def get_feat_imp(mdl, features):
    if hasattr(mdl, "feature_importances_"):
        imp = mdl.feature_importances_
        pairs = sorted(zip(features, imp.tolist()), key=lambda x: -x[1])
        return [{"feature": f, "importance": round(v, 4)} for f, v in pairs]
    elif hasattr(mdl, "coef_"):
        imp = np.abs(mdl.coef_[0]) if mdl.coef_.ndim > 1 else np.abs(mdl.coef_)
        pairs = sorted(zip(features, imp.tolist()), key=lambda x: -x[1])
        return [{"feature": f, "importance": round(v, 4)} for f, v in pairs]
    return []


def train_classification_suite(df, features, target, seed=42):
    X = df[features].values
    y = df[target].values
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    scaler = StandardScaler()
    Xtr_s, Xte_s = scaler.fit_transform(Xtr), scaler.transform(Xte)

    models = {
        "Decision Tree":     DecisionTreeClassifier(max_depth=5, random_state=seed),
        "Bagging (RF)":      RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed),
        "AdaBoost":          AdaBoostClassifier(n_estimators=100, random_state=seed, algorithm="SAMME"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed),
        "Logistic (GLM)":    LogisticRegression(max_iter=1000, random_state=seed),
    }
    results = {}
    for name, mdl in models.items():
        Xt, Xv = (Xtr_s, Xte_s) if name == "Logistic (GLM)" else (Xtr, Xte)
        mdl.fit(Xt, ytr)
        proba = mdl.predict_proba(Xv)[:, 1]
        pred  = mdl.predict(Xv)
        fpr, tpr, _ = roc_curve(yte, proba)
        cm = confusion_matrix(yte, pred)
        results[name] = {
            "auc": round(roc_auc_score(yte, proba), 4),
            "accuracy": round(accuracy_score(yte, pred), 4),
            "f1": round(f1_score(yte, pred), 4),
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "cm": cm.tolist(),
            "feat_imp": get_feat_imp(mdl, features),
        }
        if name == "Decision Tree":
            results[name]["tree_rules"] = export_text(mdl, feature_names=features, max_depth=3)

    # GLM via statsmodels
    Xsm, Xsm_te = sm.add_constant(Xtr_s), sm.add_constant(Xte_s)
    try:
        glm = sm.GLM(ytr, Xsm, family=sm.families.Binomial()).fit(disp=False)
        gp  = glm.predict(Xsm_te)
        gpr = (gp > 0.5).astype(int)
        fpr, tpr, _ = roc_curve(yte, gp)
        results["GLM (statsmodels)"] = {
            "auc": round(roc_auc_score(yte, gp), 4),
            "accuracy": round(accuracy_score(yte, gpr), 4),
            "f1": round(f1_score(yte, gpr), 4),
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "cm": confusion_matrix(yte, gpr).tolist(),
            "feat_imp": [],
            "coef":    dict(zip(["const"] + features, glm.params.round(4).tolist())),
            "pvalues": dict(zip(["const"] + features, glm.pvalues.round(4).tolist())),
        }
    except Exception:
        pass

    results["_meta"] = {
        "n_train": len(ytr), "n_test": len(yte),
        "target_rate": round(y.mean(), 4),
        "features": features,
        "sample": df.head(8).to_dict(orient="records")
    }
    return results


def train_regression_suite(df, features, target, seed=42):
    X = df[features].values
    y = df[target].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    scaler = StandardScaler()
    Xtr_s, Xte_s = scaler.fit_transform(Xtr), scaler.transform(Xte)

    models = {
        "Decision Tree":     DecisionTreeRegressor(max_depth=5, random_state=seed),
        "Random Forest":     RandomForestRegressor(n_estimators=100, max_depth=8, random_state=seed),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=seed),
        "Tweedie GLM":       TweedieRegressor(power=1.5, max_iter=500),
    }
    results = {}
    for name, mdl in models.items():
        Xt, Xv = (Xtr_s, Xte_s) if name == "Tweedie GLM" else (Xtr, Xte)
        mdl.fit(Xt, ytr)
        pred = mdl.predict(Xv).clip(0)
        results[name] = {
            "rmse": round(np.sqrt(mean_squared_error(yte, pred)), 2),
            "mae":  round(mean_absolute_error(yte, pred), 2),
            "r2":   round(r2_score(yte, pred), 4),
            "pred_sample":   pred[:200].tolist(),
            "actual_sample": yte[:200].tolist(),
            "feat_imp": get_feat_imp(mdl, features),
        }

    # statsmodels Gamma GLM
    Xsm, Xsm_te = sm.add_constant(Xtr_s), sm.add_constant(Xte_s)
    try:
        gm  = sm.GLM(ytr, Xsm, family=sm.families.Gamma(sm.families.links.Log())).fit(disp=False)
        gp  = gm.predict(Xsm_te).clip(0)
        results["GLM Gamma (statsmodels)"] = {
            "rmse": round(np.sqrt(mean_squared_error(yte, gp)), 2),
            "mae":  round(mean_absolute_error(yte, gp), 2),
            "r2":   round(r2_score(yte, gp), 4),
            "pred_sample":   gp[:200].tolist(),
            "actual_sample": yte[:200].tolist(),
            "feat_imp": [],
            "coef": dict(zip(["const"] + features, gm.params.round(4).tolist())),
        }
    except Exception:
        pass

    results["_meta"] = {
        "n_train": len(ytr), "n_test": len(Xte),
        "target_mean": round(y.mean(), 2),
        "features": features,
        "sample": df.head(8).to_dict(orient="records")
    }
    return results


def train_fraud_detection(df, seed=42):
    features = ["amount", "hour", "velocity", "distance_km", "merchant_risk"]
    X = df[features].values
    y = df["is_fraud"].values
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)

    iso      = IsolationForest(n_estimators=100, contamination=0.05, random_state=seed)
    iso.fit(Xtr)
    iso_score = -iso.score_samples(Xte)
    iso_pred  = (iso.predict(Xte) == -1).astype(int)

    dt  = DecisionTreeClassifier(max_depth=5, random_state=seed, class_weight="balanced")
    dt.fit(Xtr, ytr)
    dt_pred, dt_proba = dt.predict(Xte), dt.predict_proba(Xte)[:, 1]

    gb  = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed)
    gb.fit(Xtr, ytr)
    gb_pred, gb_proba = gb.predict(Xte), gb.predict_proba(Xte)[:, 1]

    fpr_dt, tpr_dt, _ = roc_curve(yte, dt_proba)
    fpr_gb, tpr_gb, _ = roc_curve(yte, gb_proba)
    sc_idx = np.random.choice(len(Xte), min(400, len(Xte)), replace=False)

    return {
        "isolation_forest": {
            "accuracy": round(accuracy_score(yte, iso_pred), 4),
            "auc":      round(roc_auc_score(yte, iso_score), 4),
            "cm":       confusion_matrix(yte, iso_pred).tolist(),
        },
        "decision_tree": {
            "accuracy": round(accuracy_score(yte, dt_pred), 4),
            "auc":      round(roc_auc_score(yte, dt_proba), 4),
            "f1":       round(f1_score(yte, dt_pred), 4),
            "fpr": fpr_dt.tolist(), "tpr": tpr_dt.tolist(),
            "cm": confusion_matrix(yte, dt_pred).tolist(),
            "feat_imp": get_feat_imp(dt, features),
            "tree_rules": export_text(dt, feature_names=features, max_depth=3),
        },
        "gradient_boosting": {
            "accuracy": round(accuracy_score(yte, gb_pred), 4),
            "auc":      round(roc_auc_score(yte, gb_proba), 4),
            "f1":       round(f1_score(yte, gb_pred), 4),
            "fpr": fpr_gb.tolist(), "tpr": tpr_gb.tolist(),
            "cm": confusion_matrix(yte, gb_pred).tolist(),
            "feat_imp": get_feat_imp(gb, features),
        },
        "scatter": {
            "amount":   Xte[sc_idx, 0].tolist(),
            "velocity": Xte[sc_idx, 2].tolist(),
            "label":    yte[sc_idx].tolist(),
        },
        "_meta": {
            "fraud_rate": round(y.mean(), 4),
            "n_fraud_test": int(yte.sum()),
            "n_test": len(yte),
            "features": features,
            "sample": df.head(8).to_dict(orient="records")
        }
    }


# ──────────────────────────────────────────────────────────────
# STARTUP — GENERATE DATA & TRAIN ALL MODELS
# ──────────────────────────────────────────────────────────────

print("⚙️  Generating synthetic datasets …")
DF_CREDIT    = gen_credit_risk()
DF_INSURANCE = gen_insurance()
DF_CHURN     = gen_churn()
DF_FRAUD     = gen_fraud()

CR_FEATURES  = ["age","income","loan_amount","credit_score","employment_years",
                 "debt_to_income","num_accounts","late_payments","ltv"]
INS_FEATURES = ["age","vehicle_age","annual_mileage","accident_history",
                 "vehicle_value","credit_score","urban","coverage_type"]
CHN_FEATURES = ["tenure_months","num_products","balance","is_active",
                 "credit_score","age","salary","complaints","satisfaction"]

print("🤖  Training credit-risk models …")
CR_RESULTS  = train_classification_suite(DF_CREDIT,   CR_FEATURES,  "default")
print("🤖  Training insurance pricing models …")
INS_RESULTS = train_regression_suite(DF_INSURANCE, INS_FEATURES, "premium")
print("🤖  Training churn models …")
CHN_RESULTS = train_classification_suite(DF_CHURN,    CHN_FEATURES, "churned")
print("🤖  Training fraud detection models …")
FRD_RESULTS = train_fraud_detection(DF_FRAUD)
print("✅  All models ready.\n")


# ──────────────────────────────────────────────────────────────
# HELPER
# ──────────────────────────────────────────────────────────────

def _j(obj):
    return json.loads(json.dumps(obj, cls=PlotlyJSONEncoder))


def _prep_dist(df):
    return {col: df[col].head(500).tolist() for col in df.columns}


def _churn_bands():
    df = DF_CHURN.copy()
    df["sat_bin"] = pd.cut(df["satisfaction"], bins=5).astype(str)
    df["ten_bin"] = pd.cut(df["tenure_months"], bins=[0,6,12,24,48,120],
                           labels=["0-6","6-12","12-24","24-48","48+"])
    sat = df.groupby("sat_bin")["churned"].mean().reset_index()
    ten = df.groupby("ten_bin", observed=False)["churned"].mean().reset_index()
    return (sat["sat_bin"].tolist(), sat["churned"].round(4).tolist(),
            ten["ten_bin"].astype(str).tolist(), ten["churned"].round(4).tolist())


chn_sat_x, chn_sat_y, chn_ten_x, chn_ten_y = _churn_bands()


# ──────────────────────────────────────────────────────────────
# LOAD HTML TEMPLATE
# ──────────────────────────────────────────────────────────────

with open("templates/index.html", "r") as f:
    BASE_HTML = f.read()


# ──────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    from flask import render_template_string
    return render_template_string(
        BASE_HTML,
        cr_auc      = CR_RESULTS["Gradient Boosting"]["auc"],
        chn_auc     = CHN_RESULTS["Gradient Boosting"]["auc"],
        frd_auc     = FRD_RESULTS["gradient_boosting"]["auc"],
        ins_r2      = INS_RESULTS["Gradient Boosting"]["r2"],
        frd_rate    = f'{FRD_RESULTS["_meta"]["fraud_rate"]*100:.1f}%',
        frd_iso_auc = FRD_RESULTS["isolation_forest"]["auc"],
        frd_gb_auc  = FRD_RESULTS["gradient_boosting"]["auc"],
        frd_dt_auc  = FRD_RESULTS["decision_tree"]["auc"],
        cr_json     = json.dumps(_j(CR_RESULTS)),
        ins_json    = json.dumps(_j(INS_RESULTS)),
        chn_json    = json.dumps(_j(CHN_RESULTS)),
        frd_json    = json.dumps(_j(FRD_RESULTS)),
        cr_sample   = json.dumps(CR_RESULTS["_meta"]["sample"]),
        ins_sample  = json.dumps(INS_RESULTS["_meta"]["sample"]),
        chn_sample  = json.dumps(CHN_RESULTS["_meta"]["sample"]),
        frd_sample  = json.dumps(FRD_RESULTS["_meta"]["sample"]),
        cr_full_sample  = json.dumps(DF_CREDIT.head(20).to_dict(orient="records")),
        ins_full_sample = json.dumps(DF_INSURANCE.head(20).to_dict(orient="records")),
        chn_full_sample = json.dumps(DF_CHURN.head(20).to_dict(orient="records")),
        frd_full_sample = json.dumps(DF_FRAUD.head(20).to_dict(orient="records")),
        cr_dist     = json.dumps(_prep_dist(DF_CREDIT)),
        ins_dist    = json.dumps(_prep_dist(DF_INSURANCE)),
        chn_dist    = json.dumps(_prep_dist(DF_CHURN)),
        frd_dist    = json.dumps(_prep_dist(DF_FRAUD)),
        chn_sat_x   = json.dumps(chn_sat_x),
        chn_sat_y   = json.dumps(chn_sat_y),
        chn_ten_x   = json.dumps(chn_ten_x),
        chn_ten_y   = json.dumps(chn_ten_y),
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_loaded": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)