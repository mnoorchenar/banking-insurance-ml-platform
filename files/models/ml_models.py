"""
ML Models
Covers: Logistic Regression (GLM), Decision Tree, Random Forest (Bagging),
        Gradient Boosting (Boosting) — all trained on synthetic banking/insurance data.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
    confusion_matrix, roc_curve,
)

from models.data_generator import get_banking_data, get_insurance_data

warnings.filterwarnings('ignore')


# ── Feature preparation ──────────────────────────────────────────────────────

def _prepare_banking(df: pd.DataFrame):
    df = df.copy()
    for col in ['home_ownership', 'loan_purpose']:
        df[col] = LabelEncoder().fit_transform(df[col])
    X = df.drop('default', axis=1)
    y = df['default']
    return X, y, X.columns.tolist()


def _prepare_insurance(df: pd.DataFrame):
    df = df.copy()
    df['region'] = LabelEncoder().fit_transform(df['region'])
    X = df.drop('high_claim', axis=1)
    y = df['high_claim']
    return X, y, X.columns.tolist()


def _load(dataset: str):
    if dataset == 'banking':
        df = get_banking_data()
        return _prepare_banking(df)
    df = get_insurance_data()
    return _prepare_insurance(df)


def _split(X, y, test_size, random_state):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def _base_metrics(y_test, y_pred, y_prob) -> dict:
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    return {
        'metrics': {
            'accuracy':  round(float(accuracy_score(y_test, y_pred)), 4),
            'auc':       round(float(roc_auc_score(y_test, y_prob)), 4),
            'f1':        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            'recall':    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        },
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'roc': {
            'fpr': [round(v, 4) for v in fpr.tolist()],
            'tpr': [round(v, 4) for v in tpr.tolist()],
        },
    }


# ── GLM ──────────────────────────────────────────────────────────────────────

def train_glm(dataset='banking', test_size=0.2, C=1.0, penalty='l2', random_state=42) -> dict:
    X, y, feat_names = _load(dataset)
    X_tr, X_te, y_tr, y_te = _split(X, y, test_size, random_state)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=C, penalty=penalty, solver='lbfgs', max_iter=2000, random_state=random_state)),
    ])
    pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_te)
    y_prob = pipe.predict_proba(X_te)[:, 1]
    result = _base_metrics(y_te, y_pred, y_prob)

    coef = pipe.named_steps['clf'].coef_[0].tolist()
    result['coefficients'] = {
        'features': feat_names,
        'values':   [round(c, 5) for c in coef],
        'abs':      [round(abs(c), 5) for c in coef],
    }
    result['odds_ratios'] = {f: round(float(np.exp(c)), 4) for f, c in zip(feat_names, coef)}
    result['train_size']  = len(X_tr)
    result['test_size']   = len(X_te)
    result['target_rate'] = round(float(y_tr.mean()), 4)

    # Calibration curve buckets
    buckets    = pd.cut(y_prob, bins=10)
    cal_data   = pd.DataFrame({'bucket': buckets, 'actual': y_te.values, 'prob': y_prob})
    cal_grp    = cal_data.groupby('bucket', observed=True).agg(mean_prob=('prob','mean'), actual_rate=('actual','mean'), n=('actual','count')).reset_index()
    result['calibration'] = {
        'mean_prob':   cal_grp['mean_prob'].round(4).tolist(),
        'actual_rate': cal_grp['actual_rate'].round(4).tolist(),
        'n':           cal_grp['n'].tolist(),
    }
    return result


# ── Decision Tree ─────────────────────────────────────────────────────────────

def train_decision_tree(dataset='banking', test_size=0.2, max_depth=4,
                         min_samples_split=20, criterion='gini', random_state=42) -> dict:
    X, y, feat_names = _load(dataset)
    X_tr, X_te, y_tr, y_te = _split(X, y, test_size, random_state)

    clf = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_split=min_samples_split,
        criterion=criterion, random_state=random_state
    )
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]
    result = _base_metrics(y_te, y_pred, y_prob)

    result['feature_importance'] = {
        'features': feat_names,
        'values':   [round(v, 5) for v in clf.feature_importances_.tolist()],
    }

    # Depth complexity curve
    depths      = list(range(1, 13))
    tr_scores   = []
    te_scores   = []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, random_state=random_state)
        m.fit(X_tr, y_tr)
        tr_scores.append(round(m.score(X_tr, y_tr), 4))
        te_scores.append(round(m.score(X_te, y_te), 4))
    result['depth_curve'] = {'depths': depths, 'train': tr_scores, 'test': te_scores}

    cv = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    result['cv_auc']      = {'mean': round(float(cv.mean()), 4), 'std': round(float(cv.std()), 4)}
    result['tree_rules']  = export_text(clf, feature_names=feat_names, max_depth=4)
    result['n_leaves']    = clf.get_n_leaves()
    result['actual_depth']= clf.get_depth()
    result['train_size']  = len(X_tr)
    result['test_size']   = len(X_te)
    return result


# ── Random Forest (Bagging) ───────────────────────────────────────────────────

def train_random_forest(dataset='banking', test_size=0.2, n_estimators=100,
                         max_depth=None, max_features='sqrt', random_state=42) -> dict:
    X, y, feat_names = _load(dataset)
    X_tr, X_te, y_tr, y_te = _split(X, y, test_size, random_state)

    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        max_features=max_features, oob_score=True,
        random_state=random_state, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]
    result = _base_metrics(y_te, y_pred, y_prob)

    result['feature_importance'] = {
        'features': feat_names,
        'values':   [round(v, 5) for v in clf.feature_importances_.tolist()],
    }
    result['oob_score'] = round(clf.oob_score_, 4)

    # Learning curve vs n_estimators
    steps = list(range(5, n_estimators + 1, max(5, n_estimators // 15)))
    if steps[-1] != n_estimators:
        steps.append(n_estimators)
    oob_sc, te_sc = [], []
    for n in steps:
        m = RandomForestClassifier(
            n_estimators=n, max_depth=max_depth, max_features=max_features,
            oob_score=True, random_state=random_state, n_jobs=-1,
        )
        m.fit(X_tr, y_tr)
        oob_sc.append(round(m.oob_score_, 4))
        te_sc.append(round(m.score(X_te, y_te), 4))
    result['learning_curve'] = {'n_estimators': steps, 'oob': oob_sc, 'test': te_sc}

    # Single-tree baseline comparison
    st = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    st.fit(X_tr, y_tr)
    result['single_tree_acc'] = round(float(accuracy_score(y_te, st.predict(X_te))), 4)
    result['single_tree_auc'] = round(float(roc_auc_score(y_te, st.predict_proba(X_te)[:, 1])), 4)

    result['train_size'] = len(X_tr)
    result['test_size']  = len(X_te)
    return result


# ── Gradient Boosting ─────────────────────────────────────────────────────────

def train_gradient_boosting(dataset='banking', test_size=0.2, n_estimators=100,
                              learning_rate=0.1, max_depth=3, subsample=0.8,
                              random_state=42) -> dict:
    X, y, feat_names = _load(dataset)
    X_tr, X_te, y_tr, y_te = _split(X, y, test_size, random_state)

    clf = GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, subsample=subsample,
        random_state=random_state,
    )
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]
    result = _base_metrics(y_te, y_pred, y_prob)

    result['feature_importance'] = {
        'features': feat_names,
        'values':   [round(v, 5) for v in clf.feature_importances_.tolist()],
    }

    # Staged (boosting) learning curve — subsample every 5 steps
    all_tr_pred = list(clf.staged_predict(X_tr))
    all_te_pred = list(clf.staged_predict(X_te))
    indices   = [i for i in range(n_estimators) if i % 5 == 0 or i == n_estimators - 1]
    tr_staged = [round(float(accuracy_score(y_tr, all_tr_pred[i])), 4) for i in indices]
    te_staged = [round(float(accuracy_score(y_te, all_te_pred[i])), 4) for i in indices]
    st_range  = [i + 1 for i in indices]
    result['staged_scores'] = {'stages': st_range, 'train': tr_staged, 'test': te_staged}

    result['train_size'] = len(X_tr)
    result['test_size']  = len(X_te)
    return result


# ── Model comparison ──────────────────────────────────────────────────────────

def compare_all_models(dataset='banking', test_size=0.2, random_state=42) -> dict:
    """Train all four model families and return a unified comparison object."""
    X, y, feat_names = _load(dataset)
    X_tr, X_te, y_tr, y_te = _split(X, y, test_size, random_state)

    configs = {
        'GLM': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, random_state=random_state)),
        ]),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
    }

    results = {}
    roc_data = {}
    for name, model in configs.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        results[name] = {
            'accuracy':  round(float(accuracy_score(y_te, y_pred)), 4),
            'auc':       round(float(roc_auc_score(y_te, y_prob)), 4),
            'f1':        round(float(f1_score(y_te, y_pred, zero_division=0)), 4),
            'precision': round(float(precision_score(y_te, y_pred, zero_division=0)), 4),
            'recall':    round(float(recall_score(y_te, y_pred, zero_division=0)), 4),
        }
        roc_data[name] = {
            'fpr': [round(v, 4) for v in fpr.tolist()],
            'tpr': [round(v, 4) for v in tpr.tolist()],
        }

    return {
        'metrics': results,
        'roc_data': roc_data,
        'model_names': list(configs.keys()),
        'train_size': len(X_tr),
        'test_size': len(X_te),
        'target_rate': round(float(y_te.mean()), 4),
    }
