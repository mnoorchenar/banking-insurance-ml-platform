import json
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request

from models.data_generator import get_banking_data, get_insurance_data
from models.ml_models import (
    train_glm, train_decision_tree,
    train_random_forest, train_gradient_boosting,
    compare_all_models,
)

api_bp = Blueprint('api', __name__)


# ── helpers ──────────────────────────────────────────────────────────────────

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _ok(data):
    return json.loads(json.dumps(data, cls=_NpEncoder))


def _dataset_from_args():
    return request.args.get('dataset', 'banking')


# ── Data Explorer ─────────────────────────────────────────────────────────────

@api_bp.route('/data/overview')
def data_overview():
    ds = _dataset_from_args()
    df = get_banking_data() if ds == 'banking' else get_insurance_data()
    target_col = 'default' if ds == 'banking' else 'high_claim'

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols     = df.select_dtypes(exclude='number').columns.tolist()

    stats = df[numeric_cols].describe().round(3).to_dict()

    return jsonify(_ok({
        'n_rows':      len(df),
        'n_cols':      len(df.columns),
        'numeric_cols': numeric_cols,
        'cat_cols':     cat_cols,
        'target_col':   target_col,
        'target_rate':  round(float(df[target_col].mean()), 4),
        'stats':        stats,
        'columns':      df.columns.tolist(),
    }))


@api_bp.route('/data/distributions')
def data_distributions():
    ds = _dataset_from_args()
    df = get_banking_data() if ds == 'banking' else get_insurance_data()
    target_col = 'default' if ds == 'banking' else 'high_claim'

    numeric_cols = [c for c in df.select_dtypes(include='number').columns if c != target_col]
    dists = {}
    for col in numeric_cols:
        vals = df[col].tolist()
        dists[col] = {
            'values':    vals,
            'by_target': {
                str(t): df.loc[df[target_col] == t, col].tolist()
                for t in df[target_col].unique()
            },
        }
    return jsonify(_ok({'distributions': dists, 'target_col': target_col}))


@api_bp.route('/data/correlation')
def data_correlation():
    ds = _dataset_from_args()
    df = get_banking_data() if ds == 'banking' else get_insurance_data()
    for col in df.select_dtypes(exclude='number').columns:
        from sklearn.preprocessing import LabelEncoder
        df[col] = LabelEncoder().fit_transform(df[col])
    corr = df.corr().round(3)
    return jsonify(_ok({
        'cols':   corr.columns.tolist(),
        'matrix': corr.values.tolist(),
    }))


@api_bp.route('/data/sample')
def data_sample():
    ds = _dataset_from_args()
    df = get_banking_data() if ds == 'banking' else get_insurance_data()
    sample = df.head(50).fillna('').to_dict(orient='records')
    return jsonify(_ok({'records': sample, 'columns': df.columns.tolist()}))


@api_bp.route('/data/categorical')
def data_categorical():
    ds = _dataset_from_args()
    df = get_banking_data() if ds == 'banking' else get_insurance_data()
    target_col = 'default' if ds == 'banking' else 'high_claim'
    cat_cols   = df.select_dtypes(exclude='number').columns.tolist()
    result = {}
    for col in cat_cols:
        grp = df.groupby(col)[target_col].agg(['mean', 'count']).reset_index()
        result[col] = {
            'categories': grp[col].tolist(),
            'target_rate': grp['mean'].round(4).tolist(),
            'count':       grp['count'].tolist(),
        }
    return jsonify(_ok(result))


# ── Model training endpoints ──────────────────────────────────────────────────

@api_bp.route('/model/glm', methods=['POST'])
def model_glm():
    p = request.get_json(force=True) or {}
    result = train_glm(
        dataset=p.get('dataset', 'banking'),
        test_size=float(p.get('test_size', 0.2)),
        C=float(p.get('C', 1.0)),
        penalty=p.get('penalty', 'l2'),
    )
    return jsonify(_ok(result))


@api_bp.route('/model/decision-tree', methods=['POST'])
def model_decision_tree():
    p = request.get_json(force=True) or {}
    result = train_decision_tree(
        dataset=p.get('dataset', 'banking'),
        test_size=float(p.get('test_size', 0.2)),
        max_depth=int(p.get('max_depth', 4)),
        min_samples_split=int(p.get('min_samples_split', 20)),
        criterion=p.get('criterion', 'gini'),
    )
    return jsonify(_ok(result))


@api_bp.route('/model/random-forest', methods=['POST'])
def model_random_forest():
    p = request.get_json(force=True) or {}
    result = train_random_forest(
        dataset=p.get('dataset', 'banking'),
        test_size=float(p.get('test_size', 0.2)),
        n_estimators=int(p.get('n_estimators', 100)),
        max_depth=p.get('max_depth') or None,
        max_features=p.get('max_features', 'sqrt'),
    )
    return jsonify(_ok(result))


@api_bp.route('/model/gradient-boosting', methods=['POST'])
def model_gradient_boosting():
    p = request.get_json(force=True) or {}
    result = train_gradient_boosting(
        dataset=p.get('dataset', 'banking'),
        test_size=float(p.get('test_size', 0.2)),
        n_estimators=int(p.get('n_estimators', 100)),
        learning_rate=float(p.get('learning_rate', 0.1)),
        max_depth=int(p.get('max_depth', 3)),
        subsample=float(p.get('subsample', 0.8)),
    )
    return jsonify(_ok(result))


@api_bp.route('/model/compare', methods=['POST'])
def model_compare():
    p = request.get_json(force=True) or {}
    result = compare_all_models(
        dataset=p.get('dataset', 'banking'),
        test_size=float(p.get('test_size', 0.2)),
    )
    return jsonify(_ok(result))
