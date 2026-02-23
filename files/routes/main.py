from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    return render_template('index.html')


@main_bp.route('/data-explorer')
def data_explorer():
    return render_template('data_explorer.html')


@main_bp.route('/glm')
def glm():
    return render_template('glm.html')


@main_bp.route('/decision-tree')
def decision_tree():
    return render_template('decision_tree.html')


@main_bp.route('/random-forest')
def random_forest():
    return render_template('random_forest.html')


@main_bp.route('/gradient-boosting')
def gradient_boosting():
    return render_template('gradient_boosting.html')


@main_bp.route('/model-comparison')
def model_comparison():
    return render_template('model_comparison.html')


@main_bp.route('/stakeholder')
def stakeholder():
    return render_template('stakeholder.html')
