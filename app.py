from flask import Flask
from routes.main import main_bp
from routes.api import api_bp

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'banking-insurance-ml-2024'
    app.config['JSON_SORT_KEYS'] = False
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=7860, debug=False)
