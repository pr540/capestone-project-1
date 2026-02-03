import os
from app import app
with app.app_context():
    print("DB URI:", app.config['SQLALCHEMY_DATABASE_URI'])
    print("DB Path Exists:", os.path.exists(app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')))
