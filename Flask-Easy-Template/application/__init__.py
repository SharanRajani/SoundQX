from flask import Flask
import logging
from flask_cache import Cache
from flask_sqlalchemy import SQLAlchemy

UPLOAD_FOLDER='/home/thegodfather/Desktop/A2IoT/for_interns/Test/'

app = Flask('application')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



# app.config.from_object('config.ProductionConfig')
app.config.from_object('config.DevelopmentConfig')

db = SQLAlchemy(app)

cache = Cache(app, config={'CACHE_TYPE': 'simple'})
cache.init_app(app)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from application import views
