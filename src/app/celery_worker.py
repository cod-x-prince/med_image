from med_image.app.app import create_app
from med_image.app.celery_utils import make_celery

app = create_app()
celery = make_celery(app)
