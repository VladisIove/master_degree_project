import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'signal_modeling.settings')

application = get_wsgi_application()
