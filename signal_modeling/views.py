from django.views.generic.edit import FormView
from django.views.generic.base import TemplateView

class MainPage(TemplateView):
    template_name = 'main.html'