from django.contrib import admin
from django.urls import path, include

from signal_modeling.views import MainPage

urlpatterns = [
    path('', MainPage.as_view(), name='main-page'),
    path('', include('time_analysis.urls')),
]
