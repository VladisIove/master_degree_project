from django.urls import path

from time_analysis.views import TimeAnalysisView

urlpatterns = [
    path('time_analysis', TimeAnalysisView.as_view(), name='time-analysis')
]
