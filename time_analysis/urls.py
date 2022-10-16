from django.urls import path

from time_analysis.views import CustomAnalyticView, TimeAnalysisView

urlpatterns = [
    path('time_analysis', TimeAnalysisView.as_view(), name='time-analysis'),
    path('custom_time_analysis', CustomAnalyticView.as_view(), name='custom-time-analysis'),
]
