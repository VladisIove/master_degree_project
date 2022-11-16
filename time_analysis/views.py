from functools import wraps
import json
import sys
from time import time
from typing import Any, Dict

from django.http import JsonResponse
from utils.mixins import ExportFileViewMixin
from utils.base_forms import DownloadAnalyticFilesForm
from django.views.generic.edit import FormView
from pandas import DataFrame

from time_analysis.forms import CustomAnalyitcForm, TimeAnalyticForm

# Create your views here.

class TimeAnalysisView(ExportFileViewMixin, FormView):
    template_name = "time_analysis.html"
    form_class = TimeAnalyticForm
    
    def get_context_data(self, **kwargs: Any) -> Dict[str, Any]:
        context_data = super().get_context_data(**kwargs)
        # with open('result.json', 'w') as f:
        #     json.dump(kwargs['data'], f)
        if kwargs:
            context_data['export_file_form'] = DownloadAnalyticFilesForm(data={'context_data_field': kwargs.get('data', {}).get('calculated_data_json'), 'file_type': DownloadAnalyticFilesForm.FileType.TXT})
        else: 
            context_data['export_file_form'] = DownloadAnalyticFilesForm
        return context_data
    
    
    def form_valid(self, form):
        income_dataframe: DataFrame = form.get_dataframe_from_file()
        data = {}
        data['headers'] = income_dataframe.columns.tolist()
        data['calculated_data'] = form.calculation_data(income_dataframe)
        data['calculated_data_json'] = json.dumps(data['calculated_data'])
        income_dataframe = income_dataframe.round(5)
        data['data_for_graph'] = income_dataframe.to_dict('list')
        context = self.get_context_data(data=data, form=form)
        return self.render_to_response(context)


class CustomAnalyticView(ExportFileViewMixin, FormView):
    
    template_name = "time_analysis.html"
    form_class = CustomAnalyitcForm
    
    def default_data(self):
        
        with open('./time_analysis/result.json', 'r') as fcc_file:
            data = json.load(fcc_file)
        return {
            'data': data,
            'form': self.form_class(
                    data = dict(
                    type_of_signal = self.form_class.SignalType.SIN,
                    mean = 3,
                    scope = 4,
                    snr=1,
                    count_of_periods = 2,
                    frequency_sampling = 500,
                    period_sampling = 0.002,
                    frequency = 99,
                    count_of_dots=22,
                    checker_count_of_dot_or_period_sampling=True
                    )
            )
        }
    
    def get_context_data(self, **kwargs: Any) -> Dict[str, Any]:
        if not kwargs:
            kwargs = self.default_data()  
        context_data = super().get_context_data(**kwargs)
        if kwargs:
            context_data['export_file_form'] = DownloadAnalyticFilesForm(data={'context_data_field': kwargs.get('data', {}).get('calculated_data_json'), 'file_type': DownloadAnalyticFilesForm.FileType.TXT})
        else: 
            context_data['export_file_form'] = DownloadAnalyticFilesForm
        return context_data
    
    def form_valid(self, form):
        income_dataframe: DataFrame = form.get_dataframe()
        data = {}
        data['headers'] = income_dataframe.columns.tolist()
        data['calculated_data'] = form.calculation_data(income_dataframe)
        data['calculated_data_json'] = json.dumps(data['calculated_data'])
        income_dataframe = income_dataframe.round(5)
        data['data_for_graph'] = income_dataframe.to_dict('list')
        context = self.get_context_data(data=data, form=form)
        return JsonResponse(context['data'])