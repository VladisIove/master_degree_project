import json
from typing import Any, Dict
from utils.mixins import ExportFileViewMixin
from utils.base_forms import DownloadAnalyticFilesForm
from django.views.generic.edit import FormView
from pandas import DataFrame

from time_analysis.forms import TimeAnalyticForm

# Create your views here.

class TimeAnalysisView(ExportFileViewMixin, FormView):
    template_name = "time_analysis.html"
    form_class = TimeAnalyticForm
    
    def get_context_data(self, **kwargs: Any) -> Dict[str, Any]:
        context_data = super().get_context_data(**kwargs)
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
