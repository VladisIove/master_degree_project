
import json
from django.http import HttpResponse
from utils.base_forms import DownloadAnalyticFilesForm
from utils.parser_files import convertor_df_to_file
from pandas import DataFrame

class ExportFileViewMixin:
    export_form = DownloadAnalyticFilesForm
    
    CONTENT_TYPE = {
        'txt': 'text/txt; charset=UTF-8',
        'csv': 'text/csv',
        'xls': 'application/ms-excel',
        'xlsx': 'application/ms-excel',
    }
    
    def export_file(self, file_type, context_data):
        df = DataFrame(json.loads(context_data) or None)
        return convertor_df_to_file.convert(df, file_type)
    
    def post(self, request, *args, **kwargs):
        """
        Handle POST requests: instantiate a form instance with the passed
        POST variables and then check if it's valid.
        """
        if request.GET.get('export'):
            file_type = request.POST['file_type']
            io_file = self.export_file(file_type, request.POST['context_data_field'])
            response = HttpResponse(io_file ,content_type=self.CONTENT_TYPE[file_type])
            response['Content-Disposition'] = f'attachment; filename=export_file.{file_type}'
            return response
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form)
        else:
            return self.form_invalid(form)