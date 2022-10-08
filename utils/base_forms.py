from utils.parser_files import convertor_file_to_df
from django.forms import FileField, Form, ChoiceField, HiddenInput, JSONField
from pandas import DataFrame

from utils.base_validators import validate_file_extension


class AnalyticBaseForm(Form):
    
    input_data_file = FileField(label='Файл з сирими даними', required=False, allow_empty_file=False, validators=[validate_file_extension])
    
    def is_valid(self) -> bool:
        is_valid = super().is_valid()
        if self.cleaned_data['input_data_file'] is None:
            self.errors['input_data_file'] = ['Будь ласка, оберіть файл', ]
            return False
        return is_valid
            
    def get_dataframe_from_file(self) -> DataFrame:
        file = self.cleaned_data['input_data_file'].file
        filename = self.cleaned_data['input_data_file'].name
        
        return convertor_file_to_df.convert(file, filename)
    

class DownloadAnalyticFilesForm(Form):
    
    class FileType:
        TXT = 'txt'
        CSV = 'csv'
        XLS = 'xls'
        XLSX = 'xlsx'
    
    FILE_TYPE_CHOICE = (
        (FileType.TXT, '.txt'),
        (FileType.CSV, '.csv'),
        (FileType.XLS, '.xls'),
        (FileType.XLSX, '.xlsx')
    )
    
    file_type = ChoiceField(choices=FILE_TYPE_CHOICE,required=True, label='Тип файлу')
    context_data_field = JSONField(widget = HiddenInput(), required = True)