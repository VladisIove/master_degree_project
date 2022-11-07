
from utils.custom_range import crange
import numpy as np
from pandas import DataFrame
from utils.base_forms import AnalyticBaseForm 

from utils.parser_files import convertor_file_to_df
from django.forms import FileField, FloatField, IntegerField, ChoiceField, BooleanField

from utils.base_validators import validate_file_extension

class TimeAnalyticForm(AnalyticBaseForm):
    
    class SignalType:
        DETERMINATION = 'determination'
        STOCHASTIC = 'stochastic'

    SIGNAL_TYPE = (
        (SignalType.DETERMINATION, 'детермінований'),
        (SignalType.STOCHASTIC, 'стохастичний (випадковий)')
    ) 
    
    signal_type = ChoiceField(choices=SIGNAL_TYPE, label='Тип сигналу', required=False)
    input_data_file = FileField(label='Файл з сирими даними', required=False, allow_empty_file=False, validators=[validate_file_extension])
    
    def calculation_data(self, df: DataFrame) -> dict:
        signal_type_calculation = {
            self.SignalType.DETERMINATION: self._determination_data,
            self.SignalType.STOCHASTIC: self._stochastic_data
        }
        analytics_data = signal_type_calculation[self.cleaned_data['signal_type']](df)
        graphs_data = self._get_graphs_data(df.copy())
        return {
            'analytics_data': analytics_data,
            'graphs_data': graphs_data
        }
        
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
    
class CustomAnalyitcForm(AnalyticBaseForm):
    class SignalType:
        COS = 'cos'
        SIN = 'sin'

    SIGNAL_TYPE = (
        (SignalType.COS, 'cos'),
        (SignalType.SIN, 'sin')
    ) 
    
    type_of_signal = ChoiceField(required=False, choices=SIGNAL_TYPE)
    mean = FloatField(label='Середне значення', required=True)
    scope = FloatField(label='Розмах', required=True)
    checker_count_of_dot_or_period_sampling = BooleanField(label='Обраховувати через кількість періодів')
    count_of_periods = IntegerField(label='Кількість періодів', required=True)
    count_of_dots = IntegerField(label='Кількість точок', required=True)
    period_sampling = FloatField(label='Період дискретизації', required=True)
    frequency_sampling = FloatField(label='Частота дискретизації', required=True)
    frequency = IntegerField(label='Частота', required=True)
    
    def calculation_data(self, df: DataFrame) -> dict:
        analytics_data = self._determination_data(df)
        graphs_data = self._get_graphs_data(df.copy())
        return {
            'analytics_data': analytics_data,
            'graphs_data': graphs_data
        }
        
    def get_dataframe(self) -> DataFrame:
        f = self.cleaned_data['frequency']
        Td = self.cleaned_data['period_sampling']
        
        type_of_signal = self.cleaned_data['type_of_signal']
        rozmah = self.cleaned_data['scope']
        mean = self.cleaned_data['mean']
        p = self.cleaned_data['count_of_periods']
        
        T = 1/f
        t = crange(Td,p*T,Td)
        z = 2*np.pi*t*f
        zz = rozmah*getattr(np, type_of_signal)(z)
        y = float(mean)+ zz
        
        return DataFrame({'t': t.tolist(), 'y': y.tolist()})
    
    def is_valid(self) -> bool:
        is_valid = super().is_valid()
        frequency = self.cleaned_data['frequency']
        frequency_sampling = self.cleaned_data['frequency_sampling']
        period_sampling = self.cleaned_data['period_sampling']
        
        if not ( frequency * 2 <= frequency_sampling) :
            self.errors['frequency'] = ['Частота та частота дискретизації не співпадають за теоремою Найквіста Коперніка', ]
            return False
        
        if not ( round(1/frequency_sampling, 10) == round(period_sampling, 10) ) :
            self.errors['period_sampling'] = ['Період дискретизації та частота дискретизації взаємопов\'язані за формулою період дискретизації = 1/частота дискретизації', ]
            self.errors['frequency_sampling'] = ['Період дискретизації та частота дискретизації взаємопов\'язані за формулою період дискретизації = 1/частота дискретизації', ]
            return False
        
        return is_valid

