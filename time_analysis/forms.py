from cmath import inf
from email.policy import default
import numpy as np

from pandas import DataFrame
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import hilbert, periodogram
from utils.base_forms import AnalyticBaseForm 

from django.forms.fields import ChoiceField

class TimeAnalyticForm(AnalyticBaseForm):
    
    class SignalType:
        DETERMINATION = 'determination'
        STOCHASTIC = 'stochastic'
    
    SIGNAL_TYPE = (
        (SignalType.DETERMINATION, 'детермінований'),
        (SignalType.STOCHASTIC, 'стохастичний (випадковий)')
    ) 
    
    signal_type = ChoiceField(choices=SIGNAL_TYPE, label='Тип сигналу', required=False)
    
    def calculation_data(self, df: DataFrame) -> dict:
        signal_type_calculation = {
            self.SignalType.DETERMINATION: self._determination_data,
            self.SignalType.STOCHASTIC: self._stochastic_data
        }
        analytics_data = signal_type_calculation[self.cleaned_data['signal_type']](df)
        graphs_data = self._get_graphs_data(df)
        return {
            'analytics_data': analytics_data,
            'graphs_data': graphs_data
        }
        
    def _get_graphs_data(self, df: DataFrame) -> dict:
        kilkist_vidlikiv = self._get_kilkist_vidlikiv(df)
        # period_descritiatcii = self._get_period_descritiatcii(df)
        chastota_descritiatcii = self._get_chastota_descritiatcii(df)
        
        fft_data = self._get_fft_data(df)
        periodogram_data = self._get_periodogram_data(df)
        triangle_periodogram_data = self._get_triangle_periodogram_data(df)
        hann_periodogram_data = self._get_hann_periodogram_data(df)
        return {
            # 'period_descritiatcii': period_descritiatcii,
            'kilkist_vidlikiv': kilkist_vidlikiv,
            'chastota_descritiatcii': chastota_descritiatcii,
            
            'fft': fft_data,
            'periodogram': periodogram_data,
            'triangle_periodogram': triangle_periodogram_data,
            'hann_periodogram': hann_periodogram_data
        }
    
    def _stochastic_data(self, df: DataFrame) -> dict:
        data = {}
        data['min'] = self._get_min(df) 
        data['max'] = self._get_max(df) 
        data['median'] = self._get_median(df)
        data['mean'] = self._get_mean(df)
        data['quantile'] = self._get_quantile(df)
        data['dispersion'] = self._get_dispersion(df)
        data['std'] = self._get_std(df)
        data['mathematical_expectation'] = self._get_mathematical_expectation(df)
        df = self._get_amplitude_modulation(df)
        return data
     
    
    def _determination_data(self, df:DataFrame) -> dict:
        data = {}
        data['min'] = self._get_min(df) 
        data['max'] = self._get_max(df) 
        data['median'] = self._get_median(df)
        data['mean'] = self._get_mean(df)
        data['quantile'] = self._get_quantile(df)
        df = self._get_amplitude_modulation(df)
        return data
    
    @staticmethod
    def _get_min(df: DataFrame) -> dict:
        min = df.min()
        return {
            'label': 'Мінімальне значення',
            'value': min.to_dict()
        } 
        
    @staticmethod
    def _get_max(df: DataFrame) -> dict:
        max = df.max()
        return {
            'label': 'Максимальне значення',
            'value': max.to_dict()
        }
        
    @staticmethod
    def _get_median(df: DataFrame) -> dict:
        median = df.median()
        return {
            'label': 'Медіана значення',
            'value': median.to_dict()
        } 
        
    @staticmethod
    def _get_mean(df: DataFrame) -> dict:
        mean = df.mean()
        return {
            'label': 'Cередне значення',
            'value': mean.to_dict()
        }
        
    @staticmethod
    def _get_quantile(df: DataFrame) -> dict:
        quantile = df.quantile(0.5)
        return {
            'label': 'Розмах',
            'value': quantile.to_dict()
        }
        
    @staticmethod
    def _get_dispersion(df: DataFrame) -> dict:
        dispersion = df.var()
        return {
            'label': 'Дисперсія',
            'value': dispersion.to_dict()
        }
        
    @staticmethod
    def _get_std(df: DataFrame) -> dict:
        std = df.std()
        return {
            'label': 'Середньоквадратичне відхилення',
            'value': std.to_dict()
        }
    
    @staticmethod
    def _get_mathematical_expectation(df: DataFrame) -> dict:
        headers = df.columns.tolist()
        val1 = (df[headers[0]] * df[headers[1]]).sum() / df[headers[1]].sum()
        val2 = (df[headers[1]] * df[headers[0]]).sum() / df[headers[0]].sum()
        
        return {
            'label': 'Математичне сподівання',
            'value': {headers[0]:val1, headers[1]: val2}  
        }
        
    @staticmethod
    def _get_amplitude_modulation(df: DataFrame) -> dict:
        headers = df.columns.tolist()
        analytic_signal = np.abs(hilbert(df[headers[1]]))
        df['ampl'] = analytic_signal
        for column_name in df.columns.tolist():
            df[column_name] = df[column_name].round(2)
        return df 
    
    @staticmethod
    def _get_kilkist_vidlikiv(df: DataFrame) -> float:
        X_header_name = df.columns.tolist()[0]
        return len(df[X_header_name].to_list())
    
    def _get_chastota_descritiatcii(self, df: DataFrame) -> float:
        return list(fftfreq(self._get_kilkist_vidlikiv(df)))

    def _get_period_descritiatcii(self, df: DataFrame) -> float:
        freq = self._get_chastota_descritiatcii(df)
        return [1/f if 1/f != np.inf else 0  for f in freq] 
    
    @staticmethod
    def _get_period(df: DataFrame) -> int:
        pass
    
    def _get_fft_data(self, df: DataFrame) -> dict:
        '''
            chastota discritizatcii
        '''
        Y_header_name = df.columns.tolist()[1]
        y = df[Y_header_name].to_list()
        yf = np.abs(fft(y))
        xf = self._get_chastota_descritiatcii(df)
        return DataFrame({'y': list(yf), 'x': list(xf)}).to_dict('records')
    
    def _get_periodogram_data(self, df: DataFrame) -> dict:
        Y_header_name = df.columns.tolist()[1]
        y = df[Y_header_name].to_list()
        x, y = periodogram(y, 1)
        return DataFrame({'y': list(y), 'x': list(x)}).to_dict('list')
    
    def _get_triangle_periodogram_data(self, df: DataFrame) -> dict:
        Y_header_name = df.columns.tolist()[1]
        y = df[Y_header_name].to_list()
        x, y = periodogram(y, 1, window='triang')
        return DataFrame({'y': list(y), 'x': list(x)}).to_dict('list')
    
    def _get_hann_periodogram_data(self, df: DataFrame) -> dict:
        Y_header_name = df.columns.tolist()[1]
        y = df[Y_header_name].to_list()
        x, y = periodogram(y, 1, window='hann')
        return DataFrame({'y': list(y), 'x': list(x)}).to_dict('list')
        