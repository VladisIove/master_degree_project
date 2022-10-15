
import math
import numpy as np

from pandas import DataFrame
from scipy.fft import fft, fftshift
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
        graphs_data = self._get_graphs_data(df.copy())
        return {
            'analytics_data': analytics_data,
            'graphs_data': graphs_data
        }
        
    def _get_graphs_data(self, df: DataFrame) -> dict:
        kilkist_vidlikiv = self._get_kilkist_vidlikiv(df)
        period_descritiatcii = self._get_period_descritiatcii(df)
        chastota_descritiatcii = self._get_chastota_descritiatcii(df)
        
        fft_data = self._get_fft_data(df)

        return {
            'period_descritiatcii': period_descritiatcii,
            'kilkist_vidlikiv': kilkist_vidlikiv,
            'chastota_descritiatcii': chastota_descritiatcii,
            
            'fft': fft_data,
            'periodogram': self._get_periodogram_data(df),
            'triangle_periodogram': self._get_triangle_periodogram_data(df),
            'hann_periodogram': self._get_hann_periodogram_data(df),
            
            'blackman_periodogram': self._get_blackman_periodogram_data(df),
            'hamming_periodogram': self._get_hamming_periodogram_data(df),
            'bartlett_periodogram': self._get_bartlett_periodogram_data(df),
            'flattop_periodogram': self._get_flattop_periodogram_data(df),
            'parzen_periodogram': self._get_parzen_periodogram_data(df),
            'bohman_periodogram': self._get_bohman_periodogram_data(df),
            'blackmanharris_periodogram': self._get_blackmanharris_periodogram_data(df),
            'nuttall_periodogram': self._get_nuttall_periodogram_data(df),
            'barthann_periodogram': self._get_barthann_periodogram_data(df),
            'cosine_periodogram': self._get_cosine_periodogram_data(df),
            'exponential_periodogram': self._get_exponential_periodogram_data(df),
            'tukey_periodogram': self._get_tukey_periodogram_data(df),
            'taylor_periodogram': self._get_taylor_periodogram_data(df)
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
        headers = df.columns.tolist()
        x_rozmah = df[headers[0]].max() + math.fabs(df[headers[0]].min())
        y_rozmah = df[headers[1]].max() + math.fabs(df[headers[1]].min())
        
        return {
            'label': 'Розмах',
            'value': {headers[0]:x_rozmah, headers[1]: y_rozmah} 
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
        return df
    
    @staticmethod
    def _get_kilkist_vidlikiv(df: DataFrame) -> float:
        X_header_name = df.columns.tolist()[0]
        return len(df[X_header_name].to_list())
    
    def _get_chastota_descritiatcii(self, df: DataFrame) -> float:
        return 1 / self._get_period_descritiatcii(df) 

    @staticmethod
    def _get_period_descritiatcii(df: DataFrame) -> float:
        X_header_name = df.columns.tolist()[0]
        x = df[X_header_name].to_list()
        for index, item in enumerate(x):
            if x[index+1] - item:  
                return math.fabs(item)
    
    @staticmethod
    def _get_period(df: DataFrame) -> int:
        pass
    
    def _get_fft_data(self, df: DataFrame) -> dict:
        '''
            chastota discritizatcii
        '''
        Y_header_name = df.columns.tolist()[1]
        y = df[Y_header_name].to_list()
        fd = self._get_chastota_descritiatcii(df)
        yf = fftshift(np.abs(fft(y)/len(y)))
        xf = np.arange(-fd/2, fd/2-fd/len(y), fd/len(y)) 
        if xf.shape[0] != yf.shape[0]:
            xf = np.arange(-fd/2, fd/2, fd/len(y)) 
        return DataFrame({'y': list(yf), 'x': list(xf.round(2))}).to_dict('list')
    
    def _get_periodogram_data_by_widnow(self, df: DataFrame, window: str) -> dict: 
        Y_header_name = df.columns.tolist()[1]
        y = df[Y_header_name].to_list()
        fd = self._get_chastota_descritiatcii(df)
        xp, yp = periodogram(y, fd, window)
        xp = xp.round(5)
        return DataFrame({'y': list(yp), 'x': list(xp)}).to_dict('list')
    
    def _get_periodogram_data(self, df: DataFrame) -> dict:
        return self._get_periodogram_data_by_widnow(df, 'boxcar')
    
    def _get_triangle_periodogram_data(self, df: DataFrame) -> dict:
        return self._get_periodogram_data_by_widnow(df, 'triang')
    
    def _get_hann_periodogram_data(self, df: DataFrame) -> dict:
        return self._get_periodogram_data_by_widnow(df, 'hann')
    
    def _get_blackman_periodogram_data(self, df: DataFrame) -> dict:
        return self._get_periodogram_data_by_widnow(df, 'blackman')
    
    def _get_hamming_periodogram_data(self, df: DataFrame) -> dict:
        return self._get_periodogram_data_by_widnow(df, 'hamming')
    
    def _get_bartlett_periodogram_data(self, df: DataFrame) -> dict:
        return self._get_periodogram_data_by_widnow(df, 'bartlett')
    
    def _get_flattop_periodogram_data(self, df: DataFrame) -> dict:
        return self._get_periodogram_data_by_widnow(df, 'flattop')
    
    def _get_parzen_periodogram_data(self, df: DataFrame) -> dict:
        return self._get_periodogram_data_by_widnow(df, 'parzen')
    
    def _get_bohman_periodogram_data(self, df: DataFrame) -> dict:
        return self._get_periodogram_data_by_widnow(df, 'bohman')
    
    def _get_blackmanharris_periodogram_data(self, df: DataFrame) -> dict:
        return self._get_periodogram_data_by_widnow(df, 'blackmanharris')
    
    def _get_nuttall_periodogram_data(self, df: DataFrame) -> dict:
        return self._get_periodogram_data_by_widnow(df, 'nuttall')
    
    def _get_barthann_periodogram_data(self, df: DataFrame) -> dict:
        return self._get_periodogram_data_by_widnow(df, 'barthann')
    
    def _get_cosine_periodogram_data(self, df: DataFrame) -> dict:
        return self._get_periodogram_data_by_widnow(df, 'cosine')
    
    def _get_exponential_periodogram_data(self, df: DataFrame) -> dict:
        return self._get_periodogram_data_by_widnow(df, 'exponential')
    
    def _get_tukey_periodogram_data(self, df: DataFrame) -> dict:
        return self._get_periodogram_data_by_widnow(df, 'tukey')
    
    def _get_taylor_periodogram_data(self, df: DataFrame) -> dict:
        return self._get_periodogram_data_by_widnow(df, 'taylor')
    
    
