from functools import cached_property
import math
from utils.custom_range import crange
import numpy as np

from pandas import DataFrame
from scipy.fft import fft, fftshift
from scipy.signal import hilbert, periodogram
from scipy.signal.windows import (
    hann, taylor, boxcar, triang, blackman, hamming, bartlett, flattop, 
    parzen, bohman, blackmanharris, nuttall, barthann, cosine,
    exponential, tukey
)

from django.forms.fields import ChoiceField
from django.forms import Form, ChoiceField, HiddenInput, JSONField
from pandas import DataFrame

class AnalyticBaseForm(Form):
        
    def _get_graphs_data(self, df: DataFrame) -> dict:
        kilkist_vidlikiv = self._get_kilkist_vidlikiv(df)
        
        period_descritiatcii = self._get_period_descritiatcii(df)
        chastota_descritiatcii = self._get_chastota_descritiatcii(df)
        
        fft_data = self._get_fft_data(df)
        periodograms_without_signal = self._get_periodograms_without_signal(df)
        periodograms_without_signal_frequency = self._get_periodograms_without_signal_frequency(periodograms_without_signal)
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
            'taylor_periodogram': self._get_taylor_periodogram_data(df),
            
            'periodograms_without_signal': periodograms_without_signal,
            'periodograms_without_signal_frequency': periodograms_without_signal_frequency,
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
        # df = self._get_amplitude_modulation(df)
        return data
     
    
    def _determination_data(self, df:DataFrame) -> dict:
        data = {}
        data['min'] = self._get_min(df) 
        data['max'] = self._get_max(df) 
        data['median'] = self._get_median(df)
        data['mean'] = self._get_mean(df)
        data['quantile'] = self._get_quantile(df)
        # df = self._get_amplitude_modulation(df)
        return data
    
    @staticmethod
    def _get_min(df: DataFrame) -> dict:
        min = df.min().round(3)
        _min = min.to_dict()
        k0 = list(_min.keys())[0]
        k1 = list(_min.keys())[1]
        return {
            'label': 'Мінімальне значення',
            'value':{'x_min': round(_min[k0], 3),  'y_min': round(_min[k1], 3)}
        } 
        
    @staticmethod
    def _get_max(df: DataFrame) -> dict:
        max = df.max().round(3)
        _max = max.to_dict()
        k0 = list(_max.keys())[0]
        k1 = list(_max.keys())[1]
        return {
            'label': 'Максимальне значення',
            'value': {'x_max': round(_max[k0], 3),  'y_max': round(_max[k1], 3)}
        }
        
    @staticmethod
    def _get_median(df: DataFrame) -> dict:
        median = df.median().round(3)
        _median = median.to_dict()
        k1 = list(_median.keys())[1]
        return {
            'label': 'Медіана значення',
            'value': {'x_median': '-',  'y_median': round(_median[k1], 3)}
        } 
        
    @staticmethod
    def _get_mean(df: DataFrame) -> dict:
        mean = df.mean()
        _mean =mean.to_dict()
        k1 = list(_mean.keys())[1]
        return {
            'label': 'Cередне значення',
            'value': {'x_mean': '-',  'y_mean': round(_mean[k1], 3)}
        }
        
    @staticmethod
    def _get_quantile(df: DataFrame) -> dict:
        headers = df.columns.tolist()
        # x_rozmah = df[headers[0]].max() - math.fabs(df[headers[0]].min())
        y_rozmah = df[headers[1]].max() - math.fabs(df[headers[1]].min())
        
        return {
            'label': 'Розмах',
            'value': {'x_quantile':'-', 'y_quantile': round(y_rozmah, 3)} 
        }
        
    @staticmethod
    def _get_dispersion(df: DataFrame) -> dict:
        dispersion = df.var()
        _dispersion = dispersion.to_dict()
        # k0 = list(_dispersion.keys())[0]
        k1 = list(_dispersion.keys())[1]
        return {
            'label': 'Дисперсія',
            'value': {'x_dispersion': '-',  'y_dispersion': round(_dispersion[k1], 3)}
        }
        
    @staticmethod
    def _get_std(df: DataFrame) -> dict:
        std = df.std()
        _std = std.to_dict()
        # k0 = list(_std.keys())[0]
        k1 = list(_std.keys())[1]
        return {
            'label': 'Середньоквадратичне відхилення',
            'value': {'x_std': '-',  'y_std': round(_std[k1], 3)}
        }
    
    @staticmethod
    def _get_mathematical_expectation(df: DataFrame) -> dict:
        headers = df.columns.tolist()
        val2 = (df[headers[1]] * df[headers[0]]).sum() / df[headers[0]].sum()
        
        return {
            'label': 'Математичне сподівання',
            'value': {'x_mathematical_expectation':'-', 'y_mathematical_expectation': round(val2, 3)}  
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
        return math.fabs(x[1] - x[0])

    
    def _get_fft_data(self, df: DataFrame) -> dict:
        '''
            chastota discritizatcii
        '''
        Y_header_name = df.columns.tolist()[1]
        y = df[Y_header_name].to_list()
        fd = self._get_chastota_descritiatcii(df)
        N = len(y)
        yf = fftshift(np.abs(fft(y)))/N
        start = -fd/2
        end = fd/2
        step = fd/N
        xf = np.arange(start, end, step)
        
        return DataFrame({'y': list(yf), 'x': list(xf.round(0))}).to_dict('list')
    
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
    
    def _get_periodograms_without_signal(self, df: DataFrame) -> dict:
        Y_header_name = df.columns.tolist()[1]
        y = df[Y_header_name].to_list()
        N = len(y)
        return {
            'periodogram': list(boxcar(N)),
            'triangle_periodogram': list(triang(N)),
            'hann_periodogram': list(hann(N)),
            'blackman_periodogram': list(blackman(N)),
            'hamming_periodogram': list(hamming(N)),
            'bartlett_periodogram': list(bartlett(N)),
            'flattop_periodogram': list(flattop(N)),
            'parzen_periodogram': list(parzen(N)),
            'bohman_periodogram': list(bohman(N)),
            'blackmanharris_periodogram': list(blackmanharris(N)),
            'nuttall_periodogram': list(nuttall(N)),
            'barthann_periodogram': list(barthann(N)),
            'cosine_periodogram': list(cosine(N)),
            'exponential_periodogram': list(exponential(N)),
            'tukey_periodogram': list(tukey(N)),
            'taylor_periodogram': list(taylor(N)),
        }
    
    def _get_periodograms_without_signal_frequency(self, periodograms_without_signal: dict) -> dict: 
        
        return {
            'periodogram': self._get_periodogram_boxcar_without_signal(periodograms_without_signal['periodogram']),
            'triangle_periodogram': self._get_periodogram_triangl_without_signal(periodograms_without_signal['triangle_periodogram']),
            'hann_periodogram': self._get_periodogram_hann_without_signal(periodograms_without_signal['hann_periodogram']),
            'blackman_periodogram': self._get_periodogram_blackman_without_signal(periodograms_without_signal['blackman_periodogram']),
            'hamming_periodogram': self._get_periodogram_hamming_without_signal(periodograms_without_signal['hamming_periodogram']),
            'bartlett_periodogram': self._get_periodogram_bartlett_without_signal(periodograms_without_signal['bartlett_periodogram']),
            'flattop_periodogram': self._get_periodogram_flattop_without_signal(periodograms_without_signal['flattop_periodogram']),
            'parzen_periodogram': self._get_periodogram_parzen_without_signal(periodograms_without_signal['parzen_periodogram']),
            'bohman_periodogram': self._get_periodogram_bohman_without_signal(periodograms_without_signal['bohman_periodogram']),
            'blackmanharris_periodogram': self._get_periodogram_blackmanharris_without_signal(periodograms_without_signal['blackmanharris_periodogram']),
            'nuttall_periodogram': self._get_periodogram_blackmanharris_without_signal(periodograms_without_signal['nuttall_periodogram']),
            'barthann_periodogram': self._get_periodogram_blackmanharris_without_signal(periodograms_without_signal['barthann_periodogram']),
            'cosine_periodogram': self._get_periodogram_blackmanharris_without_signal(periodograms_without_signal['cosine_periodogram']),
            'exponential_periodogram': self._get_periodogram_blackmanharris_without_signal(periodograms_without_signal['exponential_periodogram']),
            'tukey_periodogram': self._get_periodogram_blackmanharris_without_signal(periodograms_without_signal['tukey_periodogram']),
            'taylor_periodogram': self._get_periodogram_blackmanharris_without_signal(periodograms_without_signal['taylor_periodogram']),
        }
        

    def _get_periodogram_boxcar_without_signal(self, window: np.arange) -> dict:
        A = fft(window, 512) / (len(window)/2.0)
        freq = np.linspace(-0.5, 0.5, len(A))
        response = np.abs(fftshift(A / abs(A).max()))
        response = 20 * np.log10(np.maximum(response, 1e-10))
        return {
            'freq': list(freq),
            'response': list(response)
        }
    
    def _get_periodogram_triangl_without_signal(self, window: np.arange) -> dict:
        A = fft(window, 512) / (len(window)/2.0)
        freq = np.linspace(-0.5, 0.5, len(A))
        response = np.abs(fftshift(A / abs(A).max()))
        response = 20 * np.log10(np.maximum(response, 1e-10))
        return {
            'freq': list(freq),
            'response': list(response)
        }
    
    def _get_periodogram_hann_without_signal(self, window: np.arange) -> dict:
        A = fft(window, 512) / (len(window)/2.0)
        freq = np.linspace(-0.5, 0.5, len(A))
        response = np.abs(fftshift(A / abs(A).max()))
        response = 20 * np.log10(np.maximum(response, 1e-10))
        return {
            'freq': list(freq),
            'response': list(response)
        }
    
    def _get_periodogram_blackman_without_signal(self, window: np.arange) -> dict:
        A = fft(window, 512) / (len(window)/2.0)
        freq = np.linspace(-0.5, 0.5, len(A))
        response = np.abs(fftshift(A / abs(A).max()))
        response = 20 * np.log10(np.maximum(response, 1e-10))
        return {
            'freq': list(freq),
            'response': list(response)
        }
    
    def _get_periodogram_hamming_without_signal(self, window: np.arange) -> dict:
        A = fft(window, 512) / (len(window)/2.0)
        freq = np.linspace(-0.5, 0.5, len(A))
        response = np.abs(fftshift(A / abs(A).max()))
        response = 20 * np.log10(np.maximum(response, 1e-10))
        return {
            'freq': list(freq),
            'response': list(response)
        }
        
    def _get_periodogram_bartlett_without_signal(self, window: np.arange) -> dict:
        A = fft(window, 512) / (len(window)/2.0)
        freq = np.linspace(-0.5, 0.5, len(A))
        response = np.abs(fftshift(A / abs(A).max()))
        response = 20 * np.log10(np.maximum(response, 1e-10))
        return {
            'freq': list(freq),
            'response': list(response)
        }
    
    def _get_periodogram_flattop_without_signal(self, window: np.arange) -> dict:
        A = fft(window, 512) / (len(window)/2.0)
        freq = np.linspace(-0.5, 0.5, len(A))
        response = np.abs(fftshift(A / abs(A).max()))
        response = 20 * np.log10(np.maximum(response, 1e-10))
        return {
            'freq': list(freq),
            'response': list(response)
        }
    
    def _get_periodogram_parzen_without_signal(self, window: np.arange) -> dict:
        A = fft(window, 512) / (len(window)/2.0)
        freq = np.linspace(-0.5, 0.5, len(A))
        response = np.abs(fftshift(A / abs(A).max()))
        response = 20 * np.log10(np.maximum(response, 1e-10))
        return {
            'freq': list(freq),
            'response': list(response)
        }
    
    def _get_periodogram_bohman_without_signal(self, window: np.arange) -> dict:
        A = fft(window, 512) / (len(window)/2.0)
        freq = np.linspace(-0.5, 0.5, len(A))   
        response = np.abs(fftshift(A / abs(A).max()))
        response = 20 * np.log10(np.maximum(response, 1e-10))
        return {
            'freq': list(freq),
            'response': list(response)
        }
    
    def _get_periodogram_blackmanharris_without_signal(self, window: np.arange) -> dict:
        A = fft(window, 512) / (len(window)/2.0)
        freq = np.linspace(-0.5, 0.5, len(A))   
        response = np.abs(fftshift(A / abs(A).max()))
        response = 20 * np.log10(np.maximum(response, 1e-10))
        freq[np.isnan(freq)] = 0
        response[np.isnan(response)] = 0
        return {
            'freq': list(freq),
            'response': list(response)
        }
    

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