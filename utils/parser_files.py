from io import BytesIO
from typing import Any, Callable, Optional, IO
from pandas import DataFrame, ExcelWriter, read_excel, read_csv


class ConvertFileDataToDataFrame:
    '''
        This class represent fabric of methods
        of the parcing files which extensions
        txt, csv, xls, xlsx
    '''
    
    def convert(self, file: IO, filename:str, delimetr: Optional[str]=None) -> DataFrame:
        convertor = self._get_convert_method(filename)
        return convertor(file, delimetr)
        
    def _get_convert_method(self, filename: str) -> Callable:
        convertors = {
            'txt': self._txt_convertor, 
            'csv': self._csv_convertor,
            'xls': self._xls_convertor,
            'xlsx': self._xls_convertor
        }
        return convertors.get(filename.split('.')[1])
        
    def _xls_convertor(self, file: IO, delimetr: str) -> DataFrame:
        return read_excel(file)
    
    def _csv_convertor(self, file: IO, delimetr: str) -> DataFrame:
        return read_csv(file)

    def _txt_convertor(self, file: IO, delimetr: str) -> DataFrame:
        return read_csv(file, delimiter=delimetr)


class ConvertDataFrameToFileData:
    '''
        This class represent fabric of methods
        of the parcing dataframe to file which extensions
        txt, csv, xls, xlsx
    '''

    def _get_convert_method(self, type_to_convert: str) -> Callable:
        convertors = {
            'txt': self._txt_convertor, 
            'csv': self._csv_convertor,
            'xls': self._xls_convertor,
            'xlsx': self._xls_convertor
        }
        return convertors.get(type_to_convert)
    
    def _xls_convertor(self, df: DataFrame) -> IO:
        io = BytesIO()
        writer = ExcelWriter(io)
        df.to_excel(writer, 'Дані', index=False)
        writer.save()
        return io.getvalue()
    
    def _csv_convertor(self, df: DataFrame) -> IO:
        io = BytesIO()
        df.to_csv(path_or_buf=io)
        return io.getvalue()
        
    def _txt_convertor(self, df: DataFrame) -> IO:
        return self._csv_convertor(df) 
    
    def convert(self, df: DataFrame, type_to_convert: str) -> IO:
        convertor = self._get_convert_method(type_to_convert)
        return convertor(df)
        

convertor_file_to_df = ConvertFileDataToDataFrame()
convertor_df_to_file = ConvertDataFrameToFileData()