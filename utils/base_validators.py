import os
from django.core.exceptions import ValidationError

def validate_file_extension(value):
    ext = os.path.splitext(value.name)[1]  # [0] returns path+filename
    valid_extensions = ['.csv', '.txt', '.xlsx', '.xls']
    if not ext.lower() in valid_extensions:
        raise ValidationError('Файд з даним розширенням не підтримується. Підтримуються csv, txt, xlsx, xls розширення')