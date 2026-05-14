import numpy as np


# Функция условий протекания процессов
def fU(t,  # Моменты времени
       UParametersSystemParameters  # U-параметры системы
       ):
    # Получаем параметры токов
    [Ie] = UParametersSystemParameters

    # Получаем массив токов
    return np.full_like(t, Ie, dtype=np.double)
