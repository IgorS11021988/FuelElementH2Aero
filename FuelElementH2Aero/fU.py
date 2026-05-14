import numpy as np


# Функция условий протекания процессов
def fU(t,  # Моменты времени
       UParametersSystemParameters  # U-параметры системы
       ):
    # Выделяем параметры динамики
    [OmegaVAlpha,  # Частота колебаний внешнего потока
     AvAlpha  # Амплитуда колебаний внешнего потока
     ] = UParametersSystemParameters  # Постоянная составляющая тока

    # Рассчитываем поток вещества
    vAlpha = np.abs(AvAlpha * OmegaVAlpha * np.sin(OmegaVAlpha * t))  # Учитываем колебания

    # Выводим результат
    return vAlpha
