import numpy as np

from .StationFunctions import funCbin
from .fICharge import fICharge


# Функция состояния для литий-ионного аккумулятора
def CharacteristicsFunction(t,  # Моменты времени
                            stateCoordinates,  # Координаты состояния
                            reducedTemp,  # Приведенные температуры
                            systemParameters  # Параметры системы
                            ):
    # Получаем динамику тока
    (Icur, otherSystemParameters) = fICharge(np.array(t, dtype=np.double),  # Моменты времени
                                             systemParameters  # Параметры системы
                                             )
    Icur = np.array(Icur, dtype=np.double).reshape(-1)  # Приводим токи к одномерному массиву

    # Получаем координаты состояния
    qbinp = stateCoordinates[:, 0]  # Заряд на положительном двойном слое
    qm = stateCoordinates[:, 1]  # Заряд на мембране
    qbinn = stateCoordinates[:, 2]  # Заряд на отрицательном двойном слое
    nuH2Op = stateCoordinates[:, 3]  # Число молей воды в приэлектродной области положительного электрода
    nuH2On = stateCoordinates[:, 4]  # Число молей воды в приэлектродной области отрицательного электрода
    nuH2OStp = stateCoordinates[:, 5]  # Приведенное число молей воды в камере положительного электрода
    nuH2OStn = stateCoordinates[:, 6]  # Приведенное число молей воды в камере отрицательного электрода
    nuO2 = stateCoordinates[:, 7]  # Число молей кислорода
    nuH2 = stateCoordinates[:, 8]  # Число молей водорода

    # Температура аккумулятора
    TFEl = reducedTemp[:, 0] - 273.15
    TElp = reducedTemp[:, 1] - 273.15
    TEln = reducedTemp[:, 2] - 273.15

    # Получаем параметры
    Cbin0p = otherSystemParameters[28]  # Емкость положительного двойного слоя
    Cm = otherSystemParameters[29]  # Емкость мембраны
    Cbin0n = otherSystemParameters[30]  # Емкость отрицательного двойного слоя
    alphaCQp = otherSystemParameters[59]  # Зарядовый коэффициент емкости положительного электрода, 1/Кл
    alphaCQn = otherSystemParameters[60]  # Зарядовый коэффициент емкости отрицательного электрода, 1/Кл

    betaCQ2p = otherSystemParameters[93]
    betaCQ2n = otherSystemParameters[94]
    betaCQ3p = otherSystemParameters[95]
    betaCQ3n = otherSystemParameters[96]

    # Получаем сопротивление клемм
    Rkl = otherSystemParameters[-1]

    # Определяем емкости двойных слоев
    (Cbinp, Cbinn) = funCbin(qbinp, qbinn, alphaCQp, alphaCQn, Cbin0p, Cbin0n,
                             betaCQ2p, betaCQ2n, betaCQ3p, betaCQ3n)

    # Рассчитываем напряжения двойных слоев
    Ubinp = qbinp / Cbinp  # Положительный двойной слой
    Um = qm / Cm  # Мембрана
    Ubinn = qbinn / Cbinn  # Отрицательный двойной слой

    # Напряжение на клеммах
    Ukl = Ubinp + Um + Ubinn - Icur * Rkl

    # Выводм результат
    return (t, Ukl, Ubinp, Ubinn, Um,
            TFEl, TElp, TEln,
            nuH2Op, nuH2On, nuH2OStp, nuH2OStn,
            nuO2, nuH2)
