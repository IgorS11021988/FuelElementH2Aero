from MathProtEnergyProcBase.IndexFunctions import GetIndex, GetIndexes

from .AttributesNames import stateCoordinatesNames, reducedTemperaturesEnergyPowersNames, USystemParametersNames, otherSystemParametersNames
from .StationFunctions import funCbin


# Индексы координат состояния
qbinpInd = GetIndex(stateCoordinatesNames, "qbinp")  # Индекс заряда положительного электрода
qmInd = GetIndex(stateCoordinatesNames, "qm")  # Индекс заряда мембраны
qbinnInd = GetIndex(stateCoordinatesNames, "qbinn")  # Индекс заряда отрицательного электрода
nuH2OpInd = GetIndex(stateCoordinatesNames, "nuH2Op")  # Индекс зарядового числа молей воды в приэлектродной области положительного электрода
nuH2OnInd = GetIndex(stateCoordinatesNames, "nuH2On")  # Индекс зарядового числа молей воды в приэлектродной области отрицательного электрода
nuH2OStpInd = GetIndex(stateCoordinatesNames, "nuH2OStp")  # Индекс зарядового числа молей воды в камере положительного электрода
nuH2OStnInd = GetIndex(stateCoordinatesNames, "nuH2OStn")  # Индекс зарядового числа молей воды в камере отрицательного электрода
nuO2Ind = GetIndex(stateCoordinatesNames, "nuO2")  # Индекс зарядового числа молей кислорода
nuH2Ind = GetIndex(stateCoordinatesNames, "nuH2")  # Индекс зарядового числа молей водорода

# Индексы приведенных температур
TFElInd = GetIndex(reducedTemperaturesEnergyPowersNames, "TFEl")  # Температура топливного элемента
TElpInd = GetIndex(reducedTemperaturesEnergyPowersNames, "TElp")  # Температура в камере положительного электрода
TElnInd = GetIndex(reducedTemperaturesEnergyPowersNames, "TEln")  # Температура в камере отрицательного электрода

# Индексы переменных параметров системы
IInd = GetIndex(USystemParametersNames, "I")  # Индекс тока

# Индексы параметров системы
systemParametersIndexes = GetIndexes(otherSystemParametersNames, ["Cbin0p",  # Емкость положительного двойного слоя
                                                                  "Cm",  # Емкость мембраны
                                                                  "Cbin0n",  # Емкость отрицательного двойного слоя
                                                                  "alphaCQp",  # Зарядовый коэффициент емкости положительного электрода
                                                                  "alphaCQn",  # Зарядовый коэффициент емкости отрицательного электрода

                                                                  "betaCQ2p",
                                                                  "betaCQ2n",
                                                                  "betaCQ3p",
                                                                  "betaCQ3n",

                                                                  "Rkl"  # Сопротивление клемм
                                                                  ])


# Функция состояния для литий-ионного аккумулятора
def CharacteristicsFunction(t,  # Моменты времени
                            stateCoordinates,  # Координаты состояния
                            reducedTemp,  # Приведенные температуры
                            USystemParameters,  # U-параметры системы
                            otherSystemParameters  # Прочие параметры системы
                            ):
    # Получаем динамику тока
    Icur = USystemParameters[:, IInd]  # Ток в текущие моменты времени

    # Получаем координаты состояния
    qbinp = stateCoordinates[:, qbinpInd]  # Заряд на положительном двойном слое
    qm = stateCoordinates[:, qmInd]  # Заряд на мембране
    qbinn = stateCoordinates[:, qbinnInd]  # Заряд на отрицательном двойном слое
    nuH2Op = stateCoordinates[:, nuH2OpInd]  # Зарядовое число молей воды в приэлектродной области положительного электрода
    nuH2On = stateCoordinates[:, nuH2OnInd]  # Зарядовое число молей воды в приэлектродной области отрицательного электрода
    nuH2OStp = stateCoordinates[:, nuH2OStpInd]  # Зарядовое число молей воды в камере положительного электрода
    nuH2OStn = stateCoordinates[:, nuH2OStnInd]  # Зарядовое число молей воды в камере отрицательного электрода
    nuO2 = stateCoordinates[:, nuO2Ind]  # Зарядовое число молей кислорода
    nuH2 = stateCoordinates[:, nuH2Ind]  # Зарядовое число молей водорода

    # Температура аккумулятора
    TFEl = reducedTemp[:, TFElInd] - 273.15  # Температура топливного элемента
    TElp = reducedTemp[:, TElpInd] - 273.15  # Температура в камере положительного электрода
    TEln = reducedTemp[:, TElnInd] - 273.15  # Температура в камере отрицательного электрода

    # Получаем параметры
    [Cbin0p,  # Емкость положительного двойного слоя
     Cm,  # Емкость мембраны
     Cbin0n,  # Емкость отрицательного двойного слоя
     alphaCQp,  # Зарядовый коэффициент емкости положительного электрода
     alphaCQn,  # Зарядовый коэффициент емкости отрицательного электрода

     betaCQ2p,
     betaCQ2n,
     betaCQ3p,
     betaCQ3n,

     Rkl  # Сопротивление клемм
     ] = otherSystemParameters[systemParametersIndexes]

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
    return (t.reshape(-1,), Ukl, Ubinp, Ubinn, Um,
            TFEl, TElp, TEln,
            nuH2Op, nuH2On, nuH2OStp, nuH2OStn,
            nuO2, nuH2)
