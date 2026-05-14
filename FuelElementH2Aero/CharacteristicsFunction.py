import numpy as np

from MathProtEnergyProcBase.IndexFunctions import GetIndex, GetIndexes

from .AttributesNames import stateCoordinatesNames, reducedTemperaturesEnergyPowersNames, USystemParametersNames, otherSystemParametersNames, processCoordinatesNames


# Индексы координат состояния
nuH2OpInd = GetIndex(stateCoordinatesNames, "nuH2Op")  # Индекс зарядового числа молей воды в приэлектродной области положительного электрода
nuH2OnInd = GetIndex(stateCoordinatesNames, "nuH2On")  # Индекс зарядового числа молей воды в приэлектродной области отрицательного электрода
nuH2OStpInd = GetIndex(stateCoordinatesNames, "nuH2OStp")  # Индекс зарядового числа молей воды в камере положительного электрода
nuH2OStnInd = GetIndex(stateCoordinatesNames, "nuH2OStn")  # Индекс зарядового числа молей воды в камере отрицательного электрода
nuO2Ind = GetIndex(stateCoordinatesNames, "nuO2")  # Индекс зарядового числа молей кислорода
nuO2dpInd = GetIndex(stateCoordinatesNames, "nuO2dp")  # Индекс зарядового числа молей растворенного кислорода в области положительного электрода
nuO2dnInd = GetIndex(stateCoordinatesNames, "nuO2dn")  # Индекс зарядового числа молей растворенного кислорода в области отрицательного электрода
nuH2Ind = GetIndex(stateCoordinatesNames, "nuH2")  # Индекс зарядового числа молей водорода
nuH2dpInd = GetIndex(stateCoordinatesNames, "nuH2dp")  # Индекс зарядового числа молей растворенного водорода в области положительного электрода
nuH2dnInd = GetIndex(stateCoordinatesNames, "nuH2dn")  # Индекс зарядового числа молей растворенного водорода в области отрицательного электрода

# Индексы приведенных температур
TFElInd = GetIndex(reducedTemperaturesEnergyPowersNames, "TFEl")  # Температура топливного элемента
TElpInd = GetIndex(reducedTemperaturesEnergyPowersNames, "TElp")  # Температура в камере положительного электрода
TElnInd = GetIndex(reducedTemperaturesEnergyPowersNames, "TEln")  # Температура в камере отрицательного электрода

# Индексы переменных параметров системы
IInd = GetIndex(USystemParametersNames, "I")  # Индекс тока

# Индексы токов в элементе
IElInd = GetIndexes(processCoordinatesNames, ["dqbinp", "dqm", "dqbinn", "utFuelp", "utFueln"])

# Индексы параметров системы
RklInd = GetIndex(otherSystemParametersNames, "Rkl")


# Функция состояния для литий-ионного аккумулятора
def CharacteristicsFunction(t,  # Моменты времени
                            stateCoordinates,  # Координаты состояния
                            reducedTemp,  # Приведенные температуры
                            USystemParameters,  # U-параметры системы
                            otherSystemParameters,  # Прочие параметры системы
                            nEqSysQ  # Указатель на функцию системы
                            ):
    # Рассчитываем аттрибуты системы
    def GetElAttr(ind):
        # Рассчитываем аттрибуты
        nEqSysQ.CountSystem(stateCoordinates[ind],  # Координаты состояния
                            reducedTemp[ind],  # Приведенные температуры энергетических степеней свободы
                            np.hstack([USystemParameters[ind], otherSystemParameters])  # Параметры системы
                            )

        # Получаем напряжения двойных слоев
        Ubinp = nEqSysQ.GetStateFunction().GetIndepStateFunction().GetUbinp()  # Положительный двойной слой
        Ubinn = nEqSysQ.GetStateFunction().GetIndepStateFunction().GetUbinn()  # Отрицательный двойной слой

        # Получаем напряжение мембраны
        Um = nEqSysQ.GetStateFunction().GetIndepStateFunction().GetUm()

        # Получаем токи через двойные слои
        (Ibinp, Im, Ibinn, vUtFuelp, vUtFueln) = nEqSysQ.GetVProcessCoordinates()[IElInd]

        # Выводим результат
        return (Ibinp, Im, Ibinn, Ubinp, Um, Ubinn, vUtFuelp, vUtFueln)
    inds = np.arange(t.shape[0])  # Массив индексов
    GetAttrs = np.vectorize(GetElAttr)
    (Ibinp, Im, Ibinn, Ubinp, Um, Ubinn, vUtFuelp, vUtFueln) = GetAttrs(inds)

    # Получаем динамику тока
    Icur = USystemParameters[:, IInd]  # Ток в текущие моменты времени

    # Получаем координаты состояния
    nuH2Op = stateCoordinates[:, nuH2OpInd]  # Зарядовое число молей воды в приэлектродной области положительного электрода
    nuH2On = stateCoordinates[:, nuH2OnInd]  # Зарядовое число молей воды в приэлектродной области отрицательного электрода
    nuH2OStp = stateCoordinates[:, nuH2OStpInd]  # Зарядовое число молей воды в камере положительного электрода
    nuH2OStn = stateCoordinates[:, nuH2OStnInd]  # Зарядовое число молей воды в камере отрицательного электрода
    nuO2 = stateCoordinates[:, nuO2Ind]  # Зарядовое число молей кислорода
    nuO2dp = stateCoordinates[:, nuO2dpInd]  # Число молей растворенного кислорода в области положительного электрода
    nuO2dn = stateCoordinates[:, nuO2dnInd]  # Число молей растворенного кислорода в области отрицательного электрода
    nuH2 = stateCoordinates[:, nuH2Ind]  # Зарядовое число молей водорода
    nuH2dp = stateCoordinates[:, nuH2dpInd]  # Число молей растворенного водорода в области положительного электрода
    nuH2dn = stateCoordinates[:, nuH2dnInd]  # Число молей растворенного водорода в области отрицательного электрода

    # Температура аккумулятора
    TFEl = reducedTemp[:, TFElInd] - 273.15  # Температура топливного элемента
    TElp = reducedTemp[:, TElpInd] - 273.15  # Температура в камере положительного электрода
    TEln = reducedTemp[:, TElnInd] - 273.15  # Температура в камере отрицательного электрода

    # Получаем сопротивление на клеммах
    Rkl = otherSystemParameters[RklInd]

    # Напряжение на клеммах
    Ukl = Ubinp + Um + Ubinn - Icur * Rkl

    # Выводм результат
    return (t.reshape(-1,), Ukl, Ubinp, Ubinn, Um,
            TFEl, TElp, TEln, Icur, Ibinp, Im, Ibinn,
            nuH2Op, nuH2On, nuH2OStp, nuH2OStn,
            nuO2, nuH2, nuO2dp, nuO2dn, nuH2dp, nuH2dn,
            vUtFuelp, vUtFueln)
