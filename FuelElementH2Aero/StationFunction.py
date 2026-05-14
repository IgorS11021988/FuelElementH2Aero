import numpy as np

from .StationFunctions import funNuEMat, funMuMat, funADNu
from MathProtEnergyProc import NonEqSystemQBase


# Функция состояния для литий-ионного аккумулятора
def IndepStateFunction(stateCoordinates,
                       reducedTemp,
                       systemParameters):
    # Получаем числа молей материалов
    [nuMat,  # Число молей недеградированного материала
     nuMatDeg  # Число молей деградированного материала
     ] = stateCoordinates

    # Получаем температуру
    [TDegMat,  # Температура деградирующегося материала
     TMat  # Температура недеградирующегося материала
     ] = reducedTemp

    # Получаем параметры
    [vAlpha,  # Внешний поток
     Tokr,  # Температура окружающей среды
     sMuDeg,  # Пороговый химический потенциал разрушения
     CMuDegMat,  # Коэффициент химического потенциала деградирующегося материала
     nuMats,  # Характерное число молей
     NuAll,  # Общее число молей материала
     ADNuMat0,  # Коэффициент перестройки расположения атомов
     ADNuMatDeg0,  # Коэффициент деградации материала
     KDegMat,  # Коэффициент теплопередачи деградируемого материала недеградированному
     CQDegMat,  # Теплоемкость деградируемого материала
     KMat,  # Коэффициент теплоотдачи недеградируемого материала
     CQMat,  # Теплоемкость недеградируемого материала
     alphaADNuMatT,  # Экспоненциальный коэффициент перестройки расположения атомов по температуре
     alphaADNuMatDegT,  # Экспоненциальный коэффициент дегарадции материала по температуре
     bADNuMatT,  # Граничная температура по коэффициенту перестройки расположения атомов
     bADNuMatDegT,  # Граничная температура по коэффициенту деградации материала
     rCADNuMatT,  # Постоянный коэффициент температурной зависимости коэффициента перестройки расположения атомов
     rCADNuMatDegT,  # Постоянный коэффициент температурной зависимости дегарадации материала

     betaADNuMatDeg1,
     betaADNuMatDeg2,
     betaADNuMatDeg3,
     betaMu2,
     betaMu3,
     betaADNuMatT2,
     betaADNuMatDegT2,
     betaADNuMatT3,
     betaADNuMatDegT3
     ] = systemParameters

    # Рассчитываем корректировочный коэффициент потока
    kVAlpha = (1 - np.exp(-nuMat / nuMats)) * (np.sign(nuMat) + 1) / 2

    # Расчитываем внешний поток
    sVAlpha = -vAlpha * kVAlpha

    # Определяем число молей возбужденных молекул
    (nuEMat, rNuEMat) = funNuEMat(nuMat, nuMatDeg, NuAll)

    # Определяем химические потенциалы деградированного и недеградированного материала
    (muMat, muMatDeg) = funMuMat(nuEMat, rNuEMat,
                                 CMuDegMat, sMuDeg,
                                 betaMu2, betaMu3)

    # Матрица Якоби приведенной энтропии по числам молей
    JSNu = np.array([muMat, muMatDeg], dtype=np.double) / TDegMat

    # Матрица Гесса приведенной энтропии по температуре и электрическим зарядам
    HSNuT = np.vstack([-JSNu / TDegMat,
                       np.zeros_like(JSNu)])

    # Приведенные первые и вторые производные приведенной энтропии по температуре
    JST = np.array([CQDegMat, CQMat], dtype=np.double) / reducedTemp
    HSTT = -JST / reducedTemp

    # Определяем сопротивления двойных слоев и мембраны
    ADNu = funADNu(rNuEMat, TDegMat, muMat, muMatDeg, ADNuMat0, ADNuMatDeg0,
                   alphaADNuMatT, alphaADNuMatDegT, bADNuMatT, bADNuMatDegT, rCADNuMatT, rCADNuMatDegT,
                   betaADNuMatT2, betaADNuMatDegT2, betaADNuMatT3, betaADNuMatDegT3,
                   betaADNuMatDeg1, betaADNuMatDeg2, betaADNuMatDeg3) * TDegMat / NonEqSystemQBase.GetTbase()

    # Коэффициенты теплообмена
    KQMat = np.array([KDegMat * TDegMat, KMat * Tokr], dtype=np.double) * TMat / NonEqSystemQBase.GetTbase()

    # Выводим результат
    return (Tokr, sVAlpha,
            JSNu, JST, HSNuT, HSTT,
            ADNu, KQMat)
