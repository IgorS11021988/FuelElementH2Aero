import numpy as np

from .StationFunctions import funHMuH2O, funHMuCam, funCbin, funRbin, funRm, funEvH2O
from MathProtEnergyProc import NonEqSystemQBase

from MathProtEnergyProc.CorrectionModel import ReluFilter


# Функция состояния для литий-ионного аккумулятора
def StateFunction(stateCoordinates,
                  reducedTemp,
                  systemParameters):
    # получаем электрические заряды
    [qbinp,  # Электрический заряд положительного двойного слоя
     qm,  # Электрический заряд мембраны
     qbinn,  # Электрический заряд отрицательного двойного слоя
     nuH2Op,  # Число молей воды в приэлектродной области положительного электрода
     nuH2On,  # Число молей воды в приэлектродной области отрицательного электрода
     nuH2OStp,  # Приведенное число молей воды в камере положительного электрода
     nuH2OStn,  # Приведенное число молей воды в камере отрицательного электрода
     nuO2,  # Число молей кислорода
     nuH2  # Число молей водорода
     ] = stateCoordinates

    # Получаем температуру
    [TFEl,
     TElp,
     TEln] = reducedTemp

    # Получаем параметры
    [I,  # Ток во внешней цепи
     Tokr,  # Температура окружающей среды
     Econ,  # Контактная ЭДС
     hH2Os,  # Характерный тепловой потенциал воды в мембране водородно-воздушного топливного элемента
     dhH2Os,  # Характерное приращение теплового потенциала воды в мембране водородно-воздушного топливного элемента
     muH2Os,  # Характерный химический потенциал воды в мембране водородно-воздушного топливного элемента
     dmuH2Os,  # Характерное приращение химического потенциала воды в мембране водородно-воздушного топливного элемента
     nuH2Os,  # Характерное число молей воды в мембране водородно-воздушного топливного элемента
     hH2OStsp,  # Характерный тепловой потенциал воды в камере положительного электрода
     dhH2OStsp,  # Характерное приращение теплового потенциала воды в камере положительного электрода
     muH2OStsp,  # Характерный химический потенциал воды в камере положительного электрода
     dmuH2OStsp,  # Характерное приращение химического потенциала воды в камере положительного электрода
     nuH2OStsp,  # Характерное число молей воды в камере положительного электрода
     hH2OStsn,  # Характерный тепловой потенциал воды в камере отрицательного электрода, В
     dhH2OStsn,  # Характерное приращение теплового потенциала воды в камере отрицательного электрода, В
     muH2OStsn,  # Характерный химический потенциал воды в камере отрицательного электрода
     dmuH2OStsn,  # Характерное приращение химического потенциала воды в камере отрицательного электрода
     nuH2OStsn,  # Характерное число молей воды в камере отрицательного электрода
     hO2s,  # Характерный тепловой потенциал кислорода
     dhO2s,  # Характерное приращение теплового потенциала кислорода
     muO2s,  # Характерный химический потенциал кислорода
     dmuO2s,  # Характерное приращение химического потенциала кислорода
     nuO2s,  # Характерное число молей кислорода
     hH2s,  # Характерный тепловой потенциал водорода
     dhH2s,  # Характерное приращение теплового потенциала водорода
     muH2s,  # Характерный химический потенциал водорода
     dmuH2s,  # Характерное приращение химического потенциала водорода
     nuH2s,  # Характерное число молей водорода
     THMus,  # Характерная температура химических потенциалов и тепловых эффектов воды и газов
     Cbin0p,  # Емкость положительного двойного слоя
     Cm,  # Емкость мембраны
     Cbin0n,  # Емкость отрицательного двойного слоя
     Rbin0p,  # Сопротивление положительного двойного слоя
     Rm0,  # Сопротивление мембраны
     Rbin0n,  # Сопротивление отрицательного двойного слоя
     KFEl,  # Коэффициент теплопередачи водородно-воздушного топливного элемента
     KElp,  # Характерный коэффициент теплопередачи содержимого камеры положительного электрода
     KEln,  # Характерный коэффициент теплопередачи содержимого камеры отрицательного электрода
     KElTop,  # Коэффициент теплопередачи к камере положительного электрода
     KElTon,  # Коэффициент теплопередачи к камере отрицательного электрода
     dKElTEvp0,  # Характерное приращение по испарению коэффициента теплопередачи к камере положительного электрода
     dKElTEvn0,  # Характерное приращение по испарению коэффициент теплопередачи к камере отрицательного электрода
     dKElTQp0,  # Характерное приращение по электродным реакциям коэффициента теплопередачи к камере положительного электрода
     dKElTQn0,  # Характерное приращение по электродным реакциям коэффициент теплопередачи к камере отрицательного электрода
     CFEls,  # Теплоемкость водородно-воздушного топливного элемента
     CElsp,  # Теплоемкость содержимого камеры положительного электрода
     CElsn,  # Теплоемкость содержимого камеры отрицательного электрода
     cFElH2O,  # Удельная теплоемкость водородно-воздушного топливного элемента по воде в мембране
     cElH2OStp,  # Удельная теплоемкость содержимого камеры положительного электрода по воде
     cElH2OStn,  # Удельная теплоемкость содержимого камеры отрицательного электрода по воде
     cElO2p,  # Удельная теплоемкость содержимого камеры положительного электрода по кислороду
     cElH2n,  # Удельная теплоемкость содержимого камеры отрицательного электрода по водороду
     alphaRIp,  # Коэффициент сопротивления по току положительного двойного слоя
     alphaRIn,  # Коэффициент сопротивления по току отрицательного двойного слоя
     alphaRTp,  # Экспоненциальный коэффициент сопротивления по температуре положительного электрода
     alphaRTm,  # Экспоненциальный коэффициент сопротивления по температуре мембраны
     alphaRTn,  # Экспоненциальный коэффициент сопротивления по температуре отрицательного электрода
     bRTp,  # Граничная температура по сопротивлению положительного электрода
     bRTm,  # Граничная температура по сопротивлению мембраны
     bRTn,  # Граничная температура по сопротивлению отрицательного электрода
     alphaCQp,  # Зарядовый коэффициент емкости положительного электрода
     alphaCQn,  # Зарядовый коэффициент емкости отрицательного электрода
     kDiffH2O0,  # Характерный коэффициент диффузии воды в водородно-воздушным топливном элементе
     dKDiffH2O0,  # Характерное приращение коэффициента диффузии воды в водородно-воздушным топливном элементе
     kEvH2Osp,  # Коэффициент испарения воды в камеру положительного электрода
     kEvH2Osn,  # Коэффициент испарения воды в камеру отрицательного электрода
     alphaKTEvH2Osp,  # Температурный показатель коэффициента испарения воды в камеру положительного электрода
     alphaKTEvH2Osn,  # Температурный показатель коэффициента испарения воды в камеру отрицательного электрода
     bTKEvH2Osp,  # Температурная граница коэффициента испарения воды в камеру положительного электрода
     bTKEvH2Osn,  # Температурная граница коэффициента испарения воды в камеру отрицательного электрода
     evExtH2Osp,  # Поток водяного пара в камеру положительного электрода
     evExtH2Osn,  # Поток водяного пара в камеру отрицательного электрода
     evExtO2s,  # Поток кислорода в камеру положительного электрода
     evExtH2s,  # Поток водорода в камеру отрицательного электрода
     qExtp,  # Внешний поток теплоты на камеру положительного электрода
     qExtn,  # Внешний поток теплоты на камеру отрицательного электрода
     nuH2Osm,  # Характерное число молей воды в мембране
     nuH2OsEvp,  # Характерное число молей воды в приэлектродной области и камере положительного электрода
     nuH2OsEvn,  # Характерное число молей воды в приэлектродной области и камере отрицательного электрода
     crRmDiffH2O,  # Коэффициент прекрестности диффузии воды и ионов водородв в мембране
     crEvH20KElp,  # Коэффициент перекрестности испарения воды и теплообмена с камерой положительного электрода
     crEvH20KEln,  # Коэффициент перекрестности испарения воды и теплообмена с камерой отрицательного электродаэлектрода
     crQKElp,  # Коэффициент перекрестности электродной реакции и теплообмена с камерой положительного электрода
     crQKEln,  # Коэффициент перекрестности электродной реакции и теплообмена с камерой отрицательного электрода

     # Получаем довесочные коэффициенты
     betaRI2p,
     betaRI2n,
     betaRI3p,
     betaRI3n,
     betaRT2p,
     betaRT2m,
     betaRT2n,
     betaRT3p,
     betaRT3m,
     betaRT3n,
     betaCQ2p,
     betaCQ2n,
     betaCQ3p,
     betaCQ3n,
     betaKRmH2O2,
     betaKRmH2O3,
     betaKTEvH2Op2,
     betaKTEvH2On2,
     betaKNuEvH2Op2,
     betaKNuEvH2On2,
     betaKTEvH2Op3,
     betaKTEvH2On3,
     betaKNuEvH2Op3,
     betaKNuEvH2On3,
     betaMuH2O2,
     betaMuH2O3,
     betaHH2O2,
     betaHH2O3,
     betaMuH2OStp,
     betaMuH2OStO2p,
     betaMuO2p,
     betaHH2OStp,
     betaHH2OStO2p,
     betaHO2p,
     betaMuH2OStn,
     betaMuH2OStH2n,
     betaMuH2n,
     betaHH2OStn,
     betaHH2OStH2n,
     betaHH2n,

     Rkl  # Сопротивление клемм
     ] = systemParameters

    # Матрица баланса
    balanceMatrix = np.array([])

    # Определяем отток воды
    evExtH2Op = -evExtH2Osp * (nuH2OStp / nuH2OStsp - 1)
    evExtH2On = -evExtH2Osn * (nuH2OStn / nuH2OStsn - 1)

    # Определяем приток кислорода и водорода
    evExtO2 = -evExtO2s * (nuO2 / nuO2s - 1)
    evExtH2 = -evExtH2s * (nuH2 / nuH2s - 1)

    # Внешние потоки зарядов
    stateCoordinatesStreams = np.array([-I, -I, -I, evExtH2Op, evExtH2On, evExtO2, evExtH2], dtype=np.double)

    # Выделившаяся джоулева теплота в клеммах
    QKl = Rkl * np.power(I, 2)

    # Внешние потоки теплоты
    heatEnergyPowersStreams = np.array([QKl, qExtp, qExtn], dtype=np.double)

    # Выводим температуры
    energyPowerTemperatures = np.array([TFEl, TElp, TEln, Tokr], dtype=np.double)

    # Определяем химический потенциал воды в приэлектродных областях мембраны
    (muH2Op, hH2Op) = funHMuH2O(nuH2Op, TFEl,
                                nuH2Os, THMus,
                                muH2Os, dmuH2Os,
                                hH2Os, dhH2Os,
                                betaMuH2O2, betaMuH2O3,
                                betaHH2O2, betaHH2O3,
                                cFElH2O)  # Положительный электрод
    (muH2On, hH2On) = funHMuH2O(nuH2On, TFEl,
                                nuH2Os, THMus,
                                muH2Os, dmuH2Os,
                                hH2Os, dhH2Os,
                                betaMuH2O2, betaMuH2O3,
                                betaHH2O2, betaHH2O3,
                                cFElH2O)  # Отрицательный электрод

    # Определяем химические потенциалы кислорода, водорода и воды в камерах электродов при стандартной температуре
    (muH2OStp, muO2, hH2OStp, hO2) = funHMuCam(nuH2OStp, nuO2, TElp,
                                               nuH2OStsp, nuO2s, THMus,
                                               muH2OStsp, dmuH2OStsp, muO2s, dmuO2s,
                                               hH2OStsp, dhH2OStsp, hO2s, dhO2s,
                                               betaMuH2OStp, betaMuH2OStO2p, betaMuO2p,
                                               betaHH2OStp, betaHH2OStO2p, betaHO2p,
                                               cElH2OStp, cElO2p)  # Определяем химические потенциалы кислорода и воды в камере положительного электрода
    (muH2OStn, muH2, hH2OStn, hH2) = funHMuCam(nuH2OStn, nuH2, TEln,
                                               nuH2OStsn, nuH2s, THMus,
                                               muH2OStsn, dmuH2OStsn, muH2s, dmuH2s,
                                               hH2OStsn, dhH2OStsn, hH2s, dhH2s,
                                               betaMuH2OStn, betaMuH2OStH2n, betaMuH2n,
                                               betaHH2OStn, betaHH2OStH2n, betaHH2n,
                                               cElH2OStn, cElH2n)  # Определяем химические потенциалы кислорода и воды в камере отрицательного электрода

    # Определяем емкости двойных слоев
    (Cbinp, Cbinn) = funCbin(qbinp, qbinn, alphaCQp, alphaCQn, Cbin0p, Cbin0n,
                             betaCQ2p, betaCQ2n, betaCQ3p, betaCQ3n)

    # Определяем напряжения на двойных слоях
    dissUbinp =  Econ - qbinp / Cbinp  # Положительный двойной слой
    dissUbinn = -Econ - qbinn / Cbinn  # Отрицательный двойной слой

    # Потенциалы взаимодействия энергетических степеней свободы
    potentialInter = np.array([dissUbinp, -qm / Cm, dissUbinn,
                               -muH2Op, -muH2On, -muH2OStp, -muH2OStn,
                               -muO2, -muH2], dtype=np.double)

    # Потенциалы взаимодействия между энергетическими степенями свободы
    potentialInterBet = np.array([])

    # Доли распределения некомпенсированной теплоты
    beta = np.array([])

    # Определяем сопротивления двойных слоев (вместе с теплообменом с камерами электродов)
    (Rbinp, Rbinn,
     dKElTQp, dKElTQn,
     crKElTQp, crKElTQn) = funRbin(TFEl, dissUbinp, dissUbinn, alphaRIp, alphaRIn,
                                   alphaRTp, alphaRTn, bRTp, bRTn, Rbin0p, Rbin0n,
                                   dKElTQp0, dKElTQn0, crQKElp, crQKEln, betaRI2p,
                                   betaRI2n, betaRI3p, betaRI3n, betaRT2p, betaRT2n,
                                   betaRT3p, betaRT3n)

    # Определяем сопротивления мембраны (вместе с диффузией воды)
    (Rm, kDiffH2O, crKDiffH2O) = funRm(TFEl, nuH2Op, nuH2On, nuH2Osm, alphaRTm, bRTm,
                                       Rm0, kDiffH2O0, dKDiffH2O0, crRmDiffH2O,
                                       betaRT2m, betaRT3m, betaKRmH2O2, betaKRmH2O3)

    # Определяем коэфициенты испарения воды (вместе с теплообменом с камерами электродов)
    (kEvH2Op, kEvH2On,
     dKElTEvp, dKElTEvn,
     crKElTEvp, crKElTEvn) = funEvH2O(TFEl, TElp, TEln, nuH2Op, nuH2On, nuH2OStp,
                                      nuH2OStn, nuH2OsEvp, nuH2OsEvn, kEvH2Osp,
                                      kEvH2Osn, dKElTEvp0, dKElTEvn0, crEvH20KElp,
                                      crEvH20KEln, alphaKTEvH2Osp, alphaKTEvH2Osn,
                                      bTKEvH2Osp, bTKEvH2Osn, betaKTEvH2Op2,
                                      betaKTEvH2On2, betaKNuEvH2Op2, betaKNuEvH2On2,
                                      betaKTEvH2Op3, betaKTEvH2On3, betaKNuEvH2Op3,
                                      betaKNuEvH2On3)

    # Главный блок кинетической матрицы по процессам
    kineticMatrixPCPC = np.array([1 / Rbinp, 1 / Rm, 1 / Rbinn,
                                  kDiffH2O, kEvH2Op, kEvH2On,
                                  crKDiffH2O, crKDiffH2O], dtype=np.double) * Tokr / (4.642 * NonEqSystemQBase.GetTbase())

    # Перекрестные блоки кинетической матрицы по процессам
    kineticMatrixPCHeat = np.array([crKElTEvp, crKElTEvn, crKElTQp, crKElTQn], dtype=np.double) * Tokr / 4.642
    kineticMatrixHeatPC = np.array([crKElTEvp, crKElTEvn, crKElTQp, crKElTQn], dtype=np.double) * Tokr / 4.642

    # Главный блок кинетической матрицы по теплообмену
    KFEl = ReluFilter(KFEl)
    KElTop = ReluFilter(KElTop)
    KElTon = ReluFilter(KElTon)
    KElp = ReluFilter(KElp)
    KEln = ReluFilter(KEln)
    kineticMatrixHeatHeat = np.array([KFEl,
                                      KElTop + dKElTEvp + dKElTQp,
                                      KElTon + dKElTEvn + dKElTQn,
                                      KElp, KEln], dtype=np.double) * np.power(Tokr, 2) / (21.55 * NonEqSystemQBase.GetTbase())

    # Определяем теплоемкости
    CFEl = CFEls + cFElH2O * (nuH2Op + nuH2On)  # Темплоемкость элемента
    CElp = CElsp + cElH2OStp * nuH2OStp + cElO2p * nuO2  # Теплоемкость положительного электрода
    CEln = CElsn + cElH2OStn * nuH2OStn + cElH2n * nuH2  # Теплоемкость отрицательного электрода

    # Обратная теплоемкость водородно-воздушного топливного элемента
    invHeatCapacityMatrixCf = 1 / np.array([CFEl, CElp, CEln], dtype=np.double)

    # Приведенные тепловые эффекты водородно-воздушного топливного элемента
    heatEffectsGH2O = -np.array([hH2Op, hH2On, hH2OStp, hH2OStn, hO2, hH2], dtype=np.double)
    heatEffectMatrixCf = np.hstack([potentialInter[0:3], heatEffectsGH2O]) / np.array([CFEl, CFEl, CFEl, CFEl, CFEl, CElp, CEln, CElp, CEln], dtype=np.double)

    # Выводим результат
    return (balanceMatrix,
            stateCoordinatesStreams,
            heatEnergyPowersStreams,
            energyPowerTemperatures,
            potentialInter,
            potentialInterBet,
            beta, kineticMatrixPCPC,
            kineticMatrixPCHeat,
            kineticMatrixHeatPC,
            kineticMatrixHeatHeat,
            invHeatCapacityMatrixCf,
            heatEffectMatrixCf)
