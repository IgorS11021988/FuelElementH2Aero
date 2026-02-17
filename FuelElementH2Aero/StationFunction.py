import numpy as np

from .StationFunctions import funJHSzTEl, funJHSzTCam, funCbin, funRbin, funRm, funEvH2O
from MathProtEnergyProc import NonEqSystemQBase


# Функция состояния для литий-ионного аккумулятора
def IndepStateFunction(stateCoordinates,
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
     cRTp,  # Температурный коэффициент по сопротивлению положительного электрода
     cRTm,  # Температурный коэффициент по сопротивлению мембраны
     cRTn,  # Температурный коэффициент по сопротивлению отрицательного электрода
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
     cTKEvH2Osp,  # Температурный коэффициент испарения воды в камеру положительного электрода
     cTKEvH2Osn,  # Температурный коэффициент испарения воды в камеру отрицательного электрода
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
     TOkrs,  # Постоянная температура окружающей среды

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

    # Определяем отток воды
    evExtH2Op = evExtH2Osp * (nuH2OStp / nuH2OStsp - 1)
    evExtH2On = evExtH2Osn * (nuH2OStn / nuH2OStsn - 1)

    # Определяем приток кислорода и водорода
    evExtO2 = -evExtO2s * (nuO2 / nuO2s - 1)
    evExtH2 = -evExtH2s * (nuH2 / nuH2s - 1)

    # Выделившаяся джоулева теплота в клеммах
    QKl = Rkl * np.power(I, 2)

    # Определяем емкости двойных слоев
    (Cbinp, Cbinn) = funCbin(qbinp, qbinn, alphaCQp, alphaCQn, Cbin0p, Cbin0n,
                             betaCQ2p, betaCQ2n, betaCQ3p, betaCQ3n)

    # Определяем химический потенциал воды в приэлектродных областях мембраны
    (JSzEl, HSzTEl,
     JSTEl, HSTTEl,
     dissUbinp, dissUbinn) = funJHSzTEl(qbinp, qm, qbinn,
                                        nuH2Op, nuH2On, TFEl,
                                        Cbinp, Cm, Cbinn, Econ,
                                        nuH2Os, THMus, muH2Os,
                                        dmuH2Os, hH2Os, dhH2Os,
                                        betaMuH2O2, betaMuH2O3,
                                        betaHH2O2, betaHH2O3,
                                        cFElH2O, CFEls)

    # Определяем химические потенциалы кислорода, водорода и воды в камерах электродов при стандартной температуре
    (JSzCamp, HSzTCamp,
     JSTCamp, HSTTCamp) = funJHSzTCam(nuH2OStp, nuO2, TElp,
                                      nuH2OStsp, nuO2s, THMus,
                                      muH2OStsp, dmuH2OStsp, muO2s, dmuO2s,
                                      hH2OStsp, dhH2OStsp, hO2s, dhO2s,
                                      betaMuH2OStp, betaMuH2OStO2p, betaMuO2p,
                                      betaHH2OStp, betaHH2OStO2p, betaHO2p,
                                      cElH2OStp, cElO2p, CElsp)  # Определяем химические потенциалы кислорода и воды в камере положительного электрода
    (JSzCamn, HSzTCamn,
     JSTCamn, HSTTCamn) = funJHSzTCam(nuH2OStn, nuH2, TEln,
                                      nuH2OStsn, nuH2s, THMus,
                                      muH2OStsn, dmuH2OStsn, muH2s, dmuH2s,
                                      hH2OStsn, dhH2OStsn, hH2s, dhH2s,
                                      betaMuH2OStn, betaMuH2OStH2n, betaMuH2n,
                                      betaHH2OStn, betaHH2OStH2n, betaHH2n,
                                      cElH2OStn, cElH2n, CElsp)  # Определяем химические потенциалы кислорода и воды в камере отрицательного электрода

    # Определяем сопротивления двойных слоев (вместе с теплообменом с камерами электродов)
    (sbinp, sbinn) = funRbin(TFEl, dissUbinp, dissUbinn, alphaRIp,
                             alphaRIn, alphaRTp, alphaRTn, bRTp,
                             bRTn, cRTp, cRTn, betaRI2p, betaRI2n,
                             betaRI3p, betaRI3n, betaRT2p, betaRT2n,
                             betaRT3p, betaRT3n)

    # Определяем коэфициенты испарения воды (вместе с теплообменом с камерами электродов)
    (kbinp, kbinn) = funEvH2O(TFEl, TElp, TEln, nuH2Op, nuH2On, nuH2OStp,
                              nuH2OStn, nuH2OsEvp, nuH2OsEvn, kEvH2Osp,
                              kEvH2Osn, dKElTEvp0, dKElTEvn0, crEvH20KElp,
                              crEvH20KEln, alphaKTEvH2Osp, alphaKTEvH2Osn,
                              bTKEvH2Osp, bTKEvH2Osn, cTKEvH2Osp, cTKEvH2Osn, 
                              betaKTEvH2Op2, betaKTEvH2On2, betaKNuEvH2Op2,
                              betaKNuEvH2On2, betaKTEvH2Op3, betaKTEvH2On3,
                              betaKNuEvH2Op3, betaKNuEvH2On3)

    # Определяем сопротивления мембраны (вместе с диффузией воды)
    sm = funRm(TFEl, nuH2Op, nuH2On, nuH2Osm,
               alphaRTm, bRTm, cRTm, betaRT2m,
               betaRT3m, betaKRmH2O2, betaKRmH2O3)

    # Определяем обратимые и необратимые составляющие кинетической матрицы положительной камеры
    kInvMatrixElEvs = np.array([0, 0, 1], dtype=np.double).reshape(-1, 1)
    kInvMatrixElpEchCr = (np.sqrt(np.array([1 / Rbin0p, 0, dKElTQp0],
                                            dtype=np.double)) * np.array([1, 1, crQKElp],
                                                                         dtype=np.double)).reshape(-1, 1)
    kInvMatrixElpEvCr = (np.sqrt(np.array([0, kEvH2Osp, dKElTEvp0],
                                           dtype=np.double)) * np.array([1, 1, crEvH20KElp],
                                                                        dtype=np.double)).reshape(-1, 1)
    kNoInvMatrixElp = np.array([KElTop * Tokr / TOkrs + \
                                dKElTQp0 * (Tokr / TOkrs - np.power(crQKElp, 2)) * sbinp + \
                                dKElTEvp0 * (Tokr / TOkrs - np.power(crEvH20KElp, 2)) * kbinp,
                                sbinp, kbinp], dtype=np.double) * Tokr / (4.642 * NonEqSystemQBase.GetTbase())

    # Определяем обратимые и необратимые составляющие кинетической матрицы отрицательной камеры
    kInvMatrixElnEchCr = (np.sqrt(np.array([1 / Rbin0n, 0, dKElTQn0],
                                            dtype=np.double)) * np.array([1, 1, crQKEln],
                                                                         dtype=np.double)).reshape(-1, 1)
    kInvMatrixElnEvCr = (np.sqrt(np.array([0, kEvH2Osn, dKElTEvn0],
                                           dtype=np.double)) * np.array([1, 1, crEvH20KEln],
                                                                        dtype=np.double)).reshape(-1, 1)
    kNoInvMatrixEln = np.array([KElTon * Tokr / TOkrs + \
                                dKElTQn0 * (Tokr / TOkrs - np.power(crQKEln, 2)) * sbinn + \
                                dKElTEvn0 * (Tokr / TOkrs - np.power(crEvH20KEln, 2)) * kbinn,
                                sbinn, kbinn], dtype=np.double) * Tokr / (4.642 * NonEqSystemQBase.GetTbase())

    # Определяем обратимые и необратимые составляющие кинетической матрицы мембраны
    kInvMatrixElmDiffs = np.array([0, 1], dtype=np.double).reshape(-1, 1)
    kInvMatrixElmCr = (np.sqrt(np.array([1 / Rm0, dKDiffH2O0],
                                        dtype=np.double)) * np.array([1, crRmDiffH2O],
                                                                     dtype=np.double)).reshape(-1, 1)
    kNoInvMatrixElm = np.array([kDiffH2O0 + dKDiffH2O0 * (1 - np.power(crRmDiffH2O, 2)) * sm,
                                sm], dtype=np.double) * Tokr / (4.642 * NonEqSystemQBase.GetTbase())

    # Определяем необратимые составляющие динамической матрицы по теплообмену с окружающей средой
    kQOkr = np.array([KFEl, KElp, KEln], dtype=np.double) * np.power(Tokr, 2) / (4.642 * TOkrs * NonEqSystemQBase.GetTbase())

    # Выводим результат
    return (evExtH2Op, evExtH2On,
            evExtO2, evExtH2, QKl,
            JSzEl, HSzTEl,
            JSTEl, HSTTEl,
            JSzCamp, HSzTCamp,
            JSTCamp, HSTTCamp,
            JSzCamn, HSzTCamn,
            JSTCamn, HSTTCamn,
            kInvMatrixElEvs,
            kInvMatrixElpEchCr,
            kInvMatrixElpEvCr,
            kNoInvMatrixElp,
            kInvMatrixElnEchCr,
            kInvMatrixElnEvCr,
            kNoInvMatrixEln,
            kInvMatrixElmDiffs,
            kInvMatrixElmCr,
            kNoInvMatrixElm,
            kQOkr, I, Tokr,
            qExtp, qExtn)
