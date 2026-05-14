import numpy as np

from .StationFunctions import funJHSzTEl, funJHSzTCam, funCbin, funRbin, funRm, funEvH2O, funKDiss, funUtKH2O2

from MathProtEnergyProc import NonEqSystemQBase


# Функция состояния для литий-ионного аккумулятора
class IndepStateFunction(object):
    # Тело функции
    def __call__(self,

                 stateCoordinates,
                 reducedTemp,
                 systemParameters):
        # получаем электрические заряды
        [qbinp,  # Электрический заряд положительного двойного слоя
         qm,  # Электрический заряд мембраны
         qbinn,  # Электрический заряд отрицательного двойного слоя
         nuH2Op,  # Зарядовое число молей воды в области положительного электрода
         nuH2On,  # Зарядовое число молей воды в области отрицательного электрода
         nuH2OStp,  # Зарядовое число молей воды в камере положительного электрода
         nuH2OStn,  # Зарядовое число молей воды в камере отрицательного электрода
         nuO2,  # Число молей кислорода
         nuO2dp,  # Число молей растворенного кислорода в области положительного электрода
         nuO2dn,  # Число молей растворенного кислорода в области отрицательного электрода
         nuH2,  # Число молей водорода
         nuH2dp,  # Число молей растворенного водорода в области положительного электрода
         nuH2dn  # Число молей растворенного водорода в области отрицательного электрода
         ] = stateCoordinates

        # Получаем температуру
        [TFEl,  # Температура топливного элемента
         TElp,  # Температура в камере положительного электрода
         TEln  # Температура в камере отрицательного электрода
         ] = reducedTemp

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
         hH2OStsn,  # Характерный тепловой потенциал воды в камере отрицательного электрода
         dhH2OStsn,  # Характерное приращение теплового потенциала воды в камере отрицательного электрода
         muH2OStsn,  # Характерный химический потенциал воды в камере отрицательного электрода
         dmuH2OStsn,  # Характерное приращение химического потенциала воды в камере отрицательного электрода
         nuH2OStsn,  # Характерное число молей воды в камере отрицательного электрода
         hO2s,  # Характерный тепловой потенциал кислорода
         hO2ds,  # Характерный тепловой потенциал растворенного кислорода
         dhO2s,  # Характерное приращение теплового потенциала кислорода
         dhO2ds,  # Характерное приращение теплового потенциала растворенного кислорода
         muO2s,  # Характерный химический потенциал кислорода
         muO2ds,  # Характерный химический потенциал растворенного кислорода
         dmuO2s,  # Характерное приращение химического потенциала кислорода
         dmuO2ds,  # Характерное приращение химического потенциала растворенного кислорода
         nuO2s,  # Характерное число молей кислорода
         nuO2ds,  # Характерное число молей растворенного кислорода
         hH2s,  # Характерный тепловой потенциал водорода
         hH2ds,  # Характерный тепловой потенциал растворенного водорода
         dhH2s,  # Характерное приращение теплового потенциала водорода
         dhH2ds,  # Характерное приращение теплового потенциала растворенного водорода
         muH2s,  # Характерный химический потенциал водорода
         muH2ds,  # Характерный химический потенциал растворенного водорода
         dmuH2s,  # Характерное приращение химического потенциала водорода
         dmuH2ds,  # Характерное приращение химического потенциала растворенного водорода
         nuH2s,  # Характерное число молей водорода
         nuH2ds,  # Характерное число молей растворенного водорода
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
         cFElO2,  # Удельная теплоемкость водородно-воздушного топливного элемента по кислороду в мембране
         cFElH2,  # Удельная теплоемкость водородно-воздушного топливного элемента по водороду в мембране
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
         qExtsp,  # Внешний поток теплоты на камеру положительного электрода
         qExtsn,  # Внешний поток теплоты на камеру отрицательного электрода
         nuH2Osm,  # Характерное число молей воды в мембране
         nuH2OsEvp,  # Характерное число молей воды в приэлектродной области и камере положительного электрода
         nuH2OsEvn,  # Характерное число молей воды в приэлектродной области и камере отрицательного электрода
         crRmDiffH2O,  # Коэффициент прекрестности диффузии воды и ионов водородв в мембране
         crEvH20KElp,  # Коэффициент перекрестности испарения воды и теплообмена с камерой положительного электрода
         crEvH20KEln,  # Коэффициент перекрестности испарения воды и теплообмена с камерой отрицательного электрода
         crQKElp,  # Коэффициент перекрестности электродной реакции и теплообмена с камерой положительного электрода
         crQKEln,  # Коэффициент перекрестности электродной реакции и теплообмена с камерой отрицательного электрода
         TOkrs,  # Постоянная температура окружающей среды
         nuH2OStsEp,  # Задаваемое число молей воды в камере положительного электрода
         nuH2OStsEn,  # Задаваемое число молей воды в камере отрицательного электрода
         nuO2Es,  # Задаваемое число молей кислорода в камере положительного электрода
         nuH2Es,  # Задаваемое число молей водорода в камере отрицательного электрода
         qExtH2Osp,  # Удельная теплота отвода с водой из камеры положительного электрода
         qExtH2Osn,  # Удельная теплота отвода с водой из камеры отрицательного электрода
         qExtO2sp,  # Удельная теплота поступления с кислородом из камеры положительного электрода
         qExtH2sn,  # Удельная теплота поступления с водородом из камеры отрицательного электрода
         KDissO2,  # Коэффициент растворения кислорода в камере положительного электрода
         KDissH2,  # Коэффициент растворения водорода в камере отрицательного электрода
         dKQDissO2,  # Коэффициент передачи теплоты в процессе растворения кислорода в камере положительного электрода
         dKQDissH2,  # Коэффициент передачи теплоты в процессе растворения водорода в камере отрицательного электрода
         crQKDissO2,  # Перекрестный по теплоте коэффициент растворения кислорода в камере положительного электрода
         crQKDissH2,  # Перекрестный по теплоте коэффициент растворения водорода в камере отрицательного электрода
         rKDissO2,  # Относительный коэффициент растворения кислорода по нерастворенному кислороду
         rKDissO2d,  # Относительный коэффициент растворения кислорода по растворенному кислороду
         rKDissH2,  # Относительный коэффициент растворения водорода по нерастворенному водороду
         rKDissH2d,  # Относительный коэффициент растворения водорода по растворенному водороду
         KDiffO2,  # Коэффициент диффузии кислорода в мембране
         KDiffH2,  # Коэффициент диффузии водорода в мембране
         dKDiffH2OO2,  # Добавочный коэффициент диффузии воды по кислороду
         dKDiffH2OH2,  # Добавочный коэффициент диффузии воды по водороду
         rKDiffO2H2O,  # Перекрестный коэффициент диффузии кислорода в мембране с водой
         rKDiffH2H2O,  # Перекрестный коэффициент диффузии водорода в мембране с водой
         rKDiffO2p,  # Относительный коэффициент диффузии кислорода в мембране по области положительного электрода
         rKDiffO2n,  # Относительный коэффициент диффузии кислорода в мембране по области отрицательного электрода
         rKDiffH2p,  # Относительный коэффициент диффузии водорода в мембране по области положительного электрода
         rKDiffH2n,  # Относительный коэффициент диффузии водорода в мембране по области отрицательного электрода
         utKH2O2s,  # Коэффициент утилизации кислорода и водорода

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
         betaMuO22,
         betaMuO23,
         betaHO22,
         betaHO23,
         betaMuH22,
         betaMuH23,
         betaHH22,
         betaHH23,
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
         betakUtH2O2,
         betakUtH2O2O2,
         betakUtH2O2H2,
         betaDissO22,
         betaDissO2d2,
         betaDissO2cd2,
         betaDissH22,
         betaDissH2d2,
         betaDissH2cd2,
         betaDiffO22,
         betaDiffO2d2,
         betaDiffO2cd2,
         betaDiffH22,
         betaDiffH2d2,
         betaDiffH2cd2,
         betaMuO2H2O,
         betaMuH2H2O,

         Rkl  # Сопротивление клемм
         ] = systemParameters

        # Определяем отток воды
        evExtH2Op = evExtH2Osp * (nuH2OStp / nuH2OStsEp - 1)
        evExtH2On = evExtH2Osn * (nuH2OStn / nuH2OStsEn - 1)

        # Определяем приток кислорода и водорода
        evExtO2 = -evExtO2s * (nuO2 / nuO2Es - 1)
        evExtH2 = -evExtH2s * (nuH2 / nuH2Es - 1)

        # Выделившаяся джоулева теплота в клеммах
        QKl = Rkl * np.power(I, 2)

        # Потоки теплоты в камеру извне
        qExtp = qExtsp + qExtO2sp * evExtO2 - qExtH2Osp * evExtH2Op
        qExtn = qExtsn + qExtH2sn * evExtH2 - qExtH2Osn * evExtH2On

        # Определяем емкости двойных слоев
        (Cbinp, Cbinn) = funCbin(qbinp, qbinn, alphaCQp, alphaCQn, Cbin0p, Cbin0n,
                                 betaCQ2p, betaCQ2n, betaCQ3p, betaCQ3n)

        # Определяем химический потенциал воды в приэлектродных областях мембраны
        (JSzEl, HSzTEl,
         JSTEl, HSTTEl,
         Ubinp, Um, Ubinn,
         dissUbinp, dissUbinn,
         rNuO2dp, rNuO2dn,
         rNuH2dp, rNuH2dn,
         JSzElH2Op, JSzElH2On,
         JSzElO2p, JSzElO2n,
         JSzElH2p, JSzElH2n) = funJHSzTEl(qbinp, qm, qbinn,
                                          nuH2Op, nuO2dp, nuO2dn,
                                          nuH2On, nuH2dp, nuH2dn, TFEl,
                                          Cbinp, Cm, Cbinn, Econ, THMus,
                                          nuH2Os, nuO2ds, nuH2ds,
                                          muH2Os, muO2ds, muH2ds,
                                          dmuH2Os, dmuO2ds, dmuH2ds,
                                          hH2Os, hO2ds, hH2ds,
                                          dhH2Os, dhO2ds, dhH2ds,
                                          betaMuH2O2, betaMuH2O3,
                                          betaHH2O2, betaHH2O3,
                                          betaMuO22, betaMuO23,
                                          betaHO22, betaHO23,
                                          betaMuH22, betaMuH23,
                                          betaHH22, betaHH23,
                                          betaMuO2H2O, betaMuH2H2O,
                                          cFElH2O, cFElO2, cFElH2, CFEls)
        self.__Ubinp = Ubinp
        self.__Um = Um
        self.__Ubinn = Ubinn

        # Определяем химические потенциалы кислорода, водорода и воды в камерах электродов при стандартной температуре
        (JSzCamp, HSzTCamp,
         JSTCamp, HSTTCamp,
         rNuO2) = funJHSzTCam(nuH2OStp, nuO2, TElp,
                              nuH2OStsp, nuO2s, THMus,
                              muH2OStsp, dmuH2OStsp, muO2s, dmuO2s,
                              hH2OStsp, dhH2OStsp, hO2s, dhO2s,
                              betaMuH2OStp, betaMuH2OStO2p, betaMuO2p,
                              betaHH2OStp, betaHH2OStO2p, betaHO2p,
                              cElH2OStp, cElO2p, CElsp)  # Определяем химические потенциалы кислорода и воды в камере положительного электрода
        (JSzCamn, HSzTCamn,
         JSTCamn, HSTTCamn,
         rNuH2) = funJHSzTCam(nuH2OStn, nuH2, TEln,
                              nuH2OStsn, nuH2s, THMus,
                              muH2OStsn, dmuH2OStsn, muH2s, dmuH2s,
                              hH2OStsn, dhH2OStsn, hH2s, dhH2s,
                              betaMuH2OStn, betaMuH2OStH2n, betaMuH2n,
                              betaHH2OStn, betaHH2OStH2n, betaHH2n,
                              cElH2OStn, cElH2n, CElsn)  # Определяем химические потенциалы кислорода и воды в камере отрицательного электрода

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
        kInvMatrixElEvs = np.array([0, 0, 0, 1], dtype=np.double).reshape(-1, 1)
        kInvMatrixElpEchCr = (np.sqrt(np.array([1 / Rbin0p, 0, 0, dKElTQp0],
                                               dtype=np.double)) * np.array([1, 1, 1, crQKElp],
                                                                            dtype=np.double)).reshape(-1, 1)
        kInvMatrixElpEvCr = (np.sqrt(np.array([0, kEvH2Osp, 0, dKElTEvp0],
                                              dtype=np.double)) * np.array([1, 1, 1, crEvH20KElp],
                                                                           dtype=np.double)).reshape(-1, 1)
        kInvMatrixElpEvDiss = (np.sqrt(np.array([0, 0, KDissO2, dKQDissO2],
                                                 dtype=np.double)) * np.array([1, 1, 1, crQKDissO2],
                                                                              dtype=np.double)).reshape(-1, 1)
        rKDissO2 = funKDiss(rNuO2, rNuO2dp,
                            rKDissO2, rKDissO2d,
                            betaDissO22,
                            betaDissO2d2,
                            betaDissO2cd2)
        kNoInvMatrixElp = np.array([KElTop * Tokr / TOkrs + \
                                    dKElTQp0 * (Tokr / TOkrs - np.power(crQKElp, 2)) * sbinp + \
                                    dKElTEvp0 * (Tokr / TOkrs - np.power(crEvH20KElp, 2)) * kbinp + \
                                    dKQDissO2 * (Tokr / TOkrs - np.power(crQKDissO2, 2)) * rKDissO2,
                                    sbinp, kbinp, rKDissO2], dtype=np.double) * Tokr / (4.642 * NonEqSystemQBase.GetTbase())

        # Определяем обратимые и необратимые составляющие кинетической матрицы отрицательной камеры
        kInvMatrixElnEchCr = (np.sqrt(np.array([1 / Rbin0n, 0, 0, dKElTQn0],
                                               dtype=np.double)) * np.array([1, 1, 1, crQKEln],
                                                                            dtype=np.double)).reshape(-1, 1)
        kInvMatrixElnEvCr = (np.sqrt(np.array([0, kEvH2Osn, 0, dKElTEvn0],
                                              dtype=np.double)) * np.array([1, 1, 1, crEvH20KEln],
                                                                           dtype=np.double)).reshape(-1, 1)
        kInvMatrixElnEvDiss = (np.sqrt(np.array([0, 0, KDissH2, dKQDissH2],
                                                 dtype=np.double)) * np.array([1, 1, 1, crQKDissH2],
                                                                              dtype=np.double)).reshape(-1, 1)
        rKDissH2 = funKDiss(rNuH2, rNuH2dn,
                            rKDissH2, rKDissH2d,
                            betaDissH22,
                            betaDissH2d2,
                            betaDissH2cd2)
        kNoInvMatrixEln = np.array([KElTon * Tokr / TOkrs + \
                                    dKElTQn0 * (Tokr / TOkrs - np.power(crQKEln, 2)) * sbinn + \
                                    dKElTEvn0 * (Tokr / TOkrs - np.power(crEvH20KEln, 2)) * kbinn + \
                                    dKQDissH2 * (Tokr / TOkrs - np.power(crQKDissH2, 2)) * rKDissH2,
                                    sbinn, kbinn, rKDissH2], dtype=np.double) * Tokr / (4.642 * NonEqSystemQBase.GetTbase())

        # Определяем обратимые и необратимые составляющие кинетической матрицы мембраны
        kInvMatrixElmDiffs = np.array([0, 0, 0, 1], dtype=np.double).reshape(-1, 1)
        kInvMatrixElmCr = (np.sqrt(np.array([1 / Rm0, 0, 0, dKDiffH2O0],
                                            dtype=np.double)) * np.array([1, 1, 1, crRmDiffH2O],
                                                                         dtype=np.double)).reshape(-1, 1)
        kInvMatrixElmDiffO2 = (np.sqrt(np.array([0, KDiffO2, 0, dKDiffH2OO2],
                                                dtype=np.double)) * np.array([1, 1, 1, rKDiffO2H2O],
                                                                             dtype=np.double)).reshape(-1, 1)
        kInvMatrixElmDiffH2 = (np.sqrt(np.array([0, 0, KDiffH2, dKDiffH2OH2],
                                                dtype=np.double)) * np.array([1, 1, 1, rKDiffH2H2O],
                                                                             dtype=np.double)).reshape(-1, 1)
        rKDiffO2 = funKDiss(rNuO2dp, rNuO2dn,
                            rKDiffO2p, rKDiffO2n,
                            betaDiffO22,
                            betaDiffO2d2,
                            betaDiffO2cd2)
        rKDiffH2 = funKDiss(rNuH2dp, rNuH2dn,
                            rKDiffH2p, rKDiffH2n,
                            betaDiffH22,
                            betaDiffH2d2,
                            betaDiffH2cd2)
        kNoInvMatrixElm = np.array([kDiffH2O0 + \
                                    dKDiffH2O0 * (1 - np.power(crRmDiffH2O, 2)) * sm + \
                                    dKDiffH2OO2 * (1 - np.power(rKDiffO2H2O, 2)) * rKDiffO2 + \
                                    dKDiffH2OH2 * (1 - np.power(rKDiffH2H2O, 2)) * rKDiffH2,
                                    sm, rKDiffO2, rKDiffH2], dtype=np.double) * Tokr / (4.642 * NonEqSystemQBase.GetTbase())

        # Коэффициенты утилизации кислорода и водорода в мембране
        rUtKH2O2p = funUtKH2O2(rNuO2dp, rNuH2dp,
                               -JSzElH2Op, -JSzElH2p, -JSzElO2p,
                               betakUtH2O2,
                               betakUtH2O2O2,
                               betakUtH2O2H2)
        rUtKH2O2n = funUtKH2O2(rNuO2dn, rNuH2dn,
                               -JSzElH2On, -JSzElH2n, -JSzElO2n,
                               betakUtH2O2,
                               betakUtH2O2O2,
                               betakUtH2O2H2)
        utKH2O2 = utKH2O2s * np.array([rUtKH2O2p, rUtKH2O2n], dtype=np.double)

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
                kInvMatrixElpEvDiss,
                kInvMatrixElnEvDiss,
                kInvMatrixElmDiffO2,
                kInvMatrixElmDiffH2,
                utKH2O2, kQOkr, I, Tokr,
                qExtp, qExtn)

    # Выводим напряжение положительного двойного слоя
    def GetUbinp(self):
        return self.__Ubinp

    # Выводим напряжение ммбораны
    def GetUm(self):
        return self.__Um

    # Выводим напряжение отрицательного двойного слоя
    def GetUbinn(self):
        return self.__Ubinn
