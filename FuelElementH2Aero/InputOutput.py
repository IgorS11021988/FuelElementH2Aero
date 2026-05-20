import numpy as np

from MathProtEnergyProcSynDatas.TimesMoments import LinearTimesMoments
from MathProtEnergyProcSynDatas.Indicate import PlotGraphicIndicate, SaveDynamicToFileIndicate
from MathProtEnergyProcSynDatas.File import DynamicSaveAndSaveGraphics

from MathProtEnergyProc.CorrectionModel import ReluFilter


# Корректировка перекрестных коэффициентов
def crCfCorr(Pars,  # Параметры

             crCfName  # Имя перекрестного коэффициента
             ):
    # Получаем индексы перекрестных коэффициентов, больших 1
    gtParsBInd = (Pars[crCfName] > 1)

    # Корректируем перекрестные коэффициенты, большие 1
    Pars.loc[gtParsBInd, crCfName] = 1

    # Получаем индексы перекрестных коэффициентов, меньших -1
    ltParsBInd = (Pars[crCfName] < -1)

    # Корректируем перекрестные коэффициенты, меньшие -1
    Pars.loc[ltParsBInd, crCfName] = -1


# Функция расчета динамики
def InputArrayCreate(Pars,  # Параметры

                     integrateAttributes  # Аттрибуты интегрирования
                     ):  # Формирование массивов входных параметров
    # Корректируем начальное состояние
    Pars[["nuH2OStp", "nuH2OStn", "nuO2", "nuH2", "nuO2dp", "nuO2dn", "nuH2dp", "nuH2dn"]] *= Pars[["nuH2OStsEp", "nuH2OStsEn", "nuO2Es", "nuH2Es", "nuO2ds", "nuO2ds", "nuH2ds", "nuH2ds"]].to_numpy()  # Корректируем начальное числа молей воды
    Pars[["TFEl", "TElp", "TEln"]] += Pars[["Tokr"]].to_numpy()  # Корректируем начальные температуры, заданные относительно температуры окружающей среды
    Pars["qbinp"] *= (Pars["muO2s"] / 4 - Pars["muH2Os"] / 2 + Pars["Econ"]) * Pars["Cbin0p"]  # Заряд на положительном двойном слое, Кл
    Pars["qbinn"] *= (Pars["muH2s"] / 2 - Pars["Econ"]) * Pars["Cbin0n"]  # Заряд на отрицательном двойном слое, Кл

    # Корректируем главные кинетические коэффициенты
    mainKinCf = ["kEvH2Osp",
                 "kEvH2Osn",
                 "dKElTEvp0",
                 "dKElTEvn0",
                 "Rm0",
                 "kDiffH2O0",
                 "dKDiffH2O0",
                 "Rbin0p",
                 "Rbin0n",
                 "dKElTQp0",
                 "dKElTQn0"]
    Pars[mainKinCf] = ReluFilter(Pars[mainKinCf])

    # Корректируем перекрестные коэффициенты
    crCfCorr(Pars, "crQKElp")
    crCfCorr(Pars, "crQKEln")
    crCfCorr(Pars, "crRmDiffH2O")
    crCfCorr(Pars, "crEvH20KElp")
    crCfCorr(Pars, "crEvH20KEln")

    # Переводим температуру в кельвины
    Pars[["TFEl", "TElp", "TEln", "Tokr", "THMus", "bRTp", "bRTm", "bRTn", "bTKEvH2Osp", "bTKEvH2Osn"]] += 273.15

    #  Моменты времени
    Tints = integrateAttributes["Tint"].to_numpy()  # Времена интегрирования
    NPoints = np.array(integrateAttributes["NPoints"], dtype=np.int32)  # Числа точек интегрирования
    ts = LinearTimesMoments(Tints,  # Времена интегрирования
                            NPoints  # Числа точек интегрирования
                            )

    # Возвращаем исходные данные динамики системы
    return (Pars, Tints, ts)


# Обработка результатов моделирования динамик
def OutputValues(dyns, index,
                 saveDynamicFun):
    # Получаем величины из кортежа
    (t, Ukl, Ubinp, Ubinn, Um,
     TFEl, TElp, TEln, Icur, Ibinp, Im, Ibinn,
     qH2Op, qH2On, qH2OStp, qH2OStn,
     qO2, qH2, qO2dp, qO2dn, qH2dp, qH2dn,
     vUtFuelp, vUtFueln) = dyns

    # Заголовки и динамики
    dynamicsHeaders = {"Time": t,
                       "Ukl": Ukl,
                       "Ubinp": Ubinp,
                       "Ubinn": Ubinn,
                       "Um": Um,
                       "Ibinp": Ibinp,
                       "Ibinn": Ibinn,
                       "Im": Im,
                       "I": Icur,
                       "TFEl": TFEl,
                       "TElp": TElp,
                       "TEln": TEln,
                       "qH2Op": qH2Op,
                       "qH2On": qH2On,
                       "qH2OStp": qH2OStp,
                       "qH2OStn": qH2OStn,
                       "qO2": qO2,
                       "qH2": qH2,
                       "qO2dp": qO2dp,
                       "qH2dp": qH2dp,
                       "qO2dn": qO2dn,
                       "qH2dn": qH2dn,
                       "vUtFuelp": vUtFuelp,
                       "vUtFueln": vUtFueln
                       }

    # Одиночные графики на полотне
    oneTimeValueGraphics = [{"values": Ukl,  # Величины в моменты времени
                             "graphName": "Напряжение на клеммах",  # Имя полотна
                             "yAxesName": "Напряжение, В",  # Имя оси ординат
                             "graphFileBaseName": "ElVoltage",  # Имя файла графика
                             "color": (0, 0, 0)  # Цвет в RGB-шкале
                             }]

    # Группы графиков на полотне
    timesValuesGraphics = [{"listValues": [TElp, TEln, TFEl],  # Список величин в моменты времени
                            "listValuesNames": ["Камера положительного электрода",
                                                "Камера отрицательного электрода",
                                                "Элемент"],  # Список имен величин (в моменты времени)
                            "graphName": "Температуры топливного элемента",  # Имя полотна
                            "yAxesName": "Температура, град С",  # Имя оси
                            "graphFileBaseName": "ElTemperatures",  # Имя файла графика
                            "color": [(1, 0, 0), (0, 0, 1), (0, 1, 0)]  # Цвет в RGB-шкале
                            },

                           {"listValues": [Ubinp, Ubinn, Um],  # Список величин в моменты времени
                            "listValuesNames": ["Положительный двойной слой",
                                                "Отрицательный двойной слой",
                                                "Мембрана"],  # Список имен величин (в моменты времени)
                            "graphName": "Напряжения в топливном элементе",  # Имя полотна
                            "yAxesName": "Напряжение, В",  # Имя оси
                            "graphFileBaseName": "InElVoltages",  # Имя файла графика
                            "color": [(1, 0, 0), (0, 0, 1), (0, 1, 0)]  # Цвет в RGB-шкале
                            },

                           {"listValues": [Ibinp, Ibinn, Im, Icur],  # Список величин в моменты времени
                            "listValuesNames": ["Положительный двойной слой",
                                                "Отрицательный двойной слой",
                                                "Мембрана",
                                                "Ток во внешней цепи"],  # Список имен величин (в моменты времени)
                            "graphName": "Токи в топливном элементе",  # Имя полотна
                            "yAxesName": "Ток, А",  # Имя оси
                            "graphFileBaseName": "ElCur",  # Имя файла графика
                            "color": [(1, 0, 0), (0, 0, 1), (0, 1, 0), (0.21, 0.51, 0)]  # Цвет в RGB-шкале
                            },

                           {"listValues": [qH2Op, qH2On],  # Список величин в моменты времени
                            "listValuesNames": ["Положительный электрод",
                                                "Отрицательный электрод"],  # Список имен величин (в моменты времени)
                            "graphName": "Количество воды в приэлектродных областях",  # Имя полотна
                            "yAxesName": "Зарядовое число молей воды, Кл",  # Имя оси
                            "graphFileBaseName": "InElWaterMoles",  # Имя файла графика
                            "color": [(1, 0, 0), (0, 0, 1)]  # Цвет в RGB-шкале
                            },

                           {"listValues": [qH2OStp, qH2OStn],  # Список величин в моменты времени
                            "listValuesNames": ["Положительный электрод",
                                                "Отрицательный электрод"],  # Список имен величин (в моменты времени)
                            "graphName": "Количество воды в электродных камерах",  # Имя полотна
                            "yAxesName": "Зарядовое число молей воды, Кл",  # Имя оси
                            "graphFileBaseName": "CamElWaterMoles",  # Имя файла графика
                            "color": [(1, 0, 0), (0, 0, 1)]  # Цвет в RGB-шкале
                            },

                           {"listValues": [qH2, qO2],  # Список величин в моменты времени
                            "listValuesNames": ["Водород",
                                                "Кислород"],  # Список имен величин (в моменты времени)
                            "graphName": "Количество газов в электродных камерах",  # Имя полотна
                            "yAxesName": "Зарядовое число молей газа, Кл",  # Имя оси
                            "graphFileBaseName": "ElGases",  # Имя файла графика
                            "color": [(0, 0, 1), (1, 0, 0)]  # Цвет в RGB-шкале
                            },

                           {"listValues": [qO2dp, qO2dn],  # Список величин в моменты времени
                            "listValuesNames": ["Положительный электрод",
                                                "Отрицательный электрод"],  # Список имен величин (в моменты времени)
                            "graphName": "Количество растворенного кислорода в приэлектродных областях",  # Имя полотна
                            "yAxesName": "Зарядовое число молей кислорода, Кл",  # Имя оси
                            "graphFileBaseName": "ElO2d",  # Имя файла графика
                            "color": [(1, 0, 0), (0, 0, 1)]  # Цвет в RGB-шкале
                            },

                           {"listValues": [qH2dp, qH2dn],  # Список величин в моменты времени
                            "listValuesNames": ["Положительный электрод",
                                                "Отрицательный электрод"],  # Список имен величин (в моменты времени)
                            "graphName": "Количество растворенного водорода в приэлектродных областях",  # Имя полотна
                            "yAxesName": "Зарядовое число молей водорода, Кл",  # Имя оси
                            "graphFileBaseName": "ElH2d",  # Имя файла графика
                            "color": [(1, 0, 0), (0, 0, 1)]  # Цвет в RGB-шкале
                            },

                           {"listValues": [vUtFuelp, vUtFueln],  # Список величин в моменты времени
                            "listValuesNames": ["Положительный электрод",
                                                "Отрицательный электрод"],  # Список имен величин (в моменты времени)
                            "graphName": "Скорость утилизации топлива в приэлектродных областях",  # Имя полотна
                            "yAxesName": "Зарядовая cкорость утилизации топлива, А",  # Имя оси
                            "graphFileBaseName": "vUtFuel",  # Имя файла графика
                            "color": [(1, 0, 0), (0, 0, 1)]  # Цвет в RGB-шкале
                            }]

    # Сохраняем динамику в .csv файл и отображаем графики
    DynamicSaveAndSaveGraphics(dynamicsHeaders,  # Словарь динамик с заголовками
                               saveDynamicFun,  # Имя функции сохранения динамики

                               t,  # Моменты времени
                               oneTimeValueGraphics,  # Один график на одном полотне
                               timesValuesGraphics,  # Несколько графиков на одном полотне

                               showGraphics=True,  # Необходимость отображения графиков

                               saveDynamicIndicator=SaveDynamicToFileIndicate,  # Индикатор сохранения динамики
                               saveGraphicIndicator=PlotGraphicIndicate,  # Индикатор отображения графиков

                               index=index  # Индекс динамики
                               )


# Обработка результатов оптимизационного моделирования динамик
def OutputValuesOptimize(dyns, index,
                         saveDynamicFun):
    # Получаем величины из кортежа
    (t, Ukl, TFEl, TElp, TEln, Icur,
     qH2OStp, qH2OStn, qO2, qH2) = dyns

    # Заголовки и динамики
    dynamicsHeaders = {"Time": t,
                       "Ukl": Ukl,
                       "TFEl": TFEl,
                       "TElp": TElp,
                       "TEln": TEln,
                       "qH2OStp": qH2OStp,
                       "qH2OStn": qH2OStn,
                       "qO2": qO2,
                       "qH2": qH2
                       }

    # Сохраняем динамику
    print("Dynamic: " + str(index))
    saveDynamicFun.SaveDynamic(dynamicsHeaders, index)
