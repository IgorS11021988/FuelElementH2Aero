import numpy as np

from MathProtEnergyProcSynDatas.TimesMoments import LinearTimesMoments
from MathProtEnergyProcSynDatas.Indicate import PlotGraphicIndicate, SaveDynamicToFileIndicate
from MathProtEnergyProcSynDatas.File import DynamicSaveAndSaveGraphics

from MathProtEnergyProc.CorrectionModel import ReluFilter

from .fU import UParametersSystemParametersNames, otherSystemParametersNames
from .StationFunction import stateCoordinatesNames, reducedTemperaturesEnergyPowersNames


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
    Pars[["nuH2OStp", "nuH2OStn", "nuO2", "nuH2"]] *= Pars[["nuH2OStsEp", "nuH2OStsEn", "nuO2Es", "nuH2Es"]].to_numpy()  # Корректируем начальное числа молей воды
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

    # Массив параметров
    USystemParametersNames = UParametersSystemParametersNames + otherSystemParametersNames
    systemParameters = Pars[USystemParametersNames].to_numpy()

    # Массив начальных состояний
    stateCoordinates0 = Pars[stateCoordinatesNames].to_numpy()
    reducedTemp0 = Pars[reducedTemperaturesEnergyPowersNames].to_numpy()

    #  Моменты времени
    Tints = integrateAttributes["Tint"].to_numpy()  # Времена интегрирования
    NPoints = np.array(integrateAttributes["NPoints"], dtype=np.int32)  # Числа точек интегрирования
    ts = LinearTimesMoments(Tints,  # Времена интегрирования
                            NPoints  # Числа точек интегрирования
                            )

    # Возвращаем исходные данные динамики системы
    return (Tints,
            stateCoordinates0,
            reducedTemp0,
            systemParameters,
            ts)


# Обработка результатов моделирования динамик
def OutputValues(dyns, fileName,
                 sep, dec, index,
                 plotGraphics=False  # Необходимость построения графиков
                 ):
    # Получаем величины из кортежа
    (t, Ukl, Ubinp, Ubinn, Um,
     TFEl, TElp, TEln,
     qH2Op, qH2On, qH2OStp, qH2OStn,
     qO2, qH2) = dyns

    # Заголовки и динамики
    dynamicsHeaders = {"Time": t,
                       "Ukl": Ukl,
                       "Ubinp": Ubinp,
                       "Ubinn": Ubinn,
                       "Um": Um,
                       "TFEl": TFEl,
                       "TElp": TElp,
                       "TEln": TEln,
                       "qH2Op": qH2Op,
                       "qH2On": qH2On,
                       "qH2OStp": qH2OStp,
                       "qH2OStn": qH2OStn,
                       "qO2": qO2,
                       "qH2": qH2
                       }

    # Одиночные графики на полотне
    oneTimeValueGraphics = [{"values": Ukl,  # Величины в моменты времени
                             "graphName": "Напряжение на клеммах",  # Имя полотна
                             "yAxesName": "Напряжение, В",  # Имя оси ординат
                             "graphFileBaseName": "ElVoltage"  # Имя файла графика
                             }]

    # Группы графиков на полотне
    timesValuesGraphics = [{"listValues": [TEln, TElp, TFEl],  # Список величин в моменты времени
                            "listValuesNames": ["Камера положительного электрода",
                                                "Камера отрицательного электрода",
                                                "Элемент"],  # Список имен величин (в моменты времени)
                            "graphName": "Температуры топливного элемента",  # Имя полотна
                            "yAxesName": "Температура, град С",  # Имя оси
                            "graphFileBaseName": "ElTemperatures"  # Имя файла графика
                            },

                           {"listValues": [Ubinn, Ubinp, Um],  # Список величин в моменты времени
                            "listValuesNames": ["Отрицательный двойной слой",
                                                "Положительный двойной слой",
                                                "Мембрана"],  # Список имен величин (в моменты времени)
                            "graphName": "Напряжения в топливном элементе",  # Имя полотна
                            "yAxesName": "Напряжение, В",  # Имя оси
                            "graphFileBaseName": "InElVoltages"  # Имя файла графика
                            },

                           {"listValues": [qH2On, qH2Op],  # Список величин в моменты времени
                            "listValuesNames": ["Отрицательный электрод",
                                                "Положительный электрод"],  # Список имен величин (в моменты времени)
                            "graphName": "Количество воды в приэлектродных областях",  # Имя полотна
                            "yAxesName": "Зарядовое число молей воды, Кл",  # Имя оси
                            "graphFileBaseName": "InElWaterMoles"  # Имя файла графика
                            },

                           {"listValues": [qH2OStn, qH2OStp],  # Список величин в моменты времени
                            "listValuesNames": ["Отрицательный электрод",
                                                "Положительный электрод"],  # Список имен величин (в моменты времени)
                            "graphName": "Количество воды в электродных камерах",  # Имя полотна
                            "yAxesName": "Зарядовое число молей воды, Кл",  # Имя оси
                            "graphFileBaseName": "CamElWaterMoles"  # Имя файла графика
                            },

                           {"listValues": [qH2, qO2],  # Список величин в моменты времени
                            "listValuesNames": ["Водород",
                                                "Кислород"],  # Список имен величин (в моменты времени)
                            "graphName": "Количество газов в электродных камерах",  # Имя полотна
                            "yAxesName": "Зарядовое число молей газа, Кл",  # Имя оси
                            "graphFileBaseName": "ElGases"  # Имя файла графика
                            }]

    # Сохраняем динамику в .csv файл и отображаем графики
    DynamicSaveAndSaveGraphics(dynamicsHeaders,  # Словарь динамик с заголовками
                               fileName,  # Имя файла динамик

                               t,  # Моменты времени
                               oneTimeValueGraphics,  # Один график на одном полотне
                               timesValuesGraphics,  # Несколько графиков на одном полотне

                               plotGraphics,  # Необходимость построения графиков

                               sep, dec,   # Разделители (csv и десятичный соответственно)

                               saveDynamicIndicator=SaveDynamicToFileIndicate,  # Индикатор сохранения динамики
                               saveGraphicIndicator=PlotGraphicIndicate,  # Индикатор отображения графиков
                               index=index  # Индекс динамики
                               )
