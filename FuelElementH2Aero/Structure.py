import numpy as np

from .StationFunction import IndepStateFunction
from .AttributesNames import stateCoordinatesNames, reducedTemperaturesEnergyPowersNames, processCoordinatesNames

from MathProtEnergyProc.CorrectionModel import KineticMatrixQ, KineticMatrixFromPosSubMatrix, CreateBlockMatrix
from MathProtEnergyProc.HeatPowerValues import IntPotentialsOne, HeatValuesOne

from MathProtEnergyProc.CorrectionModel import ReluFilter, PosLinearFilter


# Функция структуры аккумулятора
def StructureFunction():
    # Описываем структуру водородно-воздушного топливного элемента
    energyPowersNames = ["EnPowFEl", "EnPowElp", "EnPowEln", "EnPowOkr"]  # Имена энергетических степеней свободы
    energyPowersBetNames = []  # Имена взаимодействий между энергетическими степенями свободы
    heatTransfersNames = ["Qexp", "QFElp", "QFEln", "Qexpp", "Qexpn"]  # Имена потоков переноса теплоты
    heatTransfersOutputEnergyPowersNames = ["EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowElp", "EnPowEln"]  # Имена энергетических степеней свободы, с которых уходит теплота
    heatTransfersInputEnergyPowersNames = ["EnPowOkr", "EnPowElp", "EnPowEln", "EnPowOkr", "EnPowOkr"]  # Имена энергетических степеней свободы, на которые приходит теплота
    stateCoordinatesStreamsNames = ["qbinp", "qm", "qbinn", "nuH2OStp", "nuH2OStn", "nuO2", "nuH2"]  # Имена координат состояния, изменяемых в результате внешних потоков
    heatEnergyPowersStreamsNames = ["EnPowFEl", "EnPowElp", "EnPowEln"]  # Имена потоков теплоты на энергетические степени свободы
    stateCoordinatesVarBalanceNames = []  # Имена переменных коэффициентов матрицы баланса по координатам состояния
    processCoordinatesVarBalanceNames = []  # Имена переменных коэффициентов матрицы баланса по координатам процессов
    energyPowersVarTemperatureNames = ["EnPowFEl", "EnPowElp", "EnPowEln", "EnPowOkr"]  # Имена переменных температур энергетических степеней свободы
    stateCoordinatesVarPotentialsInterNames = ["qbinp", "qm", "qbinn", "nuH2Op", "nuH2On", "nuH2OStp", "nuH2OStn", "nuO2", "nuH2", "nuO2dp", "nuO2dn", "nuH2dp", "nuH2dn"]  # Имена переменных потенциалов взаимодействия по координатам состояния
    energyPowersVarPotentialsInterNames = ["EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowElp", "EnPowEln", "EnPowElp", "EnPowEln", "EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowFEl"]  # Имена переменных потенциалов взаимодействия по энергетическим степеням свободы
    stateCoordinatesVarPotentialsInterBetNames = []  # Имена переменных потенциалов взаимодействия для взаимодействий между энергетическими степенями свободы по координатам состояния
    energyPowersVarPotentialsInterBetNames = []  # Имена переменных потенциалов взаимодействия для взаимодействий между энергетическими степенями свободы по энергетическим степеням свободы
    energyPowersVarBetaNames = []  # Имена переменных долей распределения некомпенсированной теплоты энергетических степеней свободы
    processCoordinatesVarBetaNames = []  # Имена переменных долей распределения некомпенсированной теплоты координат процессов
    reducedTemperaturesEnergyPowersVarInvHeatCapacityNames = ["TFEl", "TElp", "TEln"]  # Имена переменных коэффициентов обратных теплоемкостей по отношению к приведенным температурам
    energyPowersVarInvHeatCapacityNames = ["EnPowFEl", "EnPowElp", "EnPowEln"]  # Имена переменных коэффициентов обратных теплоемкостей по отношению к энергетическим степеням свободы
    reducedTemperaturesEnergyPowersVarHeatEffectNames = ["TFEl", "TFEl", "TFEl", "TFEl", "TFEl", "TElp", "TEln", "TElp", "TEln", "TFEl", "TFEl", "TFEl", "TFEl"]  # Имена переменных коэффициентов обратных теплоемкостей по отношению к приведенным температурам
    stateCoordinatesVarHeatEffectNames = ["qbinp", "qm", "qbinn", "nuH2Op", "nuH2On", "nuH2OStp", "nuH2OStn", "nuO2", "nuH2", "nuO2dp", "nuO2dn", "nuH2dp", "nuH2dn"]  # Имена переменных коэффициентов обратных теплоемкостей по отношению к координатам состояния
    varKineticPCPCNames = ["dqbinp", "evH2Op", "dqbinn", "evH2On", "dissO2p", "dissH2n", "dqm", "diffH2O", "diffO2m", "diffH2m", "dqm", "diffO2m", "diffH2m", "diffH2O", "diffH2O", "diffH2O", "utFuelp", "utFueln"]  # Имена сопряженностей между собой координат процессов
    varKineticPCPCAffNames = ["dqbinp", "evH2Op", "dqbinn", "evH2On", "dissO2p", "dissH2n", "dqm", "diffH2O", "diffO2m", "diffH2m", "diffH2O", "diffH2O", "diffH2O", "dqm", "diffO2m", "diffH2m", "utFuelp", "utFueln"]  # Имена сопряженностей между собой термодинамических сил
    varKineticPCHeatNames = ["evH2Op", "dqbinp", "evH2On", "dqbinn", "dissO2p", "dissH2n"]  # Имена сопряженностей координат процессов с теплопереносами
    varKineticPCHeatAffNames = ["QFElp", "QFElp", "QFEln", "QFEln", "QFElp", "QFEln"]  # Имена сопряженностей термодинамических сил с теплопереносами
    varKineticHeatPCNames = ["QFElp", "QFElp", "QFEln", "QFEln", "QFElp", "QFEln"]  # Имена сопряженностей теплопереносов с координатами процессов
    varKineticHeatPCAffNames = ["evH2Op", "dqbinp", "evH2On", "dqbinn", "dissO2p", "dissH2n"]  # Имена сопряженностей теплопереносов с термодинамическими силами
    varKineticHeatHeatNames = ["QFElp", "QFEln", "Qexp", "Qexpp", "Qexpn"]  # Имена сопряженностей между собой перенесенных теплот
    varKineticHeatHeatAffNames = ["QFElp", "QFEln", "Qexp", "Qexpp", "Qexpn"]  # Имена сопряженностей между собой термодинамических сил по переносу теплот
    stateCoordinatesVarStreamsNames = ["qbinp", "qm", "qbinn", "nuH2OStp", "nuH2OStn", "nuO2", "nuH2"]  # Имена переменных внешних потоков
    heatEnergyPowersVarStreamsNames = ["EnPowFEl", "EnPowElp", "EnPowEln"]  # Имена переменных внешних потоков теплоты

    # Коэффициенты кинетической матрицы
    kinMatrixEl = KineticMatrixQ(["dqbinp", "evH2Op", "dqbinn", "evH2On", "dissO2p", "dissH2n", "dqm", "diffH2O", "diffO2m", "diffH2m",     "dqm", "diffO2m", "diffH2m", "diffH2O", "diffH2O", "diffH2O"],  # Имена сопряженностей между собой координат процессов
                                 ["dqbinp", "evH2Op", "dqbinn", "evH2On", "dissO2p", "dissH2n", "dqm", "diffH2O", "diffO2m", "diffH2m", "diffH2O", "diffH2O", "diffH2O",     "dqm", "diffO2m", "diffH2m"],  # Имена сопряженностей между собой термодинамических сил
                                 ["evH2Op", "dqbinp", "evH2On", "dqbinn", "dissO2p", "dissH2n"],  # Имена сопряженностей координат процессов с теплопереносами
                                 [ "QFElp",  "QFElp",  "QFEln",  "QFEln",   "QFElp",   "QFEln"],  # Имена сопряженностей термодинамических сил с теплопереносами
                                 [ "QFElp",  "QFElp",  "QFEln",  "QFEln",   "QFElp",   "QFEln"],  # Имена сопряженностей теплопереносов с координатами процессов
                                 ["evH2Op", "dqbinp", "evH2On", "dqbinn", "dissO2p", "dissH2n"],  # Имена сопряженностей теплопереносов с термодинамическими силами
                                 ["QFElp", "QFEln"],  # Имена сопряженностей между собой перенесенных теплот
                                 ["QFElp", "QFEln"],  # Имена сопряженностей между собой термодинамических сил по переносу теплот

                                 [["dqbinp", "evH2Op", "dissO2p", "QFElp"],
                                  ["dqbinn", "evH2On", "dissH2n", "QFEln"],
                                  ["dqm", "diffO2m", "diffH2m", "diffH2O"]]  # Массив массивов имен координат процессов (в том числе и перенесенных теплот) по кинетической матрице
                                 )  # Кинетическая матрица по водородно-воздушному топливному элементу

    # Потенциалы взаимодействия в топливном элементе и камерах
    potentialInterElAll = IntPotentialsOne(["qbinp", "qm", "qbinn", "nuH2Op", "nuH2On", "nuO2dp", "nuO2dn", "nuH2dp", "nuH2dn", "nuH2OStp", "nuO2", "nuH2OStn", "nuH2"],  # Имена координат состояния
                                           ["EnPowFEl", "EnPowElp", "EnPowEln"],  # Имена энергетических степеней свободы

                                           [   "qbinp",       "qm",    "qbinn",   "nuH2Op",   "nuH2On", "nuH2OStp", "nuH2OStn",     "nuO2",     "nuH2",   "nuO2dp",   "nuO2dn",   "nuH2dp",   "nuH2dn"],  # Имена переменных потенциалов взаимодействия по координатам состояния
                                           ["EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowElp", "EnPowEln", "EnPowElp", "EnPowEln", "EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowFEl"]  # Имена переменных потенциалов взаимодействия по энергетическим степеням свободы
                                           )

    # Приведенные обратные теплоемкости и тепловые эффекты
    heatValuesElAll = HeatValuesOne(["qbinp", "qm", "qbinn", "nuH2Op", "nuH2On", "nuO2dp", "nuO2dn", "nuH2dp", "nuH2dn", "nuH2OStp", "nuO2", "nuH2OStn", "nuH2"],  # Имена координат состояния
                                    ["EnPowFEl", "EnPowElp", "EnPowEln"],  # Имена энергетических степеней свободы

                                    ["EnPowFEl", "EnPowElp", "EnPowEln"],  # Имена переменных коэффициентов обратных теплоемкостей по отношению к энергетическим степеням свободы
                                    ["EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowElp", "EnPowEln", "EnPowElp", "EnPowEln", "EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowFEl"],  # Имена переменных коэффициентов обратных теплоемкостей по отношению к приведенным температурам
                                    [   "qbinp",       "qm",    "qbinn",   "nuH2Op",   "nuH2On", "nuH2OStp", "nuH2OStn",     "nuO2",     "nuH2",   "nuO2dp",   "nuO2dn",   "nuH2dp",   "nuH2dn"]  # Имена переменных коэффициентов обратных теплоемкостей по отношению к координатам состояния
                                    )

    # Функция состояния для литий-ионного аккумулятора
    class StateFunction(object):
        # Инициализация класса
        def __init__(self,

                     indepStateFunction  # Независимая функция состояния
                     ):
            # Заполняем поле
            self.__indepStateFunction = indepStateFunction  # Независимая функция состояния

        # Выводим независимую функцию состояния
        def GetIndepStateFunction(self):
            return self.__indepStateFunction

        # Тело функтора
        def __call__(self,

                     stateCoordinates,
                     reducedTemp,
                     systemParameters):
            # Получаем независимые составляющие свойств веществ и процессов
            (evExtH2Op, evExtH2On,
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
             qExtp, qExtn) = self.__indepStateFunction(stateCoordinates,
                                                       reducedTemp,
                                                       systemParameters)

            # Внешние потоки зарядов
            stateCoordinatesStreams = np.array([-I, -I, -I, -evExtH2Op, -evExtH2On, evExtO2, evExtH2], dtype=np.double)

            # Внешние потоки теплоты
            heatEnergyPowersStreams = np.array([QKl, qExtp, qExtn], dtype=np.double)

            # Выводим температуры
            energyPowerTemperatures = np.hstack([reducedTemp, [Tokr]])

            # Матрица баланса
            balanceMatrix = np.array([])

            # Потенциалы взаимодействия энергетических степеней свободы
            JFz = np.hstack([JSzEl, JSzCamp, JSzCamn])  # Матрица Якоби приведенной энтропии по всем координатам состояния
            potentialInter = potentialInterElAll(JFz, reducedTemp)

            # Потенциалы взаимодействия между энергетическими степенями свободы
            potentialInterBet = np.array([])

            # Доли распределения некомпенсированной теплоты
            beta = np.array([])

            # Определяем кинетическую матрицу положительной камеры
            kMatrixElp = KineticMatrixFromPosSubMatrix(ReluFilter(kNoInvMatrixElp),  # Положительные определенные составляющие атрицы
                                                       [kInvMatrixElEvs, kInvMatrixElpEchCr, kInvMatrixElpEvCr, kInvMatrixElpEvDiss]  # Податрицы баланса
                                                       )

            # Определяем кинетическую матрицу отрицательной камеры
            kMatrixEln = KineticMatrixFromPosSubMatrix(ReluFilter(kNoInvMatrixEln),  # Положительные определенные составляющие атрицы
                                                       [kInvMatrixElEvs, kInvMatrixElnEchCr, kInvMatrixElnEvCr, kInvMatrixElnEvDiss]  # Податрицы баланса
                                                       )

            # Определяем кинетическую матрицу мембраны
            kMatrixElm = KineticMatrixFromPosSubMatrix(PosLinearFilter(kNoInvMatrixElm),  # Положительные определенные составляющие атрицы
                                                       [kInvMatrixElmDiffs, kInvMatrixElmCr, kInvMatrixElmDiffO2, kInvMatrixElmDiffH2]  # Податрицы баланса
                                                       )

            # Определение кинеттической матрицы топливного элемента без учета теплообмена с окружающей средой
            (kineticMatrixPCPC,
             kineticMatrixPCHeat,
             kineticMatrixHeatPC,
             kineticMatrixHeatHeat) = kinMatrixEl([kMatrixElp,
                                                   kMatrixEln,
                                                   kMatrixElm])

            # Главный блок кинетической матрицы по прочим координатам процессов
            kineticMatrixPCPC = np.hstack([kineticMatrixPCPC,
                                           ReluFilter(utKH2O2)])

            # Главный блок кинетической матрицы по теплообмену
            kineticMatrixHeatHeat = np.hstack([kineticMatrixHeatHeat,
                                               ReluFilter(kQOkr)])

            # Определяем обратную теплоемкость и приведенные тепловые эффекты топливного элемента
            HSzTElAll = CreateBlockMatrix([HSzTEl, HSzTCamp, HSzTCamn])  # Полная матрица Гесса приведенной энтропии по температуре и по координатам состояния
            JSTElAll = np.hstack([JSTEl, JSTCamp, JSTCamn])  # Первые производные приведенной энтропии по температуре
            HSTTElAll = np.hstack([HSTTEl, HSTTCamp, HSTTCamn])  # Вторые производные приведенной энтропии по температуре
            (invHeatCapacityMatrixCf,  # Обратная теплоемкость водородно-воздушного топливного элемента
             heatEffectMatrixCf  # Приведенные тепловые эффекты водородно-воздушного топливного элемента
             ) = heatValuesElAll(JSTElAll,  # Якобиан приведенной энтропии по температурам
                                 HSTTElAll,  # Матрица Гесса приведенной энтропии по температурам
                                 HSzTElAll,  # Матрица Гесса приведенной энтропии по температурам и координатам состояния
                                 reducedTemp  # Температуры
                                 )

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

    # Выводим структуру литий-ионного аккумулятора
    return (stateCoordinatesNames,  # Имена координат состояния
            processCoordinatesNames,  # Имена координат процессов
            energyPowersNames,  # Имена энергетических степеней свободы
            reducedTemperaturesEnergyPowersNames,  # Имена приведенных температур энергетических степеней свободы
            energyPowersBetNames,  # Имена взаимодействий между энергетическими степенями свободы
            heatTransfersNames,  # Имена потоков переноса теплоты
            heatTransfersOutputEnergyPowersNames,  # Имена энергетических степеней свободы, с которых уходит теплота
            heatTransfersInputEnergyPowersNames,  # Имена энергетических степеней свободы, на которые приходит теплота
            stateCoordinatesStreamsNames,  # Имена координат состояния, изменяемых в результате внешних потоков
            heatEnergyPowersStreamsNames,  # Имена потоков теплоты на энергетические степени свободы
            StateFunction(IndepStateFunction()),  # Функция состояния
            stateCoordinatesVarBalanceNames,  # Имена переменных коэффициентов матрицы баланса по координатам состояния
            processCoordinatesVarBalanceNames,  # Имена переменных коэффициентов матрицы баланса по координатам процессов
            energyPowersVarTemperatureNames,  # Имена переменных температур энергетических степеней свободы
            stateCoordinatesVarPotentialsInterNames,  # Имена переменных потенциалов взаимодействия по координатам состояния
            energyPowersVarPotentialsInterNames,  # Имена переменных потенциалов взаимодействия по энергетическим степеням свободы
            stateCoordinatesVarPotentialsInterBetNames,  # Имена переменных потенциалов взаимодействия для взаимодействий между энергетическими степенями свободы по координатам состояния
            energyPowersVarPotentialsInterBetNames,  # Имена переменных потенциалов взаимодействия для взаимодействий между энергетическими степенями свободы по энергетическим степеням свободы
            energyPowersVarBetaNames,  # Имена переменных долей распределения некомпенсированной теплоты энергетических степеней свободы
            processCoordinatesVarBetaNames,  # Имена переменных долей распределения некомпенсированной теплоты координат процессов
            reducedTemperaturesEnergyPowersVarInvHeatCapacityNames,  # Имена переменных коэффициентов обратных теплоемкостей по отношению к приведенным температурам
            energyPowersVarInvHeatCapacityNames,  # Имена переменных коэффициентов обратных теплоемкостей по отношению к энергетическим степеням свободы
            reducedTemperaturesEnergyPowersVarHeatEffectNames,  # Имена переменных коэффициентов обратных теплоемкостей по отношению к приведенным температурам
            stateCoordinatesVarHeatEffectNames,  # Имена переменных коэффициентов обратных теплоемкостей по отношению к координатам состояния
            varKineticPCPCNames,  # Имена сопряженностей между собой координат процессов
            varKineticPCPCAffNames,  # Имена сопряженностей между собой термодинамических сил
            varKineticPCHeatNames,  # Имена сопряженностей координат процессов с теплопереносами
            varKineticPCHeatAffNames,  # Имена сопряженностей термодинамических сил с теплопереносами
            varKineticHeatPCNames,  # Имена сопряженностей теплопереносов с координатами процессов
            varKineticHeatPCAffNames,  # Имена сопряженностей теплопереносов с термодинамическими силами
            varKineticHeatHeatNames,  # Имена сопряженностей между собой перенесенных теплот
            varKineticHeatHeatAffNames,  # Имена сопряженностей между собой термодинамических сил по переносу теплот
            stateCoordinatesVarStreamsNames,  # Имена переменных внешних потоков
            heatEnergyPowersVarStreamsNames  # Имена переменных внешних потоков теплоты
            )


# Функция постоянных параметров литий-ионного аккумулятора
def ConstParametersFunction(sysStructure  # Структура системы
                            ):
    # Задаем связь между коордиинатами состояния и процессами
    sysStructure.SetBalanceStateCoordinatesConstElement("qbinp", "dqbinp", 1)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuH2Op", "dqbinp", 1.0 / 2)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuO2", "dqbinp", -1.0 / 4)
    sysStructure.SetBalanceStateCoordinatesConstElement("qm", "dqm", 1)
    sysStructure.SetBalanceStateCoordinatesConstElement("qbinn", "dqbinn", 1)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuH2", "dqbinn", -1.0 / 2)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuH2Op", "diffH2O", -1)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuH2On", "diffH2O", 1)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuH2Op", "evH2Op", -1)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuH2OStp", "evH2Op", 1)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuH2On", "evH2On", -1)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuH2OStn", "evH2On", 1)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuO2dp", "dissO2p", 1)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuO2", "dissO2p", -1)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuH2dn", "dissH2n", 1)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuH2", "dissH2n", -1)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuO2dn", "diffO2m", 1)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuO2dp", "diffO2m", -1)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuH2dp", "diffH2m", 1)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuH2dn", "diffH2m", -1)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuH2Op", "utFuelp", 2)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuH2dp", "utFuelp", -2)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuO2dp", "utFuelp", -1)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuH2On", "utFueln", 2)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuH2dn", "utFueln", -2)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuO2dn", "utFueln", -1)

    # Задаем доли распределения некомпенсированной теплоты
    sysStructure.SetBetaConstElement("EnPowFEl", "dqbinp", 0.8)
    sysStructure.SetBetaConstElement("EnPowElp", "dqbinp", 0.2)
    sysStructure.SetBetaConstElement("EnPowFEl", "dqm", 1.0)
    sysStructure.SetBetaConstElement("EnPowFEl", "dqbinn", 0.8)
    sysStructure.SetBetaConstElement("EnPowEln", "dqbinn", 0.2)
    sysStructure.SetBetaConstElement("EnPowFEl", "diffH2O", 1.0)
    sysStructure.SetBetaConstElement("EnPowFEl", "evH2Op", 0.6)
    sysStructure.SetBetaConstElement("EnPowElp", "evH2Op", 0.4)
    sysStructure.SetBetaConstElement("EnPowFEl", "evH2On", 0.6)
    sysStructure.SetBetaConstElement("EnPowEln", "evH2On", 0.4)
    sysStructure.SetBetaConstElement("EnPowFEl", "dissH2n", 0.6)
    sysStructure.SetBetaConstElement("EnPowEln", "dissH2n", 0.4)
    sysStructure.SetBetaConstElement("EnPowFEl", "dissO2p", 0.6)
    sysStructure.SetBetaConstElement("EnPowElp", "dissO2p", 0.4)
    sysStructure.SetBetaConstElement("EnPowFEl", "diffO2m", 1.0)
    sysStructure.SetBetaConstElement("EnPowFEl", "diffH2m", 1.0)
    sysStructure.SetBetaConstElement("EnPowFEl", "utFuelp", 1.0)
    sysStructure.SetBetaConstElement("EnPowFEl", "utFueln", 1.0)
