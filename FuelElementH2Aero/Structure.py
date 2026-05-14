import numpy as np

from .StationFunction import IndepStateFunction
from .AttributesNames import stateCoordinatesNames, reducedTemperaturesEnergyPowersNames

from MathProtEnergyProc.HeatPowerValues import IntPotentialsOne, HeatValuesOne

from MathProtEnergyProc.CorrectionModel import ReluFilter, PosLinearFilter


# Функция структуры аккумулятора
def StructureFunction():
    # Описываем структуру водородно-воздушного топливного элемента
    processCoordinatesNames = ["dqbinp", "dqm", "dqbinn", "diffH2O", "evH2Op", "evH2On"]  # Имена координат процессов
    processCoordinatesNames = ["dnuMatp", "dnuMatn"]  # Имена координат процессов
    energyPowersNames = ["EnPowDegMat", "EnPowMat", "EnPowOkr"]  # Имена энергетических степеней свободы
    energyPowersBetNames = []  # Имена взаимодействий между энергетическими степенями свободы
    heatTransfersNames = ["QDegMatMat", "QMatExp"]  # Имена потоков переноса теплоты
    heatTransfersOutputEnergyPowersNames = ["EnPowDegMat", "EnPowMat"]  # Имена энергетических степеней свободы, с которых уходит теплота
    heatTransfersInputEnergyPowersNames = ["EnPowMat", "EnPowOkr"]  # Имена энергетических степеней свободы, на которые приходит теплота
    stateCoordinatesStreamsNames = ["nuMat"]  # Имена координат состояния, изменяемых в результате внешних потоков
    heatEnergyPowersStreamsNames = []  # Имена потоков теплоты на энергетические степени свободы
    stateCoordinatesVarBalanceNames = []  # Имена переменных коэффициентов матрицы баланса по координатам состояния
    processCoordinatesVarBalanceNames = []  # Имена переменных коэффициентов матрицы баланса по координатам процессов
    energyPowersVarTemperatureNames = ["EnPowDegMat", "EnPowMat", "EnPowOkr"]  # Имена переменных температур энергетических степеней свободы
    stateCoordinatesVarPotentialsInterNames = ["nuMat", "nuMatDeg"]  # Имена переменных потенциалов взаимодействия по координатам состояния
    energyPowersVarPotentialsInterNames = ["EnPowDegMat", "EnPowDegMat"]  # Имена переменных потенциалов взаимодействия по энергетическим степеням свободы
    stateCoordinatesVarPotentialsInterBetNames = []  # Имена переменных потенциалов взаимодействия для взаимодействий между энергетическими степенями свободы по координатам состояния
    energyPowersVarPotentialsInterBetNames = []  # Имена переменных потенциалов взаимодействия для взаимодействий между энергетическими степенями свободы по энергетическим степеням свободы
    energyPowersVarBetaNames = []  # Имена переменных долей распределения некомпенсированной теплоты энергетических степеней свободы
    processCoordinatesVarBetaNames = []  # Имена переменных долей распределения некомпенсированной теплоты координат процессов
    reducedTemperaturesEnergyPowersVarInvHeatCapacityNames = ["TDegMat", "TMat"]  # Имена переменных коэффициентов обратных теплоемкостей по отношению к приведенным температурам
    energyPowersVarInvHeatCapacityNames = ["EnPowDegMat", "EnPowMat"]  # Имена переменных коэффициентов обратных теплоемкостей по отношению к энергетическим степеням свободы
    reducedTemperaturesEnergyPowersVarHeatEffectNames = ["TDegMat", "TDegMat"]  # Имена переменных коэффициентов обратных теплоемкостей по отношению к приведенным температурам
    stateCoordinatesVarHeatEffectNames = ["nuMat", "nuMatDeg"]  # Имена переменных коэффициентов обратных теплоемкостей по отношению к координатам состояния
    varKineticPCPCNames = ["dnuMatp", "dnuMatn"]  # Имена сопряженностей между собой координат процессов
    varKineticPCPCAffNames = ["dnuMatp", "dnuMatn"]  # Имена сопряженностей между собой термодинамических сил
    varKineticPCHeatNames = []  # Имена сопряженностей координат процессов с теплопереносами
    varKineticPCHeatAffNames = []  # Имена сопряженностей термодинамических сил с теплопереносами
    varKineticHeatPCNames = []  # Имена сопряженностей теплопереносов с координатами процессов
    varKineticHeatPCAffNames = []  # Имена сопряженностей теплопереносов с термодинамическими силами
    varKineticHeatHeatNames = ["QDegMatMat", "QMatExp"]  # Имена сопряженностей между собой перенесенных теплот
    varKineticHeatHeatAffNames = ["QDegMatMat", "QMatExp"]  # Имена сопряженностей между собой термодинамических сил по переносу теплот
    stateCoordinatesVarStreamsNames = ["nuMat"]  # Имена переменных внешних потоков
    heatEnergyPowersVarStreamsNames = []  # Имена переменных внешних потоков теплоты

    # Потенциалы взаимодействия в топливном элементе и камерах
    potentialInterMat = IntPotentialsOne(stateCoordinatesNames,  # Имена координат состояния
                                         ["EnPowDegMat", "EnPowMat"],  # Имена энергетических степеней свободы

                                         [      "nuMat",     "nuMatDeg"],  # Имена переменных потенциалов взаимодействия по координатам состояния
                                         ["EnPowDegMat",  "EnPowDegMat"]  # Имена переменных потенциалов взаимодействия по энергетическим степеням свободы
                                         )

    # Приведенные обратные теплоемкости и тепловые эффекты
    heatValuesMat = HeatValuesOne(stateCoordinatesNames,  # Имена координат состояния
                                  ["EnPowDegMat", "EnPowMat"],  # Имена энергетических степеней свободы

                                  ["EnPowDegMat", "EnPowMat"],  # Имена переменных коэффициентов обратных теплоемкостей по отношению к энергетическим степеням свободы
                                  ["EnPowDegMat", "EnPowDegMat"],  # Имена переменных коэффициентов обратных теплоемкостей по отношению к приведенным температурам
                                  [      "nuMat",    "nuMatDeg"]  # Имена переменных коэффициентов обратных теплоемкостей по отношению к координатам состояния
                                  )

    # Функция состояния для литий-ионного аккумулятора
    def StateFunction(stateCoordinates,
                      reducedTemp,
                      systemParameters):
        # Получаем независимые свойства веществ и процессов
        (Tokr, sVAlpha,
         JSNu, JST, HSNuT, HSTT,
         ADNu, KQMat) = IndepStateFunction(stateCoordinates,
                                           reducedTemp,
                                           systemParameters)

        # Матрица баланса
        balanceMatrix = np.array([])

        # Внешние потоки зарядов
        stateCoordinatesStreams = np.array([sVAlpha], dtype=np.double)

        # Внешние потоки теплоты
        heatEnergyPowersStreams = np.array([])

        # Выводим температуры
        energyPowerTemperatures = np.hstack([reducedTemp, [Tokr]])

        # Доли распределения некомпенсированной теплоты
        beta = np.array([])

        # Потенциалы взаимодействия энергетических степеней свободы
        potentialInter = potentialInterMat(JSNu, reducedTemp)

        # Потенциалы взаимодействия между энергетическими степенями свободы
        potentialInterBet = np.array([])

        # Главный блок кинетической матрицы по процессам
        kineticMatrixPCPC = PosLinearFilter(ADNu)

        # Перекрестные блоки кинетической матрицы по процессам
        kineticMatrixPCHeat = np.array([])
        kineticMatrixHeatPC = np.array([])

        # Главный блок кинетической матрицы по теплообмену
        kineticMatrixHeatHeat = ReluFilter(KQMat)

        # Обратная теплоемкость и приведенные тепловые эффекты литий-ионного аккумулятора
        (invHeatCapacityMatrixCf,  # Обратная теплоемкость водородно-воздушного топливного элемента
         heatEffectMatrixCf  # Приведенные тепловые эффекты водородно-воздушного топливного элемента
         ) = heatValuesMat(JST,  # Якобиан приведенной энтропии по температурам
                           HSTT,  # Матрица Гесса приведенной энтропии по температурам
                           HSNuT,  # Матрица Гесса приведенной энтропии по температурам и координатам состояния
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
            StateFunction,  # Функция состояния
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
    sysStructure.SetBalanceStateCoordinatesConstElement("nuMat", "dnuMatp", 1)
    sysStructure.SetBalanceStateCoordinatesConstElement("nuMatDeg", "dnuMatn", 1)

    # Задаем доли распределения некомпенсированной теплоты
    sysStructure.SetBetaConstElement("EnPowDegMat", "dnuMatp", 1.0)
    sysStructure.SetBetaConstElement("EnPowDegMat", "dnuMatn", 1.0)
