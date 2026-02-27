import numpy as np

from .StationFunction import IndepStateFunction

from MathProtEnergyProc.CorrectionModel import KineticMatrixQ, KineticMatrixFromPosSubMatrix, CreateBlockMatrix
from MathProtEnergyProc.HeatPowerValues import IntPotentialsOne, HeatValuesOne

from MathProtEnergyProc.CorrectionModel import ReluFilter, PosLinearFilter


# Функция структуры аккумулятора
def StructureFunction():
    # Описываем структуру водородно-воздушного топливного элемента
    stateCoordinatesNames = ["qbinp", "qm", "qbinn", "nuH2Op", "nuH2On", "nuH2OStp", "nuH2OStn", "nuO2", "nuH2"]  # Имена координат состояния
    processCoordinatesNames = ["dqbinp", "dqm", "dqbinn", "diffH2O", "evH2Op", "evH2On"]  # Имена координат процессов
    energyPowersNames = ["EnPowFEl", "EnPowElp", "EnPowEln", "EnPowOkr"]  # Имена энергетических степеней свободы
    reducedTemperaturesEnergyPowersNames = ["TFEl", "TElp", "TEln"]  # Имена приведенных температур энергетических степеней свободы
    energyPowersBetNames = []  # Имена взаимодействий между энергетическими степенями свободы
    heatTransfersNames = ["Qexp", "QFElp", "QFEln", "Qexpp", "Qexpn"]  # Имена потоков переноса теплоты
    heatTransfersOutputEnergyPowersNames = ["EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowElp", "EnPowEln"]  # Имена энергетических степеней свободы, с которых уходит теплота
    heatTransfersInputEnergyPowersNames = ["EnPowOkr", "EnPowElp", "EnPowEln", "EnPowOkr", "EnPowOkr"]  # Имена энергетических степеней свободы, на которые приходит теплота
    stateCoordinatesStreamsNames = ["qbinp", "qm", "qbinn", "nuH2OStp", "nuH2OStn", "nuO2", "nuH2"]  # Имена координат состояния, изменяемых в результате внешних потоков
    heatEnergyPowersStreamsNames = ["EnPowFEl", "EnPowElp", "EnPowEln"]  # Имена потоков теплоты на энергетические степени свободы
    stateCoordinatesVarBalanceNames = []  # Имена переменных коэффициентов матрицы баланса по координатам состояния
    processCoordinatesVarBalanceNames = []  # Имена переменных коэффициентов матрицы баланса по координатам процессов
    energyPowersVarTemperatureNames = ["EnPowFEl", "EnPowElp", "EnPowEln", "EnPowOkr"]  # Имена переменных температур энергетических степеней свободы
    stateCoordinatesVarPotentialsInterNames = ["qbinp", "qm", "qbinn", "nuH2Op", "nuH2On", "nuH2OStp", "nuH2OStn", "nuO2", "nuH2"]  # Имена переменных потенциалов взаимодействия по координатам состояния
    energyPowersVarPotentialsInterNames = ["EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowElp", "EnPowEln", "EnPowElp", "EnPowEln"]  # Имена переменных потенциалов взаимодействия по энергетическим степеням свободы
    stateCoordinatesVarPotentialsInterBetNames = []  # Имена переменных потенциалов взаимодействия для взаимодействий между энергетическими степенями свободы по координатам состояния
    energyPowersVarPotentialsInterBetNames = []  # Имена переменных потенциалов взаимодействия для взаимодействий между энергетическими степенями свободы по энергетическим степеням свободы
    energyPowersVarBetaNames = []  # Имена переменных долей распределения некомпенсированной теплоты энергетических степеней свободы
    processCoordinatesVarBetaNames = []  # Имена переменных долей распределения некомпенсированной теплоты координат процессов
    reducedTemperaturesEnergyPowersVarInvHeatCapacityNames = ["TFEl", "TElp", "TEln"]  # Имена переменных коэффициентов обратных теплоемкостей по отношению к приведенным температурам
    energyPowersVarInvHeatCapacityNames = ["EnPowFEl", "EnPowElp", "EnPowEln"]  # Имена переменных коэффициентов обратных теплоемкостей по отношению к энергетическим степеням свободы
    reducedTemperaturesEnergyPowersVarHeatEffectNames = ["TFEl", "TFEl", "TFEl", "TFEl", "TFEl", "TElp", "TEln", "TElp", "TEln"]  # Имена переменных коэффициентов обратных теплоемкостей по отношению к приведенным температурам
    stateCoordinatesVarHeatEffectNames = ["qbinp", "qm", "qbinn", "nuH2Op", "nuH2On", "nuH2OStp", "nuH2OStn", "nuO2", "nuH2"]  # Имена переменных коэффициентов обратных теплоемкостей по отношению к координатам состояния
    varKineticPCPCNames = ["dqbinp", "evH2Op", "dqbinn", "evH2On", "dqm", "diffH2O", "dqm", "diffH2O"]  # Имена сопряженностей между собой координат процессов
    varKineticPCPCAffNames = ["dqbinp", "evH2Op", "dqbinn", "evH2On", "dqm", "diffH2O", "diffH2O", "dqm"]  # Имена сопряженностей между собой термодинамических сил
    varKineticPCHeatNames = ["evH2Op", "dqbinp", "evH2On", "dqbinn"]  # Имена сопряженностей координат процессов с теплопереносами
    varKineticPCHeatAffNames = ["QFElp", "QFElp", "QFEln", "QFEln"]  # Имена сопряженностей термодинамических сил с теплопереносами
    varKineticHeatPCNames = ["QFElp", "QFElp", "QFEln", "QFEln"]  # Имена сопряженностей теплопереносов с координатами процессов
    varKineticHeatPCAffNames = ["evH2Op", "dqbinp", "evH2On", "dqbinn"]  # Имена сопряженностей теплопереносов с термодинамическими силами
    varKineticHeatHeatNames = ["QFElp", "QFEln", "Qexp", "Qexpp", "Qexpn"]  # Имена сопряженностей между собой перенесенных теплот
    varKineticHeatHeatAffNames = ["QFElp", "QFEln", "Qexp", "Qexpp", "Qexpn"]  # Имена сопряженностей между собой термодинамических сил по переносу теплот
    stateCoordinatesVarStreamsNames = ["qbinp", "qm", "qbinn", "nuH2OStp", "nuH2OStn", "nuO2", "nuH2"]  # Имена переменных внешних потоков
    heatEnergyPowersVarStreamsNames = ["EnPowFEl", "EnPowElp", "EnPowEln"]  # Имена переменных внешних потоков теплоты

    # Коэффициенты кинетической матрицы
    kinMatrixElp = KineticMatrixQ(["dqbinp", "evH2Op"],  # Имена сопряженностей между собой координат процессов
                                  ["dqbinp", "evH2Op"],  # Имена сопряженностей между собой термодинамических сил
                                  ["evH2Op", "dqbinp"],  # Имена сопряженностей координат процессов с теплопереносами
                                  [ "QFElp",  "QFElp"],  # Имена сопряженностей термодинамических сил с теплопереносами
                                  [ "QFElp",  "QFElp"],  # Имена сопряженностей теплопереносов с координатами процессов
                                  ["evH2Op", "dqbinp"],  # Имена сопряженностей теплопереносов с термодинамическими силами
                                  ["QFElp"],  # Имена сопряженностей между собой перенесенных теплот
                                  ["QFElp"],  # Имена сопряженностей между собой термодинамических сил по переносу теплот

                                  [["dqbinp", "evH2Op", "QFElp"]]  # Массив массивов имен координат процессов (в том числе и перенесенных теплот) по кинетической матрице
                                  )  # Кинетическая матрица по камере положительного электрода
    kinMatrixEln = KineticMatrixQ(["dqbinn", "evH2On"],  # Имена сопряженностей между собой координат процессов
                                  ["dqbinn", "evH2On"],  # Имена сопряженностей между собой термодинамических сил
                                  ["evH2On", "dqbinn"],  # Имена сопряженностей координат процессов с теплопереносами
                                  [ "QFEln",  "QFEln"],  # Имена сопряженностей термодинамических сил с теплопереносами
                                  [ "QFEln",  "QFEln"],  # Имена сопряженностей теплопереносов с координатами процессов
                                  ["evH2On", "dqbinn"],  # Имена сопряженностей теплопереносов с термодинамическими силами
                                  ["QFEln"],  # Имена сопряженностей между собой перенесенных теплот
                                  ["QFEln"],  # Имена сопряженностей между собой термодинамических сил по переносу теплот

                                  [["dqbinn", "evH2On", "QFEln"]]  # Массив массивов имен координат процессов (в том числе и перенесенных теплот) по кинетической матрице
                                  )  # Кинетическая матрица по камере отрицательного электрода
    kinMatrixElm = KineticMatrixQ(["dqm", "diffH2O",     "dqm", "diffH2O"],  # Имена сопряженностей между собой координат процессов
                                  ["dqm", "diffH2O", "diffH2O",     "dqm"],  # Имена сопряженностей между собой термодинамических сил
                                  [],  # Имена сопряженностей координат процессов с теплопереносами
                                  [],  # Имена сопряженностей термодинамических сил с теплопереносами
                                  [],  # Имена сопряженностей теплопереносов с координатами процессов
                                  [],  # Имена сопряженностей теплопереносов с термодинамическими силами
                                  [],  # Имена сопряженностей между собой перенесенных теплот
                                  [],  # Имена сопряженностей между собой термодинамических сил по переносу теплот

                                  [["dqm", "diffH2O"]]  # Массив массивов имен координат процессов (в том числе и перенесенных теплот) по кинетической матрице
                                  )  # Кинетическая матрица по мембране элемента

    # Потенциалы взаимодействия в топливном элементе и камерах
    potentialInterElAll = IntPotentialsOne(["qbinp", "qm", "qbinn", "nuH2Op", "nuH2On", "nuH2OStp", "nuO2", "nuH2OStn", "nuH2"],  # Имена координат состояния
                                           ["EnPowFEl", "EnPowElp", "EnPowEln"],  # Имена энергетических степеней свободы

                                           [   "qbinp",       "qm",    "qbinn",   "nuH2Op",   "nuH2On", "nuH2OStp", "nuH2OStn",     "nuO2",     "nuH2"],  # Имена переменных потенциалов взаимодействия по координатам состояния
                                           ["EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowElp", "EnPowEln", "EnPowElp", "EnPowEln"]  # Имена переменных потенциалов взаимодействия по энергетическим степеням свободы
                                           )

    # Приведенные обратные теплоемкости и тепловые эффекты
    heatValuesElAll = HeatValuesOne(["qbinp", "qm", "qbinn", "nuH2Op", "nuH2On", "nuH2OStp", "nuO2", "nuH2OStn", "nuH2"],  # Имена координат состояния
                                    ["EnPowFEl", "EnPowElp", "EnPowEln"],  # Имена энергетических степеней свободы

                                    ["EnPowFEl", "EnPowElp", "EnPowEln"],  # Имена переменных коэффициентов обратных теплоемкостей по отношению к энергетическим степеням свободы
                                    ["EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowFEl", "EnPowElp", "EnPowEln", "EnPowElp", "EnPowEln"],  # Имена переменных коэффициентов обратных теплоемкостей по отношению к приведенным температурам
                                    [   "qbinp",       "qm",    "qbinn",   "nuH2Op",   "nuH2On", "nuH2OStp", "nuH2OStn",     "nuO2",     "nuH2"]  # Имена переменных коэффициентов обратных теплоемкостей по отношению к координатам состояния
                                    )

    # Функция состояния для литий-ионного аккумулятора
    def StateFunction(stateCoordinates,
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
         kQOkr, I, Tokr,
         qExtp, qExtn) = IndepStateFunction(stateCoordinates,
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
        kMatrixElp = KineticMatrixFromPosSubMatrix(PosLinearFilter(kNoInvMatrixElp),  # Положительные определенные составляющие атрицы
                                                   [kInvMatrixElEvs, kInvMatrixElpEchCr, kInvMatrixElpEvCr]  # Податрицы баланса
                                                   )
        (kineticMatrixPCPCElp,
         kineticMatrixPCHeatElp,
         kineticMatrixHeatPCElp,
         kineticMatrixHeatHeatElp) = kinMatrixElp([kMatrixElp])

        # Определяем кинетическую матрицу отрицательной камеры
        kMatrixEln = KineticMatrixFromPosSubMatrix(PosLinearFilter(kNoInvMatrixEln),  # Положительные определенные составляющие атрицы
                                                   [kInvMatrixElEvs, kInvMatrixElnEchCr, kInvMatrixElnEvCr]  # Податрицы баланса
                                                   )
        (kineticMatrixPCPCEln,
         kineticMatrixPCHeatEln,
         kineticMatrixHeatPCEln,
         kineticMatrixHeatHeatEln) = kinMatrixEln([kMatrixEln])

        # Определяем кинетическую матрицу мембраны
        kMatrixElm = KineticMatrixFromPosSubMatrix(PosLinearFilter(kNoInvMatrixElm),  # Положительные определенные составляющие атрицы
                                                   [kInvMatrixElmDiffs, kInvMatrixElmCr]  # Податрицы баланса
                                                   )
        (kineticMatrixPCPCElm,
         kineticMatrixPCHeatElm,
         kineticMatrixHeatPCElm,
         kineticMatrixHeatHeatElm) = kinMatrixElm([kMatrixElm])

        # Главный блок кинетической матрицы по процессам
        kineticMatrixPCPC = np.hstack([kineticMatrixPCPCElp,
                                       kineticMatrixPCPCEln,
                                       kineticMatrixPCPCElm])

        # Перекрестные блоки кинетической матрицы по процессам
        kineticMatrixPCHeat = np.hstack([kineticMatrixPCHeatElp,
                                         kineticMatrixPCHeatEln,
                                         kineticMatrixPCHeatElm])
        kineticMatrixHeatPC = np.hstack([kineticMatrixHeatPCElp,
                                         kineticMatrixHeatPCEln,
                                         kineticMatrixHeatPCElm])

        # Главный блок кинетической матрицы по теплообмену
        kineticMatrixHeatHeat = np.hstack([kineticMatrixHeatHeatElp,
                                           kineticMatrixHeatHeatEln,
                                           kineticMatrixHeatHeatElm,
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
