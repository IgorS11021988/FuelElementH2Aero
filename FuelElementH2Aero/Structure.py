from .StationFunction import StateFunction


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
    stateFunction = StateFunction  # Функция состояния
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
    varKineticPCPCNames = ["dqbinp", "dqm", "dqbinn", "diffH2O", "evH2Op", "evH2On", "dqm", "diffH2O"]  # Имена сопряженностей между собой координат процессов
    varKineticPCPCAffNames = ["dqbinp", "dqm", "dqbinn", "diffH2O", "evH2Op", "evH2On", "diffH2O", "dqm"]  # Имена сопряженностей между собой термодинамических сил
    varKineticPCHeatNames = ["evH2Op", "evH2On", "dqbinp", "dqbinn"]  # Имена сопряженностей координат процессов с теплопереносами
    varKineticPCHeatAffNames = ["QFElp", "QFEln", "QFElp", "QFEln"]  # Имена сопряженностей термодинамических сил с теплопереносами
    varKineticHeatPCNames = ["QFElp", "QFEln", "QFElp", "QFEln"]  # Имена сопряженностей теплопереносов с координатами процессов
    varKineticHeatPCAffNames = ["evH2Op", "evH2On", "dqbinp", "dqbinn"]  # Имена сопряженностей теплопереносов с термодинамическими силами
    varKineticHeatHeatNames = ["Qexp", "QFElp", "QFEln", "Qexpp", "Qexpn"]  # Имена сопряженностей между собой перенесенных теплот
    varKineticHeatHeatAffNames = ["Qexp", "QFElp", "QFEln", "Qexpp", "Qexpn"]  # Имена сопряженностей между собой термодинамических сил по переносу теплот
    stateCoordinatesVarStreamsNames = ["qbinp", "qm", "qbinn", "nuH2OStp", "nuH2OStn", "nuO2", "nuH2"]  # Имена переменных внешних потоков
    heatEnergyPowersVarStreamsNames = ["EnPowFEl", "EnPowElp", "EnPowEln"]  # Имена переменных внешних потоков теплоты

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
            stateFunction,  # Функция состояния
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
