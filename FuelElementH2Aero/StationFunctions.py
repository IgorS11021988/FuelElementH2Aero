import numpy as np

from MathProtEnergyProc.CorrectionModel import ReluFilter


# Вспомогательные функции
def funRI(alphaRI, dissU):  # Мультипликативная корректировка по току через двойной слой
    # Проверяем падение напряжения
    _dissU = alphaRI * dissU
    if (np.fabs(_dissU) > 0.001):
        return 2 * _dissU / (np.exp(_dissU) - np.exp(-_dissU))
    else:
        return 1


def funRT(alphaRT, bRT, cRT, TFEl):  # Мультипликативная корректировка по температуре
    return 1 + cRT * np.exp(-alphaRT * (TFEl - bRT))


def funCQbin(qbin, alphaCQ):  # Емкость двойного слоя в зависимости от заряда
    return np.exp(alphaCQ * np.abs(qbin))


def funHMuLin(rNu, HMus, dHMus,
              betaHMu2, betaHMu3):
    # Основная составляющая
    HMu = HMus + dHMus * rNu

    # Добавляем довески
    return HMu + dHMus * (betaHMu2 * np.power(rNu, 2) + betaHMu3 * np.power(rNu, 3))


def funJHSzTEl(qbinp, qm, qbinn,
               nuH2Op, nuH2On, TFEl,
               Cbinp, Cm, Cbinn, Econ,
               nuH2Os, THMus, muH2Os,
               dmuH2Os, hH2Os, dhH2Os,
               betaMuH2O2, betaMuH2O3,
               betaHH2O2, betaHH2O3,
               cFElH2O, CFEls):  # Характерный химический потенциал кислорода в мембране
    # Относительное число молей пропитывающей воды
    rNuH2Op = nuH2Op / nuH2Os
    rNuH2On = nuH2On / nuH2Os

    # Химический потенциал пропитывающей воды при стандартной температуре
    muH2Op = funHMuLin(rNuH2Op, muH2Os, dmuH2Os,
                       betaMuH2O2, betaMuH2O3)
    muH2On = funHMuLin(rNuH2On, muH2Os, dmuH2Os,
                       betaMuH2O2, betaMuH2O3)

    # Тепловой потенциал пропитывающей воды при стандартной температуре
    hH2Ops = funHMuLin(rNuH2Op, hH2Os, dhH2Os,
                       betaHH2O2, betaHH2O3)
    hH2Ons = funHMuLin(rNuH2On, hH2Os, dhH2Os,
                       betaHH2O2, betaHH2O3)

    # Приведенные температуры
    rTFEl = TFEl / THMus  # Относительная температура
    dTFEl = TFEl - THMus  # Температура относительно уровня
    lTFEl = dTFEl - TFEl * np.log(rTFEl)

    # Падения напряжений на двойных слоях
    dissUbinp =  Econ - qbinp / Cbinp  # Положительный двойной слой
    dissUbinn = -Econ - qbinn / Cbinn  # Отрицательный двойной слой

    # Теплоемкость топливного элемента
    CFEl = CFEls + cFElH2O * (nuH2Op + nuH2On)

    # Матрица Якоби приведенной энтропии по координатам состояния
    JSzElH2Op = -hH2Ops - (muH2Op - hH2Ops) * rTFEl - cFElH2O * lTFEl
    JSzElH2On = -hH2Ons - (muH2On - hH2Ons) * rTFEl - cFElH2O * lTFEl
    JSzEl = np.array([dissUbinp, -qm / Cm, dissUbinn,
                      JSzElH2Op, JSzElH2On], dtype=np.double) / TFEl

    # Матрица Гесса приведенной энтропии по координатам состояния и температуре
    HSzTElH2Op = hH2Ops + cFElH2O * dTFEl
    HSzTElH2On = hH2Ons + cFElH2O * dTFEl
    HSzTEl = np.array([-dissUbinp, qm / Cm, -dissUbinn,
                       HSzTElH2Op, HSzTElH2On], dtype=np.double) / np.power(TFEl, 2)

    # Приведенные первая и вторая производные приведенной энтропии по температуре
    JSTEl = CFEl * dTFEl / np.power(TFEl, 2)
    HSTTEl = CFEl * (2 * THMus - TFEl) / np.power(TFEl, 3)

    # Выводим результат
    return (JSzEl, HSzTEl, JSTEl, HSTTEl, dissUbinp, dissUbinn)


def funHMuLog(rNu, HMus, dHMus):
    return HMus + dHMus * np.log(rNu)


def funJHSzTCam(nuH2OSt, nuG, TCam,
                nuH2OSts, nuGs, THMus,
                muH2OSts, dmuH2OSts, muGs, dmuGs,
                hH2OSts, dhH2OSts, hGs, dhGs,
                betaMuH2OSt, betaMuH2OStG, betaMuG,
                betaHH2OSt, betaHH2OStG, betaHG,
                cElH2OSt, cElG, CEls):  # Определяем химические потенциалы кислорода и воды в камере положительного электрода
    # Определяем отнисительные числа молей воды и газа
    rNuH2OStG = np.sqrt(nuH2OSts * nuGs)
    rNuH2OSt = nuH2OSt / nuH2OSts
    rNuG = nuG / nuGs
    rCrNuH2OSt = nuH2OSt / rNuH2OStG
    rCrNuG = nuG / rNuH2OStG

    # Определяем химические потенциалы воды и газа при стандартной температуре
    muH2OSts = funHMuLog(rNuH2OSt, muH2OSts, dmuH2OSts) + betaMuH2OSt * rNuH2OSt + betaMuH2OStG * rCrNuG
    muGs = funHMuLog(rNuG, muGs, dmuGs) + betaMuH2OStG * rCrNuH2OSt + betaMuG * rNuG

    # Определяем тепловые потенциалы воды и газа при стандартной температуре
    hH2OSts = funHMuLog(rNuH2OSt, hH2OSts, dhH2OSts) + betaHH2OSt * rNuH2OSt + betaHH2OStG * rCrNuG
    hGs = funHMuLog(rNuG, hGs, dhGs) + betaHH2OStG * rCrNuH2OSt + betaHG * rNuG

    # Приведенные температуры
    rTCam = TCam / THMus  # Относительная температура
    dTCam = TCam - THMus  # Температура относительно уровня
    lTCam = dTCam - TCam * np.log(rTCam)

    # Определяем теплоемкость камеры
    CCam = CEls + cElH2OSt * nuH2OSt + cElG * nuG

    # Определяем матрицу Якоби по числам молей
    JSzCamH2OSt = -hH2OSts - (muH2OSts - hH2OSts) * rTCam - cElH2OSt * lTCam
    JSzCamG = -hGs - (muGs - hGs) * rTCam - cElG * lTCam
    JSzCam = np.array([JSzCamH2OSt, JSzCamG], dtype=np.double) / TCam

    # Определяем Гесса приведенной энтропии по координатам состояния и температуре
    HSzTCamH2OSt = hH2OSts + cElH2OSt * dTCam
    HSzTCamG = hGs + cElG * dTCam
    HSzTCam = np.array([HSzTCamH2OSt, HSzTCamG], dtype=np.double) / np.power(TCam, 2)

    # Приведенные первая и вторая производные приведенной энтропии по температуре
    JSTCam = CCam * dTCam / np.power(TCam, 2)
    HSTTCam = CCam * (2 * THMus - TCam) / np.power(TCam, 3)

    # Выводим результат
    return (JSzCam, HSzTCam, JSTCam, HSTTCam)


def funKrmH2O(nuH2Op, nuH2On, nuH2Osm):
    return 2 * nuH2Osm / (nuH2Op + nuH2On)


def funNuEvH2O(nuH2O, nuH2OSt, nuH2OsEv):
    return (nuH2O + nuH2OSt) / (2 * nuH2OsEv)


def funKTEvH2O(TFEl, TEl, alphaKTEvH2Os, bTKEvH2Os, cTKEvH2Os):
    return cTKEvH2Os * np.exp(alphaKTEvH2Os * ((TEl + TFEl) / 2 - bTKEvH2Os))


# Функции для свойств веществ и процессов
def funRbin(TFEl, dissUbinp, dissUbinn, alphaRIp,
            alphaRIn, alphaRTp, alphaRTn, bRTp,
            bRTn, cRTp, cRTn, betaRI2p, betaRI2n,
            betaRI3p, betaRI3n, betaRT2p, betaRT2n,
            betaRT3p, betaRT3n):  # Функция сопротивления двойных слоев
    # Определяем корректировку сопротивления двойных слоев через токи двойных слоев
    sIbinp = 1 / funRI(alphaRIp, dissUbinp)  # Положительный двойной слой
    sIbinn = 1 / funRI(alphaRIn, dissUbinn)  # Отрицательный двойной слой

    # Добавляем довесочные члены к корректировкам сопротивления двойных слоев через токи двойных слоев
    sIbinp += betaRI2p * np.power(1 - sIbinp, 2) + betaRI3p * np.power(1 - sIbinp, 3)
    sIbinn += betaRI2n * np.power(1 - sIbinn, 2) + betaRI3n * np.power(1 - sIbinn, 3)

    # Определяем корректировку сопротивления двойных слоев через температуру
    sTbinp = 1 / funRT(alphaRTp, bRTp, cRTp, TFEl)
    sTbinn = 1 / funRT(alphaRTn, bRTn, cRTn, TFEl)

    # Добавляем довесочные члены к корректировкам сопротивления двойных слоев через температуру
    sTbinp += betaRT2p * np.power(1 - sTbinp, 2) + betaRT3p * np.power(1 - sTbinp, 3)
    sTbinn += betaRT2n * np.power(1 - sTbinn, 2) + betaRT3n * np.power(1 - sTbinn, 3)

    # Выводим результат
    return (sIbinp * sTbinp,
            sIbinn * sTbinn)


def funRm(TFEl, nuH2Op, nuH2On, nuH2Osm,
          alphaRTm, bRTm, cRTm, betaRT2m,
          betaRT3m, betaKRmH2O2, betaKRmH2O3):  # Функция сопротивления мембраны
    # Определяем температурный коэффициент сопротивления мембраны
    sm = 1 / funRT(-alphaRTm, bRTm, cRTm, TFEl)

    # Добавляем довесочные члены к температурному коэффициенту сопротивления мембраны
    sm += betaRT2m * np.power(1 - sm, 2) + betaRT3m * np.power(1 - sm, 3)

    # Определяем увлажняющий коэффициент сопротивления мембраны
    krmH2O = funKrmH2O(nuH2Op, nuH2On, nuH2Osm)

    # Добавляем довесочные члены к увлажняющему коэффициенту сопротивления мембраны
    krmH2O += betaKRmH2O2 * np.power(krmH2O, 2) + betaKRmH2O3 * np.power(krmH2O, 3)

    # Выводим резуль(тат
    return sm / (1 + ReluFilter(krmH2O))


def funCbin(qbinp, qbinn, alphaCQp, alphaCQn, Cbin0p, Cbin0n,
            betaCQ2p, betaCQ2n, betaCQ3p, betaCQ3n):  # Функция емкостей двойных слоев
    # Определяем корректировочный коэффициент емкости двойного слоя
    rCbinQp = funCQbin(qbinp, alphaCQp)  # Положительный двойной слой
    rCbinQn = funCQbin(qbinn, alphaCQn)  # Отрицательный двойной слой

    # Учитываем довесочные слагаемые коэффициента емкости двойного слоя
    rCbinQp1 = rCbinQp - 1
    rCbinQp += np.power(rCbinQp1, 2) + np.power(rCbinQp1, 3)  # Положительный двойной слой
    rCbinQn1 = rCbinQn - 1
    rCbinQn += np.power(rCbinQn1, 2) + np.power(rCbinQn1, 3)  # Отрицательный двойной слой

    # Выводим результат
    return (Cbin0p * rCbinQp, Cbin0n * rCbinQn)


def funEvH2O(TFEl, TElp, TEln, nuH2Op, nuH2On, nuH2OStp,
             nuH2OStn, nuH2OsEvp, nuH2OsEvn, kEvH2Osp,
             kEvH2Osn, dKElTEvp0, dKElTEvn0, crEvH20KElp,
             crEvH20KEln, alphaKTEvH2Osp, alphaKTEvH2Osn,
             bTKEvH2Osp, bTKEvH2Osn, cTKEvH2Osp, cTKEvH2Osn,
             betaKTEvH2Op2, betaKTEvH2On2, betaKNuEvH2Op2,
             betaKNuEvH2On2, betaKTEvH2Op3, betaKTEvH2On3,
             betaKNuEvH2Op3, betaKNuEvH2On3):
    # Температурные корректировочные коэффициенты
    kTEvH2Op = funKTEvH2O(TFEl, TElp, alphaKTEvH2Osp, bTKEvH2Osp, cTKEvH2Osp)
    kTEvH2On = funKTEvH2O(TFEl, TEln, alphaKTEvH2Osn, bTKEvH2Osn, cTKEvH2Osn)

    # Добавляем довески к корректировочному коэффициенту по температуре
    kTEvH2Op += betaKTEvH2Op2 * np.power(kTEvH2Op, 2) + betaKTEvH2Op3 * np.power(kTEvH2Op, 3) + 1
    kTEvH2On += betaKTEvH2On2 * np.power(kTEvH2On, 2) + betaKTEvH2On3 * np.power(kTEvH2On, 3) + 1

    # Корректировочные коэффициенты по числу молей воды
    kNuEvH2Op = funNuEvH2O(nuH2Op, nuH2OStp, nuH2OsEvp)
    kNuEvH2On = funNuEvH2O(nuH2On, nuH2OStn, nuH2OsEvn)

    # Добавляем довески к корректировочному коэффициенту по числу молей воды
    kNuEvH2Op += betaKNuEvH2Op2 * np.power(kNuEvH2Op - 1, 2) + betaKNuEvH2Op3 * np.power(kNuEvH2Op - 1, 3)
    kNuEvH2On += betaKNuEvH2On2 * np.power(kNuEvH2On - 1, 2) + betaKNuEvH2On3 * np.power(kNuEvH2On - 1, 3)

    # Итоговый корректирующий коэффициент
    kbinp = kTEvH2Op * kNuEvH2Op
    kbinn = kTEvH2On * kNuEvH2On

    # Выводим результат
    return (kTEvH2Op * kNuEvH2Op,
            kTEvH2On * kNuEvH2On)
