import numpy as np

from MathProtEnergyProc.CorrectionModel import PosLinearFilter


# Вспомогательные функции
def funRI(alphaRI, dissU):  # Мультипликативная корректировка по току через двойной слой
    # Проверяем падение напряжения
    _dissU = alphaRI * dissU
    if (np.fabs(_dissU) > 0.001):
        return 2 * _dissU / (np.exp(_dissU) - np.exp(-_dissU))
    else:
        return 1


def funRT(alphaRT, bRT, cRT, TFEl):  # Мультипликативная корректировка по температуре
    return cRT * np.exp(-alphaRT * (TFEl - bRT)) + 1


def funCQbin(qbin, alphaCQ):  # Емкость двойного слоя в зависимости от заряда
    return np.exp(alphaCQ * np.abs(qbin))


def funHMuLin(rNu, HMus, dHMus,
              betaHMu2, betaHMu3):
    # Основная составляющая
    HMu = HMus + dHMus * rNu

    # Добавляем довески
    return HMu + dHMus * (betaHMu2 * np.power(rNu, 2) + betaHMu3 * np.power(rNu, 3))


def funHMuH2O(nuH2O, TFEl,
              nuH2Os, THMus,
              muH2Os, dmuH2Os,
              hH2Os, dhH2Os,
              betaMuH2O2, betaMuH2O3,
              betaHH2O2, betaHH2O3,
              cFElH2O):  # Характерный химический потенциал кислорода в мембране
    # Относительное число молей пропитывающей воды
    rNuH2O = nuH2O / nuH2Os

    # Химический потенциал пропитывающей воды при стандартной температуре
    muH2O = funHMuLin(rNuH2O, muH2Os, dmuH2Os,
                      betaMuH2O2, betaMuH2O3)

    # Тепловой потенциал пропитывающей воды при стандартной температуре
    hH2O = funHMuLin(rNuH2O, hH2Os, dhH2Os,
                     betaHH2O2, betaHH2O3)

    # Приведенные температуры
    rTFEl = TFEl / THMus  # Относительная температура
    dTFEl = TFEl - THMus  # Температура относительно уровня
    lTFEl = dTFEl - TFEl * np.log(rTFEl)

    # Химический потенциал пропитывающей воды
    muH2O += (muH2O - hH2O) * (rTFEl - 1) + cFElH2O * lTFEl

    # Тепловой потенциал пропитывающей воды
    hH2O += cFElH2O * dTFEl

    # Выводим результат
    return (muH2O, hH2O)


def funHMuLog(rNu, HMus, dHMus):
    return HMus + dHMus * np.log(rNu)


def funHMuCam(nuH2OSt, nuG, TCam,
              nuH2OSts, nuGs, THMus,
              muH2OSts, dmuH2OSts, muGs, dmuGs,
              hH2OSts, dhH2OSts, hGs, dhGs,
              betaMuH2OSt, betaMuH2OStG, betaMuG,
              betaHH2OSt, betaHH2OStG, betaHG,
              cElH2OSt, cElG):  # Определяем химические потенциалы кислорода и воды в камере положительного электрода
    # Определяем отнисительные числа молей воды и газа
    rNuH2OStG = np.sqrt(nuH2OSts * nuGs)
    rNuH2OSt = nuH2OSt / nuH2OSts
    rNuG = nuG / nuGs
    rCrNuH2OSt = nuH2OSt / rNuH2OStG
    rCrNuG = nuG / rNuH2OStG

    # Определяем химические потенциалы воды и газа при стандартной температуре
    muH2OSt = funHMuLog(rNuH2OSt, muH2OSts, dmuH2OSts) + betaMuH2OSt * rNuH2OSt + betaMuH2OStG * rCrNuG
    muG = funHMuLog(rNuG, muGs, dmuGs) + betaMuH2OStG * rCrNuH2OSt + betaMuG * rNuG

    # Определяем тепловые потенциалы воды и газа при стандартной температуре
    hH2OSt = funHMuLog(rNuH2OSt, hH2OSts, dhH2OSts) + betaHH2OSt * rNuH2OSt + betaHH2OStG * rCrNuG
    hG = funHMuLog(rNuG, hGs, dhGs) + betaHH2OStG * rCrNuH2OSt + betaHG * rNuG

    # Приведенные температуры
    rTCam = TCam / THMus  # Относительная температура
    dTCam = TCam - THMus  # Температура относительно уровня
    lTCam = dTCam - TCam * np.log(rTCam)

    # Определяем химические потенциалы воды и газа
    muH2OSt += (muH2OSt - hH2OSt) * (rTCam - 1) + cElH2OSt * lTCam
    muG += (muG - hG) * (rTCam - 1) + cElG * lTCam

    # Определяем тепловые потенциалы воды и газа при стандартной температуре
    hH2OSt += cElH2OSt * dTCam
    hG += cElG * dTCam

    # Выводим результат
    return (muH2OSt, muG, hH2OSt, hG)


def funKrmH2O(nuH2Op, nuH2On, nuH2Osm):
    return 2 * nuH2Osm / (nuH2Op + nuH2On)


def funNuEvH2O(nuH2O, nuH2OSt, nuH2OsEv):
    return (nuH2O + nuH2OSt) / (2 * nuH2OsEv)


def funKTEvH2O(TFEl, TEl, alphaKTEvH2Os, bTKEvH2Os, cTKEvH2Os):
    return 1 + cTKEvH2Os * np.exp(alphaKTEvH2Os * ((TEl + TFEl) / 2 - bTKEvH2Os))


# Функции для свойств веществ и процессов
def funRbin(TFEl, dissUbinp, dissUbinn, alphaRIp,
            alphaRIn, alphaRTp, alphaRTn, bRTp,
            bRTn, cRTp, cRTn, Rbin0p, Rbin0n, dKElTQp0, dKElTQn0,
            crQKElp, crQKEln, betaRI2p, betaRI2n,
            betaRI3p, betaRI3n, betaRT2p, betaRT2n,
            betaRT3p, betaRT3n):  # Функция сопротивления двойных слоев
    # Определяем корректировку сопротивления двойных слоев через токи двойных слоев
    rIbinp = funRI(alphaRIp, dissUbinp)  # Положительный двойной слой
    rIbinn = funRI(alphaRIn, dissUbinn)  # Отрицательный двойной слой

    # Добавляем довесочные члены к корректировкам сопротивления двойных слоев через токи двойных слоев
    rIbinp += betaRI2p * np.power(rIbinp, 2) + betaRI3p * np.power(rIbinp, 3)
    rIbinn += betaRI2n * np.power(rIbinn, 2) + betaRI3n * np.power(rIbinn, 3)

    # Определяем корректировку сопротивления двойных слоев через температуру
    rTbinp = funRT(alphaRTp, bRTp, cRTp, TFEl)
    rTbinn = funRT(alphaRTn, bRTn, cRTn, TFEl)

    # Добавляем довесочные члены к корректировкам сопротивления двойных слоев через температуру
    rTbinp += betaRT2p * np.power(rTbinp, 2) + betaRT3p * np.power(rTbinp, 3)
    rTbinn += betaRT2n * np.power(rTbinn, 2) + betaRT3n * np.power(rTbinn, 3)

    # Определяем корректирующие коэффициенты сопротивлений двойных слоев
    rbinp = rIbinp * rTbinp  # Положительный двойной слой
    rbinn = rIbinn * rTbinn  # Отрицательный двойной слой

    # Выводим результат
    rbinp = PosLinearFilter(rbinp)
    rbinn = PosLinearFilter(rbinn)
    return (Rbin0p * rbinp,
            Rbin0n * rbinn,
            dKElTQp0 / rbinp,
            dKElTQn0 / rbinn,
            crQKElp * np.sqrt(dKElTQp0 / Rbin0p) / rbinp,
            crQKEln * np.sqrt(dKElTQn0 / Rbin0n) / rbinn)


def funRm(TFEl, nuH2Op, nuH2On, nuH2Osm,
          alphaRTm, bRTm, cRTm, Rm0, kDiffH2O0,
          dKDiffH2O0, crRmDiffH2O, betaRT2m,
          betaRT3m, betaKRmH2O2, betaKRmH2O3):  # Функция сопротивления мембраны
    # Определяем температурный коэффициент сопротивления мембраны
    rm = funRT(-alphaRTm, bRTm, cRTm, TFEl)

    # Добавляем довесочные члены к температурному коэффициенту сопротивления мембраны
    rm += betaRT2m * np.power(rm, 2) + betaRT3m * np.power(rm, 3)

    # Определяем увлажняющий коэффициент сопротивления мембраны
    krmH2O = funKrmH2O(nuH2Op, nuH2On, nuH2Osm)

    # Добавляем довесочные члены к увлажняющему коэффициенту сопротивления мембраны
    krmH2O += betaKRmH2O2 * np.power(krmH2O, 2) + betaKRmH2O3 * np.power(krmH2O, 3)

    # Определяем корректирующий коэффициент сопротивления мембраны
    cf = rm * (1 + krmH2O)

    # Выводим результат
    cf = PosLinearFilter(cf)
    return (Rm0 * cf,
            kDiffH2O0 + dKDiffH2O0 / cf,
            crRmDiffH2O * np.sqrt(dKDiffH2O0 / Rm0) / cf)


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
    kTEvH2Op += betaKTEvH2Op2 * np.power(kTEvH2Op, 2) + betaKTEvH2Op3 * np.power(kTEvH2Op, 3)
    kTEvH2On += betaKTEvH2On2 * np.power(kTEvH2On, 2) + betaKTEvH2On3 * np.power(kTEvH2On, 3)

    # Корректировочные коэффициенты по числу молей воды
    kNuEvH2Op = funNuEvH2O(nuH2Op, nuH2OStp, nuH2OsEvp)
    kNuEvH2On = funNuEvH2O(nuH2On, nuH2OStn, nuH2OsEvn)

    # Добавляем довески к корректировочному коэффициенту по числу молей воды
    kNuEvH2Op += betaKNuEvH2Op2 * np.power(kNuEvH2Op, 2) + betaKNuEvH2Op3 * np.power(kNuEvH2Op, 3)
    kNuEvH2On += betaKNuEvH2On2 * np.power(kNuEvH2On, 2) + betaKNuEvH2On3 * np.power(kNuEvH2On, 3)

    # Итоговый корректирующий коэффициент
    cfp = kTEvH2Op * kNuEvH2Op
    cfn = kTEvH2On * kNuEvH2On

    # Выводим результат
    cfp = PosLinearFilter(cfp)
    cfn = PosLinearFilter(cfn)
    return (kEvH2Osp * cfp, kEvH2Osn * cfn,
            dKElTEvp0 * cfp, dKElTEvn0 * cfn,
            crEvH20KElp * np.sqrt(kEvH2Osp * dKElTEvp0) * cfp,
            crEvH20KEln * np.sqrt(kEvH2Osn * dKElTEvn0) * cfn)
