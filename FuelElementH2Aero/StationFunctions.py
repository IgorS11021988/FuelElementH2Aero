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
               cFElH2O, cFElO2, cFElH2, CFEls):  # Характерный химический потенциал кислорода в мембране
    # Относительное числа молей
    rNuH2Op = nuH2Op / nuH2Os
    rNuH2On = nuH2On / nuH2Os
    rNuO2dp = nuO2dp / nuO2ds
    rNuO2dn = nuO2dn / nuO2ds
    rNuH2dp = nuH2dp / nuH2ds
    rNuH2dn = nuH2dn / nuH2ds

    # Относительные перекрестные числа молей
    crNuH2OO2s = np.sqrt(nuH2Os * nuO2ds)
    crNuH2OH2s = np.sqrt(nuH2Os * nuH2ds)
    crNuH2OO2p = nuH2Op / crNuH2OO2s
    crNuH2OO2n = nuH2On / crNuH2OO2s
    crNuH2OH2p = nuH2Op / crNuH2OH2s
    crNuH2OH2n = nuH2On / crNuH2OH2s
    crNuO2dp = nuO2dp / crNuH2OO2s
    crNuO2dn = nuO2dn / crNuH2OO2s
    crNuH2dp = nuH2dp / crNuH2OH2s
    crNuH2dn = nuH2dn / crNuH2OH2s

    # Химический потенциал пропитывающей воды при стандартной температуре
    muH2Op = funHMuLin(rNuH2Op, muH2Os, dmuH2Os,
                       betaMuH2O2, betaMuH2O3) + betaMuO2H2O * crNuO2dp + betaMuH2H2O * crNuH2dp
    muH2On = funHMuLin(rNuH2On, muH2Os, dmuH2Os,
                       betaMuH2O2, betaMuH2O3) + betaMuO2H2O * crNuO2dn + betaMuH2H2O * crNuH2dn
    muO2p = funHMuLin(rNuO2dp, muO2ds, dmuO2ds,
                      betaMuO22, betaMuO23) + betaMuO2H2O * crNuH2OO2p
    muO2n = funHMuLin(rNuO2dn, muO2ds, dmuO2ds,
                      betaMuO22, betaMuO23) + betaMuO2H2O * crNuH2OO2n
    muH2p = funHMuLin(rNuH2dp, muH2ds, dmuH2ds,
                      betaMuH22, betaMuH23) + betaMuH2H2O * crNuH2OH2p
    muH2n = funHMuLin(rNuH2dn, muH2ds, dmuH2ds,
                      betaMuH22, betaMuH23) + betaMuH2H2O * crNuH2OH2n

    # Тепловой потенциал пропитывающей воды при стандартной температуре
    hH2Ops = funHMuLin(rNuH2Op, hH2Os, dhH2Os,
                       betaHH2O2, betaHH2O3)
    hH2Ons = funHMuLin(rNuH2On, hH2Os, dhH2Os,
                       betaHH2O2, betaHH2O3)
    hO2ps = funHMuLin(rNuO2dp, hO2ds, dhO2ds,
                      betaHO22, betaHO23)
    hO2ns = funHMuLin(rNuO2dn, hO2ds, dhO2ds,
                      betaHO22, betaHO23)
    hH2ps = funHMuLin(rNuH2dp, hH2ds, dhH2ds,
                      betaHH22, betaHH23)
    hH2ns = funHMuLin(rNuH2dn, hH2ds, dhH2ds,
                      betaHH22, betaHH23)

    # Приведенные температуры
    rTFEl = TFEl / THMus  # Относительная температура
    dTFEl = TFEl - THMus  # Температура относительно уровня
    lTFEl = dTFEl - TFEl * np.log(rTFEl)

    # Напряжения на двойных слоях
    Ubinp = qbinp / Cbinp  # Положительный двойной слой
    Ubinn = qbinn / Cbinn  # Отрицательный двойной слой

    # Напряжение на мембране
    Um = qm / Cm

    # Падения напряжений на двойных слоях
    dissUbinp =  Econ - Ubinp  # Положительный двойной слой
    dissUbinn = -Econ - Ubinn  # Отрицательный двойной слой

    # Теплоемкость топливного элемента
    CFEl = CFEls + cFElH2O * (nuH2Op + nuH2On) + cFElO2 * (nuO2dp + nuO2dn) + cFElH2 * (nuH2dp + nuH2dn)

    # Матрица Якоби приведенной энтропии по координатам состояния
    JSzElH2Op = -hH2Ops - (muH2Op - hH2Ops) * rTFEl - cFElH2O * lTFEl
    JSzElH2On = -hH2Ons - (muH2On - hH2Ons) * rTFEl - cFElH2O * lTFEl
    JSzElO2p = -hO2ps - (muO2p - hO2ps) * rTFEl - cFElO2 * lTFEl
    JSzElO2n = -hO2ns - (muO2n - hO2ns) * rTFEl - cFElO2 * lTFEl
    JSzElH2p = -hH2ps - (muH2p - hH2ps) * rTFEl - cFElH2 * lTFEl
    JSzElH2n = -hH2ns - (muH2n - hH2ns) * rTFEl - cFElH2 * lTFEl
    JSzEl = np.array([dissUbinp, -Um, dissUbinn,
                      JSzElH2Op, JSzElH2On,
                      JSzElO2p, JSzElO2n,
                      JSzElH2p, JSzElH2n], dtype=np.double) / TFEl

    # Матрица Гесса приведенной энтропии по координатам состояния и температуре
    HSzTElH2Op = hH2Ops + cFElH2O * dTFEl
    HSzTElH2On = hH2Ons + cFElH2O * dTFEl
    HSzTElO2p = hO2ps + cFElO2 * dTFEl
    HSzTElO2n = hO2ns + cFElO2 * dTFEl
    HSzTElH2p = hH2ps + cFElH2 * dTFEl
    HSzTElH2n = hH2ns + cFElH2 * dTFEl
    HSzTEl = np.array([-dissUbinp, Um, -dissUbinn,
                       HSzTElH2Op, HSzTElH2On,
                       HSzTElO2p, HSzTElO2n,
                       HSzTElH2p, HSzTElH2n], dtype=np.double) / np.power(TFEl, 2)

    # Приведенные первая и вторая производные приведенной энтропии по температуре
    JSTEl = CFEl * dTFEl / np.power(TFEl, 2)
    HSTTEl = CFEl * (2 * THMus - TFEl) / np.power(TFEl, 3)

    # Выводим результат
    return (JSzEl, HSzTEl, JSTEl, HSTTEl,
            Ubinp, Um, Ubinn, dissUbinp, dissUbinn,
            rNuO2dp, rNuO2dn,
            rNuH2dp, rNuH2dn,
            JSzElH2Op, JSzElH2On,
            JSzElO2p, JSzElO2n,
            JSzElH2p, JSzElH2n)


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
    return (JSzCam, HSzTCam, JSTCam, HSTTCam, rNuG)


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

    # Выводим результат
    return (kTEvH2Op * kNuEvH2Op,
            kTEvH2On * kNuEvH2On)


# Функция для расчета относительного коэффициента растворения
def funKDiss(rNuG, rNuGd,
             rKDissG, rKDissGd,
             betaG2, betaGd2, betaGcd2):
    # Рассчитываем коэффициент растворения и выводим результат
    return rKDissG * (rNuG + betaG2 * np.power(rNuG, 2)) + rKDissGd * (rNuGd + betaGd2 * np.power(rNuGd, 2)) + rKDissG * rKDissGd * betaGcd2 * rNuG * rNuGd


# Функция коэффициента утилизации топлива
def funUtKH2O2(rNuO2, rNuH2,
               muH2O, muH2, muO2,
               betakUtH2O2,
               betakUtH2O2O2,
               betakUtH2O2H2):
    # Вычисляем приведенное химическое сродство
    aUtFuel = ReluFilter(2 * muH2 + muO2 - 2 * muH2O) * rNuO2 * rNuH2 * (betakUtH2O2 + betakUtH2O2O2 * rNuO2 + betakUtH2O2H2 * rNuH2)

    # Выводим результат
    return aUtFuel
