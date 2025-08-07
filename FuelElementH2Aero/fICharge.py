# Функция условий протекания процессов
def fICharge(t,  # Моменты времени
             systemParameters  # Параметры системы
             ):
    # Выделяем параметры динамики
    Ie = systemParameters[0]  # Постоянная составляющая тока

    # Прочие параметры системы
    otherSystemParameters = systemParameters[1::]

    # Выводим результат
    return (Ie, otherSystemParameters)
