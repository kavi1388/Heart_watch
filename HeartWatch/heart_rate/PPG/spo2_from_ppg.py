import random


def spo2(heartRate):
    spo2h_buff0 = [88, 89, 90, 91, 92, 93]
    spo2h_buff1 = [94, 95, 96, 97]
    spo2h_buff2 = [97, 98, 99]

    iRandom = int(random.random() * 10)
    if heartRate < 45:
        spo2h = 0
    elif heartRate < 50:
        spo2h = spo2h_buff0[0]
    elif heartRate < 60:
        iRandom = iRandom % 6
        spo2h = spo2h_buff0[iRandom]
    elif heartRate < 70:
        iRandom = iRandom % 4
        spo2h = spo2h_buff1[iRandom]
    elif heartRate <= 100:
        iRandom = iRandom % 3
        spo2h = spo2h_buff2[iRandom]
    else:
        spo2h = spo2h_buff2[2]

    return spo2h
