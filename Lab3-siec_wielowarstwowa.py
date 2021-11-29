import numpy as np
import matplotlib.pyplot as plt
import random
from keras.datasets import mnist
import math

def czytajMNISTbiblioteka(): #wczytanie danych
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    return train_X, train_y, test_X, test_y

def calkowite_pobudzenie_1(x, w, bias):
    suma=0
    for a in range (0, len(x)):
        for b in range (len(x[a])):
            suma+= round(x[a][b], 2) * round(w,2) +bias
    return suma

def calkowite_pobudzenie_2(x, w, bias):
    suma=0
    for a in range (0, len(x)):
        suma+=x[a]*w+bias
    return suma

def funkcja_liniowa(suma):
    return suma

def funkcja_aktywacji_sigmoidalna(suma):
    return 1 / (1 + np.exp(-suma))

def funkcja_aktywacji_tangens_hiperboliczny(suma):
    return np.tanh(suma)

def relu(z):
    a = math.log(np.exp(z)+1)
    return np.where(a<10, a, 10)

def pochodna_funkcji_liniowej(z):
    return 1

def pochodna_funkcji_sigmoidalnej(z):
    return (1 / (1 + np.exp(-1*z))) * ( 1- (1 / (1 + np.exp(-1 * z))))

def pochodna_funkcji_tangens_hiperboliczny(z):
    return 1 - (funkcja_aktywacji_tangens_hiperboliczny(z)**2)

def pochodna_z_funkcji_relu(z):
    return 1 /(1+np.exp(-1*z))

def warstwa_wejsciowa():
    train_X, train_y, test_X, test_y = czytajMNISTbiblioteka()
    return train_X[:1000], train_y[:1000], train_X[1000:1100], train_y[1000:]

def warstwa_wejsciowa_paczki(ile_ma_paczka, i):
    train_X, train_y, test_X, test_y = czytajMNISTbiblioteka()
    return train_X[:1000], train_y[:1000]

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum(axis=0)

def softmax_dla_sumy(sumy):
    suma_calkowita = 0
    for a in range(len(sumy)):
        suma_calkowita+=np.exp(sumy[a])
    return max(sumy)/suma_calkowita

def softmax_gradient(s): 
    jacobian_m = np.diag(s)
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1-s[i])
            else: 
                jacobian_m[i][j] = -s[i]*s[j]
    return jacobian_m 

def warstwa_ukryta(x, ile_neuronow, funkcja_aktywacji, w_dla_wszystkich, bias_dla_wszystkich, ktora_warstwa):
    #tą metodę można wywoływać wielokrotnie, zależnie od tego ile chcemy wartsw ukrytych w modelu
    a_dla_wszystkich = []
    lista_sum_wszystkich = []
    for c in range(len(x)):
        a = []
        lista_sum=[]
        for b in range(0, ile_neuronow):
            if ktora_warstwa == 1:
                suma = calkowite_pobudzenie_1(x[c], w_dla_wszystkich[b][c], bias_dla_wszystkich[b])
            else:
                suma = calkowite_pobudzenie_2(x[c], w_dla_wszystkich[b][c], bias_dla_wszystkich[b])
            lista_sum.append(suma)
            if (funkcja_aktywacji == "funkcja_aktywacji_tangens_hiperboliczny"):
                a.append(funkcja_aktywacji_tangens_hiperboliczny(suma))
            elif(funkcja_aktywacji=="funkcja_aktywacji_sigmoidalna"):
                a.append(funkcja_aktywacji_sigmoidalna(suma))
            elif(funkcja_aktywacji=="relu"):
                a.append(relu(suma))
        a_dla_wszystkich.append(a)
        lista_sum_wszystkich.append(lista_sum)
    return a_dla_wszystkich, lista_sum_wszystkich

def warstwa_wyjsciowa(a2, ile_neuronow, w_dla_wszystkich, bias_dla_wszystkich):
    a_dla_wszystkich = []
    wynik = []
    for c in range(len(a2)):
        a=[]
        for b in range(0, ile_neuronow):
            a.append(calkowite_pobudzenie_2(a2[c], w_dla_wszystkich[b][c], bias_dla_wszystkich[b]))
        a_dla_wszystkich.append(a)

    for i in range(len(a_dla_wszystkich)):
        wynik.append(softmax(a_dla_wszystkich[i]))
   
    return wynik, a_dla_wszystkich

def bladDlaJednostekWyjsciowych(wynik, y, suma):
    delta = []
    skutecznosc = 0
    for i in range(len(wynik)):
        delta.append((y[i] - max(wynik[i]))*softmax_dla_sumy(suma[i]))
        if delta[i] == 0 or delta[i] < 0.5: 
            skutecznosc+=1
    return delta, skutecznosc/len(wynik)

def bladDlaJednostekUkrytych(w, delty, funkcja_aktywacji, sumy, bias):
    delta_ukryta=[]
    for i in range(len(sumy)):
        for j in range(len(sumy[i])):
            if (funkcja_aktywacji == "funkcja_aktywacji_tangens_hiperboliczny"):
                z = pochodna_funkcji_tangens_hiperboliczny(sumy[i][j])
            elif(funkcja_aktywacji=="funkcja_aktywacji_sigmoidalna"):
                z = pochodna_funkcji_sigmoidalnej(sumy[i][j])
            elif(funkcja_aktywacji=="relu"):
                z = pochodna_z_funkcji_relu(sumy[i][j])
            suma=0
            suma+=delty[i]*w[j][i]+bias[j]
        delta_ukryta.append(z*suma)
    return delta_ukryta

def bladDlaJednostekUkrytych_Nesterov(w, delty, funkcja_aktywacji, sumy, bias, gamma):
    delta_ukryta=[]
    for i in range(len(sumy)):
        for j in range(len(sumy[i])):
            if (funkcja_aktywacji == "funkcja_aktywacji_tangens_hiperboliczny"):
                z = pochodna_funkcji_tangens_hiperboliczny(sumy[i][j]+gamma*w[j][i])
            elif(funkcja_aktywacji=="funkcja_aktywacji_sigmoidalna"):
                z = pochodna_funkcji_sigmoidalnej(sumy[i][j]+gamma*w[j][i])
            elif(funkcja_aktywacji=="relu"):
                z = pochodna_z_funkcji_relu(sumy[i][j]+gamma*w[j][i])
            suma=0
            suma+=delty[i]*w[j][i]+bias[j]
        delta_ukryta.append(z*suma)
    return delta_ukryta

def losowanie_wag(x, ile_neuronow):
    w_dla_wszytskich = []
    bias_dla_wszystkich = [] 
    momentum_dla_wszystkich = []
    for j in range (0, ile_neuronow):
        w_dla_neuronu = []
        momentum_dla_neuronu = []
        for i in range (0, x):
            w_dla_neuronu.append(random.uniform(-0.2, 0.2))
            momentum_dla_neuronu.append(0)
        w_dla_wszytskich.append(w_dla_neuronu)
        momentum_dla_wszystkich.append(momentum_dla_neuronu)
        bias_dla_wszystkich.append(random.uniform(-0.2, 0.2))

    return w_dla_wszytskich, bias_dla_wszystkich, momentum_dla_wszystkich

def losowanie_wag_xavier(x, ile_neuronow):
    w_dla_wszytskich = []
    bias_dla_wszystkich = [] 
    momentum_dla_wszystkich = []
    for j in range (0, ile_neuronow):
        w_dla_neuronu = []
        momentum_dla_neuronu = []
        for i in range (0, x):
            w_dla_neuronu.append(random.uniform(-0.2, 0.2)*np.sqrt((2/(ile_neuronow+x))))
            momentum_dla_neuronu.append(0)
        w_dla_wszytskich.append(w_dla_neuronu)
        momentum_dla_wszystkich.append(momentum_dla_neuronu)
        bias_dla_wszystkich.append(random.uniform(-0.2, 0.2))

    return w_dla_wszytskich, bias_dla_wszystkich, momentum_dla_wszystkich

def losowanie_wag_he(x, ile_neuronow):
    w_dla_wszytskich = []
    bias_dla_wszystkich = [] 
    momentum_dla_wszystkich = []
    for j in range (0, ile_neuronow):
        w_dla_neuronu = []
        momentum_dla_neuronu = []
        for i in range (0, x):
            w_dla_neuronu.append(random.uniform(-0.2, 0.2)*np.sqrt((2/ile_neuronow)))
            momentum_dla_neuronu.append(0)
        w_dla_wszytskich.append(w_dla_neuronu)
        momentum_dla_wszystkich.append(momentum_dla_neuronu)
        bias_dla_wszystkich.append(random.uniform(-0.2, 0.2))

    return w_dla_wszytskich, bias_dla_wszystkich, momentum_dla_wszystkich

def uaktualnienie_wag(w, mi, a, delty):
    for i in range(len(w)):
        for j in range(len(w[i])):
            w[i][j]+=mi*a[j][i]*delty[j]
    return w

def uaktualnienie_wag_momentum_SGD(w, mi, a, delty, gamma, momentum, bias):
    for i in range(len(w)):
        for j in range(len(w[i])):
            momentum[i][j]=momentum[i][j]*gamma - mi*a[j][i]*delty[j]
            w[i][j]+=momentum[i][j]
        bias[i] += sum(momentum[i])/len(momentum[i])
    return w, momentum, bias

def uaktualnienie_wag_momentum_nesterov(w, mi, a, delty, gamma, momentum, bias):
    for i in range(len(w)):
        temp_bias = []
        for j in range(len(w[i])):
            temp = momentum[i][j]
            temp_bias.append(temp)
            momentum[i][j]=momentum[i][j]*gamma - mi*a[j][i]*delty[j]
            w[i][j]+= -gamma*temp + (1+gamma)*momentum[i][j]
        bias[i] += -gamma * (sum(temp_bias)/len(temp_bias)) +(1+gamma) * sum(momentum[i])/len(momentum[i])
    return w, momentum, bias

def uaktualnienie_wag_adagrad(w, mi, a, delty, gamma, dane, bias):
    for i in range(len(w)):
        for j in range(len(w[i])):
            dane[i][j] = (a[j][i]*delty[j]**2)
            w[i][j]+= (-1*mi*a[j][i]*delty[j]) / (np.sqrt(dane[i][j]) + np.finfo(np.float).eps)
        bias[i] += (-1*mi* (sum(a[i])/len(a[i]))*(sum(delty)/len(delty))) / (np.sqrt(sum(dane[i])/len(dane[i])) + np.finfo(np.float).eps)
    return w, dane, bias

def uaktualnienie_wag_adadelta(w, mi, a, delty, gamma, dane, bias):
    for i in range(len(w)):
        for j in range(len(w[i])):
            dane[i][j] = (gamma *dane[i][j]) + ((1-gamma) * a[j][i]*delty[j]**2)
            w[i][j]+= (-1*mi*a[j][i]*delty[j]) / (np.sqrt(dane[i][j]) + np.finfo(np.float).eps)
        bias[i] += (-1*mi* (sum(a[i])/len(a[i]))*(sum(delty)/len(delty))) / (np.sqrt(sum(dane[i])/len(dane[i])) + np.finfo(np.float).eps)
    return w, dane, bias

def uaktualnienie_wag_momentum_adam(w, mi, a, delty, gamma, momentum, dane, bias):
    for i in range(len(w)):
        for j in range(len(w[i])):
            momentum[i][j] = 0.9 *momentum[i][j] + (1-0.9)*a[j][i]*delty[j]
            dane[i][j] = (0.999 *dane[i][j]) + ((1-0.999) * a[j][i]*delty[j]**2)
            w[i][j]+= (-1*mi*momentum[i][j] / (0.1**j)) / (np.sqrt(dane[i][j]/(0.001**j)) + np.finfo(np.float).eps)
        bias[i]+= (-1*mi*sum(momentum[i])/len(momentum[i])/0.1**i) / (np.sqrt(sum(dane[i])/len(dane[i])/(0.001**i)) + np.finfo(np.float).eps)
    return w, momentum, dane, bias

def uaktualnienie_bias(bias, mi, delty):
    for i in range(len(bias)):
        bias[i]+=mi*sum(delty)/len(delty)
    return bias

def obliczenie_progu_walidacyjnego(x_walidacyjny, y_walidacyjny, blad_stary, w_dla_wszystkich, w_dla_wszystkich_2, w_dla_wszystkich_3, bias_dla_wszystkich, bias_dla_wszystkich_2, bias_dla_wszystkich_3):
    a, lista_sum_wszystkich_1 = warstwa_ukryta(x_walidacyjny, 55, "relu", w_dla_wszystkich, bias_dla_wszystkich, 1)
    a1, lista_sum_wszystkich_2 = warstwa_ukryta(a, 35, "funkcja_aktywacji_tangens_hiperboliczny", w_dla_wszystkich_2, bias_dla_wszystkich_2, 2)
    wynik, a_dla_wszystkich= warstwa_wyjsciowa(a1, 10, w_dla_wszystkich_3, bias_dla_wszystkich_3)
    delty = bladDlaJednostekWyjsciowych(wynik, y_walidacyjny, a_dla_wszystkich)
    delty_koncowe=[]
    
    for i in range(len(delty)):
        delty_koncowe.append(delty[i]**2)
    blad = sum(delty_koncowe)/2

    return blad - blad_stary 

def algorytm_dzialania_paczek(prog, ile_ma_paczka, ile_obrazkow):
    w_dla_wszystkich, bias_dla_wszystkich, momentum_dla_wszystkich = losowanie_wag_he(ile_ma_paczka, 35)
    w_dla_wszystkich_3, bias_dla_wszystkich_3, momentum_dla_wszystkich_3 = losowanie_wag_he(ile_ma_paczka, 10)
    dane = []
    dane.append(np.zeros(shape=(55, ile_ma_paczka)))
    dane.append(np.zeros(shape=(35, ile_ma_paczka)))
    dane.append(np.zeros(shape=(10, ile_ma_paczka)))
    blad_calkowity = 10000
    epoki=0
    skutecznosc_tab = []
    while blad_calkowity>prog:
        epoki+=1
        progi=[]
        x_caly, y_caly = warstwa_wejsciowa_paczki(ile_ma_paczka, 0)
        random.shuffle(x_caly) # w tym miejscu w każdej epoce mieszam obrazki pomiędzy paczkami
        random.shuffle(y_caly)
        for i in range(int(ile_obrazkow/ ile_ma_paczka)):
            x, y = x_caly[i*ile_ma_paczka: i*ile_ma_paczka+ile_ma_paczka], y_caly[i*ile_ma_paczka: i*ile_ma_paczka+ile_ma_paczka]
            a, lista_sum_wszystkich_1 = warstwa_ukryta(x, 35, "funkcja_aktywacji_sigmoidalna", w_dla_wszystkich, bias_dla_wszystkich, 1)
            wynik, a_dla_wszystkich= warstwa_wyjsciowa(a, 10, w_dla_wszystkich_3, bias_dla_wszystkich_3)
            delty, skutecznosc = bladDlaJednostekWyjsciowych(wynik, y, a_dla_wszystkich)
            skutecznosc_tab.append(skutecznosc)
            delty_ukryte_1 = bladDlaJednostekUkrytych(w_dla_wszystkich, delty, "funkcja_aktywacji_sigmoidalna", lista_sum_wszystkich_1, bias_dla_wszystkich)
            #w_dla_wszystkich_3, momentum_dla_wszystkich_3, bias_dla_wszystkich_3 = uaktualnienie_wag(w_dla_wszystkich_3, 0.01, wynik, delty, 0.7, momentum_dla_wszystkich_3, bias_dla_wszystkich_3)
            w_dla_wszystkich_3 = uaktualnienie_wag(w_dla_wszystkich_3, 0.01, wynik, delty)
            #w_dla_wszystkich, momentum_dla_wszystkich, bias_dla_wszystkich = uaktualnienie_wag(w_dla_wszystkich, 0.01, a, delty_ukryte_1, 0.7, momentum_dla_wszystkich, bias_dla_wszystkich)
            w_dla_wszystkich = uaktualnienie_wag(w_dla_wszystkich, 0.01, a, delty_ukryte_1)
            delty_koncowe=[]
            for i in range(len(delty)):
                delty_koncowe.append(delty[i]**2)
            blad = sum(delty_koncowe)/2
            print("Wielkość błędu: "+str(blad))
            progi.append(blad)
        blad_calkowity = sum(progi)/len(progi)
    print("skutecznosc: "+str(sum(skutecznosc_tab)/len(skutecznosc_tab)*100))
    return epoki

if __name__ == '__main__':
    tab=[]
    for i in range(0,10):
        tab.append(algorytm_dzialania_paczek(0.1, 100, 1000))
        print("Wartość pojedynczego: "+str(tab[i]))
    print("Tab:")
    print(tab)
    