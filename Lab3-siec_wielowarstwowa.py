import numpy as np
import matplotlib.pyplot as plt
import random
from keras.datasets import mnist

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
    return np.where(z<0, 0, z)

def pochodna_funkcji_liniowej(z):
    return 1

def pochodna_funkcji_sigmoidalnej(z):
    return round(z*(1-z),5)

def pochodna_funkcji_tangens_hiperboliczny(z):
    return 1 - (funkcja_aktywacji_tangens_hiperboliczny(z)**2)

def pochodna_z_funkcji_relu(z):
    return 1

def warstwa_wejsciowa():
    train_X, train_y, test_X, test_y = czytajMNISTbiblioteka()
    return train_X[:1000], train_y[:1000], train_X[500:600], train_y[500:600]

def warstwa_wejsciowa_paczki(ile_ma_paczka, i):
    train_X, train_y, test_X, test_y = czytajMNISTbiblioteka()
    return train_X[:100], train_y[:100]
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
    for i in range(len(wynik)):
        delta.append((y[i] - max(wynik[i]))*softmax_dla_sumy(suma[i]))
    return delta

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

def losowanie_wag(x, ile_neuronow):
    w_dla_wszytskich = []
    bias_dla_wszystkich = [] 
    for j in range (0, ile_neuronow):
        w_dla_neuronu = []
        for i in range (0, x):
            w_dla_neuronu.append(random.uniform(-0.2, 0.2))
        w_dla_wszytskich.append(w_dla_neuronu)
        bias_dla_wszystkich.append(random.uniform(-0.2, 0.2))

    return w_dla_wszytskich, bias_dla_wszystkich

def uaktualnienie_wag(w, mi, a, delty):
    for i in range(len(w)):
        for j in range(len(w[i])):
            w[i][j]+=mi*a[j][i]*delty[j]
    return w

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
    w_dla_wszystkich, bias_dla_wszystkich = losowanie_wag(ile_ma_paczka, 55)
    w_dla_wszystkich_2, bias_dla_wszystkich_2 = losowanie_wag(ile_ma_paczka, 35)
    w_dla_wszystkich_3, bias_dla_wszystkich_3 = losowanie_wag(ile_ma_paczka, 10)
    blad_calkowity = 10000
    while blad_calkowity>prog:
        progi=[]
        x_caly, y_caly = warstwa_wejsciowa_paczki(ile_ma_paczka, 0)
        random.shuffle(x_caly) # w tym miejscu w każdej epoce mieszam obrazki pomiędzy paczkami
        random.shuffle(y_caly)
        for i in range(int(ile_obrazkow/ ile_ma_paczka)):
            x, y = x_caly[i*ile_ma_paczka: i*ile_ma_paczka+ile_ma_paczka], y_caly[i*ile_ma_paczka: i*ile_ma_paczka+ile_ma_paczka]
            a, lista_sum_wszystkich_1 = warstwa_ukryta(x, 55, "relu", w_dla_wszystkich, bias_dla_wszystkich, 1)
            a1, lista_sum_wszystkich_2 = warstwa_ukryta(a, 35, "funkcja_aktywacji_tangens_hiperboliczny", w_dla_wszystkich_2, bias_dla_wszystkich_2, 2)
            wynik, a_dla_wszystkich= warstwa_wyjsciowa(a1, 10, w_dla_wszystkich_3, bias_dla_wszystkich_3)
            delty = bladDlaJednostekWyjsciowych(wynik, y, a_dla_wszystkich)
            delty_ukryte_2 = bladDlaJednostekUkrytych(w_dla_wszystkich_2, delty, "funkcja_aktywacji_tangens_hiperboliczny", lista_sum_wszystkich_2, bias_dla_wszystkich_2)
            delty_ukryte_1 = bladDlaJednostekUkrytych(w_dla_wszystkich, delty_ukryte_2, "relu", lista_sum_wszystkich_1, bias_dla_wszystkich)
            print("1--------------------------------------------------------------")
            w_dla_wszystkich_3 = uaktualnienie_wag(w_dla_wszystkich_3, 0.1, wynik, delty)
            print("2--------------------------------------------------------------")
            w_dla_wszystkich_2 = uaktualnienie_wag(w_dla_wszystkich_2, 0.1, a1, delty_ukryte_2)
            print("3--------------------------------------------------------------")
            w_dla_wszystkich = uaktualnienie_wag(w_dla_wszystkich, 0.1, a, delty_ukryte_1)
            bias_dla_wszystkich_3 = uaktualnienie_bias(bias_dla_wszystkich_3, 0.1, delty)
            print(len(delty_ukryte_2))
            print(len(bias_dla_wszystkich_2))
            bias_dla_wszystkich_2 = uaktualnienie_bias(bias_dla_wszystkich_2, 0.1, delty_ukryte_2)
            bias_dla_wszystkich = uaktualnienie_bias(bias_dla_wszystkich, 0.1, delty_ukryte_1)
            delty_koncowe=[]
            for i in range(len(delty)):
                delty_koncowe.append(delty[i]**2)
            blad = sum(delty_koncowe)/2
            print("Wielkość błędu: "+str(blad))
            progi.append(blad)
        blad_calkowity = sum(progi)/len(progi)
       
def algorytm_dzialania(prog, prog_walidacyjny):
    x, y, x_walidacyjny, y_walidacyjny = warstwa_wejsciowa()
    w_dla_wszystkich, bias_dla_wszystkich = losowanie_wag(len(x), 200)
    w_dla_wszystkich_2, bias_dla_wszystkich_2 = losowanie_wag(len(x), 100)
    w_dla_wszystkich_3, bias_dla_wszystkich_3 = losowanie_wag(len(x), 10)
    blad = 10000
    blad_walidacyjny = 0
    epoki = 0 
    while blad>prog:
        epoki+=1
        a, lista_sum_wszystkich_1 = warstwa_ukryta(x, 200, "funkcja_aktywacji_tangens_hiperboliczny", w_dla_wszystkich, bias_dla_wszystkich, 1)
        print(a)
        a1, lista_sum_wszystkich_2 = warstwa_ukryta(a, 100, "funkcja_aktywacji_sigmoidalna", w_dla_wszystkich_2, bias_dla_wszystkich_2, 2)
        wynik, a_dla_wszystkich= warstwa_wyjsciowa(a1, 10, w_dla_wszystkich_3, bias_dla_wszystkich_3)
        delty = bladDlaJednostekWyjsciowych(wynik, y, a_dla_wszystkich)
        delty_ukryte_2 = bladDlaJednostekUkrytych(w_dla_wszystkich_2, delty, "funkcja_aktywacji_sigmoidalna", lista_sum_wszystkich_2, bias_dla_wszystkich_2)
        delty_ukryte_1 = bladDlaJednostekUkrytych(w_dla_wszystkich, delty_ukryte_2, "funkcja_aktywacji_tangens_hiperboliczny", lista_sum_wszystkich_1, bias_dla_wszystkich)

        print("1--------------------------------------------------------------")
        w_dla_wszystkich_3 = uaktualnienie_wag(w_dla_wszystkich_3, 0.1, wynik, delty)
        print("2--------------------------------------------------------------")
        w_dla_wszystkich_2 = uaktualnienie_wag(w_dla_wszystkich_2, 0.1, a1, delty_ukryte_2)
        print("3--------------------------------------------------------------")
        w_dla_wszystkich = uaktualnienie_wag(w_dla_wszystkich, 0.1, a, delty_ukryte_1)
        bias_dla_wszystkich_3 = uaktualnienie_bias(bias_dla_wszystkich_3, 0.1, delty)
        bias_dla_wszystkich_2 = uaktualnienie_bias(bias_dla_wszystkich_2, 0.1, delty_ukryte_2)
        bias_dla_wszystkich = uaktualnienie_bias(bias_dla_wszystkich, 0.1, delty_ukryte_1)
        delty_koncowe=[]
        
        for i in range(len(delty)):
            delty_koncowe.append(delty[i]**2)
        blad = sum(delty_koncowe)/2
        print("Wielkość błędu: "+str(blad))
    return epoki

if __name__ == '__main__':
    algorytm_dzialania(0.1, 5)
    #algorytm_dzialania_paczek(0.1, 10, 100)
    