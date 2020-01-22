
'''
Aluno: Gabriel de Souza Nogueira da Silva
Matricula: 398847
'''
import random
import math
import matplotlib.pyplot as plp
import re
from scipy import stats
import numpy as np

#quant_neuronio = int(input('Digite a quantidade de neuronios: '))
quant_neuronio = 10


def somatorioQe(vet_y, vet_y_linha):
    # cria o somatorio de Qe
    somador = 0
    for k in range(0, len(vet_y)):
        somador = somador + (vet_y[k][0] - vet_y_linha[k][0])**2
    return somador


def somatorioYy(y):
    # cria o somatorio de yy
    somador = 0
    y_media = np.sum(y)/len(y)
    for k in range(0, len(y)):
        somador = somador + (y[k][0] - y_media)**2
    return somador


def normaliza(x):
    return stats.zscore(x)


def cria_vetor(y):
    vet = np.ones((2, len(y)))
    for i in range(0, 2):
        for j in range(0, len(y)):
            if i != 0:
                vet[i][j] = y[j]
    return vet


def cria_vetor_sem_peso_bias(y):
    vet = np.ones((1, len(y)))
    for m in range(0, len(y)):
        vet[0][m] = y[m]
    return vet


def cria_vetor_w():
    vet = np.ones((quant_neuronio, 2))
    for i in range(0, quant_neuronio):
        for j in range(0, 2):
            # valores aleatorios em uma distribuiçao normal
            vet[i][j] = random.normalvariate(0, 0.1)
    return vet


def cria_vetor_u(x, w):
    # funçao np.dot() multiplica matrizes
    return np.dot(w, x)


def cria_vetor_z(u):
    for i in range(0, quant_neuronio):
        for j in range(0, 2250):
            # aplicando funçao de ativaçao
            u[i][j] = 1 / (1 + math.exp((-1)*u[i][j]))
    return u


def cria_vetor_z_linha(z):
    vet = np.ones((len(z)+1, 2250))
    for i in range(1, len(z)+1):
        for j in range(0, 2250):
            vet[i][j] = z[i-1][j]
    return vet


def cria_vetor_a(m, z):
    return np.dot(z, m)


x = []  # vetor de entradas
d = []  # vetor de saidas

dados = open("aerogerador.dat", "r")
for line in dados:
    # separando o que é x do que é d
    line = line.strip()  # quebra no \n
    line = re.sub('\s+', ',', line)  # trocando os espaços vazios por virgula
    x1, y = line.split(",")  # quebra nas virgulas e retorna 2 valores
    x.append(float(x1))
    d.append(float(y))
dados.close()

X = cria_vetor(normaliza(x))

D = cria_vetor_sem_peso_bias(normaliza(d))

# como a funçao cria_vetor() ja cria com peso do bias
# o vetor X1 sao meus dados sem o uso do peso
X1 = cria_vetor_sem_peso_bias(normaliza(x))

W = cria_vetor_w()

U = cria_vetor_u(X, W)

Z = cria_vetor_z(U)
# vetor Z com o bias
Z_linha = cria_vetor_z_linha(Z)

# criando a matriz "treinada"
M = np.dot(np.dot(D, np.transpose(Z_linha)), np.linalg.inv(
    np.dot(Z_linha, np.transpose(Z_linha))))

a = cria_vetor_a(Z_linha, M)

# usando a formula dada nos slides de regressao
r2 = 1 - (somatorioQe(np.transpose(D), np.transpose(a)) /
          somatorioYy(np.transpose(D)))
print("Valor de R2: " + str(r2))


# Plotando os graficos
plp.plot(np.transpose(X1), np.transpose(a), color='black')
plp.scatter(X1, D, marker=".")
plp.title('R2 = %f' % (r2))
plp.show()
