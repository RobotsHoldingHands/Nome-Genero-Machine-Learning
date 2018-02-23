# -*- coding: utf-8 -*-

import csv
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import tempfile

import re

#Classificadores
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import NearestCentroid

#########################################################

from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict,train_test_split
from sklearn.metrics import confusion_matrix,precision_score,f1_score
import itertools





## Função geradora da Matriz de Confusao
def plot_confusion_matrix(cm, classes, normalize=False, title="Matriz de Confusao", cmap=plt.cm.Blues):
    #Função de plot da matriz de confusão
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print "Matriz de confusão normalizada."
    else:
        print "Matriz de confusão sem normalização."

    print cm

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



## Importando a base de dados

nomes_split = []
sexos = []
homens = 0
mulheres = 0

'''
## Abertura de arquivo antigo

with open('TB_Nomes.csv', 'rb') as arquivo:
	leitor = csv.reader(arquivo, delimiter=',')
	for linha in leitor:
		nomes.append(linha[1].decode(encoding='iso8859_10'))
		nomes_split.append(list(linha[1].decode(encoding='iso8859_10')))
		if linha[2] == "M":
			sexos.append(1)
			homens += 1
		else:
			sexos.append(0)
			mulheres += 1

'''

with open('nomes.csv', 'rb') as arquivo:
	leitor = csv.reader(arquivo, delimiter=',')
	for linha in leitor:
		for i in range(int(linha[2])):
			nomes_split.append(list(linha[1].upper().decode(encoding='utf-8')))
			if linha[0] == "M":
				sexos.append(1)
				homens += 1
			else:
				sexos.append(0)
				mulheres += 1
	
print "Total de amostas: %d" % len(nomes_split)

## Shuffle para evitar viés (a ordem é mantida pois é usada uma seed)
import random

random.Random(4938).shuffle(nomes_split)
random.Random(4938).shuffle(sexos)

## Representando os dados num grafico

objetos = ('Mulheres', 'Homens')
y_pos = np.arange(len(objetos))
performance = [mulheres, homens]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objetos)
plt.ylabel('Quantidade')
plt.title('Quantidade por Sexo')

plt.show()

## Inicializando o vetorizador

#Vetorizador para palavras inteiras
#vetorizador = TfidfVectorizer()
#X = vetorizador.fit_transform(nomes)

#Vetorizador para caracteres separados
vetorizador = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
X = vetorizador.fit_transform(nomes_split)
y = sexos


## Separando treino/teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.5)

## Criando o classificador e treinando-o
## Testando diversos classificadores #Acuracia  [F1score-0 F1score-1]

#classificador = MultinomialNB(alpha=0.01) #0.82 [0.82 0.82]
#classificador = RidgeClassifier(tol=1e-2, solver="sag") #0.87 [0.86 0.88]
#classificador = PassiveAggressiveClassifier(max_iter=80) #0.85 [0.85 0.86]
#classificador = LinearSVC(penalty="l1", dual=False, tol=1e-3) #0.87 [0.87 0.87]
#classificador = Perceptron(max_iter=32) #0.81 [0.79 0.82]
#classificador = BernoulliNB(alpha=.01) #0.79 [0.8 0.79]
#classificador = MultinomialNB(alpha=.01) #0.82 [0.82 0.83]
#classificador = NearestCentroid() #0.77 [0.77, 0.77]
#classificador = SGDClassifier(alpha=.002, max_iter=50, penalty="elasticnet") #0.85  [0.85 0.86]

classificador = RandomForestClassifier(n_estimators=50) #0.99 [0.99 0.99] 

classificador.fit(X_treino, y_treino)

## Fazendo testes para acurácia
y_pred = classificador.predict(X_teste)

matriz = confusion_matrix(y_teste, y_pred)

np.set_printoptions(precision = 2)

nomesDeClasse = [0, 1]

print ("Acuracia: %s" % classificador.score(X_teste, y_teste))
print ("F1: %s" % str(f1_score(y_teste, y_pred, average=None)))

plt.figure()
plot_confusion_matrix(matriz, classes=nomesDeClasse, title="Matriz de Confusao")

plt.show()


while True:
	nome = raw_input("Insira um nome: ")
	if nome == "":
		break
	nome = list(nome.upper().decode(encoding='utf-8'))

	probabilidades = classificador.predict_proba(vetorizador.transform([nome]))[0]
	print "Feminino: {}% de chance | Masculino: {}% de chance.".format(probabilidades[0]*100, probabilidades[1]*100)

