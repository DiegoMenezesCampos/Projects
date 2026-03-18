import pandas as pd
import numpy as np
import os as os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, 
                                     GridSearchCV, 
                                     StratifiedShuffleSplit,
                                     StratifiedKFold)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             precision_score,
                             recall_score,
                             f1_score,
                             auc,
                             roc_auc_score,
                             roc_curve,
                             classification_report)
from sklearn import metrics
import argparse
import pickle

sns.set()
pd.options.mode.chained_assignment = None

#defining arguments

parser = argparse.ArgumentParser(
    description='Argumentos para rodar o processamento de arquivos HDF para treinar modelo de machine learning.')

parser.add_argument('--n_gases', dest="n_gases", type=int, required = True,
                    help='Numero de gases utilizado. Ex: 3')
parser.add_argument('--gases_escolhidos', dest="chosen_gases", type=str, required = True,
                    help='Gases que serão utilizados ex: "1,2,3"')
parser.add_argument('--precisa_treinar', dest="needs_training", type=bool, required = True,
                    help='variavel para decidir se vai treinar ou nao ex: True or False')
parser.add_argument('--num_dias_para_treino', dest="training_days", type=int, required = True,
                    help='Numero de dias para usar para treinar o modelo e para verificar qual modelo usar')
parser.add_argument('--num_dias_utilizados_para_cada_previsao', dest="num_days_used_for_each_prediction", type=int, required = True,
                    help='Numero de dias usado para iterar e prever (3 dias prever o 4) ex: 3')


args = parser.parse_args()

print("Diretorio Atual: {0}".format(os.getcwd()))

print("\nCarregando dados.....")

# Loading the test .csv and getting user inputs
 
data = pd.read_csv('../Dataframes_criados/df_final_para_previsao.csv', sep=',')

n_gases = args.n_gases
gases = args.chosen_gases.split(",")

#df_lists will contain the dataframes of the selected gases

if len(gases) == n_gases:
     #getting only the columns of the gases defined in the gases list
     list_columns_to_use=[]
     for column_index in range(0,len(data.columns)):
          for gas_num in gases:
               if "gas_" + str(gas_num) in data.columns[column_index]:
                    list_columns_to_use.append(data.columns[column_index])
     
     list_columns_to_use.append("date")
     list_columns_to_use.append("choveu")
else:
     print("\nErro! Numero de gases informados nao bate\n")

interpolated_df = []

final_dataset = data[list_columns_to_use]
final_dataset['date']= final_dataset['date'].replace('-','/', regex = True)
final_dataset['date'] = pd.to_datetime(final_dataset['date'])
final_dataset.sort_values(by=['date'], inplace = True)
final_dataset = final_dataset.reset_index(drop=True)
final_dataset = final_dataset.set_index("date")
#interpolated_df = final_dataset[final_dataset.columns.difference(['date', 'choveu'])].interpolate()
interpolated_df = final_dataset.fillna(final_dataset.mean())
interpolated_df['choveu'] = pd.Series(final_dataset['choveu'])


# Now, let's separate the train test split.
training_days = args.training_days

X = interpolated_df.loc[:, interpolated_df.columns != "choveu"].head(training_days)
Y = interpolated_df.choveu.head(training_days)
scaler = StandardScaler().fit(X)
X_transformed = pd.DataFrame(scaler.transform(X), columns = X.columns, index = X.index)
X_train, X_test, Y_train, Y_test = train_test_split(X_transformed,
                                                    Y,
                                                    test_size=0.25,
                                                    stratify= Y,
                                                    random_state = 12)

# Separating the date from the index and creating the necessary variables
interpolated_df = interpolated_df.reset_index(level=0)
LR_accuracy = []
NB_accuracy = []
SVM_accuracy = []
LR_predict = []
NB_predict = []
SVM_predict = []
LR_result = []
NB_result = []
SVM_result = []
predicted_indexes = []
Pkl_LR = f"Modelos_Exportados/LR_Model_{n_gases}_gases.pkl"
Pkl_NB = f"Modelos_Exportados/NB_Model_{n_gases}_gases.pkl"
Pkl_SVM = f"Modelos_Exportados/SVM_Model_{n_gases}_gases.pkl"

###### train models  ###########
needs_training = args.needs_training
# Logistic Regression 
if needs_training==True:
     lr=LogisticRegression()
     log_regression=lr.fit(X_train,Y_train)
    
     with open(Pkl_LR, 'wb') as file:  
          pickle.dump(log_regression, file)

     # Naive Bayes 

     nb = GaussianNB()
     naive_bayes_model = nb.fit(X_train,Y_train)
     
     with open(Pkl_NB, 'wb') as file:  
          pickle.dump(naive_bayes_model, file)

     # Support Vector Machines

     kernel = "linear"
     SVM = svm.SVC(kernel=kernel)
     svm_model = SVM.fit(X_train,Y_train)
     
     with open(Pkl_SVM, 'wb') as file:  
          pickle.dump(svm_model, file)

else:
     print("Usuario nao vai fazer treino dos dados, entao ja deve possuir os modelos que precisa em pickle")


num_days_used_for_each_prediction = args.num_days_used_for_each_prediction
###### Performing the loop to store hits and errors considering each four days
i = interpolated_df.index[0]
while (i) < interpolated_df.index[len(interpolated_df.index)-(num_days_used_for_each_prediction - 1)]:
     
     if interpolated_df.loc[i:i+num_days_used_for_each_prediction, interpolated_df.columns == "date"].isnull().values.any() == False:
            
          # Creating X, Y and X_transformed
          X = interpolated_df.loc[i:i+num_days_used_for_each_prediction, interpolated_df.columns != "choveu"]
          X = X.set_index("date")
          Y = interpolated_df.loc[i:i+num_days_used_for_each_prediction, ['choveu','date']]
          Y = Y.set_index("date")
          scaler = StandardScaler().fit(X)
          X_transformed = pd.DataFrame(scaler.transform(X), columns = X.columns, index = X.index)
          
          with open(Pkl_LR, 'rb') as file:
               log_regression = pickle.load(file)
          result = log_regression.predict(X_transformed)
          LR_predict.append(result[-1])
          LR_accuracy.append(result[num_days_used_for_each_prediction-1] == Y.iloc[num_days_used_for_each_prediction-1,0])
          LR_result.append(result[num_days_used_for_each_prediction-1])
          
          with open(Pkl_NB, 'rb') as file:
               nb = pickle.load(file)
          result = nb.predict(X_transformed)
          NB_predict.append(result[-1])
          NB_accuracy.append(result[num_days_used_for_each_prediction-1] == Y.iloc[num_days_used_for_each_prediction-1,0])
          NB_result.append(result[num_days_used_for_each_prediction-1])
          
          with open(Pkl_SVM, 'rb') as file:
               SVM = pickle.load(file)
          result = SVM.predict(X_transformed)
          SVM_predict.append(result[-1])
          SVM_accuracy.append(result[num_days_used_for_each_prediction-1] == Y.iloc[num_days_used_for_each_prediction-1,0])
          SVM_result.append(result[num_days_used_for_each_prediction-1])
          
          predicted_indexes.append(X.index[num_days_used_for_each_prediction])
          
          i = i + 1
     else:
          print('Erro: Não existem previsões de todos os dias, impossível prever valores para a data ' + str(interpolated_df.loc[[i+num_days_used_for_each_prediction], interpolated_df.columns == "date"].values))

real_result = []
for i in range(0,(len(list(interpolated_df.choveu)))):
     if i < (num_days_used_for_each_prediction - 1):
          list(interpolated_df.choveu)[i] = list(interpolated_df.choveu)[i]
     else:
          real_result.append(list(interpolated_df.choveu)[i])

# Confusion matrix, FAR, FRR and EER for Logistic Regression

confusion_matrix = metrics.confusion_matrix(real_result, LR_result)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.savefig('Confusion_Matrix_LR.jpeg', bbox_inches='tight')
tn, fp, fn, tp = metrics.confusion_matrix(real_result, LR_result).ravel()
FAR_LR = fp/(fp + tn)
FRR_LR = fn/(fn + tp)
ERR_LR = (FAR_LR + FRR_LR)/2

# Plotting Logistic Regression

fig, ax = plt.subplots()
ax.step(predicted_indexes, LR_accuracy)
plt.ylabel('1 - Acerto, 0 - Erro',fontsize = 18)
plt.xlabel('Dia', fontsize=18)
ax.set_title('Regressão Logística', fontsize = 18)
plt.savefig('Graficos/Regressão_Logística_data.jpeg', bbox_inches='tight')

fig, ax = plt.subplots()
ax.step(range(0,len(LR_accuracy)), LR_accuracy)
plt.ylabel('1 - Acerto, 0 - Erro',fontsize = 18)
plt.xlabel('Dia', fontsize=18)
ax.set_title('Regressão Logística', fontsize = 18)
plt.savefig('Graficos/Regressão_Logística_dia.jpeg', bbox_inches='tight')

# Confusion matrix, FAR, FRR and EER for Naive Bayes

confusion_matrix = metrics.confusion_matrix(real_result, NB_result)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.savefig('Confusion_Matrix_NB.jpeg', bbox_inches='tight')
tn, fp, fn, tp = metrics.confusion_matrix(real_result, NB_result).ravel()
FAR_NB = fp/(fp + tn)
FRR_NB = fn/(fn + tp)
ERR_NB = (FAR_NB + FRR_NB)/2

#Plotting Naive Bayes

fig, ax = plt.subplots()
ax.step(predicted_indexes, NB_accuracy)
plt.ylabel('1 - Acerto, 0 - Erro',fontsize = 18)
plt.xlabel('Dia', fontsize=18)
ax.set_title('Naive Bayes', fontsize = 18)
plt.savefig('Graficos/Naive_Bayes_data.jpeg', bbox_inches='tight')

fig, ax = plt.subplots()
ax.step(range(0,len(NB_accuracy)), NB_accuracy)
plt.ylabel('1 - Acerto, 0 - Erro',fontsize = 18)
plt.xlabel('Dia', fontsize=18)
ax.set_title('Naive Bayes', fontsize = 18)
plt.savefig('Graficos/Naive_Bayes_dia.jpeg', bbox_inches='tight')

# Confusion matrix, FAR, FRR and EER for SVM

confusion_matrix = metrics.confusion_matrix(real_result, SVM_result)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.savefig('Confusion_Matrix_SVM.jpeg', bbox_inches='tight')
tn, fp, fn, tp = metrics.confusion_matrix(real_result, SVM_result).ravel()
FAR_SVM = fp/(fp + tn)
FRR_SVM = fn/(fn + tp)
ERR_SVM = (FAR_SVM + FRR_SVM)/2

#Plotting SVM

fig, ax = plt.subplots()
ax.step(predicted_indexes, SVM_accuracy)
plt.ylabel('1 - Acerto, 0 - Erro',fontsize = 18)
plt.xlabel('Dia', fontsize=18)
ax.set_title('Support Vector Machines', fontsize = 18)
plt.savefig('Graficos/SVM_data.jpeg', bbox_inches='tight')

fig, ax = plt.subplots()
ax.step(range(0,len(SVM_accuracy)), SVM_accuracy)
plt.ylabel('1 - Acerto, 0 - Erro',fontsize = 18)
plt.xlabel('Dia', fontsize=18)
ax.set_title('Support Vector Machines', fontsize = 18)
plt.savefig('Graficos/SVM_dia.jpeg', bbox_inches='tight')

# Dataframe for FAR, FRR and EER

d = {'FAR': [FAR_LR, FAR_NB, FAR_SVM], 'FRR': [FRR_LR, FRR_NB, FRR_SVM], 'ERR': [ERR_LR, ERR_NB, ERR_SVM]}
Errors_Dataframe = pd.DataFrame(data = d, index = ['Regressão Logística', 'Naive Bayes', 'Support Vector Machines'])
print(Errors_Dataframe)
Errors_Dataframe.to_csv(r'C:\Users\diego\OneDrive\Área de Trabalho\Freelance\Concentração de Gases em Porto Velho\arquivos principais\dataframe_erros.csv')

# Selecting the best accuracy

LR_accuracy = sum(LR_accuracy)/len(LR_accuracy)
NB_accuracy = sum(NB_accuracy)/len(NB_accuracy)
SVM_accuracy = sum(SVM_accuracy)/len(SVM_accuracy)
accuracy_dict = {'Regressão Logística':LR_accuracy, 'Naive Bayes':NB_accuracy, 'SVM':SVM_accuracy}
print('Modelo de Melhor Acurácia: ' + max(accuracy_dict, key=accuracy_dict.get) + '\nAcurácia: ' + str(100*accuracy_dict[max(accuracy_dict, key=accuracy_dict.get)]))

# Creating accuracy dataframe
Accuracies = [LR_accuracy, NB_accuracy, SVM_accuracy]
Accuracies_Dataframe = pd.DataFrame(Accuracies,columns = ['Acurácia'], index = ['Regressão Logística', 'Naive Bayes', 'Support Vector Machines'])
Accuracies_Dataframe.to_csv('Dataframes_resultado/acuracias.csv')

# Plotting comparison

fig, ax = plt.subplots()
bars = plt.bar(['Regressão Logística', 'Naive Bayes', 'Support Vector Machines'], Accuracies_Dataframe['Acurácia'].values)
ax.bar_label(bars)
plt.xlabel("Métodos")
plt.ylabel("Acurácia")
plt.title("Comparação dos métodos")
plt.savefig('Comparação.png', bbox_inches='tight')

# Creating prediction dataframes
list_LR_predict_with_indexes = []
list_NB_predict_with_indexes = []
list_SVM_predict_with_indexes = []

for x in range(0, len(predicted_indexes)):
    list_LR_predict_with_indexes.append([predicted_indexes[x],LR_predict[x]]) 
    list_NB_predict_with_indexes.append([predicted_indexes[x],NB_predict[x]]) 
    list_SVM_predict_with_indexes.append([predicted_indexes[x],SVM_predict[x]]) 

df_predict_LR = pd.DataFrame(list_LR_predict_with_indexes, columns=['date','predicao_choveu'])
df_predict_LR.to_csv('Dataframes_resultado/dataframe_predict_LR.csv')

df_predict_NB = pd.DataFrame(list_NB_predict_with_indexes, columns=['date','predicao_choveu'])
df_predict_NB.to_csv('Dataframes_resultado/dataframe_predict_NB.csv')

df_predict_SVM = pd.DataFrame(list_SVM_predict_with_indexes, columns=['date','predicao_choveu'])
df_predict_SVM.to_csv('Dataframes_resultado/dataframe_predict_SVM.csv')