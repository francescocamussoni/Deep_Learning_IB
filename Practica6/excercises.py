import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

####################
         ####
            #
         ####
        #   #
         ####
####################
from sklearn.model_selection import train_test_split

carseats=pd.read_csv('carseats.csv', delimiter=',')
carseats['High']=[1 if carseats['Sales'][i]>=8 else 0 for i in range(len(carseats['Sales']))]
carseats=pd.get_dummies(carseats)

train, test = train_test_split(carseats, test_size=0.2, random_state=42, stratify=carseats['High'])

####################42
        #
        #
        #####
        #   #
        #####
####################
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import uniform

''' #Esto era para ver si habia alguna correlación clara entre las variables
attributes = ['CompPrice', 'Income', 'Advertising', 'Population', 'Price',
       'Age', 'Education']
corr_plot = sns.pairplot(carseats, hue='High', diag_kind= 'hist',
             vars=carseats[attributes],
             plot_kws=dict(alpha=0.5),
             diag_kws=dict(alpha=0.5))
plt.suptitle('Correlaciones entre variables y dependencia con la salida')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
corr_plot.savefig('Corr_b.pdf', format='pdf')
plt.show()
'''

x_train, y_train = train.drop(['High', 'Sales'], axis=1), train['High']
x_test, y_test = test.drop(['High', 'Sales'], axis=1), test['High']

TC=DecisionTreeClassifier()
TC.fit(x_train, y_train)
print('Resultados sobre los datos de training:')
y_true, y_pred = y_train, TC.predict(x_train)
print(classification_report(y_true, y_pred))

print('Resultados sobre los datos de test:')
y_true, y_pred = y_test, TC.predict(x_test)
print(classification_report(y_true, y_pred))

y_scores=TC.predict_proba(x_train)
fpr, tpr, thresholds = roc_curve(y_train, y_scores[:, 1])
print('AUC training ', roc_auc_score(y_train, y_scores[:, 1]))
y_scores_test=TC.predict_proba(x_test)
fpr_test, tpr_test, thresholds = roc_curve(y_test, y_scores_test[:, 1])
print('AUC test', roc_auc_score(y_test, y_scores_test[:, 1]))

fig = plt.figure(figsize=(9,6))
plt.plot(fpr, tpr, label='Training')
plt.plot(fpr_test, tpr_test, label='Test')
plt.title('ROC curve', fontsize=16)
plt.xlabel('False positive ratio', fontsize=14)
plt.ylabel('True positive ratio')
plt.legend(fontsize=14)
plt.savefig('ROC_b_1.pdf', format='pdf')
plt.show()

sns.reset_orig()
fig = plt.figure(figsize=(10,10))
tree.plot_tree(TC, filled=True)
#plt.suptitle('Arbol clasificador')
plt.savefig('Arbol_b_1.pdf', format='pdf', bbox_inches = "tight")
plt.show()
sns.set(style='whitegrid')

zip_iterator = zip(x_train.keys(), TC.feature_importances_)
feature_importance = dict(zip_iterator)
feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
for i in range(len(feature_importance)):
        print(str(i+1)+' '+feature_importance[i][0]+':', feature_importance[i][1])

##########################################
 #Hago una busqueda de parametros decente
##########################################

param_grid= [{'max_depth': [4, 5, 6, 7, 8, 9, 10], #primero hice una corrida con los params default y obtuve una profundidad de 10
                'ccp_alpha': uniform(loc=0, scale=1)
                }]

randm_search = RandomizedSearchCV(TC,
                        param_grid,
                        cv=5,
                        n_iter=100,
                        scoring='roc_auc',
                        return_train_score=True)

randm_search.fit(x_train, y_train)

#cvres =randm_search.cv_results_
#for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#    print(np.sqrt(mean_score), params) #los printeo

print('Mejores parámetros:')
print(randm_search.best_params_)
final_model=randm_search.best_estimator_

print('Resultados sobre los datos de training:')
y_true, y_pred = y_train, final_model.predict(x_train)
print(classification_report(y_true, y_pred))

print('Resultados sobre los datos de test:')
y_true, y_pred = y_test, final_model.predict(x_test)
print(classification_report(y_true, y_pred))

y_scores=final_model.predict_proba(x_train)
fpr, tpr, thresholds = roc_curve(y_train, y_scores[:, 1])
print('AUC training ', roc_auc_score(y_train, y_scores[:, 1]))
y_scores_test=final_model.predict_proba(x_test)
fpr_test, tpr_test, thresholds = roc_curve(y_test, y_scores_test[:, 1])
print('AUC test', roc_auc_score(y_test, y_scores_test[:, 1]))

fig = plt.figure(figsize=(9,6))
plt.plot(fpr, tpr, label='Training')
plt.plot(fpr_test, tpr_test, label='Test')
plt.title('ROC curve', fontsize=16)
plt.xlabel('False positive ratio', fontsize=14)
plt.ylabel('True positive ratio')
plt.legend(fontsize=14)
plt.savefig('ROC_b_2.pdf', format='pdf')
plt.show()

sns.reset_orig()
fig = plt.figure(figsize=(10,10))
tree.plot_tree(final_model, filled=True)
#plt.suptitle('Arbol clasificador')
plt.savefig('Arbol_b_2.pdf', format='pdf', bbox_inches = "tight")
plt.show()
sns.set(style='whitegrid')

zip_iterator = zip(x_train.keys(), final_model.feature_importances_)
feature_importance = dict(zip_iterator)
feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
for i in range(len(feature_importance)):
        print(str(i+1)+' '+feature_importance[i][0]+':', feature_importance[i][1])
####################
        #####
        #
        #
        #
        #####
####################
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
#from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from pandas.plotting import scatter_matrix
from sklearn.metrics import max_error, mean_squared_error
from scipy.stats import uniform

#Esto era para haber si habia una correlación clara entre las variables.
#Podría haberme quedado con los 3 atributos mas representativos a ver si mejoraba el modelo
attributes = ['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price',
       'Age', 'Education']
scatter_matrix(carseats[attributes], figsize=(12, 8))
plt.suptitle('Correlaciones de las variables numéricas')
plt.savefig('Corr_c.pdf', format='pdf')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
print(np.abs(carseats[attributes].corr()['Sales']).sort_values(ascending=False))

x_train, y_train = train.drop(['High', 'Sales'], axis=1), train['Sales']
x_test, y_test = test.drop(['High', 'Sales'], axis=1), test['Sales']

TR=DecisionTreeRegressor()
TR.fit(x_train, y_train)

print('\nResultados sobre los datos de training:')
y_true, y_pred = y_train, TR.predict(x_train)
print('Maximo error: ', max_error(y_true, y_pred))
print('RMSE: ', np.sqrt(mean_squared_error(y_true, y_pred)))

print('\nResultados sobre los datos de test:')
y_true_test, y_pred_test = y_test, TR.predict(x_test)
print('Maximo error: ', max_error(y_true_test, y_pred_test))
print('RMSE: ', np.sqrt(mean_squared_error(y_true_test, y_pred_test)))

sns.reset_orig()
tree.plot_tree(TR, filled=True)
#plt.suptitle('Arbol clasificador')
plt.savefig('Arbol_c.pdf', format='pdf', bbox_inches = "tight")
plt.show()
sns.set(style='whitegrid')

fig = plt.figure(figsize=(9,6))
plt.plot(y_true, y_true, color='C0', label='Identidad')
plt.plot(y_true_test, y_true_test, color='C0')
plt.scatter(y_true, y_pred, color='C1', label='Resultados de training', s=12, alpha=0.7)
plt.scatter(y_true_test, y_pred_test, color='C2', label='Resultados de test', s=12, alpha=0.7)
plt.title('Resultados para de predicción para training y test', fontsize=16)
plt.xlabel('Precio real', fontsize=14)
plt.ylabel('Precio estimado', fontsize=14)
plt.legend(fontsize=14)
plt.savefig('result_c.pdf', format='pdf')
plt.show()

zip_iterator = zip(x_train.keys(), TR.feature_importances_)
feature_importance = dict(zip_iterator)
feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
for i in range(len(feature_importance)):
        print(str(i+1)+' '+feature_importance[i][0]+':', feature_importance[i][1])
####################
        ######
        #    #
        ######
        #
        ######
####################
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
#from pandas.plotting import scatter_matrix
from sklearn.metrics import max_error, mean_squared_error
from scipy.stats import uniform

x_train, y_train = train.drop(['High', 'Sales'], axis=1), train['Sales']
x_test, y_test = test.drop(['High', 'Sales'], axis=1), test['Sales']

TR=DecisionTreeRegressor()

param_grid= [{'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], #En la corrida anterior obtuve profundidad 16
                'ccp_alpha': uniform(loc=0, scale=1)
                }]

randm_search = RandomizedSearchCV(TR,
                        param_grid,
                        cv=5,
                        n_iter=100,
                        scoring='neg_mean_squared_error',
                        return_train_score=True)

randm_search.fit(x_train, y_train)

print('Mejores parámetros:')
print(randm_search.best_params_)
final_model=randm_search.best_estimator_

#print('Mean test score')
#print(randm_search.cv_results_['mean_test_score'])

#cvres = randm_search.cv_results_
#for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#    print(np.sqrt(-mean_score), params) #los printeo

print('\nResultados sobre los datos de training:')
y_true, y_pred = y_train, final_model.predict(x_train)
print('Maximo error: ', max_error(y_true, y_pred))
print('RMSE: ', np.sqrt(mean_squared_error(y_true, y_pred)))

print('\nResultados sobre los datos de test:')
y_true, y_pred = y_test, final_model.predict(x_test)
print('Maximo error: ', max_error(y_true, y_pred))
print('RMSE: ', np.sqrt(mean_squared_error(y_true, y_pred)))

sns.reset_orig()
tree.plot_tree(final_model, filled=True)
#plt.suptitle('Arbol clasificador')
plt.savefig('Arbol_e.pdf', format='pdf', bbox_inches = "tight")
plt.show()
sns.set(style='whitegrid')

fig = plt.figure(figsize=(9,6))
plt.plot(y_true, y_true, color='C0', label='Identidad')
plt.plot(y_true_test, y_true_test, color='C0')
plt.scatter(y_true, y_pred, color='C1', label='Resultados de training', s=12, alpha=0.7)
plt.scatter(y_true_test, y_pred_test, color='C2', label='Resultados de test', s=12, alpha=0.7)
plt.title('Resultados para de predicción para training y test', fontsize=16)
plt.xlabel('Precio real', fontsize=14)
plt.ylabel('Precio estimado', fontsize=14)
plt.legend(fontsize=14)
plt.savefig('result_e.pdf', format='pdf')
plt.show()

zip_iterator = zip(x_train.keys(), final_model.feature_importances_)
feature_importance = dict(zip_iterator)
feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
for i in range(len(feature_importance)):
        print(str(i+1)+' '+feature_importance[i][0]+':', feature_importance[i][1])
#####################
        ######
        #
        ####
        #
        #
#####################
from sklearn import tree
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
#from pandas.plotting import scatter_matrix
from sklearn.metrics import max_error, mean_squared_error
from scipy.stats import uniform

x_train, y_train = train.drop(['High', 'Sales'], axis=1), train['Sales']
x_test, y_test = test.drop(['High', 'Sales'], axis=1), test['Sales']

BR=BaggingRegressor(final_model)

param_grid= [{'n_estimators': [50, 60, 70, 80, 90, 100], #En la corrida anterior obtuve profundidad 16
                'max_samples': uniform(loc=0, scale=1),
                'bootstrap': ['True'],
                }]

randm_search = RandomizedSearchCV(BR,
                        param_grid,
                        cv=5,
                        n_iter=100,
                        scoring='neg_mean_squared_error',
                        return_train_score=True)

randm_search.fit(x_train, y_train)

print('Mejores parámetros:')
print(randm_search.best_params_)
final_model=randm_search.best_estimator_

#print('Mean test score')
#print(randm_search.cv_results_['mean_test_score'])

#cvres = randm_search.cv_results_
#for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#    print(np.sqrt(-mean_score), params) #los printeo

print('\nResultados sobre los datos de training:')
y_true, y_pred = y_train, final_model.predict(x_train)
print('Maximo error: ', max_error(y_true, y_pred))
print('RMSE: ', np.sqrt(mean_squared_error(y_true, y_pred)))

print('\nResultados sobre los datos de test:')
y_true_test, y_pred_test = y_test, final_model.predict(x_test)
print('Maximo error: ', max_error(y_true_test, y_pred_test))
print('RMSE: ', np.sqrt(mean_squared_error(y_true_test, y_pred_test)))


fig = plt.figure(figsize=(9,6))
plt.plot(y_true, y_true, color='C0', label='Identidad')
plt.plot(y_true_test, y_true_test, color='C0')
plt.scatter(y_true, y_pred, color='C1', label='Resultados de training', s=12, alpha=0.7)
plt.scatter(y_true_test, y_pred_test, color='C2', label='Resultados de test', s=12, alpha=0.7)
plt.title('Resultados para de predicción para training y test', fontsize=16)
plt.xlabel('Precio real', fontsize=14)
plt.ylabel('Precio estimado', fontsize=14)
plt.legend(fontsize=14)
plt.savefig('result_f.pdf', format='pdf')
plt.show()

'''No existe esto para bagging
sns.reset_orig()
tree.plot_tree(final_model, filled=True)
#plt.suptitle('Arbol clasificador')
plt.savefig('Arbol_e.pdf', format='pdf', bbox_inches = "tight")
plt.show()
sns.set(style='whitegrid')
'''
feature_importance = np.mean([
    tree.feature_importances_ for tree in final_model.estimators_
], axis=0)

zip_iterator = zip(x_train.keys(), feature_importance)
feature_importance = dict(zip_iterator)
feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
for i in range(len(feature_importance)):
        print(str(i+1)+' '+feature_importance[i][0]+':', feature_importance[i][1])


######################
         #####
        #    #
         #####
             #
         ####
######################
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
#from pandas.plotting import scatter_matrix
from sklearn.metrics import max_error, mean_squared_error
from scipy.stats import uniform

x_train, y_train = train.drop(['High', 'Sales'], axis=1), train['Sales']
x_test, y_test = test.drop(['High', 'Sales'], axis=1), test['Sales']

RFR=RandomForestRegressor()

param_grid= [{'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], #En la corrida anterior obtuve profundidad 16
                'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                'ccp_alpha': uniform(loc=0, scale=0.01),
                'max_features': uniform(loc=0.5, scale=0.5)
                }]

randm_search = RandomizedSearchCV(RFR,
                        param_grid,
                        cv=5,
                        n_iter=1000,
                        scoring='neg_mean_squared_error',
                        return_train_score=True)

randm_search.fit(x_train, y_train)

print('Mejores parámetros:')
print(randm_search.best_params_)
final_model=randm_search.best_estimator_

#print('Mean test score')
#print(randm_search.cv_results_['mean_test_score'])

#cvres = randm_search.cv_results_
#for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#    print(np.sqrt(-mean_score), params) #los printeo

print('\nResultados sobre los datos de training:')
y_true, y_pred = y_train, final_model.predict(x_train)
print('Maximo error: ', max_error(y_true, y_pred))
print('RMSE: ', np.sqrt(mean_squared_error(y_true, y_pred)))

print('\nResultados sobre los datos de test:')
y_true_test, y_pred_test = y_test, final_model.predict(x_test)
print('Maximo error: ', max_error(y_true_test, y_pred_test))
print('RMSE: ', np.sqrt(mean_squared_error(y_true_test, y_pred_test)))

'''
sns.reset_orig()
tree.plot_tree(final_model, filled=True)
#plt.suptitle('Arbol clasificador')
plt.savefig('Arbol_g.pdf', format='pdf', bbox_inches = "tight")
plt.show()
sns.set(style='whitegrid')
'''

fig = plt.figure(figsize=(9,6))
plt.plot(y_true, y_true, color='C0', label='Identidad')
plt.plot(y_true_test, y_true_test, color='C0')
plt.scatter(y_true, y_pred, color='C1', label='Resultados de training', s=12, alpha=0.7)
plt.scatter(y_true_test, y_pred_test, color='C2', label='Resultados de test', s=12, alpha=0.7)
plt.title('Resultados para de predicción para training y test', fontsize=16)
plt.xlabel('Precio real', fontsize=14)
plt.ylabel('Precio estimado', fontsize=14)
plt.legend(fontsize=14)
plt.savefig('result_g.pdf', format='pdf')
plt.show()

zip_iterator = zip(x_train.keys(), final_model.feature_importances_)
feature_importance = dict(zip_iterator)
feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
for i in range(len(feature_importance)):
        print(str(i+1)+' '+feature_importance[i][0]+':', feature_importance[i][1])

ccp_alpha=randm_search.best_params_['ccp_alpha']
max_depth=randm_search.best_params_['max_depth']
n_estimators=randm_search.best_params_['n_estimators']
max_features=randm_search.best_params_['max_features']

error_test=[]
for i in range(len(x_train.keys())):
        RFR=RandomForestRegressor(ccp_alpha=ccp_alpha,
                                max_depth=max_depth,
                                n_estimators=n_estimators,
                                max_features=i+1)
        RFR.fit(x_train, y_train)
        y_true_test, y_pred_test = y_test, RFR.predict(x_test)
        error_test.append(np.sqrt(mean_squared_error(y_true_test, y_pred_test)))
fig=plt.figure(figsize=(9,6))
plt.scatter(range(len(x_train.keys())), error_test)
plt.plot(range(len(x_train.keys())), error_test)
plt.title('Error de test en funcion de max_features', fontsize=16)
plt.xlabel('max_features', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.savefig('g_error_max_feature.pdf', format='pdf')
plt.show()

error_test=[]
range_depth=50
for i in range(range_depth):
        RFR=RandomForestRegressor(ccp_alpha=ccp_alpha,
                                max_depth=i+1,
                                n_estimators=n_estimators,
                                max_features=max_features)
        RFR.fit(x_train, y_train)
        y_true_test, y_pred_test = y_test, RFR.predict(x_test)
        error_test.append(np.sqrt(mean_squared_error(y_true_test, y_pred_test)))
fig=plt.figure(figsize=(9,6))
plt.scatter(range(range_depth), error_test)
plt.plot(range(range_depth), error_test)
plt.title('Error de test en funcion de max_features', fontsize=16)
plt.xlabel('max_features', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.savefig('g_error_max_depth.pdf', format='pdf')
plt.show()

######################
         #
         #
         #####
         #   #
         #   #
######################
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
#from pandas.plotting import scatter_matrix
from sklearn.metrics import max_error, mean_squared_error
from scipy.stats import uniform

x_train, y_train = train.drop(['High', 'Sales'], axis=1), train['Sales']
x_test, y_test = test.drop(['High', 'Sales'], axis=1), test['Sales']

ABR=AdaBoostRegressor()

param_grid= [{'base_estimator': final_model,
                'learning_rate': uniform(loc=0, scale=5)
                }]

randm_search = RandomizedSearchCV(ABR,
                        param_grid,
                        cv=5,
                        n_iter=100,
                        scoring='neg_mean_squared_error',
                        return_train_score=True)

randm_search.fit(x_train, y_train)

print('Mejores parámetros:')
print(randm_search.best_params_)
final_model=randm_search.best_estimator_

#print('Mean test score')
#print(randm_search.cv_results_['mean_test_score'])

#cvres = randm_search.cv_results_
#for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#    print(np.sqrt(-mean_score), params) #los printeo

print('\nResultados sobre los datos de training:')
y_true, y_pred = y_train, final_model.predict(x_train)
print('Maximo error: ', max_error(y_true, y_pred))
print('RMSE: ', np.sqrt(mean_squared_error(y_true, y_pred)))

print('\nResultados sobre los datos de test:')
y_true_test, y_pred_test = y_test, final_model.predict(x_test)
print('Maximo error: ', max_error(y_true_test, y_pred_test))
print('RMSE: ', np.sqrt(mean_squared_error(y_true_test, y_pred_test)))

'''
sns.reset_orig()
tree.plot_tree(final_model, filled=True)
#plt.suptitle('Arbol clasificador')
plt.savefig('Arbol_g.pdf', format='pdf', bbox_inches = "tight")
plt.show()
sns.set(style='whitegrid')
'''

fig = plt.figure(figsize=(9,6))
plt.plot(y_true, y_true, color='C0', label='Identidad')
plt.plot(y_true_test, y_true_test, color='C0')
plt.scatter(y_true, y_pred, color='C1', label='Resultados de training', s=12, alpha=0.7)
plt.scatter(y_true_test, y_pred_test, color='C2', label='Resultados de test', s=12, alpha=0.7)
plt.title('Resultados para de predicción para training y test', fontsize=16)
plt.xlabel('Precio real', fontsize=14)
plt.ylabel('Precio estimado', fontsize=14)
plt.legend(fontsize=14)
plt.savefig('result_h.pdf', format='pdf')
plt.show()

zip_iterator = zip(x_train.keys(), final_model.feature_importances_)
feature_importance = dict(zip_iterator)
feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
for i in range(len(feature_importance)):
        print(str(i+1)+' '+feature_importance[i][0]+':', feature_importance[i][1])


