#!/usr/bin/env python
# coding: utf-8

# In[1]:


# performing linear algebra 
import numpy as np 

# data processing 
import pandas as pd 
pd.set_option("display.max_columns", None)

# visualisation 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[2]:


df = pd.read_excel(r"C:\Users\Admin\Downloads\WA_Fn-UseC_-HR-Employee-Attrition.xlsx", header=0) 
df.head()


# In[3]:


df.info()


# In[4]:


df.describe(include="all")


# In[5]:


sns.distplot(df["Age"])


# In[6]:


sns.distplot(df["DailyRate"])


# In[7]:


sns.set_style('darkgrid') 
sns.countplot(x ='Attrition', data = df) 


# In[8]:


import plotly.express as px
fig = px.histogram(df, x='Age', color='Attrition', barmode='group', nbins=20, title='Attrition by Age')
fig.update_layout(xaxis_title='Age', yaxis_title='Count')
fig.show()


# In[9]:


fig = px.sunburst(df,path=['Gender','Attrition'], title='Gender Distribution')
fig.update_traces(textinfo='label+percent parent')
fig.update_layout(title_x=0.5)
fig.show()


# In[10]:


plt.figure(figsize=(10,4))
sns.countplot(x="JobLevel", hue="Attrition", data=df, palette='pink')
plt.title("Attrition by Job Level")
plt.show()


# In[11]:


plt.figure(figsize=(10,4))
sns.countplot(x="Department", hue="Attrition", data=df, palette ='husl')
plt.title("Attrition by Department")
plt.show()


# In[12]:


for i in df.columns:
    print({i:df[i].unique()})


# In[13]:


df.drop('EmployeeCount', axis = 1, inplace = True) 
df.drop('StandardHours', axis = 1, inplace = True) 
df.drop('EmployeeNumber', axis = 1, inplace = True) 
df.drop('Over18', axis = 1, inplace = True) 

print(df.shape)


# In[14]:


df.isnull().sum()


# In[15]:


colname=[]
for x in df.columns:
    if df[x].dtype=='object':
        colname.append(x)
colname


# In[16]:


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for x in colname:
    df[x]=le.fit_transform(df[x])


# In[17]:


df.head()


# In[18]:


X=df.drop(["Attrition"],axis=1)
Y=df["Attrition"]


# In[19]:


print(X.shape)
print(Y.shape)


# In[ ]:





# In[ ]:





# In[20]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)
X = scaler.transform(X)


# In[22]:


print(X)


# In[23]:


from sklearn.model_selection import train_test_split

#Split the data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,stratify=Y,
                                                    random_state=10)  


# In[24]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[25]:


from sklearn.linear_model import LogisticRegression
#create a model
classifier=LogisticRegression()
#fitting training data to the model
classifier.fit(X_train,Y_train)


# In[26]:


classifier.score(X_train,Y_train)


# In[27]:


Y_pred=classifier.predict(X_test)
print(Y_pred)


# In[29]:


Y_pred_prob=classifier.predict_proba(X_test)
np.set_printoptions(suppress=True)
Y_pred_prob


# In[30]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)


# In[31]:


from sklearn.tree import DecisionTreeClassifier

model_DT=DecisionTreeClassifier(random_state=10, 
                                         criterion="gini")
#min_samples_leaf, min_samples_split, max_depth, max_features, max_leaf_nodes

#fit the model on the data and predict the values
model_DT.fit(X_train,Y_train)
Y_pred=model_DT.predict(X_test)
#print(Y_pred)
#print(list(zip(Y_test,Y_pred)))


# In[32]:


model_DT.score(X_train,Y_train)


# In[33]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)


# In[34]:


from sklearn.ensemble import RandomForestClassifier

model_RandomForest=RandomForestClassifier(n_estimators=100,
                                          random_state=10, bootstrap=True,
                                         n_jobs=-1)

#fit the model on the data and predict the values
model_RandomForest.fit(X_train,Y_train)

Y_pred=model_RandomForest.predict(X_test)


# In[35]:


model_RandomForest.score(X_train,Y_train)


# In[36]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)


# In[37]:


df.Attrition.value_counts()


# In[38]:


print("Before OverSampling, counts of label '1': ", (sum(Y_train == 1)))
print("Before OverSampling, counts of label '0': ", (sum(Y_train == 0)))
  
# import SMOTE from imblearn library
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 10,k_neighbors=5)
X_train_res, Y_train_res = sm.fit_resample(X_train, Y_train)
  
print('After OverSampling, the shape of train_X: ', (X_train_res.shape))
print('After OverSampling, the shape of train_y: ', (Y_train_res.shape))
  
print("After OverSampling, counts of label '1': ", (sum(Y_train_res == 1)))
print("After OverSampling, counts of label '0': ", (sum(Y_train_res == 0)))


# In[39]:


from sklearn.linear_model import LogisticRegression
#create a model
classifier=LogisticRegression()
#fitting training data to the model
classifier.fit(X_train_res,Y_train_res)

#print(classifier.intercept_)
#print(classifier.coef_)


# In[40]:


classifier.score(X_train_res,Y_train_res)


# In[41]:


Y_pred=classifier.predict(X_test)


# In[42]:


Y_pred_prob=classifier.predict_proba(X_test)
Y_pred_prob


# In[43]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)


# In[44]:


from sklearn.tree import DecisionTreeClassifier

model_DT=DecisionTreeClassifier(random_state=10, 
                                         criterion="gini")
#min_samples_leaf, min_samples_split, max_depth, max_features, max_leaf_nodes

#fit the model on the data and predict the values
model_DT.fit(X_train_res,Y_train_res)
Y_pred=model_DT.predict(X_test)
#print(Y_pred)
#print(list(zip(Y_test,Y_pred)))


# In[45]:


model_DT.score(X_train_res,Y_train_res)


# In[46]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)


# In[47]:


model_DT.feature_importances_


# In[48]:


model_DT=DecisionTreeClassifier(random_state=10, 
                                         criterion="gini") #fixed parameters should be passsed here

#parameters for trial and error should be passed here
parameter_space = {
           
    'max_depth':[10,15,20],
    'min_samples_leaf':[1,2,3,4,5,6,7]
    }
from sklearn.model_selection import GridSearchCV #RandomizedSearchCV
clf = GridSearchCV(model_DT, parameter_space, n_jobs=-1, cv=5)


# In[49]:


clf.fit(X_train_res,Y_train_res)


# In[50]:


print('Best parameters found:\n', clf.best_params_)


# In[51]:


clf.best_score_ #accuracy of the best params using the 5-fold CV


# In[52]:


Y_pred=clf.predict(X_test)


# In[53]:


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))


# In[ ]:





# In[54]:


from sklearn.ensemble import RandomForestClassifier

model_RandomForest=RandomForestClassifier(n_estimators=100,
                                          random_state=10, bootstrap=True,
                                         n_jobs=-1)

#fit the model on the data and predict the values
model_RandomForest.fit(X_train_res,Y_train_res)

Y_pred=model_RandomForest.predict(X_test)


# In[55]:


model_RandomForest.score(X_train_res,Y_train_res)


# In[56]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)


# In[ ]:


model_RandomForest=RandomForestClassifier(random_state=10, bootstrap=True,
                                         n_jobs=-1) #fixed parameters should be passsed here

#parameters for trial and error should be passed here
parameter_space = {
    'n_estimators':[100,200,300,500,1000],       #np.arange(100, 1001,50),
    'max_depth':[10,15,20],
    'min_samples_leaf':[1,3,4,5,6,7]
    }
from sklearn.model_selection import GridSearchCV #RandomizedSearchCV
clf = GridSearchCV(model_RandomForest, parameter_space, n_jobs=-1, cv=5)


# In[ ]:


clf.fit(X_train_res,Y_train_res)


# In[ ]:


print('Best parameters found:\n', clf.best_params_)


# In[ ]:


clf.best_score_ #accuracy of the best params using the 5-fold CV


# In[ ]:


Y_pred=clf.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))


# In[57]:


from sklearn.ensemble import GradientBoostingClassifier

model_GradientBoosting=GradientBoostingClassifier(n_estimators=100,
                                                  random_state=10)

#fit the model on the data and predict the values
model_GradientBoosting.fit(X_train_res,Y_train_res)

Y_pred=model_GradientBoosting.predict(X_test)


# In[58]:


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))


# In[59]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn = knn.fit(X_train_res, Y_train_res)
Y_pred=knn.predict(X_test)


# In[60]:


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))


# In[61]:


from sklearn.svm import SVC

knn = SVC()
knn = knn.fit(X_train_res, Y_train_res)
Y_pred=knn.predict(X_test)


# In[62]:


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))


# In[ ]:




