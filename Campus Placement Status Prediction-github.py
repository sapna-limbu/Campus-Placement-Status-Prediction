#!/usr/bin/env python
# coding: utf-8

# # Campus Placement Status Prediction

# # Business Problems and data understanding
# ## Objective:
# Predict the likelihood of campus placement success for students based on their academic performance, background, and other relevant factors.
# 
# ## Constraints:
# Optimize the performance of the predictive models in terms of accuracy, precision, recall, and other relevant evaluation metrics to ensure reliable predictions and actionable insights for campus placement management.
#
# In[43]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder


# In[10]:


df = pd.read_csv("Placement_Data.csv")


# In[11]:


df.head(5)


# In[34]:


df.tail(5)


# In[4]:


# as salary and sl_no columns are not required for placement status prediction so we drop it
df.drop(['sl_no', ], axis=1, inplace=True)


# In[5]:


df.head(2)


# In[5]:


df.shape


# In[7]:


df.info()


# ## EDA

# In[ ]:





# In[106]:


# checking distributions of all features
fig, axs = plt.subplots(ncols=6, nrows=3, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k, v in df.items():
    sns.distplot(v, ax=axs[index])
    index += 1

# Since you have 18 plots, and you want to remove the last one,
# you need to access index 17, not index 18, which is out of bounds
fig.delaxes(axs[index - 1])  # Corrected index
plt.tight_layout(pad=0.3, w_pad=0.2, h_pad=4.5)
plt.show()


# In[ ]:





# In[23]:


df["degree_t"].value_counts().plot.bar()
plt.show()


# From the above plot we can see most students have degrees in Commerce and Management, followed by Science and Technology.

# In[ ]:





# In[24]:


df["status"].value_counts().plot.bar()
plt.show()


# Number of Placed students is above 140 and Not Placed is below 80

# In[ ]:





# In[25]:


plt.figure(figsize=(5,4))
sns.countplot(x='specialisation', data=df)
plt.title("specialisation")
plt.xticks(rotation=90)
#plt.text(5,6,"Hello")
plt.show()


# Most of the students have done Specialisation in Marketing and Finance

# In[ ]:





# In[26]:


# Categorical vs Categorical bivariate
# hue => Categorical column. 
plt.figure(figsize=(5,4))
sns.countplot(x='status',hue='degree_t', data=df)
plt.title("Sales By Region")
plt.xticks(rotation=90)
plt.show()


# Above plot illustrates the relationship between Status and Degree:
# - From the plots above, we observe that the majority of students are placed and hold degrees in Commerce and Management.
# - The count of placed students is higher than that of students who are not placed.

# In[ ]:





# In[27]:


df["degree_t"].value_counts().plot.pie(autopct="%.1f%%")
plt.show()


# Above pie chart shows that the percentage of Degree holder in different branches:
# - Commerce and Management has 67.4%, Science&Technology 27.4% and Others 5.1%

# In[ ]:





# In[28]:


sns.distplot(df['salary'], kde=True)
plt.title("salary")
plt.show()


# Most individuals fall within a specific salary range, with fewer outliers earning significantly more or less.

# In[ ]:





# In[29]:


# hue converts univariate to Bivariate 
sns.kdeplot(x='salary', data=df, hue='degree_t')
plt.title("KDE plot salary& degree_t ")
plt.show()


# Commerce & Management degree holders tend to have a higher salary peak compared to the other two categories.

# In[ ]:





# In[30]:


df.boxplot(column="salary")
plt.show()


# - In this boxplot the green box represents the interquartile range (IQR), where the middle 50% of salaries fall.
#   Inside the box, a line marks the median salary value.
# 
# - Above the whiskers, several circles represent outliers—individuals who earn significantly more than the majority.
# - Most salaries cluster between approximately 200,000 and 300,000, with some outliers reaching up to 900,000.

# In[ ]:





# In[32]:


plt.figure(figsize=(10,5))
sns.violinplot(x='salary', data=df)
plt.show()


# In[ ]:





# In[37]:


sns.pairplot(df)
plt.show()


# In[ ]:





# In[39]:


sns.pairplot(df, hue='gender')
plt.show()


# In[ ]:





# In[43]:


# Bivariate
sns.violinplot(x='workex',y='etest_p', data=df)
plt.show()


# In[ ]:





# In[44]:


# Bivariate
sns.violinplot(x='degree_t',y='salary', data=df)
plt.show()


# In[ ]:





# ## Heatmap

# In[86]:


df.corr()


# In[ ]:





# In[55]:


np.triu(df.corr().values)


# In[87]:


sns.heatmap(df.corr(), cmap='Reds', annot=True)
plt.show()


# This heatmap provides insights into how different educational and test performance variables correlate with salary
# - The diagonal cells (from top left to bottom right) are dark red with a value of 1, indicating each variable’s perfect positive correlation with itself.
# - Some pairs exhibit relatively high positive correlations:
#   ssc_p - hsc_p
#   ssc_p - degree_p
#   hsc_p - degree_p
# - Other correlations are weaker, such as mba_p - salary and etest_p - salary.

# In[ ]:





# In[59]:


# Categorical vs categorical
plt.figure(figsize=(5,5))
sns.heatmap(pd.crosstab(df['gender'], df['status']), annot=True, cmap='Blues')
plt.show()


# The heatmap visualizes the correlation between two categorical variables: gender (with categories F for female and M for male) and status (with categories Not Placed and Placed).
# - Each cell in the heatmap contains a numerical value representing the correlation coefficient between the corresponding gender-status combination.
# 
# Notable observations:
# - Females (F):
#   Not Placed: 28 individuals
#   Placed: 48 individuals
# - Males (M):
#   Not Placed: 39 individuals
#   Placed: Over 100 individuals (1e+02)
# - This heatmap highlights the relationship between gender and placement status, with more males being placed compared to females.

# In[ ]:





# In[61]:


sns.barplot(x='salary', y='degree_t', data=df,ci=None,orient = "h")
plt.xticks(rotation = 90)
plt.show()


# Above plot illustrates the average salaries based on different degree types:
# - Sci&Tech: Individuals with a Sci&Tech degree have the highest average salary, exceeding 25,000.
# - Sci&Tech graduates tend to earn more than those in Comm&Mgmt or other fields.

# In[ ]:





# In[62]:


sns.barplot(x='degree_t', y='etest_p', data=df,ci=None,estimator=sum)
plt.xticks(rotation = 90)
plt.show()


# In[ ]:





# ## Checking outliers

# In[180]:


fig, axs = plt.subplots(ncols=6, nrows=3, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k, v in df.items():
    sns.boxplot(y=v, ax=axs[index])
    index += 1

# Since you have 18 plots, and you want to remove the last one,
# you need to access index 17, not index 18, which is out of bounds
fig.delaxes(axs[index - 1])  # Corrected index
plt.tight_layout(pad=0.3, w_pad=0.5, h_pad=4.5)
plt.show()


# In[ ]:





# In[82]:


df[(df['degree_t']=="Sci&Tech") & (df['status']=="Placed")].sort_values(by="salary",ascending=False).head()


# In[ ]:





# ## Data Preprocessing

# In[63]:


#Finding missing values
df.isna().sum()


# In[88]:


#cheking Duplicates
df.duplicated().sum()


# In[91]:


# checking column values data type
df.dtypes


# In[14]:


# as salary and sl_no columns are not required for placement status prediction so we drop it
df.drop(['salary'], axis=1, inplace=True)


# In[ ]:





# In[16]:


df.describe().T


# In[ ]:





# ## Label Encoding Data

# In[17]:


# label encoding needs to be done to ensure all values in the dataset is numeric
# hsc_s, degree_t columns needs to be splitted into columns (get_dummies needs to be applied)
features_to_split = ['hsc_s','degree_t']
for feature in features_to_split:
    dummy = pd.get_dummies(df[feature])
    df = pd.concat([df, dummy], axis=1)
    df.drop(feature, axis=1, inplace=True)


# In[18]:


df.head(2)


# In[19]:


df.rename(columns={"Others": "Other_Degree"},inplace=True)


# In[20]:


encoder = LabelEncoder() # to encode string to the values like 0,1,2 


# In[21]:


columns_to_encode = ['gender','ssc_b', 'hsc_b','workex','specialisation','status']
for column in columns_to_encode:
    df[column] = encoder.fit_transform(df[column])


# In[22]:


df.head(2)


# In[23]:


df.describe()


# In[ ]:





# ## Splitting the data

# In[24]:


X = df.drop('status', axis=1)
y =df['status']


# In[25]:


X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=23)


# In[26]:


sc= StandardScaler()
x_scaled = sc.fit_transform(X) # for standardising the features
x_scaled = pd.DataFrame(x_scaled)


# In[ ]:





# In[28]:


from sklearn.feature_selection import RFE

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)
rfe.fit(X, y)


# In[74]:


from mixed_naive_bayes import MixedNB

mnb = MixedNB()
mnb.fit(X_train, y_train)


# In[75]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = mnb.predict(X_test)


# In[76]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))
print("Precision: {:.2f}%".format(precision*100))
print("Recall: {:.2f}%".format(recall*100))
print("F1-score: {:.2f}%".format(f1*100))


# Conclusion, the classification model developed for predicting placement status based on the provided dataset 
# - demonstrates promising performance with an accuracy of 70.37%. 
# - The precision of 75.00% indicates the proportion of correctly predicted positive placements out of all placements predicted by the model. 
# - Additionally, the model exhibits a recall of 83.33%, suggesting its ability to correctly identify the majority of actual positive placements. 
# - The F1-score, a harmonic mean of precision and recall, stands at 78.95%, indicating a balanced performance between precision and recall.

# In[ ]:





# In[32]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score,ConfusionMatrixDisplay


# In[33]:


ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred)).plot()


# In[34]:


print(classification_report(y_test,y_pred))


# In[ ]:





# In[ ]:





# ## Import The models

# In[35]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from mixed_naive_bayes import MixedNB
from sklearn.ensemble import GradientBoostingClassifier


# ##  Model Training

# In[36]:


lr = LogisticRegression()
lr.fit(X_train,y_train)

svm = svm.SVC()
svm.fit(X_train,y_train)

knn=KNeighborsClassifier()
knn.fit(X_train,y_train)

dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)

rf=RandomForestClassifier()
rf.fit(X_train,y_train)

mnb = MixedNB()
mnb.fit(X_train, y_train)

gb=GradientBoostingClassifier()
gb.fit(X_train,y_train)


# ## Prediction on Test Data

# In[37]:


y_pred1 = lr.predict(X_test)
y_pred2 = svm.predict(X_test)
y_pred3 = knn.predict(X_test)
y_pred4 = dt.predict(X_test)
y_pred5 = rf.predict(X_test)
y_pred6 = mnb.predict(X_test)
y_pred7 =gb.predict(X_test)


# ##  Evaluating the Algorithms

# In[38]:


from sklearn.metrics import accuracy_score


# In[39]:


score1=accuracy_score(y_test,y_pred1)
score2=accuracy_score(y_test,y_pred2)
score3=accuracy_score(y_test,y_pred3)
score4=accuracy_score(y_test,y_pred4)
score5=accuracy_score(y_test,y_pred5)
score6=accuracy_score(y_test,y_pred6)
score7=accuracy_score(y_test,y_pred7)


# In[40]:


print(score1,score2,score3,score4,score5,score6,score7)


# In[41]:


final_data = pd.DataFrame({'Models':['LR','SVC','KNN','DT','RF','MNB','GB'],
            'ACC':[score1*100,
                  score2*100,
                  score3*100,
                  score4*100,
                  score5*100,score6*100,score7*100]})


# In[42]:


final_data


# In[54]:


from sklearn.metrics import precision_score


# In[55]:


score1=precision_score(y_test,y_pred1)
score2=precision_score(y_test,y_pred2)
score3=precision_score(y_test,y_pred3)
score4=precision_score(y_test,y_pred4)
score5=precision_score(y_test,y_pred5)
score6=precision_score(y_test,y_pred6)
score7=precision_score(y_test,y_pred7)


# In[56]:


print(score1,score2,score3,score4,score5,score6,score7)


# In[57]:


final_data_precision_score = pd.DataFrame({'Models':['LR','SVC','KNN','DT','RF','MNB','GB'],
            'ACC':[score1*100,
                  score2*100,
                  score3*100,
                  score4*100,
                  score5*100,score6*100,score7*100]})


# In[58]:


final_data_precision_score


# In[ ]:





# In[64]:


from sklearn.metrics import recall_score


# In[65]:


score1=recall_score(y_test,y_pred1)
score2=recall_score(y_test,y_pred2)
score3=recall_score(y_test,y_pred3)
score4=recall_score(y_test,y_pred4)
score5=recall_score(y_test,y_pred5)
score6=recall_score(y_test,y_pred6)
score7=recall_score(y_test,y_pred7)


# In[66]:


print(score1,score2,score3,score4,score5,score6,score7)


# In[67]:


final_data_recall_score = pd.DataFrame({'Models':['LR','SVC','KNN','DT','RF','MNB','GB'],
            'ACC':[score1*100,
                  score2*100,
                  score3*100,
                  score4*100,
                  score5*100,score6*100,score7*100]})


# In[68]:


final_data_recall_score


# In[ ]:





# In[69]:


from sklearn.metrics import f1_score


# In[70]:


score1=f1_score(y_test,y_pred1)
score2=f1_score(y_test,y_pred2)
score3=f1_score(y_test,y_pred3)
score4=f1_score(y_test,y_pred4)
score5=f1_score(y_test,y_pred5)
score6=f1_score(y_test,y_pred6)
score7=f1_score(y_test,y_pred7)


# In[71]:


print(score1,score2,score3,score4,score5,score6,score7)


# In[72]:


final_data_f1_score = pd.DataFrame({'Models':['LR','SVC','KNN','DT','RF','MNB','GB'],
            'ACC':[score1*100,
                  score2*100,
                  score3*100,
                  score4*100,
                  score5*100,score6*100,score7*100]})


# In[73]:


final_data_f1_score


# In[ ]:





# In[ ]:




