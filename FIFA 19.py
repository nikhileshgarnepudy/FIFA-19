#!/usr/bin/env python
# coding: utf-8

# # FIFA 19 -  VISUALIZATION | ANALYSIS | PLAYER 'RATING' PREDICTION
# 

# # IMPORTING REQUIRED LIBRARIES & DATA

# In[57]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None


# In[2]:


data = pd.read_csv('fifa19.csv')


# # DATA ANALYSIS

# In[3]:


data


# In[4]:


data.describe()


# In[5]:


data.info()


# # DATA CLEANING

# NULL VALUES 

# In[6]:


plt.figure(figsize=[30,5])
sns.heatmap(data.isnull(),cmap='Blues',yticklabels=False)


# NULL VALUES - ASSIGNING THE MEAN OF THE COLUMN TO VOID SPACES

# In[7]:


data['Volleys'].fillna(data['Volleys'].mean(), inplace = True)
data['Dribbling'].fillna(data['Dribbling'].mean(), inplace = True)
data['Curve'].fillna(data['Curve'].mean(), inplace = True)
data['FKAccuracy'].fillna(data['FKAccuracy'], inplace = True)
data['LongPassing'].fillna(data['LongPassing'].mean(), inplace = True)
data['BallControl'].fillna(data['BallControl'].mean(), inplace = True)
data['HeadingAccuracy'].fillna(data['HeadingAccuracy'].mean(), inplace = True)
data['Finishing'].fillna(data['Finishing'].mean(), inplace = True)
data['Crossing'].fillna(data['Crossing'].mean(), inplace = True)
data['Weight'].fillna('200lbs', inplace = True)
data['Contract Valid Until'].fillna(2019, inplace = True)
data['Height'].fillna("5'11", inplace = True)
data['Loaned From'].fillna('None', inplace = True)
data['Joined'].fillna('Jul 1, 2018', inplace = True)
data['Jersey Number'].fillna(8, inplace = True)
data['Body Type'].fillna('Normal', inplace = True)
data['Position'].fillna('ST', inplace = True)
data['Club'].fillna('No Club', inplace = True)
data['Work Rate'].fillna('Medium/ Medium', inplace = True)
data['Skill Moves'].fillna(data['Skill Moves'].median(), inplace = True)
data['Weak Foot'].fillna(3, inplace = True)
data['Preferred Foot'].fillna('Right', inplace = True)
data['International Reputation'].fillna(1, inplace = True)
data['Wage'].fillna('€200K', inplace = True)


# In[8]:


plt.figure(figsize=[30,5])
sns.heatmap(data.isnull(),cmap='Blues',yticklabels=False)


# DROP ROWS WHO'S VOID SPACES CANNOT BE PREDICTED BY ANY MEANS

# In[9]:


data = data.dropna()


# In[10]:


plt.figure(figsize=[30,5])
sns.heatmap(data.isnull(),cmap='Blues',yticklabels=False,cbar=False)


# In[11]:


data.columns


# DROPPING COLUMNS, UNNECESSARY FOR TRAINING

# In[12]:


data.drop(['Unnamed: 0', 'ID', 'Name','Photo', 'Nationality', 'Flag','Club', 'Club Logo','Work Rate', 'Body Type', 'Real Face','Jersey Number', 'Joined', 'Loaned From', 'Contract Valid Until',
       'Height', 'Weight','LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
       'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB','Release Clause','Value','Wage'],axis=1,inplace=True)


# In[13]:


data.iloc[:5,0:10]


# # DATA VISUALIZATION

# PLAYER - AGE

# In[14]:


plt.figure(figsize=[20,6])
data['Age'].hist(bins=5)


# OVERALL AND POTENTIAL DISTRIBUTION OF PLAYERS

# In[15]:


plt.figure(figsize=[30,8])
plt.subplot(1,2,1)
sns.countplot(x='Overall',data=data,palette='colorblind')
plt.subplot(1,2,2)
sns.countplot(x='Potential',data=data,palette='colorblind')


# LEFT FOOT AND RIGHT FOOT PLAYERS DISTRIBUTION

# In[16]:


sns.countplot(x='Preferred Foot',data=data,palette='colorblind')


# PLAYER SUMMARY

# In[17]:


plt.figure(figsize=[6,8])
labels = '1','2','3','4','5'
sizes = data['International Reputation'].value_counts()
explode = [0.1, 0.1, 0.2, 0.5, 0.9]

plt.pie(sizes, labels = labels,  explode = explode, shadow = True)
plt.title('International Repuatation for the Football Players', fontsize = 20)
plt.legend()
plt.show()


# In[18]:


labels = ['5', '4', '3', '2', '1'] 
size = data['Weak Foot'].value_counts()
colors = plt.cm.Wistia(np.linspace(0, 1, 5))
explode = [0, 0, 0, 0, 0.1]
plt.figure(figsize=[6,8])
plt.pie(size, labels = labels, colors = colors, explode = explode, shadow = True, startangle = 90)
plt.title('Distribution of Week Foot', fontsize = 25)
plt.legend()
plt.show()


# In[19]:


plt.figure(figsize=[15,3])
plt.title('Players Playing at different Positions',fontsize=20)
sns.countplot(x='Position',data=data,palette='bone')


# In[20]:


data.iloc[:5,10:20]


# In[21]:


plt.figure(figsize=[15,5])
sns.countplot(x='Skill Moves',data=data,palette='pastel')


# In[22]:


plt.figure(figsize=[20,7])
plt.style.use('seaborn-dark-palette')

sns.boxenplot(data['Overall'], data['Age'], hue = data['Preferred Foot'], palette = 'Greys')
plt.title('Comparison of Overall Scores and age wrt Preferred foot', fontsize = 20)
plt.show()

plt.rcdefaults()


# In[23]:


plt.figure(figsize=[15,7])
plt.scatter(data['Overall'], data['International Reputation'], s = data['Age']*1000, c = 'pink')
plt.xlabel('Overall Ratings', fontsize = 20)
plt.ylabel('International Reputation', fontsize = 20)
plt.title('Ratings vs Reputation', fontweight = 20, fontsize = 20)
plt.show()


# A CO-RELATION MATRIX DEPICTING THE RELATION BETWEEN VARIOUS DATA POINTS

# In[24]:


plt.figure(figsize=[30,15])
sns.heatmap(data.corr(),annot=True)


# In[25]:


data.iloc[data.groupby(data['Position'])['Overall'].idxmax()][['Position', 'Age']]


# In[26]:


sns.lineplot(data['Age'], data['Potential'], palette = 'Wistia')
plt.title('Age vs Potential', fontsize = 20)

plt.show()


# In[27]:


data.iloc[:2,0:10]


# # FEATURE ENGINEERING

# CONVERTING THE 'PREFERRED FOOT' COLUMN INTO DUMMY INTEGERS FOR TRAINING

# In[28]:


foot = pd.get_dummies(data['Preferred Foot'],drop_first=True)


# In[29]:


data = pd.concat([data,foot],axis=1)
data.drop('Preferred Foot',axis=1,inplace=True)
data


# CONVERTING THE 'POSTION' COLUMN INTO DUMMY INTEGERS FOR TRAINING

# In[30]:


pos = pd.get_dummies(data['Position'],drop_first=True)


# In[31]:


data = pd.concat([data,pos],axis=1)
data.drop('Position',axis=1,inplace=True)
data


# CATEGORIZING THE TARGET COLUMN INTO CATEGORIES

# In[32]:


def overall(data):
    if data>=90:
        return 5
    elif data>=80 and data<90:
        return 4
    elif data>=70 and data<80:
        return 2
    elif data>=60 and data<70:
        return 2
    elif data<60:
        return 1


# In[33]:


data['Overall'] = data['Overall'].apply(overall)


# In[34]:


data


# In[35]:


data.rename(columns = {'Overall':'Rating'}, inplace = True) 


# In[36]:


data


# In[37]:


X = data.drop('Rating',axis=1)


# In[38]:


X


# In[39]:


y = data['Rating']


# In[40]:


y


# # FEATURE SCALING

# In[41]:


from sklearn.preprocessing import StandardScaler


# In[42]:


sc = StandardScaler()
X = sc.fit_transform(X)


# In[43]:


X


# In[44]:


y = y.values


# In[45]:


y


# # TRAINING THE DATASET

# SPLITTING THE DATA INTO TRAINING AND TESTING

# In[46]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35)


# In[47]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# # TESTING THE DATASET

# CLASSIFICATION MATRIX

# In[48]:


y_predict = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True, fmt="d")


# CLASSIFICATION REPORT

# In[58]:


print(classification_report(y_test, y_predict))


# # REVISITING FOR BETTER RESULTS

# In[50]:


data.columns


# DROPPING ADDITIONAL COLUMNS OF LESS IMPORTANCE

# In[51]:


data.drop(['CB', 'CDM', 'CF', 'CM', 'LAM', 'LB', 'LCB', 'LCM', 'LDM',
       'LF', 'LM', 'LS', 'LW', 'LWB', 'RAM', 'RB', 'RCB', 'RCM', 'RDM', 'RF',
       'RM', 'RS', 'RW', 'RWB', 'ST'],axis=1,inplace=True)


# In[52]:


X = data.drop('Rating',axis=1)
y = data['Rating']
sc = StandardScaler()
X = sc.fit_transform(X)


# SPLITTING THE DATA 

# In[53]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35)


# # TRAINING THE DATASET - 2nd iter

# In[54]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# # TESTING THE DATASET - 2nd iter

# In[55]:


y_predict = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True, fmt="d")


# In[59]:


print(classification_report(y_test, y_predict))

