# Everest Death zone summit prediction model.
#A logical regression model to predict whether a climber can successfully summit the mount everest.

#importing the required libraries
import pandas as pd #for ops on dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder #for data transformation
from numpy import set_printoptions #to fit the model.
import numpy as np #for numerical ops
import matplotlib.pyplot as plt #for data visualization
import seaborn as sns
from sklearn.linear_model import LogisticRegression #logistic model
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve #to evaluate the accuracy of the model
from sklearn.model_selection import train_test_split #splitting the data
from sklearn.feature_selection import RFE #Recursive feature elimination
from sklearn.model_selection import cross_val_score #cross validation
from sklearn.metrics import classification_report #for classification report of the model


#now, getting the dataset.
df = pd.read_csv("everest_summit_master.csv")

df.head()

#Exploring the data:
df.info()

#Checking for missing data
df.isnull().sum()

#Perfect there are no missing data. we can move ahead with exploring the dataset

print("Rows:", df.shape[0], "Columns:", df.shape[1])
#Rows = 45000, Columns = 25

#performing descriptive stats
df.describe()

'''
Step 2: Data Visualization  
Visualize the data
'''
#summited vs season
plt.figure(figsize=(6,4))
sns.barplot(df, x="season", y="summited", hue = "season")
plt.xlabel("Seasons")
plt.ylabel("Summited")
plt.title("Impact of seasons on summit of Mt. Everest")
plt.show()

 #summited vs route
plt.figure(figsize=(6,4))
sns.barplot(df, x="route", y="summited", hue='route')
plt.xlabel("Routes")
plt.ylabel("Summited")
plt.title("Different routes taken for the summit of Mt. Everest")
plt.show()

#summited vs sex
plt.figure(figsize=(6,4))
sns.barplot(df, x="sex", y="summited")
plt.xlabel("Sex")
plt.ylabel("Summited")
plt.title("Gender differences on summit of Mt. Everest")
plt.show()

#Experience in climbing vs sex
plt.figure(figsize=(6,4))
sns.barplot(df, y="years_climbing", x="summited")
plt.xlabel("Summited")
plt.ylabel("Experience in Climbing")
plt.title("Experience in climbing vs Summit on Mt. Everest")
plt.show()


#summited vs Turnaround Reason
plt.figure(figsize=(10,8))
sns.barplot(df, x="turnaround_reason", hue="turnaround_reason")
plt.xlabel("Turnaround Reason")
plt.ylabel("Summited")
plt.title("Turnaround Reasons for not summitting the  of Mt. Everest")
plt.show()

#before we encode, let's take a backup
df_bkp = pd.DataFrame(df)

#Encoding the dataset for further analysis
col = ['season', 'route', 'operator_tier', 'sex']
ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
encoder_array = ohe.fit_transform(df[col])

encoded_df = pd.DataFrame(
    encoder_array,
    columns=ohe.get_feature_names_out(col),
    index=df.index
)

df = pd.concat([df.drop(columns=col), encoded_df], axis=1)
df.head()

#Dropping 'turnaround_reason' as it is a leakage & 'climber_id'  as it serves no purpose and 'o2_start_altitude_m', 'jetstream_risk' as they are highly correlated:
df = df.drop(columns=['turnaround_reason','climber_id','o2_start_altitude_m', 'jetstream_risk'])

#Let's split the data in x and y variables for model building
x = df.drop(columns='summited')
y = df['summited']

print(f"x_shape: {x.shape}, y_shape: {y.shape}")


#now, let's split the data for testing
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

print(f"X train: {x_train.shape}, Y train: {y_train.shape}, X test: {x_test.shape}, Y test: {y_test.shape}")


#Now let's scale the data:
continuos_cols = ["year","age", "years_climbing", "prior_8000m_peaks", "prior_everest_attempts", "highest_prev_altitude_m", "team_size", "sherpa_ratio", "days_acclimatizing", "camps_established", "summit_window_days","weather_score", "queue_minutes_above_8000"]

scale = StandardScaler()
x_train = x_train.copy()
x_test = x_test.copy()

x_train[continuos_cols] = scale.fit_transform(x_train[continuos_cols])
x_test[continuos_cols] = scale.transform(x_test[continuos_cols])


# model building
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

#predictive analysis
y_pred_test = classifier.predict(x_test)


#Predicting the accuracy:
accuracy = accuracy_score(y_test, y_pred_test)
print("Model accuracy", accuracy)

#confusion matrix:
print("\nConfusion Matrix (test set)\n", confusion_matrix(y_test, y_pred_test))

#AUC Score:
auc = roc_auc_score(y_test, classifier.predict_proba(x_test)[:,1])
print("Test AUC:", auc)

#plotting the ROC AUC curve

#Get probability values for the positive class (summited == 1)
y_pred_proba = classifier.predict_proba(x_test)[:,1]
y_pred_proba

#Calculate the ROC curve metrics

fpr,tpr,threshold = roc_curve(y_test, y_pred_proba)

#calculate the AUC score:
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {auc_score:.3f}")

#plot ROC Curve
plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,color="red", lw=2, label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0,1], [0,1], color= 'red', linestyle= "--", label="Random Guessing")

#Customizing plot
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True positive rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc= "lower right")
plt.grid(True)


#Train Accuracy:
y_pred_train = classifier.predict(x_train)
print(f"Train Accuracy: {accuracy_score(y_train, y_pred_train)}")


'''
# Model Enhancements:

1. L2 penalty:
'''
#building a model with L2 penalty
classifier_l2 = LogisticRegression(penalty="l2", C=0.1, solver="liblinear", random_state=42)
classifier_l2.fit(x_train, y_train)
#predicting
y_pred_l2 = classifier_l2.predict(x_test)
#accuracy scores with L2 penalty
accuracy = accuracy_score(y_test, y_pred_l2)
print("Accuracy Score with L2 penalty = ",accuracy)

'''
2. Recursive Feature Elimination (RFE):
Selects the top n most important features by recursively removing the least significant feature.

Why?  
Improves generalization by eliminating noisy/irrelevant features.  
Reduces training and model complexity.
'''

from sklearn.feature_selection import RFE

#selects top 3 features

selector = RFE(classifier, n_features_to_select=3, step=1)
selector.fit(x_train, y_train)
print("Selection feature:", x_train.columns[selector.support_])

#Selection feature: Index(['prior_8000m_peaks', 'sherpa_ratio', 'uses_oxygen'], dtype='object')

df_2 = df[['prior_8000m_peaks', 'sherpa_ratio', 'uses_oxygen', "summited"]]

x_2 = df_2.iloc[:, :3]
y_2 = df_2.iloc[:, 3]

x2_train, x2_test, y2_train, y2_test = train_test_split(x_2,y_2, test_size=0.2, random_state=42)
print(f"X2 train shape: {x2_train.shape}")
print(f"X2 test shape: {x2_test.shape}")
print(f"Y2 train shape: {y2_train.shape}")
print(f"Y2 test shape: {y2_test.shape}")

#Scaling the spilt RFE dataset:
continuos_cols = ['prior_8000m_peaks', 'sherpa_ratio']

scale = StandardScaler()
x2_train = x2_train.copy()
x2_test = x2_test.copy()

x2_train[continuos_cols] = scale.fit_transform(x2_train[continuos_cols])
x2_test[continuos_cols] = scale.transform(x2_test[continuos_cols])


#Modeling the RFE dataset:
classifier_rfe = LogisticRegression()
classifier_rfe.fit(x2_train, y2_train)
#prediction
y2_pred_test = classifier_rfe.predict(x2_test)
#accuracy score
accuracy = accuracy_score(y2_test, y2_pred_test)
print("Model accuracy", accuracy)



'''
Cross Validation:
A method to evaluate model performance by splitting the data into multiple train-test folds  

It provides a single robust estimate of generalization error.  
Reduces reliance on single train-test split
'''

from sklearn.model_selection import cross_val_score

#5-fold cross validation
cv_scores = cross_val_score(classifier, x_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validated Accuracy: {cv_scores.mean():.3f} (\u00b1 {cv_scores.std():.3f})")


"""
Classification Report:
"""

#for baseline model:
print("For baseline model:\n\n",classification_report(y_test, y_pred_test))

#for model with L2 penalty
print("For model with L2 penalty:\n\n",classification_report(y_test, y_pred_l2))

#for model with RFE
print(classification_report(y2_test, y2_pred_test)) 