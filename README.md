# Capstone Project | Data Science Bootcamp

## Introduction

Chronic diabetes is one of the most common diseases in the modern world. Type 2 diabetes is an epidemic in high-income countries. This disease is in the list of the most common causes of disability, early disability, poor quality of life, and death. Type 2 diabetes mellitus increase risks of various complications, both specific for patients with diabetes, and cardiac, renal, and other diseases related to various organs of the human body. 

Early prediction of future complications is especially important for patients with diabetes, as health workers will take preventive measures against predicted complications. Early therapy will reduce the probability of negative cardiovascular outcomes, such as atrial fibrillation, chronic kidney disease, and others. Health workers will be use qualitative prediction of future microvascular complications for the task of selection of compensating carbohydrate metabolism therapy. (Derevitskii,I. & Kovalchuk,S., 2020)

Machine Learning algorithms are being increasingly utilized in medicine to combine complex sets of clinical and diagnostic information to generate predictive algorithms for improving health outcomes. The application of machine learning and data mining methods have gained considerable attention particularly in diabetes research. These machine-learning strategies have been applied primarily to develop better closed loop insulin delivery systems utilizing measures of glycaemic variability (GV), hyperglycaemia, and hypoglycaemia in patients with Type 1 diabetes within the context of personalized decision support systems and blood glucose alarm events for optimal diabetes self-management. (Elhadd, T. et al, 2020)

The purpose of this project is to determine the variables that have the most correlation with the diabetes diagnosis and create a Machine Learning model that can predict, based on the information provided, whether a patient has the disease or not.

This project uses data from the Center for Disease Control and Prevention (CDC) on factors that can contribute to a person having diabetes, the dataset contains more than 250K observations of symptoms and conditions of patients who have or not the disease.

## Dataset

The used dataset had 22 features and 253680 …


### Features:

Response Variable
  1.	(Ever told) you have diabetes (If "Yes" and respondent is female, ask "Was this only when you were pregnant?" --> Diabetes_binary
High Blood Pressure:
  3.	Adults who have been told they have high blood pressure by a doctor, nurse, or other health professional --> 'HighBP'
High Cholesterol:
  3.	Have you EVER been told by a doctor, nurse or other health professional that your blood cholesterol is high? --> 'HighChol'
  4.	Cholesterol check within past five years --> 'CholCheck'
BMI
  5.	Body Mass Index (BMI) --> ‘BMI’
Smoking
  6.	Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes] --> 'Smoker'
Other Chronic Health Conditions
  7.	(Ever told) you had a stroke. --> 'Stroke'
  8.	Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI) --> 'HeartDiseaseorAttack' 
Physical Activity
  9.	Adults who reported doing physical activity or exercise during the past 30 days other than their regular job --> 'PhysActivity'
Diet
  10.	Consume Fruit 1 or more times per day --> 'Fruits'
  11.	Consume Vegetables 1 or more times per day --> 'Veggies'
Alcohol Consumption
  12.	Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week) --> 'HvyAlcoholConsump'
Health Care
  13.	Do you have any kind of health care coverage, including health insurance, prepaid plans such as HMOs, or government plans such as Medicare, or Indian Health Service? --> 'AnyHealthcare'
  14.	Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? --> 'NoDocbcCost'
Health General and Mental Health
  15.	Would you say that in general your health is: --> 'GenHlth'
  16.	Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? --> 'MentHlth'
  17.	Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? --> 'PhysHlth'
  18.	Do you have serious difficulty walking or climbing stairs? --> 'DiffWalk'
Demographics
  19.	Indicate sex of respondent. --> 'Sex'
  20.	Fourteen-level age category --> _'Age'
  21.	What is the highest grade or year of school you completed? --> 'Education'
  22.	Is your annual household income from all sources --> 'Income'

## Exploratory Data Analysis

The following graph shows the distribution of the target feature (0: No diabetes, 1: Diabetes). With this information it is known that we are working with an imbalanced dataset since one class contains many more observations than the other class, so later we will apply different techniques to reduce the negative effect that it can cause in the prediction models.
 
![image](https://user-images.githubusercontent.com/67977294/145473207-a7462a95-ce80-4844-9553-a810f835e8b1.png)

![image](https://user-images.githubusercontent.com/67977294/145473245-ca1932b8-cf07-4a61-96c0-61739e5fd117.png)

The age column in the dataset was made as a range of values between 1 and 13, this is because each value corresponds to a range of 5 years:
•	1: 18-24.
•	2: 25-29.
•	3: 30-34.
•	...
•	13: 80 or older.
So, in the next plot we can see the ranges of ages with most cases of diabetes, people with an age between 65-69 are more likely to have diabetes.

![image](https://user-images.githubusercontent.com/67977294/145473285-5dfc4f07-158c-4ff7-8892-659f9200d516.png)

 
The next plot shows that if you have high cholesterol levels you are basically twice as likely to have diabetes as people without high cholesterol.
The second plot conclude that the people that checks their cholesterol level in the past 5 years have the most negatives diabetes cases, this can be due to the fact that people who didn't check their cholesterol don’t receive an alert to avoid diabetes.

![image](https://user-images.githubusercontent.com/67977294/145473321-8fa14772-a07a-496d-b6a5-91c4d6cbc41b.png)


The following plot shows that people who do physical activities are less likely to have diabetes.
Since exercising helps improve cholesterol levels, blood pressure and body weight, also one of the most important factors is that it helps improve resistance to insulin.
 
![image](https://user-images.githubusercontent.com/67977294/145473373-1c1afc3b-47e9-4810-93d0-c160803fe800.png)

The following graph shows us that the column BMI is not a factor related to diabetes, since both categories of the column diabetes (0, 1) have relatively the same distribution.

![image](https://user-images.githubusercontent.com/67977294/145473530-e9c618e2-8f33-4040-b29d-d316f1c16570.png)


Next, we created a plot that shows the amount of people that consume fruits and vegetables. As we can see, there is more risk of having Diabetes if people don’t eat vegetables or fruits.

![image](https://user-images.githubusercontent.com/67977294/145473557-5ed381a2-84a8-47f2-af86-c94d7a1ba198.png)

 
Also, we have some insights on the Diabetes diagnosis for the people that consumes Alcohol or are Smokers.
As we could see in following plot, it seems to not be correlated with Diabetes because of the minimum differences in the positive and negative diagnosis in both subplots.
 
![image](https://user-images.githubusercontent.com/67977294/145473590-51f484a5-d436-480f-801e-5ba872c4f468.png)


At the beginning we thought that the Education and Income columns would not be related to whether the person has diabetes or not, but the following plots do show us a difference between each of the classes, as the ratios are quite different between them. So, we are going to keep these features in the creation of the model.
 

![image](https://user-images.githubusercontent.com/67977294/145473621-8ef4ec4e-7d19-4a98-9e26-024891fa6d4e.png)


## Machine Learning Models

Five separate machine learning techniques including Random Forest, K-Nearest Neighbor, XG Boost Classifier, Neuronal Network and Extras Trees were applied. In Table 1 we registered the accuracy score, AUC score and F1 for every model.

Table 1. Models Score with imbalance dataset
|     Score       |     Random forest    |     K-Nearest Neighbor    |     XGBoost Classifier    |     Neural Network    |     Extra Trees    |
|-----------------|:--------------------:|:-------------------------:|:-------------------------:|:---------------------:|:------------------:|
|     Accuracy    |         0.8674       |           0.8606          |           0.8411          |         0.8667        |        0.8515      |
|     ROC-AUC     |         0.5533       |           0.5399          |           0.5965          |         0.5705        |        0.5688      |
|     F1-Score    |         0.1991       |           0.1609          |           0.3100          |         0.2507        |        0.2490      |

When observation in one class is higher than the observation in other classes then there exists a class imbalance, if the dataset is imbalance, then in such cases, you get a pretty high accuracy just by predicting the majority class, but you fail to capture the minority class, which is most often the point of creating the model in the first place, and as we could see from the Table 1, the results shows that the dataset was imbalance.

A widely adopted technique for dealing with highly unbalanced datasets is called resampling. It consists of removing samples from the majority class (under-sampling) and/or adding more examples from the minority class (over-sampling).

For this project, the applied technique was over-sampling, since with this technique observations that may have data and patterns that can help the model to make better predictions are not discarded. In Table 2 we registered the results for every model using the balanced dataset

Table 2. Models Score with balance dataset

|     Score       |     Random forest    |     K-Nearest Neighbor    |     XGBoost Classifier    |     Neural Network    |     Extra Trees    |
|-----------------|:--------------------:|:-------------------------:|:-------------------------:|:---------------------:|:------------------:|
|     Accuracy    |         0.7861       |           0.7747          |           0.9337          |         0.7565        |        0.9562      |
|     ROC-AUC     |         0.7862       |           0.7748          |           0.9337          |         0.7565        |        0.9562      |
|     F1-Score    |         0.7988       |           0.7853          |           0.9369          |         0.7673        |        0.9574      |


## Results Analysis

Machine Learning models demonstrated high accuracy in diagnosing diabetes with 'AUC' ranging from 0.7565 to 0.9562 being **Extra Trees** the model that have the best performance with an 'Accuracy Score' of 0.9562. It’s important to mention that the Extra Trees model was implemented with the python library PyCaret and from the analysis made by this classifier it determined that BMI and Age features were the most important for the model. This result is visualized on the following graph

![image](https://user-images.githubusercontent.com/67977294/145492517-f36e8ca9-0dde-4296-a37c-17e4afebc3b5.png)

