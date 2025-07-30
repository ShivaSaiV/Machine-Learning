# Labor-Economics-Project
Labor Economics Project Proposal
Project Title: Wage Prediction and Fairness Auditing Using Machine Learning
Shiva Sai Vummaji
Labor Economics
Prof. Van Kammen
Objective:
This project aims to build a machine learning model to predict individual wages based on human
capital, educational qualifications, race & demographics, and occupational features using real
labor market data. Beyond prediction (if I am successful and have time), I will assess whether the
model amplifies bias along sensitive attributes such as gender and race, connecting the project to
themes of wage inequality and fairness in labor markets.
Data Source:
I will use the U.S. Census Bureau's Current Population Survey (CPS) or American Community
Survey (ACS) microdata, focusing on individuals aged 18–65 in the labor force. The dataset
includes relevant variables such as hourly wage/annual income, gender, race, age, industry and
occupation, education, hours worked per week, irregular/regular hours.
Methodology:
1. Data Collection & Cleaning:
a. I will get data from IPUMS and select relevant variables, similar to how I did for
Data Explorer assignments.
b. I will convert categorical variables (e.g., SEX, RACE, EDUC) to labels and
remove any duplicates/invalid/missing values.
2. Exploratory Analysis:
a. Understand/graph wage distribution by education, gender, race, etc.
i. I can even utilize graphs from Data Explorers for this
b. I will create a correlation matrix for the variables.
c. My goal here would basically be to understand the data better.
3. Model Building:
a. I will utilize 2 types of Machine Learning models: Linear/Logistic Regression and
Neural Networks.
b. c. Linear Regression (sklearn, numpy, pandas):
i. Split dataset into train/test data
ii. Using the linear regression library from sklearn, I will build the model and
fit it on the training data.
iii. I will then predict using the test data.
Neural Network (PyTorch, numpy, pandas):
i. Using the torch module, I will build a neural network and train it on the
training data.
I will then create a testing mechanism.
ii. 4. Model Evaluation:
a. I will utilize metrics such as RMSE, MAE, R² to evaluate both Linear Regression
and Neural Network.
b. I will also plot actual vs. predicted wages for both models and compare.
5. Fairness Analysis:
a. I will group based on variables such as SEX and RACE and see if the model has
any bias.
b. I will calculate the difference in accuracy between original models vs. true data
and see if there are any overestimates/underestimates based on race or sex.
c. I can also maybe implement a Python library called fairlearn which is a more
sophisticated way to check bias.
6. Deliverables:
a. Cleaned dataset from Step 1
b. Graphs from Step 2
c. Code (with detailed comments) and link to Github repository
d. 1-2 page report regarding findings/interpretation/results
e. Visuals for evaluation
Tech Stack:
1. Programming Language: Python 3.10
2. Data Source: IPUMS
3. Data Analysis: Pandas, Numpy, Scikit-Learn, Matplotlib
4. Machine Learning: PyTorch, Scikit-Learn
5. Environment: Google Colab/Jupyter Notebook
6. Codebase: Github Repository
