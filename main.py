import pandas as pd
import numpy as np

file = pd.ExcelFile('data\Data.xlsx')

#parse the first/default sheet in the Excel
original_data = file.parse(file.sheet_names[0])

#make 2 dataframes of same type for testing and training
training_data = pd.DataFrame(columns=original_data.columns)
testing_data = pd.DataFrame(columns=original_data.columns)

#split the original into 1:3 testing:training
for i, row in original_data.iterrows():
    if(i%4 == 0):
        testing_data = testing_data._append(row, ignore_index=True)
    else:
        training_data = training_data._append(row, ignore_index=True)

#for finding mean of all columns in dataframe
#input: data(dataframe)
#output: mean of all features(dataframe)
def get_mean(data):
    mean_data = pd.DataFrame(columns=original_data.columns)

    for col in data.columns:
        if(col != 'Class (malignant = 0 , benign = 1)'):
            mean_data.at[0, col] = data[col].mean()

    return mean_data

#for finding standard deviation of all columns in dataframe
#input: data(dataframe)
#output: standard deviation of all features(dataframe)
def get_SD(data):
    SD_data = pd.DataFrame(columns=original_data.columns)
    
    for col in data.columns:
        if(col != 'Class (malignant = 0 , benign = 1)'):
            mean = data[col].mean()
            total = 0
            counter = 0
            for row in data[col]:
                total = total + ((row - mean) ** 2)
                counter += 1
            SD_data.at[0, col] = np.sqrt(total / (counter - 1)) #ERRRRRROOOOOOOOOOOORRRRRRRRRRRRRRRRRRRRRRR maybe

    return SD_data

#based on x and the corresponding mean and SD, it should calculate the probability
#input: x(double), mean(double), and standard deviation(double)
#output: probability(double)
def PDF(x, mean, SD):
    return (np.e ** -(((x - mean) ** 2) / (2 * (SD ** 2)))) / (2 * np.pi * (SD ** 2)) ** .5

#adding new test data
def add(array, data):
    data.loc[len(data)] = array

#start calculating probability of m and b for test cases
#input: test set(dataframe), all data(dataframe)
def main(test, training):
    #organize all the data/info-----------------------------------------------------------------
    #split the training data based on last column m=0 b=1
    m_training_data = pd.DataFrame(columns=original_data.columns)
    b_training_data = pd.DataFrame(columns=original_data.columns)
    for i, row in training.iterrows():
        if(row['Class (malignant = 0 , benign = 1)'] == 0):
            m_training_data = m_training_data._append(row, ignore_index = True)
        else:
            b_training_data = b_training_data._append(row, ignore_index = True)

    #get all the mean and SD of the 3 different sets
    m_training_mean = get_mean(m_training_data)
    b_training_mean = get_mean(b_training_data)
    training_mean = get_mean(training_data)
    m_training_SD = get_SD(m_training_data)
    b_training_SD = get_SD(b_training_data)
    training_SD = get_SD(training_data)

    #calculate P(malignant) and P(benign)
    prob_m = len(m_training_data)/len(training)
    prob_b = len(b_training_data)/len(training)

    #organize all the data/info-----------------------------------------------------------------

    #record accuracy
    correct = 0
    total = 0

    #start iterrating through all rows of the test dataframe------------------------------------
    for i, row in test.iterrows():
        #get actual class
        given = row['Class (malignant = 0 , benign = 1)']
        predicted = -1

        #calculate P(X)
        prob_X = 1
        for col in test.columns:
            if(col != 'Class (malignant = 0 , benign = 1)'):
                prob_X = prob_X * PDF(row[col], training_mean.at[0,col], training_SD.at[0,col])

        #calculate P(X|malignant)
        prob_X_m = 1
        #calculate P(x|malignant)
        for col in test.columns:
            if(col != 'Class (malignant = 0 , benign = 1)'):
                prob_X_m = prob_X_m * PDF(row[col], m_training_mean.at[0,col], m_training_SD.at[0,col])

        #calculate P(X|benign)
        prob_X_b = 1
        #calculate P(x|benign)
        for col in test.columns:
            if(col != 'Class (malignant = 0 , benign = 1)'):
                prob_X_b = prob_X_b * PDF(row[col], b_training_mean.at[0,col], b_training_SD.at[0,col])

        #prediction time
        prob_m_X = (prob_X_m * prob_m) / prob_X
        prob_b_X = (prob_X_b * prob_b) / prob_X
        if(prob_m_X > prob_b_X):
            predicted = 0
        else:
            predicted = 1

        #check if it is correct
        if(predicted == given):
            correct += 1
            
        total += 1

    return(correct / total)



#make a dataframe for new testing guessing it is benign
X = [13.0, 15.0, 85.0, 500.0, 0.1, 0.15, 0.1, 0.05, 0.2, 0.08, 0.5, 1.5, 4.0, 70.0, 0.01, 0.02, 0.02, 0.01,
0.015, 0.002, 14.0, 20.0, 90.0, 600.0, 0.2, 0.25, 0.2, 0.1, 0.3, 0.1, 1]
X_data = pd.DataFrame(columns=original_data.columns)
add(X, X_data)

#execute
test_result = main(testing_data, training_data)
X_result = main(X_data, training_data)
print("The accuracy of the algorithm on the test data is:")
print(test_result)
print("If it is 0, X is malignant; if it is 1, it is benign")
print(X_result)