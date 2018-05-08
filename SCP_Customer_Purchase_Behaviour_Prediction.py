import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression as lr


#Function to generate Train Dataset
def fun_generate_train_data(num_people, num_products, num_weeks):
    matrix = [list((i,j,t)) for t in np.arange(0,num_weeks) for i in np.arange(0,num_people) for j in np.arange(0,num_products)]
    output_df = pd.DataFrame(matrix,columns = ['i','j','t']) 
    return output_df


#Function to generate Test Dataset
def fun_generate_test_data(num_people, num_products):
    matrix = [list((i,j,49)) for i in np.arange(0,num_people) for j in np.arange(0,num_products)]
    output_df = pd.DataFrame(matrix,columns = ['i','j','t']) 
    return output_df
    
if __name__ == '__main__':

    #Compute Start Time
    start_time = time.time()

    #define data variables
    NUM_PEOPLE = 2000
    NUM_WEEKS = 49
    NUM_PRODUCTS = 40
    
    #define model parameters
    RANDOM_STATE = 42
    C = 0.17
    PENALTY = 'l2'
    
    #Input File names (Please place all the files in the same directory as the script file)
    
    INPUT_HIST_DATA_FILE = 'train.csv'
    PRODUCT_PROMOTION_SCHEDULE_FILE = 'promotion_schedule.csv' 
    
    #Output File for storing prediction results
    PROBABILITY_PREDICTION_FILE = 'predictions.csv'
    
    
    print("Training data preparation started....\n\n")

    #Generate Training Data
    train_df = fun_generate_train_data(NUM_PEOPLE,NUM_PRODUCTS,NUM_WEEKS)


    #Read Historical Product purchase data from file
    read_data_df = pd.read_csv(INPUT_HIST_DATA_FILE)
    
    #Group by customer, product and advertised
    prod_pref_adv = read_data_df.groupby(['i','j','advertised']).size().reset_index(name='counts_per_prod_adv')
    #Group by based on customer and product
    prod_pref_overall = read_data_df.groupby(['i','j']).size().reset_index(name='counts_per_product')
    #Group by based on customer
    customer_buy_cnt = read_data_df.groupby(['i']).size().reset_index(name='counts_total_purchase')
    

    #Calculate probability of buying a product
    read_data_df = pd.merge(read_data_df,pd.merge(prod_pref_adv,pd.merge(prod_pref_overall,customer_buy_cnt, on = ['i'], 
                                                                         how = 'left'),on = ['i','j'], 
                                                           how = 'left'),
                                                           on = ['i','j','advertised'], 
                                                           how = 'left')
    read_data_df['prod_adv_prob'] = read_data_df['counts_per_prod_adv']/read_data_df['counts_per_product']
    read_data_df['overall_purchase_prob'] = read_data_df['counts_per_product']/read_data_df['counts_total_purchase']
    
    #Add Labels to the historical data
    read_data_df['labels'] = 1

    #Drop unnecessary columns
    read_data_df.drop(['counts_per_prod_adv', 'counts_per_product','counts_total_purchase'], axis=1, inplace = True)

    #Join input data with the generated data
    train_df = pd.merge(train_df,read_data_df, on = ['i','j','t'], how = 'left')


    #Get Product Price List from the historical data
    prod_price_df = read_data_df[read_data_df['advertised'] == 0][['j','price']].drop_duplicates().sort_values(by = 'j').reset_index(drop = True)


    #Map the product prices of the unknown values
    train_df.loc[train_df['price'].isnull(),'price'] = train_df['j'].map(prod_price_df.price)

    #Fill NaN values with zeros
    train_df['advertised'].fillna(0, inplace = True)
    train_df['prod_adv_prob'].fillna(0, inplace = True)
    train_df['overall_purchase_prob'].fillna(0, inplace = True)
    train_df['labels'].fillna(0, inplace = True)

    #Convert datatypes into int
    train_df['advertised'] = train_df['advertised'].astype(int)
    train_df['labels'] = train_df['labels'].astype(int)
    
    #Generate Additional Feature for advertised prices only in train data
    train_df.insert(loc=7, column='adv_prices', value=train_df['price'] * train_df['advertised'])
    
    #Shuffle Training Set Data Frame
    train_df = train_df.sample(frac=1, random_state = RANDOM_STATE).reset_index(drop=True)
    
    print("Training data preparation completed. %s rows generated with %s features." % (train_df.shape[0], (train_df.shape[1]-1)))
          
          
    print("\n\nTest Data Preparation Started...\n\n")

    #Generating augmented data for testing dataset
    test_df = fun_generate_test_data(NUM_PEOPLE,NUM_PRODUCTS)

    #Read Promotion Schedule for Week 50
    promotion_df = pd.read_csv(PRODUCT_PROMOTION_SCHEDULE_FILE)

    #Use promotion data to compute prices of 50th week
    prod_price_df = pd.merge(prod_price_df,promotion_df, on = ['j'], how = 'inner')

    #Apply Discount on the advertised product for the 50th week
    prod_price_df.loc[prod_price_df['advertised'] == 1,'price'] = prod_price_df[prod_price_df['advertised'] == 1]['price'] * (1-prod_price_df[prod_price_df['advertised'] == 1]['discount'])
    prod_price_df.drop(['discount'], axis=1, inplace = True)


    #Join test data with the product prices data for the 50th week
    test_df = pd.merge(test_df,prod_price_df, on = ['j'], how = 'left')
    
    #Join read data with the generated test data in order to determine the product preference probability
    test_df = pd.merge(test_df,read_data_df[['i','j','advertised','prod_adv_prob','overall_purchase_prob']], 
                       on = ['i','j','advertised'], how = 'left') 
    
    #Generate Additional Feature for advertised prices only in test data
    test_df['adv_prices'] = test_df['price'] * test_df['advertised']
    
    #Add zeros to unknown  values
    test_df['prod_adv_prob'].fillna(0, inplace = True)
    test_df['overall_purchase_prob'].fillna(0, inplace = True)
    
    #Drop Duplicates from test data
    test_df.drop_duplicates(inplace = True)

    print("Test data preparation completed. %s rows generated with %s features." % (test_df.shape[0], test_df.shape[1]))
          
    print("\n\nTraining Logistic Regression Model...\n\n")

    #Logistic Regression to fit training data with selected features
    X, Y = train_df.iloc[:,[0,1,3,4,5,6,7]], train_df.iloc[:,-1]
    
    model = lr(C=C, penalty = PENALTY).fit(X,Y)
        
    print("Predicting Product Buying probabilty for 50th week using Logistic Regression Model...\n\n")

    #Predict probability of buying the product in the 50th week
    test_df['prediction'] = model.predict_proba(test_df.iloc[:,[0,1,3,4,5,6,7]])[:,1]

    #Dump the output data into the csv file
    test_df[['i','j','prediction']].to_csv(PROBABILITY_PREDICTION_FILE, header = True, index = False)
    
    print("Script completed and prediction output file generated successfully.\n")
    
    print("--- Script execution time : %s seconds ---" % (time.time() - start_time))

