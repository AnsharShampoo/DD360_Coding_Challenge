import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
def get_cols_with_no_nans(df,col_type):
    '''
    Arguments :
    df : The dataframe to process
    col_type : 
          num : to only get numerical columns with no nans
          no_num : to only get nun-numerical columns with no nans
          all : to get any columns with no nans    
    '''
    if (col_type == 'num'):
        predictors = df.select_dtypes(exclude=['object'])
    elif (col_type == 'no_num'):
        predictors = df.select_dtypes(include=['object'])
    elif (col_type == 'all'):
        predictors = df
    else :
        print('Error : choose a type (num, no_num, all)')
        return 0
    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no_nans

raw_data = pd.read_csv(r'reto_precios.csv')
num_cols = get_cols_with_no_nans(raw_data , 'num')
combined = raw_data[num_cols]
train_data = raw_data[num_cols]
train = train_data[:700]
train_x = train.drop(['final_price','price_square_meter','price_mod','id','age_in_years','m2','since_value','days_on_site'], axis=1)
train_y = train['price_square_meter']
model = RandomForestRegressor()
model.fit(train_x,train_y)
with open('rbmodel.pkl', 'wb') as li:
    pickle.dump(model,li)
