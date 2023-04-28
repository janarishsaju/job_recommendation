import os
os.chdir(r'C:\Users\Janarish\Desktop\job_recommendation')

import pandas as pd
import numpy as np
import turicreate as tc
import time

users = pd.read_csv('data/uid.csv') 

jobs = pd.read_csv('data/jobs.csv') 

jobseekers = pd.read_csv('data/jobseekers.csv') 
jobseekers.rename(columns = {'Gender':'preferred_gender', 'Languages Known':'preferred_languages'}, inplace = True) 

transactions  = pd.read_csv('data/transactions.csv')
transactions = transactions[['uid', 'nid']]
transactions['uid'] = transactions['uid'].apply(str)
transactions['nid'] = transactions['nid'].apply(str)
transactions = transactions.groupby(['uid'])['nid'].apply('|'.join).reset_index()

transactions['nid'] = transactions['nid'].apply(lambda x: [int(i) for i in x.split('|')])

data = pd.melt(transactions.set_index('uid')['nid'].apply(pd.Series).reset_index(), 
              id_vars=['uid'],
              value_name='nid') \
    .dropna().drop(['variable'], axis=1) \
    .groupby(['uid', 'nid']) \
    .agg({'nid': 'count'}) \
    .rename(columns={'nid': 'visit_count'}) \
    .reset_index() \
    .rename(columns={'nid': 'nid'})
data['nid'] = data['nid'].astype(np.int64)

# =============================================================================
# Change filters
# =============================================================================
start_time = time.time()
#Filtering Operations
data_input = data.merge(jobs, on=['nid'], how='left', indicator=True)
users_input = users.merge(jobseekers, on=['uid'], how='left', indicator=True)

job_type = ['fulltime', 'both']  #Change as per need
job_type_pat = r'(\b{}\b)'.format('|'.join(job_type))

gender = ['Male']  #Change as per need
gender_pat = r'(\b{}\b)'.format('|'.join(gender))

languages = ['English']  #Change as per need
languages_pat = r'(\b{}\b)'.format('|'.join(languages))

data_input = data_input[data_input["jobtype"].str.contains(job_type_pat, case=False, na=False) & data_input["preferred_gender"].str.contains(gender_pat, case=False, na=False) & data_input["preferred_languages"].str.contains(languages_pat, case=False, na=False)]
data_input = data_input[['uid', 'nid', 'visit_count']]

users_input = users_input[users_input["preferred_gender"].str.contains(gender_pat, case=False, na=False) & users_input["preferred_languages"].str.contains(languages_pat, case=False, na=False)]
users_input = users_input[['uid']]

train_data = tc.SFrame(data_input)

# constant variables to define field names include:
user_id = 'uid'
item_id = 'nid'
users_to_recommend = list(users_input['uid'])
n_rec = 10 # number of items to recommend
n_display = 30 # to display the first few rows in an output dataset

def model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display):
    if name == 'popularity':
        model = tc.popularity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target)
    elif name == 'cosine':
        model = tc.item_similarity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target, 
                                                    similarity_type='cosine')
    elif name == 'pearson':
        model = tc.item_similarity_recommender.create(train_data, 
                                                        user_id=user_id, 
                                                        item_id=item_id, 
                                                        target=target, 
                                                        similarity_type='pearson')
            
        recom = model.recommend(users=users_to_recommend, k=n_rec)
        recom.print_rows(n_display)
    return model

name = 'cosine'
target = 'visit_count'
cos = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

popularity_recomm = cos.recommend(users=users_to_recommend, k=n_rec)
df_rec = popularity_recomm.to_dataframe()

df_rec['recommendednid'] = df_rec.groupby([user_id])[item_id].transform(lambda x: '|'.join(x.astype(str)))
df_output = df_rec[['uid', 'recommendednid']].drop_duplicates().sort_values('uid').set_index('uid')

df_output.to_csv("predictions/output.csv")

print("--- %s seconds ---" % (time.time() - start_time))