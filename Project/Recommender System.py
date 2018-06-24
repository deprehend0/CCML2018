# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:52:43 2018

@author: Gebruiker
"""

#Based on the structure on https://www.kaggle.com/gspmoreira/recommender-systems-in-python-101
#Totale runtime: 1:11:00:00 

#%%
from __future__ import division
import numpy as np
import scipy
import seaborn as sns
import pandas as pd
import math
import random
import sklearn
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df = pd.read_excel("../data/Persoon_Skills.xlsx")
Skill_list = pd.read_excel("../data/Skill_lijst.xlsx")
#%%
#Removing duplicate Skills in one Person
df['ColumnA'] = df[df.columns[0:2]].apply(lambda x: ','.join(x.dropna().astype(str).astype(str)),axis=1)
df = df.sort_values('first-name', ascending=False).drop_duplicates('ColumnA').sort_index()
X = df.drop(['ColumnA'], axis=1, inplace=True)

#Removing Duplicate Skills from list
Skill_list['ColumnA'] = Skill_list[Skill_list.columns[0:1]].apply(lambda x: ','.join(x.dropna().astype(str).astype(str)),axis=1)
Skill_list = Skill_list.sort_values('Skill', ascending=False).drop_duplicates('ColumnA').sort_index()
X = Skill_list.drop(['ColumnA'], axis=1, inplace=True)

event_type_strength = {0}

#%%
### Recommending Skills to People
Skill_list['Skill'] = Skill_list['Skill'].astype(str)
Skill_list['contentId'] = le.fit_transform(Skill_list['Skill'])
articles_df = Skill_list.copy()
articles_df = articles_df.reset_index()
articles_df = articles_df[['Skill', 'contentId']]
articles_df['ColumnA'] = articles_df[articles_df.columns[0:2]].apply(lambda x: ','.join(x.dropna().astype(str).astype(str)),axis=1)
articles_df = articles_df.sort_values('Skill', ascending=False).drop_duplicates('ColumnA').sort_index()
X = articles_df.drop(['ColumnA'], axis=1, inplace=True)

df['personId'] = le.fit_transform(df['first-name'])
df['Skill'] = df['Skill'].astype(str)
df['contentId'] = le.fit_transform(df['Skill'])

person_skills = df[['personId', 'first-name', 'contentId', 'Skill']]

#%%
### Recommending People based on Skills
df['Skill'] = df['Skill'].astype(str)
df['personId'] = le.fit_transform(df['Skill'])
df['contentId'] = le.fit_transform(df['first-name'])
articles_df = df.copy()
articles_df = articles_df.reset_index()
articles_df = articles_df[['first-name', 'contentId']]
articles_df['ColumnA'] = articles_df[articles_df.columns[0:2]].apply(lambda x: ','.join(x.dropna().astype(str).astype(str)),axis=1)
articles_df = articles_df.sort_values('first-name', ascending=False).drop_duplicates('ColumnA').sort_index()
X = articles_df.drop(['ColumnA'], axis=1, inplace=True)

#%%
df['eventStrength'] = 1


users_interactions_count_df = df.groupby(['personId', 'contentId']).size().groupby('personId').size()
print('# users: %d' % len(users_interactions_count_df))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 3].reset_index()[['personId']]
print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))
          
print('# of interactions: %d' % len(df))
interactions_from_selected_users_df = df.merge(users_with_enough_interactions_df, 
               how = 'right',
               left_on = 'personId',
               right_on = 'personId')
print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df)) 

def smooth_user_preference(x):
    return math.log(1+x, 2)

interactions_full_df = interactions_from_selected_users_df \
                    .groupby(['personId', 'contentId'])['eventStrength'].sum() \
                    .apply(smooth_user_preference).reset_index()
print('# of unique user/item interactions: %d' % len(interactions_full_df))
interactions_full_df.head(10)    

interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                   stratify=interactions_full_df['personId'], 
                                   test_size=0.40,
                                   random_state=42)

interactions_full_df = interactions_from_selected_users_df \
                    .groupby(['personId', 'contentId'])['eventStrength'].sum() \
                    .apply(smooth_user_preference).reset_index()
print('# of unique user/item interactions: %d' % len(interactions_full_df))
interactions_full_df.head(10)

print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))
      
interactions_full_indexed_df = interactions_full_df.set_index('personId')
interactions_train_indexed_df = interactions_train_df.set_index('personId')
interactions_test_indexed_df = interactions_test_df.set_index('personId')
#%% Recommending People
stopwords_list = stopwords.words('dutch') + stopwords.words('english')
vectorizer = TfidfVectorizer(analyzer='word', stop_words=stopwords_list)

item_ids = articles_df['contentId'].tolist()
tfidf_matrix = vectorizer.fit_transform(articles_df['first-name'])
tfidf_feature_names = vectorizer.get_feature_names()
tfidf_matrix

#%% Recommending Skills
stopwords_list = stopwords.words('dutch') + stopwords.words('english')
vectorizer = TfidfVectorizer(analyzer='word', stop_words=stopwords_list)

item_ids = articles_df['contentId'].tolist()
tfidf_matrix = vectorizer.fit_transform(articles_df['Skill'])
tfidf_feature_names = vectorizer.get_feature_names()
tfidf_matrix

#%%
def get_items_interacted(person_id, interactions_df):
    interacted_items = interactions_df.loc[person_id]['contentId']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 1000

class ModelEvaluator:


    def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
        interacted_items = get_items_interacted(person_id, interactions_full_indexed_df)
        all_items = set(articles_df['contentId'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):        
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index

    def evaluate_model_for_user(self, model, person_id):
        #Getting the items in test set
        interacted_values_testset = interactions_test_indexed_df.loc[person_id]
        if type(interacted_values_testset['contentId']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['contentId'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['contentId'])])  
        interacted_items_count_testset = len(person_interacted_items_testset) 
        
        

        #Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_items(person_id, 
                                               items_to_ignore=get_items_interacted(person_id, 
                                                                                    interactions_train_indexed_df), 
                                               topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        hits_at_X_count = 0
        #For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            #Getting a random sample (1000) items the user has not interacted 
            #(to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id, 
                                                                          sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, 
                                                                          seed=item_id%(2**32))

            #Combining the current interacted item with the 1000 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            #Filtering only recommendations that are either the interacted item or from a random sample of 1000 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['contentId'].isin(items_to_filter_recs)]                    
            valid_recs = valid_recs_df['contentId'].values
            #Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10
            hit_at_X, index_at_X = self._verify_hit_top_n(item_id, valid_recs, 100)
            hits_at_X_count += hit_at_X

        #Recall is the rate of the interacted items that are ranked among the Top-N recommended items, 
        #when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)
        precision_at_5 = hits_at_5_count / float((hits_at_X_count) + 0.0000000001)
        precision_at_10 = hits_at_10_count / float((hits_at_X_count) + 0.0000000001)
        #F1_at_5 = 2 * ((precision_at_5 * recall_at_5) / float(precision_at_5 + recall_at_5))
        #F1_at_10 = 2 * ((precision_at_10 * recall_at_10) / float(precision_at_10 + recall_at_10))

        person_metrics = {'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'interacted_count_test': interacted_items_count_testset,
                          'hits@X_count': hits_at_X_count,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10,
                          'precision@5': precision_at_5,
                          'precision@10': precision_at_10}
         #                 'F1-Score@5': F1_at_5,
          #                'F1-Score@10': F1_at_10}
        return person_metrics

    def evaluate_model(self, model):
        #print('Running evaluation for users')
        people_metrics = []
        for idx, person_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
            #if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, person_id)  
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics) \
                            .sort_values(['interacted_count_test','hits@X_count'], ascending=False)
                            
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count_test'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count_test'].sum())
        global_precision_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['hits@X_count'].sum() + 0.0001)
        global_precision_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['hits@X_count'].sum() + 0.0001)
        global_F1_at_5 = 2 * ((global_precision_at_5 * global_recall_at_5) / float(global_precision_at_5 + global_recall_at_5))
        global_F1_at_10 = 2 * ((global_precision_at_10 * global_recall_at_10) / float(global_precision_at_10 + global_recall_at_10))
        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10,
                          'precision@5': global_precision_at_5,
                          'precision@10': global_precision_at_10,
                          'F1-Score@5': global_F1_at_5,
                          'F1-Score@10': global_F1_at_10}    
        return global_metrics, detailed_results_df
    
model_evaluator = ModelEvaluator() 

item_popularity_df = interactions_full_df.groupby('contentId')['eventStrength'].sum().sort_values(ascending=False).reset_index()
item_popularity_df.head(10)   

class PopularityRecommender:
    
    MODEL_NAME = 'Popularity'
    
    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=100, verbose=False):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['contentId'].isin(items_to_ignore)] \
                               .sort_values('eventStrength', ascending = False) \
                               .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['eventStrength', 'PersonId', 'Omschrijving']]


        return recommendations_df
    
popularity_model = PopularityRecommender(item_popularity_df, articles_df)

print('Evaluating Popularity recommendation model...')
pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model)
print('\nGlobal metrics:\n%s' % pop_global_metrics)
pop_detailed_results_df.head(10)

def get_item_profile(item_id):
    idx = item_ids.index(item_id)
    item_profile = tfidf_matrix[idx:idx+1]
    return item_profile

def get_item_profiles(ids):
    item_profiles_list = [get_item_profile(x) for x in ids]
    item_profiles = scipy.sparse.vstack(item_profiles_list)
    return item_profiles

def build_users_profile(person_id, interactions_indexed_df):
    interactions_person_df = interactions_indexed_df.loc[person_id]
    user_item_profiles = get_item_profiles(interactions_person_df['contentId'])
    
    user_item_strengths = np.array(interactions_person_df['eventStrength']).reshape(-1,1)
    #Weighted average of item profiles by the interactions strength
    user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths)
    user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
    return user_profile_norm

def build_users_profiles(): 
    interactions_indexed_df = interactions_full_df[interactions_full_df['contentId'] \
                                                   .isin(articles_df['contentId'])].set_index('personId')
    user_profiles = {}
    for person_id in interactions_indexed_df.index.unique():
        user_profiles[person_id] = build_users_profile(person_id, interactions_indexed_df)
    return user_profiles

user_profiles = build_users_profiles()
len(user_profiles)

    
class ContentBasedRecommender:
    
    MODEL_NAME = 'Content-Based'
    
    def __init__(self, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        #Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(user_profiles[person_id], tfidf_matrix)
        #Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        #Sort the similar items by similarity
        similar_items = sorted([(item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=100, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        #Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['contentId', 'recStrength']) \
                                    .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrength', 'contentId', 'PersonId', 'Omschrijving']]


        return recommendations_df
    
content_based_recommender_model = ContentBasedRecommender(articles_df)

print('Evaluating Content-Based Filtering model...')
cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(content_based_recommender_model)
print('\nGlobal metrics:\n%s' % cb_global_metrics)
cb_detailed_results_df.head(10)

users_items_pivot_matrix_df = interactions_train_df.pivot(index='personId', 
                                                          columns='contentId', 
                                                          values='eventStrength').fillna(0)

users_items_pivot_matrix = users_items_pivot_matrix_df.as_matrix()
users_ids = list(users_items_pivot_matrix_df.index)

NUMBER_OF_FACTORS_MF = 15
U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
sigma = np.diag(sigma)

U.shape
Vt.shape
sigma.shape

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
all_user_predicted_ratings

cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
len(cf_preds_df.columns)

class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=100, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
                                    .reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['contentId'].isin(items_to_ignore)] \
                               .sort_values('recStrength', ascending = False) \
                               .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrength', 'contentId', 'PersonId', 'Omschrijving']]


        return recommendations_df
    
cf_recommender_model = CFRecommender(cf_preds_df, articles_df)

print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)
print('\nGlobal metrics:\n%s' % cf_global_metrics)
cf_detailed_results_df.head(10)

class HybridRecommender:
    
    MODEL_NAME = 'Hybrid'
    
    def __init__(self, cb_rec_model, cf_rec_model, items_df):
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=100, verbose=False):
        #Getting the top-1000 Content-based filtering recommendations
        cb_recs_df = self.cb_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose,
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCB'})
        
        #Getting the top-1000 Collaborative filtering recommendations
        cf_recs_df = self.cf_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose, 
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCF'})
        
        #Combining the results by contentId
        recs_df = cb_recs_df.merge(cf_recs_df,
                                   how = 'inner', 
                                   left_on = 'contentId', 
                                   right_on = 'contentId')
        
        #Computing a hybrid recommendation score based on CF and CB scores
        recs_df['recStrengthHybrid'] = recs_df['recStrengthCB'] * recs_df['recStrengthCF']
        
        #Sorting recommendations by hybrid score
        recommendations_df = recs_df.sort_values('recStrengthHybrid', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrengthHybrid', 'contentId', 'PersonId']]


        return recommendations_df
    
hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, articles_df)

print('Evaluating Hybrid model...')
hybrid_global_metrics, hybrid_detailed_results_df = model_evaluator.evaluate_model(hybrid_recommender_model)
print('\nGlobal metrics:\n%s' % hybrid_global_metrics)
hybrid_detailed_results_df.head(10)

global_metrics_df = pd.DataFrame([pop_global_metrics, cf_global_metrics, cb_global_metrics, hybrid_global_metrics]) \
                        .set_index('modelName')
global_metrics_df

ax = global_metrics_df.transpose().plot(kind='bar', figsize=(15,8))
for p in ax.patches:
    ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(5, 10), textcoords='offset points')
    
def inspect_interactions(person_id, test_set=True):
    if test_set:
        interactions_df = interactions_test_indexed_df
    else:
        interactions_df = interactions_train_indexed_df
    return interactions_df.loc[person_id].merge(articles_df, how = 'left', 
                                                      left_on = 'contentId', 
                                                      right_on = 'contentId') \
                          .sort_values('eventStrength', ascending = False)[['eventStrength', 
                                                                          'contentId',
                                                                          'Omschrijving']]
#%%               
popularity_model.recommend_items(6927, topn=50, verbose=False)
#%%
content_based_recommender_model.recommend_items(9048, topn=50, verbose=False)
#%%
cf_recommender_model.recommend_items(9048, topn=50, verbose=False)
#%%
hybrid_recommender_model.recommend_items(9048, topn=50, verbose=False)                          
#%%
filename = 'Popularity_Model.sav'
joblib.dump(popularity_model, filename)
filename = 'Content_Based_Model.sav'
joblib.dump(content_based_recommender_model, filename)
filename = 'Collaboration_Model.sav'
joblib.dump(cf_recommender_model, filename)
filename = 'Hybrid_Model.sav'
joblib.dump(hybrid_recommender_model, filename)