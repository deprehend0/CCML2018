# Cognitive Computational Modeling of Language and Web Interaction
-----
### Description
The goal of this project was to make different models to make recommendations for skills or people based on a LinkedIn scrape. In this repo the following files are provided:

### Files
The files provided can be split into Excel files and Python files. In the case of the Python files the version used in Python 3.6.4. Older versions are possible, but not advised.

The excel files are: Persoon_Skills, Skill_lijst and User_Info. Persoon_Skills are the interactions between people and the skills they possess. This excel file is a nested table, so multiple rows can consist of one person with multiple skills. Skill_lijst is the excel file with all the unique skills that are present within the dataset. User_Info are the additional features of the user such as country and educational background. Also the index numbering of the similarity matrix is added, so persons can be traced back from the similarity matrix. With this the direct and indirect path can be found. 

### Python files
The python files are the recommender system and graph file. 

**Recommender system**
The recommender system file is based on a kernel of Kaggle were an implementation of a recommender system is described. https://www.kaggle.com/gspmoreira/recommender-systems-in-python-101

In that kernel only recall was used as a metric, in this project the metrics: precision and F1-Score were also added. See the report on how these were calculated. 

The code is split into sections and also need to be executed that way. This is because the recommender skills and recommender person approach are integrated into the same code. The only difference is the label encoding based on what you want to have recommended. In the code these specific sections have comment lines added to them, such that each section is clear on how they should be executed. 

The notable code lines in this file are the following:

```sh
df['ColumnA'] = df[df.columns[0:2]].apply(lambda x: ','.join(x.dropna().astype(str).astype(str)),axis=1)
df = df.sort_values('first-name', ascending=False).drop_duplicates('ColumnA').sort_index()
X = df.drop(['ColumnA'], axis=1, inplace=True)
```
This removes all duplicates user-item columns from the dataset. This makes sure that no user can have duplicate skills. This does not include stemmed variations on the skills. 

```sh
event_type_strength = {0}
```
In our project no weights were given to specific values, however in case studies specific values can have varying weights. Due to fairness, all values were weighted the same in the project. 

```sh
stopwords_list = stopwords.words('dutch') + stopwords.words('english')
vectorizer = TfidfVectorizer(analyzer='word', stop_words=stopwords_list)

item_ids = articles_df['contentId'].tolist()
tfidf_matrix = vectorizer.fit_transform(articles_df['Skill'])
tfidf_feature_names = vectorizer.get_feature_names()
tfidf_matrix
```
Vectorization of the skill strings. This section is used in the case of recommending skills to a specific user. In the case of recommending people based on skills, the "Skill" column is replaced by the "first-name" column. This is again shown in the code by the different sections. 

```sh
users_interactions_count_df = df.groupby(['personId', 'contentId']).size().groupby('personId').size()
print('# users: %d' % len(users_interactions_count_df))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 3].reset_index()[['personId']]
print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))
```

The 3 is used as the limit for how many skills a person needs to be used in the training. The same holds true for the amount of people with that specific skill. Lowering this below 3 may result in a problem in which the train and test set are not created properly due to the cold-start problem. Increasing the threshold lowers the amount of users in the dataset,

```sh
hybrid_recommender_model.recommend_items(9048, topn=50, verbose=False)  
```
Each model can recommend the items and the top-n recommendations. The number is based on the ContentId and needs to be checked in the interactions_full_index_df to see if the person you are looking for is at least present in the dataframe. 

```sh
filename = 'Popularity_Model.sav'
joblib.dump(popularity_model, filename)
```
This exports the model, such that it can be used in the future separate from the code. 

**Graph**
This file creates a tf-idf similarity matrix based on the User_Info excel file. Stopwords are removed from the strings and after the TFIDF vectorization a linear kernel is applied to check the similarity between users. 

Lines of code that need special attention are:

```sh
sq_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
sq_matrix[sq_matrix >= 0.5] = 1
sq_matrix[sq_matrix < 0.5] = 0
```
Lowers or heightens the similarity matrix between persons. The lower the threshold the more likely persons can be considered similar and can have a direct path. While increasing the threshold lowers the amount of persons with a direct path. 

```sh
X = np.array(sq_matrix[1851])
Y = np.array(sq_matrix[454])
Z = X + Y
```
X in this situation is the recruiter, while Y is the possible candidate. Z adds all rows between X and Y together to see if there are any shared friends between them. This can be seen if one of the rows results into 2.0. Checking the X array to see if the number recorded in Y has got a 1 or 0 in that row. In the case of 0 there is at least no direct path, is there a 1 there is a direct path. 

```sh
poi = 18 # person of interest, index number
relevant_persons = nx.ego_graph(gr, poi, center=True, undirected=False, distance=None)

# To get the other (i.e., out of range people) you calculate the subgraph within range with the code above.
# Then you give that graph with the tfidf_matrix to the code below and you get the persons that are out of range.
def get_out_of_range(in_range_graph, matrix):
    in_range_nodes = in_range_graph.nodes()
    tmp_array = []
    for i in range(len(matrix)):
        if i in in_range_nodes:
            tmp_array.append(matrix[i])

    return tmp_array
```
Used to create and visualize the social network between persons that are mostly similar to the POI. 

### Conclusion
Any problems or unclarities can be put into the issue forum of this repo. All libraries used in the Python files are mentioned in there and can be checked to see if there are any missing packages in your own personal DIY run of this code. 




