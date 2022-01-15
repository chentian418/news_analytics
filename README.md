# The Determinants of Analyst Forecast Revisions: Exploratory analysis through Word embeddings and Topic modeling
## Social Science Question
Outside analysts make monthly forecast about target companies' earnings per share (EPS) at certain future times, say 2-year-ahead EPS. As their estimates would have effect on investors' decision making, and more importantly, any deviation from the realized EPS and estimated ones would reflect on the corresponding stock prices. Therefore, we are interested in understanding the decision-making process of the analysts' revision of the monthly EPS forecast. 

The revision is approximately made monthly, and as an ocean of news come in during the monthly interval, the value-relevant information embedded in this unstructured datasets are of vital importance. The research question is, **what are the determinants for analysts to make forecast revisions for two-year-ahead EPS estimates?** In other words, we want to understand what factors contribute to analysts’ decision making when revising the forecast before the actual earnings announcement. While the determinants would have different degrees of correlations with the forecast revision, it’s more ideal that we could develop a prediction model based on the impacting factors.

Therefore, an exploratory analysis of the news data would be a tremendous start. To investigate possible determinants of analyst forecast revisions, the first step is to examine the dimensions that have connection with analyst forecast revisions supported by previous literatures. A topic modeling approach could also help us to have an overview of the trends of dynamic news. Next, we can construct these cultural dimensions using huge corpora of news text from the past and the word embedding models.
## Research Design and Plan
a. Propose hypothesis about possible determinants of analyst forecast revisions using evidence from relevant literatures.

b. Collect dynamic news data on Wall Street Journal from ProQuest text and data mining (TDM) studios.

c. Get the xml file content, parse the files and convert data of different months into DataFrame using MultiProcessing Pool.

d. Build topic modeling models: use Scikit-learn CountVecterizer to produce term-document matrix and use Scikit-learn NMF model to train the topic modeling, parallelizing using joblib and Dask in the backend.

e. Train dynamic monthly word-embedding models using rolling monthly data in the most recent two years using Word2Vec and Dask DataFrame, extracting the cultural dimensions that are relevant to analyst forecast revisions we mentioned below.

Some future research to be extended:

g. Train word-embedding models using the most recent five-year data using Word2Vec and extract the cultural dimensions that have significant explanatory power for analyst forecast revision. For the news happening in the most recent month before the analyst forecast revisions are made, we use the centroid-based text summarization method to extract the centroid vectors of the texts, and project them onto the cultural dimensions we have, to see the direction of impacts the most recent news have on the analyst forecast revisions through the channel of the constructed cultural dimensions.

h.	On the other hand, we can also fit the word embedding models to the monthly news before the analyst making the revision, and the word vectors can feed into a machine learning models to further explain or predict the magnitude and dimensions of the forecast revisions.

## Dataset and ProQuest TDM Studio
ProQuest TDM Studio is a cloud-based tool for performing text and data mining on content that the Library licenses from ProQuest. This includes current and historical newspapers, news wires, trade journals and magazines, and theses and dissertations. TDM Studio gives researchers the ability to analyze large textual datasets that had previously been unavailable due to copyright and technical limitations.

We select news from Wall Street Journal from December 1, 2019 to November 30, 2021, which adds up to 96,599 articles, in the form of xml files. These datasets are stored in S3 buckets on AWS, and can be accessed and processed from the SageMaker Notebook on the ProQuest TDM Studio.

## Convert Large Batches of xml Files to Dataframe Using Multiprocessing Pool

**Multiprocessing** is a useful python package that enables the user to utilize multiple processors on a given machine for more efficient progress. The **Pool object allows data parallelism**--making the function execution of multiple input values more convenient through split processes. This sample script displays the use of Multiprocessing Pool in parsing large numbers of XML files.

Multiprocessing is preferred when calling functions on larger sets of data. The concept of data-parallelism allows independent processes to run simultaneously without having to communicate with other processes to perform the particular function on its data.

Our script [MultiprocessingPool](https://github.com/lsc4ss-a21/final-project-news_analytics/blob/main/ConvertToDataframeMultiprocessing.ipynb) displays the use of multiprocessing in parsing the xml files. It creates the function for parsing, creates a pool object, then calls the function using that pool object to run multiprocessing. 

The following codes define a thread Pool to process multiple XML files simultaneously and apply functions with Pool to the corpus, utilizing all the available cores.
```
# When using multiple processes, important to eventually close them to avoid memory/resource leaks
try:
    # Define a thread Pool to process multiple XML files simultaneously and 
    # Default set to num_cores, but may change number of processes depending on instance
    p = Pool(processes=num_cores)
    
    # Apply function with Pool to corpus, may limit number of articles by using split
    processed_lists = p.map(make_lists, articles)

except:
    print("Error in processing document")
    
finally:
    p.close()
```

## Word Embedding Models ``Word2Vec``
Word-embedding models input large collections of digitalized text and output a high-dimensional vector-space model (Mikolov, Yih, and Zweig 2013; Pennington, Socher, and Manning 2014), in a way that best preserved the distances between words across all contexts. Specifically, each word is represented by a word vector, and words sharing similar contexts within the text will be positioned nearby in the vector space. 

Below is a schematic illustration of the word embedding problems(Kozlowski, Taddy, and Evans 2019), where we represent all Words from a corpus within a k-dimensional space that best preserves distances between words in their local contexts. The solution bolded on the right hand side is an n-by-k matrix of values, where k « m. Moreover, distance between words in an embedding space is typically assessed using “cosine similarity,” the cosine of the angle between two word vectors.

![word_embeddings](https://github.com/lsc4ss-a21/final-project-news_analytics/blob/main/word_embeddings.png)
In this research, we will deploy the most widely used word embedding algorithm, word2Vec, which uses a shallow, two-layered neural network architecture which optimizes the prediction of words based on shared contexts with other words (Kozlowski, Taddy, and Evans 2019). According to their ideas, derived dimensions of word embedding vector spaces can correspond closely to “cultural dimensions”, which includes the potential impacting factors of the analyst forecast revisions, such us uncertainty, positivity and completeness. 

The codes of the Word2Vec model can be found in [word2vec](https://github.com/lsc4ss-a21/final-project-news_analytics/blob/main/word2vec.ipynb). After loading the data into **Dask Dataframe**, we tokenize the data using the customized ``tokenize_ob`` function from the [dwe](https://github.com/lsc4ss-a21/final-project-news_analytics/blob/main/dwe.py) module. Using Dask Dataframe allows us to delay all the DataFrame operations that requires large computing capabilities and compute them all at one in parallel at the end. One example of this is shown below:

```
# Tokenize the text by dwe
dict_dask2 = {}
months = df_dask['YM'].unique()
df_group = df_dask.groupby('YM')
for month in months:
    dict_dask2[month] = dwe.tokenize_ob(df_group.get_group(month).Text).compute()
```

After tokenization, We use Gensim's Word2Vec Model to train the model. By specifying **``workers=-1``**, under the hood **gensim** automatically utilize all the available workers to **parallelize** the training process, as shown below:
```
#Use Gensim's Word2Vec implementation to transform words into vectors
model = {}
for month, tokens in dict_dask2.items():
    model[month] = models.word2vec.Word2Vec(tokens, workers=-1)
    model[month].save('w2v_{}.model'.format(month))

model = {str(month): models.KeyedVectors.load('w2v_{}.model'.format(month)) for month in months}
```

## Potential determinants of Analyst forecast revisions
By combining word vectors relevant to this dimensions, the corresponding dimensions can be further constructed. For example, an approximation of the positivity dimension can be captured not only by $\overrightarrow{good} - \overrightarrow{bad}$, but also by any other pairs of words whose semantic difference corresponds to the same cultural dimension of interests, like $\overrightarrow{best} - \overrightarrow{worst}$ and $\overrightarrow{awesome} - \overrightarrow{awful}$. We can take the arithmetic mean of these a set of such pairs to represent the dimensions.

Below are some examples of the dimensions of interests and possible ways to construct the corresponding dimensions:

1. Market sentiment

Market sentiment is a prominent factor valued by firms and analysts. Some pre-existing measures include CBOE Volatility Index, High-low index, Bullish Percent Index, moving average, etc. Market sentiment can be customized using the difference between word vectors, for example $\overrightarrow{optimistic} - \overrightarrow{perssimistic}$.

2. Macroeconomic uncertainty and cyclicality

Macroeconomic uncertainty jointly affect management and analyst forecast(Yang and Chen 2021). Specifically, increasing macroeconomic uncertainty reduce the tendency for management to issue earnings forecast, which are heavily driven by firms with low cyclicality. When macroeconomic uncertainty is high, analysts issue more accurate earnings forecast than management for firms with high cyclicality. Therefore, customized measures around macroeconomic uncertainty and cyclicality can be constructed from the corpus of the past five years before the earnings forecast revisions.

3. Information Uncertainty

Information uncertainty stems from two sources: the volatility of a ﬁrm’s underlying fundamentals and poor information (see Zhang 2006). The author uses dispersion in analysts’ earnings forecasts as a proxy for information uncertainty, which is measured as the standard deviation of analyst forecasts made four months prior to ﬁscal year-end, scaled by the prior year-end stock price.

Then we could construct relevant dimensions from sets of word vectors in the Word2Vec model; for example,

The positivity dimension:
```
# This is the vector representation of a word:
model['2021-09'].wv['good'] - model['2021-09'].wv['bad']

array([-0.01246194, -0.00423844, -0.00754492, -0.00094433, -0.0076013 ,
       -0.0039413 ,  0.00354507,  0.00443552, -0.0053397 , -0.00448217,
        0.00812644, -0.01373298,  0.00528097, -0.00261526, -0.00158261,
        0.00662444,  0.00707565, -0.00618023,  0.00849694, -0.00534548,
        0.00836258, -0.00739764,  0.00376004,  0.00388304, -0.0100096 ,
        0.00204969,  0.00824579, -0.00489771,  0.01671576,  0.00554534,
       -0.00456122, -0.00910323,  0.01165122,  0.00956067,  0.00326219,
       -0.01616808,  0.0046078 , -0.0040546 , -0.00817754, -0.01651328,
        0.00313228,  0.00101648,  0.00323665,  0.01068779,  0.01575492,
        0.00822953,  0.00199978,  0.00343103, -0.007583  ,  0.00367907,
        0.00315927,  0.00935874, -0.00433332,  0.00300984, -0.00906182,
        0.00901577, -0.01358785,  0.00736532, -0.01367095, -0.01630726,
       -0.01458818,  0.0046094 ,  0.0012912 ,  0.00343539, -0.00633125,
        0.013795  ,  0.00129037,  0.00793879, -0.00089897,  0.0101614 ,
        0.01685372,  0.00706067,  0.00144489, -0.00490889,  0.0013182 ,
       -0.00081446, -0.00467878, -0.01054413,  0.00248013,  0.01453103,
       -0.00503271, -0.01719952, -0.00407277,  0.00951267,  0.00539712,
       -0.00497514,  0.00241731, -0.00613017, -0.00511101,  0.00093622,
        0.00444427,  0.00315092,  0.00913856,  0.01368533,  0.01408858,
       -0.00586777,  0.01804875, -0.00802949,  0.00137819,  0.0010875 ],
      dtype=float32)
```
The certainty dimension:
```
model['2021-09'].wv['certain']-model['2021-09'].wv['uncertain']

array([ 0.00238759, -0.00905487,  0.01684976, -0.00259188,  0.0003051 ,
        0.00352847, -0.00081572,  0.00749601,  0.0084063 ,  0.01165399,
        0.00863019,  0.01082332, -0.01708635, -0.00474653,  0.00330676,
        0.00573033, -0.00324709, -0.00581241,  0.00340554, -0.00061988,
        0.00110295,  0.00274357, -0.00974507,  0.00344604, -0.00409065,
        0.00578468, -0.00585447, -0.00065301,  0.00258926,  0.00149495,
        0.00045994,  0.01031436,  0.01083726, -0.00607581,  0.01331528,
        0.01213955, -0.01037435,  0.00162626,  0.00698686,  0.00200312,
       -0.01169665,  0.00290381, -0.00250407, -0.00675524, -0.00913734,
        0.0042626 ,  0.00784279, -0.00172653, -0.00687744, -0.00050499,
        0.00071931,  0.01472743,  0.01618569,  0.00239614,  0.00531416,
       -0.01645597,  0.0083224 , -0.00171803, -0.01376249, -0.00318439,
       -0.01074727, -0.00494634, -0.00136675,  0.01503177, -0.0109706 ,
       -0.01716124, -0.00405496,  0.01767059,  0.0030091 ,  0.00022994,
       -0.00567238,  0.00348791,  0.01505918, -0.0071577 , -0.01308936,
       -0.00430985,  0.00710196,  0.00366743, -0.00639988,  0.01252693,
       -0.00842993, -0.00423205,  0.00103543,  0.01127114, -0.00267334,
       -0.01296536,  0.00593457,  0.0010532 , -0.00085927,  0.01390136,
        0.00681996,  0.0144337 , -0.01616817,  0.00972665,  0.01733634,
       -0.00524847, -0.00066928, -0.00880299, -0.0104724 ,  0.00024109],
      dtype=float32)
```
We can also find the most similar words by comparing the distance between word vectors to the target word vectors:

```
print(model['2021-09'].wv.most_similar('good'))

[('bronx', 0.36144378781318665), ('auto', 0.3456558287143707), ('studentloan', 0.3406968116760254), ('heartbreak', 0.332455575466156), ('dens', 0.3287903666496277), ('cent', 0.32342860102653503), ('geometr', 0.3194636106491089), ('shanksvill', 0.3189867436885834), ('margaret', 0.31608015298843384), ('kenneth', 0.31579887866973877)]

print(model['2021-09'].wv.most_similar('bad'))

[('rational', 0.41860443353652954), ('widest', 0.35977911949157715), ('brake', 0.3488847315311432), ('soot', 0.32659947872161865), ('venkatprwsjcom', 0.32314053177833557), ('iea', 0.31359636783599854), ('rail', 0.30930137634277344), ('retent', 0.3029954433441162), ('broken', 0.3028797507286072), ('charley', 0.30199503898620605)]

print(model['2021-10'].wv.most_similar('certain'))

[('parisroubaix', 0.3387027084827423), ('kyrsten', 0.3372078537940979), ('cf', 0.33583611249923706), ('gamechang', 0.3256683647632599), ('martyn', 0.3209635615348816), ('aniston', 0.3203233778476715), ('amount', 0.3187102675437927), ('oxfam', 0.31554776430130005), ('coup', 0.3130550980567932), ('humana', 0.3085482120513916)]

print(model['2021-10'].wv.most_similar('uncertain'))

[('amish', 0.3379340171813965), ('worldview', 0.33166059851646423), ('smooth', 0.3286925256252289), ('invers', 0.3266356289386749), ('propos', 0.3259832561016083), ('homerenov', 0.3255958557128906), ('speaker', 0.31199002265930176), ('cohort', 0.3119129240512848), ('oil', 0.3117658495903015), ('diablo', 0.3100152611732483)]
```

Aligning Diachronic Word Embeddings allows us to compare word vectors generated from dynamic embedding models using data from different time:

```
#assign two models
dwe.smart_procrustes_align_gensim(model['2020-11'], model['2021-11'])

Alignment Complete

print('2020-11', model['2020-11'].wv.distance('good', 'certain'))
print('2021-11', model['2021-11'].wv.distance('good', 'certain'))

2020-11 1.1797177493572235
2021-11 0.9868703782558441
```

## Topic Modeling

We use the `sklearn.decomposition.NMF` for topic modeling, and `scikit-learn CountVectorizer` for tokenization and vocabulary creation. NMF is short for Non-Negative Matrix Factorization, which aims to find two non-negative matrices (W, H) whose product approximates the non- negative matrix X. It is believed to provide more coherent results than LDA models, another popular model for topic modeling.

The objective function is:
```
0.5 * ||X - WH||_Fro^2
+ alpha * l1_ratio * ||vec(W)||_1
+ alpha * l1_ratio * ||vec(H)||_1
+ 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2
+ 0.5 * alpha * (1 - l1_ratio) * ||H||_Fro^2
```
where
```
||A||_Fro^2 = \sum_{i,j} A_{ij}^2 (Frobenius norm)
||vec(A)||_1 = \sum_{i,j} abs(A_{ij}) (Elementwise L1 norm)
```
The code of this part can be referred at [topic_modeling.ipynb](https://github.com/lsc4ss-a21/final-project-news_analytics/blob/main/topic_modeling.ipynb). The first of the code loads data using Multiprocessing Pool as mentioned above, and we use data from 2021/09 for illustration. It can be easily scaled up if we want to use larger dataset.

We use scikit-learn CountVectorizer to create a vectorizer object which will then be used to create a term-document matrix in the next cell. This countVectorizer includes lots of valuable pre-processing parameters and optionsincluding as follows:
```
#Set parameters for the tokenization and vocabulary creation. read from files 
vectorizer = CountVectorizer(max_df=.8, min_df=10, stop_words='english', max_features=5000)

#Produce term-document matrix and use joblib for paralleling
with joblib.parallel_backend("dask"):
    X = vectorizer.fit_transform(df_metadata_text['text'].tolist())

features = vectorizer.get_feature_names()
```
It should be noted that we use **`joblib` with `dask` specification to speed up the pre-processing**.

Then we fit the topic modling model using `NMF`, and we also use **`joblib` with `dask` specification** to speed up the calculation. We set the number of topics as 30 for our analysis.
```
model = NMF(n_components=number_of_topics, init='random', random_state=0)
with joblib.parallel_backend("dask"):
    W = model.fit_transform(X.toarray())

df_W = pd.DataFrame(W)

H = model.components_
df_H = pd.DataFrame(H)
```
We later play around the 30 topics with 10 words each. As it is monthly data, we graph the trend of each topic accross the mothon. Some examples as shown as below.


## Results from Topic Modeling

Next, we look that the results from topic modeling. The words within each topic are quite coherent as we suggessted at the beginning.

The most popular topic is shown below. This topic is closely related to people's daily life which is frequently mentioned each day. 
![image](https://user-images.githubusercontent.com/72136058/145681460-504b1d07-2a8b-4e1c-a78b-8ea0d92bcd40.png)

The second most popular topic is closely related to social media. Tokens such as `instagram` and `facebook` are mentioned. Tokens `teen` and `teens` together appear twice. This infers that teens' use of social media is a heated topic during the middle of the month.
![image](https://user-images.githubusercontent.com/72136058/145681750-51a18878-024f-4f99-b8f1-153cc8633209.png)

Unsurprisingly, the topic related to Covid-19 is frequently mentioned, with the special concern on Delta variation. Topic 11 also shows that vaccines are regraded as an efficient way to prevent Covid, and people were considering getting boosted shots to deal with the changing nature of the covid virus.
![image](https://user-images.githubusercontent.com/72136058/145682309-1bb8377a-1bac-4a31-8bce-044c5d158b8c.png)
![image](https://user-images.githubusercontent.com/72136058/145682948-f74c5ff7-3423-4811-bba8-700afd1386dd.png)


Sino-the U.S. relation is also a popular topic in the Wall Street Journal during the September 2021. Trade is one of the tensions.
![image](https://user-images.githubusercontent.com/72136058/145682660-2e8f69dc-b38c-445a-bdad-9ab65fbef6d5.png)

Please refer to the complete 30 topics and their trends during the month at the end of [topic_modeling](https://github.com/lsc4ss-a21/final-project-news_analytics/blob/main/topic_modeling.ipynb).





## Reference
*	Mikolov, Tomáš, Wen-tau Yih, and Geoffrey Zweig. 2013. "Linguistic regularities in continuous space word representations." Proceedings of the 2013 conference of the north american chapter of the association for computational linguistics: Human language technologies.
*	Pennington, Jeffrey, Richard Socher, and Christopher D Manning. 2014. "Glove: Global vectors for word representation." Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP).
*	Kozlowski, Austin C, Matt Taddy, and James A Evans. 2019. "The geometry of culture: Analyzing the meanings of class through word embeddings."  American Sociological Review 84 (5):905-949.
*   Cichocki, Andrzej, and P. H. A. N. Anh-Huy. “Fast local algorithms for large scale nonnegative matrix and tensor factorizations.” IEICE transactions on fundamentals of electronics, communications and computer sciences 92.3: 708-721, 2009.
*  Fevotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix factorization with the beta-divergence. Neural Computation, 23(9).



