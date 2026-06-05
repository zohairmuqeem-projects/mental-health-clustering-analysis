import streamlit as st
<<<<<<< HEAD

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors


# ── Section 1: Introduction ──────────────────────────────────────────────────

st.write("""
# Global Mental Health Rates Unsupervised Learning Analysis

# 1. Introduction

Currently, "One in seven 10-19-year-olds experiences a mental disorder, accounting for 15% of the global burden of disease in this age group … yet these remain largely unrecognized and untreated" (World Health Organization, as cited in Cosgrove, 2025). Even from those diagnosed, we observe an alarming rate of associated mental disorders, and it's important now than ever to develop high-impact strategies that create effective intervention for both those diagnosed and those who may need help.

Additionally, The Global Trends in Mental Health Disorder dataset contains information regarding the prevalence of various mental health disorders in countries across the globe. Furthermore, there may be associations within the inherent clusters in the dataset that provide a better understanding of shared mental health profiles between groups of nations. These mental health profiles in the inherent clusters can help describe the varied prevalence of mental health issues, and allow for clustered groups of nations that share similar mental health disorders. This can be especially useful for world health organizations, who can develop specific strategies developed from the cluster analysis that can target similar groups of countries at once. Through identifying these patterns, world health organizations can develop regional task forces that specialize in alleviating specific mental health disorder profiles in a given cluster of countries. Together, the cluster analysis can help optimize resource allocation to assist in treating the mental health crisis, and provide insight into constructing solutions to target and support groups of countries.

While an unsupervised learning analysis is useful, it is also important as well to recognize the possible downsides of our cluster analysis. First, if the algorithm chosen fails to identify an inherent cluster, this can directly lead to neglect of a group of countries as the prevalent mental health disorder is not represented. Further, the group of countries may be lumped in with a more common prevalent disorder, which can push forward underfunding and lack of preventative measures for those select countries. Conversely, if the algorithm "splits" an inherent cluster with very similar mental health profiles, this can ruin the optimization goal for world health organizations as solutions between both clusters overlap, thus raising administrative and policy costs due to redundancy. This is due to the fact that the organization could have created a single unified strategy which solved both clusterings.

Together, we can use various clustering techniques to uncover patterns within mental health rates. First, dendrograms can help illustrate hierarchical relationships between nations, and allow researchers to observe how countries slowly come together to form clusters, or how certain countries diverge from main clusters due to a certain mental health disorder. Next, a harder partitioning method such as K-Means can be essential for creating actionable policy. Through direct assignment to a cluster, health organizations can create policies which translate to a direct cluster of countries, which can help reduce policy costs and create direct health initiatives. In addition, K-Nearest Neighbors (KNN) can help identify regions of prevalent mental health disorders, and identify possible noise/outliers of countries which share unique mental health profiles compared to the larger group of countries.

Through our unsupervised learning analysis, we hope to gain the following in each section:

Section 2: In Dataset Discussion we will describe the source of the mental health rates dataset, and justify our selection of variables.

Section 3: In Basic Dataset Cleaning and Exploration we will describe our dataset, observe missing values, outliers, and noise and then adjust the dataset to maintain integrity in our clustering results.

Section 4: In Basic Descriptive Analytics we will provide basic summary statistics of the dataset, and investigate pairwise relationships between mental health disorders.

Section 5: In Scaling Decisions we justify whether the mental health dataset is dominated by different mental health disorders, and implement proper scaling methods to prepare the data for unsupervised learning analysis.

Section 6: In Clusterability and Clustering Structure Questions, we create t-SNE plots to observe whether the dataset is clusterable, and infer basic underlying structure between clusters.

Section 7: In Clustering Algorithm Selection Motivation, we justify our two clustering algorithms and analyze whether our ideal dataset properties are met.

Section 8: In Clustering and Post-Cluster Analysis (1), We cluster the dataset using K-Means and view performance metrics and analyze the post-clustering performance and corroboration with our research goals.

Section 9: In Clustering and Post-Cluster Analysis (2), We cluster the dataset using Hierarchical Agglomerative Clustering with Complete Linkage and view performance metrics and analyze the post-clustering performance and corroboration with our research goals.

Section 10: In Discussion, we assess our clustering using metrics to compare and contrast our clustering algorithms and unsupervised learning results.

Section 11: In conclusion, we will summarize our key findings, and give recommendations garnered from our analysis and indicate steps for future implementation beyond the unsupervised learning analysis.
""")


# ── Section 2: Dataset Discussion ────────────────────────────────────────────

st.write("""
# 2. Dataset Discussion

The data for the analysis was retrieved via Kaggle from user Amit. In our analysis, we specifically utilized a subset of the "Uncover Global Trends in Mental Health Disorder" dataset which is restricted to the year 2017. While sourced through Kaggle, the original dataset Our World in Data is compiled from a separate source and associated blog. Links below have been provided to the Kaggle Source, Primary Source, and Amit's data.world profile.

The following links were accessed and the associated dataset was downloaded on May 12th, 2026.

Kaggle Source: https://www.kaggle.com/datasets/thedevastator/uncover-global-trends-in-mental-health-disorder/data

Primary Source: https://ourworldindata.org and Blog Post: https://ourworldindata.org/mental-health#all-charts-preview

Amit's data.hub profile: https://data.world/amitd?preview=vizzup%2Fmental-health-depression-disorder-data
""")

code = """import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

df = pd.read_csv('mental_health_countries.csv')
df.head()"""
st.code(code, language="python")

df = pd.read_csv('mental_health_countries.csv')
st.dataframe(df, use_container_width=True)

st.write("""
Before dataset cleaning and exploration, the dataset contains 195 rows and 9 columns. The dataset's observations according to Our World in Data are sourced from Hospitals across the globe, but may not include clinical trials. Each row or observation in the dataset describes a country, its region, and respective diagnosed mental health disorder rates in terms of the general population of each country.

In our analysis, we have already retrieved the dataset with the following selected variables: Country, Region, Schizophrenia (%),  Bipolar disorder (%), Eating disorders (%), Anxiety disorders (%), Drug use disorders (%), Depression (%), and Alcohol use disorders (%). Importantly, within the dataset sourced from kaggle and Our World in Data, the following differences have been observed. In our analysis, the column titled "index" which was present in the csv is not a named column and has been suppressed. Additionally, the original dataset contained additional columns which were removed as they were deemed unnecessary for the analysis. As per the Year Filter mentioned in Section 3.3, the year column was dropped as data was restricted to 2017.
""")


# ── Section 3: Dataset Cleaning and Exploration ──────────────────────────────

st.write("# 3. Dataset Cleaning and Exploration")

code = """import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import fcluster"""
st.code(code, language="python")

code = """df.columns"""
st.code(code, language="python")
st.write(df.columns.tolist())

code = """df_use = df[['Country',
             'Region',
             'Schizophrenia (%)',
             'Bipolar disorder (%)',
             'Eating disorders (%)',
             'Anxiety disorders (%)',
             'Drug use disorders (%)',
             'Depression (%)',
             'Alcohol use disorders (%)']].copy()

df_use.head()"""
st.code(code, language="python")

df_use = df[['Country',
             'Region',
             'Schizophrenia (%)',
             'Bipolar disorder (%)',
             'Eating disorders (%)',
             'Anxiety disorders (%)',
             'Drug use disorders (%)',
             'Depression (%)',
             'Alcohol use disorders (%)']].copy()
st.dataframe(df_use.head(), use_container_width=True)

code = """df_model = df_use[['Schizophrenia (%)',
                   'Bipolar disorder (%)',
                   'Eating disorders (%)',
                   'Anxiety disorders (%)',
                   'Drug use disorders (%)',
                   'Depression (%)',
                   'Alcohol use disorders (%)']].copy()

df_model.head()"""
st.code(code, language="python")

df_model = df_use[['Schizophrenia (%)',
                   'Bipolar disorder (%)',
                   'Eating disorders (%)',
                   'Anxiety disorders (%)',
                   'Drug use disorders (%)',
                   'Depression (%)',
                   'Alcohol use disorders (%)']].copy()
st.dataframe(df_model.head(), use_container_width=True)

code = """df_use.isnull().sum()"""
st.code(code, language="python")
st.write(df_use.isnull().sum())

st.write("Dataset contains no missing values, and requires no cleaning for missing values.")

code = """sns.pairplot(df_model)
plt.show()"""
st.code(code, language="python")

fig = sns.pairplot(df_model)
st.pyplot(fig)

st.write("For every pair of numerical explanatory variables, we observe potential outliers in the scatterplots. These points deviate away from the main cluster greatly, and don't attach separately to another cluster.")

st.write("KNN Distance Plot")

code = """for k in [2, 3, 4, 5]:
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs_fit = nbrs.fit(df_model)
    distances, indices = nbrs_fit.kneighbors(df_model)

    k_distances = np.sort(distances[:, k-1])

    plt.plot(k_distances)
    plt.title('%s-Nearest Neighbor Sorted Distance Plot' % (k))
    plt.xlabel('Points Sorted by Distance')
    plt.ylabel('%s-NN Distance' % (k))
    plt.show()"""
st.code(code, language="python")

for k in [2, 3, 4, 5]:
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs_fit = nbrs.fit(df_model)
    distances, indices = nbrs_fit.kneighbors(df_model)
    k_distances = np.sort(distances[:, k-1])
    fig, ax = plt.subplots()
    ax.plot(k_distances)
    ax.set_title('%s-Nearest Neighbor Sorted Distance Plot' % (k))
    ax.set_xlabel('Points Sorted by Distance')
    ax.set_ylabel('%s-NN Distance' % (k))
    st.pyplot(fig)
    plt.close()

st.write("Single Linkage Dendrogram")

code = """dm = pdist(df_model, metric='euclidean')
Z = linkage(dm, method='single')

plt.subplots(figsize=(20, 35))
dendrogram(Z, labels=df_use['Country'].values, orientation='right')
plt.title('Single Linkage Dendrogram')
plt.xlabel('Distance')
plt.ylabel('Country')
plt.show()"""
st.code(code, language="python")

dm = pdist(df_model, metric='euclidean')
Z = linkage(dm, method='single')
fig, ax = plt.subplots(figsize=(20, 35))
dendrogram(Z, labels=df_use['Country'].values, orientation='right', ax=ax)
ax.set_title('Single Linkage Dendrogram')
ax.set_xlabel('Distance')
ax.set_ylabel('Country')
st.pyplot(fig)
plt.close()

st.write("""
Through our single linkage dendrogram, we observe the presence of two outliers (New Zealand and Greenland), which join the main clustering much later than other countries. In addition, we observe a vertical spike after the 190th country on the x-axis, indicating the presence of outliers. We observe that most of the distance in KNN stays below 1, but spikes due to the presence of outliers which are far from any other data point.

### 3.1 Outlier Consideration

In the context of improving mental health awareness, we believe that it is best to keep the identified outliers and not drop them. While dropping the outlier can allow for a more general solution to approach mental health by focusing on the now more cohesive main clusters. The downside is that this country will be ignored in our analysis, when they might need to. Another benefit of dropping the outlier is that it could allow for our clustering metrics and cluster-sorted similarity matrix to be stronger. However, in the context of our research, we believe the downsides outweigh the benefits, and it's important to retain the country in our clustering analysis.

### 3.2 Noise Consideration and Identification
""")

code = """from sklearn.cluster import DBSCAN
X_numeric = df_model.copy()
X_numeric.std()

scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X_numeric)

for min_samples_val in [2, 3, 4]:
    silhouette_vals = []
    cluster_counts = []
    noise_counts = []

    for epsilon in np.arange(1, 6, 0.2):
        dbscan_model = DBSCAN(eps=epsilon, min_samples=min_samples_val, metric=\"euclidean\")
        df['cluster_label'] = dbscan_model.fit_predict(X_scaled_array)
        df_filtered = df[df['cluster_label'] != -1]
        cluster_total = len(df_filtered['cluster_label'].value_counts())
        cluster_counts.append(cluster_total)
        noise_total = len(df[df['cluster_label'] == -1])
        noise_counts.append(noise_total)
        if cluster_total > 1:
            silhouette_vals.append(silhouette_score(X_scaled_array[df['cluster_label'] != -1], df[df['cluster_label'] != -1]['cluster_label']))
        else:
            silhouette_vals.append(0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    epsilon_vals = np.arange(1, 6, 0.2)
    fig.suptitle('Min Samples = %s' % min_samples_val)
    axes[0].plot(epsilon_vals, silhouette_vals)
    axes[1].plot(epsilon_vals, noise_counts)
    axes[2].plot(epsilon_vals, cluster_counts)
    axes[0].set_title('Average Silhouette Score')
    axes[1].set_title('Number of Noise Points')
    axes[2].set_title('Number of Clusters')
    axes[0].set_xlabel('Epsilon')
    axes[1].set_xlabel('Epsilon')
    axes[2].set_xlabel('Epsilon')
    plt.show()"""
st.code(code, language="python")

X_numeric = df_model.copy()
scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X_numeric)

for min_samples_val in [2, 3, 4]:
    silhouette_vals = []
    cluster_counts = []
    noise_counts = []
    for epsilon in np.arange(1, 6, 0.2):
        dbscan_model = DBSCAN(eps=epsilon, min_samples=min_samples_val, metric="euclidean")
        df['cluster_label'] = dbscan_model.fit_predict(X_scaled_array)
        df_filtered = df[df['cluster_label'] != -1]
        cluster_total = len(df_filtered['cluster_label'].value_counts())
        cluster_counts.append(cluster_total)
        noise_total = len(df[df['cluster_label'] == -1])
        noise_counts.append(noise_total)
        if cluster_total > 1:
            silhouette_vals.append(silhouette_score(X_scaled_array[df['cluster_label'] != -1], df[df['cluster_label'] != -1]['cluster_label']))
        else:
            silhouette_vals.append(0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    epsilon_vals = np.arange(1, 6, 0.2)
    fig.suptitle('Min Samples = %s' % min_samples_val)
    axes[0].plot(epsilon_vals, silhouette_vals)
    axes[1].plot(epsilon_vals, noise_counts)
    axes[2].plot(epsilon_vals, cluster_counts)
    axes[0].set_title('Average Silhouette Score')
    axes[1].set_title('Number of Noise Points')
    axes[2].set_title('Number of Clusters')
    axes[0].set_xlabel('Epsilon')
    axes[1].set_xlabel('Epsilon')
    axes[2].set_xlabel('Epsilon')
    st.pyplot(fig)
    plt.close()

st.write("""
To determine noise, we first scaled the dataset then used DBSCAN with varying epsilon and minpts values and examined the number of clusters, noise points, and average silhouette score performance. Moreover, DBSCAN is effective as it does not force our countries into a cluster, but to defined as noise points based on their euclidean distance to other clusters.

Importantly, as epsilon increased, the number of noise points decreased and the number of clusters decreased. This indicates that DBSCAN is likely classifying our outlier countries (Greenland, New Zealand) as noise, but as epsilon increases and the tolerance for grouping countries increases, we can observe these countries fitting into a broader cluster definition.

### 3.3 Other Data Cleaning

Beyond basic data cleaning, the following steps as aforementioned were deemed suitable for our unsupervised learning analysis, and were conducted before importing the dataset:

- Year Filter: Dataset was restricted to the year 2017 to ensure comparisons between countries and regions instead of identifying trends across multiple years
- Entity Filtering: Cleared the dataset of non-national entities (i.e. World, continents, etc.)
- Column Dropping: Dropped Year column (not needed after filtering), and prevalence columns not used in analysis
""")


# ── Section 4: Basic Descriptive Analytics ───────────────────────────────────

st.write("""
# 4. Basic Descriptive Analytics

## 4.1 Boxplot Analysis
""")

code = """corr_mat = df_model.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_mat, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Mental Health Disorder Rates')
plt.show()"""
st.code(code, language="python")

corr_mat = df_model.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_mat, annot=True, cmap='coolwarm', center=0, ax=ax)
ax.set_title('Correlation Matrix of Mental Health Disorder Rates')
st.pyplot(fig)
plt.close()

st.write("""
Here, we observe that Bipolar Disorder and Eating Disorders, Schizophrenia and Eating Disorders, and Anxiety Disorders and Eating Disorders all have high positive relationships. On the other hand, no two variables have an extremely negative covariance relationship. Generally, we can observe various disorders are positively correlated with one another, meaning that the prevalence of mental health disorders may tend to co-occur across countries.

## 4.2 Summary Statistics Analysis
""")

code = """df_model.describe()"""
st.code(code, language="python")
st.dataframe(df_model.describe(), use_container_width=True)

code = """df_use['Region'].value_counts()"""
st.code(code, language="python")
st.write(df_use['Region'].value_counts())

st.write("We observe that the most common regions are Western Asia, Eastern Africa, and Western Africa. Importantly, these continents contain numerous countries, which may skew our data analysis as we didn't assign weights to adequately represent prevalence across all regions.")


# ── Section 5: Scaling Decisions ─────────────────────────────────────────────

st.write("# 5. Scaling Decisions")

code = """X_num = df_model.copy()
X_num.std()"""
st.code(code, language="python")

X_num = df_model.copy()
st.write(X_num.std())

code = """ss = StandardScaler()
X_stand_array = ss.fit_transform(X_num)
df_scaled = pd.DataFrame(X_stand_array, columns=X_num.columns)
df_scaled.describe()"""
st.code(code, language="python")

ss = StandardScaler()
X_stand_array = ss.fit_transform(X_num)
df_scaled = pd.DataFrame(X_stand_array, columns=X_num.columns)
st.dataframe(df_scaled.describe(), use_container_width=True)

st.write("""
Although all of the numerical variables in this dataset are measured as percentages, they do not have the same amount of variation, as shown by their different standard deviations. For example, schizophrenia has a much smaller spread than anxiety disorders, depression, and alcohol use disorders. If we did not scale the data, variables with larger variability would contribute more heavily to Euclidean distance calculations and thus dominate the clustering. By standardizing, we ensure that each disorder rate contributes equally to the distance metric.
""")


# ── Section 6: Clusterability and Clustering Structure ───────────────────────

st.write("# 6. Clusterability and Clustering Structure")

code = """df_scaled_labeled = pd.concat([df_use[['Country', 'Region']], df_scaled], axis=1)
df_scaled_labeled.head()"""
st.code(code, language="python")

df_scaled_labeled = pd.concat([df_use[['Country', 'Region']], df_scaled], axis=1)
st.dataframe(df_scaled_labeled.head(), use_container_width=True)

code = """for perp in [5, 10, 20, 30, 40, 50]:
    for rs in [77, 1000]:
        tsne = TSNE(n_components=2, perplexity=perp, random_state=rs)
        data_tsne = tsne.fit_transform(df_scaled)

        df_tsne = pd.DataFrame(data_tsne, columns=['x_projected', 'y_projected'])
        df_combo = pd.concat([df_scaled_labeled, df_tsne], axis=1)

        sns.scatterplot(x='x_projected', y='y_projected', data=df_combo)
        plt.title('t-SNE Plot with Perplexity Value %s and Random State %s' % (perp, rs))
        plt.show()"""
st.code(code, language="python")

for perp in [5, 10, 20, 30, 40, 50]:
    for rs in [77, 1000]:
        tsne = TSNE(n_components=2, perplexity=perp, random_state=rs)
        data_tsne = tsne.fit_transform(df_scaled)
        df_tsne = pd.DataFrame(data_tsne, columns=['x_projected', 'y_projected'])
        df_combo = pd.concat([df_scaled_labeled, df_tsne], axis=1)
        fig, ax = plt.subplots()
        sns.scatterplot(x='x_projected', y='y_projected', data=df_combo, ax=ax)
        ax.set_title('t-SNE Plot with Perplexity Value %s and Random State %s' % (perp, rs))
        st.pyplot(fig)
        plt.close()

code = """tsne = TSNE(n_components=2, perplexity=50, random_state=77)
data_tsne = tsne.fit_transform(df_scaled)

df_tsne = pd.DataFrame(data_tsne, columns=['x_projected', 'y_projected'])
df_combo = pd.concat([df_scaled_labeled, df_tsne], axis=1)

sns.scatterplot(x='x_projected', y='y_projected', data=df_combo)
plt.title('Representative t-SNE Plot: Perplexity = 50, Random State = 77')
plt.show()"""
st.code(code, language="python")

tsne = TSNE(n_components=2, perplexity=50, random_state=77)
data_tsne = tsne.fit_transform(df_scaled)
df_tsne = pd.DataFrame(data_tsne, columns=['x_projected', 'y_projected'])
df_combo = pd.concat([df_scaled_labeled, df_tsne], axis=1)
fig, ax = plt.subplots()
sns.scatterplot(x='x_projected', y='y_projected', data=df_combo, ax=ax)
ax.set_title('Representative t-SNE Plot: Perplexity = 50, Random State = 77')
st.pyplot(fig)
plt.close()

st.write("Across all of the t-SNE plots, the dataset appears to be clusterable. While the exact spacing between groups changes from plot to plot, the overall structure is fairly stable, which gives us confidence that the mental health dataset contains real underlying clustering structure. Because of this, we believe the data is clusterable. The suggested number of clusters shows 6-7 clusters within the t-SNE plots.")

code = """for var in ['Schizophrenia (%)',
            'Bipolar disorder (%)',
            'Eating disorders (%)',
            'Anxiety disorders (%)',
            'Drug use disorders (%)',
            'Depression (%)',
            'Alcohol use disorders (%)']:

    sns.scatterplot(x='x_projected',
                    y='y_projected',
                    hue=var,
                    data=df_combo,
                    palette='viridis')
    plt.title('t-SNE Plot Color-Coded by %s' % (var))
    plt.legend(bbox_to_anchor=(1,1))
    plt.show()"""
st.code(code, language="python")

for var in ['Schizophrenia (%)',
            'Bipolar disorder (%)',
            'Eating disorders (%)',
            'Anxiety disorders (%)',
            'Drug use disorders (%)',
            'Depression (%)',
            'Alcohol use disorders (%)']:
    fig, ax = plt.subplots()
    sns.scatterplot(x='x_projected', y='y_projected', hue=var, data=df_combo, palette='viridis', ax=ax)
    ax.set_title('t-SNE Plot Color-Coded by %s' % (var))
    ax.legend(bbox_to_anchor=(1, 1))
    st.pyplot(fig)
    plt.close()

st.write("When we examine the relationship between the attributes and the t-SNE-suggested structure, the results suggest that the clusters are (in most cases) associated with combinations of disorder rates rather than with only one single disorder. In particular, schizophrenia, bipolar disorder, eating disorders, anxiety disorders, and drug use disorders appear to contribute meaningfully to the separation of the clusters.")


# ── Section 7: Clustering Algorithm Selection Motivation ─────────────────────

st.write("""
# 7. Clustering Algorithm Selection Motivation

## 7.1 Algorithm #1: k-Means

We chose k-Means as our first clustering algorithm based on our previous analyses.

First, k-Means produces interpretable cluster centroids, which are useful for the kind of group-level summaries our motivation needs. Each cluster's centroid is a vector of seven disorder rates that represents the "typical" mental health profile of the countries in that cluster. So once the clustering is complete, we can directly compare cluster centroids to describe what makes each group of countries unique.

Second, after standardizing the data in Section 5, the mental health dataset is now approximately continuous and symmetric for several features. K-Means assumes approximately spherical clusters with similar spread, and while the t-SNE plots show some stretching, the broad structure appears round enough for k-Means to be a reasonable first pass.

Third, the t-SNE plot from Section 6 suggests about six to seven groups. K-Means can target a specific k, which makes it easy to test a small range of values (k=6 through k=9) and compare results using the elbow method, silhouette scores, and t-SNE corroboration.

## 7.2 Algorithm #2: Hierarchical Agglomerative Clustering (Complete Linkage)

For our second algorithm, we choose Hierarchical Agglomerative Clustering with Complete Linkage. Our t-SNE plot suggests the presence of slightly stretched clusters alongside clusters that contain very isolated points which are separated from bigger main clusters.

Together, the roughly spherical shape of the clusters in the t-SNE plot suggests that HAC with Complete Linkage is a strong candidate for our analysis. Complete Linkage merges clusters based on the maximum distance between any two points across two clusters. This leads to more compact and spherical clusters compared to single linkage, which tends to chain and Average Linkage which produces clusters of intermediate compactness. Together, this linkage strategy is stronger for detecting outliers, especially in our dataset which contains potential noise points.

Additionally, HAC is deterministic and does not require a random seed, which means results are reproducible across runs without specifying a random state. Furthermore, the dendrogram produced by HAC can be useful for directly observing the hierarchical relationship between countries and clusters, which can be useful for understanding how the countries are related to one another.
""")


# ── Section 8: Clustering Algorithm #1 ──────────────────────────────────────

st.write("""
# 8. Clustering Algorithm #1

### 8.1 Elbow Plot
""")

code = """cluster_num_list = range(1, 13)
inertia_list = []

for k in cluster_num_list:
    print('k = ' + str(k))
    kmeans = KMeans(n_clusters=k, random_state=100)
    kmeans.fit(df_scaled)
    inertia_list.append(kmeans.inertia_)

plt.plot(cluster_num_list, inertia_list)
plt.xlabel('Number of Clusters Requested in K-means')
plt.ylabel('Inertia')
plt.title('Elbow Method Results for Mental Health Data')
plt.show()"""
st.code(code, language="python")

cluster_num_list = range(1, 13)
inertia_list = []
for k in cluster_num_list:
    kmeans = KMeans(n_clusters=k, random_state=100)
    kmeans.fit(df_scaled)
    inertia_list.append(kmeans.inertia_)
fig, ax = plt.subplots()
ax.plot(cluster_num_list, inertia_list)
ax.set_xlabel('Number of Clusters Requested in K-means')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method Results for Mental Health Data')
st.pyplot(fig)
plt.close()

st.write("This elbow plot suggests that k-means is detecting clustering structure in the mental health dataset. Inertia drops sharply for the first few clusters, and then the rate of decrease begins to slow down. The bend in the curve appears around k=6 to k=7, which suggests that the data likely contains about six to seven meaningful clusters. Since the elbow is not very sharp, we cannot draw a strong conclusion from this plot alone.")

st.write("### 8.2 Average Silhouette Score for K-Means Algorithm")

code = """sil_scores = []

for k in range(2, 13):
    kmeans = KMeans(n_clusters=k, random_state=100)
    cluster_labels = kmeans.fit_predict(df_scaled)
    sil_scores.append(silhouette_score(df_scaled, cluster_labels))

plt.plot(range(2, 13), sil_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Average Silhouette Score')
plt.title('Average Silhouette Scores for k-Means')
plt.show()"""
st.code(code, language="python")

sil_scores = []
for k in range(2, 13):
    kmeans = KMeans(n_clusters=k, random_state=100)
    cluster_labels = kmeans.fit_predict(df_scaled)
    sil_scores.append(silhouette_score(df_scaled, cluster_labels))
fig, ax = plt.subplots()
ax.plot(range(2, 13), sil_scores)
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Average Silhouette Score')
ax.set_title('Average Silhouette Scores for k-Means')
st.pyplot(fig)
plt.close()

st.write("The average silhouette score plot shows that several values of k could work, with the strongest scores occurring in the range around k=6 through k=9. We do not want to use only the silhouette score data, but we should instead compare the silhouette scores with the elbow plot and the t-SNE plots. We see that the average silhouette score plot most strongly suggests about k=7 clusters.")

st.write("### 8.3 t-SNE Corroboration Plots")

code = """for k in [6, 7, 8, 9]:
    kmeans = KMeans(n_clusters=k, random_state=100)
    df_combo['predicted_cluster'] = kmeans.fit_predict(df_scaled)

    sns.scatterplot(x='x_projected', y='y_projected',
                    hue='predicted_cluster',
                    palette=sns.color_palette(\"husl\", k),
                    data=df_combo)
    plt.title('t-SNE Plot with k-Means Clustering with k=%s' % (k))
    plt.legend(bbox_to_anchor=(1,1))
    plt.show()"""
st.code(code, language="python")

for k in [6, 7, 8, 9]:
    kmeans = KMeans(n_clusters=k, random_state=100)
    df_combo['predicted_cluster'] = kmeans.fit_predict(df_scaled)
    fig, ax = plt.subplots()
    sns.scatterplot(x='x_projected', y='y_projected', hue='predicted_cluster', palette=sns.color_palette("husl", k), data=df_combo, ax=ax)
    ax.set_title('t-SNE Plot with k-Means Clustering with k=%s' % (k))
    ax.legend(bbox_to_anchor=(1, 1))
    st.pyplot(fig)
    plt.close()

st.write("To compare candidate k-means clusterings with the structure suggested by the t-SNE plots, we examined k=6,7,8, and 9. Among these, the clustering with k=7 had the strongest corroboration with the representative t-SNE plot. The k=7 solution aligns well with the visually separated groups, while smaller values like as k=6 appear to merge groups that look distinct and should not be joined.")

st.write("### 8.4 Cluster-Sorted Similarity Matrix for K-Means")

code = """dist_mat = pairwise_distances(df_scaled, metric='euclidean')

for k in [6,7,8,9]:
  kmeans = KMeans(n_clusters=k, random_state=100)
  cluster_labels = kmeans.fit_predict(df_scaled)

  sort_idx = np.argsort(cluster_labels)
  dist_mat_sorted = dist_mat[sort_idx][:, sort_idx]

  plt.figure(figsize=(8,8))
  sns.heatmap(dist_mat_sorted, cmap='viridis')
  plt.title('Cluster-Sorted Similarity Matrix for k-Means (k=%s)' % (k))
  plt.show()"""
st.code(code, language="python")

dist_mat = pairwise_distances(df_scaled, metric='euclidean')
for k in [6, 7, 8, 9]:
    kmeans = KMeans(n_clusters=k, random_state=100)
    cluster_labels = kmeans.fit_predict(df_scaled)
    sort_idx = np.argsort(cluster_labels)
    dist_mat_sorted = dist_mat[sort_idx][:, sort_idx]
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(dist_mat_sorted, cmap='viridis', ax=ax)
    ax.set_title('Cluster-Sorted Similarity Matrix for k-Means (k=%s)' % (k))
    st.pyplot(fig)
    plt.close()

st.write("The cluster-sorted similarity matrix for k=7 shows a fairly clear block structure along the diagonal, which suggests that the resulting clusters are cohesive and separated from one another. The block diagonal is not perfectly strong, which means that some clusters are closer to each other than others. However, the overall matrix still provides support for k=7.")

st.write("### 8.5 Boxplot and Region Comparison")

code = """k = 7
kmeans = KMeans(n_clusters=k, random_state=100)
df_combo['kmeans_cluster'] = kmeans.fit_predict(df_scaled)

df_combo[['Country', 'Region', 'kmeans_cluster']].head()"""
st.code(code, language="python")

k = 7
kmeans = KMeans(n_clusters=k, random_state=100)
df_combo['kmeans_cluster'] = kmeans.fit_predict(df_scaled)
st.dataframe(df_combo[['Country', 'Region', 'kmeans_cluster']].head(), use_container_width=True)

code = """for var in ['Schizophrenia (%)',
            'Bipolar disorder (%)',
            'Eating disorders (%)',
            'Anxiety disorders (%)',
            'Drug use disorders (%)',
            'Depression (%)',
            'Alcohol use disorders (%)']:
    sns.boxplot(x='kmeans_cluster', y=var, data=df_combo)
    plt.title('%s by k-Means Cluster' % (var))
    plt.show()"""
st.code(code, language="python")

for var in ['Schizophrenia (%)',
            'Bipolar disorder (%)',
            'Eating disorders (%)',
            'Anxiety disorders (%)',
            'Drug use disorders (%)',
            'Depression (%)',
            'Alcohol use disorders (%)']:
    fig, ax = plt.subplots()
    sns.boxplot(x='kmeans_cluster', y=var, data=df_combo, ax=ax)
    ax.set_title('%s by k-Means Cluster' % (var))
    st.pyplot(fig)
    plt.close()

st.write("""
The above boxplots are used to add more depth of understanding as to the composition of each cluster in context of the data.

### 8.6 Outlier Handling Check

Earlier sections suggested that the dataset contains outliers and some noise-like countries, including countries such as New Zealand and Greenland. But, in the context of our research goal, we decided not to drop these countries because they may represent unique mental health profiles that could still be relevant. While k-Means ideal data properties don't include outliers, the final clustering appears to place these countries into their own small cluster, which still allows us to identify their unique mental health profile.

### 8.7 Best Clustering Selection and Final Results Presentation for K-Means
""")

code = """k = 7
sns.scatterplot(x='x_projected', y='y_projected',
                hue='kmeans_cluster',
                palette=sns.color_palette(\"husl\", k),
                data=df_combo)
plt.title('Representative t-SNE Plot with Final k-Means Clustering (k=7)')
plt.legend(bbox_to_anchor=(1,1))
plt.show()"""
st.code(code, language="python")

k = 7
fig, ax = plt.subplots()
sns.scatterplot(x='x_projected', y='y_projected', hue='kmeans_cluster', palette=sns.color_palette("husl", k), data=df_combo, ax=ax)
ax.set_title('Representative t-SNE Plot with Final k-Means Clustering (k=7)')
ax.legend(bbox_to_anchor=(1, 1))
st.pyplot(fig)
plt.close()

st.write("""
Based on the elbow plot, silhouette scores, t-SNE corroboration, and the similarity matrix, the k-means clustering with k=7 shows the strongest evidence that we have identified the main inherent clusters in the dataset. This clustering is also the most useful for our research goal, since it gives each country a single, interpretable mental health profile group.

### 8.8 Technique Shortcomings

There are still some limitations to the k-Means tuning methods. First, the elbow plot does not show one obvious cutoff, so there is still some uncertainty between nearby values such as k=6 and k=8. Second, the clustering with the highest average silhouette score is not automatically the one that identifies all inherent clusters, since silhouette can sometimes favor more compact solutions that merge true groups. Third, k-Means is sensitive to outliers such as New Zealand and Greenland, which may be distorting the centroids of adjacent clusters.
""")


# ── Section 9: Clustering Algorithm #2 ──────────────────────────────────────

st.write("""
# 9. Clustering Algorithm #2

Although we decide on HAC Complete Linkage for algorithm 2, we include all other types of HAC (excluding Single Linkage due to poor feature fit) for pairwise analysis.

### 9.1 Silhouette Score for HAC algorithm type (Complete, Average, Ward)
""")

code = """for i in ['complete', 'average', 'ward']:
    Z = linkage(df_scaled, method= i)
    sil_scores = []
    for k in range(2, 13):
        labels = fcluster(Z, t=k, criterion='maxclust')
        sil_scores.append(silhouette_score(df_scaled, labels))
    plt.figure(figsize=(6,4))
    plt.plot(range(2, 13), sil_scores)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Silhouette Score')
    plt.title(f'Silhouette Scores for {i} Linkage')
    plt.grid(True)
    plt.show()"""
st.code(code, language="python")

for i in ['complete', 'average', 'ward']:
    Z = linkage(df_scaled, method=i)
    sil_scores_hac = []
    for k in range(2, 13):
        labels = fcluster(Z, t=k, criterion='maxclust')
        sil_scores_hac.append(silhouette_score(df_scaled, labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(2, 13), sil_scores_hac)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Average Silhouette Score')
    ax.set_title(f'Silhouette Scores for {i} Linkage')
    ax.grid(True)
    st.pyplot(fig)
    plt.close()

st.write("""
For Complete and Average Linkage functions, we observe the highest average silhouette score when k=2. Our highest average silhouette scorings for both average and complete occur early, while our highest average silhouette scoring for Ward's Linkage appears at k=10. This does not show strong corroboration with our suggested number of clusters, given from our t-SNE analysis in Section 3.

### 9.2 Test for Natural Number of Clusters: Merge Distance vs. k

For each linkage method, we plot the merge distance required to obtain k clusters. Sharp drops between adjacent k values correspond to natural cut points in the dendrogram.
""")

code = """for i in [\"complete\", \"average\", \"ward\"]:
    Z = linkage(df_scaled, method= i)
    heights = [Z[-(k - 1), 2] for k in range(2, 13)]
    plt.plot(range(2, 13), heights, \"o-\", label= i)
plt.gca().invert_xaxis()
plt.legend()
plt.show()"""
st.code(code, language="python")

fig, ax = plt.subplots()
for i in ["complete", "average", "ward"]:
    Z = linkage(df_scaled, method=i)
    heights = [Z[-(k - 1), 2] for k in range(2, 13)]
    ax.plot(range(2, 13), heights, "o-", label=i)
ax.invert_xaxis()
ax.legend()
st.pyplot(fig)
plt.close()

st.write("Complete linkage is nearly flat. Every merge costs about the same from k=12 down to k=4, with slightly elevated tilt k = 2 to 4. Together, this implies that the elbow plot doesn't contain a strong indicator for a clear ideal number of clusters. However, we can observe a higher silhouette plot plateau at k=6,7,8.")

code = """for i in [\"complete\", \"average\", \"ward\"]:
  dm = pdist(df_scaled, metric='euclidean')
  Z = linkage(dm, method= i)

  plt.subplots(figsize=(20, 35))
  dendrogram(Z, labels=df_use['Country'].values, orientation='right', color_threshold=5)
  plt.title(f'{i} Linkage Dendrogram')
  plt.xlabel('Distance')
  plt.ylabel('Country')
  plt.show()"""
st.code(code, language="python")

for i in ["complete", "average", "ward"]:
    dm = pdist(df_scaled, metric='euclidean')
    Z = linkage(dm, method=i)
    fig, ax = plt.subplots(figsize=(20, 35))
    dendrogram(Z, labels=df_use['Country'].values, orientation='right', color_threshold=5, ax=ax)
    ax.set_title(f'{i} Linkage Dendrogram')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Country')
    st.pyplot(fig)
    plt.close()

st.write("The Complete Linkage dendrogram at distance 5.9 identifies seven colored sub-trees, highlighting a significant gap between 6.4 and 9.6 that clearly separates outliers like New Zealand and Greenland from the main data body.")

st.write("### 9.3 t-SNE for Inherent Clusters\n\nWe plot the representative t-SNE plot from Section 6 by complete-linkage labels at several candidate k values.")

code = """Z_complete = linkage(df_scaled, method= 'complete')
for k in [4, 5, 6, 7, 8, 9]:
    labels = fcluster(Z_complete, t=k, criterion='maxclust')
    df_combo['predicted_cluster'] = labels
    sns.scatterplot(x='x_projected', y='y_projected',
                    hue='predicted_cluster',
                    palette=sns.color_palette(\"husl\", k),
                    data=df_combo)
    plt.title('t-SNE Plot with Complete Linkage with k=%s' % (k))
    plt.legend(bbox_to_anchor=(1,1))
    plt.show()"""
st.code(code, language="python")

Z_complete = linkage(df_scaled, method='complete')
for k in [4, 5, 6, 7, 8, 9]:
    labels = fcluster(Z_complete, t=k, criterion='maxclust')
    df_combo['predicted_cluster'] = labels
    fig, ax = plt.subplots()
    sns.scatterplot(x='x_projected', y='y_projected', hue='predicted_cluster', palette=sns.color_palette("husl", k), data=df_combo, ax=ax)
    ax.set_title('t-SNE Plot with Complete Linkage with k=%s' % (k))
    ax.legend(bbox_to_anchor=(1, 1))
    st.pyplot(fig)
    plt.close()

st.write("""
### 9.4 Cluster-Sorted Similarity Matrices

We compute Euclidean distance matrices sorted by complete-linkage labels at several candidate k values. Once again, a well-separated, well-cohesive clustering produces a block-diagonal pattern: dark squares on the diagonal (low intra-cluster distances) and lighter regions off-diagonal (higher inter-cluster distances) in an ideal clustering.
""")

code = """dist_mat = pairwise_distances(df_scaled, metric='euclidean')

for k in [5,6,7,8]:
    Z = linkage(df_scaled, method= 'complete')
    labels = fcluster(Z, t=k, criterion='maxclust')
    sort_idx = np.argsort(labels)
    dist_sorted = dist_mat[sort_idx][:, sort_idx]
    plt.figure(figsize=(6, 6))
    sns.heatmap(dist_sorted, cmap='viridis', cbar_kws={'label': 'Euclidean distance'})
    plt.title(f'Cluster-sorted similarity matrix Complete Linkage, k = {k}')
    plt.xlabel('Country (sorted by cluster)')
    plt.ylabel('Country (sorted by cluster)')
    plt.show()"""
st.code(code, language="python")

dist_mat = pairwise_distances(df_scaled, metric='euclidean')
for k in [5, 6, 7, 8]:
    Z = linkage(df_scaled, method='complete')
    labels = fcluster(Z, t=k, criterion='maxclust')
    sort_idx = np.argsort(labels)
    dist_sorted = dist_mat[sort_idx][:, sort_idx]
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(dist_sorted, cmap='viridis', cbar_kws={'label': 'Euclidean distance'}, ax=ax)
    ax.set_title(f'Cluster-sorted similarity matrix Complete Linkage, k = {k}')
    ax.set_xlabel('Country (sorted by cluster)')
    ax.set_ylabel('Country (sorted by cluster)')
    st.pyplot(fig)
    plt.close()

st.write("""
At k=5, we observe that the bottom right quadrant is one large sparse block, this larger generalization of the data points may be obscuring clustering substructure. At k=7, the bottom right quadrant resolves into two visible blocks, with darker coloring indicating stronger separation.

## 9.5 Outlier Handling Check

In 3.1 we committed to retaining New Zealand and Greenland as cluster members rather than dropping them or isolating them as singletons. We verify here that the chosen clustering preserves this behavior.
""")

code = """Z_best = linkage(df_scaled, method= 'complete')
labels = fcluster(Z_best, t= 7 , criterion='maxclust')
df_check = df_use[['Country', 'Region']].copy()
df_check['hier_cluster'] = labels
outliers = ['New Zealand', 'Greenland']
print(\"Outlier assignments under the chosen clustering:\")
print(df_check[df_check['Country'].isin(outliers)])
print(\"Size of each cluster:\")
print(df_check['hier_cluster'].value_counts().sort_index())"""
st.code(code, language="python")

Z_best = linkage(df_scaled, method='complete')
labels = fcluster(Z_best, t=7, criterion='maxclust')
df_check = df_use[['Country', 'Region']].copy()
df_check['hier_cluster'] = labels
outliers = ['New Zealand', 'Greenland']
st.write("Outlier assignments under the chosen clustering:")
st.dataframe(df_check[df_check['Country'].isin(outliers)], use_container_width=True)
st.write("Size of each cluster:")
st.write(df_check['hier_cluster'].value_counts().sort_index())

st.write("""
The outliers are not in singleton clusters, but they're in very small clusters. New Zealand sits in a cluster of 2 (cluster 1), and Greenland sits in a cluster of 3 (cluster 2). Both clusters together account for just 5 countries, which aligns with our suggested clustering in Section 3 using Single Linkage.

## 9.6 Best Clustering Selection and Final Results Presentation

Now, we refit with the selected parameters and display the representative t-SNE color-coded by the Complete Linkage labels for 7 clusters.
""")

code = """Z_final = linkage(df_scaled, method= 'complete')
labels = fcluster(Z_final, t= 7, criterion='maxclust')
df_combo['hier_cluster'] = labels
plt.figure(figsize=(8, 6))
sns.scatterplot(x='x_projected', y='y_projected',
                hue='hier_cluster',
                palette=sns.color_palette(\"husl\", 7),
                data=df_combo, s=50)
plt.title(f'Representative t-SNE color-coded by k=7 Complete Linkage Clustering')
plt.legend(bbox_to_anchor=(1, 1))
plt.show()"""
st.code(code, language="python")

Z_final = linkage(df_scaled, method='complete')
labels = fcluster(Z_final, t=7, criterion='maxclust')
df_combo['hier_cluster'] = labels
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='x_projected', y='y_projected', hue='hier_cluster', palette=sns.color_palette("husl", 7), data=df_combo, s=50, ax=ax)
ax.set_title('Representative t-SNE color-coded by k=7 Complete Linkage Clustering')
ax.legend(bbox_to_anchor=(1, 1))
st.pyplot(fig)
plt.close()

st.write("""
Based on the elbow plot, silhouette scores, t-SNE corroboration, and the similarity matrix, the Complete Linkage clustering with k=7 shows the strongest evidence that we have identified the main inherent clusters in the dataset.

## 9.7 Technique Shortcomings

Unfortunately, our clustering algorithm had poor corroboration between determining the number of clusters. Our highest average silhouette plots suggested a far smaller number of clusters than our dendrogram analysis. However, it's important to recognize that the highest average silhouette score does not always indicate the method that identifies the inherent clusters. Also, our cluster sorted matrix does not show strong block structure, indicating that the clusters may not be well separated.
""")


# ── Section 10: Discussion ───────────────────────────────────────────────────

st.write("""
# 10. Discussion

## 10.1 Clustering Comparison
""")

code = """ari = adjusted_rand_score(df_combo['kmeans_cluster'], df_combo['hier_cluster'])
print(\"Adjusted Rand Score between k-Means and HAC Complete Linkage (both k=7)\")
crosstab = pd.crosstab(df_combo['kmeans_cluster'], df_combo['hier_cluster'], margins=True, margins_name='Total')
print(\"\\nCross-tabulation (rows = k-Means clusters, cols = HAC clusters):\")
print(crosstab)
print(\"-----------------------\")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.scatterplot(x='x_projected', y='y_projected', hue='kmeans_cluster', palette=sns.color_palette(\"husl\", 7), data=df_combo, ax=axes[0])
axes[0].set_title('k-Means clusters (k=7)')
axes[0].legend(bbox_to_anchor=(1, 1), fontsize=8)
sns.scatterplot(x='x_projected', y='y_projected', hue='hier_cluster', palette=sns.color_palette(\"husl\", 7), data=df_combo, ax=axes[1])
axes[1].set_title('HAC Complete Linkage clusters (k=7)')
axes[1].legend(bbox_to_anchor=(1, 1), fontsize=8)
plt.tight_layout()
plt.show()"""
st.code(code, language="python")

ari = adjusted_rand_score(df_combo['kmeans_cluster'], df_combo['hier_cluster'])
st.write("Adjusted Rand Score between k-Means and HAC Complete Linkage (both k=7):", ari)
crosstab = pd.crosstab(df_combo['kmeans_cluster'], df_combo['hier_cluster'], margins=True, margins_name='Total')
st.write("Cross-tabulation (rows = k-Means clusters, cols = HAC clusters):")
st.dataframe(crosstab, use_container_width=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.scatterplot(x='x_projected', y='y_projected', hue='kmeans_cluster', palette=sns.color_palette("husl", 7), data=df_combo, ax=axes[0])
axes[0].set_title('k-Means clusters (k=7)')
axes[0].legend(bbox_to_anchor=(1, 1), fontsize=8)
sns.scatterplot(x='x_projected', y='y_projected', hue='hier_cluster', palette=sns.color_palette("husl", 7), data=df_combo, ax=axes[1])
axes[1].set_title('HAC Complete Linkage clusters (k=7)')
axes[1].legend(bbox_to_anchor=(1, 1), fontsize=8)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.write("""
The ARI between k-Means and HAC at k=7 is ~0.337, which indicates slight agreement between both clustering algorithms. This indication may imply that there is some general agreement on data shape, but plentiful disagreement on cluster assortments.

## 10.2 Best Clustering Overall

Across both algorithms, k-Means with k=7 provides the strongest evidence we've identified the inherent clusters, with one important caveat about outlier handling. Evidence of corroboration with the t-SNE plot is more visible, k-Means' cluster labels align with the t-SNE data structures more closely compared to HAC's.

## 10.3 Different Insights From Each Algorithm

For our analysis, k-Means provides valuable insights regarding the larger cloud of data observed in our t-SNE plot. Specifically, the centered cloud of data is labeled into separate clusters, which can provide valuable information regarding each group of countries. Compared to HAC with Complete Linkage, we can analyze differences between the centered graph clusters and produce solutions based on the cluster profiles. On the other hand, HAC Complete Linkage identifies the outliers more distinctly, which can allow for an analysis of specific countries with unique mental health profiles.

## 10.4 Separation and Sparsity (using k-Means similarity matrix)
""")

code = """dist_mat = pairwise_distances(df_scaled, metric='euclidean')
labels = df_combo['kmeans_cluster'].values
sort_idx = np.argsort(labels)
dist_sorted = dist_mat[sort_idx][:, sort_idx]
plt.figure(figsize=(7, 7))
sns.heatmap(dist_sorted, cmap='viridis', cbar_kws={'label': 'Euclidean distance'})
plt.title('Cluster-sorted similarity matrix — k-Means k=7 (final)')
plt.xlabel('Country (sorted by cluster)')
plt.ylabel('Country (sorted by cluster)')
plt.show()"""
st.code(code, language="python")

dist_mat = pairwise_distances(df_scaled, metric='euclidean')
labels = df_combo['kmeans_cluster'].values
sort_idx = np.argsort(labels)
dist_sorted = dist_mat[sort_idx][:, sort_idx]
fig, ax = plt.subplots(figsize=(7, 7))
sns.heatmap(dist_sorted, cmap='viridis', cbar_kws={'label': 'Euclidean distance'}, ax=ax)
ax.set_title('Cluster-sorted similarity matrix — k-Means k=7 (final)')
ax.set_xlabel('Country (sorted by cluster)')
ax.set_ylabel('Country (sorted by cluster)')
st.pyplot(fig)
plt.close()

code = """Z = linkage(df_scaled, method= 'complete')
labels = fcluster(Z, t= 7, criterion='maxclust')
sort_idx = np.argsort(labels)
dist_sorted = dist_mat[sort_idx][:, sort_idx]
plt.figure(figsize=(6, 6))
sns.heatmap(dist_sorted, cmap='viridis', cbar_kws={'label': 'Euclidean distance'})
plt.title(f'Cluster-sorted similarity matrix Complete Linkage, k = 7')
plt.xlabel('Country (sorted by cluster)')
plt.ylabel('Country (sorted by cluster)')
plt.show()"""
st.code(code, language="python")

Z = linkage(df_scaled, method='complete')
labels = fcluster(Z, t=7, criterion='maxclust')
sort_idx = np.argsort(labels)
dist_sorted = dist_mat[sort_idx][:, sort_idx]
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(dist_sorted, cmap='viridis', cbar_kws={'label': 'Euclidean distance'}, ax=ax)
ax.set_title('Cluster-sorted similarity matrix Complete Linkage, k = 7')
ax.set_xlabel('Country (sorted by cluster)')
ax.set_ylabel('Country (sorted by cluster)')
st.pyplot(fig)
plt.close()

st.write("""
For our k-Means similarity matrix, we observe several distinct diagonal blocks representing clusters. These clusters are mostly dark blue representing strong inter-cluster distances. However, in the top left quadrant representing the large cloud of data, we observe that the intra clusters are quite blue colored, indicating that the intra-cluster distance is high.

## 10.5 Attribute Descriptions — Profile of Each Cluster
""")

code = """disorder_vars = ['Schizophrenia (%)', 'Bipolar disorder (%)', 'Eating disorders (%)',
                 'Anxiety disorders (%)', 'Drug use disorders (%)',
                 'Depression (%)', 'Alcohol use disorders (%)']
plt.figure(figsize=(10, 4))
profile = df_combo.groupby('kmeans_cluster')[disorder_vars].mean()
sns.heatmap(profile, annot=True)
plt.title('Mean disorder rates per cluster:')
plt.show()

plt.figure(figsize=(10, 4))
region_ct = pd.crosstab(df_combo['kmeans_cluster'], df_combo['Region'])
region_ct.plot(kind='bar', ax=plt.gca())
plt.title('Region composition of each k-Means clusters')
plt.xlabel('k-Means linkage cluster')
plt.ylabel('Number of countries')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for i, var in enumerate(disorder_vars):
    sns.boxplot(x='kmeans_cluster', y=var, data=df_combo, ax=axes[i], palette='Set2')
    axes[i].set_title(var, fontsize=12)
    axes[i].set_xlabel('k-Means Cluster')
    axes[i].set_ylabel(f'{var} Scaled Mean Values')
axes[-1].set_visible(False)
plt.tight_layout()
plt.show()"""
st.code(code, language="python")

disorder_vars = ['Schizophrenia (%)', 'Bipolar disorder (%)', 'Eating disorders (%)',
                 'Anxiety disorders (%)', 'Drug use disorders (%)',
                 'Depression (%)', 'Alcohol use disorders (%)']

fig, ax = plt.subplots(figsize=(10, 4))
profile = df_combo.groupby('kmeans_cluster')[disorder_vars].mean()
sns.heatmap(profile, annot=True, ax=ax)
ax.set_title('Mean disorder rates per cluster:')
st.pyplot(fig)
plt.close()

fig, ax = plt.subplots(figsize=(10, 4))
region_ct = pd.crosstab(df_combo['kmeans_cluster'], df_combo['Region'])
region_ct.plot(kind='bar', ax=ax)
ax.set_title('Region composition of each k-Means clusters')
ax.set_xlabel('k-Means linkage cluster')
ax.set_ylabel('Number of countries')
ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()
st.pyplot(fig)
plt.close()

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for i, var in enumerate(disorder_vars):
    sns.boxplot(x='kmeans_cluster', y=var, data=df_combo, ax=axes[i], palette='Set2')
    axes[i].set_title(var, fontsize=12)
    axes[i].set_xlabel('k-Means Cluster')
    axes[i].set_ylabel(f'{var} Scaled Mean Values')
axes[-1].set_visible(False)
plt.tight_layout()
st.pyplot(fig)
plt.close()

code = """Z_complete = linkage(df_scaled, method= 'complete')
labels = fcluster(Z_complete, t=7, criterion='maxclust')
df_combo['predicted_cluster'] = labels
plt.figure(figsize=(10, 4))
profile = df_combo.groupby('predicted_cluster')[disorder_vars].mean()
sns.heatmap(profile, annot=True)
plt.title('Mean disorder rates per cluster:')
plt.show()

plt.figure(figsize=(10, 4))
region_ct = pd.crosstab(df_combo['predicted_cluster'], df_combo['Region'])
region_ct.plot(kind='bar', ax=plt.gca())
plt.title('Region composition of each HAC complete linkage cluster')
plt.xlabel('HAC complete linkage cluster')
plt.ylabel('Number of countries')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for i, var in enumerate(disorder_vars):
    sns.boxplot(x='predicted_cluster', y=var, data=df_combo, ax=axes[i], palette='Set2')
    axes[i].set_title(var, fontsize=12)
    axes[i].set_xlabel('HAC Cluster')
    axes[i].set_ylabel(f'{var} Scaled Mean Values')
axes[-1].set_visible(False)
plt.tight_layout()
plt.show()"""
st.code(code, language="python")

Z_complete = linkage(df_scaled, method='complete')
labels = fcluster(Z_complete, t=7, criterion='maxclust')
df_combo['predicted_cluster'] = labels

fig, ax = plt.subplots(figsize=(10, 4))
profile = df_combo.groupby('predicted_cluster')[disorder_vars].mean()
sns.heatmap(profile, annot=True, ax=ax)
ax.set_title('Mean disorder rates per cluster:')
st.pyplot(fig)
plt.close()

fig, ax = plt.subplots(figsize=(10, 4))
region_ct = pd.crosstab(df_combo['predicted_cluster'], df_combo['Region'])
region_ct.plot(kind='bar', ax=ax)
ax.set_title('Region composition of each HAC complete linkage cluster')
ax.set_xlabel('HAC complete linkage cluster')
ax.set_ylabel('Number of countries')
ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()
st.pyplot(fig)
plt.close()

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for i, var in enumerate(disorder_vars):
    sns.boxplot(x='predicted_cluster', y=var, data=df_combo, ax=axes[i], palette='Set2')
    axes[i].set_title(var, fontsize=12)
    axes[i].set_xlabel('HAC Cluster')
    axes[i].set_ylabel(f'{var} Scaled Mean Values')
axes[-1].set_visible(False)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.write("""
For k-Means, we observe on the heatmap that Clusters 6 and 3 represent groups with higher overall rates across conditions, with Cluster 6 showing peaks in schizophrenia, eating disorders, anxiety, and drug use mental health disorders. Conversely, Cluster 2 captures a group with relatively low rates across all mental health disorders. Cluster 4 contains a semi-high alcohol use disorder rate, and Cluster 5 for drug use disorders.

For HAC Complete Linkage, Clusters 1 and 2 represent groups with a high overall mean disorder rates, with Cluster 1 showing exceptionally high rates across almost all categories except alcohol use. Cluster 2 peaks sharply in drug use disorders. Conversely, Cluster 4 represents a group with very low disorder rates across every single disorder, while Cluster 3 isolates individuals with extremely high alcohol use disorders.

In our bar plots, we observe the frequency of regions across the various clustering labels. According to our plot, we don't observe a clear relationship between region and mental health disorder profiles.
""")


# ── Section 11: Conclusion ───────────────────────────────────────────────────

st.write("""
# 11. Conclusion

In our analysis, we discovered several associations between clusters of countries and associated mental health disorders. Based on our analysis, it is beneficial to create broad mental health solutions based on our unsupervised learning analysis. Through k-Means, we observed clustering of the larger group of countries, with some clusters sporting higher than average mean values for certain mental health disorders such as schizophrenia. This finding can be useful for health organizations, who for example may construct solutions that focus slightly more on targeting schizophrenia among other disorders. Moreover, for countries that show low rates across all boxplots, it may be beneficial to adapt the countries approach towards mental health into their own solutions. On the other hand, countries which hope to target unique mental health profiles may observe our clustering labels for HAC Complete Linkage, and observe countries related to our outliers.

While our unsupervised analysis was successful, we did encounter several shortcomings during our analysis. Foremost, the presence of outliers in the dataset worsened the performance of our clustering algorithms, and may have obscured true relationships in the data. Also, our clusters themselves didn't have exceptionally strong associations between clusters, or agreement amongst algorithms. Our Adjusted Rand Score was low, and our mental health profile heatmaps in Section 10 didn't indicate clear patterns amongst our cluster labels.

In the future, it may be beneficial to minimize the number of attributes, and ensure a hard partitioning to create more general solutions for the world health organizations research goals. Additionally, it's possible that more recent data may indicate different trends than our current analysis using data from 2017.


# References

Kaggle Dataset:
The Devastator. (n.d.). Uncover global trends in mental health disorder [Dataset]. Kaggle. https://www.kaggle.com/datasets/thedevastator/uncover-global-trends-in-mental-health-disorder/data

Primary Source & Blog:
Saloni Dattani, Lucas Rodés-Guirao, Hannah Ritchie, and Max Roser (2023) - "Mental Health" Published online at OurWorldinData.org. Retrieved from: https://ourworldindata.org/mental-health and https://ourworldindata.org/mental-health#all-charts-preview

Amit's data.hub profile:
Amit. (n.d.). Mental health depression disorder data [Data set]. data.world. https://data.world/amitd?preview=vizzup%2Fmental-health-depression-disorder-data

Cosgrove, L. (2025). Addressing the global mental health crisis: How a human rights approach can help end the search for pharmaceutical magic bullets. Health and Human Rights Journal, 27(1). https://pmc.ncbi.nlm.nih.gov/articles/PMC12799051/
""")
=======
import pandas as pd
df = pd.read_csv("mental_health_countries.csv")

st.write("""
# My first app
Hello *world!*
""")

#Testing if data set works
st.dataframe(df, use_container_width=True)
>>>>>>> 5c9ff4f46d10e9ca5b42b85a317172ff2efb9c6f
