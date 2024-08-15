# Visualize results of literature on Bayesian optimization, active learning, optimal experimental design, metamodels.

# optionally create network representation of keywords depending on their connections

# %%
load_pickle = False

from scholarly import scholarly

def query_google_scholar(keywords):
    search_query = scholarly.search_pubs(keywords)
    results = []
    for i in range(75):  # Adjust the range as needed, but be cautious of rate limits
        try:
            result = next(search_query)
            results.append(result)
            print("Result number {} is added.".format(i))
        except StopIteration:
            break
    return results

keywords = ["Bayesian optimization", "active learning", "optimal experimental design", "metamodels", "adaptive data collection", "simulation based inference", "model-free strategies", "bayesian experimental design"]

import pickle

if not load_pickle:
    all_results = []
    for k in keywords:
        scholar_results = query_google_scholar(k)
        all_results.append(scholar_results)
        print("Number of results for keyword {}: {}".format(k, len(scholar_results)))

    with open("/Users/ajivani/Desktop/Research/SurrBayesED/misc/scholar_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

else:
    print("Load data from pickle.")
    with open("/Users/ajivani/Desktop/Research/SurrBayesED/misc/scholar_results.pkl", "rb") as f:
        all_results = pickle.load(f)



# %%
def extract_abstracts(results):
    abstracts = []
    for result in results:
        if 'abstract' in result['bib']:
            abstracts.append(result['bib']['abstract'])
    return abstracts

all_abstracts = []
for i, results in enumerate(all_results):
    abstracts = extract_abstracts(results)
    all_abstracts.append(abstracts)
    print("Number of abstracts for keyword {}: {}".format(keywords[i], len(abstracts)))

# %% Python wordcloud library

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


abstract_list = []
for aa in all_abstracts:
    for a in aa:
        abstract_list.append(a)

combined_text = " ".join(abstract_list)
create_wordcloud(combined_text)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def preprocess_and_extract_keywords(docs):
    # vectorizer = TfidfVectorizer(stop_words='english', max_features=100)  # Limit to top 50 features for simplicity
    vectorizer = CountVectorizer(stop_words='english', max_features=100)
    X = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()
    return X, feature_names

X, feature_names = preprocess_and_extract_keywords(abstract_list)
print("Extracted keywords: ", feature_names)

# %%
import numpy as np

def create_cooccurrence_matrix(X):
    Xc = (X.T * X)  # This is the co-occurrence matrix in CSR format
    Xc.setdiag(0)  # We don't need self-cooccurrence
    return Xc

co_occurrence_matrix = create_cooccurrence_matrix(X)
# %%
import networkx as nx
import matplotlib.pyplot as plt

def draw_network(co_occurrence_matrix, terms):
    G = nx.Graph()

    # Adding nodes
    for i in range(len(terms)):
        G.add_node(terms[i])

    # Adding edges
    co_occurrence_matrix = co_occurrence_matrix.toarray()
    for i in range(len(terms)):
        for j in range(i+1, len(terms)):
            if co_occurrence_matrix[i, j] > 0:
                G.add_edge(terms[i], terms[j], weight=co_occurrence_matrix[i, j])

    pos = nx.spring_layout(G, k=1.5)  # positions for all nodes
    plt.figure(figsize=(12, 12))

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='#66b3ff')
    # edges
    nx.draw_networkx_edges(G, pos, edgelist=G.edges, width=1, edge_color='gray', alpha=0.2)
    # labels
    nx.draw_networkx_labels(G, pos, font_size=15)

    plt.title('Co-occurrence Network of Terms')
    plt.axis('off')
    plt.show()

draw_network(co_occurrence_matrix, feature_names)
# %%
