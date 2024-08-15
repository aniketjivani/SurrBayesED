# Visualize results of literature on Bayesian optimization, active learning, optimal experimental design, metamodels.

# optionally create network representation of keywords depending on their connections

# %%
load_pickle = True

from scholarly import scholarly

def query_google_scholar(keywords):
    search_query = scholarly.search_pubs(keywords)
    results = []
    for i in range(100):  # Adjust the range as needed, but be cautious of rate limits
        try:
            result = next(search_query)
            results.append(result)
            print("Result number {} is added.".format(i))
        except StopIteration:
            break
    return results

keywords = ["Bayesian optimization", "active learning", "optimal experimental design", "metamodels", "adaptive data collection", "simulation based inference"]


if not load_pickle:
    all_results = []
    for k in keywords:
        scholar_results = query_google_scholar(k)
        all_results.append(scholar_results)
        print("Number of results for keyword {}: {}".format(k, len(scholar_results)))
else:
    print("Load data from pickle.")

# %% save to / load from pickle

# import pickle

# with open("/Users/ajivani/Desktop/Research/SurrBayesED/misc/scholar_results.pkl", "wb") as f:
#     pickle.dump(all_results, f)

import pickle

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
