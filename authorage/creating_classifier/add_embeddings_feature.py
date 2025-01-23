###
from scipy.spatial.distance import cosine

def get_embeddings(titles_list):
    if isinstance(titles_list, str):
        return model.encode(titles_list)
    else:
        return model.encode('')

def get_cosine_dist(embedding_1, embedding_2):
    """
    This function will get the indiviual cosine distance for a particular line to the avg overall cosine distances for the fan fic and the normative corporea
    :return:
    """
    cosine_dist = cosine(embedding_1, embedding_2)
    return cosine_dist


# set up the model for creating embeddings
from sentence_transformers import SentenceTransformer

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')
###

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('result_df_replicated2.csv')

    df['VIAF_embeddings'] = df['record_enumerated_titles'].apply(get_embeddings)
    df['S2_embeddings'] = df['title_list'].apply(get_embeddings)
    df['cos_sim'] =  df.apply(lambda row:(get_cosine_dist(row['VIAF_embeddings'], row['S2_embeddings'])), axis=1)
