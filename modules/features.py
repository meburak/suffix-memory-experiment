#features.py
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine


similarity_f = []
embeddings_f = []

# <NOTE> For now, all functions follow sim_df, embedding_df order. 
# I will make this more modular and robust. 


def set_embedding_space(similarity, embeddings):
    """
    use for setting global variables of similarity and embeddings
    """
    global similarity_f
    similarity_f = similarity
    global embeddings_f
    embeddings_f = embeddings

def find_most_similiar(group, sim_df, embeddings_df): 
    length = len(group)-1
    similarities = []
    distances = []
    pos = []
    words = group["Presented Word"].values #get rid of pandas indexing
    
    i = 0
    while i <= length: 
        most_sim_index = i 
        current_word = words[i]
        most_sim_val = -2  

        j = 0
        while j < i: 
            comparison_word = words[j]
            try: 
                similarity = sim_df.loc[current_word, comparison_word]
                if pd.notna(similarity) and similarity > most_sim_val:
                    most_sim_val, most_sim_index = similarity, j
            except KeyError: 
                print (f"{current_word} or {comparison_word} is not found on object")
                pass
            j += 1

        if most_sim_val < -1: 
            similarities.append(np.nan)
            distances.append(np.nan)
            pos.append(np.nan)
        else:
            similarities.append(most_sim_val)
            distances.append(i-most_sim_index)
            pos.append(most_sim_index)
        i += 1
    return pd.DataFrame(data = {"MOST_SIM_VAL": similarities,
                             "MOST_SIM_DIST": distances, "DEBUG_MOST_SIM_POS": pos}, index=group.index)

def avg_sim_previous(group, sim_df, embeddings_df, n): 
    length = len(group)
    avg_similarities = []
    words = group["Presented Word"].values 

    i = 0
    while i < length:
        current_word = words[i]
        
        # THE FIX: This determines the start of your look-back window.
        # If i=5 and n=3, start_idx is 2. (It checks j=2, 3, 4)
        # If i=1 and n=3, start_idx is 0. (It checks j=0)
        start_idx = max(0, i - n)
        
        sim_sum = 0
        valid_count = 0
        
        # Loop ONLY from the start_idx up to the current word
        j = start_idx
        while j < i:
            comparison_word = words[j]
            try:
                similarity = sim_df.loc[current_word, comparison_word]
                
                # If the similarity exists, add it to our running total
                if pd.notna(similarity):
                    sim_sum += similarity
                    valid_count += 1
            except KeyError:
                pass
            j += 1
            
        # Calculate the average. If no valid words were found (or i=0), return NaN.
        if valid_count > 0:
            avg_similarities.append(sim_sum / valid_count)
        else:
            avg_similarities.append(np.nan)
            
        i += 1
        
    # Return a DataFrame with a dynamic column name based on 'n'
    return pd.DataFrame(
        data={f"AVG_SIM_PREV_{n}": avg_similarities}, 
        index=group.index
    )

def sim_last_word(group, sim_df, embeddings_df): 
    length = len(group)-1
    similarities = []
    words = group["Presented Word"].values #get rid of pandas indexing
    
    i = 0
    while i <= length:
        current_word = words[i]
        if i>0: 
            comparison_word = words[i-1]
            try: 
                similarity = sim_df.loc[current_word, comparison_word]
                similarities.append(similarity)
            except KeyError: 
                similarities.append(np.nan)
        else: 
            similarities.append(np.nan)
        i += 1
    return pd.DataFrame(data = {"PREV_SIM": similarities}, index = group.index)

def contrast(group, sim_df, embeddings_df, n): 
    length = len(group)
    distances = []
    words = group["Presented Word"].values
    
    i = 0
    while i < length:
        current_word = words[i]
        
        # 1. Safely fetch the target vector first. 
        # If the target word isn't in FastText, we can't calculate a distance.
        try:
            current_word_vector = embeddings_df[current_word]
        except KeyError:
            distances.append(np.nan)
            i += 1
            continue

        # 2. Safe Boundaries (prevents IndexError at the end of the list)
        # Using slice-style boundaries so we don't need +1 in the range later
        neighbors_idx_begin = max(0, i - n)
        neighbors_idx_end = min(length, i + n + 1) 
        
        neighbor_vectors = []
        
        # 3. Iterate through the safe window
        for j in range(neighbors_idx_begin, neighbors_idx_end):
            if j == i:
                continue # Skip the target word itself
                
            word_to_fetch = words[j]
            try:
                # Only add the vector if FastText actually has it
                vector = embeddings_df[word_to_fetch]
                neighbor_vectors.append(vector)
            except KeyError:
                pass
                
        # 4. Calculate Centroid and Distance
        # We must ensure we actually found valid neighbors before taking the mean
        if len(neighbor_vectors) > 0:
            centroid = np.mean(neighbor_vectors, axis=0)
            dist = 1- cosine(current_word_vector, centroid)
            distances.append(dist)
        else:
            distances.append(np.nan)
            
        i += 1
    return pd.DataFrame(
        data={f"AVG_CONTRAST_{n}": distances}, 
        index=group.index
    )

def fatigue(group, sim_df, embeddings_df): 
    length = len(group)
    fatigue = [i for i in range(1, length+1)]
    return pd.DataFrame({"FATIGUE":fatigue}, index=group.index)

def position_zone(group, sim_df, embeddings_df): 
    positions = group["Present Position"].values
    zones = []
    for pos in positions:
        if pos <= 4:
            zones.append(1)
        elif pos <= 9: 
            zones.append(2)
        else: 
            zones.append(3)
    return pd.DataFrame({"PRESENT_ZONE":zones}, index = group.index)


"""
<NOTE> Logic 

simliarity_features_vd = df.groupby(["Participant ID", "Participant Group", "ListID"]).apply(find_most_similiar, word_sim_df)
clean_features = simliarity_features_vd.droplevel([0,1,2])
df = df.join(clean_features)

"""



def apply_join(df_main, function, similarity_df = None, embeddings_df = None, groupby_groups = None, **kwargs):
    """
    a function that applies other functions and returns the dataframe clearly. 
    """
    if similarity_df == None:
        similarity_df = similarity_f
    if embeddings_df == None:
        embeddings_df = embeddings_f

    temp = df_main.groupby(by = groupby_groups).apply(function, similarity_df,
                                                      embeddings_df, include_groups = False,
                                                      **kwargs)
    temp = temp.droplevel([i for i in range(len(groupby_groups))])
    out = df_main.join(temp)

    return out 

def to_1d_array(x):
    if isinstance(x, str):
        cleaned = x.replace('[', '').replace(']', '').strip()
        return np.fromstring(cleaned, sep=' ')
    elif hasattr(x, 'numpy'):
        return x.detach().cpu().numpy().flatten()
    else:
        return np.array(x).flatten()
    
def bert_find_most_similiar(group, sim_df, embeddings_df): 
    length = len(group)-1
    similarities = []
    distances = []
    pos = []
    words = group["Presented Word"].values #get rid of pandas indexing

    vectors = [to_1d_array(v) for v in group["BERT"].values]
    
    i = 0
    while i <= length: 
        most_sim_index = i 
        current_word = words[i]
        current_word_vector = vectors[i]
        most_sim_val = -2  

        j = 0
        while j < i: 
            comparison_word = words[j]
            comparison_word_vector = vectors [j]
            try: 
                similarity = 1- cosine(current_word_vector, comparison_word_vector)
                if pd.notna(similarity) and similarity > most_sim_val:
                    most_sim_val, most_sim_index = similarity, j
            except KeyError: 
                print (f"{current_word} or {comparison_word} is not found on object")
                pass
            j += 1

        if most_sim_val < -1: 
            similarities.append(np.nan)
            distances.append(np.nan)
            pos.append(np.nan)
        else:
            similarities.append(most_sim_val)
            distances.append(i-most_sim_index)
            pos.append(most_sim_index)
        i += 1
    return pd.DataFrame(data = {"MOST_SIM_VAL": similarities,
                             "MOST_SIM_DIST": distances, "DEBUG_MOST_SIM_POS": pos}, index=group.index)


def bert_avg_sim_previous(group, sim_df, embeddings_df, n):
    length = len(group)
    avg_similarities = []
    vectors = [to_1d_array(v) for v in group["BERT"].values]

    i = 0
    while i < length:
        current_word = vectors[i]
        
        # THE FIX: This determines the start of your look-back window.
        # If i=5 and n=3, start_idx is 2. (It checks j=2, 3, 4)
        # If i=1 and n=3, start_idx is 0. (It checks j=0)
        start_idx = max(0, i - n)
        
        sim_sum = 0
        valid_count = 0
        
        # Loop ONLY from the start_idx up to the current word
        j = start_idx
        while j < i:
            comparison_word = vectors[j]
            try:
                similarity = 1- cosine(current_word, comparison_word)
                
                # If the similarity exists, add it to our running total
                if pd.notna(similarity):
                    sim_sum += similarity
                    valid_count += 1
            except KeyError:
                pass
            j += 1
            
        # Calculate the average. If no valid words were found (or i=0), return NaN.
        if valid_count > 0:
            avg_similarities.append(sim_sum / valid_count)
        else:
            avg_similarities.append(np.nan)
            
        i += 1
        
    # Return a DataFrame with a dynamic column name based on 'n'
    return pd.DataFrame(
        data={f"AVG_SIM_PREV_{n}": avg_similarities}, 
        index=group.index
    )

def bert_sim_last_word(group, sim_df, embeddings_df): 
    length = len(group)-1
    similarities = []
    words = group["Presented Word"].values #get rid of pandas indexing
    vectors = [to_1d_array(v) for v in group["BERT"].values]

    i = 0
    while i <= length:
        current_word = vectors[i]
        if i>0: 
            comparison_word = vectors[i-1]
            try: 
                similarity = 1- cosine(current_word, comparison_word)
                similarities.append(similarity)
            except KeyError: 
                similarities.append(np.nan)
        else: 
            similarities.append(np.nan)
        i += 1
    return pd.DataFrame(data = {"PREV_SIM": similarities}, index = group.index)


def bert_contrast(group, sim_df, embeddings_df, n): 
    length = len(group)
    distances = []
    vectors = [to_1d_array(v) for v in group["BERT"].values]
    
    i = 0
    while i < length:
        current_word = vectors[i]
        
        # 1. Safely fetch the target vector first. 
        # If the target word isn't in FastText, we can't calculate a distance.
        try:
            current_word_vector = current_word
        except KeyError:
            distances.append(np.nan)
            i += 1
            continue

        # 2. Safe Boundaries (prevents IndexError at the end of the list)
        # Using slice-style boundaries so we don't need +1 in the range later
        neighbors_idx_begin = max(0, i - n)
        neighbors_idx_end = min(length, i + n + 1) 
        
        neighbor_vectors = []
        
        # 3. Iterate through the safe window
        for j in range(neighbors_idx_begin, neighbors_idx_end):
            if j == i:
                continue # Skip the target word itself
                
            try:
                # Only add the vector if FastText actually has it
                vector = vectors[j]
                neighbor_vectors.append(vector)
            except KeyError:
                pass
                
        # 4. Calculate Centroid and Distance
        # We must ensure we actually found valid neighbors before taking the mean
        if len(neighbor_vectors) > 0:
            centroid = np.mean(neighbor_vectors, axis=0)
            dist = 1- cosine(current_word_vector, centroid)
            distances.append(dist)
        else:
            distances.append(np.nan)
            
        i += 1
    return pd.DataFrame(
        data={f"AVG_CONTRAST_{n}": distances}, 
        index=group.index
    )

def bert_stm_max_sim_n(group, sim_df, embeddings_df, n): 
    length = len(group)-1
    similarities = []
    pos = []
    distances = []
    words = group["Presented Word"].values #get rid of pandas indexing

    vectors = [to_1d_array(v) for v in group["BERT"].values]
    
    i = 0
    while i <= length: 
        most_sim_index = i 
        current_word = words[i]
        current_word_vector = vectors[i]
        most_sim_val = -2  

        j = max(0, i-n)
        while j < i: 
            comparison_word = words[j]
            comparison_word_vector = vectors [j]
            try: 
                similarity = 1- cosine(current_word_vector, comparison_word_vector)
                if pd.notna(similarity) and similarity > most_sim_val:
                    most_sim_val, most_sim_index = similarity, j
            except KeyError: 
                print (f"{current_word} or {comparison_word} is not found on object")
                pass
            j += 1

        if most_sim_val < -1: 
            similarities.append(1) #word itself
            distances.append(0) #word itself
            pos.append(most_sim_index)
        else:
            similarities.append(most_sim_val)
            distances.append(i-most_sim_index)
            pos.append(most_sim_index)
        i += 1
    return pd.DataFrame(data = {"STM_MAX_SIM": similarities,
                             "STM_MAX_SIM_DIST": distances, "STM_MAX_SIM_POS": pos}, index=group.index)

"""
def bert_stm_semantic_pairs(group, sim_df, embeddings_df, n):
    length = len(group)-1
    similarities = []
    pos = []
    distances = []
    words = group["Presented Word"].values #get rid of pandas indexing

    vectors = [to_1d_array(v) for v in group["BERT"].values]
"""

def bert_stm_cumulative_semantic_density(group, sim_df, embeddings_df, n):
    """avg pairwise similarity from beginnig to i for rolling window n"""
    length = len(group)-1
    semantic_density = []
    words = group["Presented Word"].values #get rid of pandas indexing
    vectors = [to_1d_array(v) for v in group["BERT"].values]
    i = 0
    while i <= length: # 0 1 2 3 | i=4
        left = max(0, i-n)
        pairwise_cosine = []
        while left < i: 
            current_word = vectors[left]
            for p in range(i - left -1): 
                p += 1 
                comparison_word = vectors[left+p]
                pairwise_cosine.append(1- cosine(current_word, comparison_word))
            left += 1
        if i == 0: 
            semantic_density.append(0)
        else:
            semantic_density.append(sum(pairwise_cosine))
        i += 1
    return pd.DataFrame(data = {"STM_SEMANTIC_D": semantic_density}, index=group.index)
    




def bert_ltm_contrast():
    """contrast to list centroid thus far"""
    pass

def bert_attention(group, sim_df, embeddings_df, n):
    pass


def ft_stm_max_sim_n(group, sim_df, embeddings_df, n):
    length = len(group)-1
    similarities = []
    pos = []
    distances = []
    words = group["Presented Word"].values #get rid of pandas indexing
    
    i = 0
    while i <= length: 
        most_sim_index = i 
        current_word = words[i]
        most_sim_val = -2  

        j = max(0, i-n)
        while j < i: 
            comparison_word = words[j]
            try: 
                similarity = sim_df.loc[current_word, comparison_word]
                if pd.notna(similarity) and similarity > most_sim_val:
                    most_sim_val, most_sim_index = similarity, j
            except KeyError: 
                print (f"{current_word} or {comparison_word} is not found on object")
                pass
            j += 1

        if most_sim_val < -1: 
            similarities.append(1) #word itself
            distances.append(0) #word itself
            pos.append(most_sim_index)
        else:
            similarities.append(most_sim_val)
            distances.append(i-most_sim_index)
            pos.append(most_sim_index)
        i += 1
    return pd.DataFrame(data = {"STM_MAX_SIM": similarities,
                             "STM_MAX_SIM_DIST": distances, "STM_MAX_SIM_POS": pos}, index=group.index)

def ft_stm_cumulative_semantic_density(group, sim_df, embeddings_df, n):
    """avg pairwise similarity from beginnig to i for rolling window n"""
    length = len(group)-1
    semantic_density = []
    words = group["Presented Word"].values #get rid of pandas indexing
    i = 0
    current_word = words[i]
    while i <= length: # 0 1 2 3 | i=4
        left = max(0, i-n)
        pairwise_cosine = []
        while left < i: 
            for p in range(i - left -1): 
                p += 1 
                comparison_word = words[left+p]
                pairwise_cosine.append(sim_df.loc[current_word,comparison_word])
            left += 1
        if i == 0: 
            semantic_density.append(0)
        else:
            semantic_density.append(sum(pairwise_cosine))
        i += 1
    return pd.DataFrame(data = {"STM_SEMANTIC_D": semantic_density}, index=group.index)
    
def stm_uniqueness(group, sim_df, embeddings_df, n):
    """uniqueness value, cosine distnace to centroid of rolling window n"""
    length = len(group)-1
    uniquenes = []
    words = group["Presented Word"].values #get rid of pandas indexing
    i = 0
    current_word = words[i]
    current_vector = to_1d_array(embeddings_df[current_word].T)
    while i <= length: # 0 1 2 3 | i=4
        left = max(0, i-n)
        neighboring_words = []
        while left < i: 
            for p in range(i - left -1): 
                p += 1 
                comparison_word = words[left+p]
                neighboring_words.append(to_1d_array(embeddings_df[comparison_word].T))
            left += 1
        centroid = np.mean(neighboring_words, axis = 0)
        if i == 0: 
            uniquenes.append(1)
        else:
            uniquenes.append(cosine(current_vector, centroid))
        i += 1
    return pd.DataFrame(data = {"UNIQUENESS": uniquenes}, index=group.index)