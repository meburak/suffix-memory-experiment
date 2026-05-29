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
        try:
            current_word_vector = embeddings_df[current_word]
        except KeyError:
            distances.append(np.nan)
            i += 1
            continue

        neighbors_idx_begin = max(0, i - n)
        neighbors_idx_end = min(length, i + n + 1) 
        
        neighbor_vectors = []
        for j in range(neighbors_idx_begin, neighbors_idx_end):
            if j == i:
                continue 
            word_to_fetch = words[j]
            try:
                vector = embeddings_df[word_to_fetch]
                neighbor_vectors.append(vector)
            except KeyError:
                pass
                
        if len(neighbor_vectors) > 0:
            centroid = np.mean(neighbor_vectors, axis=0)
            dist = cosine(current_word_vector, centroid)
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
    fatigue = [(-(i-70)**2)**(1/2) for i in range(1, length+1)]
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
        try:
            current_word_vector = current_word
        except KeyError:
            distances.append(np.nan)
            i += 1
            continue

        neighbors_idx_begin = max(0, i - n)
        neighbors_idx_end = min(length, i + n + 1) 
        
        neighbor_vectors = []
        for j in range(neighbors_idx_begin, neighbors_idx_end):
            if j == i:
                continue 
            try:
                vector = vectors[j]
                neighbor_vectors.append(vector)
            except KeyError:
                pass
                
        if len(neighbor_vectors) > 0:
            centroid = np.mean(neighbor_vectors, axis=0)
            dist = cosine(current_word_vector, centroid)
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
    
    while i <= length: 
        left = max(0, i-n)
        pairwise_cosine = []
        while left < i: 
            current_word = words[left] 
            for p in range(i - left -1): 
                p += 1 
                comparison_word = words[left+p]
                pairwise_cosine.append(sim_df.loc[current_word,comparison_word])
            left += 1
            
        if i == 0 or len(pairwise_cosine) == 0: 
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
    
    while i <= length: 
        current_word = words[i]
        current_vector = to_1d_array(embeddings_df[current_word].T)
        
        left = max(0, i-n)
        neighboring_words = []
        
        for j in range(left, i):
            comparison_word = words[j]
            neighboring_words.append(to_1d_array(embeddings_df[comparison_word].T))
            
        if i == 0 or len(neighboring_words) == 0: 
            uniquenes.append(1)
        else:
            centroid = np.mean(neighboring_words, axis=0)
            uniquenes.append(cosine(current_vector, centroid))
        i += 1
    return pd.DataFrame(data = {"UNIQUENESS": uniquenes}, index=group.index)

def categories_in_zone(group, sim_df, embeddings_df, categories_df, n):
    length = len(group)
    words = group["Presented Word"].values
    
    zones = group["PRESENT_ZONE"].values 
  
    cluster_map = dict(zip(categories_df["WORDS"], categories_df["CLUSTER"]))
    
    words_z1 = words[zones == 1]
    words_z2 = words[zones == 2]
    words_z3 = words[zones == 3]

    clusters_z1 = [cluster_map[word] for word in words_z1]
    clusters_z2 = [cluster_map[word] for word in words_z2]
    clusters_z3 = [cluster_map[word] for word in words_z3]


    zone_1_match = []
    zone_2_match = []
    zone_3_match = []

    for i in range(length):
        f1, f2, f3 = 0,0,0
        word = words[i]
        cluster = cluster_map[word]
        if cluster in clusters_z1:
            f1 = 1
        if cluster in clusters_z2:
            f2 = 1
        if cluster in clusters_z3:
            f3 = 1
        zone_1_match.append(f1)
        zone_2_match.append(f2)
        zone_3_match.append(f3)

    return pd.DataFrame(
        data={
            f"SAME_CLUSTER_IN_Z1": zone_1_match,
            f"SAME_CLUSTER_IN_Z2": zone_2_match,
            f"SAME_CLUSTER_IN_Z3": zone_3_match
        }, 
        index=group.index
    )

def categories_in_zone_count(group, sim_df, embeddings_df, categories_df, n):
    length = len(group)
    words = group["Presented Word"].values
    
    zones = group["PRESENT_ZONE"].values 
  
    cluster_map = dict(zip(categories_df["WORDS"], categories_df["CLUSTER"]))
    
    words_z1 = words[zones == 1]
    words_z2 = words[zones == 2]
    words_z3 = words[zones == 3]

    clusters_z1 = [cluster_map[word] for word in words_z1]
    clusters_z2 = [cluster_map[word] for word in words_z2]
    clusters_z3 = [cluster_map[word] for word in words_z3]


    zone_1_match = []
    zone_2_match = []
    zone_3_match = []

    for i in range(length):
        f1, f2, f3 = 0,0,0
        word = words[i]
        cluster = cluster_map[word]
        for cluster_compare in clusters_z1:
            if cluster == cluster_compare:
                f1 += 1
        for cluster_compare in clusters_z2:
            if cluster == cluster_compare:
                f2 += 1
        for cluster_compare in clusters_z3:
            if cluster == cluster_compare:
                f3 += 1

        zone_1_match.append(f1)
        zone_2_match.append(f2)
        zone_3_match.append(f3)

    return pd.DataFrame(
        data={
            f"SAME_CLUSTER_IN_Z1_COUNT": zone_1_match,
            f"SAME_CLUSTER_IN_Z2_COUNT": zone_2_match,
            f"SAME_CLUSTER_IN_Z3_COUNT": zone_3_match
        }, 
        index=group.index
    )

def category_in_stm_n(group, sim_df, embeddings_df, categories_df, n):
    length = len(group)
    words = group["Presented Word"].values
    
    # Hızlı arama için pandas df'ini sözlüğe çeviriyoruz
    cluster_map = dict(zip(categories_df["WORDS"], categories_df["CLUSTER"]))
    
    stm_match = []
    
    for i in range(length):
        current_word = words[i]
        current_cluster = cluster_map.get(current_word, None)
        
        match_flag = 0
        
        # Eğer güncel kelimenin bir kategorisi varsa geçmiş n kelimeye (STM) bak
        if current_cluster is not None:
            start_idx = max(0, i - n)
            
            for j in range(start_idx, i):
                prev_word = words[j]
                prev_cluster = cluster_map.get(prev_word, None)
                
                # Eğer aynı kategoriden bir kelime STM penceresinde varsa
                if prev_cluster == current_cluster:
                    match_flag = 1
                    break  # Bulduğumuz an gerisine bakmaya gerek yok, 1 yap çık
                    
        stm_match.append(match_flag)
        
    return pd.DataFrame(
        data={f"CATEGORY_IN_STM_{n}": stm_match}, 
        index=group.index
    )

def category_fetch_clusters(group, sim_df, embeddings_df, categories_df, n):
    length = len(group)
    words = group["Presented Word"].values
    
    cluster_map = dict(zip(categories_df["WORDS"], categories_df["CLUSTER"]))
    
    cluster_output = []
    
    for i in range(length):
        current_word = words[i]
        current_cluster = cluster_map.get(current_word, None)
        
        cluster_output.append(current_cluster)
        
    return pd.DataFrame(
        data={f"CLUSTER": cluster_output}, 
        index=group.index
    )
