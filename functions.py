# Title: Interactive Bangla Sentence Annotation via Embedding Projection & Clustering
#
# This code provides an interactive tool (for notebooks) to visualize sentence
# embeddings in 2D, cluster them, and let a user label points by clicking.
# It supports multiple dimensionality reducers (UMAP, MDS, t-SNE, PCA) and
# an interactive Plotly scatter plot with ipywidgets controls. Labeled
# sentences can be saved to CSV as a gold dataset with inferred target
# word offsets.
#
# 1) Prepare helper utilities (text wrapping, highlighting).
# 2) Cluster embeddings (KMeans) and project to 2D via a chosen reducer.
# 3) Render an interactive Plotly scatter plot; let the user click to assign labels.
# 4) Display grouped sentences per label and save labeled selections to disk
#    with (lemma, sentence, sense, start, end).
# 5) Provide robust word-position utilities: exact match first, then a
#    Levenshtein-distance fallback to mark the target span.
#
# Source acknowledgement & references:
# The interactive functions and workflow in this `functions.py` are adapted from the
# following repository and paper. Please cite them when using or extending this code.
#
# - Goworek, R. (n.d.) *projecting_sentences*. GitHub. Available at:
#   https://github.com/roksanagow/projecting_sentences (Accessed: 17 August 2025).
#
# - Goworek, R., Karlcut, H., Shezad, H., Darshana, N., Mane, A., Bondada, S., Sikka, R.,
#   Mammadov, U., Allahverdiyev, R., Purighella, S., Gupta, P., Ndegwa, M., Tran, B.K.
#   and Dubossarsky, H. (n.d.) ‘SenWiCh: Sense-Annotation of Low-Resource Languages for
#   WiC using Hybrid Methods’. [Preprint].


import re  # Regular expressions for word searches and token-like matching
from IPython.display import display, HTML  # Notebook display utilities (rich HTML, widgets output)
import plotly.graph_objects as go  # Plotly graph objects for interactive scatter
import ipywidgets as widgets  # Interactive widgets (buttons, layout)
import pandas as pd  # Tabular data handling for saving labeled sentences
import os  # Filesystem paths and directory creation
from sklearn.cluster import KMeans  # KMeans clustering for grouping points
from sklearn.decomposition import PCA  # PCA option for dimensionality reduction
from sklearn.manifold import TSNE, MDS  # t-SNE / MDS reducers for 2D projection
from umap import UMAP  # UMAP reducer for 2D projection


# Function: split_text
# Purpose : Soft-wrap a long string into <br>-separated chunks for plot hovers.
# Inputs  : 
#   text (str) - the raw sentence; max_line_length (int) - chunk size.
# Outputs : 
#   (str) HTML string with <br> inserted every max_line_length chars.
def split_text(text, max_line_length):  # Define a text-wrapping helper
    return '<br>'.join(text[i:i + max_line_length] for i in range(0, len(text), max_line_length))  # Chunk and join with <br>


# Function: make_word_bold
# Purpose : Bold a target word inside a sentence for hover display.
# Uses simple .replace (all occurrences), HTML-safe in Plotly hover.
# Inputs  : 
#   text (str) - sentence; word (str) - target token to bold.
# Outputs : 
#   (str) HTML string with <b>word</b> substituted.
def make_word_bold(text, word):  # Define a helper to bold a word
    return text.replace(word, f"<b>{word}</b>")  # Wrap occurrences with <b> tags


# Function: project_group_and_scatter_plot_embeddings_interactive
# Purpose : Cluster sentence embeddings, reduce to 2D, show an interactive
# scatter plot for manual labeling; provide controls to switch labels,
# finalize groupings, and save labeled sentences to CSV (gold set).
# Clicking assigns the current label color to selected points.
# "Finish Labelling" prints groups; "Save Sentences" writes CSV to saved_sentences/.
# Word offsets are computed via get_positions (exact match → edit-distance fallback).
# Inputs  :
#   - embeddings (array-like [N,D]): sentence vectors
#   - sentences (List[str])        : raw sentences (length N)
#   - words (List[str])            : per-sentence target word (length N; first used as lemma)
#   - n_clusters (int)             : initial KMeans cluster count
#   - dim_reducer (str)            : 'umap' | 'mds' | 'tsne' | 'pca'
# Outputs :
#   - fig (plotly.graph_objs.FigureWidget): interactive scatter widget
def project_group_and_scatter_plot_embeddings_interactive(  # Main interactive visualization + labeling function
    embeddings, sentences, words, n_clusters=5, dim_reducer='mds'  # Accept embeddings, sentences, word list, cluster count, and reducer
):
    if n_clusters > len(embeddings):  # Safety: cap clusters to number of points
        n_clusters = len(embeddings)  # Adjust cluster count if too large

    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)  # Fit KMeans on embeddings
    labels = kmeans.labels_  # Retrieve cluster labels (not directly used for coloring here)

    # Dimensionality reduction
    reducer = {  # Map reducer name to corresponding object
        'umap': UMAP(n_components=2, random_state=42),
        'mds': MDS(n_components=2, random_state=42),
        'tsne': TSNE(n_components=2, random_state=42),
        'pca': PCA(n_components=2)
    }.get(dim_reducer)  # Pick the reducer based on the string key
    if reducer is None:  # Validate the reducer name
        raise ValueError(f"Unknown dimensionality reducer: {dim_reducer}")  # Inform about bad input

    embeddings_2d = reducer.fit_transform(embeddings)  # Compute 2D coordinates
    x, y = embeddings_2d[:, 0], embeddings_2d[:, 1]  # Split into x/y arrays

    label_colors = ["green", "red", "blue", "orange", "purple"]  # Predefined color palette for labels
    current_label = [0]  # Mutable current label index (list for closure mutability)
    point_labels = [-1] * len(sentences)  # Initialize per-point labels (-1 = unassigned)
    click_groups = [[] for _ in label_colors]  # Containers for sentences per color/label

    hover_text = [make_word_bold(split_text(s, 95), w) for s, w in zip(sentences, words)]  # Build hover strings

    scatter_trace = go.Scatter(  # Define the scatter trace
        x=x,  # X coords
        y=y,  # Y coords
        mode='markers',  # Marker-only plot
        marker=dict(size=[10] * len(sentences), color=["grey"] * len(sentences)),  # Start grey/unlabeled
        text=hover_text,  # Hover text (HTML)
        hoverinfo="text"  # Only show text on hover
    )
    fig = go.FigureWidget([scatter_trace])  # Create an interactive FigureWidget
    fig.update_layout(  # Style/layout of the figure
        title=f'Embeddings of sentences projected using {dim_reducer.upper()}',  # Title with reducer name
        xaxis_title='Dimension 1',  # X label
        yaxis_title='Dimension 2',  # Y label
        hovermode='closest',  # Hover nearest point
        width=700,  # Width in pixels
        height=700  # Height in pixels
    )

    output = widgets.Output()  # Output area for logs/messages
    scatter_trace = fig.data[0]  # Convenience handle to the trace

    def on_click(trace, points, selector):  # Callback: assign current label/color to clicked points
        c = list(scatter_trace.marker.color)  # Get modifiable color list
        for i in points.point_inds:  # For each clicked point index
            point_labels[i] = current_label[0]  # Assign current label id
            c[i] = label_colors[current_label[0]]  # Update color accordingly
        with fig.batch_update():  # Batch UI updates for efficiency
            scatter_trace.marker.color = c  # Apply updated colors

    scatter_trace.on_click(on_click)  # Register click handler on the trace

    def next_label(_):  # Widget callback: cycle to next label color
        current_label[0] = (current_label[0] + 1) % len(label_colors)  # Increment and wrap
        with output:  # Log in the output widget
            print(f"Switched to label {current_label[0]} ({label_colors[current_label[0]]})")  # Inform user

    def finish_labeling(_):  # Widget callback: finalize and show groups per label
        for label in range(len(label_colors)):  # For each label/color
            group = [sentences[i] for i, lbl in enumerate(point_labels) if lbl == label]  # Collect sentences
            click_groups[label] = group  # Save group

        with output:  # Print the groups and counts
            print("✅ Sentence groups by label:")  # Header
            for i, group in enumerate(click_groups):  # Iterate groups
                color = label_colors[i]  # Color name
                display(HTML(  # Rich HTML header per group
                    f"<div style='color:{color}; font-weight: bold; margin-top:10px;'>"
                    f"Label {i} ({color}): {len(group)} sentence(s):</div>"
                ))
                for sent in group:  # Print sentences in plain text
                    print(sent)

            print("\n📋 Full result (list of lists):")  # Show raw structure
            for i, group in enumerate(click_groups):  # Enumerate groups
                print(f"[Label {i}] {group}")  # Print group content

    def save_sentences(_):  # Widget callback: persist labeled sentences to CSV with offsets
        saved_sentences = pd.DataFrame(columns=['lemma','sentence','sense','start','end'])  # Create empty table
        word = words[0]  # Use the first provided word as the lemma key
        for i, group in enumerate(click_groups):  # For each label group
            for sent in group:  # For each sentence in the group
                start, end = get_positions(sent, word)  # Compute character offsets
                saved_sentences.loc[len(saved_sentences)] = [word, sent, i, start, end]  # Append row

        count = 1  # File version suffix
        # if saved_sentences/ doesn't exist, create it
        if not os.path.exists('saved_sentences'):  # Ensure target folder exists
            os.makedirs('saved_sentences')  # Create folder
        while os.path.exists(f'saved_sentences/{word}_labelled_sentences{count}.csv'):  # Find next free suffix
            count += 1  # Increment count if file exists
        filename = f'saved_sentences/{word}_labelled_sentences{count}.csv' if count > 1 else f'saved_sentences/{word}_labelled_sentences.csv'  # First file has no numeric suffix

        saved_sentences.to_csv(filename, index=False)  # Write CSV without index
        with output:  # Log path
            print(f"💾 Saved to {filename}")  # Confirmation message

    btn_save = widgets.Button(description="Save Sentences", button_style='warning')  # Save button
    btn_save.on_click(save_sentences)  # Bind save callback
    btn_next = widgets.Button(description="Next Label", button_style='info')  # Next-label button
    btn_finish = widgets.Button(description="Finish Labelling", button_style='success')  # Finish button
    btn_next.on_click(next_label)  # Bind cycle callback
    btn_finish.on_click(finish_labeling)  # Bind finalize callback

    display(widgets.HBox([btn_next, btn_finish, btn_save]))  # Show buttons in a row
    display(fig, output)  # Render the plot and the output widget

    with output:  # Initial instruction
        print(f"💡 Click to label. Current label: {current_label[0]} ({label_colors[0]})")  # Guide the user

    return fig  # Return the interactive figure widget to the caller


# Function: edit_distance
# Purpose : Compute Levenshtein edit distance between two strings.
# Iterative dynamic programming; used as a fuzzy match fallback.
# Inputs  : 
#   s1 (str), s2 (str) - strings to compare.
# Outputs : 
#   (int) edit distance (insertions + deletions + substitutions).
def edit_distance(s1, s2):  # Define Levenshtein distance
    """Calculate the Levenshtein distance between two strings."""  # Inline docstring (kept as-is)
    if len(s1) < len(s2):  # Ensure s1 is the longer string (swap if needed)
        return edit_distance(s2, s1)  # Recurse with swapped args

    if len(s2) == 0:  # If second string is empty
        return len(s1)  # Distance equals length of first string

    previous_row = range(len(s2) + 1)  # Initialize DP previous row
    for i, c1 in enumerate(s1):  # Iterate characters of s1
        current_row = [i + 1]  # Start current row
        for j, c2 in enumerate(s2):  # Iterate characters of s2
            insertions = previous_row[j + 1] + 1  # Cost of insertion
            deletions = current_row[j] + 1  # Cost of deletion
            substitutions = previous_row[j] + (c1 != c2)  # Cost of substitution
            current_row.append(min(insertions, deletions, substitutions))  # Pick minimum
        previous_row = current_row  # Advance DP row

    return previous_row[-1]  # Final cell is the distance


# Function: find_position_of_similar_word
# Purpose : In a sentence, find the word with minimal edit distance to the
# target and return its (start, end) character offsets.
# Uses a word-like regex to iterate candidates; ties choose first occurrence.
# Relies on edit_distance(...) above for scoring.
# Inputs  :
#   - word (str)     : target lemma/string
#   - sentence (str) : sentence to search in
# Outputs :
#   - best_position (tuple[int,int] | None): (start, end) of closest word; None if none
def find_position_of_similar_word(word, sentence):  # Define fuzzy position finder
    """
    Find the start and end positions of the word in the sentence that has the smallest Levenshtein distance
    to the target word. If multiple words have the same smallest distance, the first one is returned.

    Args:
        word (str): The target word to find a similar match for.
        sentence (str): The sentence to search within.

    Returns:
        tuple: A tuple containing the start and end positions (start_pos, end_pos) of the similar word.
               If no match is found, returns None.
    """
    # Find all whole words in the sentence
    matches = list(re.finditer(r'\b\w+\'?\w*\b', sentence))  # Token-like regex (ASCII word chars + optional apostrophe)
    min_distance = float('inf')  # Initialize best distance as +inf
    best_position = None  # Placeholder for best (start, end)

    # Iterate through each word in the sentence
    for match in matches:  # Loop over candidate words
        sentence_word = match.group(0)  # Extract word text
        distance = edit_distance(word, sentence_word)  # Compute edit distance
        
        # Update the best position if a smaller distance is found
        if distance < min_distance:  # Found better match
            min_distance = distance  # Update best score
            best_position = (match.start(), match.end())  # Store char offsets
        # If the same minimum distance is found, retain the first occurrence
        elif distance == min_distance:  # Tie: keep earlier one
            pass  # No change

    return best_position  # Could be None if no matches


# Function: get_positions
# Purpose : Locate exact character span for a target word inside a sentence.
# If exact whole-word match fails, fall back to fuzzy nearest match.
# Exact match uses word boundaries; fallback via find_position_of_similar_word.
# Returns [None, None] if even fuzzy match fails (uncommon).
# Inputs  :
#   - sentence (str): sentence to search within
#   - word (str)    : target token/lemma to locate
# Outputs :
#   - [start_pos, end_pos] (List[int,int]): character indices for the match
def get_positions(sentence, word):  # Define robust span finder for the target word
    """
    Finds the start and end character positions of a word in a sentence. If the word is found as a complete word,
    returns its start and end positions. If not found, returns the start and end positions of the word with the
    smallest Levenshtein edit distance from the target word.

    Args:
        sentence (str): The sentence to search within.
        word (str): The word to find the positions of.

    Returns:
        list: A list containing the start and end positions [start_pos, end_pos] of the word or the closest match.
    """
    match = re.search(rf'\b{re.escape(word)}\b', sentence)  # Try exact whole-word match (regex-escaped)
    if match:  # If exact hit
        start_pos = match.start()  # Start offset
        end_pos = match.end()  # End offset
    else:  # Otherwise use fuzzy nearest word
        start_pos, end_pos = find_position_of_similar_word(word, sentence)  # May return None
    return [start_pos, end_pos]  # Return list [start, end] for CSV compatibility
