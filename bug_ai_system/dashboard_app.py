# ==============================
# 🐞 Bug AI Dashboard with Clustering Visualization (Optimized for Large Data)
# Purpose: Upload bug CSV → Clean step-by-step → Compare clustering methods + Visualize
# ==============================

import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np
import plotly.express as px

# ------------------------------
# NLP Downloads
# ------------------------------
nltk.download('stopwords')
nltk.download('wordnet')

# ------------------------------
# Session State Initialization
# ------------------------------
for key in ['df_stage1','df_stage2','df_stage3','df_final',
            'X','df_clustered_som','df_clustered_jaccard','df_clustered_kmeans',
            'removed_words_stage2','removed_words_stage3']:
    if key not in st.session_state:
        st.session_state[key] = None

# ------------------------------
# Dashboard Title & Developer Info
# ------------------------------
st.title("🐞 AI Bug Clustering Dashboard")
st.markdown('<h3 style="font-size:20px;">Developed by ATTIKA AHMED</h3>', unsafe_allow_html=True)
st.markdown('<h3 style="font-size:18px;">Developer Contact: <a href="mailto:codebyattika@gmail.com">codebyattika@gmail.com</a></h3>', unsafe_allow_html=True)

# ------------------------------
# Limit rows and points for large datasets
# ------------------------------
MAX_DISPLAY_ROWS = 500
MAX_PLOT_POINTS = 2000

def display_df_limited(df, stage_name=""):
    if len(df) > MAX_DISPLAY_ROWS:
        st.warning(f"Showing only first {MAX_DISPLAY_ROWS} rows for {stage_name}. Total rows: {len(df)}")
        st.dataframe(df.head(MAX_DISPLAY_ROWS))
    else:
        st.dataframe(df)

# ------------------------------
# Step 1: File Upload
# ------------------------------
uploaded_file = st.file_uploader("Upload CSV file with 'description' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Raw Data:")
    display_df_limited(df, "Raw Data")

    # ------------------------------
    # Stage 1: Remove duplicates & empty
    # ------------------------------
    if st.button("Run Stage 1: Remove duplicates & empty"):
        df_stage1 = df.drop_duplicates()
        df_stage1 = df_stage1[df_stage1['description'].notna()]
        st.session_state['df_stage1'] = df_stage1
        st.success("✅ Stage 1 Done: Duplicates & empty descriptions removed")

    if st.session_state['df_stage1'] is not None:
        st.write("### Stage 1 Result")
        display_df_limited(st.session_state['df_stage1'], "Stage 1")

        col1, col2 = st.columns(2)
        proceed1 = col1.button("Proceed to Stage 2", key="p1")
        if col2.button("Exit", key="exit1"):
            st.stop()

        # ------------------------------
        # Stage 2: Normalization
        # ------------------------------
        if proceed1:
            df2 = st.session_state['df_stage1'].copy()
            def normalize(text):
                text = text.lower()
                text = text.translate(str.maketrans('', '', string.punctuation))
                return ' '.join(text.split())
            df2['clean_desc'] = df2['description'].apply(normalize)
            st.session_state['df_stage2'] = df2
            st.session_state['removed_words_stage2'] = {
                row['description']: set(row['description'].lower().split()) - set(row['clean_desc'].split())
                for _, row in df2.iterrows()
            }

    # ------------------------------
    # Stage 2 Output
    # ------------------------------
    if st.session_state['df_stage2'] is not None:
        st.write("### Stage 2: Normalization")
        display_df_limited(st.session_state['df_stage2'], "Stage 2")
        st.write("Words removed in this stage (punctuation & lowercase):")
        st.write(st.session_state['removed_words_stage2'])

        col1, col2 = st.columns(2)
        proceed2 = col1.button("Proceed to Stage 3", key="p2")
        if col2.button("Exit", key="exit2"):
            st.stop()

        # ------------------------------
        # Stage 3: NLP Cleaning (Stopwords + Lemmatization)
        # ------------------------------
        if proceed2:
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()
            df3 = st.session_state['df_stage2'].copy()

            removed_words_stage3 = {}
            def nlp_clean(text):
                words = text.split()
                filtered = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
                removed_words_stage3[text] = set(words) - set(filtered)
                return ' '.join(filtered)

            df3['clean_desc'] = df3['clean_desc'].apply(nlp_clean)
            st.session_state['df_stage3'] = df3
            st.session_state['removed_words_stage3'] = removed_words_stage3

    # ------------------------------
    # Stage 3 Output
    # ------------------------------
    if st.session_state['df_stage3'] is not None:
        st.write("### Stage 3: NLP Cleaning")
        display_df_limited(st.session_state['df_stage3'], "Stage 3")
        st.write("Words removed in this stage (stopwords + lemmatization):")
        st.write(st.session_state['removed_words_stage3'])

        col1, col2 = st.columns(2)
        proceed3 = col1.button("Proceed to Stage 4 (Vectorization)", key="p3")
        if col2.button("Exit", key="exit3"):
            st.stop()

        # ------------------------------
        # Stage 4: Vectorization
        # ------------------------------
        if proceed3:
            df_final = st.session_state['df_stage3'].copy()
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(df_final['clean_desc'])
            st.session_state['df_final'] = df_final
            st.session_state['X'] = X

    # ------------------------------
    # Final Cleaned Data
    # ------------------------------
    if st.session_state['df_final'] is not None:
        st.write("### Final Cleaned Data")
        display_df_limited(st.session_state['df_final'], "Final Cleaned Data")
        csv_clean = st.session_state['df_final'].to_csv(index=False).encode('utf-8')
        st.download_button("Download Cleaned Data", csv_clean, "cleaned_data.csv")

        # ------------------------------
        # Clustering Selection
        # ------------------------------
        st.write("### Choose Clustering Method")
        method = st.multiselect("Select one or more clustering methods:", 
                                ["Standard SOM", "Improved SOM (Jaccard)", "K-Means"])

        # ------------------------------
        # Run Selected Clustering
        # ------------------------------
        if st.button("Run Selected Clustering", key="cluster_btn"):
            df_final = st.session_state['df_final'].copy()
            X = st.session_state['X']

            # ---------- Jaccard similarity function ----------
            def jaccard(v1, v2):
                set1 = set(np.where(v1.toarray()[0] > 0)[0])
                set2 = set(np.where(v2 > 0)[0])
                return len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0

            # ---------- Standard SOM ----------
            if "Standard SOM" in method:
                num_clusters = 3
                centers = [X[i].toarray()[0] for i in range(num_clusters)]
                clusters = []
                for i in range(X.shape[0]):
                    similarities = [np.dot(X[i].toarray()[0], center) for center in centers]
                    cluster = np.argmax(similarities)
                    clusters.append(cluster)
                df_som = df_final.copy()
                df_som['Cluster'] = clusters
                st.session_state['df_clustered_som'] = df_som

            # ---------- Improved SOM with Jaccard ----------
            if "Improved SOM (Jaccard)" in method:
                num_clusters = 3
                centers = [X[i].toarray()[0] for i in range(num_clusters)]
                clusters = []
                for i in range(X.shape[0]):
                    similarities = [jaccard(X[i], center) for center in centers]
                    cluster = np.argmax(similarities)
                    clusters.append(cluster)
                df_jaccard = df_final.copy()
                df_jaccard['Cluster'] = clusters
                st.session_state['df_clustered_jaccard'] = df_jaccard

            # ---------- K-Means ----------
            if "K-Means" in method:
                km = KMeans(n_clusters=3, random_state=42)
                km.fit(X)
                df_kmeans = df_final.copy()
                df_kmeans['Cluster'] = km.labels_
                st.session_state['df_clustered_kmeans'] = df_kmeans

            st.success("✅ Clustering Completed!")

    # ------------------------------
# ===============================
# 📊 Advanced Cluster Visualization (Disk View) + Silhouette Scores
# ===============================
st.subheader("📊 Advanced Cluster Visualization (Disk View)")

clustering_dfs = {
    "Standard SOM": st.session_state.get('df_clustered_som'),
    "Improved SOM (Jaccard)": st.session_state.get('df_clustered_jaccard'),
    "K-Means": st.session_state.get('df_clustered_kmeans')
}

sil_scores = {}  # Store silhouette scores

for name, df_cluster in clustering_dfs.items():
    if df_cluster is not None:
        st.markdown(f"### 🔵 {name} Clusters")

        # Sample points if dataset is large
        if len(df_cluster) > MAX_PLOT_POINTS:
            df_sample = df_cluster.sample(MAX_PLOT_POINTS, random_state=42)
            st.info(f"Showing {MAX_PLOT_POINTS} sampled points for visualization")
        else:
            df_sample = df_cluster

        # Reduce dimensions for visualization
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(st.session_state['X'].toarray())
        if len(df_cluster) > MAX_PLOT_POINTS:
            X_vis = X_vis[df_sample.index]

        # Prepare visualization dataframe
        df_vis = df_sample.copy()
        df_vis['PC1'] = X_vis[:, 0]
        df_vis['PC2'] = X_vis[:, 1]

        # Disk-style scatter plot with colored clusters
        fig = px.scatter(
            df_vis,
            x='PC1',
            y='PC2',
            color=df_vis['Cluster'].astype(str),       # categorical coloring
            hover_data=['description', 'clean_desc'],
            title=f"{name} Clusters (PCA Projection)",
            size_max=12,
            opacity=0.9,
            color_discrete_sequence=px.colors.qualitative.Safe  # 11 distinct colors
        )

        # Add disk style
        fig.update_traces(
            marker=dict(
                size=18,
                line=dict(width=1, color='DarkSlateGrey'),
            )
        )

        # Add cluster centroids
        centroids = df_vis.groupby('Cluster')[['PC1','PC2']].mean().reset_index()
        fig.add_scatter(
            x=centroids['PC1'],
            y=centroids['PC2'],
            mode='markers',
            marker=dict(color='black', size=25, symbol='x'),
            name='Centroids'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Download clustered data
        csv_cluster = df_cluster.to_csv(index=False).encode('utf-8')
        st.download_button(f"Download {name} Clustered Data", csv_cluster, f"{name}_clustered.csv")

        # ------------------------------
        # Compute silhouette score
        # ------------------------------
        try:
            labels = df_cluster['Cluster'].values
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X_vis, labels)
                sil_scores[name] = score
            else:
                sil_scores[name] = -1
        except Exception:
            sil_scores[name] = -1

# ------------------------------
# Suggested Best Algorithm
# ------------------------------
if sil_scores:
    st.subheader("💡 Suggested Best Clustering Algorithm Based on Silhouette Score")
    
    # Display all scores in a table
    score_df = pd.DataFrame(list(sil_scores.items()), columns=["Algorithm", "Silhouette Score"])
    score_df['Silhouette Score'] = score_df['Silhouette Score'].round(3)
    st.dataframe(score_df)

    # Highlight best algorithm
    best_algo = max(sil_scores, key=sil_scores.get)
    st.success(f"✅ Recommended Algorithm: **{best_algo}** (Silhouette Score: {sil_scores[best_algo]:.3f})")
    st.info("Higher silhouette score indicates better-defined clusters.")

    