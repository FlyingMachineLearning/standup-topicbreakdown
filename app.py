import re
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Stand-Up Topic Modeler",
    page_icon="🎙️",
    layout="wide",
)

DEFAULT_COMEDY_STOPWORDS = {
    "yeah", "yep", "yes", "okay", "ok", "right", "like", "know", "mean",
    "thing", "things", "gonna", "gotta", "wanna", "really", "just", "actually",
    "literally", "uh", "um", "hmm", "hey", "oh", "ah", "man", "dude",
    "people", "person", "say", "said", "says", "go", "going", "went",
    "get", "got", "getting", "come", "came", "look", "think",
}

TIMESTAMP_RE = re.compile(
    r"(?m)^\s*(?:\[\s*)?(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?(?:\s*\])?\s*[-–—:]?\s*"
)
BRACKET_CUE_RE = re.compile(
    r"\[(?:applause|laughter|laughs|music|cheering|cheers|audience|inaudible)[^\]]*\]",
    flags=re.I,
)


def clean_transcript(text: str) -> str:
    text = text.replace("\ufeff", " ").replace("\r\n", "\n")
    text = TIMESTAMP_RE.sub("", text)
    text = BRACKET_CUE_RE.sub(" ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunk_words = max(40, int(chunk_words))
    overlap_words = max(0, min(int(overlap_words), chunk_words - 20))
    step = max(20, chunk_words - overlap_words)

    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_words]
        if len(piece) < max(30, chunk_words // 4) and chunks:
            break
        chunks.append(" ".join(piece))
        if start + chunk_words >= len(words):
            break
    return chunks


def make_corpus(files, chunk_words: int, overlap_words: int) -> pd.DataFrame:
    rows = []
    for f in files:
        raw = f.getvalue().decode("utf-8", errors="replace")
        cleaned = clean_transcript(raw)
        for i, chunk in enumerate(chunk_text(cleaned, chunk_words, overlap_words), start=1):
            rows.append(
                {
                    "source": f.name,
                    "chunk_id": i,
                    "text": chunk,
                    "word_count": len(chunk.split()),
                }
            )
    return pd.DataFrame(rows)


def fit_topic_model(
    docs: list[str],
    requested_topics: int,
    stop_words: list[str],
    use_bigrams: bool,
):
    min_df = 2 if len(docs) >= 12 else 1
    max_df = 0.92 if len(docs) >= 8 else 1.0

    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']{1,}\b",
        ngram_range=(1, 2 if use_bigrams else 1),
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        max_features=12000,
    )
    X = vectorizer.fit_transform(docs)

    max_topics = max(1, min(X.shape[0] - 1, X.shape[1] - 1))
    n_topics = max(1, min(int(requested_topics), max_topics))

    model = NMF(
        n_components=n_topics,
        init="nndsvda",
        random_state=42,
        max_iter=700,
        l1_ratio=0.05,
    )
    W = model.fit_transform(X)
    H = model.components_
    terms = np.asarray(vectorizer.get_feature_names_out())
    return vectorizer, model, X, W, H, terms


def topic_table(H, terms, top_n: int) -> pd.DataFrame:
    rows = []
    for topic_idx, weights in enumerate(H):
        ids = weights.argsort()[::-1][:top_n]
        words = terms[ids]
        vals = weights[ids]
        label = " · ".join(words[:3])
        rows.append(
            {
                "topic": topic_idx + 1,
                "label": label,
                "top_terms": ", ".join(words),
                "term_weights": vals.tolist(),
                "terms": words.tolist(),
            }
        )
    return pd.DataFrame(rows)


def representative_chunks(corpus, W, topic_idx: int, n: int = 5):
    scores = W[:, topic_idx]
    order = np.argsort(scores)[::-1][:n]
    out = corpus.iloc[order][["source", "chunk_id", "text"]].copy()
    out["topic_strength"] = scores[order]
    return out


def topic_prevalence(W):
    totals = W.sum(axis=0)
    if totals.sum() == 0:
        return np.repeat(1 / len(totals), len(totals))
    return totals / totals.sum()


def make_word_web(
    corpus: pd.DataFrame,
    W: np.ndarray,
    H: np.ndarray,
    terms: np.ndarray,
    topic_idx: int,
    top_words: int = 18,
    max_edges: int = 40,
):
    component = H[topic_idx]
    ids = component.argsort()[::-1][:top_words]
    selected_terms = terms[ids].tolist()
    selected_weights = component[ids]

    dominant = W.argmax(axis=1)
    relevant_ids = np.where(dominant == topic_idx)[0]

    # If the topic rarely wins outright, use its strongest chunks instead.
    if len(relevant_ids) < 4:
        relevant_ids = np.argsort(W[:, topic_idx])[::-1][: min(20, len(corpus))]

    G = nx.Graph()
    max_node_weight = max(float(selected_weights.max()), 1e-9)
    for term, weight in zip(selected_terms, selected_weights):
        G.add_node(term, weight=float(weight) / max_node_weight)

    edge_scores = {}
    for doc_idx in relevant_ids:
        text = corpus.iloc[int(doc_idx)]["text"].lower()
        present = []
        for term in selected_terms:
            if re.search(r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)", text):
                present.append(term)

        chunk_weight = float(W[int(doc_idx), topic_idx])
        for a, b in combinations(present, 2):
            key = tuple(sorted((a, b)))
            edge_scores[key] = edge_scores.get(key, 0.0) + chunk_weight

    for (a, b), score in sorted(
        edge_scores.items(), key=lambda kv: kv[1], reverse=True
    )[:max_edges]:
        if score > 0:
            G.add_edge(a, b, weight=score)

    # Preserve isolated top terms rather than hiding them.
    if len(G.nodes) == 0:
        return go.Figure()

    pos = nx.spring_layout(G, seed=42, weight="weight", k=0.75)

    edge_x, edge_y = [], []
    edge_hover = []
    for a, b, data in G.edges(data=True):
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_hover.extend(
            [
                f"{a} ↔ {b}<br>co-occurrence: {data['weight']:.3f}",
                f"{a} ↔ {b}<br>co-occurrence: {data['weight']:.3f}",
                None,
            ]
        )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        hoverinfo="text",
        text=edge_hover,
        line={"width": 1},
        showlegend=False,
    )

    node_x, node_y, node_text, node_size = [], [], [], []
    degree = dict(G.degree(weight="weight"))
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(
            f"<b>{node}</b><br>"
            f"topic weight: {data['weight']:.3f}<br>"
            f"weighted connections: {degree.get(node, 0):.3f}"
        )
        node_size.append(18 + 30 * data["weight"])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(G.nodes()),
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker={"size": node_size, "line": {"width": 1}},
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis={"visible": False},
        yaxis={"visible": False},
        hovermode="closest",
        height=600,
    )
    return fig


def make_topic_map(H, topic_labels):
    sim = cosine_similarity(H)
    G = nx.Graph()
    for i, label in enumerate(topic_labels):
        G.add_node(i, label=f"T{i+1}: {label}")

    # Connect each topic to its two closest neighbors when similarity is nonzero.
    for i in range(len(topic_labels)):
        candidates = np.argsort(sim[i])[::-1]
        added = 0
        for j in candidates:
            if i == j or sim[i, j] <= 0:
                continue
            G.add_edge(i, int(j), weight=float(sim[i, j]))
            added += 1
            if added >= min(2, len(topic_labels) - 1):
                break

    pos = nx.spring_layout(G, seed=42, weight="weight", k=1.1)

    edge_x, edge_y = [], []
    for a, b in G.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        hoverinfo="skip",
        line={"width": 1},
        showlegend=False,
    )

    node_x, node_y, labels = [], [], []
    for node, attrs in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        labels.append(attrs["label"])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        hovertext=labels,
        hoverinfo="text",
        marker={"size": 28, "line": {"width": 1}},
        showlegend=False,
    )

    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=500,
    )
    return fig


st.title("🎙️ Stand-Up Topic Modeler")
st.caption(
    "A lightweight exploratory topic model for comedy transcripts: "
    "TF-IDF → NMF → chunk-level topic assignments → word co-occurrence webs."
)

with st.sidebar:
    st.header("Model controls")
    requested_topics = st.slider("Number of topics", 3, 20, 8)
    chunk_words = st.slider("Chunk size (words)", 60, 400, 160, step=20)
    overlap_words = st.slider("Chunk overlap (words)", 0, 100, 30, step=10)
    top_terms = st.slider("Top terms per topic", 5, 20, 10)
    use_bigrams = st.toggle("Include two-word phrases", value=True)

    st.divider()
    st.subheader("Comedy stopwords")
    extra_stopword_text = st.text_area(
        "Edit/add filler words (comma-separated)",
        value=", ".join(sorted(DEFAULT_COMEDY_STOPWORDS)),
        height=180,
    )

files = st.file_uploader(
    "Upload one or more stand-up transcripts",
    type=["txt", "md"],
    accept_multiple_files=True,
    help="Use one file per special/set/comedian when possible.",
)

if not files:
    st.info(
        "Drop in .txt or .md transcripts. Nothing is uploaded anywhere by this app; "
        "analysis runs in the Streamlit process."
    )
    st.stop()

corpus = make_corpus(files, chunk_words, overlap_words)

if len(corpus) < 4:
    st.error(
        "I need at least four usable chunks. Upload more text or reduce the chunk size."
    )
    st.stop()

custom_stopwords = {
    w.strip().lower()
    for w in re.split(r"[,\n]", extra_stopword_text)
    if w.strip()
}
stop_words = sorted(set(ENGLISH_STOP_WORDS) | custom_stopwords)

try:
    vectorizer, model, X, W, H, terms = fit_topic_model(
        corpus["text"].tolist(),
        requested_topics,
        stop_words,
        use_bigrams,
    )
except ValueError as e:
    st.error(
        "The vocabulary collapsed after filtering. Try removing some stopwords, "
        "turning off bigrams, or uploading more transcript text."
    )
    st.exception(e)
    st.stop()

topic_df = topic_table(H, terms, top_terms)
prevalence = topic_prevalence(W)

dominant = W.argmax(axis=1)
confidence = W.max(axis=1) / np.maximum(W.sum(axis=1), 1e-9)

corpus_out = corpus.copy()
corpus_out["topic"] = dominant + 1
corpus_out["topic_label"] = [
    topic_df.loc[topic_df["topic"] == t, "label"].iloc[0]
    for t in corpus_out["topic"]
]
corpus_out["topic_confidence"] = confidence

metric_cols = st.columns(4)
metric_cols[0].metric("Transcripts", len(files))
metric_cols[1].metric("Chunks", len(corpus))
metric_cols[2].metric("Topics", len(topic_df))
metric_cols[3].metric("Vocabulary", len(terms))

tab1, tab2, tab3, tab4 = st.tabs(
    ["Topic overview", "Word web", "Transcript × topic", "Chunk explorer"]
)

with tab1:
    st.subheader("Top-level topics")
    overview = topic_df[["topic", "label", "top_terms"]].copy()
    overview["prevalence"] = prevalence
    st.dataframe(
        overview,
        use_container_width=True,
        hide_index=True,
        column_config={
            "prevalence": st.column_config.ProgressColumn(
                "Prevalence",
                min_value=0.0,
                max_value=float(max(prevalence.max(), 0.01)),
                format="%.3f",
            )
        },
    )

    prev_df = pd.DataFrame(
        {
            "topic": [f"T{i+1}: {label}" for i, label in enumerate(topic_df["label"])],
            "prevalence": prevalence,
        }
    ).sort_values("prevalence", ascending=True)

    fig = px.bar(
        prev_df,
        x="prevalence",
        y="topic",
        orientation="h",
        title="Corpus-wide topic prevalence",
    )
    st.plotly_chart(fig, use_container_width=True)

    if len(topic_df) > 1:
        st.subheader("Topic neighborhood")
        st.caption(
            "Topics are connected to their nearest neighbors using cosine similarity "
            "between NMF term-weight profiles."
        )
        st.plotly_chart(
            make_topic_map(H, topic_df["label"].tolist()),
            use_container_width=True,
        )

with tab2:
    st.subheader("Topic word web")
    selected_topic_num = st.selectbox(
        "Choose a topic",
        options=topic_df["topic"].tolist(),
        format_func=lambda t: (
            f"T{t}: "
            + topic_df.loc[topic_df["topic"] == t, "label"].iloc[0]
        ),
    )
    topic_idx = int(selected_topic_num) - 1

    selected_row = topic_df.iloc[topic_idx]
    st.write(f"**Top terms:** {selected_row['top_terms']}")
    st.caption(
        "Node size = importance to the topic. Connections = those terms appearing "
        "together in chunks associated with the topic."
    )
    web = make_word_web(
        corpus,
        W,
        H,
        terms,
        topic_idx,
        top_words=min(20, max(top_terms + 5, 12)),
    )
    st.plotly_chart(web, use_container_width=True)

    st.subheader("Representative chunks")
    reps = representative_chunks(corpus, W, topic_idx, n=min(6, len(corpus)))
    for _, row in reps.iterrows():
        with st.expander(
            f"{row['source']} — chunk {int(row['chunk_id'])} "
            f"(strength {row['topic_strength']:.3f})"
        ):
            st.write(row["text"])

with tab3:
    st.subheader("Transcript × topic")
    weights_df = pd.DataFrame(
        W,
        columns=[f"T{i+1}" for i in range(W.shape[1])],
    )
    weights_df["source"] = corpus["source"].values
    source_topic = weights_df.groupby("source").sum()
    source_topic = source_topic.div(
        source_topic.sum(axis=1).replace(0, 1),
        axis=0,
    )

    heatmap = px.imshow(
        source_topic,
        aspect="auto",
        text_auto=".2f",
        labels={"x": "Topic", "y": "Transcript", "color": "Share"},
        title="Relative topic mix by transcript",
    )
    st.plotly_chart(heatmap, use_container_width=True)

    long_df = source_topic.reset_index().melt(
        id_vars="source",
        var_name="topic",
        value_name="share",
    )
    stacked = px.bar(
        long_df,
        x="source",
        y="share",
        color="topic",
        title="Topic composition by transcript",
    )
    stacked.update_layout(barmode="stack")
    st.plotly_chart(stacked, use_container_width=True)

with tab4:
    st.subheader("Chunk explorer")
    topic_filter = st.multiselect(
        "Filter topics",
        options=topic_df["topic"].tolist(),
        default=topic_df["topic"].tolist(),
        format_func=lambda t: f"T{t}",
    )
    source_filter = st.multiselect(
        "Filter transcripts",
        options=sorted(corpus_out["source"].unique()),
        default=sorted(corpus_out["source"].unique()),
    )

    filtered = corpus_out[
        corpus_out["topic"].isin(topic_filter)
        & corpus_out["source"].isin(source_filter)
    ].copy()

    st.dataframe(
        filtered[
            [
                "source",
                "chunk_id",
                "topic",
                "topic_label",
                "topic_confidence",
                "word_count",
                "text",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "topic_confidence": st.column_config.ProgressColumn(
                "Dominance",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            )
        },
    )

st.divider()
st.subheader("Export")

topic_export = topic_df[["topic", "label", "top_terms"]].copy()
topic_export["prevalence"] = prevalence

c1, c2 = st.columns(2)
c1.download_button(
    "Download topics CSV",
    data=topic_export.to_csv(index=False).encode("utf-8"),
    file_name="standup_topics.csv",
    mime="text/csv",
)
c2.download_button(
    "Download chunk assignments CSV",
    data=corpus_out.to_csv(index=False).encode("utf-8"),
    file_name="standup_chunk_topics.csv",
    mime="text/csv",
)

with st.expander("How to read this without fooling yourself"):
    st.markdown(
        """
- **Topics are latent term bundles, not truth.** The three-word labels are shorthand.
- **Change the topic count.** If the same conceptual groups survive across 6, 8, and 10 topics, that's a good sign.
- **Read the representative chunks.** A coherent-looking term list can still be nonsense in context.
- **Use multiple comedians/specials.** One transcript can produce "topics" that are really just long-running bits.
- **Keep an eye on filler.** Add recurring verbal tics to the stopword box if they dominate.
- **Bigrams help comedy.** Phrases like "high school", "my wife", or "New York" are often more informative than isolated words.
        """
    )
