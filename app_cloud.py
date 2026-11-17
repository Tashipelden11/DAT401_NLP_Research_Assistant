
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai
from google.genai import types


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

st.set_page_config(
    page_title="Evidence-Grounded Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Evidence-Grounded Research Assistant")
st.caption(
    "Research assistance on Generative AI in higher education, "
    "using 154 peer-reviewed research articles."
)


@st.cache_resource
def load_research_resources():

    chunks = pd.read_csv(
        RESULTS_DIR / "DAT401_Text_Chunks.csv"
    ).reset_index(drop=True)

    with open(
        RESULTS_DIR / "DAT401_Full_Corpus_BM25.pkl", "rb"
    ) as f:
        bm25_index = pickle.load(f)

    with open(
        RESULTS_DIR / "DAT401_Full_Corpus_Dense_Index.pkl", "rb"
    ) as f:
        dense_index = pickle.load(f)

    metadata = pd.read_csv(
        RESULTS_DIR / "DAT401_Verified_Article_Metadata.csv",
        low_memory=False
    )

    metadata_columns = [
        "Article_ID",
        "Crossref_Title",
        "Crossref_Year",
        "Crossref_Journal",
        "Crossref_Authors",
        "DOI"
    ]

    article_metadata = metadata[metadata_columns].copy()

    evidence_chunks = (
        chunks.drop(columns=["DOI"])
        .join(
            article_metadata.set_index("Article_ID"),
            on="Article_ID",
            validate="many_to_one"
        )
    )

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    cross_encoder = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return (
        chunks,
        evidence_chunks,
        bm25_index,
        dense_index,
        embedding_model,
        cross_encoder
    )


def retrieve_evidence(question, resources, top_k=5):

    (
        chunks,
        evidence_chunks,
        bm25_index,
        dense_index,
        embedding_model,
        cross_encoder
    ) = resources

    # BM25 retrieval
    tokens = question.lower().split()
    bm25_scores = bm25_index.get_scores(tokens)

    bm25_indices = np.argsort(
        -bm25_scores, kind="stable"
    )[:20]

    # Dense retrieval
    query_embedding = embedding_model.encode(
        question,
        convert_to_numpy=True
    )

    _, dense_indices = dense_index.kneighbors(
        query_embedding.reshape(1, -1),
        n_neighbors=20
    )

    dense_indices = dense_indices[0]

    dense_ranks = {
        index: rank
        for rank, index in enumerate(dense_indices, 1)
    }

    bm25_ranks = {
        index: rank
        for rank, index in enumerate(bm25_indices, 1)
    }

    candidates = set(dense_ranks) | set(bm25_ranks)

    # Reciprocal Rank Fusion
    hybrid_scores = {
        index:
        (1 / (60 + dense_ranks[index]) if index in dense_ranks else 0)
        +
        (1 / (60 + bm25_ranks[index]) if index in bm25_ranks else 0)
        for index in candidates
    }

    ranked_indices = sorted(
        candidates,
        key=lambda index: (-hybrid_scores[index], index)
    )[:20]

    hybrid_results = chunks.iloc[ranked_indices].copy()

    # Cross-Encoder reranking
    pairs = [
        (question, text)
        for text in hybrid_results["Chunk_Text"]
    ]

    scores = cross_encoder.predict(pairs)

    hybrid_results["Rerank_Score"] = scores

    final_results = (
        hybrid_results
        .sort_values("Rerank_Score", ascending=False)
        .head(top_k)
    )
    print("\nQUERY:", question)
    print("\nRERANKING SCORES:")

    print(
        final_results[
        ["Article_ID", "Rerank_Score", "Chunk_Text"]
         ].to_string(index=False)
    )
    
    evidence = []

    for rank, (index, row) in enumerate(
        final_results.iterrows(), 1
    ):

        metadata = evidence_chunks.loc[index]

        evidence.append({
            "Evidence_ID": f"E{rank:02d}",
            "Article_ID": row["Article_ID"],
            "Title": metadata["Crossref_Title"],
            "Authors": metadata["Crossref_Authors"],
            "Year": metadata["Crossref_Year"],
            "Journal": metadata["Crossref_Journal"],
            "DOI": metadata["DOI"],
            "Page": row["Page"],
            "Passage": row["Chunk_Text"]
        })

    return evidence


def build_context(evidence):

    sections = []

    for item in evidence:
        sections.append(
            f"""
[{item['Evidence_ID']}]
Article ID: {item['Article_ID']}
Title: {item['Title']}
Authors: {item['Authors']}
Year: {item['Year']}
Journal: {item['Journal']}
DOI: {item['DOI']}
Page: {item['Page']}

Supporting passage:
{item['Passage']}
"""
        )

    return "\n\n".join(sections)


def generate_answer(question, context, client):

    prompt = f"""
You are an evidence-grounded research assistant specialising
in Generative AI in higher education.

Answer using ONLY the supplied research evidence.

STRICT RULES:
1. Do not invent facts, statistics, findings or citations.
2. Cite factual claims using [E01, p.8] format.
3. Use only evidence IDs and page numbers supplied below.
4. Distinguish perceptions, reported capabilities,
   potential benefits and experimentally demonstrated outcomes.
5. Do not present perceptions as proof of improved academic performance.
6. If evidence is insufficient, clearly state this.
7. Provide a concise academic answer.

RESEARCH QUESTION:
{question}

AVAILABLE RESEARCH EVIDENCE:
{context}

EVIDENCE-GROUNDED ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1200
        )
    )

    if not response.text or not response.text.strip():
        raise ValueError("The language model returned an empty answer.")

    return response.text.strip()


def validate_citations(answer, evidence):

    citation_pattern = r"\[(E\d+),\s*p\.\s*(\d+)\]"
    import re

    citations = re.findall(citation_pattern, answer)

    lookup = {
        item["Evidence_ID"]: item
        for item in evidence
    }

    results = []

    for evidence_id, page in citations:

        record = lookup.get(evidence_id)

        if record is None:
            status = "Invalid evidence ID"
        elif str(record["Page"]) != page:
            status = "Page mismatch"
        else:
            status = "Valid"

        results.append({
            "Evidence": evidence_id,
            "Cited page": page,
            "Status": status
        })

    return results


# Sidebar
with st.sidebar:
    st.header("Research assistant")
    st.write("**Corpus:** 154 research articles")
    st.write("**Chunks:** 11,273")
    st.write("**Retrieval:** Hybrid RRF + Cross-Encoder")
    st.write("**Answer generation:** Gemini")

    # Read the API key securely from Streamlit Cloud Secrets.
    try:
        secret_api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_api_key = ""

    default_api_key = (
        secret_api_key or os.environ.get("GEMINI_API_KEY", "")
    )

    # Password input remains available for local testing.
    api_key = st.text_input(
        "Gemini API key",
        type="password",
        value=default_api_key,
        help="The key is stored in Streamlit Secrets or entered for this session."
    )

    st.caption(
        "Answers are grounded in the indexed research corpus. "
        "Insufficient evidence should be reported explicitly."
    )


# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask a research question about Generative AI in higher education..."
)

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        if not api_key.strip():
            st.warning("Enter your Gemini API key in the sidebar.")
            st.stop()

        try:
            with st.spinner("Loading research resources..."):
                resources = load_research_resources()

            client = genai.Client(api_key=api_key)

            with st.spinner("Searching research articles..."):
                evidence = retrieve_evidence(question, resources)

            if not evidence:
                st.warning("No supporting passages were retrieved.")
                st.stop()

            context = build_context(evidence)

            with st.spinner("Generating an evidence-grounded answer..."):
                answer = generate_answer(
                    question, context, client
                )

            st.markdown(answer)

            citation_results = validate_citations(
                answer, evidence
            )

            with st.expander("Retrieved research evidence"):
                for item in evidence:
                    st.markdown(
                        f"**{item['Evidence_ID']} — {item['Title']}**"
                    )
                    st.caption(
                        f"{item['Authors']} ({item['Year']}) | "
                        f"{item['Journal']} | Page {item['Page']}"
                    )
                    st.write(item["Passage"])

                    if pd.notna(item["DOI"]):
                        st.write(f"DOI: {item['DOI']}")

                    st.divider()

            with st.expander("Citation validation"):
                if citation_results:
                    st.dataframe(
                        pd.DataFrame(citation_results),
                        use_container_width=True
                    )
                else:
                    st.info("No formatted evidence citations were detected.")

            invalid = [
                item for item in citation_results
                if item["Status"] != "Valid"
            ]

            if invalid:
                st.warning(
                    "Some citations did not match the retrieved evidence."
                )
            elif citation_results:
                st.success("All detected citations match the retrieved evidence.")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as error:
            st.error(f"An error occurred: {error}")
            st.exception(error)


st.divider()
st.caption(
    "DAT401 Natural Language Processing Group Project | "
    "Evidence-Grounded Research Assistant"
)
