"""
Streamlit page for embedding prompt/example files from `data/describe`.

The page lets users:
1) Browse supported files (`.csv`, `.xlsx`) under the `data/describe` directory/folder,
2) Preview the selected file,
3) Generate embeddings for the `user` column, and
4) Save the enriched table to a mirrored `data/embeddings` path as parquet.

Design assumptions:
- Input files contain a `user` column with text to embed.
- The output location is derived by replacing `describe` with `embeddings` in the original path.
"""

# Library imports
import streamlit as st
import pandas as pd
import tiktoken
import os
from utils.utils import normalize_text


from utils.page_components import add_common_page_elements

from classes.embeddings import Embeddings


def get_format(path):
    """
    Resolve file extension and Pandas reader for a data file.

    Args:
        path: Absolute or relative file path selected in the UI.

    Returns:
        tuple[str, callable]: The extension (e.g. `.csv`) and the matching
        pandas read function (`pd.read_csv` or `pd.read_excel`).

    Raises:
        ValueError: If the extension is not supported by this page.
    """
    file_format = "." + path.split(".")[-1]
    if file_format == ".xlsx":
        read_func = pd.read_excel
    elif file_format == ".csv":
        read_func = pd.read_csv
    else:
        raise ValueError(f"File format {file_format} not supported.")
    return file_format, read_func


def embed(file_path, embeddings):
    """
    Create embeddings for a selected file and save the result as parquet.

    Workflow:
    - Read the source table (`.csv` or `.xlsx`).
    - Filter out rows whose `user` text exceeds token limits.
    - Normalize text for common formatting artifacts.
    - Compute one embedding per `user` text.
    - Persist to the mirrored `data/embeddings` location.

    Args:
        file_path: Source file under `data/describe`.
        embeddings: `Embeddings` service instance used to call the model.

    Side effects:
        Writes a parquet file to disk and prints progress/data previews in UI.
    """
    file_format, read_func = get_format(file_path)

    df = read_func(file_path)
    # Keep folder structure mirrored between describe/ and embeddings/.
    embedding_path = file_path.replace("describe", "embeddings").replace(
        file_format, ".parquet"
    )

    st.write(f"Embedding file: {file_path}")
    # Skip rows that exceed model context/token constraints.
    tokenizer = tiktoken.get_encoding("cl100k_base")
    df["user_tokens"] = df["user"].apply(lambda x: len(tokenizer.encode(x)))
    df = df[df.user_tokens < 8192]
    df = df.drop("user_tokens", axis=1)

    # Normalize text before embedding to reduce avoidable noise.
    df["user"] = df["user"].apply(lambda x: normalize_text(x))

    # Store embeddings as strings to keep parquet serialization simple/stable.
    df["user_embedded"] = df["user"].apply(
        lambda x: str(embeddings.return_embedding(x))
    )

    # Ensure destination directory exists before writing output parquet.
    directory = os.path.dirname(embedding_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    st.write("Embedded file:")
    st.write(df)
    df.to_parquet(embedding_path, index=False)


# Shared page chrome (title/sidebar/help) used across Streamlit pages.
sidebar_container = add_common_page_elements()

st.divider()

# Embedding client used by this page's embed action.
embeddings = Embeddings()

# Discover candidate source files recursively inside data/describe.
describe_folder = "data/describe"
available_files = []

for root, dirs, files in os.walk(describe_folder):
    for file in files:
        if not file.endswith(".DS_Store"):
            # Keep relative path for cleaner dropdown display.
            rel_path = os.path.relpath(os.path.join(root, file), describe_folder)
            available_files.append(rel_path)

if not available_files:
    st.warning("No files found in data/describe folder")
else:
    # File selection and preview.
    selected_file = st.selectbox(
        "Select a file to embed",
        options=available_files,
        index=0
    )

    # Rebuild absolute path from selected relative path.
    full_path = os.path.join(describe_folder, selected_file)
    st.info(f"Selected file: {full_path}")

    # Read and display current source content for inspection.
    try:
        file_format, read_func = get_format(full_path)
        df = read_func(full_path)
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

    # If output parquet already exists, switch CTA text to "Re-embed".
    try:
        embedding_path = full_path.replace("describe", "embeddings").replace(
            file_format, ".parquet"
        )
        is_embedded = os.path.exists(embedding_path)
    except:
        is_embedded = False
    
    # Embedding action (idempotent: can overwrite existing parquet).
    button_label = "Re-embed File" if is_embedded else "Embed File"
    if st.button(button_label, type="primary"):
        st.write(f"Starting to embed {selected_file}...")
        try:
            embed(full_path, embeddings)
            st.success(f"Successfully embedded {selected_file}!")
        except Exception as e:
            st.error(f"Error embedding file: {str(e)}")
