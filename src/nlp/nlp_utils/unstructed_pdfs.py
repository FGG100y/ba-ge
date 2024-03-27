"""load pdfs and extract texts and tables.

Supported tables and texts chunks seperated by `unstructured` (which using
table_transformers from Microsoft)
"""

import glob
import os
from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf


def load_and_split(directory_path, fext="pdf"):
    files = load_file(directory_path, fext=fext)
    return split_block(files) 


def split_block(files):
    docs = []
    tables = []
    for pdf_file in files:
        table_elements, text_elements = get_elements(pdf_file)
        docs.append(text_elements)
        tables.append(table_elements)

    return docs, tables


def load_file(directory_path, fext="pdf"):
    pdfs = [
        pdf_file
        for pdf_file in glob.glob(os.path.join(directory_path, f"*.{fext}"))
    ]
    return pdfs


class Element(BaseModel):
    type: str
    text: Any


def get_elements(filename="LLAMA2.pdf"):
    """ """
    raw_pdf_elements = get_raw_elements(filename)
    # Categorize by type
    categorized_elements = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(
                Element(type="table", text=str(element))
            )
        elif "unstructured.documents.elements.CompositeElement" in str(
            type(element)
        ):
            categorized_elements.append(
                Element(type="text", text=str(element))
            )

    # Tables
    table_elements = [e for e in categorized_elements if e.type == "table"]
    print(len(table_elements))

    # Text
    text_elements = [e for e in categorized_elements if e.type == "text"]
    print(len(text_elements))

    return table_elements, text_elements


def get_raw_elements(filename):
    """ """
    raw_pdf_elements = partition_pdf(
        filename=filename,
        # Unstructured first finds embedded image blocks
        extract_images_in_pdf=False,
        # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
        # Titles are any sub-section of the document
        infer_table_structure=True,
        # Post processing to aggregate text once we have the title
        chunking_strategy="by_title",
        # Chunking params to aggregate text blocks
        # Attempt to create a new chunk 3800 chars
        # Attempt to keep chunks > 2000 chars
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=directory_path,
    )

    # Create a dictionary to store counts of each type
    category_counts = {}

    for element in raw_pdf_elements:
        category = str(type(element))
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1

    print(category_counts)

    #  # Unique_categories will have unique elements
    #  unique_categories = set(category_counts.keys())

    return raw_pdf_elements


if __name__ == "__main__":
    #  os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    directory_path = "data/pdfs/en"
    docs, tables = load_and_split(directory_path)
    # docs -> List; docs[0][0] -> Element Obj
    print(docs[0][0].text)  # -> str
