#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core document processing module for ESG report analysis.

Provides the ESGReport class, which handles:
    - PDF parsing with PyMuPDF (fitz)
    - Text chunking with LangChain's RecursiveCharacterTextSplitter
    - Vector store creation and persistence using FAISS + OpenAI Embeddings
    - Semantic retrieval of relevant chunks for each ESG query
    - Saving retrieved chunks to JSON for audit / reproducibility

Usage:
    from esg_documents_clean import ESGReport

    report = ESGReport(
        path='./esg_reports/NYSE_AAPL_2022.pdf',
        db_path='esg_summary/vector_db/NYSE_AAPL_2022',
        retrieved_chunks_path='esg_summary/retrieved_chunks/NYSE_AAPL_2022',
    )
    # report.section_text_dict contains retrieved chunks for each query key

Requirements:
    - PyMuPDF (fitz)
    - langchain
    - faiss-cpu
    - openai
    - Set OPENAI_API_KEY as an environment variable before running.
"""

import os
import re
import json
import time

import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings


# ---------------------------------------------------------------------------
# OpenAI API key
# Set OPENAI_API_KEY in your environment or in a .env file.
# Do NOT hard-code your key here.
# ---------------------------------------------------------------------------
if 'OPENAI_API_KEY' not in os.environ:
    raise EnvironmentError(
        'OPENAI_API_KEY is not set. '
        'Export it in your shell or add it to a .env file before running.'
    )

# =============================================================================
# Configuration
# =============================================================================
TOP_K         = 20
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 20

# ESG queries used for semantic retrieval.
# Each key maps to the question sent to the vector store retriever.
QUERIES = {
    'basic_1':        'What sector does the company belong to?',
    'emissions_1':    'What are the greenhouse gas emissions for Scope 1, Scope 2, and Scope 3?',
    'emissions_2':    'What is the carbon intensity (emissions per unit of product or revenue)?',
    'emissions_3':    'What are the carbon reduction targets, and what progress has been made?',
    'emissions_4':    'How are air pollutants, such as nitrogen oxides and sulfur oxides, managed?',
    'resource_1':     'What is the total energy consumption, and what proportion of it comes from renewable sources?',
    'resource_2':     'How is water used, including total water consumption and the recycling rate?',
    'resource_3':     'How efficient is the use of raw materials, and what recycling initiatives are in place?',
    'resource_4':     'How is waste managed, and what are the methods and recycling targets?',
    'environment_1':  'What green products or services does the company offer?',
    'environment_2':  'How much investment has been made in environmental innovation or R&D? What projects are included?',
    'environment_3':  'Is the company participating in initiatives promoting the circular economy?',
    'biodiversity_1': 'What policies are in place to protect biodiversity or natural habitats?',
    'biodiversity_2': 'Does the company participate in biodiversity conservation projects? If so, what are they?',
    'biodiversity_3': 'What is the impact of the supply chain on land use and the environment?',
    'ems_1':          'Does the company have publicly available environmental policies? If so, what are they?',
    'ems_2':          'Has the company obtained certifications such as ISO 14001 or similar? If so, which ones?',
    'ems_3':          'How transparent is the company regarding environmental data and goals?',
}


# =============================================================================
# ESGReport class
# =============================================================================
class ESGReport:
    """
    Parse a PDF ESG report, build a FAISS vector store, and retrieve the most
    relevant text chunks for each predefined ESG query.

    Attributes:
        section_text_dict (dict): Maps each query key to a list of retrieved
                                  LangChain Document objects.
        page_idx          (list): Maps each chunk index to its source PDF page number.
        title             (str):  Document title inferred from the largest font text.

    Args:
        path                  (str):  Local path to the PDF file.
        url                   (str):  URL of the PDF (mutually exclusive with path).
        store_path            (str):  Destination path to copy the PDF (optional).
        top_k                 (int):  Number of chunks to retrieve per query.
        db_path               (str):  Directory for saving/loading the FAISS index.
        retrieved_chunks_path (str):  Directory for saving retrieved chunks as JSON.
    """

    def __init__(
        self,
        path=None,
        url=None,
        store_path=None,
        top_k=TOP_K,
        db_path=None,
        retrieved_chunks_path=None,
    ):
        assert (path is None) != (url is None), \
            'Provide exactly one of path or url, not both.'

        self.path                  = path
        self.url                   = url
        self.store_path            = store_path
        self.top_k                 = top_k
        self.db_path               = db_path
        self.retrieved_chunks_path = retrieved_chunks_path
        self.queries               = QUERIES
        self.chunks                = []
        self.page_idx              = []

        self.pdf   = fitz.open(self.path)
        self.title = self.get_title()
        self.parse_pdf()

    # -------------------------------------------------------------------------
    def parse_pdf(self):
        """
        Extract text from the PDF, build the vector store, run all queries,
        and save the retrieved chunks to JSON.
        """
        self.pdf       = fitz.open(self.path)
        self.text_list = [page.get_text() for page in self.pdf]
        self.all_text  = ' '.join(self.text_list)

        start_time = time.time()

        self.retriever, self.vector_db = self._get_retriever(self.db_path)
        self.section_text_dict         = self._retrieve_chunks()

        elapsed = time.time() - start_time
        print(f'  Retrieval completed in {elapsed:.1f}s')

        # Save retrieved chunks to JSON for audit / reproducibility
        os.makedirs(self.retrieved_chunks_path, exist_ok=True)
        retrieved_json_path = os.path.join(self.retrieved_chunks_path, 'retrieved.json')
        with open(retrieved_json_path, 'w') as f:
            to_dump = {
                key: {
                    chunk.metadata['source']: [chunk.page_content, chunk.metadata['page']]
                    for chunk in chunk_list
                }
                for key, chunk_list in self.section_text_dict.items()
            }
            json.dump(to_dump, f)

        self.section_text_dict['title'] = self.title
        self.pdf.close()

    # -------------------------------------------------------------------------
    def get_title(self):
        """
        Infer the document title by finding text rendered in the largest font size.

        Scans all pages and collects font sizes. Strings whose font size matches
        the top-2 largest sizes and is longer than 4 characters are concatenated
        to form the title.

        Returns:
            str: Inferred document title.
        """
        doc            = self.pdf
        max_font_sizes = [0]

        # First pass: collect all font sizes
        for page in doc:
            for block in page.get_text('dict')['blocks']:
                if block['type'] == 0 and block['lines'] and block['lines'][0]['spans']:
                    font_size = block['lines'][0]['spans'][0]['size']
                    max_font_sizes.append(font_size)

        max_font_sizes.sort()
        top_sizes = {max_font_sizes[-1], max_font_sizes[-2]}

        # Second pass: collect strings at the largest font sizes
        title_parts = []
        for page in doc:
            for block in page.get_text('dict')['blocks']:
                if block['type'] == 0 and block['lines'] and block['lines'][0]['spans']:
                    span      = block['lines'][0]['spans'][0]
                    font_size = span['size']
                    text      = span['text']
                    if any(abs(font_size - s) < 0.3 for s in top_sizes) and len(text) > 4:
                        title_parts.append(text)

        return ' '.join(title_parts).replace('\n', ' ')

    # -------------------------------------------------------------------------
    def _get_retriever(self, db_path):
        """
        Build a FAISS vector store from the PDF text and return a retriever.

        Each page is split into chunks; chunk-to-page mappings are stored in
        self.page_idx so that source references can later be resolved to page
        numbers.

        Args:
            db_path (str): Directory where the FAISS index will be saved.

        Returns:
            tuple: (retriever, FAISS vector store)
        """
        embeddings = OpenAIEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=['\n\n', '\n', ' '],
        )

        for i, page in enumerate(self.pdf):
            page_chunks = text_splitter.split_text(page.get_text())
            self.page_idx.extend([i + 1] * len(page_chunks))
            self.chunks.extend(page_chunks)

        doc_search = FAISS.from_texts(
            self.chunks,
            embeddings,
            metadatas=[
                {'source': str(i), 'page': str(page_num)}
                for i, page_num in enumerate(self.page_idx)
            ],
        )
        doc_search.save_local(db_path)

        retriever = doc_search.as_retriever(search_kwargs={'k': self.top_k})
        return retriever, doc_search

    # -------------------------------------------------------------------------
    def _retrieve_chunks(self):
        """
        Run semantic search for every query in QUERIES and collect results.

        Returns:
            dict: Maps each query key to a list of retrieved Document objects.
        """
        return {
            key: self.retriever.get_relevant_documents(query)
            for key, query in self.queries.items()
        }


# =============================================================================
# Utility
# =============================================================================
def search_page(content, search_list):
    """
    Check whether any keyword from search_list appears in the given text.

    Args:
        content     (str):  Text to search within.
        search_list (list): List of keyword strings.

    Returns:
        bool: True if at least one keyword is found (case-insensitive).
    """
    return bool(re.search('|'.join(search_list), content.lower()))
