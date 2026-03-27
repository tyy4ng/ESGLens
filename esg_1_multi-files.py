#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-process ESG/sustainability report PDFs and extract structured information
using an LLM (GPT-4o-mini).

For each PDF in the report directory, the script:
    1. Parses the PDF and builds a FAISS vector store (via ESGReport).
    2. Retrieves the most relevant text chunks for each ESG query.
    3. Calls the LLM to extract content, tone, and numerical data in JSON format.
    4. Maps chunk indices back to actual PDF page numbers.
    5. Saves three CSV files per report: summary_context, summary_tone, summary_numeric.

Output structure (under SUMMARY_DIR):
    <SUMMARY_DIR>/
        <report_name>/
            summary_context.csv   — key content points with source pages
            summary_tone.csv      — tone analysis (Positive / Neutral / Negative)
            summary_numeric.csv   — extracted numerical values

Requirements:
    - langchain
    - openai
    - faiss-cpu
    - esg_documents (local module)
"""

import os
import json
import warnings
import pandas as pd

from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

from esg_documents import ESGReport

warnings.filterwarnings('ignore')


# =============================================================================
# Settings
# =============================================================================
REPORT_DIR           = './esg_report'
DESTINATION_DIR      = 'esg_summary/pdf'             # Not used
VECTOR_DB_DIR        = 'esg_summary/vector_db'
RETRIEVED_CHUNKS_DIR = 'esg_summary/retrieved_chunks'
SUMMARY_DIR          = 'esg_summary/summary_df'

LLM_NAME  = 'gpt-4o-mini'
TOP_K     = 20
MAX_TOKENS = 1024


# =============================================================================
# ESG queries
# =============================================================================
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

SYSTEM_PROMPT = (
    "You are an AI assistant in the role of a Senior Equity Analyst with expertise "
    "in climate science that analyzes companies' sustainability reports."
)

GRI_PROMPT_TEMPLATE = """
You are an ESG (Environmental, Social, and Governance) expert assigned to analyze a company's sustainability report. Your task is to analyze the extracted parts of the report and answer the given QUESTIONS. If the information required to answer a question is not present in the provided context, respond with: "No data available." Do not fabricate answers.

Your response must follow these requirements:
1. **Format your answer in JSON** with three structured sections: `ContentExtraction`, `ToneAnalysis`, and `NumericalData`.
2. Each section must list **exactly one point**, including its corresponding `source`, `tone`, and (if applicable) `value`.
3. **Tone** must be classified as one of: `Neutral`, `Positive`, or `Negative`.
4. **Source** must be an integer and represent the associated reference number for the information.

### **JSON Output Format**
Your FINAL_ANSWER should be structured as follows:
{{
    "ContentExtraction": [
        {{
            "point": "The company aims to reduce carbon emissions by 30% by 2025.",
            "tone": "Positive",
            "value": "Null",
            "source": 8
        }}
    ],
    "ToneAnalysis": [
        {{
            "point": "The tone is Positive, emphasizing future goals and achievements.",
            "tone": "Positive",
            "value": "Null",
            "source": 8
        }}
    ],
    "NumericalData": [
        {{
            "point": "Energy consumption in 2023 was 810 GW, including electricity, natural gas, and renewable sources.",
            "tone": "Neutral",
            "value": 810,
            "source": 12
        }}
    ]
}}

### **Handling Missing Data**
If relevant data is not available for a specific section, use the following structure:
{{
    "ContentExtraction": [
        {{
            "point": "No data available.",
            "tone": "Neutral",
            "value": "Null",
            "source": 0
        }}
    ],
    "ToneAnalysis": [
        {{
            "point": "No data available.",
            "tone": "Neutral",
            "value": "Null",
            "source": 0
        }}
    ],
    "NumericalData": [
        {{
            "point": "No data available.",
            "tone": "Neutral",
            "value": "Null",
            "source": 0
        }}
    ]
}}

### **Instructions**
Now, using the provided {{context}} and {{question}}, answer the question while adhering to the JSON format above. Ensure each section is labeled and includes all required fields (`point`, `tone`, `value`, `source`).
"""


# =============================================================================
# Helper functions
# =============================================================================
def docs_to_string(docs, num_docs=TOP_K):
    """
    Convert a list of LangChain Document objects into a formatted string
    with content and source index for use in LLM prompts.

    Args:
        docs     (list): List of Document objects returned by the retriever.
        num_docs (int):  Maximum number of documents to include.

    Returns:
        str: Formatted string with alternating content and source blocks.
    """
    output = ''
    for doc in docs[:num_docs]:
        output += f'Content: {doc.page_content}\n'
        output += f'Source: {doc.metadata["source"]}\n'
        output += '\n---\n'
    return output


def replace_sources(data, source_list):
    """
    Replace chunk index references in LLM output with actual PDF page numbers.

    Args:
        data        (list): List of dicts from the LLM JSON output, each with a 'source' key.
        source_list (list): Ordered list of page numbers corresponding to chunk indices.

    Returns:
        list: Same list with 'source' values replaced by page numbers.
              Falls back to -1 if the index is out of range.
    """
    for item in data:
        try:
            index = int(item['source'])
            item['source'] = source_list[index]
        except (ValueError, IndexError):
            item['source'] = -1
    return data


def json_to_df(section_dict):
    """
    Flatten a per-category dict of LLM output records into a DataFrame.

    Args:
        section_dict (dict): Keys are query names (e.g. 'emissions_1'),
                             values are lists of dicts with keys
                             'point', 'tone', 'value', 'source'.

    Returns:
        pd.DataFrame: Columns — category, point, tone, value, source.
    """
    rows = []
    for category, records in section_dict.items():
        for record in records:
            rows.append({
                'category': category,
                'point':    record['point'],
                'tone':     record['tone'],
                'value':    record['value'],
                'source':   record['source'],
            })
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    report_paths = [
        os.path.join(REPORT_DIR, f)
        for f in os.listdir(REPORT_DIR)
        if f.endswith('.pdf')
    ]
    print(f'Found {len(report_paths)} PDF reports in {REPORT_DIR}')

    llm        = ChatOpenAI(model=LLM_NAME, temperature=0, max_tokens=MAX_TOKENS)
    gri_prompt = PromptTemplate(
        template=GRI_PROMPT_TEMPLATE,
        input_variables=['context', 'question']
    )

    failed_files = []

    for pdf_path in report_paths:
        report_name = os.path.basename(pdf_path).replace('.pdf', '')
        print(f'\nProcessing: {report_name}')

        try:
            # --- Parse PDF and build vector store ---
            report = ESGReport(
                path=pdf_path,
                url=None,
                store_path=os.path.join(DESTINATION_DIR, report_name + '.pdf'),
                db_path=os.path.join(VECTOR_DB_DIR, report_name),
                retrieved_chunks_path=os.path.join(RETRIEVED_CHUNKS_DIR, report_name),
            )

            # --- Query LLM for each ESG topic ---
            contexts = {}
            tones    = {}
            numerics = {}

            for key, question in QUERIES.items():
                context_str = docs_to_string(report.section_text_dict[key])

                message = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=gri_prompt.format(
                        context=context_str,
                        question=question,
                    )),
                ]

                output_text = llm(message).content
                data        = json.loads(output_text)

                contexts[key] = replace_sources(data['ContentExtraction'], report.page_idx)
                tones[key]    = replace_sources(data['ToneAnalysis'],      report.page_idx)
                numerics[key] = replace_sources(data['NumericalData'],     report.page_idx)

            # --- Convert to DataFrames ---
            df_contexts = json_to_df(contexts)
            df_tones    = json_to_df(tones)
            df_numerics = json_to_df(numerics)

            # --- Save CSVs ---
            output_dir = os.path.join(SUMMARY_DIR, report_name)
            os.makedirs(output_dir, exist_ok=True)

            df_contexts.to_csv(os.path.join(output_dir, 'summary_context.csv'), index=False)
            df_tones.to_csv(   os.path.join(output_dir, 'summary_tone.csv'),    index=False)
            df_numerics.to_csv(os.path.join(output_dir, 'summary_numeric.csv'), index=False)

            print(f'  [OK] Saved to {output_dir}')

        except Exception as e:
            print(f'  [FAIL] {report_name}: {e}')
            failed_files.append(pdf_path)

    # --- Summary ---
    print(f'\nDone. {len(report_paths) - len(failed_files)}/{len(report_paths)} reports processed.')
    if failed_files:
        print(f'Failed files ({len(failed_files)}):')
        for f in failed_files:
            print(f'  {f}')
