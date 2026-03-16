"""Prompts for adaptive filtering-based generation (online retrieval)."""

FILTER_SYSTEM = (
    "You are a helpful assistant responding to questions about data "
    "in the tables provided."
)

FILTER_PROMPT = """\
# Role
You are a helpful assistant responding to questions about data in the tables provided.

# Goal
Generate a response consisting of a list of key points that respond to the \
user's question, summarizing all relevant information in the input data tables.
You should use the data provided in the data tables below as the primary context \
for generating the response.
If you don't know the answer or if the input data tables do not contain \
sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important \
the point is in answering the user's question. An 'I don't know' type of \
response should have a score of 0.

The response should be JSON formatted as follows:
{{"points": [
    {{
        "description": "Description of point 1",
        "score": score_value
    }}
]}}

# User Question
{question}

# Data tables
{context_data}

Output:
"""

MERGE_SYSTEM = (
    "You are a helpful assistant responding to questions and may use the "
    "provided data as a reference."
)

MERGE_PROMPT = """\
# Role
You are a helpful assistant responding to questions and may use the provided \
data as a reference.

# Goal
You should incorporate insights from all the reports from multiple analysts \
who focused on different parts of the dataset to support your answer. \
Please note that the provided information may contain inaccuracies or be unrelated. \
If the provided information does not address the question, please respond using \
what you know:
- A response that utilizes the provided information, ensuring that all irrelevant \
details from the analysts' reports are removed.
- A response to the user's query based on your existing knowledge when \
<Analyst Reports> is empty.

The final response should merge the relevant information into a comprehensive \
answer that clearly explains all key points and implications, tailored to the \
appropriate response length and format.
Do not include information where the supporting evidence for it is not provided.

# Target response length and format
{response_format}

# User Question
{question}

# Analyst Reports
{report_data}

Output:
"""
