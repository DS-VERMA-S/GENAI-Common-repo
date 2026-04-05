BASE_PROMPT = """You are a data assistant. Convert the user's question into a JSON action plan.
The plan must follow the schema exactly.

Metadata:
{metadata}

Question:
{question}

{format_instructions}
"""

RETRY_PROMPT = """The previous action plan failed validation.
Errors: {errors}

Please return a corrected JSON action plan that matches the schema.

Metadata:
{metadata}

Question:
{question}

{format_instructions}
"""

