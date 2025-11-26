import os

import pandas as pd
from litellm import completion

os.environ["LM_STUDIO_API_BASE"] = "http://localhost:1234/v1"
os.environ["LM_STUDIO_API_KEY"] = "lm-studio"

# Step 1: Raw input markdown (from your example)
# Step 1: Load Markdown file
with open("phonepe-statement.md", "r", encoding="utf-8") as f:
    raw_markdown = f.read()

# Step 2: Prompt to clean and structure it
prompt = f"""
You are given a transaction statement in Markdown format. 
Extract all transactions into a clean JSON array with fields:

- date (string, ISO format preferred)
- details (string)
- type (Debit/Credit)
- amount (number in INR)

Only return valid JSON (no explanations, no markdown).
Here is the statement:
{raw_markdown}
"""

# Step 3: Call the local LLM (LM Studio, Ollama, etc.)
response = completion(
    model="lm_studio/qwen/qwen3-4b-2507",  # ⚠️ choose your local model
    messages=[{"role": "user", "content": prompt}],
)

# Step 4: Parse the model output (JSON) into a DataFrame
import json

raw_output = response["choices"][0]["message"]["content"]

# Step 5: Save the JSON response as a file
with open("transactions.json", "w", encoding="utf-8") as f:
    f.write(raw_output)


try:
    structured_data = json.loads(raw_output)
    df = pd.DataFrame(structured_data)
    print(df.head())  # preview
    df.to_csv("transactions.csv", index=False)
    df.to_excel("transactions.xlsx", index=False)
except Exception as e:
    print("Error parsing LLM output:", e)
    print("Raw output was:", response["choices"][0]["message"]["content"])
