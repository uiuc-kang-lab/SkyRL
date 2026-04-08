import json
import datasets

system_prompt = """# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "execute_sql", "description": "Execute a SQL query against the database and return the results. The returned dataframe will be truncated to 50 rows if the result is too long.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The SQL query to execute against the database"}}, "required": ["query"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=execute_sql>
<parameter=query>
YOUR SQL QUERY HERE
</parameter>
</function>
</tool_call>

Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question within limited turns. You should breakdown the problem, draft your reasoning process, and generate the solution.

The task you will receive should have the following format:

Database Engine:
<<engin>>

Database Schema:
<<db_details>>
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

External Knowledge:
<<external_knowledge>>

Question:
<<question>>

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- If the external knowledge is not empty, you must use all the information provided in the external knowledge to help you generate the SQL query.
- Before generating the final SQL query, please think through the steps of how to write the query. It should include detailed considerations such as analyzing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, thinking of how to call SQL tools, and revisiting previous steps.


Format:
- Conduct thinking every time you get new observation or information.
- You can use the execute_sql tool to explore data or verify result. You will receive the execution results or error information of the SQL in a tool response. Based on this information, you can think again and refine.
- The returned dataframe will be truncated in 50 rows if observation is too long.
- Only if you find no further exploration is needed or reach max turns, you directly provide the final SQL query solution inside <solution>...</solution>.
- Do not request a SQL tool execution and provide a solution in the same response.
"""

prompt_template = """Database Engine:
{engine}

Database Schema:
{db_details}

External Knowledge:
{external_knowledge}

Question:
{question}
"""

def load_synsql():
    # make sure to clone https://huggingface.co/datasets/seeklhy/SynSQL-2.5M/tree/main first
    with open("SynSQL-2.5M/tables.json", "r") as f:
        tables = json.load(f)
    dbid2ddl = {item["db_id"]: item["ddls"] for item in tables}

    def process_fn(example, idx):
        db_details = dbid2ddl[example["db_id"]]
        prompt = prompt_template.format(
            engine="sqlite",
            db_details=db_details,
            external_knowledge=example["external_knowledge"],
            question=example["question"],
        )
        return {
            "data_source": "synsql",
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "data": "synsql",
            "env_class": "text2sql",
            "reward_spec": {"method": "sql", "ground_truth": example["sql"]},
            "extra_info": {"index": idx, "answer": example["sql"]},
            "db_id": example["db_id"],
        }

    # Load raw entries with json.load() (small per-entry footprint), then
    # use Dataset.from_generator() to stream processed entries into Arrow
    # on disk one at a time.  This avoids both the PyArrow JSON reader's
    # int32 block_size limit (>2 GB files) and holding all DDL-expanded
    # prompts in memory simultaneously.
    with open("SynSQL-2.5M/data.json", "r") as f:
        raw_data = json.load(f)

    def generate():
        for idx, example in enumerate(raw_data):
            yield process_fn(example, idx)

    processed_dataset = datasets.Dataset.from_generator(generate)
    return processed_dataset

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and process datasets for training.")
    parser.add_argument("--output_path", type=str, default="synsql_dataset.parquet", help="Path to save the processed dataset")
    args = parser.parse_args()
    
    train_data = load_synsql()
    train_data.to_parquet(args.output_path)
