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

def load_synsql(output_path):
    # make sure to clone https://huggingface.co/datasets/seeklhy/SynSQL-2.5M/tree/main first
    import pyarrow as pa
    import pyarrow.parquet as pq

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

    # Stream JSON → process → write parquet in chunks.
    # Never holds more than CHUNK_SIZE processed rows in memory.
    import os
    import time

    CHUNK_SIZE = 10_000
    data_file = "SynSQL-2.5M/data.json"
    file_size = os.path.getsize(data_file)

    chunk: list[dict] = []
    writer = None
    n_total = 0
    t_start = time.time()
    t_last_log = t_start

    for idx, (example, bytes_read) in enumerate(
        _iter_json_array(data_file)
    ):
        chunk.append(process_fn(example, idx))
        if len(chunk) >= CHUNK_SIZE:
            table = pa.Table.from_pylist(chunk)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            n_total += len(chunk)
            chunk.clear()

            now = time.time()
            if now - t_last_log >= 2.0:  # log every 2 seconds
                elapsed = now - t_start
                pct = bytes_read / file_size * 100
                rate = n_total / elapsed
                eta = (file_size - bytes_read) / (bytes_read / elapsed) if bytes_read else 0
                print(
                    f"  {n_total:,} rows | {pct:.1f}% of file | "
                    f"{rate:,.0f} rows/s | ETA {eta:.0f}s",
                    flush=True,
                )
                t_last_log = now

    if chunk:
        table = pa.Table.from_pylist(chunk)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
        n_total += len(chunk)

    if writer is not None:
        writer.close()

    elapsed = time.time() - t_start
    print(f"Done — {n_total:,} rows written to {output_path} in {elapsed:.1f}s")
    return datasets.load_dataset("parquet", data_files=output_path, split="train")


def _iter_json_array(filepath, buf_size=8 * 1024):
    """Stream items from a JSON array file without loading it into memory.

    Yields ``(object, bytes_consumed)`` tuples so callers can report progress
    against the known file size.
    """
    decoder = json.JSONDecoder()
    with open(filepath, "r", encoding="utf-8") as f:
        bytes_consumed = 0

        # --- locate opening '[' ---
        text = ""
        while True:
            chunk = f.read(buf_size)
            if not chunk:
                return
            bytes_consumed += len(chunk)
            text += chunk
            pos = text.find("[")
            if pos != -1:
                text = text[pos + 1 :]
                break

        # --- yield one object at a time ---
        while True:
            text = text.lstrip(" \t\n\r,")
            if not text:
                chunk = f.read(buf_size)
                if not chunk:
                    return
                bytes_consumed += len(chunk)
                text = chunk.lstrip(" \t\n\r,")
                if not text:
                    continue

            if text[0] == "]":
                return

            try:
                obj, end = decoder.raw_decode(text)
            except json.JSONDecodeError:
                chunk = f.read(buf_size)
                if not chunk:
                    return
                bytes_consumed += len(chunk)
                text += chunk
                continue

            text = text[end:]
            yield obj, bytes_consumed

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and process datasets for training.")
    parser.add_argument("--output_path", type=str, default="synsql_dataset.parquet", help="Path to save the processed dataset")
    args = parser.parse_args()
    
    load_synsql(args.output_path)
