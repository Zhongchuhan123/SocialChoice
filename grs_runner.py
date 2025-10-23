"""
GRS pipeline for wide-format input (one row = user, each column = item rating).
Assumes structure: user_id, item_1, item_2, ..., groupId
"""

import os
import json
import pandas as pd
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# ====== CONFIG ======
LOCAL_MODEL_PATH = "./dataroot/models/Qwen/Qwen3-8B-Base"  # path to downloaded Qwen model
CSV_PATH = "./groups/groups_[2, 4, 8]members_5items_totalgroups250.csv"
OUTPUT_CSV = "./results/grs_qwen_results.csv"
AGG_STRATEGIES = ["ADD", "APP", "LMS", "MPL"]
TOP_K = 10
APP_THRESHOLD = 6
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

PROMPT_TEMPLATE = """
You are an expert in making group recommendations based on a table of ratings presented below. 
That information includes users (user_ids) and information on which items they like (item_x). The rating is a scale from 0 to 10. 
You recommend an item to the group. For the recommendation, you simply mention the item name. 

The table is found below:
{desc}

Aggregation strategy:
{strat}

Only reture the recommendation in the following JSON format (maximum 10 items; do not give explanations):
```json
{{
  "strategy": "{strategy}",
  "recommendation": ["item_1", "item_2"]
}}
"""

STRATEGY_TEXTS = {
    'ADD': 'ADD sums all ratings per item and recommend the item with the highest sum (Senot et al. 2010). For example, the first item has a 4 rating and 5 rating (sum=9). The second item has a 6 and 7 rating (sum=13). Recommend the second item because its sum is higher than the sum of the first item. Use ADD to refer to this strategy.',
    'APP': 'APP is a majority-based strategy. A predefined threshold is set at 6. For each item, you count the number of times it has been rated above 6. Recommend the item which has been rated above the threshold the most. For example, the first item has a rating of 7 and 8. The second item has a rating of 9 and 5. Recommend the first item because it has more ratings above 6. Use APP to refer to this strategy.',
    'LMS': 'LMS recommends the item which has the highest rating if you only take the lowest rating per item into account (Senot et al. 2010). For example, the first item has a rating of 5 and 6. The second item has a rating of 2 and 9. Recommend the first item because its lowest rating (5) is higher than the lowest rating of the second item (2). Use LMS to refer to this strategy.',
    'MPL': 'MPL recommends the item with the highest single rating across all relevant individuals (Senot et al. 2010). For example, the first item has a rating of 5 and 6. The second item has a rating of 2 and 9. Recommend the second item because 9 is the highest rating across all items. Use MPL to refer to this strategy.'
}


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
    model.eval()
    return tokenizer, model


def generate_response(tokenizer, model, prompt, max_new_tokens=1024):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id,)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def parse_response(text: str):
    """
    从模型输出中稳健提取最后一个 JSON。
    - 自动忽略前面的 prompt 重复、示例 JSON、代码块符号等。
    - 只解析最后一个 {...}，若失败返回空列表。
    """
    # 去掉 ```json ... ``` 等 markdown 代码块符号
    text = re.sub(r"```(json)?", "", text).strip()

    # 匹配所有 JSON 对象（以 { 开头，以 } 结尾）
    matches = re.findall(r"\{[^\{\}]*\}", text, flags=re.DOTALL)

    if not matches:
        return []

    for block in reversed(matches):  # 从最后一个开始尝试
        try:
            obj = json.loads(block)
            if isinstance(obj, dict) and "recommendation" in obj:
                rec = obj["recommendation"]
                if isinstance(rec, list):
                    return rec
        except json.JSONDecodeError:
            continue

    return []


def aggregate_items(df_long: pd.DataFrame, strategy: str, top_k: int = 10, threshold: int = 6) -> List[str]:
    if strategy == "ADD":
        scores = df_long.groupby("item")["rating"].sum()
    elif strategy == "APP":
        scores = df_long[df_long["rating"] > threshold].groupby("item").size()
    elif strategy == "LMS":
        scores = df_long.groupby("item")["rating"].min()
    elif strategy == "MPL":
        scores = df_long.groupby("item")["rating"].max()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    ranked = scores.sort_values(ascending=False)
    return list(ranked.head(top_k).index)


def format_table_prompt(df_matrix: pd.DataFrame) -> str:
    return df_matrix.drop(columns=["groupId"]).to_string(index=False)


def melt_ratings(df_matrix: pd.DataFrame) -> pd.DataFrame:
    user_col = "user_id"
    item_cols = [col for col in df_matrix.columns if col.startswith("item_")]
    df_long = df_matrix.melt(id_vars=[user_col, "groupId"], value_vars=item_cols,
                             var_name="item", value_name="rating")
    return df_long


def run():
    tokenizer, model = load_model(LOCAL_MODEL_PATH)
    df = pd.read_csv(CSV_PATH)
    results = []

    for gid, df_group in df.groupby("groupId"):
        df_group = df_group.reset_index(drop=True)
        df_long = melt_ratings(df_group)
        group_size = len(df_group)
        item_size = len(df_group.columns) - 2

        for strategy in AGG_STRATEGIES:
            prompt_table = format_table_prompt(df_group)
            strat_text = STRATEGY_TEXTS[strategy]
            prompt = PROMPT_TEMPLATE.format(desc=prompt_table, strat=strat_text, strategy=strategy)

            llm_output = generate_response(tokenizer, model, prompt)
            llm_recommend = parse_response(llm_output)
            gold = aggregate_items(df_long, strategy, top_k=TOP_K, threshold=APP_THRESHOLD)

            results.append({
                "groupId": gid,
                "strategy": strategy,
                "group_size": group_size,
                "item_size": item_size,
                "llm_recommendation": llm_recommend,
                "gold_recommendation": gold
            })

        print(f"Group ID {gid} completed.")

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved results to {OUTPUT_CSV}")


if __name__ == "__main__":
    run()
