#!/usr/bin/env python3
import pandas as pd
import numpy as np
import json
import argparse
import os


def convert_numpy_to_python(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, np.ndarray):
        return [convert_numpy_to_python(x) for x in obj.tolist()]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_to_python(x) for x in obj]
    if isinstance(obj, tuple):
        return [convert_numpy_to_python(x) for x in obj]
    return obj


def safe_json_parse(val):
    """Parse JSON string to Python object; return original if parsing fails."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val
    return val


def fix_data_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the data format for training:
    - extra_info: JSON string -> dict
    - reward_model: JSON string -> dict
    - prompt: JSON string -> list (chat messages)
    - Convert numpy types to native Python types
    """
    print("[fix] Converting JSON string fields to Python objects...")

    json_fields = ["extra_info", "reward_model", "prompt"]

    for field in json_fields:
        if field not in df.columns:
            continue

        print(f"  Processing '{field}'...")
        first_val = df[field].iloc[0] if len(df) > 0 else None

        if isinstance(first_val, str):
            df[field] = df[field].apply(safe_json_parse)
            print(f"    '{field}': parsed JSON strings")
        else:
            print(f"    '{field}': already non-string (skip JSON parsing)")

        print(f"    '{field}': converting numpy types to native Python types...")
        df[field] = df[field].apply(convert_numpy_to_python)

    return df


def fix_nested_fields_in_parquet(input_file: str, output_file: str = None):
    """
    Fix extra_info / reward_model / prompt in a parquet file.

    Args:
        input_file: input parquet path
        output_file: output parquet path; if None, overwrite input_file
    """
    print(f"[load] Input: {input_file}")

    df = pd.read_parquet(input_file)
    print(f"[load] Rows: {len(df)}")
    print(f"[load] Columns: {df.columns.tolist()}")

    df = fix_data_format(df)

    if output_file is None:
        output_file = input_file
        print(f"[save] Overwriting: {output_file}")
    else:
        print(f"[save] Writing to: {output_file}")

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("[save] Saving with datasets to preserve nested structures...")
    import datasets

    dataset = datasets.Dataset.from_pandas(df, preserve_index=False)
    dataset.to_parquet(output_file)
    print(f"[save] Wrote: {output_file}")

    print("\n[verify] Loading saved parquet with datasets...")
    try:
        ds_check = datasets.load_dataset("parquet", data_files=output_file, split="train")
        print("[verify] Loaded successfully")
        print(f"[verify] Rows: {len(ds_check)}")

        if len(ds_check) > 0:
            sample = ds_check[0]
            sample_extra_info = sample.get("extra_info")
            sample_prompt = sample.get("prompt")

            print(f"[verify] extra_info type: {type(sample_extra_info).__name__}")
            if isinstance(sample_extra_info, dict):
                keys = list(sample_extra_info.keys())[:5]
                print(f"[verify] extra_info keys (first 5): {keys}")
                try:
                    idx_val = sample_extra_info.get("index", 0)
                    print(f"[verify] extra_info['index'] accessible: {idx_val}")
                    if "question" in sample_extra_info:
                        q = sample_extra_info["question"]
                        q_show = q[:50] + "..." if isinstance(q, str) and len(q) > 50 else q
                        print(f"[verify] extra_info['question'] accessible: {q_show}")
                except Exception as e:
                    print(f"[verify] extra_info access failed: {e}")
            else:
                print("[verify] WARNING: extra_info is not a dict")

            print(f"[verify] prompt type: {type(sample_prompt).__name__}")
            if isinstance(sample_prompt, list):
                print(f"[verify] prompt length: {len(sample_prompt)}")
            else:
                print("[verify] WARNING: prompt is not a list")

    except Exception as e:
        print(f"[verify] FAILED to load with datasets: {e}")
        print("[verify] Fallback: trying pandas...")
        try:
            df_check = pd.read_parquet(output_file)
            print("[verify] pandas loaded successfully")
            print(f"[verify] Rows: {len(df_check)}")
            if len(df_check) > 0 and "extra_info" in df_check.columns:
                v = df_check["extra_info"].iloc[0]
                print(f"[verify] extra_info type (pandas): {type(v).__name__}")
        except Exception as e2:
            print(f"[verify] pandas also failed: {e2}")


def main():
    parser = argparse.ArgumentParser(description="Fix nested fields (extra_info / reward_model / prompt) in a parquet file.")
    parser.add_argument(
        "--input_file",
        default="data/selector/selector_training_4k_top25.parquet",
        help="Input parquet file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/selector/selector_training_4k_top25_fix.parquet",
        help="Output parquet file path",
    )

    args = parser.parse_args()
    fix_nested_fields_in_parquet(args.input_file, args.output)


if __name__ == "__main__":
    main()
