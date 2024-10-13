"""
Utilities for data handling.
"""
import json
import logging
import os
import pandas as pd
import shutil

from typing import Dict


logger = logging.getLogger(__name__)



def convert_tsv_entries_to_dataframe(tsv_dict: Dict, header: str) -> pd.DataFrame:
  """
  Converts entries from TSV file to Pandas DataFrame for faster processing.
  """
  header_fields = header.strip().split("\t")
  data = {header: [] for header in header_fields}

  for line in tsv_dict.values():
    fields = line.strip().split("\t")
    assert len(header_fields) == len(fields)
    for field, header in zip(fields, header_fields):
      data[header].append(field)

  df = pd.DataFrame(data, columns=header_fields)
  return df


def copy_dev_test(task_name: str,
                  from_dir: os.path,
                  to_dir: os.path,
                  extension: str = ".tsv"):
  """
  Copies development and test sets (for data selection experiments) from `from_dir` to `to_dir`.
  """
  if task_name == "MNLI":
    dev_filename = "dev_matched.tsv"
    test_filename = "dev_mismatched.tsv"
  elif task_name in ["SNLI", "QNLI", "WINOGRANDE"]:
    dev_filename = f"dev{extension}"
    test_filename = f"test{extension}"
  else:
    raise NotImplementedError(f"Logic for {task_name} not implemented.")

  dev_path = os.path.join(from_dir, dev_filename)
  if os.path.exists(dev_path):
    shutil.copyfile(dev_path, os.path.join(to_dir, dev_filename))
  else:
    raise ValueError(f"No file found at {dev_path}")

  test_path = os.path.join(from_dir, test_filename)
  if os.path.exists(test_path):
    shutil.copyfile(test_path, os.path.join(to_dir, test_filename))
  else:
    raise ValueError(f"No file found at {test_path}")


def read_jsonl(file_path: str, key: str = "pairID"):
  """
  Reads JSONL file to recover mapping between one particular key field
  in the line and the result of the line as a JSON dict.
  If no key is provided, return a list of JSON dicts.
  """
  df = pd.read_json(file_path, lines=True)
  records = df.to_dict('records')
  logger.info(f"Read {len(records)} JSON records from {file_path}.")

  if key:
    assert key in df.columns
    return {record[key]: record for record in records}
  return records
