import argparse
import io
import os
import json
import sys
import tarfile
from typing import Optional, Set, List

import requests
import re
from collections import defaultdict, Counter
from lxml import etree as ET
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser

# OA_SECTION="07"
parser = argparse.ArgumentParser(description="Aggregate AI/ML tag and category statistics per journal-year")
parser.add_argument("--OA_SECTION", type=str, default="05")
# parser.add_argument("--archive", type=Path, default=Path(f"/users/rwan388/data/oa_comm_xml.PMC0{OA_SECTION}xxxxxx.baseline.2024-12-18.tar.gz"), help="Path to PMC .tar or .tar.gz archive")
# parser.add_argument("--output", type=Path, default=Path(f"/users/rwan388/result/journal_tag_stats_{OA_SECTION}.jsonl"), help="Output JSONL file (append/resume)")
parser.add_argument("--journal1", type=Path, default=Path("/scratch/rwan388/data/Dataset1_final.csv"))
parser.add_argument("--journal2", type=Path, default=Path("/scratch/rwan388/data/Dataset2_final.csv"))
parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
args = parser.parse_args()
args.archive=Path(f"/scratch/rwan388/data/oa_comm_xml.PMC0{args.OA_SECTION}xxxxxx.baseline.2024-12-18.tar.gz")
args.output=Path(f"/users/rwan388/result/journal_tag_stats_{args.OA_SECTION}.jsonl")
print(f"Archive: {args.archive}")
print(f"Output file: {args.output}")
print(f"gpu memory utilization: {args.gpu_memory_utilization}")





# Define the system and user prompts
SYSTEM_PROMPT2 =  """
You're an expert in artificial intelligence and machine learning in computer. If user's input is not state the utilization of AI or machine learning,  return empty form directly. Otherwise, check again, if yes, keep reading.
Unacceptable:
-Techniques "commonly used" in AI/ML but are not subset of Artificial intelligence/machine learning
-Field merely "potentially" related to AI are not acceptable.
-Traditional biological analysis method on fields other than Artificial intelligence
-Pure cell or effect discovery
In the following chat, you will be given an article abstract, and your task is to return a response in given format of FORM A,B. If the input does not explicitly state the use of AI or machine learning techniques, return empty forms. 
If the abstract state utilization of AI/ML, give me a list of keywords in this paper's abstract that are strictly and explicitly uses artificial intelligence or machine learning, in the form of JSON FORM A.
Use the keywords in Form A, assign the paper to different item in LIST A given, return another JSON mapping FORM B after FORM A.

LIST A: {Natural language processing, Knowledge representation and reasoning, Search methodologies, Philosophical/theoretical foundations of artificial intelligence, Distributed artificial intelligence, Computer vision, Learning paradigms, Learning settings, Machine learning approaches, Machine learning algorithms, Cross-validation}

If provided text does not contain any keywords or concepts belonging to artificial intelligence, machine learning, or any of the listed categories in LIST A, return empty FORM A and FORM B.
If the category you identified are not strictly from the LIST A provided, remove that category from FORM B,
Categorize your identified category to categories in LIST A.

Respond the FOLLOWING CHAT in the following format:

<start>
FORM A: {keyword1, keyword2, keyword3...}
<split sign>
FORM B: {category1, category2, category3...}
<end>
"""

SYSTEM_PROMPT1 =  """
You're an expert in computer, specifically in artificial intelligence and machine learning. Read the input carefully, if this study does not explicitly mention the use of AI or machine learning (ML) techniques directly reject. if it is did not specifically and strictly utilize artificial intelligence and machine learning as parts of the paper's focus/contribution/method, return "NO" capitalized.
if you cannot find keywords like "artificial neural network", "deep learning", "random forest", "large language model" or "transformer", reject
Remember, merely Statistical method or traditional experiments does not count toward AI/ML: only specifically AI/ML approach counts. Any merely potentially related ones does not counts.
reasons like the field the study uses can be related to AI is invalid. 
Terms like "simulate"，"analyse" are invalid. Implications of widely used application are invalid. traditional biology technique are invalid.
if it passes examination and is explicitly uses AI/ML in the study as its focus/contribution/method in its wording, return "YES" capitalized.
after reasoning, finally return your decision in the below format:
<START>
{Response}
<END>
"""
SYSTEM_PROMPT3 =  """
You're an expert in artificial intelligence and machine learning. A list of keywords will be given, and if it does not contains artificial intelligence and machine learning keywords, return "NO" capitalized.
Terms only linking to technologies and algorithms such as "sensor" or "distance" cannot justify "YES"
Only when you find terms strictly demonstrate artificial intelligence and machine learning return "YES" capitalized.
return in the below format:E
<START>
{Response}
<END>
"""

# Define the Ollama API URL (replace with your actual URL)
YEAR_MIN=2016
YEAR_MAX=2023
PROCESSED_FILE = Path(f"/users/rwan388/result/processed{args.OA_SECTION}.txt")   # 每行一个 ArticlePath


def _render_gemma(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = m["role"]
        # Gemma 用 model 表示助手
        if role == "assistant":
            role = "model"
        parts.append(f"<start_of_turn>{role}\n{m['content']}\n<end_of_turn>")
    # 让模型继续生成助手回复
    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)
# ----------------
# Helpers
# ----------------


def _clean_text(t: str) -> str:
    # 去掉 markdown 加粗与代码围栏，减少干扰
    t = t.replace("**", "")
    t = re.sub(r"```.*?\n(.*?)```", r"\1", t, flags=re.DOTALL)
    return t

def _find_block(label: str, text: str) -> str | None:
    """
    在文本中查找 'FORM A' 或 'FORM B' 后的第一段 {...} 或 [...]。
    允许：可选冒号(半角/全角)；中间跨行；非贪婪匹配。
    """
    # (?i) 忽略大小写，(?s) 让 '.' 跨行
    pat = re.compile(
        rf"(?is){label}\s*[:：]?\s*(\{{[^{{}}]*\}}|\[[^\[\]]*\])"
    )
    m = pat.search(text)
    return m.group(1) if m else None

def _parse_items(block: str) -> list[str]:
    if not block:
        return []
    s = block.strip()
    # 若是 {...}，转成 JSON 数组的形状，方便 json 解析
    if s.startswith("{") and s.endswith("}"):
        s_json = "[" + s[1:-1] + "]"
    else:
        s_json = s
    # 先试 JSON
    try:
        arr = json.loads(s_json)
        if isinstance(arr, list):
            return [str(x).strip().strip('"').strip("'") for x in arr if str(x).strip()]
        if isinstance(arr, str):  # 少数模型会给一个大字符串
            return [p.strip().strip('"').strip("'") for p in arr.split(",") if p.strip()]
    except Exception:
        # 退化：去掉外层括号后用逗号切
        t = s[1:-1] if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")) else s
        return [p.strip().strip('"').strip("'") for p in t.split(",") if p.strip()]
    return []

def load_processed_set(path: Path) -> set[str]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    return set()

def append_processed_batch(path: Path, batch: list[str]) -> None:
    if not batch:
        return
    with path.open("a", encoding="utf-8") as f:
        for p in batch:
            f.write(p + "\n")
def fast_metadata_filter(xml_bytes: bytes, valid_issns: Set[str]) -> Optional[str]:
    """Return ISSN_EISSN key or None."""
    buf = io.BytesIO(xml_bytes)
    year_ok = False
    issn_key = None

    def local(t):
        return t.rsplit("}", 1)[-1].lower()

    for _, el in ET.iterparse(buf, events=("start",)):
        tag = local(el.tag)
        if tag == "subject" and (el.text or "").strip().lower() == "retraction":
            return None
        if tag == "article-title" and (el.text or "").lower().startswith("[retracted"):
            return None
        if tag == "pub-date" and not year_ok:
            y = el.find("year")
            if y is not None and y.text:
                try:
                    year=int(y.text.strip())
                    year_ok = YEAR_MIN <= int(y.text.strip()) <= YEAR_MAX
                    if not year_ok:
                        return None
                except ValueError:
                    return None
        if tag == "journal-meta" and issn_key is None:
            issn_el = el.find("issn")
            if issn_el is not None and issn_el.text:
                issn = issn_el.text.strip()
                if issn not in valid_issns:
                    return None
                row = JOURNAL_DF[(JOURNAL_DF["issn"] == issn) | (JOURNAL_DF["eissn"] == issn)]
                if row.empty:
                    return None
                issn_key = row.iloc[0]["ISSN_EISSN"]
        if year_ok and issn_key is not None:
            break
    if year_ok and issn_key:
        return issn_key, year  # Return ISSN and year if both are valid
    else:
        return None  # Return None if either ISSN or year is not valid


def safe_split_issn(v):
    if isinstance(v, str) and "_" in v:
        return v.split("_", 1)
    return (None, None)

def load_journal_tables(path1: Path, path2: Path):
    j1 = pd.read_csv(path1)
    j2 = pd.read_csv(path2, encoding="ISO-8859-1")
    merged = j1.merge(j2, on="ISSN_EISSN", how="inner").copy()
    merged[["issn", "eissn"]] = merged["ISSN_EISSN"].apply(lambda s: pd.Series(safe_split_issn(s)))
    valid = set(merged["issn"].dropna()) | set(merged["eissn"].dropna())
    return merged, valid

def load_vocabulary(path: Path) -> Set[str]:
    with path.open("r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}

# Load existing journal-year statistics from previous runs
def load_previous_stats(output_file: Path):
    # Initialize stats using defaultdict with year as key and dictionaries for counts and articles
    stats = defaultdict(lambda: defaultdict(lambda: {
        "Category_Counts": Counter(),
        "Tag_Counts": Counter(),
        "Journal_Count": 0,
        "AI_Count":0,
        "Articles": {}
    }))

    # Check if the file exists before trying to open it
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)  # Parse JSON from the line
                    jkey = obj["ISSN_EISSN"]
                    year = obj.get("year")  # Extract the year from the object

                    # If the year is not present, skip this line
                    if year is None:
                        continue

                    # Update statistics for this particular journal (ISSN_EISSN) and year
                    if "Category_Counts" in obj:
                        stats[jkey][year]["Category_Counts"].update(obj["Category_Counts"])
                    if "Tag_Counts" in obj:
                        stats[jkey][year]["Tag_Counts"].update(obj["Tag_Counts"])

                    # Update the journal count for this year
                    stats[jkey][year]["Journal_Count"] += obj.get("Journal_Count", 0)
                    stats[jkey][year]["AI_Count"] += obj.get("AI_Count", 0)

                    # Optionally update article details if available
                    if "Articles" in obj:
                        for title, article_data in obj["Articles"].items():
                            if title not in stats[jkey][year]["Articles"]:
                                stats[jkey][year]["Articles"][title] = {
                                    "Meta_Tags": set(),
                                    # "Form_C_Keywords": set(),
                                    "Form_A_Keywords": set(),
                                    "Form_B_Categories": set(),
                                }
                            # Add article metadata, keywords, etc., to articles
                            stats[jkey][year]["Articles"][title]["Meta_Tags"].update(article_data.get("Meta_Tags", []))
                            # stats[jkey][year]["Articles"][title]["Form_C_Keywords"].update(article_data.get("Form_C_Keywords", []))
                            stats[jkey][year]["Articles"][title]["Form_B_Categories"].update(
                                article_data.get("Form_B_Categories", []))
                            stats[jkey][year]["Articles"][title]["Form_A_Keywords"].update(article_data.get("Form_A_Keywords", []))

                except Exception as e:
                    print(f"⚠ Error processing line: {e}",flush=True)
                    continue  # Skip the line if any error occurs

    return stats
# Extract abstract from XML
def extract_abstract(root: ET.Element) -> Optional[str]:
    abstract = root.xpath(".//*[local-name()='abstract']")
    if abstract:
        return " ".join(abstract[0].itertext()).strip()
    return None

def collect_meta_tags(article_meta: ET.Element) -> List[str]:
    tags = set()
    subj_xpath = (
        ".//*[local-name()='article-categories']/*[local-name()='subj-group']/*[local-name()='subject']"
    )
    kwd_xpath = ".//*[local-name()='kwd-group']/*[local-name()='kwd']"
    for el in article_meta.xpath(f"{subj_xpath} | {kwd_xpath}"):
        if el.text:
            tags.add(el.text.strip())
    return list(tags)

def extract_article_title(root: ET.Element) -> Optional[str]:
    title = root.xpath(".//*[local-name()='article-title']")
    if title:
        return " ".join(title[0].itertext()).strip()
    return None


# Function to send abstract to Ollama API
def send_to_ollama_api(abstract: str, system_prompt: str) -> dict:
    data = {
        "model": "gemma:7b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": abstract}
        ],
        "stream": False,
        "options": {
            "temperature": 0.5
        },
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(ollama_api_url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

def check_if_ai_related(user_prompt, system_prompt, llm, temperature=0.1):
    sp = SamplingParams(temperature=temperature, max_tokens=256, stop=["<end_of_turn>"])
    messages = [
        {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
    ]
    prompt = _render_gemma(messages)
    outputs = llm.generate([prompt], sp)
    content = outputs[0].outputs[0].text.strip()
    if "YES" in content:
        print("First Stage validations:",flush=True)
        print(content,flush=True)
        return True
    else:
        return False

def check_if_ai_related_post(keywords, system_prompt, llm, temperature=0.1):
    sp = SamplingParams(temperature=temperature, max_tokens=256, stop=["<end_of_turn>"])
    messages = [
        {"role": "user", "content": f"{system_prompt}\n\n{keywords}"},
    ]
    prompt = _render_gemma(messages)
    outputs = llm.generate([prompt], sp)
    content = outputs[0].outputs[0].text.strip()
    print(content)

    if "YES" in content:
        print("Final Stage validations:",flush=True)
        print(content,flush=True)
        return True
    else:
        return False

def extract_ai_keywords(abstract, system_prompt, llm, temperature=0.5):
    # 采样设置（注意老版 vLLM 用位置参数传 sp 更稳）
    sp = SamplingParams(
        temperature=temperature,
        max_tokens=2048,
        stop=["<end_of_turn>"],  # 生成到 turn 结束即停，避免串出下一个轮次
    )

    # 将“规则”并入 user，避免 system 角色问题
    messages = [
        {"role": "user", "content": f"{system_prompt}\n\n{abstract}"},
    ]
    prompt = _render_gemma(messages)

    # 生成（不用 chat 模板）
    outputs = llm.generate([prompt], sp)  # 老版请不要写 sampling_params=sp
    content = outputs[0].outputs[0].text.strip()
    print("Stage 2 Content:", content)

    cleaned = _clean_text(content)

    form_a_block = _find_block(r"FORM\s*A", cleaned)
    form_b_block = _find_block(r"FORM\s*B", cleaned)

    if form_a_block and form_b_block:
        form_a_list = _parse_items(form_a_block)
        form_b_list = _parse_items(form_b_block)
        form_a_keywords = {x for x in form_a_list if x}
        form_b_categories = {x for x in form_b_list if x}
        return form_a_keywords, form_b_categories
    else:
        print("⚠ FORM A, FORM B not found in the response, identified as false in stage 2",flush=True)
        return None, None

# Modify journal stats format to include year, meta tags, and form C keywords under article title
def process_article(xml_bytes: bytes, journal_stats: defaultdict,issn_key, article_year: int, file_name, llm):
    root = ET.fromstring(xml_bytes, ET.XMLParser(recover=True))

    if issn_key is None:
        return
    # Assuming 'issn_key' is the key for journal (e.g., "0928-4249_1297-9716")
    # and 'article_year' is the year of the article (e.g., 2017)
    if issn_key not in journal_stats:
        journal_stats[issn_key] = defaultdict(lambda: {
            "Category_Counts": Counter(),
            "Tag_Counts": Counter(),
            "Journal_Count": 0,
            "AI_Count": 0,
            "Articles": {}
        })

    # Initialize statistics for the specific year under the issn_key
    if article_year not in journal_stats[issn_key]:
        journal_stats[issn_key][article_year] = {
            "Category_Counts": Counter(),  # Initialize Category_Counts for this year
            "Tag_Counts": Counter(),  # Initialize Tag_Counts for this year
            "Journal_Count": 0,  # Initialize Journal_Count for this year
            "AI_Count": 0,
            "Articles": {}  # Store articles metadata for this year
        }

    # Extract the abstract, title, and meta tags of the article
    abstract = extract_abstract(root)
    article_title = extract_article_title(root)
    if abstract is None:
        print(f"No abstract contained in path{file_name}:{article_title}, pass",flush=True)
        return
    article_meta_tags = collect_meta_tags(root)
    input=abstract+"\n"+abstract

    if not abstract or not article_title:
        return
    js = journal_stats[issn_key][article_year]
    js["Journal_Count"] += 1
    # if not check_if_ai_related(abstract, SYSTEM_PROMPT1,temperature=0.5):
    #     return
    check=check_if_ai_related(input, SYSTEM_PROMPT1, llm, temperature=0.1)
    if not check:
        # print("stop futher processing",flush=True)
        return
    print(check, flush=True)
    print("Title passed first validation being:", article_title, flush=True)
    # Step 2: If AI/ML related, extract FORM A, FORM B, and FORM C
    form_a_keywords, form_b_categories = extract_ai_keywords(abstract, SYSTEM_PROMPT2,llm,temperature=0.5)
    form_a_keywords_string = "[" + ", ".join(f'"{keyword.strip()}"' for keyword in form_a_keywords) + "]"
    final_check=check_if_ai_related_post(form_a_keywords_string, SYSTEM_PROMPT3,llm, temperature=0.1)
    if not final_check:
        print("reject at post validation stage",flush=True)
        return
    print("accepted at post validation stage", flush=True)
    if form_a_keywords and form_b_categories:
        js["AI_Count"] += 1

        # Update FORM A keywords count (tags)
        for keyword in form_a_keywords:
            js["Tag_Counts"][keyword] += 1  # Increment tag count for this journal-year

        # Update FORM B categories count
        for category in form_b_categories:
            if category != "NONE" and category != '':  # Avoid adding "NONE"
                js["Category_Counts"][category] += 1  # Increment category count for this journal-year

        # If the article title is not in "Articles", initialize it
        if article_title not in js["Articles"]:
            js["Articles"][article_title] = {
                "Meta_Tags": set(),  # Initialize as a set
                "Form_A_Keywords": set(),  # Initialize as a set
                "Form_B_Categories": set(),  # Initialize as a set for Form B categoriesf
            }

        # if form_c_mapping:
        #     # Update Form C keywords under respective categories (from FORM C)
        #     for category, keywords in form_c_mapping.items():
        #         js["Articles"][article_title]["Form_C_Keywords"].update(keywords)

        # Also update the Form A keywords (general keywords from FORM A)
        js["Articles"][article_title]["Form_A_Keywords"].update(form_a_keywords)

        # Add meta tags under the article title (from article metadata)
        js["Articles"][article_title]["Meta_Tags"].update(article_meta_tags)

        # Add the category to the article's Form B categories (from FORM B)
        js["Articles"][article_title]["Form_B_Categories"].update(form_b_categories)

        # Convert sets to lists to ensure unique values, since lists allow duplicate values
        js["Articles"][article_title]["Meta_Tags"] = list(js["Articles"][article_title]["Meta_Tags"])
        js["Articles"][article_title]["Form_A_Keywords"] = list(
            js["Articles"][article_title]["Form_A_Keywords"])
        js["Articles"][article_title]["Form_B_Categories"] = list(
            js["Articles"][article_title]["Form_B_Categories"])
        # if form_c_mapping:
        #     js["Articles"][article_title]["Form_C_Keywords"] = list(
        #         js["Articles"][article_title]["Form_C_Keywords"])

        # Debugging: Log the article title and keywords added to ensure the data is being updated correctly
        print(f"Article Title Added: {article_title}",flush=True)
        print(f"Meta Tags Added: {js['Articles'][article_title]['Meta_Tags']}",flush=True)
        print(f"Form A Keywords Added: {js['Articles'][article_title]['Form_A_Keywords']}",flush=True)
        print(f"Form B Categories Added: {js['Articles'][article_title]['Form_B_Categories']}",flush=True)
        # if form_c_mapping:
        #     print(f"Form C Keywords Added: {js['Articles'][article_title]['Form_C_Keywords']}",flush=True)
        # else:
        #     print(f"Form C Keywords not found", flush=True)
    else:
        print("⚠ FORM A and FORM B not all returned in the response, classify as false", flush=True)

def save_stats(journal_stats: defaultdict, output_file: Path):
    # Write the updated statistics to the output file
    with output_file.open("w", encoding="utf-8") as f:
        for jkey, year_data in journal_stats.items():
            for year, data in year_data.items():
                # Ensure that any sets are converted to lists before serialization
                articles_data = {}
                for article_title, article_data in data.get("Articles", {}).items():
                    articles_data[article_title] = {
                        "Meta_Tags": list(article_data["Meta_Tags"]),  # Convert set to list
                        "Form_A_Keywords": list(article_data["Form_A_Keywords"]),  # Convert set to list
                        "Form_B_Categories": list(article_data["Form_B_Categories"]),  # Convert set to list
                        # "Form_C_Keywords": list(article_data["Form_C_Keywords"])  # Convert set to list
                    }

                # Save statistics including converted lists for articles data
                json.dump({
                    "ISSN_EISSN": jkey,
                    "year": year,
                    "Tag_Counts": dict(data.get("Tag_Counts", {})),
                    "Category_Counts": dict(data.get("Category_Counts", {})),
                    "Journal_Count": data.get("Journal_Count", 0),
                    "AI_Count": data.get("AI_Count", 0),
                    "Articles": articles_data
                }, f, ensure_ascii=False)
                f.write("\n")  # Write a new line after each article's stats

def get_pmc_number_from_filename(filename: str) -> int:
    """Extract PMC number from the filename after the last '/'"""
    match = re.search(r"PMC(\d+)\.xml", filename)
    if match:
        return int(match.group(1))
    return -1  # Return a default invalid number if no PMC is found

# ───────────────── robust on-the-fly ─────────────────
def process_article_on_the_fly(llm,
                               tar_path: Path,
                               processed: set[str],
                               journal_stats: defaultdict,
                               valid_issns: set,
                               flush_every: int = 500,
                               stats_path: Path | None = None
                               ):
    """
    Stream articles; every `flush_every` successful docs, flush stats & processed list.
    """
    new_batch: list[str] = []       # 收集本批新处理文章路径
    count_since_flush = 0

    # flush_bar = tqdm(total=flush_every, desc="Flush Progress", position=1, ncols=100,
    #                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} articles processed")

    def flush():
        nonlocal new_batch, count_since_flush
        if stats_path:
            save_stats(journal_stats, stats_path)
        append_processed_batch(PROCESSED_FILE, new_batch)
        new_batch.clear()
        count_since_flush = 0
        # flush_bar.n = 0  # Reset the flush progress bar
        # flush_bar.last_print_n = 0  # Reset the last printed progress
        # flush_bar.update(0)  # Update the progress bar to reset it
        print("💾 checkpoint flushed", flush=True)

    try:
        with tarfile.open(tar_path, "r|gz") as tf:
            for member in tqdm(tf, desc=f"{tar_path.name} (stream)", unit="xml",position=0):
                if not member.name.endswith(".xml") or member.name in processed:
                    continue

                    # Extract PMC number from the member name (assuming format PMCxxx.xml)
                pmc_number = get_pmc_number_from_filename(member.name)

                if pmc_number == -1:
                    continue  # Skip this member if PMC number is invalid

                base_pmc = member.name.split("/")[0]  # Extract base PMC identifier

                processed_pmc_numbers = [get_pmc_number_from_filename(pmc) for pmc in processed if
                                         pmc.startswith(base_pmc)]

                # Find the maximum PMC number already processed for the same base PMC
                max_processed_pmc = max(processed_pmc_numbers, default=-1)
                # print(max_processed_pmc,flush=True)

                # Skip the current member if its PMC number is smaller than or equal to the max processed number
                if pmc_number <= max_processed_pmc:
                    continue

                try:
                    # Extract the XML file from the tar archive
                    xml_bytes = tf.extractfile(member).read()
                except Exception as exc:
                    print(f"⚠ Error extracting {member.name}: {exc}",flush=True)
                    continue  # Skip files that couldn't be extracted

                try:
                    new_batch.append(member.name)
                    # Extract metadata and ensure that it's valid
                    metadata = fast_metadata_filter(xml_bytes, valid_issns)
                    if metadata is None:
                        continue  # Skip articles with invalid or missing metadata

                    issn_key, year = metadata

                    if not issn_key or not year:
                        continue  # Skip articles with invalid ISSN or year

                    # Process the article
                    process_article(xml_bytes, journal_stats, issn_key, year,member.name,llm)

                    count_since_flush += 1

                    # flush_bar.update(1)

                    # Flush stats every `flush_every` articles
                    if count_since_flush >= flush_every:
                        flush()

                except Exception as exc:
                    print(f"⚠ Error processing {member.name}: {exc}",flush=True)
                    continue  # Skip articles that couldn't be processed

    except tarfile.ReadError as e:
        print(f"Error reading tar file {tar_path}: {e}",flush=True)
    except Exception as e:
        print(f"Unexpected error: {e}",flush=True)

        # Final flush to save any remaining data
    flush()
    print("✅ Finished processing archive", tar_path.name,flush=True)

# ─────────────────────────────────── Main Function ────────────────────────────────────

def main():

    if not args.archive.exists():
        sys.exit(f"❌ Archive {args.archive} not found")

    global JOURNAL_DF  # for fast_metadata_filter
    JOURNAL_DF, valid_issns = load_journal_tables(args.journal1, args.journal2)

    # Load previous stats from output file, if exists
    journal_stats = load_previous_stats(args.output)
    processed = load_processed_set(PROCESSED_FILE)

    repo_id = "google/gemma-3-12b-it"  # ⚠ 若报 chat template 相关错误，建议改为 "google/gemma-7b-it"
    model_store = f"/scratch/rwan388/LLM/{repo_id}"

    llm = LLM(model=model_store,max_model_len=40000,gpu_memory_utilization=args.gpu_memory_utilization)

    # Open the tar file and process it on the fly
    process_article_on_the_fly(
        llm=llm,
        tar_path=args.archive,
        processed=processed,
        journal_stats=journal_stats,
        valid_issns=valid_issns,
        flush_every=500,  # 每 500 篇保存一次
        stats_path=args.output,
    )

    # Save the updated statistics after processing
    save_stats(journal_stats, args.output)
    print(f"✅ Tag statistics written to {args.output.resolve()}",flush=True)


if __name__ == "__main__":
    main()
