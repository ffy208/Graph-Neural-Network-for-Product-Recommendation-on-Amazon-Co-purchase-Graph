import os
import pandas as pd
import re
from pathlib import Path

# if run with colab
def convert_amazon_meta_to_csv(project_root):
    """
    Convert amazon-meta.txt into items.csv, categories.csv and reviews.csv
    Args:
        project_root (str): Root directory where the 'data' folder will be created
    """

    project_root = Path(project_root)
    data_dir = project_root / "data"

    # file paths
    input_path = os.path.join(data_dir, "amazon-meta.txt")
    items_path = os.path.join(data_dir, "items.csv")
    categories_path = os.path.join(data_dir, "categories.csv")
    reviews_path = os.path.join(data_dir, "reviews.csv")

    # if csv files already exist, skip all following steps
    if os.path.exists(items_path) and os.path.exists(categories_path) and os.path.exists(reviews_path):
        print("✔️ CSV files already exist. Skipping conversion.")
        return

    items_records = []
    categories_records = []
    reviews_records = []
    current = {}
    category_lines = []
    review_lines = []

    def save_current():
        """store current, category_lines, review_lines and reset"""
        defaults = {
            "ASIN": None,
            "title": None,
            "group": None,
            "salesrank": None,
            "similar": None,
            "category_count": None,
            "reviews_total": None,
            "reviews_downloaded": None,
            "reviews_avg_rating": None
        }
        for k, v in defaults.items():
            current.setdefault(k, v)

        # store categories
        for cat_line in category_lines:
            parts = [cat for cat in cat_line.split('|') if cat]
            rec = {"item_Id": current["item_Id"]}
            for idx, cat in enumerate(parts, start=1):
                rec[f"category_path_{idx}"] = cat
            categories_records.append(rec)
        current["category_count"] = len(category_lines)

        # store reviews
        for rline in review_lines:
            m = re.match(
                r"(\d{4}-\d{1,2}-\d{1,2})\s+cutomer:\s+(\S+)\s+rating:\s+(\d+)\s+votes:\s+(\d+)\s+helpful:\s+(\d+)",
                rline
            )
            if m:
                date, cust, rate, vote, help_ = m.groups()
                reviews_records.append({
                    "item_Id": current["item_Id"],
                    "date": date,
                    "customer_Id": cust,
                    "rating": int(rate),
                    "votes": int(vote),
                    "helpful": int(help_)
                })

        # store item 
        items_records.append(current.copy())

    # extract information from txt file line by line
    with open(input_path, "r", encoding="latin1") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("Id:"):
                if current:
                    save_current()
                current = {"item_Id": int(line.split("Id:")[1].strip())}
                category_lines = []
                review_lines = []
            elif line.strip().startswith("ASIN:"):
                current["ASIN"] = line.split("ASIN:")[1].strip()
            elif line.strip().startswith("title:"):
                current["title"] = line.split("title:")[1].strip()
            elif line.strip().startswith("group:"):
                current["group"] = line.split("group:")[1].strip()
            elif line.strip().startswith("salesrank:"):
                v = line.split("salesrank:")[1].strip()
                current["salesrank"] = int(v) if v.isdigit() else ""
            elif line.strip().startswith("similar:"):
                parts = line.split()
                cnt = int(parts[1])
                current["similar"] = ",".join(parts[2:]) if cnt > 0 else ""
            elif line.strip().startswith("categories:"):
                continue
            elif line.strip().startswith("|"):
                category_lines.append(line.strip())
            elif line.strip().startswith("reviews:"):
                parts = line.split("avg rating:")
                if len(parts) == 2:
                    try:
                        current["reviews_avg_rating"] = float(parts[1].strip())
                    except ValueError:
                        current["reviews_avg_rating"] = ""
                if "total:" in line:
                    current["reviews_total"] = int(line.split("total:")[1].split()[0])
                if "downloaded:" in line:
                    current["reviews_downloaded"] = int(line.split("downloaded:")[1].split()[0])
            elif re.match(r"\d{4}-\d{1,2}-\d{1,2}.*cutomer:", line.strip()):
                review_lines.append(line.strip())

    if current:
        save_current()

    # convert to csv file
    items_df = pd.DataFrame(items_records)
    items_df = items_df[[
        "item_Id", "ASIN", "title", "group", "salesrank",
        "similar", "category_count", "reviews_total",
        "reviews_downloaded", "reviews_avg_rating"
    ]]
    # categories_df = pd.DataFrame(categories_records).fillna("")
    categories_df = pd.DataFrame(categories_records)
    reviews_df = pd.DataFrame(reviews_records)

    # 写出文件
    os.makedirs(data_dir, exist_ok=True)
    items_df.to_csv(items_path, index=False)
    categories_df.to_csv(categories_path, index=False)
    reviews_df.to_csv(reviews_path, index=False)

    print("✔️ Generated CSVs:")
    print(f" - items: {items_path}")
    print(f" - categories: {categories_path}")
    print(f" - reviews: {reviews_path}")



# if run locally
# # 脚本所在的目录 ~/project_root/scripts
# script_dir = Path(__file__).resolve().parent
# # 项目根目录 ~/project_root
# project_root = script_dir.parent
# data_dir = project_root / "data"

# # 文件路径配置
# input_path = os.path.join(data_dir, "amazon-meta.txt")
# items_path = os.path.join(data_dir, "items.csv")
# categories_path = os.path.join(data_dir, "categories.csv")
# reviews_path = os.path.join(data_dir, "reviews.csv")

# # 如果 CSV 已存在，跳过
# if os.path.exists(items_path) and os.path.exists(categories_path) and os.path.exists(reviews_path):
#     print("✔️ CSV files already exist. Skipping conversion.")
# else:
#     # 存储列表
#     items_records = []
#     categories_records = []
#     reviews_records = []

#     # 临时变量
#     current = {}
#     category_lines = []
#     review_lines = []

#     def save_current():
#         """保存并重置 current, category_lines, review_lines"""
#         # 填充缺省字段
#         defaults = {
#             "ASIN": "",
#             "title": "",
#             "group": "",
#             "salesrank": "",
#             "similar": "",
#             "category_count": 0,
#             "reviews_total": 0,
#             "reviews_downloaded": 0,
#             "reviews_avg_rating": ""
#         }
#         for k, v in defaults.items():
#             current.setdefault(k, v)

#         # 保存 categories 子表
#         for cat_line in category_lines:
#             parts = [cat for cat in cat_line.split('|') if cat]
#             rec = {"item_Id": current["item_Id"]}
#             for idx, cat in enumerate(parts, start=1):
#                 rec[f"category_path_{idx}"] = cat
#             categories_records.append(rec)
#         current["category_count"] = len(category_lines)

#         # 保存 reviews 子表
#         for rline in review_lines:
#             m = re.match(
#                 r"(\d{4}-\d{1,2}-\d{1,2})\s+cutomer:\s+(\S+)\s+rating:\s+(\d+)\s+votes:\s+(\d+)\s+helpful:\s+(\d+)",
#                 rline
#             )
#             if m:
#                 date, cust, rate, vote, help_ = m.groups()
#                 reviews_records.append({
#                     "item_Id": current["item_Id"],
#                     "date": date,
#                     "customer_Id": cust,
#                     "rating": int(rate),
#                     "votes": int(vote),
#                     "helpful": int(help_)
#                 })

#         # 保存主表
#         items_records.append(current.copy())

#     # 逐行解析
#     with open(input_path, "r", encoding="latin1") as f:
#         for line in f:
#             line = line.rstrip()
#             if line.startswith("Id:"):
#                 # 新条目开始，先保存之前的
#                 if current:
#                     save_current()
#                 # 初始化 current
#                 current = {"item_Id": int(line.split("Id:")[1].strip())}
#                 category_lines = []
#                 review_lines = []
#             elif line.strip().startswith("ASIN:"):
#                 current["ASIN"] = line.split("ASIN:")[1].strip()
#             elif line.strip().startswith("title:"):
#                 current["title"] = line.split("title:")[1].strip()
#             elif line.strip().startswith("group:"):
#                 current["group"] = line.split("group:")[1].strip()
#             elif line.strip().startswith("salesrank:"):
#                 v = line.split("salesrank:")[1].strip()
#                 current["salesrank"] = int(v) if v.isdigit() else ""
#             elif line.strip().startswith("similar:"):
#                 parts = line.split()
#                 cnt = int(parts[1])
#                 current["similar"] = ",".join(parts[2:]) if cnt > 0 else ""
#             elif line.strip().startswith("categories:"):
#                 # ignore count line
#                 continue
#             elif line.strip().startswith("|"):
#                 category_lines.append(line.strip())
#             elif line.strip().startswith("reviews:"):
#                 parts = line.split("avg rating:")
#                 if len(parts) == 2:
#                     current["reviews_avg_rating"] = float(parts[1].strip())
#                 if "total:" in line:
#                     current["reviews_total"] = int(line.split("total:")[1].split()[0])
#                 if "downloaded:" in line:
#                     current["reviews_downloaded"] = int(line.split("downloaded:")[1].split()[0])
#             elif re.match(r"\d{4}-\d{1,2}-\d{1,2}.*cutomer:", line.strip()):
#                 review_lines.append(line.strip())

#     # 保存最后一个
#     if current:
#         save_current()

#     # 转 DataFrame 并确保列顺序
#     items_df = pd.DataFrame(items_records)
#     items_df = items_df[[
#         "item_Id", "ASIN", "title", "group", "salesrank",
#         "similar", "category_count", "reviews_total",
#         "reviews_downloaded", "reviews_avg_rating"
#     ]]

#     # categories_df
#     categories_df = pd.DataFrame(categories_records).fillna("")
#     # reviews_df
#     reviews_df = pd.DataFrame(reviews_records)

#     # 输出
#     os.makedirs(data_dir, exist_ok=True)
#     items_df.to_csv(items_path, index=False)
#     categories_df.to_csv(categories_path, index=False)
#     reviews_df.to_csv(reviews_path, index=False)

#     print("✔️ Generated CSVs:")
#     print(f" - items: {items_path}")
#     print(f" - categories: {categories_path}")
#     print(f" - reviews: {reviews_path}")

