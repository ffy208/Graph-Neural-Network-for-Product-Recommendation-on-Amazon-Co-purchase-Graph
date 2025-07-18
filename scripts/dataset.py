import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path




class FullFeatureDataset(Dataset):
    """
    每个样本返回:
      - item_id (torch.long)
      - feat    (torch.float tensor, shape=[5])
      - categories (list of list of str)
      - reviews    (list of dict)
    """


    def __init__(self):
        # 文件路径配置
        # 脚本所在的目录 ~/project_root/scripts
        script_dir = Path(__file__).resolve().parent
        # 项目根目录 ~/project_root
        project_root = script_dir.parent
        data_dir = project_root / "data"
        input_path = os.path.join(data_dir, "amazon-meta.txt")
        items_path = os.path.join(data_dir, "items.csv")
        categories_path = os.path.join(data_dir, "categories.csv")
        reviews_path = os.path.join(data_dir, "reviews.csv")

        # 1. 加载表
        items_df = pd.read_csv(items_path)
        categories_df = pd.read_csv(categories_path, dtype=str).fillna("")
        reviews_df = pd.read_csv(reviews_path, parse_dates=["date"])

        # 2. Prepare numeric features
        num_cols = ['salesrank', 'category_count', 'reviews_total', 'reviews_downloaded', 'reviews_avg_rating']
        feats = items_df[num_cols].fillna(0).values
        self.features = torch.tensor(feats, dtype=torch.float)

        # 3. Item IDs tensor
        self.ids = torch.tensor(items_df['item_Id'].values, dtype=torch.long)

        # 4. Build category mapping
        cat_cols = [c for c in categories_df.columns if c.startswith('category_path_')]
        # Ensure item_Id is int
        categories_df['item_Id'] = categories_df['item_Id'].astype(int)
        self.cat_map = categories_df.groupby('item_Id')[cat_cols] \
            .apply(lambda df: df.values.tolist()) \
            .to_dict()

        # 5. Build review mapping
        reviews_df['item_Id'] = reviews_df['item_Id'].astype(int)
        self.rev_map = reviews_df.groupby('item_Id') \
            .apply(lambda df: df.to_dict('records')) \
            .to_dict()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item_id = int(self.ids[idx].item())
        feat = self.features[idx]
        cats = self.cat_map.get(item_id, [])
        revs = self.rev_map.get(item_id, [])
        return item_id, feat, cats, revs