import pandas as pd
import os

INPUT_CSV_PATH = '../data/reviews.csv'
OUTPUT_DIR = 'dataset_recbole/interactions/'
OUTPUT_FILE = 'interactions.inter'

def convert_reviews_to_inter(input_path, output_dir, output_file):
    print(f"📥 Loading reviews from {input_path} ...")
    df = pd.read_csv(input_path)

    # 重命名
    df = df.rename(columns={
        'customer_Id': 'user_id',
        'item_Id': 'item_id',
        'date': 'timestamp'
    })

    # 转 timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['timestamp'] = df['timestamp'].astype(int) // 10**9

    # 只保留三列
    df_inter = df[['user_id', 'item_id', 'timestamp']]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    # 1) 写表头，用 tab 分隔
    with open(output_path, 'w') as fout:
        fout.write('user_id:token\titem_id:token\ttimestamp:float\n')

    # 2) 追加写数据，也用 tab
    df_inter.to_csv(output_path, mode='a', header=False, index=False, sep='\t')

    print(f"✅ Saved .inter file to {output_path}")
    print(df_inter.head())

def split_inter_by_time(input_inter_path, output_dir, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    """
    按全局时间排序划分 interactions.inter 为 train/valid/test，比例默认8:1:1。
    """
    import pandas as pd
    import os

    print(f"📥 Loading interactions from {input_inter_path} ...")
    df = pd.read_csv(input_inter_path, sep='\t')
    df = df.rename(columns={
        'user_id:token': 'user_id',
        'item_id:token': 'item_id',
        'timestamp:float': 'timestamp'
    })

    # 按时间排序
    df = df.sort_values('timestamp')
    n = len(df)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    n_test = n - n_train - n_valid

    df_train = df.iloc[:n_train]
    df_valid = df.iloc[n_train:n_train + n_valid]
    df_test = df.iloc[n_train + n_valid:]

    os.makedirs(output_dir, exist_ok=True)
    for split_name, split_df in zip(['train', 'valid', 'test'], [df_train, df_valid, df_test]):
        out_path = os.path.join(output_dir, f"{split_name}.inter")
        split_df = split_df.rename(columns={
            'user_id': 'user_id:token',
            'item_id': 'item_id:token',
            'timestamp': 'timestamp:float'
        })
        split_df.to_csv(out_path, sep='\t', index=False)
        print(f"✅ Saved {split_name}.inter to {out_path}, shape: {split_df.shape}")

def split_user_by_time(input_inter_path, output_dir, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    """
    每个用户按时间排序后，按8:1:1划分。
    输出：train_user.inter, valid_user.inter, test_user.inter
    """
    import pandas as pd
    import os

    print(f"📥 Loading interactions from {input_inter_path} ...")
    df = pd.read_csv(input_inter_path, sep='\t')
    df = df.rename(columns={
        'user_id:token': 'user_id',
        'item_id:token': 'item_id',
        'timestamp:float': 'timestamp'
    })

    train_list, valid_list, test_list = [], [], []
    for user, user_df in df.groupby('user_id'):
        user_df = user_df.sort_values('timestamp')
        n = len(user_df)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        n_test = n - n_train - n_valid
        train_list.append(user_df.iloc[:n_train])
        valid_list.append(user_df.iloc[n_train:n_train + n_valid])
        test_list.append(user_df.iloc[n_train + n_valid:])
    df_train = pd.concat(train_list)
    df_valid = pd.concat(valid_list)
    df_test = pd.concat(test_list)

    os.makedirs(output_dir, exist_ok=True)
    for split_name, split_df in zip(['train_user', 'valid_user', 'test_user'], [df_train, df_valid, df_test]):
        out_path = os.path.join(output_dir, f"{split_name}.inter")
        split_df = split_df.rename(columns={
            'user_id': 'user_id:token',
            'item_id': 'item_id:token',
            'timestamp': 'timestamp:float'
        })
        split_df.to_csv(out_path, sep='\t', index=False)
        print(f"✅ Saved {split_name}.inter to {out_path}, shape: {split_df.shape}")

def split_user_leave_one_out(input_inter_path, output_dir):
    """
    每个用户按时间排序，最后一条为test，倒数第二条为valid，其余为train。
    输出：train_loo.inter, valid_loo.inter, test_loo.inter
    """
    import pandas as pd
    import os

    print(f"📥 Loading interactions from {input_inter_path} ...")
    df = pd.read_csv(input_inter_path, sep='\t')
    df = df.rename(columns={
        'user_id:token': 'user_id',
        'item_id:token': 'item_id',
        'timestamp:float': 'timestamp'
    })

    train_list, valid_list, test_list = [], [], []
    for user, user_df in df.groupby('user_id'):
        user_df = user_df.sort_values('timestamp')
        if len(user_df) >= 2:
            train_list.append(user_df.iloc[:-2])
            valid_list.append(user_df.iloc[[-2]])
            test_list.append(user_df.iloc[[-1]])
        elif len(user_df) == 1:
            test_list.append(user_df)
    df_train = pd.concat(train_list) if train_list else pd.DataFrame(columns=df.columns)
    df_valid = pd.concat(valid_list) if valid_list else pd.DataFrame(columns=df.columns)
    df_test = pd.concat(test_list) if test_list else pd.DataFrame(columns=df.columns)

    os.makedirs(output_dir, exist_ok=True)
    for split_name, split_df in zip(['train_loo', 'valid_loo', 'test_loo'], [df_train, df_valid, df_test]):
        out_path = os.path.join(output_dir, f"{split_name}.inter")
        split_df = split_df.rename(columns={
            'user_id': 'user_id:token',
            'item_id': 'item_id:token',
            'timestamp': 'timestamp:float'
        })
        split_df.to_csv(out_path, sep='\t', index=False)
        print(f"✅ Saved {split_name}.inter to {out_path}, shape: {split_df.shape}")

if __name__ == "__main__":
    convert_reviews_to_inter(INPUT_CSV_PATH, OUTPUT_DIR, OUTPUT_FILE)
    # inter_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    # split_inter_by_time(inter_path, OUTPUT_DIR)
    # # 每个用户8:1:1划分
    # split_user_by_time(inter_path, OUTPUT_DIR)
    # # 每个用户leave-one-out划分
    # split_user_leave_one_out(inter_path, OUTPUT_DIR)
