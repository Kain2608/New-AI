import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

class FastItemCF_Scratch:
    def __init__(self, Y_data, k=200, shrink=20, min_common=2, pop_weight=0.6,
                 n_users=None, n_items=None):

        self.Y_data = Y_data.astype(np.float64)
        self.k = k
        self.shrink = shrink
        self.min_common = min_common
        self.pop_weight = pop_weight

        if n_users is None:
            self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        else:
            self.n_users = n_users

        if n_items is None:
            self.n_items = int(np.max(self.Y_data[:, 1])) + 1
        else:
            self.n_items = n_items

    def normalize_Y(self):
        self.user_mean = np.zeros(self.n_users)

        for u in range(self.n_users):
            idx = self.Y_data[:, 0] == u
            if np.any(idx):
                self.user_mean[u] = np.mean(self.Y_data[idx, 2])

        items = self.Y_data[:, 1].astype(np.int32)
        item_counts = np.bincount(items, minlength=self.n_items)
        self.item_pop = np.log1p(item_counts)
        self.item_pop_norm = self.item_pop / (np.max(self.item_pop) + 1e-8)

        self.Ybar = np.zeros((self.n_users, self.n_items))
        for row in self.Y_data:
            u, i, r = int(row[0]), int(row[1]), row[2]
            self.Ybar[u, i] = r - self.user_mean[u]

    def similarity(self):
        dot_product = self.Ybar.T @ self.Ybar

        norms = np.sqrt(np.sum(self.Ybar**2, axis=0))
        norms[norms == 0] = 1e-8
        norm_matrix = norms[:, None] @ norms[None, :]

        S = dot_product / norm_matrix

        binary_Y = (self.Ybar != 0).astype(np.float64)
        co_count = binary_Y.T @ binary_Y

        S = S * (co_count / (co_count + self.shrink))
        S = S * (co_count >= self.min_common)

        # Chặn Data Leakage
        np.fill_diagonal(S, 0)

        # TOP-K
        for i in range(self.n_items):
            row = S[i]
            if self.k < self.n_items:
                top_k_idx = np.argsort(row)[-self.k:]
                mask = np.zeros(self.n_items, dtype=bool)
                mask[top_k_idx] = True
                row[~mask] = 0

        self.S = S

    def fit(self):
        self.normalize_Y()
        self.similarity()

    def recommend(self, u, top_k=10, exclude_rated=True):
        rated_idx = self.Y_data[:, 0] == u
        rated_items = self.Y_data[rated_idx, 1].astype(np.int32)

        if exclude_rated:
            mask = np.ones(self.n_items, dtype=bool)
            mask[rated_items] = False
            candidate_items = np.where(mask)[0]
        else:
            candidate_items = np.arange(self.n_items)

        if len(rated_items) == 0:
            preds = self.item_pop_norm[candidate_items]
        else:
            sim_matrix = self.S[candidate_items][:, rated_items]
            rated_vals = (self.Ybar[u, rated_items] > 0).astype(float)

            num = np.sum(sim_matrix * rated_vals, axis=1)
            den = np.sum(np.abs(sim_matrix), axis=1) + 10.0

            preds = self.user_mean[u] + num / den + self.pop_weight * self.item_pop_norm[candidate_items]

        if len(candidate_items) == 0:
            return []

        top_k_actual = min(top_k, len(candidate_items))
        top_items_idx = np.argsort(preds)[-top_k_actual:][::-1]

        return candidate_items[top_items_idx].tolist()

    def evaluate_ranking(self, data, top_k=10, threshold=4, exclude_rated=True):
        test_dict = defaultdict(list)
        for row in data:
            u, i, r = int(row[0]), int(row[1]), row[2]
            if r >= threshold:
                test_dict[u].append(i)

        precisions, recalls = [], []

        for u, relevant in test_dict.items():
            recs = self.recommend(u, top_k, exclude_rated=exclude_rated)
            hit = len(set(recs) & set(relevant))

            precisions.append(hit / top_k)
            recalls.append(hit / len(relevant))

        if not precisions:
            return 0.0, 0.0

        return np.mean(precisions), np.mean(recalls)

    def evaluate_rmse(self, data):
        se, cnt = 0, 0

        for row in data:
            u, i, r = int(row[0]), int(row[1]), row[2]

            rated_idx = self.Y_data[:, 0] == u
            rated_items = self.Y_data[rated_idx, 1].astype(np.int32)

            if len(rated_items) == 0:
                pred = self.user_mean[u]
            else:
                sim = self.S[i, rated_items]
                r_vals = self.Ybar[u, rated_items]

                num = np.sum(sim * r_vals)
                den = np.sum(np.abs(sim)) + 5.0

                pred = self.user_mean[u] + num / den + self.pop_weight * self.item_pop_norm[i]

            se += (pred - r) ** 2
            cnt += 1

        return np.sqrt(se / cnt) if cnt > 0 else 0


def compute_f1(p, r):
    return 2 * p * r / (p + r + 1e-8)


def custom_train_val_test_split(data_array, val_size=0.1, test_size=0.1, random_seed=42):
    np.random.seed(random_seed)
    user_records = defaultdict(list)

    for row in data_array:
        user_records[int(row[0])].append(row)

    train_list, val_list, test_list = [], [], []

    for u, records in user_records.items():
        n = len(records)
        np.random.shuffle(records)

        if n < 5:
            train_list.extend(records)
        else:
            n_test = int(n * test_size)
            n_val = int(n * val_size)

            test_list.extend(records[:n_test])
            val_list.extend(records[n_test:n_test+n_val])
            train_list.extend(records[n_test+n_val:])

    return np.array(train_list), np.array(val_list), np.array(test_list)


# ================= MAIN =================
if __name__ == '__main__':
    df = pd.read_csv('ml-100k/u.data', sep='\t',
                     names=['user_id', 'item_id', 'rating', 'timestamp'])

    df = df[df['item_id'].map(df['item_id'].value_counts()) >= 10]
    df = df[df['user_id'].map(df['user_id'].value_counts()) >= 10]

    user_map = {u: i for i, u in enumerate(df['user_id'].unique())}
    item_map = {i: j for j, i in enumerate(df['item_id'].unique())}

    df['user_id'] = df['user_id'].map(user_map)
    df['item_id'] = df['item_id'].map(item_map)

    data = df[['user_id', 'item_id', 'rating']].values

    train, val, test = custom_train_val_test_split(data)

    k_values = [10, 20]
    shrink_values = [50, 100]

    best_score = -1
    best_params = None

    print("\n=== GRID SEARCH LOG ===")

    for k in k_values:
        for shrink in shrink_values:
            model = FastItemCF_Scratch(train, k=k, shrink=shrink)
            model.fit()

            # Bỏ đánh giá P/R trên Train, chỉ đánh giá trên Val
            p_val, r_val = model.evaluate_ranking(val, exclude_rated=True)
            rmse_train = model.evaluate_rmse(train)
            rmse_val = model.evaluate_rmse(val)
            f1_val = compute_f1(p_val, r_val)

            print(f"\n[k={k}, shrink={shrink}]")    
            print(f" Train -> RMSE={rmse_train:.4f}")
            print(f" Val   -> P={p_val:.4f}, R={r_val:.4f}, F1={f1_val:.4f}, RMSE={rmse_val:.4f}")

            if f1_val > best_score:
                best_score = f1_val
                best_params = {'k': k, 'shrink': shrink}

    print("\n=== BEST PARAMS ===")
    print(best_params, " | F1_val =", best_score)
    
    # ===== LEARNING CURVE =====
    # Cập nhật lại các mốc train_sizes giống hệt ảnh mẫu
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

    lc_p_val, lc_r_val = [], []
    lc_rmse_tr, lc_rmse_val = [], []

    n_users = int(np.max(data[:, 0])) + 1
    n_items = int(np.max(data[:, 1])) + 1

    print("\n=== LEARNING CURVE LOG ===")
    
    # Trộn đều dữ liệu để test Learning Curve
    np.random.seed(42)
    np.random.shuffle(train) 

    for frac in train_sizes:
        n = int(len(train) * frac)
        sub_train = train[:n]

        model = FastItemCF_Scratch(
            sub_train,
            k=best_params['k'],
            shrink=best_params['shrink'],
            n_users=n_users,
            n_items=n_items
        )
        model.fit()

        # Chỉ tính Ranking cho tập Validation
        p_v, r_v = model.evaluate_ranking(val, exclude_rated=True)
        rmse_tr = model.evaluate_rmse(sub_train)
        rmse_v = model.evaluate_rmse(val)

        print(f"\n[Train size = {frac}]")
        print(f" Train -> RMSE={rmse_tr:.4f}")
        print(f" Val   -> P={p_v:.4f}, R={r_v:.4f}, RMSE={rmse_v:.4f}")

        lc_p_val.append(p_v)
        lc_r_val.append(r_v)
        lc_rmse_tr.append(rmse_tr)
        lc_rmse_val.append(rmse_v)

    # ===== PLOT =====
    # Khôi phục style mặc định (nền trắng) thay vì seaborn-darkgrid
    plt.style.use('default') 
    
    # Label cho trục X hiển thị dạng %
    x_labels = [f"{int(x*100)}%" for x in train_sizes]

    # 1. Biểu đồ RMSE (Giống ảnh mẫu)
    plt.figure(figsize=(8, 5), dpi=300) # Thêm dpi=300 để ảnh nét hơn
    
    plt.plot(train_sizes, lc_rmse_tr, marker='o', linestyle='-', color='#1f77b4', linewidth=1.5, label="Train RMSE")
    plt.plot(train_sizes, lc_rmse_val, marker='o', linestyle='--', color='#ff7f0e', linewidth=1.5, label="Valid RMSE")
    
    plt.title("Learning Curve - Item-Based CF", fontsize=12)
    plt.xlabel("Training data size", fontsize=10)
    plt.ylabel("RMSE", fontsize=10)
    
    plt.xticks(train_sizes, x_labels)
    # Cài đặt lưới mờ giống ảnh mẫu
    plt.grid(True, color='#e6e6e6', linestyle='-', linewidth=1)
    
    plt.legend()
    plt.tight_layout()
    # Lưu file PNG
    plt.savefig('learning_curve_rmse.png', dpi=300, format='png', bbox_inches='tight')
    plt.show()

    # 2. Biểu đồ Đánh giá chất lượng Gợi ý (Validation Precision & Recall)
    plt.figure(figsize=(8, 5), dpi=300)
    
    plt.plot(train_sizes, lc_p_val, marker='o', linestyle='-', color='green', linewidth=1.5, label="Val Precision")
    plt.plot(train_sizes, lc_r_val, marker='o', linestyle='--', color='red', linewidth=1.5, label="Val Recall")
    
    plt.title("Learning Curve - Validation Ranking Metrics", fontsize=12)
    plt.xlabel("Training data size", fontsize=10)
    plt.ylabel("Score", fontsize=10)
    
    plt.xticks(train_sizes, x_labels)
    # Cài đặt lưới mờ
    plt.grid(True, color='#e6e6e6', linestyle='-', linewidth=1)
    
    plt.legend()
    plt.tight_layout()
    # Lưu file PNG
    plt.savefig('learning_curve_ranking.png', dpi=300, format='png', bbox_inches='tight')
    plt.show()