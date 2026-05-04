import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

class ItemBasedCF:
    def __init__(self, Y_data, n_users, n_items, k=200, shrink=20, min_common=2, pop_weight=0.6):
        self.Y_data = Y_data.astype(np.float64)
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
        self.shrink = shrink
        self.min_common = min_common
        self.pop_weight = pop_weight

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

    def get_rated_items(self, u):
        idx = self.Y_data[:, 0] == u
        return self.Y_data[idx, 1].astype(int)

    def predict(self, u, i):
        """Hàm dự đoán điểm cụ thể để tính RMSE"""
        if u >= self.n_users:
            return None
        
        rated_items = self.get_rated_items(u)
        if len(rated_items) == 0:
            return self.user_mean[u] if u < self.n_users else 3.0

        sim = self.S[i, rated_items]
        r_vals = self.Ybar[u, rated_items]

        den = np.sum(np.abs(sim))
        if den == 0:
            return self.user_mean[u]

        num = np.sum(sim * r_vals)
        # Cộng thêm padding (ví dụ 5.0) dưới mẫu để regularization giống base code
        pred = self.user_mean[u] + num / (den + 5.0) + self.pop_weight * self.item_pop_norm[i]
        return np.clip(pred, 1, 5)

    def predict_score_for_ranking(self, u, i):
        """Hàm dự đoán điểm tương đối để sort Ranking"""
        if u >= self.n_users:
            return None
        
        rated_items = self.get_rated_items(u)
        if len(rated_items) == 0:
            return self.item_pop_norm[i]

        sim = self.S[i, rated_items]
        # Item-Item thường tính trọng số dựa trên việc user ĐÃ THÍCH phim cũ hay chưa (>0)
        rated_vals = (self.Ybar[u, rated_items] > 0).astype(float)
        
        den = np.sum(np.abs(sim))
        if den == 0:
            return self.item_pop_norm[i]

        num = np.sum(sim * rated_vals)
        return self.user_mean[u] + num / (den + 10.0) + self.pop_weight * self.item_pop_norm[i]

    def recommend(self, u, n_rec=5):
        """API Recommender cho quá trình Deployment / Demo"""
        if u >= self.n_users:
            print(f"  [Cold-start] User {u} chưa có rating → fallback sang popularity-based")
            return [(i, None) for i in top_popular(self.Y_data, n_rec)]

        items_u = self.get_rated_items(u)
        all_items  = np.arange(self.n_items)
        unrated    = np.setdiff1d(all_items, items_u)

        preds = []
        for i in unrated:
            p = self.predict_score_for_ranking(u, i)
            if p is not None:
                preds.append((i, p))

        if len(preds) == 0:
            print(f"  [Cold-start] Không có dự đoán cho user {u} → fallback sang popularity-based")
            return [(i, None) for i in top_popular(self.Y_data, n_rec)]

        preds.sort(key=lambda x: x[1], reverse=True)
        return preds[:n_rec]

def compute_f1(p, r):
    return 2 * p * r / (p + r + 1e-8)

def top_popular(Y_data, top_n=5):
    df = pd.DataFrame(Y_data, columns=["u", "i", "r"] + (["t"] if Y_data.shape[1] > 3 else []))
    avg   = df.groupby("i")["r"].mean()
    cnt   = df.groupby("i")["r"].count()
    score = avg * np.log1p(cnt)
    top_items = score.sort_values(ascending=False).index.tolist()
    return top_items[:top_n]

def split_data(Y, train_ratio=0.7, valid_ratio=0.1):
    user_items = {}
    for row in Y:
        u, i, r, t = row[0], row[1], row[2], row[3]
        user_items.setdefault(int(u), []).append((i, r, t))

    train, valid, test = [], [], []
    for u in user_items:
        items = sorted(user_items[u], key=lambda x: x[2]) 
        n       = len(items)
        n_train = int(train_ratio * n)
        n_valid = int(valid_ratio * n)
        for i, r, _ in items[:n_train]:
            train.append([u, i, r])
        for i, r, _ in items[n_train:n_train + n_valid]:
            valid.append([u, i, r])
        for i, r, _ in items[n_train + n_valid:]:
            test.append([u, i, r])

    return np.array(train), np.array(valid), np.array(test)

def rmse(model, data):
    se, cnt = 0, 0
    for u, i, r in data:
        pred = model.predict(int(u), int(i))
        if pred is None:
            continue
        se  += (r - pred) ** 2
        cnt += 1
    return np.sqrt(se / cnt) if cnt > 0 else float('nan')

def evaluate_top_k(model, data, n_items, K=10, threshold=4, n_neg=100):
    np.random.seed(42)
    user_liked = {}
    for u, i, r in data:
        if r >= threshold:
            user_liked.setdefault(int(u), set()).add(int(i))

    precisions, recalls = [], []

    for u in user_liked:
        liked   = user_liked[u]
        items_u = set(model.get_rated_items(u))

        valid_liked = liked - items_u
        if not valid_liked:
            continue

        all_items = set(range(n_items))
        negatives = list(all_items - items_u - valid_liked)
        if len(negatives) < n_neg:
            continue

        negatives  = np.random.choice(negatives, n_neg, replace=False)
        candidates = list(valid_liked) + list(negatives)
        
        np.random.shuffle(candidates)

        preds = []
        for i in candidates:
            p = model.predict_score_for_ranking(u, i)
            if p is not None:
                p += np.random.uniform(0, 1e-6)
                preds.append((i, p))

        if not preds:
            continue

        preds.sort(key=lambda x: x[1], reverse=True)
        top_k = set([i for i, _ in preds[:K]])
        hits  = len(top_k & valid_liked)

        precisions.append(hits / K)
        recalls.append(hits / len(valid_liked))

    return np.mean(precisions) if precisions else 0.0, np.mean(recalls) if recalls else 0.0


# ========== MAIN ==========

if __name__ == "__main__":
    
    df = pd.read_csv('ml-100k/u.data',
                     sep='\t', names=['u', 'i', 'r', 't'])
    df['u'] -= 1
    df['i'] -= 1
    Y = df.values  # 4 cột: u, i, r, t

    n_users = int(np.max(Y[:, 0])) + 1
    n_items = int(np.max(Y[:, 1])) + 1

    # Chia theo timestamp
    Y_train, Y_valid, Y_test = split_data(Y)

    # ===== TUNING =====
    best_f1 = -1
    best_params    = {}
    best_model     = None

    for k in [10, 20, 30, 50]:
        for shrink in [10, 20]:
            for min_common in [3, 5]:
                model = ItemBasedCF(Y_train, n_users, n_items,
                                    k=k, shrink=shrink, min_common=min_common)
                model.fit()

                p_val, r_val = evaluate_top_k(model, Y_valid, n_items, K=10)
                f1_val = compute_f1(p_val, r_val)
                print(f"k={k}, shrink={shrink}, min_com={min_common} -> P@10={p_val:.4f}, R@10={r_val:.4f}, F1={f1_val:.4f}")
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_params    = {'k': k, 'shrink': shrink, 'min_common': min_common}
                    best_model     = model

                print(f"k={k}, shrink={shrink}, min_common={min_common} "
                      f"-> P@10={p_val:.4f}, R@10={r_val:.4f}")

    print(f"\nBest params : {best_params} | Best F1: {best_f1:.4f}")

    # ===== ĐÁNH GIÁ TRÊN TEST =====
    test_rmse      = rmse(best_model, Y_test)
    p_test, r_test = evaluate_top_k(best_model, Y_test, n_items, K=10)
    print(f"\nTest RMSE : {test_rmse:.4f}")
    print(f"Test P@10 : {p_test:.4f}")
    print(f"Test R@10 : {r_test:.4f}")
    print(f"Test F1: {compute_f1(p_test, r_test):.4f}")
    # ===== LEARNING CURVE =====
    print("\n=== VẼ LEARNING CURVE (Data Size) ===")

    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    train_rmses = []
    valid_rmses = []
    valid_p10s  = []

    # Y_train đã mất cột t do hàm split_data, nhưng dữ liệu vẫn đang được giữ nguyên thứ tự thời gian
    df_full = pd.DataFrame(Y_train, columns=["u", "i", "r"])  

    for frac in train_sizes:
        print(f"Training with {frac*100:2.0f}% data...")
        
        # Lấy prefix theo thời gian cho từng user
        Y_sub_list = []
        for u, group in df_full.groupby('u'):
            n_take = max(1, int(len(group) * frac))   # ít nhất 1 rating nếu có
            Y_sub_list.append(group.iloc[:n_take].values)
        
        Y_sub = np.vstack(Y_sub_list)
        
        # Train model
        lc_model = ItemBasedCF(Y_sub, n_users, n_items, **best_params)
        lc_model.fit()
        
        # Đánh giá
        tr_rmse = rmse(lc_model, Y_sub)
        va_rmse = rmse(lc_model, Y_valid)
        p_val, _ = evaluate_top_k(lc_model, Y_valid, n_items, K=10)
        
        train_rmses.append(tr_rmse)
        valid_rmses.append(va_rmse)
        valid_p10s.append(p_val if not np.isnan(p_val) else 0)
        
        print(f"  → Train RMSE: {tr_rmse:.4f} | Valid RMSE: {va_rmse:.4f} | P@10: {p_val:.4f}")

    # ===== VẼ BIỂU ĐỒ =====
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Training Data Size (%)')
    ax1.set_ylabel('RMSE', color='tab:blue')
    ax1.plot([int(x*100) for x in train_sizes], train_rmses, 'o-', color='tab:blue', label='Train RMSE')
    ax1.plot([int(x*100) for x in train_sizes], valid_rmses, 'o--', color='tab:orange', label='Valid RMSE')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Precision@10', color='tab:green')
    ax2.plot([int(x*100) for x in train_sizes], valid_p10s, 's-', color='tab:green', label='Valid P@10')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.legend(loc='upper right')

    plt.title('Learning Curve - Item-Based Collaborative Filtering (MovieLens 100k)')
    plt.tight_layout()
    plt.savefig('learning_curve_item_cf.png', dpi=200)
    plt.show()

    # ===== FINAL MODEL =====
    final_model = ItemBasedCF(Y[:, :3], n_users, n_items, **best_params)
    final_model.fit()

    # ===== DEMO COLD-START =====
    print("\n--- DEMO COLD-START ---")

    # Case 1: user bình thường
    print("\n[Case 1] User bình thường (user_id=0):")
    recs = final_model.recommend(0, n_rec=5)
    for item, score in recs:
        print(f"  item={item}, score={score:.4f}" if score else f"  item={item}, score=popularity-based")

    # Case 2: user mới
    new_user_id = n_users
    print(f"\n[Case 2] User mới (user_id={new_user_id}, chưa có rating):")
    recs = final_model.recommend(new_user_id, n_rec=5)
    for item, score in recs:
        print(f"  item={item}, score=popularity-based")

    # Case 3: phim mới
    new_item_id  = n_items
    n_items     += 1
    print(f"\n[Case 3] Phim mới (item_id={new_item_id}, chưa có rating):")
    print(f"  Phim {new_item_id} chưa có rating → predict trả None → không xuất hiện trong CF")
    print(f"  Hệ thống fallback sang popularity-based:")
    popular = top_popular(final_model.Y_data, top_n=5)
    for item in popular:
        print(f"  item={item}")

    # ===== LƯU MODEL =====
    with open("item_based_cf.pkl", "wb") as f:
        pickle.dump(final_model, f)
    print(f"\nĐã lưu model. Best params: {best_params}")