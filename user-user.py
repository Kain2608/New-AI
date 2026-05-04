import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

class UserBasedCF:
    def __init__(self, Y_data, n_users, n_items, k=20, shrink=20, min_common=15):
        self.Y_data = Y_data
        self.k = k
        self.n_users = n_users
        self.n_items = n_items
        self.shrink = shrink
        self.min_common = min_common

    def normalize(self):
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros(self.n_users)
        users = self.Y_data[:, 0]

        for u in range(self.n_users):
            ids = np.where(users == u)[0]
            ratings = self.Y_data[ids, 2]
            mean = np.mean(ratings) if len(ratings) > 0 else 0
            self.mu[u] = mean
            self.Ybar_data[ids, 2] = ratings - mean

        rows = self.Ybar_data[:, 0].astype(int)
        cols = self.Ybar_data[:, 1].astype(int)
        data = self.Ybar_data[:, 2]

        self.indptr = np.zeros(self.n_users + 1, dtype=int)
        for r in rows:
            self.indptr[r + 1] += 1
        self.indptr = np.cumsum(self.indptr)

        self.indices  = np.zeros(len(cols), dtype=int)
        self.csr_data = np.zeros(len(data))

        tracker = self.indptr[:-1].copy()
        for i in range(len(data)):
            r   = rows[i]
            pos = tracker[r]
            self.indices[pos]  = cols[i]
            self.csr_data[pos] = data[i]
            tracker[r] += 1

    def get_sparse_row(self, u):
        start = self.indptr[u]
        end   = self.indptr[u + 1]
        return self.indices[start:end], self.csr_data[start:end]

    def similarity(self):
        self.S = np.zeros((self.n_users, self.n_users))

        for i in range(self.n_users):
            items_i, ratings_i = self.get_sparse_row(i)

            for j in range(i, self.n_users):
                if i == j:
                    self.S[i, j] = 1.0
                    continue

                items_j, ratings_j = self.get_sparse_row(j)
                common, idx_i, idx_j = np.intersect1d(
                    items_i, items_j, return_indices=True
                )

                if len(common) < self.min_common:
                    self.S[i, j] = self.S[j, i] = 0
                    continue

                r_i = ratings_i[idx_i]
                r_j = ratings_j[idx_j]

                denom = np.sqrt(np.sum(r_i**2)) * np.sqrt(np.sum(r_j**2))
                sim   = np.sum(r_i * r_j) / denom if denom != 0 else 0

                shrink  = len(common) / (len(common) + self.shrink)
                sim    *= shrink

                self.S[i, j] = self.S[j, i] = sim

    def fit(self):
        self.normalize()
        self.similarity()

    def predict(self, u, i):
        if u >= self.n_users:
            return None

        item_indices = np.where(self.Y_data[:, 1] == i)[0]
        if len(item_indices) == 0:
            return self.mu[u]

        users_rated_i = self.Y_data[item_indices, 0].astype(int)
        users_rated_i = users_rated_i[users_rated_i != u]

        if len(users_rated_i) == 0:
            return self.mu[u]

        sims = self.S[u, users_rated_i]

        top_idx   = np.argsort(sims)[::-1][:self.k]
        top_users = users_rated_i[top_idx]
        top_sims  = np.maximum(sims[top_idx], 0)

        ratings = []
        for user in top_users:
            items_u, ratings_u = self.get_sparse_row(user)
            idx = np.where(items_u == i)[0]
            ratings.append(ratings_u[idx[0]] if len(idx) > 0 else 0)

        ratings = np.array(ratings)
        denom   = np.sum(np.abs(top_sims))
        if denom == 0:
            return self.mu[u]

        pred = self.mu[u] + np.sum(top_sims * ratings) / denom
        return np.clip(pred, 1, 5)

    def predict_score_for_ranking(self, u, i):
        item_indices = np.where(self.Y_data[:, 1] == i)[0]
        if len(item_indices) == 0:
            return self.mu[u]

        users_rated_i = self.Y_data[item_indices, 0].astype(int)
        users_rated_i = users_rated_i[users_rated_i != u]

        if len(users_rated_i) == 0:
            return self.mu[u]

        sims = self.S[u, users_rated_i]

        top_idx   = np.argsort(sims)[::-1][:self.k]
        top_users = users_rated_i[top_idx]
        top_sims  = np.maximum(sims[top_idx], 0)

        ratings = []
        for user in top_users:
            items_u, ratings_u = self.get_sparse_row(user)
            idx = np.where(items_u == i)[0]
            ratings.append(ratings_u[idx[0]] if len(idx) > 0 else 0)

        ratings = np.array(ratings)
        denom   = np.sum(np.abs(top_sims))
        if denom == 0:
            return self.mu[u]

        return self.mu[u] + np.sum(top_sims * ratings) / denom

    def recommend(self, u, n_rec=5):
        # Cold-start: user mới chưa có trong hệ thống
        if u >= self.n_users:
            print(f"  [Cold-start] User {u} chưa có rating → fallback sang popularity-based")
            return [(i, None) for i in top_popular(self.Y_data, n_rec)]

        items_u, _ = self.get_sparse_row(u)
        all_items  = np.arange(self.n_items)
        unrated    = np.setdiff1d(all_items, items_u)

        preds = []
        for i in unrated:
            p = self.predict_score_for_ranking(u, i)
            if p is not None:
                preds.append((i, p))

        # Cold-start: phim mới hoặc không có dự đoán
        if len(preds) == 0:
            print(f"  [Cold-start] Không có dự đoán cho user {u} → fallback sang popularity-based")
            return [(i, None) for i in top_popular(self.Y_data, n_rec)]

        preds.sort(key=lambda x: x[1], reverse=True)
        return preds[:n_rec]


# ========== HELPER FUNCTIONS ==========

def compute_f1(p, r):
    return 2 * p * r / (p + r + 1e-8)

def top_popular(Y_data, top_n=5):
    """Trả về top phim phổ biến nhất dựa trên điểm trung bình có trọng số."""
    df = pd.DataFrame(Y_data, columns=["u", "i", "r"])
    avg   = df.groupby("i")["r"].mean()
    cnt   = df.groupby("i")["r"].count()
    score = avg * np.log1p(cnt)  # tránh phim ít vote nhưng điểm cao
    top_items = score.sort_values(ascending=False).index.tolist()
    return top_items[:top_n]


def split_data(Y, train_ratio=0.7, valid_ratio=0.1):
    """Chia dữ liệu theo per-user + timestamp.
    Mỗi user luôn có dữ liệu trong train, thứ tự thời gian được giữ nguyên.
    """
    user_items = {}
    for u, i, r, t in Y:
        user_items.setdefault(int(u), []).append((i, r, t))

    train, valid, test = [], [], []
    for u in user_items:
        items = sorted(user_items[u], key=lambda x: x[2])  # sort theo timestamp
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
        liked      = user_liked[u]
        items_u, _ = model.get_sparse_row(u)
        items_u    = set(items_u)

        valid_liked = liked - items_u
        if not valid_liked:
            continue

        all_items = set(range(n_items))
        negatives = list(all_items - items_u - valid_liked)
        if len(negatives) < n_neg:
            continue

        negatives  = np.random.choice(negatives, n_neg, replace=False)
        candidates = list(valid_liked) + list(negatives)
        
        # FIX: Phải shuffle (trộn) candidates để phá vỡ thứ tự mặc định!
        # Nếu không shuffle, khi các item bị hoà điểm (cùng bằng mu[u]),
        # hàm sort của Python (Stable Sort) sẽ giữ nguyên valid_liked ở trên cùng.
        np.random.shuffle(candidates)

        preds = []
        for i in candidates:
            p = model.predict_score_for_ranking(u, i)
            if p is not None:
                # Cộng thêm một nhiễu siêu nhỏ để break-tie ngẫu nhiên hoàn toàn
                p += np.random.uniform(0, 1e-6)
                preds.append((i, p))

        if not preds:
            continue

        preds.sort(key=lambda x: x[1], reverse=True)
        top_k = set([i for i, _ in preds[:K]])
        hits  = len(top_k & valid_liked)

        precisions.append(hits / K)
        recalls.append(hits / len(valid_liked))

    return np.mean(precisions), np.mean(recalls)


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
                model = UserBasedCF(Y_train, n_users, n_items,
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
        lc_model = UserBasedCF(Y_sub, n_users, n_items, **best_params)
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

    plt.title('Learning Curve - User-Based Collaborative Filtering (MovieLens 100k)')
    plt.tight_layout()
    plt.savefig('learning_curve_user_cf.png', dpi=200)
    plt.show()

    # ===== FINAL MODEL =====
    final_model = UserBasedCF(Y[:, :3], n_users, n_items, **best_params)
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
    with open("user_based_cf.pkl", "wb") as f:
        pickle.dump(final_model, f)
    print(f"\nĐã lưu model. Best params: {best_params}")