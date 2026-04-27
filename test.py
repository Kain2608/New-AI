import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy import sparse
from collections import defaultdict
import pickle

class FastItemCF:
    def __init__(self, Y_data, k=30, shrink=20, min_common=2):
        self.Y_data = Y_data.astype(np.int32)
        self.k = k
        self.shrink = shrink
        self.min_common = min_common

        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

    # ================= NORMALIZE =================
    def normalize_Y(self):
        self.global_mean = np.mean(self.Y_data[:, 2])

        self.user_bias = np.zeros(self.n_users)

        # chỉ giữ user bias (KHÔNG dùng item bias)
        for u in range(self.n_users):
            idx = self.Y_data[:, 0] == u
            if np.any(idx):
                self.user_bias[u] = np.mean(self.Y_data[idx, 2]) - self.global_mean

        users = self.Y_data[:, 0].astype(np.int32)
        items = self.Y_data[:, 1].astype(np.int32)
        ratings = self.Y_data[:, 2].astype(np.float64)

        # chỉ trừ user bias
        ratings -= self.user_bias[users]

        self.Ybar = sparse.coo_matrix(
            (ratings, (users, items)),
            shape=(self.n_users, self.n_items)
        ).tocsr()

    # ================= SIMILARITY =================
    def similarity(self):
        S_sparse = cosine_similarity(self.Ybar.T, dense_output=False)

        binary = self.Ybar.copy()
        binary.data = np.ones_like(binary.data)
        co_count = (binary.T @ binary).toarray()

        S = S_sparse.toarray()

        mask = co_count >= self.min_common
        S = S * (co_count / (co_count + self.shrink))
        S[~mask] = 0

        self.S = S

    def fit(self):
        self.normalize_Y()
        self.similarity()

    # ================= PREDICT =================
    def predict(self, u, i):
        items_rated = self.Y_data[self.Y_data[:, 0] == u][:, 1]

        if len(items_rated) == 0:
            return self.global_mean

        sim = self.S[i, items_rated]

        if np.sum(np.abs(sim)) == 0:
            return self.global_mean + self.user_bias[u]

        k_actual = min(self.k, len(items_rated))
        top_k_idx = np.argpartition(sim, -k_actual)[-k_actual:]

        sim_k = sim[top_k_idx]
        r = self.Ybar[u, items_rated[top_k_idx]].toarray().flatten()

        cf_part = (r * sim_k).sum() / (np.abs(sim_k).sum() + 1e-8)

        return self.global_mean + self.user_bias[u] + cf_part

    # ================= RECOMMEND =================
    def recommend(self, u, top_k=10):
        rated_items = self.Y_data[self.Y_data[:, 0] == u][:, 1]

        mask = np.ones(self.n_items, dtype=bool)
        mask[rated_items] = False
        unrated_items = np.where(mask)[0]

        if len(rated_items) == 0:
            preds = self.global_mean * np.ones_like(unrated_items)
        else:
            sim_matrix = self.S[unrated_items][:, rated_items]
            k_actual = min(self.k, len(rated_items))

            if k_actual == 0:
                preds = self.global_mean * np.ones_like(unrated_items)
            else:
                if sim_matrix.shape[1] > k_actual:
                    top_k_idx = np.argpartition(sim_matrix, -k_actual, axis=1)[:, -k_actual:]
                    sim_k = np.take_along_axis(sim_matrix, top_k_idx, axis=1)

                    rated_vals = self.Ybar[u, rated_items].toarray().flatten()
                    r_matrix = rated_vals[top_k_idx]
                else:
                    sim_k = sim_matrix
                    rated_vals = self.Ybar[u, rated_items].toarray().flatten()
                    r_matrix = np.tile(rated_vals, (len(unrated_items), 1))

                num = np.sum(sim_k * r_matrix, axis=1)
                den = np.sum(np.abs(sim_k), axis=1) + 1e-8
                cf_part = num / den

                preds = self.global_mean + self.user_bias[u] + cf_part

        top_k_actual = min(top_k, len(unrated_items))
        top_items_idx = np.argpartition(preds, -top_k_actual)[-top_k_actual:]

        best_items = unrated_items[top_items_idx]
        best_scores = preds[top_items_idx]

        sort_idx = np.argsort(best_scores)[::-1]
        return best_items[sort_idx].tolist()

    # ================= EVALUATE =================
    def evaluate_ranking(self, test_data, top_k=10, threshold=3):  # 👈 giảm threshold
        test_dict = defaultdict(list)

        for u, i, r in test_data:
            if r >= threshold:
                test_dict[int(u)].append(int(i))

        precisions, recalls = [], []

        for u in test_dict:
            recs = self.recommend(u, top_k)
            relevant = test_dict[u]

            if len(relevant) == 0:
                continue

            hit = len(set(recs) & set(relevant))

            precisions.append(hit / top_k)
            recalls.append(hit / len(relevant))

        return np.mean(precisions), np.mean(recalls)


# ================= MAIN =================
if __name__ == '__main__':
    r_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols)

    data = df[['user_id', 'item_id', 'rating']].values
    data[:, :2] -= 1

    train, temp = train_test_split(data, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    k_candidates = [10, 20, 40, 60]
    shrink_candidates = [20, 50, 100]

    best_score = -1
    best_params = {}

    print("=== TUNING ===")

    for s in shrink_candidates:
        for k_val in k_candidates:
            model = FastItemCF(train, k=k_val, shrink=s)
            model.fit()

            p, r = model.evaluate_ranking(val, top_k=10)
            score = (p + r) / 2

            print(f"k={k_val}, shrink={s} => P={p:.4f}, R={r:.4f}")

            if score > best_score:
                best_score = score
                best_params = {'k': k_val, 'shrink': s}

    print("\nBest:", best_params)

    final_model = FastItemCF(train, **best_params)
    final_model.fit()

    with open("fast_item_cf.pkl", "wb") as f:
        pickle.dump(final_model, f)

    p, r = final_model.evaluate_ranking(test, top_k=10)

    print("\n=== TEST ===")
    print(f"P@10: {p:.4f}, R@10: {r:.4f}")