import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse 
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS

class CF(object):
    def __init__(self, Y_data, k, dist_func = cosine_similarity, uuCF = 1):
        self.uuCF = uuCF
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]
        self.k = k
        self.dist_func = dist_func
        self.Ybar_data = None
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1 
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

    def normalize_Y(self):
        users = self.Y_data[:, 0]
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))
        for n in range(self.n_users):
            ids = np.where(users == n)[0].astype(np.int32)
            if len(ids) == 0:
                self.mu[n] = 0
                continue
            ratings = self.Y_data[ids, 2]
            m = np.mean(ratings)
            self.mu[n] = m
            self.Ybar_data[ids, 2] = ratings - self.mu[n]

        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
            (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()

    def similarity(self):
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)

    def fit(self):
        self.normalize_Y()
        self.similarity()

    def __pred(self, u, i, normalized = 1):
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        sim = self.S[u, users_rated_i]
        a = np.argsort(sim)[-self.k:] 
        nearest_s = sim[a]
        r = self.Ybar[i, users_rated_i[a]]
        res = (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8)
        return res if normalized else res + self.mu[u]
    
    def recommend(self, u):
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()              
        recommended_items = []
        # Tối ưu: Chỉ duyệt qua các item phổ biến hoặc giới hạn phạm vi nếu cần
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                rating = self.__pred(u, i)
                if rating > 0: 
                    recommended_items.append(i)
        return recommended_items 

# --- KHỞI TẠO DỮ LIỆU ---
try:
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    items = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None)
    items = items[[0, 1]]
    items.columns = ['item_id', 'title']
    movie_dict = dict(zip(items.item_id, items.title))
    Y_data = ratings[['user_id','item_id','rating']].values
    
    rs = CF(Y_data, k = 10, uuCF = 1) # Tăng k lên 10 để chính xác hơn
    rs.fit()
except FileNotFoundError:
    print("Lỗi: Không tìm thấy thư mục ml-100k. Hãy đảm bảo nó nằm cùng thư mục với file test.py")

# --- FLASK APP ---
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return send_file("index.html")

@app.route("/predict")
def predict():
    user_id = request.args.get('user_id', type=int)
    if user_id is None or user_id >= rs.n_users:
        return jsonify({"error": "User ID không hợp lệ"}), 400

    # Lấy 10 gợi ý hàng đầu
    recommended_ids = rs.recommend(user_id)[:10]

    result = []
    for item in recommended_ids:
        result.append({
            "movie_id": int(item),
            "title": movie_dict.get(item, f"Movie {item}"),
            "poster": f"https://picsum.photos/seed/movie-{item}/300/450", 
            "genre": "AI Suggestion" 
        })
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)