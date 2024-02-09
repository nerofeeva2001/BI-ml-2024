import numpy as np
class KNNClassifier:
    def __init__(self, k=1):
        self.k = k
    def fit(self, X, y):
        self.train_X = np.array(X, dtype=int)
        self.train_y = np.array(y, dtype=int)      
    def compute_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.train_X.shape[0]
        distances = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                distances[i, j] = np.sum(np.abs(X[i] - self.train_X[j]))
        return distances.astype(int)       
    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.train_X.shape[0]
        distances = np.zeros((num_test, num_train))
        for i in range(num_test):
            distances[i, :] = np.sum(np.abs(self.train_X - X[i, :]), axis=1)
        return distances.astype(int) 
    def compute_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.train_X.shape[0]
        distances = np.abs(X[:, np.newaxis, :] - self.train_X).sum(axis=2)
        return distances.astype(int) 
    def predict(self, X, n_loops=0):
          if n_loops == 0:
              distances = self.compute_distances_no_loops(X) 
          elif n_loops == 1:
              distances = self.compute_distances_one_loop(X)
          else:  # n_loops == 2
              distances = self.compute_distances_two_loops(X)
          if len(np.unique(self.train_y)) == 2:
              prediction = self.predict_labels_binary(distances)          
          else:
              prediction = self.predict_labels_multiclass(distances)
          return prediction.astype(int) 
    def predict_labels_binary(self, distances)
        num_test = distances.shape[0]
        prediction = np.zeros(num_test) 
        for i in range(num_test):
            closest_y = self.train_y[np.argsort(distances[i])[:self.k]]
            print(closest_y)
            prediction[i] = np.argmax(np.bincount(closest_y))
            return prediction.astype(int)            
    def predict_labels_multiclass(self, distances):
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test, dtype=int)
        for i in range(num_test):
            closest_y = self.train_y[np.argsort(distances[i, :])[:self.k]]
            y_pred[i] = np.argmax(np.bincount(closest_y))
            return prediction.astype(int) 
