import tensorflow as tf
import numpy as np
import operator as oper
import sklearn as sk
import sklearn.decomposition
import time
import wandb

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(60000, 28 * 28).astype('float32') / 255
test_images = test_images.reshape(10000, 28 * 28).astype('float32') / 255
assert train_labels.shape == (60000,)
assert test_labels.shape == (10000,)

k_times=21
wandb.login()

run = wandb.init(
    # Set the project where this run will be logged
    project="my-awesome-project",
    # Track hyperparameters and run metadata
    config={
        "k": k_times,
    })


train_images, train_labels = train_images[:7000], train_labels[:7000]
test_images, test_labels = test_images[:3000], test_labels[:3000]
svd = sklearn.decomposition.TruncatedSVD(n_components=100)
train_images = svd.fit_transform(train_images)
train_labels = train_labels
test_images = svd.transform(test_images)
test_labels = test_labels


def euc_dist(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self, K=3):
        self.K = K
    def fit(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels
    def predict(self, test_images):
            predictions = [] 
            for i in range(len(test_images)):
                dist = np.array([euc_dist(test_images[i], t_i) for t_i in self.train_images])
                dist_sorted = dist.argsort()[:self.K]
                neigh_count = {}
                for idx in dist_sorted:
                        if self.train_labels[idx] in neigh_count:
                            neigh_count[self.train_labels[idx]] += 1
                        else:
                            neigh_count[self.train_labels[idx]] = 1
                sorted_neigh_count = sorted(neigh_count.items(),key=oper.itemgetter(1), reverse=True)
                predictions.append(sorted_neigh_count[0][0]) 
            return predictions
    
for k in range (1,21):
  model = KNN(K = k)
  start=time.process_time()
  model.fit(train_images, train_labels)
  pred = model.predict(test_images)
  end=time.process_time()
  acc = sk.metrics.classification_report(test_labels, pred)

  print("K = "+str(k)+";")
  precision, recall, fscore, _ = sk.metrics.precision_recall_fscore_support(test_labels, pred)
  print("use time: {:.3f} s".format(end - start))
  print(sk.metrics.classification_report(test_labels, pred))
  correct_count = sum([1 for item in range(len(test_labels)) if test_labels[item] == pred[item]])
  print("test_num:", len(test_labels), "correct_num:", correct_count, "test_acc:", correct_count / len(test_labels))
  wandb.log({"accuracy": correct_count / len(test_labels), "correct_num": correct_count})