# import numpy as np
# from loss_function import evaluate,rmse
# # 生成随机数据
# np.random.seed(42)

#
# # 使用fun函数计算相似度百分比
# similarity_percentage = rmse(y_predict, y_true)
#
# print("相似度百分比:", similarity_percentage)

import numpy as np

import numpy as np
y_predict = np.random.rand(32, 5)
y_true = np.random.rand(32, 5)
# def calculate_accuracy(predicted_vectors, true_vectors):
#     # 将预测向量和真实向量转换为NumPy数组
#     predicted_vectors = np.array(predicted_vectors)
#     true_vectors = np.array(true_vectors)
#     # 计算向量间的夹角余弦值
#     cosine_similarities = np.dot(predicted_vectors, true_vectors.T) / (np.linalg.norm(predicted_vectors, axis=1) * np.linalg.norm(true_vectors, axis=1))
#     print(cosine_similarities.shape)
#     # 计算准确率的平均值
#     accuracy = np.mean(cosine_similarities)
#     return accuracy
def calculate_accuracy(predicted_vectors, true_vectors):

    predicted_vectors = np.array(predicted_vectors)
    true_vectors = np.array(true_vectors)

    cosine_similarities = np.sum(predicted_vectors * true_vectors, axis=1) / (np.linalg.norm(predicted_vectors, axis=1) * np.linalg.norm(true_vectors, axis=1))
    print(cosine_similarities)
    # 计算准确率的平均值
    accuracy = np.mean(cosine_similarities)
    return accuracy

similarity = calculate_accuracy(y_predict, y_true)
print("Cosine Similarity:", similarity)