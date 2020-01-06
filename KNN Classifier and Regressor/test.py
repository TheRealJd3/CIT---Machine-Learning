#  Jason Shawn D Souza
# import numpy as np
# import math
# import os

# import time
#
# class KNN():
#
#     def __init__(self):
#         pass
#
#     def splitarray(self,arr):
#         return arr[:, :-1], arr[:, -1]
#
#     def calculateDistances(self, training_data, test_instance):
#         esubtract = np.subtract(training_data,test_instance)
#         square = np.square(subtract)
#         euclid_sum =np.sum(square,axis=1)
#         euclidean_distances = np.sqrt(euclid_sum)
#         return euclidean_distances, np.argsort(euclidean_distances)
#
#     def prediction(self,training_data, euclidean_indices, knn_k_value):
#         k_instance_class =training_data[euclidean_indices[:knn_k_value]][:, -1]
#         if knn_k_value==1:
#             if k_instance_class ==  0:
#                 return 0
#             if k_instance_class ==  1:
#                 return 1
#             if k_instance_class ==  2:
#                 return 2
#         else:
#             class_0_count = len(k_instance_class[k_instance_class == 0])
#             class_1_count = len(k_instance_class[k_instance_class == 1])
#             class_2_count = len(k_instance_class[k_instance_class == 2])
#             if class_2_count >= class_1_count:
#                 return 2
#             if class_1_count >= class_0_count:
#                 return 1
#             return 0
#
# def main():
#     training_file = "data\\classification\\trainingData.csv"
#     test_file = "data\\classification\\testData.csv"
#     training_data = np.genfromtxt(training_file, delimiter=",")
#     test_data = np.genfromtxt(test_file, delimiter=",")
#     knn = KNN()
#     training_data_values, training_data_class = knn.splitarray(training_data)
#     test_data_values, test_data_class = knn.splitarray(test_data)
#     test_ins = 0
#     correct,incorrect =0,0
#     knn_value = k =1
#     for test_instance in test_data:
#         test_instance_values = test_instance[:10]
#         test_ins +=1
#         distance,index = knn.calculateDistances(training_data_values,test_instance_values)
#         prediction = knn.prediction(training_data,index,k)
#         if prediction == test_instance[10]:
#             correct +=1
#         else:
#             incorrect +=1
#     print((correct/(correct+incorrect))*100)
#     # print(test_ins)
#
# if __name__ == '__main__':
#     main()