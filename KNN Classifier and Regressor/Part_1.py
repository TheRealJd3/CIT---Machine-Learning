#Jason Shawn D' Souza R00183051
# Change k value and options in main
import numpy as np
import matplotlib.pyplot as plt


class KNN(object):

    def __init__(self):
        pass

    @staticmethod
    def splitArray(arr):
        return arr[:, :-1], arr[:, -1]

    @staticmethod
    def calculateDistances(training_data, test_instance):
        # sqrt((x2-x1)**2+(y2-y1)**2)
        subtract = np.subtract(training_data,test_instance)
        square = np.square(subtract)
        euclid_sum =np.sum(square,axis=1)
        euclidean_distances = np.sqrt(euclid_sum)
        return euclidean_distances, np.argsort(euclidean_distances)

    def calculateDistancesManhattan(self,training_data,test_instance):
        #     |x-y| +.... so no square
        manhattan_distance = np.sum(np.absolute(np.subtract(training_data,test_instance)), axis=1)
        return manhattan_distance, np.argsort(manhattan_distance)

    @staticmethod
    def prediction(training_data, euclidean_indices, knn_k_value):
        # Get the data for the required k no. of indices and fetch the Class value/whether class 0,1,2
        k_instance_class = training_data[euclidean_indices[:knn_k_value]][:, -1]
        class_0_count = len(k_instance_class[k_instance_class == 0])
        class_1_count = len(k_instance_class[k_instance_class == 1])
        class_2_count = len(k_instance_class[k_instance_class == 2])
        if class_2_count >= class_1_count and class_2_count >= class_0_count:
            return 2
        if class_1_count >= class_0_count and class_1_count >= class_2_count:
            return 1
        if class_0_count >= class_2_count and class_0_count >= class_1_count:
            return 0

    def normalizeData(self, data):
        values = data[:, :12]  # exclude class column
        # Normalization from slide - newValue =original - minValue/maxVal-minVal
        numerator = np.subtract(values, np.min(values))
        denominator = np.subtract(np.max(values), np.min(values))
        newValue = np.divide(numerator, denominator)
        return newValue
        # if k = 1,only one index with values of class
        # Either 0,1 or 2
        # if knn_k_value == 1:
        #     if k_instance_class == 0:
        #         return 0
        #     if k_instance_class == 1:
        #         return 1
        #     if k_instance_class == 2:
        #         return 2
        # else:
        #     # Else take length of the elements with value 0,1,2
        #     class_0_count = len(k_instance_class[k_instance_class == 0])
        #     class_1_count = len(k_instance_class[k_instance_class == 1])
        #     class_2_count = len(k_instance_class[k_instance_class == 2])
        #     if class_2_count >= class_1_count and class_2_count >= class_0_count:
        #         return 2
        #     if class_1_count >= class_0_count and class_1_count >= class_2_count:
        #         return 1
        #     if class_0_count >= class_2_count and class_0_count >= class_1_count:
        #         return 0



def main():
    training_file = "data\\classification\\trainingData.csv"
    test_file = "data\\classification\\testData.csv"
    training_data = np.genfromtxt(training_file, delimiter=",")
    test_data = np.genfromtxt(test_file, delimiter=",")
    knn = KNN()
    # Uncomment below for normalized data
    # training_data = knn.normalizeData(training_data)
    # test_data = knn.normalizeData(test_data)
    training_data_values, training_data_class = knn.splitArray(training_data)
    test_data_values, test_data_class = knn.splitArray(test_data)
    test_ins = 0
    correct, incorrect = 0, 0
    knn_value = k = 1
    option=1
    # For matplot
    k_values=[]
    accuracies =[]
    # OPTION 1 FOR NORMAL,
    # OPTION 2 FOR MANHATTAN,
    # OPTION 3 FOR Varying K values without manual input and graph
    if option ==1:
        for test_instance in test_data:
            # Storing as values as we will need the actual test instance to check if prediction is correct
            test_instance_values = test_instance[:10]
            # Caluculate distance and store values
            distance, index = knn.calculateDistances(training_data_values, test_instance_values)
            # Pass the index to predict the class
            prediction = knn.prediction(training_data, index, k)
            # Calculation
            if prediction == test_instance[10]:
                correct += 1
            else:
                incorrect += 1
        print("K : {}".format(knn_value))
        print("Euclidean Correct : {} Euclidean Incorrect : {}".format(correct, incorrect))
        print("Euclidean Correct: {}".format(correct / (correct + incorrect) * 100))
        #Alternative is (correct/(correct+incorrect)*100)
    if option == 2:
        for test_instance in test_data:
            test_instance_values = test_instance[:10]
            distance, index = knn.calculateDistancesManhattan(training_data_values, test_instance_values)
            prediction = knn.prediction(training_data,index, k)
            if prediction == test_instance[10]:
                correct += 1
            else:
                incorrect += 1
        print("K : {}".format(knn_value))
        print("Manhattan Correct : {} Manhattan Incorrect : {}".format(correct, incorrect))
        print("Manhattan Correct: {}".format(correct / (correct + incorrect)* 100))
    if option == 3:
        for k_value in range(1,101):
            knn_value = k = k_value
            k_values.append(k_value)
            for test_instance in test_data:
                # Storing as values as we will need the actual test instance to check if prediction is correct
                test_instance_values = test_instance[:10]
                test_ins += 1
                distance, index = knn.calculateDistances(training_data_values, test_instance_values)
                prediction = knn.prediction(training_data, index, k)
                if prediction == test_instance[10]:
                    correct += 1
                else:
                    incorrect += 1
            print("K Value : {}".format(knn_value))
            print("Correct : {} Incorrect : {}".format(correct, incorrect))
            print((correct / len(test_data)) * 100)
            accuracies.append((correct / len(test_data)) * 100)
            # Prevent addition of corrects resulting in percentages >100 due to unnecessary addition of
            # previous correct incorrect
            correct,incorrect=0,0
        #     Matplot
        plt.plot(k_values, accuracies)
        plt.legend(['Accuracies'], loc='upper left')
        plt.xlabel('K- Value')
        plt.ylabel('Accuracy')
        plt.show()

    print("Option chosen : {}".format(option))


if __name__ == '__main__':
    main()
