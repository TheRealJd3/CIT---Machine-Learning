# Jason Shawn D' Souza R00183051
# Change k value and options in main
import numpy as np
import matplotlib.pyplot as plt
# PART - 1


class KNN(object):

    def __init__(self):
        pass

    @staticmethod
    def splitarray(arr):
        return arr[:, :-1], arr[:, -1]

    @staticmethod
    def calculateDistances(training_data, test_instance):
        # sqrt((x2-x1)**2+(y2-y1)**2)
        subtract = np.subtract(training_data, test_instance)
        square = np.square(subtract)
        euclid_sum = np.sum(square, axis=1)
        euclidean_distances = np.sqrt(euclid_sum)
        return euclidean_distances, np.argsort(euclidean_distances)

    @staticmethod
    def calculateDistancesManhattan(training_data, test_instance):
        #     |x-y| +.... so no square
        # Forgot axis earlier
        manhattan_distance = np.sum(np.absolute(np.subtract(training_data,test_instance)), axis=1)
        return manhattan_distance, np.argsort(manhattan_distance)

    @staticmethod
    def calculateDistancesMinkowski(training_data, test_instance,minkowski_factor):
        #     Factor = 1 or 2 1 for manhattan 2 for eulcidean
        #   (|x-y|**a + ....)**1/a
        minkowski_distance = np.power(np.sum(np.power(np.absolute(np.subtract(training_data, test_instance)),minkowski_factor), axis=1),
            np.divide(1,minkowski_factor))
        return minkowski_distance,np.argsort(minkowski_distance)

    # END OF PART -1
    # BEGINNNING OF PART-2
    def distance_weight_based(self,training_data, distance,euclidean_indices, knn_k_value):
        # From the training data, go to the shortest distance if k=1 or distances if k >1
        #  from this index take out the last element that contains the class
        k_instance_class =training_data[euclidean_indices[:knn_k_value]][:, -1]
        # Similarly, take the same for the distance from the dist. array
        k_instance_distance_sorted = distance[euclidean_indices[:knn_k_value]]
        # weight = sum ( (1/square of distance)  - Formula from lecture slide
        k_weight= np.divide(1,np.square(k_instance_distance_sorted))
        # Now we need to take the effecive sum of these weights
        # First check each individual class weight
        class_0_weight = np.sum(k_weight[k_instance_class == 0])
        class_1_weight = np.sum(k_weight[k_instance_class == 1])
        class_2_weight = np.sum(k_weight[k_instance_class == 2])
        # print("Class 0 : {} Class 1 : {} Class 2 : {}".format(class_0_weight,class_1_weight,class_2_weight))
        # As in the slides the class with largest weighted sum should be selected
        # as mentioned in part 1 take equal cases as well
        if class_2_weight >= class_1_weight and class_2_weight>=class_0_weight:
            # print("Returned 2")
            return 2
        if class_1_weight >= class_0_weight and class_1_weight>= class_2_weight:
            # print("Returned 1")
            return 1
        if class_0_weight >= class_1_weight and class_0_weight>= class_1_weight:
            # print("Returned 0")
            return 0
# Will be used in main option 2


def resetCounter(a,b):
    # Reset Counter
    a,b =0,0
    return a,b


def main():
    training_file = "data\\classification\\trainingData.csv"
    test_file = "data\\classification\\testData.csv"
    training_data = np.genfromtxt(training_file, delimiter=",")
    test_data = np.genfromtxt(test_file, delimiter=",")
    knn = KNN()
    training_data_values, training_data_class = knn.splitarray(training_data)
    test_data_values, test_data_class = knn.splitarray(test_data)
    test_ins = 0
    correct,incorrect =0,0
    # For matplot
    k_values = []
    accuracies = []
    ####################CHANGE K VALUE HERE

    knn_value = k =10

    #########CHANGE OPTION HERE OPTION 1 FOR PART 2(a),OPTION 2 FOR 2(B),OPTION 3 for testing with K-values#####################################


    option=3

   ########################################
    if option==1:
        for test_instance in test_data:
            test_instance_values = test_instance[:10]
            test_ins +=1
            distance,index = knn.calculateDistances(training_data_values,test_instance_values)
            weight_based = knn.distance_weight_based(training_data,distance,index,k)
            if weight_based == test_instance[10]:
                correct +=1
            else:
                incorrect +=1
        print("Correct : {} Incorrect : {}".format(correct,incorrect))
        print((correct/(correct+incorrect))*100)
    if option==2:
        for test_instance in test_data:
            test_instance_values = test_instance[:10]
            test_ins += 1
            distance, index = knn.calculateDistances(training_data_values, test_instance_values)
            euclidean_weight_based = knn.distance_weight_based(training_data, distance, index, k)
            if euclidean_weight_based == test_instance[10]:
                correct += 1
            else:
                incorrect += 1
        print("Correct Euclidean : {} Incorrect : {}".format(correct, incorrect))
        print("Euclidaean : {}".format((correct / (correct + incorrect)) * 100))
        correct,incorrect = resetCounter(correct,incorrect)
        for test_instance in test_data:
            test_instance_values = test_instance[:10]
            distance,manhattan_indices = knn.calculateDistancesManhattan(training_data_values,test_instance_values)
            manhattan_weight_based = knn.distance_weight_based(training_data,distance,manhattan_indices ,k)
            if manhattan_weight_based == test_instance[10]:
                correct +=1
            else:
                incorrect +=1
        print("Correct Manhattan : {} Incorrect Manhattan : {}".format(correct,incorrect))
        print("Manhattan : {}".format((correct/(correct+incorrect))*100))
        correct,incorrect = resetCounter(correct,incorrect)
        ################## Setting a =1 as per formula to obtain manhattan variant##########
        minkowski_manhattan_factor = 1
        for test_instance in test_data:
            test_instance_values = test_instance[:10]
            distance,  minkowski_manhattan_indices = knn.calculateDistancesMinkowski(training_data_values, test_instance_values,
                                                                          minkowski_manhattan_factor)
            minkowski_manhattan_weight_based = knn.distance_weight_based(training_data, distance, minkowski_manhattan_indices, k)
            if minkowski_manhattan_weight_based == test_instance[10]:
                correct += 1
            else:
                incorrect += 1
        print("Correct Minkowski-Manhattan : {} Incorrect Minkowski-Manhattan : {}".format(correct, incorrect))
        print("Minkowski-Manhattan : {}".format((correct / (correct + incorrect)) * 100))
        correct,incorrect = resetCounter(correct,incorrect)
        ######### Setting a =2 as per formula to obtain euclidean variant##########
        minkowski_euclidean_factor = 2
        for test_instance in test_data:
            test_instance_values = test_instance[:10]
            distance, minkowski_euclidean_indices = knn.calculateDistancesMinkowski(training_data_values, test_instance_values,
                                                                          minkowski_euclidean_factor)
            weight_based = knn.distance_weight_based(training_data, distance, minkowski_euclidean_indices, k)
            if weight_based == test_instance[10]:
                correct += 1
            else:
                incorrect += 1
        print("Correct Minkowski-Euclidean : {} Incorrect Minkowski-Manhattan : {}".format(correct, incorrect))
        print("Minkowski-Euclidean : {}".format((correct / (correct + incorrect)) * 100))



    if option ==3:
        for k_value in range(1, 101):
            knn_value = k = k_value
            k_values.append(k)
            for test_instance in test_data:
                test_instance_values = test_instance[:10]
                test_ins += 1
                distance, index = knn.calculateDistances(training_data_values, test_instance_values)
                weight_based = knn.distance_weight_based(training_data, distance, index, k)
                if weight_based == test_instance[10]:
                    correct += 1
                else:
                    incorrect += 1
            print("K-value : ",knn_value)
            print("Correct : {} Incorrect : {}".format(correct, incorrect))
            print((correct / (correct + incorrect)) * 100)
            accuracies.append((correct / (correct + incorrect)) * 100)
            correct,incorrect=0,0
        plt.plot(k_values, accuracies)
        plt.legend(['Accuracies'], loc='upper left')
        plt.xlabel('K- Value')
        plt.ylabel('Accuracy in distance-weighted variant')
        plt.show()
    print("Option Chosen : {}".format(option))
    # print(test_ins)


if __name__ == '__main__':
    main()