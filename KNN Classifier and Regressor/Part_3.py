#Jason Shawn D' Souza R00183051
# Regression
# Change k value and options in main
import numpy as np
import time

class Regression(object):
    def __init__(self):
        pass

    def splitarray(self,arr):
        return arr[:, :-1], arr[:, -1]

    def calculateDistances(self, training_data, test_instance):
        # sqrt((x2-x1)**2+(y2-y1)**2)
        subtract = np.subtract(training_data, test_instance)
        square = np.square(subtract)
        euclid_sum = np.sum(square, axis=1)
        euclidean_distances = np.sqrt(euclid_sum)
        return euclidean_distances, np.argsort(euclidean_distances)

    def distance_weight_based(self,training_data, distance,euclidean_indices, knn_k_value):
        #As commented in part 2
        k_instance_class = training_data[euclidean_indices[:knn_k_value]][:, -1]
        k_instance_distance_sorted = distance[euclidean_indices[:knn_k_value]]
        # weight = sum ( ((1/square of distance) (that nearest point))/1/square of distancedistance)  - Formula from lecture slide
        # eg. for points 1,2,3 distances of 2,3 and 4 from given point
        #  (1/4 * 1 + 1/9 *2 +1/16 * 3)/ sum of squares i.e 1/4+1/9+1/16
        k_weight = np.divide(1, np.square(k_instance_distance_sorted))
        # Numerator of the formula multiply point and distance then sum (distance square already done in part 2)
        numerator = np.sum(np.multiply(k_instance_class, k_weight))
        # Denominator ,just add the weights
        denominator = np.sum(k_weight)
        regression_weighted_sum =np.divide(numerator,denominator)
        # print(regression_weighted_sum)
        return regression_weighted_sum


    def r_square(self,regression_value,test_data):
        # Last column i.e regression value column
        values = test_data[:, -1]
        # sum of square of residual data
        # i,e numerator is  square of (regression - values of testdata) or (values- regression) ,np subtract square and then sum as in formula
        numerator = np.sum(np.square(np.subtract(values,regression_value)))
        # denominator = TOTAL sum of squares (ybar - yi)**2 + ....
        # ybar/mean/avg of values
        mean_values = np.mean(values)
        # Denominator is (np.mean(values)-actualvalue[1])**2+...+np.mean(values)-actualvalue[n])**2 so similar np as numerator
        denominator = np.sum(np.square(np.subtract(mean_values, values)))
        second_component=np.divide(numerator,denominator)
        # r**2 = 1-(numerator/denom)
        r_square = np.subtract(1,second_component)
        return r_square

    def normalizeData(self,data):
        values = data[:,:12] #exclude class column
        #Normalization from slide - newValue =original - minValue/maxVal-minVal
        numerator =np.subtract(values,np.min(values))
        denominator = np.subtract(np.max(values),np.min(values))
        newValue =np.divide(numerator,denominator)
        return newValue
    



def main():
    training_file = "data\\regression\\trainingData.csv"
    test_file = "data\\regression\\testData.csv"
    training_data = np.genfromtxt(training_file, delimiter=",")
    test_data = np.genfromtxt(test_file, delimiter=",")
    reg = Regression()

    # correct, incorrect = 0, 0
    reg_list=[]
    ################ChANGE K VALUE########
    knn_value = k = 10
    # CHANGE OPTION HERE
    # OPTION 1 WIHTOUT DATA NORMALIZATION,
    # OPTION 2 WITH DATA NORMALIZATION
    option = 2
    # Without normalized Data
    if option ==1:
        training_data_values, training_data_class = reg.splitarray(training_data)
        test_data_values, test_data_class = reg.splitarray(test_data)
        for test_instance in test_data:
            test_instance_values = test_instance[:12]
            distance, index = reg.calculateDistances(training_data_values, test_instance_values)
            weight_based_sum = reg.distance_weight_based(training_data, distance, index, k)
            reg_list.append(weight_based_sum)
        r_square = reg.r_square(reg_list, test_data)
        print("K Value : {}".format(knn_value))
        print("Accuracy {} ".format(r_square))
        print("Option chosen : {}".format(option))
    # With normalized data
    if option ==2:
        training_data = reg.normalizeData(training_data)
        test_data = reg.normalizeData(test_data)
        for test_instance in test_data:
            test_instance_values = test_instance[:12]
            distance, index = reg.calculateDistances(training_data, test_instance_values)
            weight_based_sum = reg.distance_weight_based(training_data, distance, index, k)
            reg_list.append(weight_based_sum)
        r_square = reg.r_square(reg_list, test_data)
        print("K Value : {}".format(knn_value))
        print("Accuracy WITH normalized data {} ".format(r_square))
        print("Option chosen : {}".format(option))




if __name__=="__main__":
    main()