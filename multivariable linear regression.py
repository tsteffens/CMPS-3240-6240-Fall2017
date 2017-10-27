import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error


#reads file into data
df = pd.read_csv('housing_prices.csv')

#splits data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df, df['price'],test_size = 0.3)


#intitializes variables for lists of training data
dftr = pd.DataFrame(X_train)
x_point = dftr['sqft'].tolist()
y_point = dftr['price'].tolist()
u_point = dftr['bedrooms'].tolist()
v_point = dftr['baths'].tolist()
w_point = dftr['City'].tolist()

#assign number values to cities
temp=[]
for city in w_point:
    if city not in temp:
        temp.append(city)

city_step = 1/len(temp)




#initializes variables for lists of testing data
dfts = pd.DataFrame(X_test)
test_sqft = dfts['sqft'].tolist()
test_price = dfts['price'].tolist()
test_rooms = dfts['bedrooms'].tolist()
test_baths = dfts['baths'].tolist()
test_city = dfts['City'].tolist()

#theta values for hypothesis
theta = [0,0,100000,100000]

#description of hypothesis equation
y = lambda x,u,v : theta[0] + theta[1]*x + theta[2]*u + theta[3]*v


#function for plotting hypothesis line on 2 variables
def plot_line(y, data_points):
    x_val = [i for i in range(int(min(data_points))-1, int(max(data_points))+2)]
    y_val = [y(x) for x in x_val]
    plt.plot(x_val, y_val, 'r')
    plt.show()

########### Learning Rate
learn = .000001
###########


#Solves the summation term in gradient descent
def summation(y,x_point,y_point,u_point,v_point):
    total1=0
    total2=0
    total3=0
    total4=0

    for i in range(1, len(x_point)):
        total1 += y(x_point[i], u_point[i], v_point[i]) - y_point[i]
        total2 += (y(x_point[i], u_point[i], v_point[i]) - y_point[i])* x_point[i]
        total3 += (y(x_point[i], u_point[i], v_point[i]) - y_point[i])* u_point[i]
        total4 += (y(x_point[i], u_point[i], v_point[i]) - y_point[i])* v_point[i]
        #print(total1)
        return total1/len(x_point), total2/len(x_point), total3/len(x_point), total4/len(x_point)

#updates theta values simultaneously
for i in range(1000):
    s1,s2,s3,s4 = summation(y, x_point, y_point, u_point, v_point)
    theta[0] = theta[0] - learn*s1
    theta[1] = theta[1] - learn*s2
    theta[2] = theta[2] - learn*s3
    theta[3] = theta[3] - learn*s4
    


###mean squared errors
train_predicted = []
for i in range(len(x_point)):
    train_predicted.append(y(x_point[i],u_point[i],v_point[i]))
    
print('Training Data mean squared value: ')
print(mean_squared_error(y_point, train_predicted))


test_predicted = []
for i in range(len(test_sqft)):
    test_predicted.append(y(test_sqft[i],test_rooms[i],test_baths[i]))
    #print(y(test_sqft[i]))
    
print('Testing Data mean squared value: ')
print(mean_squared_error(test_price, test_predicted))
print('Theta Values: ')
print(theta)


########Perceptron
def predict(instance, theta):
    T=800000 #threshold price for determining if a house is expensive
    activation = theta[0]
    for i in range(len(instance)):
        activation += theta[i+1]*instance[i]
    print(activation)
    return "Expensive" if activation >= T else "Not Expensive"

instance=[3392,2,2.1]
print(predict(instance,theta))
