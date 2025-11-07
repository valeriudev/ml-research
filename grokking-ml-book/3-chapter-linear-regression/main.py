import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set interactive backend
import matplotlib.pyplot as plt

# Pure NumPy way (hardcode for simplicity)
x = np.array([50, 60, 65, 70, 80, 85, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220])
y = np.array([150, 180, 195, 210, 240, 255, 270, 300, 320, 350, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560])

# parameters
X = np.vstack([np.ones(len(x)), x]).T # [1, x]

learning_rate = 0.0000001 # only very small rate works
iterations = 5000
observations_count = len(y)
w = np.ones(2)  # init function weights and bias [b, w] = [1, 1]
error = 0

for i in range(0, iterations):
    predictions = np.dot(X, w) # matrix product of weights and features, that gives a new matrix
    error = predictions - y # find the error of each prediction in the points where we know the correct value (learning data), this gives a vector
    gradient = np.dot(X.T, error) / observations_count # calculated by current weights to know how much and in which direction to should update the weights to reduce the error
    w = w - (learning_rate * gradient)

# Try to predict a new value, x = 51
predict_value = 94
x_new = np.array([1, predict_value])
y_pred = np.dot(x_new, w)  # Prediction: y = b + w_1 * x
print(f'Final error: ', np.sum(error))
print(f"Predicted value for x = {predict_value}: {y_pred:.2f}")


# Plot original data points
plt.scatter(x, y, color='blue', label='Data')

# Plot the fitted line
x_line = np.linspace(x.min(), x.max(), 100)
y_line = w[0] + w[1] * x_line
plt.plot(x_line, y_line, color='red', label=f'Fit: y = {w[0]:.2f} + {w[1]:.2f}x')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()