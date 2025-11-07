# Simple Linear Regression Exercise: Predicting House Prices

Inspired by the hands-on style in *Grokking Machine Learning* (Chapter 2, where you build linear regression from scratch with NumPy on basic datasets), this exercise uses a realistic synthetic dataset (20 houses) to predict house prices based on square meters. It's designed for practicing the least squares method and gradient calculations with NumPy only. No scikit-learn or Keras needed.

## Dataset

This is a synthetic dataset with 20 houses, representing square meters (X) and corresponding house prices in thousands of USD (Y), with realistic noise to mimic real-world data. You can copy-paste it into a file called `houses.csv` or load it directly in code.

| Square Meters | Price (thousands USD) |
|---------------|-----------------------|
| 50            | 150                   |
| 60            | 180                   |
| 65            | 195                   |
| 70            | 210                   |
| 80            | 240                   |
| 85            | 255                   |
| 90            | 270                   |
| 100           | 300                   |
| 110           | 320                   |
| 120           | 350                   |
| 130           | 380                   |
| 140           | 400                   |
| 150           | 420                   |
| 160           | 440                   |
| 170           | 460                   |
| 180           | 480                   |
| 190           | 500                   |
| 200           | 520                   |
| 210           | 540                   |
| 220           | 560                   |

### How to Load in NumPy

```python
import numpy as np

# Pure NumPy way (hardcode for simplicity)
X = np.array([50, 60, 65, 70, 80, 85, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220]).reshape(-1, 1)
y = np.array([150, 180, 195, 210, 240, 255, 270, 300, 320, 350, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560])