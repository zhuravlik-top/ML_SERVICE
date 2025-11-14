from sklearn.linear_model import LinearRegression
import joblib
import os

X= [[1], [2], [3], [4]]
y = [2, 4, 6, 8]

model = LinearRegression()
model.fit(X, y)

#os.makedirs('models', exist_ok=True)

joblib.dump(model, "D:\ML_SERVICE\models\model_v1.pkl")
print("Model saved as D:\ML_SERVICE\models\model_v1.pkl")
