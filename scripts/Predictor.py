from Predict import Predict
inference = Predict()
# Predict for new inputs
new_data = [
    [77,228,101],[28,115,180],[61,200,330]
]
predictions, classes = inference.predict(new_data)
print("Output:")
print(predictions)
print("Predicted class (0 = not at risk, 1 = at risk):")
print(classes)