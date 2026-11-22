# CSV Classifier with Flask

Upload a CSV, train a random forest and use a web form to classify a new observation. The included example uses four flower measurements to predict a numeric species label.

## Try the sample

```sh
python -m pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`, upload [the sample CSV](uploads/sample_dataset.csv), then open the prediction form. Try **5.1, 3.5, 1.4, 0.2** in order: sepal length, sepal width, petal length and petal width. The result is the class predicted by the most recently trained model.

## How training works

[The training function](train_auto.py) takes all columns except the last as features and the last column as the label. It holds out 20% of rows, fits a random forest and saves it to `models/model.pkl`. Each upload replaces that model. The [prediction form](templates/predict.html) expects four numeric inputs in the sample's column order.

The sample has ten rows, so the displayed accuracy comes from just two test examples and can change between runs. [Flask routes](app.py)
