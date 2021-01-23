import numpy as np
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify, render_template
import pickle
from flask import Markup

app = Flask(__name__, template_folder='template')
linear_model = pickle.load(open('insurance_linear_regression.pkl', 'rb'))
#poly_model = pickle.load(open('insurance_poly_regression.pkl', 'rb'))
ridge_model = pickle.load(open('insurance_ridge_regression.pkl', 'rb'))
lasso_model = pickle.load(open('insurance_lasso_regression.pkl', 'rb'))
forest_model = pickle.load(open('insurance_forest_regression.pkl', 'rb'))

@app.route('/', defaults={'page': 'insuranceLanding'})

@app.route('/<page>')
def html_lookup(page):
    try:
        return render_template('{}.html'.format(page))
    except TemplateNotFound:
        abort(404)

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    linear_prediction = linear_model.predict(final_features)
    linear_output = round(linear_prediction[0], 2)

    #poly_prediction = poly_model.predict(final_features)
    #poly_output = round(poly_prediction[0], 2)

    ridge_prediction = ridge_model.predict(final_features)
    ridge_output = round(ridge_prediction[0], 2)

    lasso_prediction = lasso_model.predict(final_features)
    lasso_output = round(lasso_prediction[0], 2)

    forest_prediction = forest_model.predict(final_features)
    forest_output = round(forest_prediction[0], 2)

    linear_result = 'Predicated Medical Insurance Cost: {}'.format(linear_output)
    #poly_result = 'Predicated Medical Insurance Cost: {}'.format(poly_output)
    ridge_result = 'Predicated Medical Insurance Cost: {}'.format(ridge_output)
    lasso_result = 'Predicated Medical Insurance Cost: {}'.format(lasso_output)
    forest_result = 'Predicated Medical Insurance Cost: {}'.format(forest_output)

    return render_template('insuranceIndex.html', prediction_text1=linear_result, prediction_text3=ridge_result, prediction_text4=lasso_result, prediction_text5=forest_result)
	
if __name__ == "__main__":
    app.run(debug=True)