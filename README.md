# Bengaluru Real Estate House Price Prediction

In this project, the main aim is to predict the prices of houses available in Bengaluru.
The price is mainly based upon the Area, Number of Bedrooms, Bathrooms, and the Locality of the house itself.

It is a Machine Learning project with frontend and backend. The front end was developed and deployed using Flask Server. The backend is powered by Sklearn.

For model and modeling, the first step was data preprocessing which includes outlier detection, anomaly detection, variation, and many other things. It was the toughest part I can say. Once the data was cleaned, the next step was model building, for that GridsearchCV was used with linear regression, lasso, and a decision tree. The straight winner was LinerRegression. Once this was done, the last step was to deploy the project using the flask server.

## Tools and Technologies Used:

### Python Modules used:
- Flask Server: For Frontend development
- Sklearn, Numpy, and Pandas: for data preprocessing and model development
- Matplotlib: For data visualization
- joblib: for loading and saving the model
- webbrowser: for frontend automation

### Softwares Used 
- Spyder 4.3.4
- CMD
- Git Bash

### OS Used:
- Windows 10
- RHEL 8

## Features  
- House price prediction based on Area, Rooms, Bathrooms, and Locality.
- It's limited to Bengaluru only

## Setting up on local machine: 
1. Open git bash 
2. git clone https://github.com/Ddhruv-IOT/Bengaluru-House-Price-Prediction.git
3. Install all the dependencies using requirements.txt (optional step):
	<pre> pip install -r requirements.txt </pre>
4. Run the following command:
	<pre> python start.py</pre>

## Demo Video 
<p align="center"> <img src="https://github.com/Ddhruv-IOT/Home-Price-Prediction-Project/blob/main/demo/demo.gif" alt="animated" /> </p>

# Thank you
- Thank you all for using my app.
- All suggestions are warmly welcomed.
