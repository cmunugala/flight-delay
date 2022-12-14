# flight-delay

In this project, me and my team (Ahmad Azizi, Kolby Devery, and Evan Chan) used large-scale flight and weather data to predict whether or not a flight would be delayed based on information two hours before the flight departure. Because the datasets were large, we were forced to use Spark for cleaning and feature engineering. 

Code was written in DataBricks notebooks in Python, PySpark, and SQL. 

The Final_Report.html walks through the project and all of design choices and results. This is the easiest way to get an idea of what the project was about. 




• In the Cleanup_feature_engineering folder, there are scripts that define our data lineage (data cleaning, joining, feature engineering, etc.). 
• In the EDA Notebooks, we have exploratory data analysis that we performed. Specifically, in the feature engineering EDA notebook, we have EDA on some of our more important engineered features such as prev_flight_delayed and mean_carrier_delay.   
• In Model_building, we have our scripts for the different models that we created, which include Logistic Regression, Random Forests, Gradient Boosting Trees, and Neural Networks.   
• The cross_validation_notebook folder contains scripts to implement our custom broken window cross validation strategy which is fitting for the time-series nature of our data.   



