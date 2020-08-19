# titanic-model-app
https://titanic-model-app.herokuapp.com/

See Jupyter notebook for details

## Checklist

1. **Define the problem from a high-level view**

    This is to understand and articulate the business logic of the problem. It should tell you:
    the nature of the problem(supervised/unsupervised, classification/regression),
    type of solutions you can develop
    what metrics you should use to measure performance?
    is machine learning the right approach to solve this problem?
    manual approach to solving the problem.
    the inherent assumptions of the problem

1. **Identify the data sources and acquire the data**

    In most cases, this step can be executed before the first step if you have the data with you and you want to define the questions(problem) around it to make better use of the incoming data.

    Based on the definition of your problem, you’d need to identify the sources of data which can be a database, a data repository, censors, etc. For an application to be deployed in production, this step should be automated by developing data pipelines to keep the incoming data flowing into the system.

    list the sources and amount of data you need.
    check if space is going to be an issue.
    check if you’re authorized to use the data for your purpose or not.
    acquire the data and convert it into a workable format.
    check the type of data(textual, categorical, numerical, time series, images)
    take aside a sample of it for final testing purposes.

1. **Initial Exploratoration of Data**

    This is the step where you study all the features that impact your outcome/prediction/target. If you have a huge chunk of data, sample it down for this step to make the analysis more manageable.
    Steps to follow:

    use jupyter notebooks as they provide an easy and intuitive interface to study the data.
    identify the target variable
    identify the types of features(categorical, numerical, textual, etc.)
    analyze the correlation between features.
    add a few data visualizations for easy interpretation of the impact of each feature on the target variable.
    document your findings.

1. **Exploratory Data Analysis to Prepare the Data**

    It’s time to execute the findings of the previous step by defining functions for data transformations, cleaning, feature selection/engineering, and scaling.

    Write functions to transform the data and automate the process for the forthcoming batches of data.
    Write functions to clean the data(imputing missing values and handling outliers)
    Write functions to select and engineer features — drop redundant features, format conversion of features, and other mathematical transformations.
    Feature scaling — standardize the features.

1. **Develop a baseline model and then explore other models to shortlist the best ones**

    Create a very basic model which should serve as the baseline for all the other complex machine learning model. Checklist of steps:

    Train a few commonly used ML models like naive bayes, linear regression, SVM, etc using default parameters.
    Measure and compare the performance of each model with the baseline and with all the others.
    Employ N-fold cross-validation for each model and compute the mean and standard deviation of the performance metrics on the N folds.
    Study the features that have the most impact on the target.
    Analyze the types of errors the models make while predicting.
    Engineer the features in a different manner.
    Repeat the above steps a few times(trial and error) to be sure that we have used the right features in the right format.
    Shortlist the top models based on their performance measures.

1. **Fine-tune your shortlisted models and check for ensemble methods**

    This needs to be one of the crucial steps where you would be moving closer to your final solution. Major steps should include:

    Hyperparameter tuning using cross-validation.
    Use automated tuning methods like random search or grid search to find out the best configuration for your best models.
    Test ensemble methods like voting classifiers etc.
    Test the models with as much data as possible.
    Once finalized, use the unseen test sample that we set aside, in the beginning, to check for overfitting or underfitting.

1. **Document Code and Communicate your solution**

    The process of communication is manifold. You need to keep in mind all the existing and potential stakeholders. Therefore the major steps include:

    Document the code as well as your approach and journey throughout your project.
    Create a dashboard like voila or an insightful presentation with close to self-explanatory visualizations.
    Write a blog/report capturing how you analyzed the features, tested different transformations, etc. Capture your learning(failures and techniques that worked)
    Conclude with the main outcome and future scope(if any)

8. **Deploy your model in production, Monitor!**

    If your project requires deployment to be tested on live data, you should create a web application or a REST API to be used across all platforms(web, android, iOS). Major steps(would vary depending on the project) include:

    Save your final trained model into an h5 or pickle file.
    Serve your model using web services, you can use Flask to develop these web services.
    Connect the input data sources and set up the ETL pipelines.
    Manage dependencies using pipenv, docker/Kubernetes(based on scaling requirements)
    You can use AWS, Azure, or Google Cloud Platform to deploy your service.
    Monitor the performance on live data or simply for people to use your model with their data.

