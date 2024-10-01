# Karan's Projects

Welcome to my collection of projects! Below is a list of the various projects I have worked on, showcasing my skills in machine learning, web development, and more.

---

## Movie Recommendation System

### Project Overview
This project is a **Movie Recommendation System** that suggests movies based on user input and historical data of user preferences. The system aims to provide users with personalized movie recommendations by leveraging advanced techniques such as machine learning, natural language processing (NLP), and APIs from **TMDb (The Movie Database)** and **YouTube**.

The project incorporates both content-based and collaborative filtering approaches to **enhance the recommendation accuracy by 65%**. Users can interact with the system via a **dynamic web application** built using **Streamlit**, integrated with **HTML** and **CSS** for an improved user experience. The application also displays movie trailers, fetched through the **YouTube API**, for a more engaging interaction.

### Technologies Used
- **Machine Learning**: 
  - Algorithms for recommendation such as cosine similarity, collaborative filtering, and content-based filtering.
- **Natural Language Processing (NLP)**:
  - Used to extract features such as movie descriptions, genres, and keywords for better content recommendations.
- **APIs**:
  - **TMDb API**: For fetching detailed information on movies such as titles, posters, and genres.
  - **YouTube API**: For displaying trailers of the recommended movies.
- **Web Technologies**:
  - **Streamlit**: For creating the interactive web interface that allows users to input their preferences.
  - **HTML & CSS**: For enhancing the visual appeal and layout of the web application.

### Key Features

1. **Personalized Movie Recommendations**:
   - Based on both user input (such as favorite movies or genres) and historical data, the system suggests movies that match user preferences.
   - Uses a hybrid approach of **content-based filtering** and **collaborative filtering** to improve recommendation relevance.

2. **NLP for Enhanced Content Recommendations**:
   - Movie plots and descriptions are processed using natural language processing to identify key features like **genres**, **keywords**, and **descriptions**. This helps improve content-based recommendations by focusing on what the user might enjoy based on past preferences.

3. **Real-Time Interaction**:
   - Users can input a movie title, and the system instantly recommends similar movies. These recommendations are based on the calculated similarity between the user-selected movie and other movies in the dataset.

4. **Movie Trailers**:
   - Once a movie is recommended, its trailer is fetched via the **YouTube API** and embedded within the application. Users can watch trailers of the recommended movies directly within the app, making the experience more interactive and immersive.

5. **User-Friendly Interface**:
   - The web app is designed using **Streamlit**, with an intuitive and responsive UI created using **HTML** and **CSS**. Users can easily navigate the interface and obtain recommendations without needing any technical expertise.

6. **Efficient Data Handling**:
   - The system can handle large datasets with thousands of movies and their related metadata. Through efficient use of **pandas** and **NumPy** libraries in Python, the application processes the data and provides recommendations quickly.

### Technical Approach

- **Data Collection**:
  - Movie data is collected from **TMDb** API, including metadata like movie titles, genres, keywords, and posters.
  
- **Content-Based Filtering**:
  - A **cosine similarity** algorithm is used to find movies similar to the user's selected movie based on movie features (e.g., genre, cast, description). The similarity matrix is built using a **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization of the movie descriptions and keywords.

- **Collaborative Filtering**:
  - The system utilizes user-rating data to recommend movies based on what other users with similar preferences have liked. This is done using matrix factorization techniques and nearest neighbors to predict user ratings for unseen movies.

- **YouTube Integration**:
  - After generating a movie recommendation, the system retrieves the corresponding movie trailer from YouTube using the **YouTube Data API**. This creates a richer and more engaging user experience.

### How It Works

1. **Input**:
   - The user enters a movie title or selects from predefined genres or categories (e.g., action, comedy, drama).
  
2. **Processing**:
   - The system calculates the similarity scores between the selected movie and all other movies in the dataset using **cosine similarity** on movie features extracted through **NLP**.
   - If using collaborative filtering, the system looks at other users' ratings for movies that are similar to those liked by the current user.

3. **Output**:
   - A list of movies that are most similar to the user-selected movie is displayed. Each recommended movie includes its title, poster, genre, and a link to watch its trailer.

4. **Trailer Display**:
   - The YouTube trailer of the recommended movie is fetched and displayed directly on the same page.

### Challenges and Solutions

- **Challenge**: Managing a large dataset of movies efficiently.
  - **Solution**: Efficient use of data processing libraries like `pandas` and `NumPy` allowed fast data manipulation and filtering.
  
- **Challenge**: Providing high-quality recommendations with minimal user input.
  - **Solution**: Implementing a hybrid recommendation approach that combines content-based filtering with collaborative filtering ensures both user preferences and general trends are taken into account.

- **Challenge**: Embedding YouTube trailers without slowing down the app.
  - **Solution**: The app dynamically fetches trailers only when a recommendation is made, reducing load time and ensuring smooth performance.

## Conclusion

This Movie Recommendation System provides a user-friendly platform for personalized movie suggestions, leveraging advanced machine learning and NLP techniques. The integration of **YouTube trailers** adds an engaging touch, making it easier for users to decide what to watch. This project showcases my ability to combine data science, web technologies, and APIs to build a practical, real-world application.

<a href="https://github.com/Karanraj-6/Movie-Recommendation-System">View Project</a>

---

## Customer Churn Prediction System

### Project Overview
The **Customer Churn Prediction System** is designed to predict whether a customer will leave a service provider (churn) based on historical data and real-time user inputs. This system enables businesses to make data-driven decisions by identifying customers at risk of churning, allowing them to take proactive steps to retain them.

A key feature of this system is the **real-time storage of user data into a database**. This ensures that every customer input and prediction result is stored securely for future analysis, tracking, and reporting. The project is developed using **machine learning** algorithms for prediction, while the web interface is built with **Flask**, **HTML**, and **CSS**.

The system was trained on the **IBM Telco Customer Churn dataset** with over 7,000 records, and significant performance improvements were achieved:
<h4>immproments by applying advance preprocessing techniques and models with hyperparameter tuning</h4>
- **Accuracy improved by 18%**
- **Precision increased by 34%**
- **Recall boosted by 36%**

## Technologies Used
- **Machine Learning**:
  - Algorithms like **Logistic Regression**, **Random Forest**, and **Gradient Boosting** were tested. The final model was selected based on the best performance using evaluation metrics such as accuracy, precision, recall, and F1-score.
  
- **Data Preprocessing**:
  - The data was cleaned and preprocessed using techniques such as handling missing values, feature scaling, and one-hot encoding for categorical variables like `InternetService`, `PaymentMethod`, and `Contract`.

- **Flask**:
  - Flask was used to build the web interface, allowing users to input customer data and view churn predictions in real-time.

- **SQLite Database**:
  - A **SQLite database** is integrated into the system to **store user inputs and prediction results**. Each time a user enters customer information, the system stores this data alongside the churn prediction, enabling future retrieval and analysis. This feature is crucial for tracking customer behavior and improving retention strategies over time.

- **HTML & CSS**:
  - For designing an intuitive and user-friendly web interface where users can input data and view predictions.

## Key Features

1. **Real-Time Churn Prediction**:
   - Users can input customer details (e.g., tenure, contract type, monthly charges), and the system predicts whether the customer is likely to churn.

2. **Data Storage in SQLite Database**:
   - Every time a prediction is made, both the **user input** and the **prediction result** are stored in an SQLite database. This data can be accessed later for analysis and reporting, making it easier for businesses to track at-risk customers and make data-driven decisions.
   - **Database Fields**:
     - Customer information: `gender`, `tenure`, `Contract`, `MonthlyCharges`, etc.
     - Prediction result: Whether the customer is likely to churn (Yes/No).

3. **Improved Model Performance**:
   - After extensive data preprocessing and feature engineering, the model improved its accuracy by 18%, with precision increased by 34% and recall by 36%. This ensures that the system provides reliable predictions to help businesses retain customers.

4. **User-Friendly Interface**:
   - Built using **Flask**, the system provides an intuitive web interface for users to easily input data, receive predictions, and track customer churn.

5. **Real-Time Data Entry and Prediction Logging**:
   - Each prediction event logs the following into the database:
     - Customer details
     - Prediction (Churn or No Churn)
   - This helps in generating reports on the churn likelihood for various customers over time.

6. **Efficient Data Handling**:
   - The system is capable of handling large datasets and real-time data input, making it scalable for real-world applications in industries such as telecommunications, banking, and e-commerce.

## Dataset Overview
The dataset used in this project is the **IBM Telco Customer Churn** dataset, consisting of 7,043 customer records and 19 features that influence customer churn, such as:

- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`
- `Churn` (target)

## Technical Approach

1. **Data Preprocessing**:
   - Missing values were handled, categorical variables were converted to numerical representations via one-hot encoding, and feature scaling was applied to numerical columns such as `tenure` and `MonthlyCharges`.

2. **Model Development**:
   - Several models were trained and evaluated, including Logistic Regression, Random Forest, and Gradient Boosting. After evaluating the models on accuracy, precision, recall, and F1-score, **Gradient Boosting** was chosen for its superior performance.

3. **Database Integration**:
   - The system uses **SQLite** to store user input and predictions. Every user interaction with the prediction system is logged in the database for future analysis.

4. **Flask Web App**:
   - Flask is used to build a web interface where users can input customer details and view predictions. The app is responsive and user-friendly, ensuring a seamless experience for non-technical users.

## How It Works

1. **User Input**:
   - The user enters customer data (e.g., gender, tenure, contract type) into the web app.
  
2. **Prediction**:
   - The machine learning model predicts whether the customer is likely to churn based on the input data.

3. **Data Storage**:
   - Both the user’s input data and the churn prediction result are stored in the **SQLite database**. This allows for future retrieval, analysis, and reporting.

4. **Churn Result**:
   - The system displays whether the customer is predicted to churn or not, helping businesses identify at-risk customers in real time.

## Future Improvements

- **Advanced Model Development**:
  - Experimenting with deep learning models, such as neural networks, to further improve prediction accuracy.

- **User Authentication**:
  - Adding a login feature to allow businesses to securely access their customer churn data and track customer behavior over time.

- **Visualization**:
  - Adding data visualization tools to show trends in customer churn predictions.

- **API Integration**:
  - Developing an API to allow integration with other business systems for seamless data exchange.

## Conclusion

The **Customer Churn Prediction System** offers a reliable solution for predicting customer churn and storing real-time data into a database. With its user-friendly interface, robust prediction capabilities, and data storage features, it provides businesses with the tools they need to identify and retain at-risk customers effectively.

## 3. **Fake News Detection System**

**Description**: A robust system for detecting fake news, built using a dataset of 8,000 news articles. Achieved perfect accuracy (1.00), precision, recall, and F1-score through extensive hyperparameter tuning of the Gradient Boosting Classifier.

**Technologies**: Python, Gradient Boosting Classifier, Machine Learning.

**Features**:
- High-accuracy fake news detection.
- Well-tuned model for real-world application.
- Easy-to-use web interface.

<a href="https://github.com/Karanraj-6/Fake-News-Detection">View Project</a>

---

## 4. **Amazon ML Challenge 2024**

**Description**: A machine learning model to extract entity values such as weight, volume, voltage, wattage, and dimensions from images. This project plays a crucial role in healthcare, e-commerce, and content moderation applications.

**Technologies**: Python, Machine Learning.

**Features**:
- Accurate extraction of entity values from images.
- Handles large datasets with 263,860 rows.
- Efficient image processing and prediction generation.

<a href="https://github.com/Karanraj-6/Amazon-ML-Challenge-2024">View Project</a>

---

## 5. **YouTube AdView Prediction Project** (InternStudio)

**Description**: Predicts ad views on YouTube videos using a semi-large dataset of 15,000 entries. Developed during an internship, this project enhances ad targeting and performance evaluation.

**Technologies**: Python, Machine Learning.

**Features**:
- Predicts YouTube ad views with high accuracy.
- Handles large data for effective model training.
- Data preprocessing for improved performance.

<a href="https://github.com/Karanraj-6/YouTube-AdView-Prediction">View Project</a>

---

## 6. **Facial Recognition System for Mood Detection** (LetsGrowMore)

**Description**: A system that detects a person’s mood using machine learning and image processing, recommending songs based on the detected mood.

**Technologies**: Python, Machine Learning, Image Processing, Music Recommendation.

**Features**:
- Mood detection from facial expressions.
- Music recommendation system based on detected mood.
- Image processing for real-time emotion analysis.

<a href="https://github.com/Karanraj-6/Facial-Recognition-Mood-Detection">View Project</a>

---

## 7. **Handwritten Equation Solver** (LetsGrowMore)

**Description**: A system that recognizes and solves handwritten mathematical equations using Convolutional Neural Networks (CNN).

**Technologies**: Python, CNN, Deep Learning, Image Processing.

**Features**:
- Solves handwritten equations accurately.
- Utilizes CNN for deep learning-based recognition.
- Efficient image processing for equation extraction.

<a href="https://github.com/Karanraj-6/Handwritten-Equation-Solver">View Project</a>

---

## 8. **Stock Market Prediction System** (LetsGrowMore)

**Description**: A stock market prediction and forecasting system using Stacked LSTM, designed to provide insights into stock trends.

**Technologies**: Python, LSTM, Machine Learning.

**Features**:
- Predicts stock market trends using LSTM.
- Time-series forecasting for financial data.
- Visualizes trends and predictions.

<a href="https://github.com/Karanraj-6/Stock-Market-Prediction">View Project</a>

---

Feel free to explore each project by clicking on the links provided!

---

**Contact Me**:  
Email: karan@example.com  
GitHub: [Karan's GitHub](https://github.com/Karanraj-6)
