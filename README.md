# Book Recommendation System

The objective of the project is to help users to discover books based on their interests and perferences. The system leverages Machine Learning and Natural Language Processing to suggest books that are most relevant to users using collabrative filtering approach.

**Tools and Technologies:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seabor, NLTK, Flask for web-based interface.

### Project Structure

```plaintext
├── static/                  
├── templates/                
├── app.py                    
├── requirements.txt
├── README.md                 
└── notebooks/                
```

### Dataset

The dataset includes:

- *Book metadata*: Titles, authors, genres, publication year, etc.
- *User data*: User information such as user-id, location, age.
- *User ratings*: Ratings given by users to books out of 10.

You can download datasets from here: [data-source](https://www.kaggle.com/datasets/saurabhbagchi/books-dataset)

### Data Preparation

 Data cleaning and preprocessing performed on the books, users and ratings datasets.

- Clean and filter book data with valid ISBN number and consider only latest edition of the book.
- Remove the books with zero ratings from the ratings dataset and calcuate the total votes, ratings and average ratings for each book and merge dataset with book data.
- Perform text cleaning on book authors name and title.
- Remove duplicates from the dataset to create final book data.
- In user data, perform text analysis and create a new features such as country and city.
- Handle outliers in age column. Create a new columns indicating missing age and age group of the user.
- Final datasets are clean and saved for further analysis.

Data-cleaning notebook can be found here: [data-cleaning.ipynb](https://github.com/madhuri-15/book-recommendation/blob/main/notebooks/data-cleaning.ipynb)

### Data Analysis
- Perform analysis and visualize the results using visualization like bar charts and line graphs to understand the user demographics and book features.
![line-chart](https://github.com/madhuri-15/book-recommendation/blob/main/images/user_demographics_analysis.jpeg)

- Calculate the weighted average ratings for each book by considering only books with minimum 30 votes to find popular books.

![weighted-average-impact](https://github.com/madhuri-15/book-recommendation/blob/main/images/weighted_rating_impact.png)

### Model Building
We have a large amount of user data with book rating. So, we use collabrative filtering approach to recommed books based on similar liking between the users.

We consider users who voted more than 120 books and books with more than 50 votes for building our model.

- **Approach 1: Simple Search Keyword**: Book recommendation by simply searching similar keywords in data.
- **Approach 2: Pearson's Correlation**: Recommendations using pearson's correlations.
- **Approach 3: Cosine Similarity**: Item-based collaborative filtering using cosine similarity.

Recommendation model building notebook can be found here: [book-recommendation-system.ipynb](https://github.com/madhuri-15/book-recommendation/blob/main/notebooks/book-recommendation-system.ipynb)

### Model Evaluation

We evaluate the performance of recommendation of a particluar book genre of topK recommendations using precision, recall and f1score metrics.

For example: performance of `Fantacy` genre book.

![model-performance](https://github.com/madhuri-15/book-recommendation/blob/main/images/model_evaluation.jpeg)

### Model deployment

The `app.py` is a script for flask application to deploy the model locally and to test the model deployment.

#### Installation and Setup

1. **Clone the Repository**:
   
   ```bash
   git clone https://github.com/madhuri-15/book-recommendation.git
   cd book-recommendation
   ```
2. **Create a Virtual Environment**:
   
   ```bash
   python -m venv env
   source env\Scripts\activate
   ```
3. **Install Dependencies**:
   
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Application**:
   
   ```bash
   python app.py
   ```

### Usage

1. Open the application in your browser (default: `http://localhost:5000`).
   ![homepage](https://github.com/madhuri-15/book-recommendation/blob/main/images/homepage.png)
2. Home page displays the all time top 50 books.
3. You can search book by book title or keyword from home page.
4. Or you can also use advanced search option (api: `http://localhost:5000/advanced-search`) for particular age.
   
