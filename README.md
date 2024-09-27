# Shipment-Analysis
# Sentiment Analysis and Feature Extraction with Web Scraping

This project demonstrates a pipeline for scraping text data from web pages, cleaning the data, extracting features using TF-IDF, and performing sentiment analysis using the `TextBlob` and `nltk` libraries. The results are then saved into a CSV file for further analysis.

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Code Explanation](#code-explanation)
  - [1. Web Scraping](#1-web-scraping)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Feature Extraction](#3-feature-extraction)
  - [4. Sentiment Analysis](#4-sentiment-analysis)
- [Usage](#usage)
- [Output](#output)

## Overview

The script performs the following tasks:

1. Scrapes data from a given website.
2. Cleans the text data by removing special characters, punctuation, and extra spaces.
3. Extracts features from the cleaned data using the `TfidfVectorizer`.
4. Analyzes the sentiment of the cleaned text data.
5. Saves the cleaned data, sentiment scores, and feature extraction results into a CSV file.

## Dependencies

- `numpy`
- `pandas`
- `requests`
- `BeautifulSoup4`
- `re`
- `nltk`
- `sklearn`
- `textblob`

Make sure to install the required libraries using the following command:

    pip install numpy pandas requests beautifulsoup4 nltk scikit-learn textblob

## Installation
1. Clone the repository:

       https://github.com/Sherryyy00/Shipment-Analysis.git

2. Navigate to the project directory:

        cd sentiment-feature-extraction

3. Install the required Python packages:

       pip install -r requirements.txt

## Code Explanation
1. Web Scraping
The script uses the requests library to fetch the content of a webpage and BeautifulSoup for parsing HTML data. It collects all hyperlinks from the page and then retrieves the text content from each link.

        url = 'https://www.sciencedirect.com/science/article/abs/pii/S1361920999000309'
        response = requests.get(url).text
        soup = bs(response, "html.parser")
        
        link = [a['href'] for a in soup.find_all('a', href=True)]
   
3. Data Cleaning
The scraped text is cleaned by removing non-alphanumeric characters, punctuation, and extra spaces. The cleaned data is then stored in a pandas DataFrame.


        for j in range(len(data_text)):
            data_text[j] = re.sub(r'\W', " ", data_text[j])
            # Further cleaning steps
   
3. Feature Extraction
TF-IDF (Term Frequency-Inverse Document Frequency) is used for feature extraction. The TfidfVectorizer transforms the cleaned text into numerical feature vectors for use in machine learning or other analysis.

        tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('English'))
        x = tfidfconverter.fit_transform(df['cleaned']).toarray()
   
5. Sentiment Analysis
The sentiment of the cleaned text is analyzed using two methods:

1. TextBlob for polarity and subjectivity scores.
2. SentimentIntensityAnalyzer from the nltk library for detailed sentiment scores (negative, positive, neutral, compound).

        df['Polarity'] = df["cleaned"].apply(lambda x: TextBlob(x).sentiment.polarity)
        df['Subjectivity'] = df["cleaned"].apply(lambda x: TextBlob(x).sentiment.subjectivity)

The SentimentIntensityAnalyzer is used to calculate various sentiment metrics.


        for i in range(len(df.index)):
          score = SentimentIntensityAnalyzer().polarity_scores(df['cleaned'][i])
          neg.append(score['neg'])
          pos.append(score['pos'])
          neu.append(score['neu'])
          com.append(score['compound'])
          
5. Saving to CSV
The results, including cleaned text, sentiment analysis scores, and extracted features, are saved into a CSV file.

        df.to_csv('Feature Extraction.csv')
   
## Usage
### To run the script:

Make sure you have installed the required dependencies.
#### Run the Python script in your terminal:

        python sentiment_analysis.py
The results will be saved in a file named Feature Extraction.csv.

## Output
### Feature Extraction.csv: A CSV file containing:
* Cleaned text
* Sentiment scores (polarity, subjectivity, positive, negative, neutral, compound)
* Extracted TF-IDF features
* plaintext

This project provides a complete pipeline from web scraping to text processing and sentiment analysis, ideal for applications in natural language processing and data mining.


This `README.md` provides a comprehensive overview of the project, its setup, and functionality. You can adapt it to your project structure and repository details as needed.











