import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('ggplot')

from bs4 import BeautifulSoup
from datetime import datetime
import nltk

# Ensure nltk is installed with necessary packages
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

# Header to set the requests as a browser request
headers = {
    'authority': 'www.amazon.com',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,/;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-language': 'en-US,en;q=0.9,bn;q=0.8',
    'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
}

# URL of the Amazon Review page
reviews_url = 'https://www.amazon.com/i9-14900K-Desktop-Processor-Integrated-Graphics/product-reviews/B0CGJDKLB8/'

# Define Page No
len_page = 44

### Functions ###

# Function to scrape Amazon reviews
def scrape_amazon_reviews(url, len_page):
    soups = []

    for page_no in range(1, len_page + 1):
        params = {
            'ie': 'UTF8',
            'reviewerType': 'all_reviews',
            'filterByStar': 'critical',
            'pageNumber': page_no,
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            soups.append(soup)
        else:
            print(f"Failed to retrieve page {page_no}. Status code: {response.status_code}")

    return soups

# Function to extract review data from HTML
def get_reviews_data(html_data):
    reviews_data = []

    boxes = html_data.select('div[data-hook="review"]')

    for box in boxes:
        try:
            name = box.select_one('[class="a-profile-name"]').text.strip()
        except Exception as e:
            name = 'N/A'

        try:
            stars = box.select_one('[data-hook="review-star-rating"]').text.strip().split(' out')[0]
        except Exception as e:
            stars = 'N/A'

        try:
            title_with_rating = box.select_one('[data-hook="review-title"]').text.strip()
            review_title = title_with_rating.split('\n', 1)[1].strip()
        except Exception as e:
            review_title = 'N/A'

        try:
            datetime_str = box.select_one('[data-hook="review-date"]').text.strip().split(' on ')[-1]
            date = datetime.strptime(datetime_str, '%B %d, %Y').strftime("%d/%m/%Y")
        except Exception as e:
            date = 'N/A'

        try:
            description = box.select_one('[data-hook="review-body"]').text.strip()
        except Exception as e:
            description = 'N/A'

        data_dict = {
            'Name': name,
            'Stars': stars,
            'Title': review_title,
            'Date': date,
            'Description': description
        }

        reviews_data.append(data_dict)

    return reviews_data

### Data Processing ###

# Scrape reviews
html_datas = scrape_amazon_reviews(reviews_url, len_page)

# Extract review data from HTML
reviews = []
for html_data in html_datas:
    reviews += get_reviews_data(html_data)

# Create DataFrame
df_reviews = pd.DataFrame(reviews)

# Print DataFrame to check
print(df_reviews.head())

# Plotting the review ratings
ax = df_reviews['Stars'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Count of Reviews by Stars',
          figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()

# Tokenizing the 50th review
example = df_reviews['Description'][50]
print(example)
tokens = nltk.word_tokenize(example)
print(tokens[:10])
tagged = nltk.pos_tag(tokens)
print(tagged[:10])
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()

# Sentiment Analysis with VADER
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
sentiment = sia.polarity_scores(example)
print(sentiment)

# Add IDs to DataFrame
ids = range(1, len(df_reviews) + 1)
df_reviews.insert(0, 'ID', ids)

from tqdm import tqdm

# Run the polarity score on the entire dataset
res = {}
for i, row in tqdm(df_reviews.iterrows(), total=len(df_reviews)):
    text = row['Description']
    myid = row['ID']
    res[myid] = sia.polarity_scores(text)

# Convert the results dictionary to DataFrame
vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'ID'})

# Merging sentiment scores with the reviews DataFrame
vaders = vaders.merge(df_reviews, on='ID')

# Displaying the first few rows of the merged DataFrame
print(vaders.head())

# Plotting the compound sentiment score by Amazon star review
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Stars', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Stars', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Stars', y='neg', ax=axs[2])
axs[0].set_title('Positive Sentiment')
axs[1].set_title('Neutral Sentiment')
axs[2].set_title('Negative Sentiment')
plt.tight_layout()
plt.show()

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Load the model and tokenizer for RoBERTa
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Function to get sentiment scores using RoBERTa
def polarity_scores_roberta(text):
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

# Apply RoBERTa sentiment analysis to the reviews
res = {}
for i, row in tqdm(df_reviews.iterrows(), total=len(df_reviews)):
    try:
        text = row['Description']
        myid = row['ID']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {f"vader_{key}": value for key, value in vader_result.items()}
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')

# Convert the results dictionary to DataFrame
results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'ID'})
results_df = results_df.merge(df_reviews, on='ID')

# Displaying the first few rows of the merged DataFrame
print(results_df.head())

# Plotting the compound sentiment scores by Amazon star review
sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                   'roberta_neg', 'roberta_neu', 'roberta_pos'],
             hue='Stars',
             palette='tab10')
plt.show()

# Check and print reviews with highest and lowest sentiment scores
if not results_df.query('Stars == 1').empty:
    print(results_df.query('Stars == 1').sort_values('roberta_pos', ascending=False)['Description'].values[0])
    print(results_df.query('Stars == 1').sort_values('vader_pos', ascending=False)['Description'].values[0])
if not results_df.query('Stars == 5').empty:
    print(results_df.query('Stars == 5').sort_values('roberta_neg', ascending=False)['Description'].values[0])
