from transformers import pipeline

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline(
    "sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Initialize the emotion analysis pipeline
emotion_analyzer = pipeline(
    "text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

# Path to your text file
file_path = 'business_H.txt'

# Initialize lists to store reviews
very_negative_reviews = []
negative_reviews = []
neutral_reviews = []
positive_reviews = []
very_positive_reviews = []

# Grouped categories
all_negative_reviews = []
all_positive_reviews = []

# Read reviews from the file
with open(file_path, 'r', encoding='utf-8') as file:
    reviews = file.readlines()

# Perform sentiment analysis on each review
for review in reviews:
    clean_review = review.strip()
    if not clean_review:
        continue
    sentiment_result = sentiment_analyzer(clean_review)
    sentiment_label = sentiment_result[0]['label']

    # Categorize reviews based on sentiment rating
    if sentiment_label == '1 star':
        very_negative_reviews.append(clean_review)
        all_negative_reviews.append(clean_review)  # Group into negative
    elif sentiment_label == '2 stars':
        negative_reviews.append(clean_review)
        all_negative_reviews.append(clean_review)  # Group into negative
    elif sentiment_label == '3 stars':
        neutral_reviews.append(clean_review)
    elif sentiment_label == '4 stars':
        positive_reviews.append(clean_review)
        all_positive_reviews.append(clean_review)  # Group into positive
    elif sentiment_label == '5 stars':
        very_positive_reviews.append(clean_review)
        all_positive_reviews.append(clean_review)  # Group into positive

# Function to calculate emotion distribution


def calculate_emotion_totals(reviews):
    emotions_count = {
        "joy": 0,
        "sadness": 0,
        "anger": 0,
        "surprise": 0,
        "disgust": 0,
        "fear": 0,
        "neutral": 0
    }

    for review in reviews:
        emotion_result = emotion_analyzer(review)
        emotion_label = emotion_result[0][0]['label']
        emotions_count[emotion_label] += 1

    return emotions_count


# Calculate emotions for each review group
very_negative_emotions = calculate_emotion_totals(very_negative_reviews)
negative_emotions = calculate_emotion_totals(negative_reviews)
neutral_emotions = calculate_emotion_totals(neutral_reviews)
positive_emotions = calculate_emotion_totals(positive_reviews)
very_positive_emotions = calculate_emotion_totals(very_positive_reviews)

# Function to print total counts


def print_review_counts():
    print("\nTotal Review Counts:")
    print("----------------------")
    print(f"Total Positive Reviews: {len(positive_reviews)}")
    print(f"Total Very Positive Reviews: {len(very_positive_reviews)}")
    print(f"Total Negative Reviews: {len(negative_reviews)}")
    print(f"Total Very Negative Reviews: {len(very_negative_reviews)}")

    print(f"\nTotal Positive Reviews (including both positive and very positive): {
          len(all_positive_reviews)}")
    print(f"Total Neutral Reviews: {len(neutral_reviews)}")
    print(f"Total Negative Reviews (including both negative and very negative): {
          len(all_negative_reviews)}")

# Function to print emotion distribution


def print_emotion_distribution():
    print("\nEmotion Distribution:")
    print("----------------------")
    print(f"Very Negative Reviews: {very_negative_emotions}")
    print(f"Negative Reviews: {negative_emotions}")
    print(f"Neutral Reviews: {neutral_emotions}")
    print(f"Positive Reviews: {positive_emotions}")
    print(f"Very Positive Reviews: {very_positive_emotions}")


# Call the functions to print
print_review_counts()
print_emotion_distribution()
