import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
from datetime import datetime
import os
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="AI Mood Predictor", page_icon="🤖")

# Title
st.title("AI Mood Predictor & Activity Recommender")
st.write("This AI-powered app analyzes your emotions using machine learning!")

# Download NLTK resources (only first time)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# ==================== LOAD AND PREPARE DATASET ====================
def load_emotion_dataset():
    """Load and prepare the emotion dataset"""
    # Check if dataset files exist
    if not all(os.path.exists(f) for f in ['train.txt', 'val.txt', 'test.txt']):
        st.error("""
        Dataset files not found! Please download the Emotion Dataset from Kaggle:
        https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp
        
        Place train.txt, val.txt, and test.txt in the same directory as app.py
        """)
        return None
    
    # Read the three dataset files
    train_data = pd.read_csv('train.txt', sep=';', names=['text', 'emotion'])
    val_data = pd.read_csv('val.txt', sep=';', names=['text', 'emotion'])
    test_data = pd.read_csv('test.txt', sep=';', names=['text', 'emotion'])
    
    # Combine all data
    df = pd.concat([train_data, val_data, test_data], ignore_index=True)
    
    # Map dataset emotions to our moods
    emotion_to_mood = {
        'sadness': 'sad',
        'joy': 'happy',
        'love': 'happy',
        'anger': 'angry',
        'fear': 'scared',
        'surprise': 'surprised'
    }
    
    # Apply mapping
    df['mood'] = df['emotion'].map(emotion_to_mood)
    
    # Drop rows with missing mood
    df = df.dropna(subset=['mood'])
    
    # Add sample data for other moods not in the dataset
    additional_data = {
        'text': [
            # Stressed
            "I'm so stressed about work", "This deadline is overwhelming me", "Too much pressure",
            "I can't handle this stress", "Feeling anxious about everything", "Work is stressing me out",
            # Tired
            "I'm exhausted and need sleep", "Feeling so tired today", "I can barely keep my eyes open",
            "Need a nap badly", "Feeling drained and fatigued", "Too tired to do anything",
            # Calm
            "Feeling peaceful and relaxed", "I'm in a calm state of mind", "Everything is serene",
            "Feeling tranquil today", "I'm at peace with myself", "So calm and centered",
            # Bored
            "I'm so bored right now", "Nothing interesting to do", "This is so dull",
            "Feeling uninterested", "I need something exciting to happen", "So bored with everything",
            # Lonely
            "Feeling lonely today", "I miss having someone around", "Feeling isolated",
            "I wish I had company", "Feeling alone in this world", "Need some human connection",
            # Excited
            "I'm so excited about this", "Can't wait for the weekend", "Feeling thrilled",
            "This is going to be amazing", "I'm pumped up", "So enthusiastic about this",
            # Grateful
            "I'm so grateful for everything", "Feeling blessed today", "Thankful for my life",
            "Appreciating all I have", "Feeling grateful for my friends", "So thankful right now",
            # Confused
            "I'm confused about this", "Don't understand what's happening", "Feeling puzzled",
            "This is so confusing", "I don't get it", "Feeling uncertain",
            # Motivated
            "I'm feeling motivated today", "Ready to tackle my goals", "Feeling driven",
            "So focused on my work", "I'm determined to succeed", "Feeling ambitious",
            # Disappointed
            "I'm disappointed with the result", "Things didn't go as planned", "Feeling let down",
            "This is disappointing", "I expected better", "Feeling unsatisfied",
            # Proud
            "I'm so proud of myself", "Feeling accomplished today", "I did a great job",
            "Proud of what I achieved", "Feeling successful", "I'm beaming with pride",
            # Nostalgic
            "Feeling nostalgic about the past", "Remembering the good old days", "Feeling sentimental",
            "I miss those times", "Feeling nostalgic for my childhood", "Remember when...",
            # Optimistic
            "I'm optimistic about the future", "Things are looking up", "Feeling hopeful",
            "I believe good things will happen", "Feeling positive about tomorrow", "So optimistic",
            # Pessimistic
            "I'm feeling pessimistic today", "Things won't get better", "Feeling hopeless",
            "This is never going to work", "Feeling negative about everything", "So pessimistic",
            # Content
            "I'm content with my life", "Feeling satisfied right now", "Everything is good",
            "I'm at peace with where I am", "Feeling fulfilled", "So content with everything",
            # Guilty
            "I'm feeling guilty about this", "I should have done better", "Feeling remorse",
            "This is my fault", "I regret my actions", "Feeling guilty"
        ],
        'mood': [
            # Stressed
            'stressed', 'stressed', 'stressed', 'stressed', 'stressed', 'stressed',
            # Tired
            'tired', 'tired', 'tired', 'tired', 'tired', 'tired',
            # Calm
            'calm', 'calm', 'calm', 'calm', 'calm', 'calm',
            # Bored
            'bored', 'bored', 'bored', 'bored', 'bored', 'bored',
            # Lonely
            'lonely', 'lonely', 'lonely', 'lonely', 'lonely', 'lonely',
            # Excited
            'excited', 'excited', 'excited', 'excited', 'excited', 'excited',
            # Grateful
            'grateful', 'grateful', 'grateful', 'grateful', 'grateful', 'grateful',
            # Confused
            'confused', 'confused', 'confused', 'confused', 'confused', 'confused',
            # Motivated
            'motivated', 'motivated', 'motivated', 'motivated', 'motivated', 'motivated',
            # Disappointed
            'disappointed', 'disappointed', 'disappointed', 'disappointed', 'disappointed', 'disappointed',
            # Proud
            'proud', 'proud', 'proud', 'proud', 'proud', 'proud',
            # Nostalgic
            'nostalgic', 'nostalgic', 'nostalgic', 'nostalgic', 'nostalgic', 'nostalgic',
            # Optimistic
            'optimistic', 'optimistic', 'optimistic', 'optimistic', 'optimistic', 'optimistic',
            # Pessimistic
            'pessimistic', 'pessimistic', 'pessimistic', 'pessimistic', 'pessimistic', 'pessimistic',
            # Content
            'content', 'content', 'content', 'content', 'content', 'content',
            # Guilty
            'guilty', 'guilty', 'guilty', 'guilty', 'guilty', 'guilty'
        ]
    }
    
    # Create DataFrame for additional data
    additional_df = pd.DataFrame(additional_data)
    
    # Combine with original dataset
    df = pd.concat([df, additional_df], ignore_index=True)
    
    return df

# ==================== ML MODEL FUNCTIONS ====================
def train_new_model():
    """Train a new ML model with real dataset"""
    # Load the emotion dataset
    df = load_emotion_dataset()
    
    if df is None:
        return None, None, "Dataset not available"
    
    st.write(f"Training with {len(df)} samples...")
    st.write("Mood distribution:")
    st.write(df['mood'].value_counts())
    
    # Preprocessing
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Feature extraction
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['mood']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(max_iter=1000, multi_class='multinomial')
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Display classification report
    st.write("Model Performance:")
    st.text(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, 'mood_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    return model, vectorizer, f"Trained model with {len(df)} samples (Accuracy: {accuracy:.2f})"

def load_ml_model():
    """Load or train the ML model"""
    model_path = 'mood_model.pkl'
    vectorizer_path = 'vectorizer.pkl'
    
    # Check if model exists
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer, "Loaded existing model"
    else:
        # Train new model with real dataset
        return train_new_model()

def detect_mood_ml(text, model, vectorizer):
    """Detect mood using ML model"""
    # Preprocess text
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    processed_text = ' '.join(tokens)
    
    # Vectorize and predict
    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)[0]
    
    # Get prediction probabilities
    probabilities = model.predict_proba(text_vector)[0]
    confidence = max(probabilities)
    
    return prediction, confidence

# ==================== UTILITY FUNCTIONS ====================
def create_text_image(title, width=200, height=300):
    """Create a text image as fallback"""
    from PIL import ImageDraw, ImageFont
    
    # Create a blank image
    img = Image.new('RGB', (width, height), color=(79, 70, 229))
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Calculate text position
    text = title if len(title) < 20 else title[:17] + "..."
    try:
        # For newer Pillow versions
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except:
        # For older Pillow versions
        text_width, text_height = draw.textsize(text, font=font)
    
    position = ((width - text_width) // 2, (height - text_height) // 2)
    
    # Draw the text
    draw.text(position, text, fill=(255, 255, 255), font=font)
    return img

def load_image(path, title=""):
    """Load image with error handling"""
    try:
        # Check if it's a local file
        if path.startswith('posters/'):
            img = Image.open(path)
            return img
        else:
            # Try to load from URL
            response = requests.get(path, timeout=5)
            img = Image.open(BytesIO(response.content))
            return img
    except:
        # Fallback to text image
        return create_text_image(title)

def detect_mood_rule_based(text):
    """Fallback rule-based mood detection"""
    text = text.lower()
    
    mood_keywords = {
        "happy": ["happy", "joy", "excited", "great", "wonderful", "amazing", "good", "love", "fantastic", "awesome"],
        "sad": ["sad", "depressed", "down", "unhappy", "crying", "tears", "grief", "miserable", "heartbroken"],
        "angry": ["angry", "mad", "furious", "annoyed", "irritated", "rage", "frustrated", "upset"],
        "stressed": ["stressed", "overwhelmed", "anxious", "worried", "nervous", "pressure", "tense"],
        "tired": ["tired", "exhausted", "fatigued", "sleepy", "drained", "worn out", "burned out"],
        "calm": ["calm", "peaceful", "relaxed", "serene", "tranquil", "chill", "at ease"],
        "bored": ["bored", "uninterested", "dull", "tedious", "monotonous", "nothing to do"],
        "lonely": ["lonely", "alone", "isolated", "abandoned", "miss", "nobody"],
        "excited": ["excited", "thrilled", "eager", "enthusiastic", "pumped", "can't wait"],
        "grateful": ["grateful", "thankful", "appreciative", "blessed", "gratitude"],
        "confused": ["confused", "uncertain", "unsure", "puzzled", "don't understand"],
        "motivated": ["motivated", "driven", "determined", "focused", "ready to work"],
        "scared": ["scared", "afraid", "terrified", "frightened", "panic"],
        "disappointed": ["disappointed", "let down", "failed", "didn't work out"],
        "proud": ["proud", "accomplished", "achieved", "success", "did it"],
        "nostalgic": ["nostalgic", "memories", "remember when", "back in the day"],
        "optimistic": ["optimistic", "hopeful", "positive", "looking forward", "bright future"],
        "pessimistic": ["pessimistic", "negative", "hopeless", "no point", "giving up"],
        "surprised": ["surprised", "shocked", "amazed", "unexpected", "wow"],
        "content": ["content", "satisfied", "fulfilled", "enough", "at peace"],
        "guilty": ["guilty", "regret", "sorry", "my fault", "shouldn't have"]
    }
    
    # Count mood keywords
    mood_scores = {mood: 0 for mood in mood_keywords}
    
    for mood, keywords in mood_keywords.items():
        for keyword in keywords:
            if keyword in text:
                mood_scores[mood] += 1
    
    # Return mood with highest score
    if max(mood_scores.values()) > 0:
        return max(mood_scores, key=mood_scores.get)
    else:
        return "neutral"

def save_mood_to_history(mood, text, method, confidence):
    """Save mood to history file"""
    # Create file if it doesn't exist
    if not os.path.exists('mood_history.csv'):
        with open('mood_history.csv', 'w') as f:
            f.write('timestamp,mood,text,method,confidence\n')
    
    # Append new mood entry
    with open('mood_history.csv', 'a') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f'{timestamp},{mood},"{text}",{method},{confidence}\n')

# ==================== RECOMMENDATIONS DATA ====================
RECOMMENDATIONS = {
    "happy": {
        "activities": [
            "Go for a nature walk",
            "Try a new recipe",
            "Call a friend",
            "Start a creative project",
            "Visit a local museum",
            "Dance to your favorite music",
            "Plan a future trip",
            "Practice gratitude journaling"
        ],
        "movies": [
            {
                "title": "The Grand Budapest Hotel",
                "poster": "posters/grand_budapest_hotel.jpeg",
                "trailer": "https://youtu.be/1Fg5iWmQjwk?si=keTOnS64U6BV2bwj"
            },
            {
                "title": "La La Land",
                "poster": "posters/la_la_land.jpeg",
                "trailer": "https://www.youtube.com/watch?v=0pdqf4P9MB8"
            }
        ],
        "music": ["Upbeat pop", "Feel-good classics", "Dance hits"],
        "books": ["Happy by Derren Brown", "The Happiness Project by Gretchen Rubin"],
        "meditation": ["Loving-kindness meditation", "Gratitude meditation"],
        "exercise": ["Dance workout", "Outdoor jogging", "Yoga flow"]
    },
    "sad": {
        "activities": [
            "Listen to calming music",
            "Write in a journal",
            "Take a warm bath",
            "Watch a comforting movie",
            "Call a supportive friend",
            "Practice self-care routines",
            "Engage in creative expression",
            "Spend time in nature"
        ],
        "movies": [
            {
                "title": "The Shawshank Redemption",
                "poster": "posters/shawshank_redemption.jpeg",
                "trailer": "https://www.youtube.com/watch?v=6hB3S9bIaco"
            },
            {
                "title": "Good Will Hunting",
                "poster": "posters/good_will_hunting.jpeg",
                "trailer": "https://www.youtube.com/watch?v=Qivj4Oc7rEw"
            }
        ],
        "music": ["Comforting ballads", "Classical music", "Ambient sounds"],
        "books": ["The Gifts of Imperfection by Brené Brown", "When Things Fall Apart by Pema Chödrön"],
        "meditation": ["Compassion meditation", "Body scan meditation"],
        "exercise": ["Gentle yoga", "Walking meditation", "Tai chi"]
    },
    "angry": {
        "activities": [
            "Practice deep breathing exercises",
            "Engage in physical activity",
            "Write down your feelings",
            "Listen to calming music",
            "Take a time-out",
            "Practice progressive muscle relaxation",
            "Talk to someone you trust",
            "Channel energy into creative projects"
        ],
        "movies": [],
        "music": ["Calming instrumental", "Nature sounds", "Meditation music"],
        "books": ["The Anger Trap by Les Carter", "Nonviolent Communication by Marshall Rosenberg"],
        "meditation": ["Loving-kindness meditation", "Mindful breathing"],
        "exercise": ["High-intensity workout", "Boxing", "Running"]
    },
    "stressed": {
        "activities": [
            "Practice deep breathing",
            "Take a short walk",
            "Listen to calming music",
            "Practice progressive muscle relaxation",
            "Sip herbal tea",
            "Take a warm bath",
            "Practice mindfulness",
            "Write down your thoughts"
        ],
        "movies": [],
        "music": ["Calming ambient", "Nature sounds", "Meditation music"],
        "books": ["The Stress-Proof Brain by Melanie Greenberg", "Full Catastrophe Living by Jon Kabat-Zinn"],
        "meditation": ["Mindful breathing", "Body scan meditation"],
        "exercise": ["Yoga", "Tai chi", "Walking"]
    },
    "tired": {
        "activities": [
            "Take a short nap",
            "Practice gentle stretching",
            "Drink a glass of water",
            "Listen to calming music",
            "Take a warm bath",
            "Practice relaxation techniques",
            "Go to bed earlier",
            "Reduce screen time"
        ],
        "movies": [],
        "music": ["Soothing lullabies", "Calming ambient", "Relaxing piano"],
        "books": ["The Sleep Revolution by Arianna Huffington", "Why We Sleep by Matthew Walker"],
        "meditation": ["Yoga nidra", "Body scan for relaxation"],
        "exercise": ["Gentle stretching", "Restorative yoga", "Walking"]
    },
    "calm": {
        "activities": [
            "Practice mindfulness meditation",
            "Enjoy a cup of tea",
            "Read a book",
            "Spend time in nature",
            "Practice gentle yoga",
            "Listen to calming music",
            "Engage in a creative hobby",
            "Take a leisurely walk"
        ],
        "movies": [
            {
                "title": "The Tree of Life",
                "poster": "posters/tree_of_life.jpeg",
                "trailer": "https://www.youtube.com/watch?v=W7QpMQ3k0s0"
            }
        ],
        "music": ["Ambient music", "Classical pieces", "Nature sounds"],
        "books": ["The Power of Now by Eckhart Tolle", "Wherever You Go, There You Are by Jon Kabat-Zinn"],
        "meditation": ["Mindfulness meditation", "Breathing awareness"],
        "exercise": ["Yoga", "Tai chi", "Walking meditation"]
    },
    "bored": {
        "activities": [
            "Learn a new skill online",
            "Try a new recipe",
            "Explore a new part of your city",
            "Start a creative project",
            "Organize your space",
            "Read a book in a new genre",
            "Try a new hobby",
            "Plan a future adventure"
        ],
        "movies": [
            {
                "title": "The Secret Life of Walter Mitty",
                "poster": "posters/walter_mitty.jpeg",
                "trailer": "https://www.youtube.com/watch?v=S5imlq6iD0o"
            }
        ],
        "music": ["Upbeat indie", "New discoveries", "Energetic playlists"],
        "books": ["Steal Like an Artist by Austin Kleon", "Big Magic by Elizabeth Gilbert"],
        "meditation": ["Open awareness meditation", "Curiosity practice"],
        "exercise": ["Dance workout", "Rock climbing", "Martial arts"]
    },
    "lonely": {
        "activities": [
            "Reach out to a friend or family member",
            "Join a club or group activity",
            "Volunteer for a cause",
            "Attend a community event",
            "Adopt a pet",
            "Take a class to learn something new",
            "Explore a new hobby",
            "Visit a public space like a cafe or library"
        ],
        "movies": [
            {
                "title": "Cast Away",
                "poster": "posters/cast_away.jpeg",
                "trailer": "https://www.youtube.com/watch?v=I_2j2GZ_4f8"
            }
        ],
        "music": ["Uplifting songs", "Feel-good classics", "Community-themed music"],
        "books": ["Loneliness: Human Nature and the Need for Social Connection by John Cacioppo", "The Art of Gathering by Priya Parker"],
        "meditation": ["Loving-kindness meditation", "Connection visualization"],
        "exercise": ["Group fitness classes", "Team sports", "Walking groups"]
    },
    "excited": {
        "activities": [
            "Plan a future adventure",
            "Share your excitement with others",
            "Channel energy into a project",
            "Try something new",
            "Dance to your favorite music",
            "Explore new places",
            "Start a creative endeavor",
            "Learn a new skill"
        ],
        "movies": [
            {
                "title": "Into the Wild",
                "poster": "posters/into_the_wild.jpeg",
                "trailer": "https://www.youtube.com/watch?v=1l5Z4tT8K3g"
            }
        ],
        "music": ["Upbeat anthems", "Energetic playlists", "Adventure-themed music"],
        "books": ["The Alchemist by Paulo Coelho", "Wild by Cheryl Strayed"],
        "meditation": ["Visualization meditation", "Gratitude practice"],
        "exercise": ["High-intensity interval training", "Adventure sports", "Dance"]
    },
    "grateful": {
        "activities": [
            "Write a thank you note",
            "Express appreciation to someone",
            "Volunteer for a cause",
            "Create a gratitude jar",
            "Share your gratitude with others",
            "Reflect on positive experiences",
            "Practice gratitude meditation",
            "Donate to a charity"
        ],
        "movies": [
            {
                "title": "It's a Wonderful Life",
                "poster": "posters/its_a_wonderful_life.jpeg",
                "trailer": "https://www.youtube.com/watch?v=iLR3gZrU2gk"
            }
        ],
        "music": ["Uplifting spiritual", "Inspirational songs", "Heartfelt ballads"],
        "books": ["The Gratitude Diaries by Janice Kaplan", "Thanks! by Robert Emmons"],
        "meditation": ["Gratitude meditation", "Loving-kindness practice"],
        "exercise": ["Walking in nature", "Gentle yoga", "Tai chi"]
    },
    "confused": {
        "activities": [
            "Write down your thoughts",
            "Talk to someone you trust",
            "Take a walk to clear your head",
            "Make a pros and cons list",
            "Practice mindfulness",
            "Do some research",
            "Take a break and come back later",
            "Try meditation"
        ],
        "movies": [
            {
                "title": "Inception",
                "poster": "posters/inception.jpeg",
                "trailer": "https://www.youtube.com/watch?v=YoHD9XEInc0"
            }
        ],
        "music": ["Calming instrumental", "Focus music", "Ambient sounds"],
        "books": ["Thinking, Fast and Slow by Daniel Kahneman", "The Art of Thinking Clearly by Rolf Dobelli"],
        "meditation": ["Mindful breathing", "Clarity meditation"],
        "exercise": ["Walking", "Yoga", "Tai chi"]
    },
    "motivated": {
        "activities": [
            "Tackle your most important task",
            "Make a to-do list",
            "Set a new goal",
            "Start a new project",
            "Exercise to boost energy",
            "Listen to motivational music",
            "Visualize success",
            "Break tasks into smaller steps"
        ],
        "movies": [
            {
                "title": "Rocky",
                "poster": "posters/rocky.jpeg",
                "trailer": "https://www.youtube.com/watch?v=JP3OEI7r4wQ"
            }
        ],
        "music": ["Motivational anthems", "Upbeat electronic", "Energetic rock"],
        "books": ["Atomic Habits by James Clear", "The 7 Habits of Highly Effective People by Stephen Covey"],
        "meditation": ["Goal visualization", "Success meditation"],
        "exercise": ["High-intensity workout", "Running", "Weight training"]
    },
    "scared": {
        "activities": [
            "Practice deep breathing",
            "Talk to someone supportive",
            "Ground yourself with the 5-4-3-2-1 technique",
            "Listen to calming music",
            "Practice mindfulness",
            "Write about your fears",
            "Create a safe space",
            "Try progressive muscle relaxation"
        ],
        "movies": [],
        "music": ["Calming piano", "Nature sounds", "Guided meditation"],
        "books": ["The Worry Trick by David Carbonell", "Dare by Barry McDonagh"],
        "meditation": ["Safety visualization", "Body scan meditation"],
        "exercise": ["Gentle stretching", "Walking", "Restorative yoga"]
    },
    "disappointed": {
        "activities": [
            "Allow yourself to feel the disappointment",
            "Talk to a friend",
            "Write about your feelings",
            "Practice self-compassion",
            "Do something kind for yourself",
            "Focus on what you can control",
            "Practice gratitude",
            "Engage in a comforting activity"
        ],
        "movies": [
            {
                "title": "Inside Out",
                "poster": "posters/inside_out.jpeg",
                "trailer": "https://www.youtube.com/watch?v=seMwpP0Y_uU"
            }
        ],
        "music": ["Comforting songs", "Uplifting ballads", "Hopeful melodies"],
        "books": ["Option B by Sheryl Sandberg", "Rising Strong by Brené Brown"],
        "meditation": ["Self-compassion meditation", "Letting go practice"],
        "exercise": ["Walking", "Gentle yoga", "Swimming"]
    },
    "proud": {
        "activities": [
            "Celebrate your achievement",
            "Share your success with others",
            "Document your accomplishment",
            "Treat yourself to something special",
            "Reflect on your journey",
            "Set your next goal",
            "Help others achieve their goals",
            "Practice gratitude for your abilities"
        ],
        "movies": [
            {
                "title": "The Pursuit of Happyness",
                "poster": "posters/pursuit_of_happyness.jpeg",
                "trailer": "https://www.youtube.com/watch?v=89Kq8b0XxgQ"
            }
        ],
        "music": ["Celebratory songs", "Triumphant anthems", "Upbeat classics"],
        "books": ["Mindset by Carol Dweck", "Grit by Angela Duckworth"],
        "meditation": ["Success visualization", "Gratitude meditation"],
        "exercise": ["Dance", "Celebratory workout", "Team sports"]
    },
    "nostalgic": {
        "activities": [
            "Look through old photos",
            "Listen to music from your past",
            "Visit a place from your childhood",
            "Connect with old friends",
            "Write about your memories",
            "Watch a favorite childhood movie",
            "Cook a childhood recipe",
            "Share stories with family"
        ],
        "movies": [
            {
                "title": "Stand By Me",
                "poster": "posters/stand_by_me.jpeg",
                "trailer": "https://www.youtube.com/watch?v=pmnXzB6i7Qs"
            }
        ],
        "music": ["Classic hits", "Throwback playlists", "Childhood favorites"],
        "books": ["The Night Circus by Erin Morgenstern", "The Book Thief by Markus Zusak"],
        "meditation": ["Memory meditation", "Gratitude for the past"],
        "exercise": ["Walking in familiar places", "Dancing to old favorites", "Gentle yoga"]
    },
    "optimistic": {
        "activities": [
            "Plan for the future",
            "Set new goals",
            "Visualize success",
            "Share your optimism with others",
            "Try something new",
            "Practice gratitude",
            "Help others",
            "Engage in creative projects"
        ],
        "movies": [
            {
                "title": "The Secret Life of Walter Mitty",
                "poster": "posters/walter_mitty.jpeg",
                "trailer": "https://www.youtube.com/watch?v=S5imlq6iD0o"
            }
        ],
        "music": ["Upbeat pop", "Inspirational songs", "Hopeful melodies"],
        "books": ["The Alchemist by Paulo Coelho", "Man's Search for Meaning by Viktor Frankl"],
        "meditation": ["Future visualization", "Gratitude practice"],
        "exercise": ["Dance", "Outdoor activities", "Team sports"]
    },
    "pessimistic": {
        "activities": [
            "Challenge negative thoughts",
            "Practice gratitude",
            "Talk to a supportive person",
            "Engage in physical activity",
            "Practice mindfulness",
            "Set small, achievable goals",
            "Limit exposure to negative news",
            "Practice self-compassion"
        ],
        "movies": [
            {
                "title": "Silver Linings Playbook",
                "poster": "posters/silver_linings_playbook.jpeg",
                "trailer": "https://www.youtube.com/watch?v=0qVS2g5dJ4g"
            }
        ],
        "music": ["Uplifting songs", "Calming melodies", "Inspirational music"],
        "books": ["Learned Optimism by Martin Seligman", "The Happiness Advantage by Shawn Achor"],
        "meditation": ["Loving-kindness meditation", "Mindful breathing"],
        "exercise": ["Walking", "Yoga", "Dance"]
    },
    "surprised": {
        "activities": [
            "Take a moment to process the surprise",
            "Talk about it with someone",
            "Write down your thoughts",
            "Practice mindfulness",
            "Embrace the unexpected",
            "Look for the positive in the surprise",
            "Share your experience",
            "Take a deep breath"
        ],
        "movies": [
            {
                "title": "The Sixth Sense",
                "poster": "posters/sixth_sense.jpeg",
                "trailer": "https://www.youtube.com/watch?v=V6d9Zk5j5x5j5x5j5x5j5x5j5x5j5x"
            }
        ],
        "music": ["Varied playlists", "Unexpected genres", "Surprising mixes"],
        "books": ["The Art of Surprise by Bernard Crettaz", "The Book of Delights by Ross Gay"],
        "meditation": ["Open awareness meditation", "Mindfulness of the unexpected"],
        "exercise": ["Dance", "Yoga", "Walking"]
    },
    "content": {
        "activities": [
            "Savor the moment",
            "Practice gratitude",
            "Share your contentment with others",
            "Engage in a favorite activity",
            "Spend time in nature",
            "Enjoy a cup of tea",
            "Read a good book",
            "Practice mindfulness"
        ],
        "movies": [
            {
                "title": "Little Miss Sunshine",
                "poster": "posters/little_miss_sunshine.jpeg",
                "trailer": "https://www.youtube.com/watch?v=6dQaV5i5j5x5j5x5j5x5j5x5j5x5j5x"
            }
        ],
        "music": ["Peaceful melodies", "Gentle classics", "Soothing sounds"],
        "books": ["The Art of Simple Living by Shunmyo Masuno", "The Book of Joy by Dalai Lama"],
        "meditation": ["Contentment meditation", "Gratitude practice"],
        "exercise": ["Walking", "Gentle yoga", "Tai chi"]
    },
    "guilty": {
        "activities": [
            "Acknowledge your feelings",
            "Practice self-forgiveness",
            "Make amends if possible",
            "Learn from the experience",
            "Talk to someone you trust",
            "Practice self-compassion",
            "Engage in positive actions",
            "Write about your feelings"
        ],
        "movies": [
            {
                "title": "Good Will Hunting",
                "poster": "posters/good_will_hunting.jpeg",
                "trailer": "https://www.youtube.com/watch?v=Qivj4Oc7rEw"
            }
        ],
        "music": ["Comforting songs", "Forgiveness-themed music", "Calming melodies"],
        "books": ["The Gifts of Imperfection by Brené Brown", "Self-Compassion by Kristin Neff"],
        "meditation": ["Forgiveness meditation", "Self-compassion practice"],
        "exercise": ["Walking", "Yoga", "Tai chi"]
    },
    "neutral": {
        "activities": [
            "Take a walk outside",
            "Try a new hobby",
            "Listen to a podcast",
            "Organize your space",
            "Call a friend",
            "Read a book",
            "Practice mindfulness",
            "Try a new recipe"
        ],
        "movies": [
            {
                "title": "Forrest Gump",
                "poster": "posters/forrest_gump.jpeg",
                "trailer": "https://www.youtube.com/watch?v=bLvqoHBptjg"
            }
        ],
        "music": ["Varied playlists", "Background music", "Easy listening"],
        "books": ["The Little Prince by Antoine de Saint-Exupéry", "The Alchemist by Paulo Coelho"],
        "meditation": ["Mindfulness meditation", "Breathing awareness"],
        "exercise": ["Walking", "Yoga", "Swimming"]
    }
}

# ==================== VISUALIZATION FUNCTIONS ====================
def plot_mood_distribution():
    """Create mood distribution pie chart"""
    # Check if mood history file exists
    if not os.path.exists('mood_history.csv'):
        return None
    
    # Read mood history
    df = pd.read_csv('mood_history.csv')
    
    # Count mood occurrences
    mood_counts = df['mood'].value_counts()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create pie chart
    ax.pie(mood_counts, labels=mood_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    plt.title('Your Mood Distribution')
    
    return fig

def plot_mood_trends():
    """Create mood trends over time chart"""
    # Check if mood history file exists
    if not os.path.exists('mood_history.csv'):
        return None
    
    # Read mood history
    df = pd.read_csv('mood_history.csv')
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    # Count moods per day
    daily_moods = df.groupby(['date', 'mood']).size().unstack(fill_value=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create stacked area chart
    daily_moods.plot(kind='area', stacked=True, ax=ax, alpha=0.7)
    
    plt.title('Your Mood Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('Mood Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def plot_model_performance():
    """Create model performance chart"""
    # Check if mood history file exists
    if not os.path.exists('mood_history.csv'):
        return None
    
    # Read mood history
    df = pd.read_csv('mood_history.csv')
    
    # Count method usage
    method_counts = df['method'].value_counts()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create bar chart
    method_counts.plot(kind='bar', ax=ax, color=['#4f46e5', '#10b981'])
    
    plt.title('AI Model vs Rule-Based Detection')
    plt.xlabel('Detection Method')
    plt.ylabel('Number of Uses')
    plt.xticks(rotation=0)
    
    return fig

# ==================== MAIN APPLICATION ====================
# Load ML model
with st.spinner("Loading AI model..."):
    model, vectorizer, model_status = load_ml_model()
    if model is not None:
        st.success(f"AI Model Ready: {model_status}")
    else:
        st.error("Failed to load model. Please check dataset files.")

# User input
user_input = st.text_area("How are you feeling today?", 
                         placeholder="I'm feeling...")

# Analyze button
if st.button("Analyze My Mood"):
    if not user_input.strip():
        st.warning("Please enter your feelings!")
    else:
        # Try ML model first
        try:
            mood, confidence = detect_mood_ml(user_input, model, vectorizer)
            method = "ML"
            st.success(f"AI Model Confidence: {confidence:.2f}")
        except:
            # Fall back to rule-based
            mood = detect_mood_rule_based(user_input)
            confidence = 0.0
            method = "Rule-Based"
            st.info("Using rule-based detection as fallback")
        
        # Save mood to history
        save_mood_to_history(mood, user_input, method, confidence)
        
        # Display mood
        st.subheader(f"Detected Mood: {mood.upper()}")
        
        # Get recommendations
        recommendations = RECOMMENDATIONS.get(mood, RECOMMENDATIONS["neutral"])
        
        # Display activities
        st.subheader("Recommended Activities")
        for activity in recommendations["activities"]:
            st.markdown(f"- {activity}")
        
        # Display meditation recommendations
        st.subheader("Meditation & Mindfulness")
        for meditation in recommendations["meditation"]:
            st.markdown(f"- {meditation}")
        
        # Display exercise recommendations
        st.subheader("Exercise Recommendations")
        for exercise in recommendations["exercise"]:
            st.markdown(f"- {exercise}")
        
        # Display music recommendations
        st.subheader("Music Recommendations")
        for music in recommendations["music"]:
            st.markdown(f"- {music}")
        
        # Display book recommendations
        st.subheader("Book Recommendations")
        for book in recommendations["books"]:
            st.markdown(f"- {book}")
        
        # Display movies if available
        if recommendations["movies"]:
            st.subheader("Movie Recommendations")
            cols = st.columns(len(recommendations["movies"]))
            
            for idx, movie in enumerate(recommendations["movies"]):
                with cols[idx]:
                    st.markdown(f"**{movie['title']}**")
                    img = load_image(movie['poster'], movie['title'])
                    st.image(img, width=200)
                    st.markdown(f"[Watch Trailer]({movie['trailer']})")
        else:
            st.write("No movie recommendations for this mood.")

# Add a section for mood analytics
st.header("Your Mood Analytics")

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Mood Distribution", "Mood Trends", "Model Performance"])

with tab1:
    st.subheader("Your Mood Distribution")
    fig = plot_mood_distribution()
    if fig:
        st.pyplot(fig)
    else:
        st.write("No mood history available yet. Analyze your mood to see visualizations!")

with tab2:
    st.subheader("Your Mood Trends Over Time")
    fig = plot_mood_trends()
    if fig:
        st.pyplot(fig)
    else:
        st.write("No mood history available yet. Analyze your mood to see visualizations!")

with tab3:
    st.subheader("AI Model Performance")
    fig = plot_model_performance()
    if fig:
        st.pyplot(fig)
    else:
        st.write("No mood history available yet. Analyze your mood to see visualizations!")

# Add a button to clear mood history
if st.button("Clear Mood History"):
    if os.path.exists('mood_history.csv'):
        os.remove('mood_history.csv')
        st.success("Mood history cleared!")
    else:
        st.write("No mood history to clear.")

# Instructions
with st.expander("How to use this app"):
    st.write("""
    1. Enter how you're feeling in the text area above
    2. Click 'Analyze My Mood' to get recommendations
    3. The AI will analyze your emotions using machine learning
    4. View your mood analytics in the tabs below
    5. Click on movie trailer links to watch previews
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Machine Learning")