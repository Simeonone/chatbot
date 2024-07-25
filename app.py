from flask import Flask, render_template, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import apply_features
import string
import random

app = Flask(__name__)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')


# The expanded CV data
cv_data = {
    "name": "Simeon Kengere Osiemo",
    "education": {
        "undergraduate": "The University of Nairobi",
        "masters": "The University of Nairobi (in progress)",
        "field": "Computer Science, specialized in Computational Intelligence",
        "expected_graduation": "20th September 2024"
    },
    "skills": "Python, JavaScript, Machine Learning, Data Science, Natural Language Processing, TensorFlow, Pandas, Scikit-learn, NLTK",
    "experience": [
        "Software Engineer at Ministry of Interior and National Administration",
        "Research Assistant at University of Nairobi",
        "Data Scientist at Fujita Corporation Japan",
        "Systems Integration Engineer at AfyaPro 2.0 Connected Care",
        "Software Engineer at Ongata Rongai Sub County Hospital"
    ],
    "projects": {
        "Handwritten digit recognition": "Implemented a CNN using TensorFlow and Keras to recognize handwritten digits from the MNIST dataset.",
        "Fall detection system": "Developed an embedded system using Arduino and AI for real-time fall detection in elderly care.",
        "Precision Farming Support": "Created a system using OpenCV and machine vision to identify defective areas in farmland."
    },
    "ai_courses": "Master of Science in Computer Science, How to build a generative A.I project from AWS, and Python for Data Science from IBM",
    "about_me": "Well, I'm Simeon Kengere Osiemo, a computer scientist with a passion for AI that borders on obsession. I'm the kind of person who dreams in Python and wakes up thinking about machine learning algorithms. By day, I'm a software engineer at the Ministry of Interior and National Administration, where I spend my time making digital systems more secure and efficient. By night, I'm in the process of publishing my Master's in Computer Science thesis, probably muttering about neural networks in my sleep. I've worn many hats in my career - from data scientist to systems integration engineer - but my favorite is my invisible 'AI wizard' hat. I've developed everything from handwritten digit recognition systems to fall detection devices for the elderly. You could say I'm on a mission to make machines as smart as possible, while keeping my own wits sharp enough to stay ahead of them. When I'm not coding or studying, you might find me writing technical blog posts with titles like 'How math makes machines intelligent: The magic behind AI.' It's my way of spreading the AI gospel to the masses. And yes, I'm proud to say I'm a card-carrying member of the IEEE - because nothing says 'cool' quite like a professional engineering membership, right? In short, I'm a techie with a sense of humor, always ready to tackle the next big challenge in the world of AI and software engineering. Just don't ask me to fix your printer - that's one problem even AI can't solve!",
    "motivation": "My fascination with computer science began when I realized I could create entire worlds with just logic and code. What really hooked me was the potential of AI and machine learning - it felt like bringing science fiction to life. I love the problem-solving aspect, whether it's developing fall detection systems for the elderly or precision farming tools. Each project is a new puzzle to crack. Plus, in a rapidly developing country like Kenya, I saw how technology could leapfrog traditional challenges. The constant evolution in this field keeps me excited - there's always something new to learn. Ultimately, computer science offers me the perfect blend of creativity, problem-solving, and the chance to make a real impact. And who knows? Maybe one day I'll create that benevolent AI overlord we've all been waiting for!"
}

cv_data["staying_current"] = "I do so through continuous learning and community engagement. I regularly take online courses on platforms like Udemy and attend webinars and workshops hosted by industry leaders like AWS. I also read academic papers from conferences like NeurIPS and follow AI research on arXiv. Engaging with the community through conferences, meetups, and online forums like Reddit, Medium and LinkedIn keeps me connected with peers and industry experts. Additionally, I work on personal projects to apply new techniques and contribute to open-source projects on GitHub. Lastly, I stay informed by following tech news websites such as TechCrunch and VentureBeat. This approach ensures I remain at the forefront of AI advancements."



# Add this function to check for ambiguous or short inputs
def is_ambiguous(input_text):
    return len(input_text.split()) < 2 or input_text.lower() in ['hmmm', 'um', 'uh', 'er', 'ah']

# Add this list of clarifying responses
clarifying_responses = [
    "I'm not sure I understand. Could you please rephrase your question?",
    "That's a bit vague. Can you be more specific about what you'd like to know?",
    "I didn't quite catch that. What aspect of my background or skills are you interested in?",
    "Hmm, could you elaborate on your question? I'd be happy to provide more detailed information.",
    "I'm here to help, but I need a bit more context. What would you like to know about my qualifications or experience?",
    "That's an interesting sound! But I'm better with words. What would you like to ask about my AI expertise or background?",
    "I'm afraid I didn't quite grasp your question. Could you try asking in a different way?",
    "I'm all ears, but I need a bit more to go on. What part of my CV are you curious about?",
]

# Preprocess input
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in string.punctuation and token not in stopwords.words('english')]
    return tokens

# Feature extractor for the classifier
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

# Training data for intent classification
training_data = [
    (["what", "is", "your", "name", "full"], "name"),
    (["tell", "me", "about", "yourself"], "about_me"),
    (["where", "did", "you", "complete", "undergraduate", "degree"], "undergraduate"),
    (["where", "did", "you", "complete", "masters", "degree"], "masters"),
    (["what", "is", "your", "field", "of", "study"], "field"),
    (["when", "do", "you", "expect", "to", "graduate"], "graduation"),
    (["what", "motivated", "you", "to", "study", "computer", "science"], "motivation"),
    (["what", "specific", "courses", "have", "you", "taken", "related", "to", "ai", "machine", "learning"], "ai_courses"),
    (["where", "have", "you", "worked", "previously"], "work_history"),
    (["what", "roles", "have", "you", "held", "in", "the", "past"], "roles"),
    (["tell", "me", "about", "your", "education"], "education"),
    (["what", "skills", "do", "you", "have"], "skills"),
    (["what", "is", "your", "work", "experience"], "experience"),
    (["tell", "me", "about", "your", "projects"], "projects"),
]

# Adding a new question to training_data
training_data.append((["how", "do", "you", "stay", "current", "with", "new", "technologies", "trends", "in", "ai"], "staying_current"))

# Prepare the classifier
all_words = [word for (sentence, intent) in training_data for word in sentence]
word_features = list(set(all_words))
training_set = apply_features(extract_features, training_data)
classifier = NaiveBayesClassifier.train(training_set)


# Sentiment analysis
sia = SentimentIntensityAnalyzer()

# Context maintenance
context = {"prev_intent": None}

# Modify the get_response function
def get_response(input_text):
    if is_ambiguous(input_text):
        return random.choice(clarifying_responses)
    
    tokens = preprocess(input_text)
    intent = classifier.classify(extract_features(tokens))
    
    # Sentiment analysis
    sentiment = sia.polarity_scores(input_text)
    sentiment_label = "positive" if sentiment["compound"] > 0 else "negative" if sentiment["compound"] < 0 else "neutral"
    
    response = ""
    
    if intent == "name":
        response = f"My full name is {cv_data['name']}."
    elif intent == "about_me":
        response = cv_data['about_me']
    elif intent == "undergraduate":
        response = f"I completed my undergraduate degree at {cv_data['education']['undergraduate']}."
    elif intent == "masters":
        response = f"I am currently pursuing my master's degree at {cv_data['education']['masters']}."
    elif intent == "field":
        response = f"My field of study is {cv_data['education']['field']}."
    elif intent == "graduation":
        response = f"I expect to graduate on {cv_data['education']['expected_graduation']}."
    elif intent == "motivation":
        response = cv_data['motivation']
    elif intent == "ai_courses":
        response = f"I have taken the following AI and machine learning related courses: {cv_data['ai_courses']}."
    elif intent == "work_history":
        response = "I have worked at the following places: " + ", ".join(cv_data['experience'])
    elif intent == "roles":
        roles = [exp.split(" at ")[0] for exp in cv_data['experience']]
        response = "I have held the following roles: " + ", ".join(set(roles))
    elif intent == "education":
        response = f"I completed my undergraduate degree at {cv_data['education']['undergraduate']} and I'm currently pursuing my master's degree at {cv_data['education']['masters']} in {cv_data['education']['field']}. I expect to graduate on {cv_data['education']['expected_graduation']}."
    elif intent == "skills":
        response = f"My key skills include: {cv_data['skills']}."
    elif intent == "experience":
        response = "My work experience includes: " + ", ".join(cv_data['experience'])
    elif intent == "projects":
        projects = ", ".join(cv_data['projects'].keys())
        response = f"Some of my notable projects are: {projects}. Would you like more details on any specific project?"
    elif intent == "staying_current":
        response = cv_data['staying_current']
    else:
        response = "I'm sorry, I don't have information about that. Could you please ask about my name, education, skills, experience, projects, how I stay current with AI trends, or my motivation for studying computer science?"

    # Handle follow-up questions
    if intent == context["prev_intent"] and intent == "projects":
        for project, description in cv_data['projects'].items():
            if any(word in tokens for word in project.lower().split()):
                response = f"Regarding the {project} project: {description}"
                break
    
    context["prev_intent"] = intent
    
    return f"[Sentiment: {sentiment_label}] {response}"


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_chatbot_response():
    user_input = request.json['message']
    response = get_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)