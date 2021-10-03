from collections import defaultdict
from enum import Enum

from fuzzywuzzy import process
import json
import requests
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


X = [
    "Hello",
    "Hi",
    "Hello there",
    "Nice to meet you",
    "Hi there!",

    "Cases near me",
    "Covid around me",
    "Recent COVID-19 cases",
    "Recent covid cases near me",
    "Nearest corona cases",

    "Cases around the world",
    "Global COVID",
    "How many cases in country?",
    "Covid globally",
    "Covid cases",

    "Do I need to wear a mask",
    "Should I wear mask",
    "Do I need use use mask when going outside?",
    "Mask",
    "When should I use a mask",

    "Should I sanitize",
    "Hand sanitizer",
    "Do I use soap or sanitizer?",
    "When should I use soap",
    "How should I use sanitizer?",

    "Should I social distance?",
    "2 meters away",
    "Do I need start social distancing",
    "Should I gather",
    "Can I be in a crowded place",

    "I feel sick",
    "I think I have symptoms",
    "Do I have COVID",
    "Should I go to a doctor",
    "I think I may have corona",
    "I have a cough",
    "I have trouble breathing",
    "I think I have a fever",
    "I have headache",
    "Do I go to the hospital"
]

X = [x.lower() for x in X]

y = ["greeting"] * 5 + ["local_case"] * 5 + ["global_case"] * 5 + ["mask"] * 5 + ["sanitize"] * \
    5 + ["distance"] * 5 + ["sick"] * 10


clf = Pipeline(
    [
        ('tfidf', TfidfVectorizer()),
        ('sgd', SGDClassifier("log"))
    ]
)

clf.fit(X, y)

response_mapping = {
    "greeting": "Hello there! I am a chatbot that can answer your questions about COVID-19.",
    "local_case": "Select what district you're in (Hong Kong only)",
    "global_case": "Select a country",
    "mask": "You should wear a mask when in crowded places indoors and comply with local laws.",
    "sanitize": "You should sanitize your hands before and after touching yourself with hand sanitizer or soap.",
    "distance": "You should keep a social distance of 2 meters when outside.",
    "sick": "If you think you are sick, go to a doctor or call an ambulance."
}

state = "asking"

def get_response(input):
    global state

    if state == "local_case":
        state = "asking"
        return get_hk_cases(input)
    elif state == "global_case":
        state = "asking"
        input = input.lower()
        return get_country_cases(input)

    input = input.lower()
    result = clf.predict([input])
    proba = clf.predict_proba([input])

    if max(proba[0]) < 0.75:
        return "I don't understand, try rephrasing."

    if result[0] == "local_case":
        state = "local_case"
    elif result[0] == "global_case":
        state = "global_case"

    return response_mapping[result[0]]


country_list = requests.get("https://api.covid19api.com/countries")
country_list = json.loads(country_list.content)

countries = [country["Slug"] for country in country_list]

cases = requests.get("https://api.covid19api.com/summary")
cases = json.loads(cases.content)
cases = cases["Countries"]


def get_country_cases(input):
    country, confidence = process.extract(input, countries, limit=1)[0]
    if confidence < 75:
        return "Unknown country"
    else:
        for case in cases:
            if case["Slug"] == country:
                return f"Cases: {case['TotalConfirmed']}. Deaths: {case['TotalDeaths']}"


hk_cases = requests.get(
    r"https://api.data.gov.hk/v2/filter?q=%7B%22resource%22%3A%22http%3A%2F%2Fwww.chp.gov.hk%2Ffiles%2Fmisc%2Fbuilding_list_eng.csv%22%2C%22section%22%3A1%2C%22format%22%3A%22json%22%7D")
hk_cases = json.loads(hk_cases.content)


def get_hk_cases(input):
    places = defaultdict(lambda: 0)
    for case in hk_cases:
        if case["District"] == input:
            places[case["Building name"]] += 1

    places_string = ""
    for key, value in places.items():
        places_string += f"{key} * {value}\n"

    if places_string == "":
        return "No cases in last 14 days."

    return places_string