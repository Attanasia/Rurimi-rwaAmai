import os
import re
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings('ignore')


try:
    from dotenv import load_dotenv
    load_dotenv()  
except ImportError:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Error: GROQ_API_KEY not set.")
    print("Run: export GROQ_API_KEY='your-key-here'")
    print("Or create a .env file with GROQ_API_KEY=your-key-here")
    exit(1)

print("Groq API key found")


print("Loading translation database...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vector_store = Chroma(
    persist_directory="./shona_vector_db",
    embedding_function=embeddings
)

all_docs = vector_store.get()
exact_dict = {}
for i, metadata in enumerate(all_docs['metadatas']):
    english = metadata.get('english', '').lower()
    shona = metadata.get('shona', '')
    if english not in exact_dict:
        exact_dict[english] = shona

print(f" Loaded {len(exact_dict)} translations\n")


client = Groq(api_key=GROQ_API_KEY)


def normalize_query(text):
    """Remove common question words to get the core phrase"""
    text = text.lower().strip()
    patterns = [
        r'^(how do you say|what is|tell me|what does|how to say|the shona word for|translate|meaning of)\s+',
        r'\s+(in shona|please|thank you)$'
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    return text.strip()

def find_translation(query):
    """Find translation with improved matching for combined phrases"""
    original = query.lower().strip()
    cleaned = normalize_query(original)
    
    if original in exact_dict:
        return exact_dict[original], "exact"
    

    if cleaned in exact_dict:
        return exact_dict[cleaned], "exact"
    

    words = original.split()
    if "and" in words:
        parts = []
        for word in words:
            if word != "and" and word in exact_dict:
                parts.append(exact_dict[word])
        if len(parts) >= 2:
            combined = " ne ".join(parts)
            return combined, "constructed"
    
  
    number_map = {
        "1": "Poshi", "one": "Poshi",
        "2": "Piri", "two": "Piri",
        "3": "Tatu", "three": "Tatu",
        "4": "Ina", "four": "Ina",
        "5": "Shanu", "five": "Shanu",
        "6": "Tanhatu", "six": "Tanhatu",
        "7": "Nomwe", "seven": "Nomwe",
        "8": "Sere", "eight": "Sere",
        "9": "Pfumbamwe", "nine": "Pfumbamwe",
        "10": "Gumi", "ten": "Gumi",
    }
    
    if original in number_map:
        return number_map[original], "exact"
    if cleaned in number_map:
        return number_map[cleaned], "exact"
    

    for num_key, num_val in number_map.items():
        if num_key in original:
            rest = original.replace(num_key, "").strip()
            for eng, sho in exact_dict.items():
                if rest in eng.lower() or eng.lower() in rest:
                    return f"{num_val} {sho}", "constructed"
    
    
    common_phrases = {
        "lets go": "Handeyi",
        "let's go": "Handeyi",
        "can you repeat that": "Dzokorora izvozvo",
        "what is this": "Chii ichi",
        "never mind": "Usazvinetse",
        "tell me": "Ndiudze",
        "show me": "Ndiratidze",
        "come here": "Huya pano",
        "go there": "Enda uko",
        "wait here": "Mirira pano",
    }
    
    if original in common_phrases:
        return common_phrases[original], "exact"
    if cleaned in common_phrases:
        return common_phrases[cleaned], "exact"
    
    
    results = vector_store.similarity_search(query, k=3)
    if results:
        return results[0].metadata.get('shona', ''), "vector"
    
    return None, "none"

def generate_friendly_response(user_input, translation, match_type):
    """Generate a friendly response with pronunciation help (auto‑generated)."""
    
    if match_type in ["exact", "constructed"]:
        
        prompt = f"""The Shona word/phrase is: "{translation}"
Provide a very brief pronunciation guide (1 short sentence) that helps an English speaker say it.
Example: "Mangwanani" -> "mah-ngwah-nah-nee"
Only output the pronunciation guide, nothing else."""
        
        try:
            pron_response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=50,
            )
            pronunciation = pron_response.choices[0].message.content.strip()
            return f"{translation}\n\n🔊 Pronunciation: {pronunciation}\n\n(From my Shona database 📖)"
        except:
            return f"{translation}\n\n(From my Shona database 📖)"
    
    elif match_type == "vector":
        
        prompt = f"""The user asked: "{user_input}"
A possible Shona translation from my database is: "{translation}"
Write a short, helpful response (1 sentence) giving this as a possible translation.
Be friendly and honest that you're not 100% sure.
Keep it very brief."""
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=80,
            )
            return response.choices[0].message.content + "\n\n(From AI with database help 🤖)"
        except Exception as e:
            return f"Possible translation: {translation}\n\n(From database, but verify with a native speaker)"
    
    else:
        
        prompt = f"""The user asked: "{user_input}"
I don't have this in my Shona database.
Write a short, helpful response (1 sentence) saying you don't have this translation yet.
Offer to help with something else.
Keep it friendly and brief."""
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=60,
            )
            return response.choices[0].message.content
        except Exception as e:
            return "I don't have that translation in my database yet. Can you try asking for something else?"
            

grammar_dict = {
    "i am hungry": "In Shona, we say 'Ndine nzara' which literally means 'I have hunger'. Shona doesn't use a separate verb 'to be' for physical states like hunger, thirst, or cold. For 'I am thirsty' it's 'Ndine nyota' (I have thirst).",
    
    "noun classes": "Shona has noun classes (like genders in European languages). Each noun class has a prefix that affects verbs, adjectives, and pronouns. Example: 'mu-' for people (munhu = person) becomes 'va-' for plural (vanhu = people). 'chi-' for things (chikafu = food) becomes 'zvi-' for plural (zvikafu = foods).",
    
    "greetings": "'Mhoro' is for one person (informal). 'Mhoroi' is for multiple people or when showing respect. Always use 'Mhoroi' for elders, authority figures, or formal situations. For 'good morning' say 'Mangwanani', 'good afternoon' is 'Masikati', 'good evening' is 'Manheru'.",
    
    "negation": "To make a verb negative in Shona, add 'ha-' to the beginning and change the final vowel to '-e'. Example: 'ndinoda' (I want) → 'handidi' (I don't want). 'unoda' (you want) → 'haudi' (you don't want).",
    
    "verb tenses": "Present tense: use 'ndi-' for I, 'u-' for you, 'a-' for he/she. Past tense: add '-ka-' after the prefix, e.g., 'ndakada' (I wanted). Future: add '-cha-' after the prefix, e.g., 'ndichada' (I will want).",
    
    "plural forms": "Most plurals are formed by changing the noun prefix. 'mu-' → 'va-' (person/people). 'chi-' → 'zvi-' (thing/things). 'i-' → 'ma-' (house/houses). Example: 'imba' (house) → 'mamba' (houses).",
    
    "possessives": "My = 'angu' (for class 1/2), 'yangu' (for class 9/10). Your = 'ako' (singular), 'enyu' (plural). His/her = 'ake'. Example: 'Bhuku rangu' (my book), 'Imba yangu' (my house).",
    
    "question words": "What? = 'Chii?'. Who? = 'Ndiani?'. Where? = 'Kupi?'. When? = 'Rini?'. Why? = 'Sei?'. How? = 'Kudini?'. How much? = 'Mari inoita?'.",
    
    "telling time": "'Inguvai?' means 'What time is it?'. To say 'It's 3 o'clock': 'Inguva nhatu'. 'Half past' = 'nehafu', 'quarter past' = 'nekota'. Example: '3:30' = 'Inguva nhatu nehafu'.",
    
    "love and relationships": "I love you = 'Ndinokuda' (informal) / 'Ndinokudai' (formal/plural). My love = 'Rudo rwangu'. Darling/sweetheart = 'Mudiwa' (mudiwa wangu = my dear).",
}

def get_grammar_explanation(topic):
    """Return a grammar explanation from the dictionary."""
    topic_lower = topic.lower().strip()
    if topic_lower in grammar_dict:
        return grammar_dict[topic_lower]
    for key, explanation in grammar_dict.items():
        if key in topic_lower or topic_lower in key:
            return explanation
    available = ", ".join(list(grammar_dict.keys())[:5]) + "..."
    return f"I don't have a grammar explanation for '{topic}' yet. Try: {available}"


def correct_sentence(user_shona):
    """Check if user's Shona sentence is correct for a given English phrase"""
    print("\nWhat English phrase are you trying to say?")
    english_guess = input("English: ").strip().lower()
    
    correct_shona, match_type = find_translation(english_guess)
    
    if not correct_shona:
        return f"I don't have '{english_guess}' in my database yet. Can you try another phrase?"
    
    if user_shona.lower().strip() == correct_shona.lower().strip():
        return f"✅ Correct! '{user_shona}' is right for '{english_guess}'. Well done!"
    else:
        return f"❌ Almost! The correct Shona for '{english_guess}' is '{correct_shona}'. You wrote '{user_shona}'. Keep practicing!"


print("="*60)
print(" Rurimi rwaAmai - Shona Language Tutor")
print("="*60)
print("Ask me for Shona translations!")
print("Type 'quit' to exit")
print("Type 'help' for examples")
print("Type '#check [shona]' to get your Shona corrected")
print("Type '#grammar [topic]' for grammar explanations")
print("-"*50 + "\n")

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("\nBot: Ndinokutendai! Keep practicing Shona! 🇿🇼\n")
        break
    
    if not user_input:
        continue
    
    if user_input.lower() == 'help':
        print("\n Translations:")
        print("  • hello, good morning, 7 in Shona, Monday, thank you, my name is John, tomato and onion, 10 beers, lets go")
        print("\n✏️ Correction command:")
        print("  • #check Mhoroi   (checks if 'Mhoroi' is correct Shona for something)")
        print("\n Grammar command:")
        print("  • #grammar noun classes   (e.g., #grammar greetings, #grammar i am hungry)")
        print("\n Ask any English phrase to get the Shona translation.\n")
        continue
    

    if user_input.lower().startswith('#grammar'):
        topic = user_input[8:].strip()
        if not topic:
            print("\nBot: Please specify a grammar topic, e.g., '#grammar noun classes'\n")
            continue
        explanation = get_grammar_explanation(topic)
        print(f"\nBot: {explanation}\n")
        continue
    
    
    if user_input.lower().startswith('#check'):
        shona_phrase = user_input[6:].strip()
        if not shona_phrase:
            print("\nBot: Please write the Shona phrase after #check, e.g., '#check Mhoroi'\n")
            continue
        result = correct_sentence(shona_phrase)
        print(f"\nBot: {result}\n")
        continue
    
    print("\nBot: ", end="", flush=True)
    translation, match_type = find_translation(user_input)
    response = generate_friendly_response(user_input, translation, match_type)
    print(response + "\n")