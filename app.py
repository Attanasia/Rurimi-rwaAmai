import streamlit as st
import os
import re
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Rurimi rwaAmai - Shona Tutor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)


if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'groq_client' not in st.session_state:
    st.session_state.groq_client = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'exact_dict' not in st.session_state:
    st.session_state.exact_dict = {}
if 'awaiting_english' not in st.session_state:
    st.session_state.awaiting_english = False
if 'pending_shona' not in st.session_state:
    st.session_state.pending_shona = ""


@st.cache_resource
def load_vector_database():
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
    for metadata in all_docs['metadatas']:
        english = metadata.get('english', '').lower()
        shona = metadata.get('shona', '')
        if english and english not in exact_dict:
            exact_dict[english] = shona
    return vector_store, exact_dict

@st.cache_resource
def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except:
            return None
    return Groq(api_key=api_key)


def normalize_query(text):
    text = text.lower().strip()
    patterns = [
        r'^(how do you say|what is|tell me|what does|how to say|the shona word for|translate|meaning of)\s+',
        r'\s+(in shona|please|thank you)$'
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    return text.strip()

def find_translation(query, exact_dict, vector_store):
    original = query.lower().strip()
    cleaned = normalize_query(original)

    if original in exact_dict:
        return exact_dict[original], "exact"
    if cleaned in exact_dict:
        return exact_dict[cleaned], "exact"

    if "my name is" in original:
        name = original.replace("my name is", "").strip()
        if name:
            return f"Zita rangu ndinonzi {name}", "constructed"

    number_map = {
        "1": "Poshi", "one": "Poshi", "2": "Piri", "two": "Piri",
        "3": "Tatu", "three": "Tatu", "4": "Ina", "four": "Ina",
        "5": "Shanu", "five": "Shanu", "6": "Tanhatu", "six": "Tanhatu",
        "7": "Nomwe", "seven": "Nomwe", "8": "Sere", "eight": "Sere",
        "9": "Pfumbamwe", "nine": "Pfumbamwe", "10": "Gumi", "ten": "Gumi",
    }
    if original in number_map:
        return number_map[original], "exact"
    if cleaned in number_map:
        return number_map[cleaned], "exact"

    if "and" in original:
        parts = []
        for word in original.split():
            if word != "and" and word in exact_dict:
                parts.append(exact_dict[word])
        if len(parts) >= 2:
            return " ne ".join(parts), "constructed"

    results = vector_store.similarity_search(query, k=3)
    if results:
        return results[0].metadata.get('shona', ''), "vector"
    return None, "none"

def get_pronunciation(shona_word, groq_client):
    if not groq_client:
        return ""
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": f"Give a very brief pronunciation guide (one short phrase) for the Shona word: '{shona_word}'. Example: 'Mangwanani' -> 'mah-ngwah-nah-nee'. Only output the guide."}],
            temperature=0.3,
            max_tokens=50,
        )
        return resp.choices[0].message.content.strip()
    except:
        return ""

grammar_dict = {
    "i am hungry": "In Shona, we say 'Ndine nzara' which literally means 'I have hunger'. Shona doesn't use a separate verb 'to be' for physical states like hunger, thirst, or cold.",
    "noun classes": "Shona has noun classes (like genders). Each class has a prefix that affects verbs and adjectives. Example: 'mu-' for people (munhu) becomes 'va-' for plural (vanhu). 'chi-' for things (chikafu) becomes 'zvi-' (zvikafu).",
    "greetings": "'Mhoro' is for one person (informal). 'Mhoroi' is for multiple people or showing respect. Use 'Mhoroi' for elders. 'Mangwanani' = good morning, 'Masikati' = good afternoon, 'Manheru' = good evening.",
    "negation": "To make a verb negative, add 'ha-' to the beginning and change the final vowel to '-e'. Example: 'ndinoda' (I want) → 'handidi' (I don't want).",
    "verb tenses": "Present: ndi- (I), u- (you), a- (he/she). Past: add '-ka-' after prefix, e.g., 'ndakada' (I wanted). Future: add '-cha-', e.g., 'ndichada' (I will want).",
    "plural forms": "Change noun prefix: mu- → va- (person/people), chi- → zvi- (thing/things), i- → ma- (house/houses). Example: 'imba' (house) → 'mamba' (houses).",
    "possessives": "My = 'angu' (class 1/2), 'yangu' (class 9/10). Your (sg) = 'ako', your (pl) = 'enyu'. His/her = 'ake'. Example: 'Bhuku rangu' (my book).",
    "question words": "What? = 'Chii?'. Who? = 'Ndiani?'. Where? = 'Kupi?'. When? = 'Rini?'. Why? = 'Sei?'. How? = 'Kudini?'. How much? = 'Mari inoita?'.",
    "telling time": "'Inguvai?' = 'What time is it?'. 'Inguva nhatu' = 3 o'clock. 'nehafu' = half past, 'nekota' = quarter past. Example: '3:30' = 'Inguva nhatu nehafu'.",
    "love and relationships": "I love you = 'Ndinokuda' (informal) / 'Ndinokudai' (formal/plural). My love = 'Rudo rwangu'. Darling = 'Mudiwa'.",
}

def get_grammar_explanation(topic):
    topic_lower = topic.lower().strip()
    if topic_lower in grammar_dict:
        return grammar_dict[topic_lower]
    for key, expl in grammar_dict.items():
        if key in topic_lower or topic_lower in key:
            return expl
    available = ", ".join(list(grammar_dict.keys())[:5]) + "..."
    return f"I don't have a grammar explanation for '{topic}' yet. Try: {available}"


def check_shona_correctness(shona_phrase, english_phrase, exact_dict, vector_store):
    correct_shona, _ = find_translation(english_phrase, exact_dict, vector_store)
    if not correct_shona:
        return f"I don't have '{english_phrase}' in my database yet. Can you try another phrase?"
    if shona_phrase.lower().strip() == correct_shona.lower().strip():
        return f"✅ Correct! '{shona_phrase}' is right for '{english_phrase}'. Well done!"
    else:
        return f"❌ Almost! The correct Shona for '{english_phrase}' is '{correct_shona}'. You wrote '{shona_phrase}'. Keep practicing!"


with st.spinner("Loading translation database..."):
    vector_store, exact_dict = load_vector_database()
    st.session_state.vector_store = vector_store
    st.session_state.exact_dict = exact_dict

with st.spinner("Connecting to Groq AI..."):
    groq_client = get_groq_client()
    st.session_state.groq_client = groq_client

if not st.session_state.groq_client:
    st.error("⚠️ Groq API key not found. Please set GROQ_API_KEY environment variable.")
    st.stop()


with st.sidebar:
    st.image("https://emojis.slackmojis.com/emojis/images/1531849430/4246/blob-flag-zimbabwe.png", width=80)
    st.markdown("## About")
    st.markdown("""
    **Rurimi rwaAmai** means *"The Mother Tongue"*.

    **Features**:
    -  Translation (English → Shona)
    -  Auto pronunciation
    -  Sentence correction (`#check`)
    -  Grammar explanations (`#grammar`)

    **Commands**:
    - `#check [shona phrase]` then enter English meaning
    - `#grammar [topic]`  e.g., `#grammar greetings`

    **Examples**:
    - `hello`
    - `good morning`
    - `my name is John`
    - `tomato and onion`
    """)
    st.markdown("---")
    if st.session_state.exact_dict:
        st.metric("Translations", len(st.session_state.exact_dict))
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.awaiting_english = False
        st.rerun()


st.markdown('<div class="main-header"><h1>Rurimi rwaAmai</h1><p>Your AI-Powered Shona Language Tutor</p></div>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "source" in msg:
            st.caption(msg["source"])


prompt = st.chat_input("Ask for a translation, or use #check / #grammar...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.awaiting_english:
        english_phrase = prompt.strip()
        result = check_shona_correctness(
            st.session_state.pending_shona,
            english_phrase,
            st.session_state.exact_dict,
            st.session_state.vector_store
        )
        with st.chat_message("assistant"):
            st.markdown(result)
            st.caption("✏️ Correction Result")
        st.session_state.messages.append({"role": "assistant", "content": result, "source": "✏️ Correction Result"})
        st.session_state.awaiting_english = False
        st.session_state.pending_shona = ""
        st.rerun()

    if prompt.lower().startswith('#grammar'):
        topic = prompt[8:].strip()
        if not topic:
            reply = "Please specify a grammar topic, e.g., `#grammar greetings`"
            source = "ℹ️ Info"
        else:
            reply = get_grammar_explanation(topic)
            source = "Grammar Reference"
        with st.chat_message("assistant"):
            st.markdown(reply)
            st.caption(source)
        st.session_state.messages.append({"role": "assistant", "content": reply, "source": source})
        st.rerun()

    if prompt.lower().startswith('#check'):
        shona_phrase = prompt[6:].strip()
        if not shona_phrase:
            reply = "Please write the Shona phrase after #check, e.g., `#check Mhoroi`"
            source = "ℹ️ Info"
            with st.chat_message("assistant"):
                st.markdown(reply)
                st.caption(source)
            st.session_state.messages.append({"role": "assistant", "content": reply, "source": source})
        else:
            st.session_state.awaiting_english = True
            st.session_state.pending_shona = shona_phrase
            reply = f"You want to check: **{shona_phrase}**\n\nNow tell me the English meaning (e.g., `hello`)."
            source = "✏️ Correction Step 1/2"
            with st.chat_message("assistant"):
                st.markdown(reply)
                st.caption(source)
            st.session_state.messages.append({"role": "assistant", "content": reply, "source": source})
        st.rerun()

    translation, match_type = find_translation(prompt, st.session_state.exact_dict, st.session_state.vector_store)

    if match_type in ["exact", "constructed"]:
        reply = translation
       
        pron = get_pronunciation(translation, st.session_state.groq_client)
        if pron:
            reply += f"\n\n🔊 Pronunciation: {pron}"
        source = "From my Shona database"
    elif match_type == "vector" and st.session_state.groq_client:
        groq_prompt = f"""The user asked: "{prompt}"
A possible Shona translation is: "{translation}"
Write a short, helpful response (1 sentence) giving this as a possible translation.
Be friendly and honest about uncertainty."""
        try:
            groq_resp = st.session_state.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": groq_prompt}],
                temperature=0.5,
                max_tokens=100,
            )
            reply = groq_resp.choices[0].message.content
            source = "From AI (with database guidance)"
        except:
            reply = f"I found: {translation}"
            source = "From database (AI unavailable)"
    else:
        reply = "I don't have that translation in my database yet. Try asking for something else!"
        source = "Translation not found"

    with st.chat_message("assistant"):
        st.markdown(reply)
        st.caption(source)
    st.session_state.messages.append({"role": "assistant", "content": reply, "source": source})
    st.rerun()


st.markdown('<div class="footer"><p>🇿🇼 Rurimi rwaAmai – Preserving and teaching Shona 🇿🇼<br>Powered by Verified Translations + Groq AI</p></div>', unsafe_allow_html=True)