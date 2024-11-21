import streamlit as st
import pymongo
from bson import ObjectId
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import process, fuzz

# MongoDB connection
client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['Course']
collection = db['Arxiv']

if "update_date_1" not in collection.index_information():
    collection.create_index([("update_date", pymongo.ASCENDING)])

# Cache MongoDB data
@st.cache_data


def fetch_documents():
    return list(collection.find({}, {
        "_id": 1, "title": 1, "abstract": 1, "categories": 1, "update_date": 1,
        "submitter": 1, "comments": 1, "journal_ref": 1, "authors": 1, "link": 1
    }))

documents = fetch_documents()

# Get unique filter values for adaptive indexing
submitters = sorted(set(doc.get("submitter", "Unknown") for doc in documents if doc.get("submitter")))
categories = sorted(set(cat for doc in documents for cat in doc.get("categories", [])))
update_dates = sorted(set(doc.get("update_date", "Unknown")[:4] for doc in documents if doc.get("update_date")))

# Prepare text data for TF-IDF
corpus = []
ids = []

for doc in documents:
    text = f"{doc.get('title', '')} {doc.get('abstract', '')} {' '.join(doc.get('categories', []))}"
    corpus.append(text)
    ids.append(doc["_id"])

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(corpus)

def recommend_documents(query, corpus, documents, limit=5):
    results = process.extract(query, corpus, scorer=fuzz.partial_ratio, limit=limit)
    recommended_docs = [documents[result[2]] for result in results]
    return recommended_docs

# Navigation using session state
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "selected_courses" not in st.session_state:
    st.session_state.selected_courses = []

def go_to_selected_courses():
    st.session_state.page = "SelectedCourses"
    st.experimental_rerun()

def go_home():
    st.session_state.page = "Home"
    st.experimental_rerun()

# Display pages based on state
if st.session_state.page == "Home":
    st.title("Enhanced Course Recommendation System")

    # Sidebar for adaptive indexing filters
    selected_update_date = st.sidebar.selectbox("Filter by Update Year", ["All"] + update_dates)  # Adaptive indexing for update date
    selected_submitter = st.sidebar.selectbox("Filter by Submitter", ["All"] + submitters)  # Adaptive indexing for submitter
    selected_category = st.sidebar.selectbox("Filter by Category", ["All"] + categories)  # Adaptive indexing for category

    # Filter documents based on sidebar selections
    # This section implements adaptive indexing based on selected filters
    filtered_documents = [
        doc for doc in documents
        if (selected_update_date == "All" or doc.get("update_date", "").startswith(selected_update_date)) and
           (selected_submitter == "All" or doc.get("submitter") == selected_submitter) and
           (selected_category == "All" or selected_category in doc.get("categories", []))
    ]

    # User query input
    query = st.text_input("Enter a search query (e.g., 'quantum computation'):")

    if query:
        corpus_filtered = [corpus[i] for i, doc in enumerate(documents) if doc in filtered_documents]
        recommended_docs = recommend_documents(query, corpus_filtered, filtered_documents,limit=50)

        st.write(f"Top recommendations for '{query}':")

        for i, rec in enumerate(recommended_docs):
            with st.expander(f"{i+1}. {rec['title']}"):
                st.write(f"**Submitter:** {rec.get('submitter', 'Unknown')}")
                st.write(f"**Update Year:** {rec.get('update_date', 'Unknown')[:4]}")
                st.write(f"**Categories:** {', '.join(rec.get('categories', []))}")
                st.write(f"**Abstract:** {rec['abstract']}")
                if st.button(f"Save Course {i+1}", key=f"save_{i}"):
                    st.session_state.selected_courses.append(rec)
                    st.success("Course saved!")

        # Button to view selected courses
        if st.button("View Selected Courses"):
            go_to_selected_courses()

elif st.session_state.page == "SelectedCourses":
    st.title("Saved Courses")

    if st.session_state.selected_courses:
        for course in st.session_state.selected_courses:
            st.write(f"### {course['title']}")
            st.write(f"**Submitter:** {course.get('submitter', 'Unknown')}")
            st.write(f"**Update Year:** {course.get('update_date', 'Unknown')[:4]}")
            st.write(f"**Categories:** {', '.join(course.get('categories', []))}")
            st.write(f"**Abstract:** {course['abstract']}")
            st.write(f"[Link to Paper]({course.get('link')})")
            st.write("---")

        if st.button("Back to Home"):
            go_home()
    else:
        st.write("No courses selected yet.")
        if st.button("Back to Home"):
            go_home()
