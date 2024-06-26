import streamlit as st
import pandas as pd
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')

def generate_summary(teacher_feedback):
    # Tokenize the feedback into sentences
    sentences = sent_tokenize(teacher_feedback)

    if len(sentences) == 0:
        st.text("No feedback available.")

    # Encode sentences into BERT embeddings
    sentence_embeddings = model.encode(sentences)

    # Calculate the mean embedding of all sentences
    mean_embedding = sentence_embeddings.mean(axis=0, keepdims=True)

    # Calculate cosine similarity between each sentence embedding and the mean embedding
    cos_similarities = cosine_similarity(sentence_embeddings, mean_embedding)

    # Sort sentences by cosine similarity in descending order
    sorted_indices = cos_similarities.flatten().argsort()[::-1]

    # Select the top two sentences as representative
    num_sentences = min(1,2) # Adjust the number of sentences as needed
    representative_sentences = [sentences[idx] for idx in sorted_indices[:num_sentences]]

    # Generate summary
    summary = ' '.join(representative_sentences)
    return summary

st.header('RAMACHANDRA COLLEGE OF ENGINEERING')
st.title('STUDENT FEEDBACK ANALYZER')
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

csv=st.file_uploader('Enter CSV')
if csv:
    df = pd.read_csv(csv)
# Load the dataset
# Specify the range of teachers to consider
    #start_teacher = 1
    #end_teacher = 5  # Adjust as needed
    # Generate summary for each teacher in the specified range
    for i in df.columns[4],df.columns[6],df.columns[8],df.columns[10],df.columns[12]:
        #st.text(i)
        if i in df.columns and not df[i].isnull().all():
            teacher_feedback = df[i].dropna().str.cat(sep=' ')
            st.text("Summary of feedback for :"+i)
            st.text(generate_summary(teacher_feedback))

        else:
            st.text("No feedback available for Teacher"+i)
