import openai
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from rank_bm25 import BM25Okapi
import os

# Set up Hugging Face API token for model access
os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'your_huggingface_token_here'

# Load and split PDF document
loader = PyPDFLoader(r'path_to_your_pdf_file')
pages = loader.load_and_split()

# Initialize Hugging Face embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# Embedding a sample query
embeddings.embed_query("sample query")

# Set up OpenAI API key
openai.api_key = 'your_openai_api_key_here'

# Create FAISS index from the PDF document using embeddings
db = FAISS.from_documents(pages, embeddings)
print(db)

# Set up the model using OpenAI's GPT-3.5 turbo
model = ChatOpenAI(openai_api_key="your_openai_api_key_here")

# Create retriever from FAISS index for document-based retrieval
retriever = db.as_retriever(search_kwargs={"k": 2})

# OpenAI client setup for making API requests
client = openai.OpenAI(api_key="your_openai_api_key_here")


# Function to ask OpenAI for a step-back question based on the given query
def ask_question(question):
    res = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "Generate a step-back question based on the given query. Give only the question, no other extra clarifying information."},
            {"role": "user", "content": f"query:{question}"}
        ]
    )
    return res.choices[0].message.content


# Function to get an answer from OpenAI based on a given context
def get_answer(question, context):
    res = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": f"Answer the following query based on this context:\n{context}"},
            {"role": "user", "content": f"query:{question}"}
        ]
    )
    return res.choices[0].message.content


# Function to generate an expanded query based on the given query
def generate_expanded_query(question):
    res = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "Generate an expanded query based on the given query. Give only the question, no other extra clarifying information."},
            {"role": "user", "content": f"query:{question}"}
        ]
    )
    return res.choices[0].message.content


# Function to get relevant context from the document using the retriever
def get_context(question):
    return retriever.get_relevant_documents(question)[0].page_content


# Function to generate a list of relevant questions based on the user's query
def generate_questions(user_query, num_questions=5):
    generated_questions = [user_query]
    for _ in range(num_questions - 1):
        question = ask_question(user_query)
        generated_questions.append(question)
    return generated_questions


# Function to generate answers for a list of questions based on the given context
def generate_answers(questions, context):
    generated_answers = []
    for q in questions:
        answer = get_answer(q, context)
        generated_answers.append(answer)
    return generated_answers


# Main execution

query = "Your initial query here"

# Get context for the query from the retriever
context = get_context(query)

# Generate an expanded version of the query
expanded_query = generate_expanded_query(query)

# Generate relevant questions based on the expanded query
questions = generate_questions(expanded_query)

# Generate answers for the questions based on the retrieved context
answers = generate_answers(questions, context)

# Function to rerank answers using the BM25 algorithm
def rerank_answers(user_query, generated_answers):
    # Create a BM25 index using the generated answers
    bm25_index = BM25Okapi(generated_answers)

    # Calculate BM25 scores for the user query
    bm25_scores = bm25_index.get_scores([user_query])

    # Sort the generated answers based on BM25 scores
    sorted_answers = [q for _, q in sorted(zip(bm25_scores, generated_answers), reverse=True)]

    # Return the top reranked answers
    return sorted_answers


# Rerank answers based on BM25 scoring
sorted_answers = rerank_answers(expanded_query, answers)

# Output the sorted answers
print("\nTop Reranked Answer: ", sorted_answers[0])

# Get the plain answer based on the original query and context
plain_answer = get_answer(query, context)
print("\nPlain Answer: ", plain_answer)
