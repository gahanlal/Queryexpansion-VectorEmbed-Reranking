import openai
import rank_bm25

# Set your OpenAI API key
api_key =""
openai.api_key = api_key

def ask_question(question):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer.You are good at creating 1 step back question and 5 relevant questions based on the given query.Your job is to create questions from the given query"},
            {"role": "user", "content": f"query:{question}"}
        ]
    )
    return response.choices[0].message.content

def generate_questions(user_query, num_questions=5):
    generated_questions = []
    for _ in range(num_questions):
        question = ask_question(user_query)
        generated_questions.append(question)
    return generated_questions

def main():
    # Example user query
    user_query = "What are the types of medicines in the pharmacy?"

    # Generate relevant questions based on the user query
    generated_questions = generate_questions(user_query, num_questions=1)

    # Print the generated questions
    for idx, question in enumerate(generated_questions, start=1):
        print(f"Question {idx}: {question}")

    # BM25 reranking
    reranked_questions = rerank_questions(user_query, generated_questions, num_rerank=3)

    # Print the reranked questions
    print("\nReranked Questions:")
    for idx, question in enumerate(reranked_questions, start=1):
        print(f"Question {idx}: {question}")

def rerank_questions(user_query, generated_questions, num_rerank=3):
    # Create a BM25 index using the user query as the query document
    bm25_index = rank_bm25.BM25MinHash([user_query])

    # Calculate the BM25 scores for the generated questions
    bm25_scores = bm25_index.get_scores(generated_questions)

    # Rerank the generated questions based on the BM25 scores
    reranked_questions = []
    for _ in range(num_rerank):
        max_score = max(bm25_scores)
        max_score_index = bm25_scores.index(max_score)
        reranked_questions.append(generated_questions[max_score_index])
        bm25_scores[max_score_index] = -1

    # Return the reranked questions
    return reranked_questions

if __name__ == "__main__":
    main()
