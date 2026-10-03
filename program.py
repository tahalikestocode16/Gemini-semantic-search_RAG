from google import genai
import numpy as np

client = genai.Client()

documents = [
    "Refund requests are allowed within 30 days.",
    "Python classes happen Monday to Friday.",
    "Students get 3 interview opportunities.",
    "Course access is available for 6 months."
]

question = "Can I get my money back in 7 days?"


# Create embeddings for documents ==================================================================

document_vectors = []

for document in documents:
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=document
    )

    vector = result.embeddings[0].values
    document_vectors.append(vector)


# Create embedding for question ==================================================================

question_result = client.models.embed_content(
    model="gemini-embedding-2",
    contents=question
)

question_vector = question_result.embeddings[0].values


# Cosine similarity ====================================================================

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# Find the most relevant document ====================================================

scores = []

for document, vector in zip(documents, document_vectors):
    score = cosine_similarity(
        question_vector,
        vector
    )

    scores.append(score)

    print(f"Score: {score:.4f} | {document}")


best_idx = np.argmax(scores)
best_document = documents[best_idx]


# Send context to Gemini ============================================================

print("\nRetrieved Context:")
print(best_document)


prompt = f"""
Answer the question only using the context below.

Context:
{best_document}

Question:
{question}

If the context does not contain enough information to answer,
say that the information is not available.
"""


response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)


print("\nAnswer:")
print(response.text)

