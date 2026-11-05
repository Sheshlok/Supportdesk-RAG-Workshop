# -*- coding: utf-8 -*-
"""
Module 4 Solutions: RAG Pipeline
================================

Solutions for all exercises in exercises.md
"""
import json
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ============================================================================
# Setup: Load data and create/load vector store
# ============================================================================
print("Loading data...")
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)
print(f"✓ Loaded {len(tickets)} support tickets")

documents = []
for ticket in tickets:
    content = f"""
Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Category: {ticket['category']}
Priority: {ticket['priority']}

Problem Description:
{ticket['description']}

Resolution:
{ticket['resolution']}
""".strip()

    doc = Document(
        page_content=content,
        metadata={
            "ticket_id": ticket['ticket_id'],
            "title": ticket['title'],
            "category": ticket['category'],
            "priority": ticket['priority']
        }
    )
    documents.append(doc)

print(f"✓ Loaded {len(documents)} documents")

print("Creating vector store...")
embeddings = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
)

persist_directory = "./rag_vector_store_solutions"

if os.path.exists(persist_directory):
    print(f"Vector store already exists at {persist_directory}. Loading existing store...")
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="solutions_test",
        collection_metadata={"hnsw:space": "cosine"}
    )
else:
    print(f"Vector store does not exist at {persist_directory}. Creating new store...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="solutions_test",
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=persist_directory
    )

print("✓ Vector store ready")

# Create retreiver
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Initialize LLM
if os.getenv("OPENAI_API_KEY"):
    print("✓ OpenAI API key found")
    llm = ChatOpenAI(
        model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'),
        temperature=0,
        timeout=120,
        max_retries=3
    )
    print(f"✓ Using OpenAI model: {llm.model_name}")
else:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

def format_docs(docs):
    """
    Convert retrieved document objects into one prompt-ready context string.

    Why this is useful?
    - Keeps all retrieved evidence in one place for the LLM.
    - Preserves chunk boundaries with separators for readability.
    - Reused across many exercises to keep chain assembly concise.
    """
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

# ============================================================================
# Exercise 1: Modify the Prompt Template (Easy)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 1: Modify the Prompt Template")
print("=" * 80)

# Version A: Concise
template_a =  """Answer the question using only the ticket context below. Cite ticket IDs.

Context: {context}

Question: {question}

Answer:"""

# Version B: Step-by-step
template_b = """You are a support assistant. Answer using ONLY the context below.

Context: {context}

Question: {question}

Think step by step:
1. What tickets are relevant?
2. What information do they contain?
3. How does this answer the question?

Answer:"""

# Version C: Bullet points
template_c = """Answer using only the context. Format as bullet points with ticket citations.

Context: {context}

Question: {question}

Answer (bullet points with sources):"""

templates = [
    ("Concise", template_a),
    ("Step-by-step", template_b),
    ("Bullet points", template_c)
]

test_query = "How do I fix authentication issues?"

print(f"\nTest Query: '{test_query}'")


# for name, template in templates:
#     prompt = ChatPromptTemplate.from_template(template)
#     chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
#     print(f"\n\n >> {name} template:")
#     response = chain.invoke(test_query)
#     print(f"  {response[:200]}...")

# Streaming Responses while saving each template's answer to compare later
answers = {}

for name, template in templates:
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm   # the component, not llm.stream()
        | StrOutputParser()
    )
    print(f"\n\n >> {name} template (streaming):")

    # Collecting the full answer as you go
    answer = ""
    for token in chain.stream(test_query):
        print(token, end="", flush=True)
        answer += token

    answers[name] = answer  # save the full answer under the template's name. Later: answers["Concise"], ...

# ============================================================================
# Exercise 2: Adjust Retrieval Parameters (Easy)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 2: Adjust Retrieval Parameters")
print("=" * 80)

test_query="Payment processing failures"

print(f"\nTest Query: '{test_query}'")

# Test different k values
for k in [1, 3, 5, 10]:
    retriever_k = vector_store.as_retriever(search_kwargs={"k": k})
    #print(f"\nRetriever with k={k}: {retriever_k}")
    docs = retriever_k.invoke(test_query)
    print(f"\nk={k}: Retrieved {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.metadata['ticket_id']}: {doc.metadata['title'][:40]}...")

print("\n\nMMR search (diverse results):")
retriever_mmr = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3, # number of documents to retrieve using MMR
        "fetch_k": 10  # number of top documents to fetch before applying MMR using cosine similarity
        # "lambda_mult": 0.5  # trade-off: 1.0 = pure relevance, 0.0 = pure diversity
                              # default 0.5 balances both equally
    }
)

docs_mmr = retriever_mmr.invoke(test_query)
print(f"MMR: Retrieved {len(docs_mmr)} documents")
for doc in docs_mmr:
    print(f"  - {doc.metadata['ticket_id']}: {doc.metadata['title'][:40]}...")

# ============================================================================
# Exercise 3: Implement Citation Formatting (Easy)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 3: Implement Citation Formatting")
print("=" * 80)

citation_prompt = """Answer the question using the context. Include inline citations [TICK-XXX] after each fact.

Example format:
"Database connection timeouts occur when the pool is undersized [TICK-002]. Increase max_connections [TICK-002]."

Context:
{context}

Question: {question}

Answer with inline citations:"""

prompt = ChatPromptTemplate.from_template(citation_prompt)

citation_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "What causes authentication failures and how do I fix them?"
print(f"\nQuery: {query}")
print(f"\nAnswer with citations:\n")
for token in citation_chain.stream(query):
    print(token, end="", flush=True)
print()

# ============================================================================
# Exercise 4: Build a Fallback System (Easy)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 4: Build a Fallback System")
print("=" * 80)

def smart_rag(query, vector_store, llm, min_similarity_score=0.45):
    docs_with_relevance_scores = vector_store.similarity_search_with_relevance_scores(query, k=3)
    best_score = docs_with_relevance_scores[0][1] if docs_with_relevance_scores else 0
    if best_score < min_similarity_score:
        print(f"\n⚠ Best match relevance ({best_score:.4f}) is below threshold ({min_similarity_score}) — too dissimilar to answer confidently")
        return "I don't have enough relevant information to answer that question confidently.", "low confidence"

    # If the best score is above the threshold, proceed with normal RAG retrieval
    docs = [doc for doc, score in docs_with_relevance_scores]
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Answer using only this context. Cite ticket IDs.
    
Context: {context}
    
Question: {query}
    
Answer:"""

    if llm:
        response = llm.invoke(prompt)
        return response.content, "high confidence"
    else:
        return "(LLM not configured)"

# Test with different queries
test_queries = [
    ("authentication problems", "high confidence expected"),
    ("how to bake cookies", "low confidence expected")
]

for query, note in test_queries:
    print(f"\nQuery: '{query}' ({note})")
    # Here you would invoke your fallback system with the query
    response, confidence = smart_rag(query, vector_store, llm)
    print(f"Response from fallback system: {response[:150]}... ({confidence})")

# ============================================================================
# Exercise 5: Compare Chain Types (Medium)
# ============================================================================
# When you have multiple retrieved documents, how do you combine them
# to generate a single answer?  This exercise compares three strategies:
#
# ── STUFF ─────────────────────────────────────────────────────────────
#   Query → Retrieve 3 docs → Concatenate all into ONE prompt → LLM → Answer
#
#   ┌──────────────────────────────────────────┐
#   │  Prompt:                                 │
#   │  Context: [Doc1] --- [Doc2] --- [Doc3]   │  ← all docs concatenated
#   │  Question: How do I fix DB timeouts?     │
#   └──────────────────────────┬───────────────┘
#                              ▼
#                         LLM (1 call)
#                              ▼
#                           Answer
#
#   Pros: Fast (1 LLM call), simple to implement.
#   Cons: Breaks if total context exceeds the model's token limit.
#
# ── MAP_REDUCE ────────────────────────────────────────────────────────
#   Query → Retrieve 3 docs → Process EACH doc separately → Combine → Answer
#
#   Two phases:
#     1. Map   — Each doc gets its own LLM call ("extract key info").
#     2. Reduce — A final LLM call combines the summaries into one answer.
#
#                       ┌─── Doc1 → LLM → Summary1
#                       │
#   Query → Retrieve ───┼─── Doc2 → LLM → Summary2    (Map: 3 LLM calls)
#                       │
#                       └─── Doc3 → LLM → Summary3
#                                   │
#                                   ▼
#                       Combine summaries → LLM → Final Answer  (Reduce: 1 call)
#
#   Pros: Handles unlimited docs (each fits in one prompt); map calls can
#         run in parallel.
#   Cons: Slower (N+1 LLM calls), more expensive, summaries may lose detail.
#
# ── REFINE ─────────────────────────────────────────────────────────────
#   Query → Retrieve 3 docs → Process sequentially, refining each time
#
#   Doc1 → LLM → Draft answer
#   Doc2 + Draft → LLM → Refined answer
#   Doc3 + Refined → LLM → Final answer
#
#   Pros: Highest quality — each step builds on the previous answer.
#   Cons: Slowest (sequential, can't parallelize), N serial LLM calls.
#
# ── For this support ticket system (small docs, k=3) ──────────────────
#   STUFF is the right default — all 3 tickets easily fit within the
#   context window, so there's no reason to pay the latency/cost of
#   multiple LLM calls.
#
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("EXERCISE 5: Compare Chain Types")
print("=" * 80)

import time

# Compare different retrieval strategies using LCEL
strategies = {
    "stuff": "Concatenate all documents into context (fast, simple)",
    "map_reduce": "Process/combine each doc separately, then combine (parallel)",
    "refine": "Iteratively refine answer with each doc (highest quality, slow)"
}

test_query = "How do I fix database timeouts?"

print(f"\nTest Query: '{test_query}'")
# ── STUFF strategy (default) ──────────────────────────────────────────
# All retrieved docs are "stuffed" into a single {context} variable
# and sent to the LLM in ONE call.
print("\n--- STUFF Strategy ---")
start_time = time.time()
stuff_prompt = ChatPromptTemplate.from_template(
    "Answer using the context:\n\nContext: {context}\n\nQuestion: {question}"
)
stuff_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | stuff_prompt
    | llm
    | StrOutputParser()
)

for token in stuff_chain.stream(test_query):
    print(token, end="", flush=True)
print()

print(f"\n  Time: {time.time() - start_time:.2f}s")

# ── MAP_REDUCE strategy ───────────────────────────────────────────────
# Phase 1 (Map):  Each doc → individual LLM call → summary
# Phase 2 (Reduce): All summaries → one LLM call → final answer
print("\n--- MAP_REDUCE Strategy ---")
start_time = time.time()
docs = retriever.invoke(test_query)

# MAP phase: Extract key info from each document separately
individual_answers = []
for doc in docs:
    single_prompt = ChatPromptTemplate.from_template(
        "Extract key info about this issue:\n{doc}\n\nKey points:"
    )
    chain = single_prompt | llm | StrOutputParser()
    individual_answers.append(chain.invoke({"doc": doc.page_content}))

# REDUCE phase: Combine all per-doc summaries into a single coherent answer
combine_prompt = ChatPromptTemplate.from_template(
    "Combine these points to answer: {question}\n\nPoints:\n{summaries}"
)

combine_chain = combine_prompt | llm | StrOutputParser()
for token in combine_chain.stream({"question": test_query, "summaries": "\n".join(individual_answers)}):
    print(token, end="", flush=True)
print()
print(f"\n Time: {time.time() - start_time:.2f}s")

# ── REFINE strategy ───────────────────────────────────────────────────
# Process docs sequentially. Start with the first doc to produce a draft,
# then feed each subsequent doc + the current draft to the LLM to refine.
#
#   Doc1 → LLM → Draft answer
#   Doc2 + Draft → LLM → Refined answer
#   Doc3 + Refined → LLM → Final answer
#
# Each step can incorporate new info AND correct earlier mistakes.
print("\n--- REFINE Strategy ---")
start_time = time.time()
docs = retriever.invoke(test_query)
# Step 1: Generate initial draft from the first document
initial_prompt = ChatPromptTemplate.from_template(
    "Answer the question using only this context.\n\n"
    "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
)
initial_chain = initial_prompt | llm | StrOutputParser()
initial_inputs = {"context": docs[0].page_content, "question": test_query}
if len(docs) > 1:
    current_answer = initial_chain.invoke(initial_inputs)  # intermediate: must be complete
else:
    # Only one doc: the initial draft is the final answer, so stream it
    print("Refined answer:\n")
    chunks = []
    for token in initial_chain.stream(initial_inputs):
        print(token, end="", flush=True)
        chunks.append(token)
    current_answer = "".join(chunks)
    print()

# Steps 2..N: Refine with each remaining doc
refine_prompt = ChatPromptTemplate.from_template(
    "Here is an existing answer to the question:\n\n"
    "Existing answer: {existing_answer}\n\n"
    "Now consider this additional context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Refine the existing answer using the new context. "
    "If the new context isn't useful, return the existing answer unchanged.\n\n"
    "Refined answer:"
)

refine_chain = refine_prompt | llm | StrOutputParser()
for i, doc in enumerate(docs[1:]):
    inputs = {"existing_answer": current_answer, "context": doc.page_content, "question": test_query}
    if i < len(docs) - 2:
        current_answer = refine_chain.invoke(inputs)       # intermediate: must be complete
    else:
        print("Refined answer:\n")
        chunks = []
        for token in refine_chain.stream(inputs):          # last step: stream
            print(token, end="", flush=True)
            chunks.append(token)
        current_answer = "".join(chunks)
        print()
print(f"\n  Time: {time.time() - start_time:.2f}s")

print("\n→ 'stuff': Fastest (1 LLM call), concatenates all docs into one prompt")
print("→ 'map_reduce': Parallel processing (N+1 calls), good for many docs")
print("→ 'refine': Iterative (N sequential calls), highest quality but slowest")

# ============================================================================
# Exercise 6: Add Metadata Filtering (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 6: Add Metadata Filtering")
print("=" * 80)

query = "system problem"

print(f"\nQuery: {query}")

# Without filter - returns any category
print("\nWithout filter:")
docs = vector_store.similarity_search(query, k=3)
for doc in docs:
    print(f"  - {doc.metadata['ticket_id']} ({doc.metadata['category']})")

# With category filter
print("\nWith category='Authentication' filter:")
docs = vector_store.similarity_search(
    query, 
    k=3,
    filter={"category": "Authentication"}
)
for doc in docs:
    print(f"  - {doc.metadata['ticket_id']} ({doc.metadata['category']})")

# With priority filter
print("\nWith priority='High' filter:")
docs = vector_store.similarity_search(
    query, 
    k=3,
    filter={"priority": "High"}
)
for doc in docs:
    print(f"  - {doc.metadata['ticket_id']} ({doc.metadata['priority']})")

# ============================================================================
# Exercise 7: Add Streaming Responses (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 7: Add Streaming Responses")
print("=" * 80)

# Build chain
prompt = ChatPromptTemplate.from_template("""Answer using the context. Provide a detailed, thorough response.
Include step-by-step troubleshooting instructions, root causes, and preventive measures.
Cite ticket IDs for every fact.

Context: {context}

Question: {question}

Detailed Answer:""")

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n Streaming response: ")
query = "What causes database connection issues and how do I troubleshoot and prevent them?"
for token in chain.stream(query):
    print(token, end="", flush=True)
print()

# ============================================================================
# Exercise 8: Multi-Turn Conversation (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 8: Multi-Turn Conversation")
print("=" * 80)

from operator import itemgetter # used for extracting specific fields from dictionaries or objects
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

# ── Step 1: Query Reformulation ───────────────────────────────────────
# Rewrites follow-ups into standalone queries using chat history
# e.g. "How do I fix it?" → "How do I fix authentication failures?"
condense_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and a follow-up question, decide whether the "
     "follow-up depends on the history.\n"
     "- If it contains pronouns or references to earlier turns "
     "(e.g. 'it', 'that', 'this issue'), rewrite it as a standalone question "
     "by replacing those references with what they refer to.\n"
     "- If it names its own subject or introduces a new topic (e.g. 'What about "
     "database issues?'), return it UNCHANGED. Do not add any topic or details "
     "from the history.\n"
     "Output only the question, with no explanation."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# Chain that rewrites the question
condense_chain = condense_prompt | llm | StrOutputParser()

# Step 2: Conversational prompt with context and history
conv_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are SupportDesk AI. Answer using ONLY the context below.
Always cite ticket IDs. If the answer isn't in the context, say "I don't have that information."

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

conv_chain = (
    {
        "context": condense_chain | retriever | format_docs,
        "chat_history": itemgetter("chat_history"),
        "question": itemgetter("question")
    }
    | conv_prompt
    | llm
    | StrOutputParser()
)

def ask_with_history(question, chat_history):
    if not chat_history:
        standalone = question
    else:
        standalone = condense_chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
    if standalone != question:
        print(f"         [Rewritten query → {standalone}]")

    answer = conv_chain.invoke({
        "chat_history": chat_history,
        "question": question # Show the original question in the prompt, not the rewritten one
    })

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    return answer

# Simulate a 3-turn conversation
# Use specific queries that match ticket content for reliable retrieval
conversation = [
    "Why are users unable to log in after a password reset?",
    "How do I fix it?",          # "it" → login issue (remembered from turn 1)
    "What about database issues?"  # new topic — no cross-contamination from history
]

print("\nSimulated conversation:")
# Start empty chat history
chat_history = []
for user_msg in conversation:
    print(f"\n>> User: {user_msg}")
    result = ask_with_history(user_msg, chat_history)
    print(f">> Assistant: {result[:200]}...")

# ============================================================================
# Bonus: Hallucination Detection (Challenge)
# ============================================================================
print("\n" + "=" * 80)
print("BONUS: Hallucination Detection")
print("=" * 80)

def detect_hallucination(query, answer, source_documents, llm):
    """
    Use LLM-as-judge to check if an answer is grounded in retrieved sources.

    Practical purpose:
    - Adds a post-generation QA gate.
    - Helps flag unsupported claims before returning responses in production.
    """
    source_text = "\n\n".join([doc.page_content for doc in source_documents])
    
    detection_prompt = f"""You are a fact-checker. Determine if the answer is fully grounded in the source documents.

SOURCE DOCUMENTS:
{source_text}

ANSWER TO CHECK:
{answer}

Is every claim in the answer supported by the source documents?
Respond with:
- "GROUNDED" if all claims are supported
- "HALLUCINATION" if any claims are not in sources
- Brief explanation

Response:"""

    response = llm.invoke(detection_prompt)
    return response.content

test_query = "How do I fix authentication issues?"
source_docs = retriever.invoke(test_query)
answer_prompt = ChatPromptTemplate.from_template(
    "Answer using ONLY the context. Cite ticket IDs.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
)
answer_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | answer_prompt
    | llm 
    | StrOutputParser()
)
answer = answer_chain.invoke(test_query)

print(f"\nQuery: {test_query}")
print(f"Answer: {answer[:200]}...")
print(f"\nHallucination check:")
check_result = detect_hallucination(test_query, answer, source_docs, llm)
print(check_result)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("ALL SOLUTIONS COMPLETE!")
print("=" * 80)
print("""
Key Takeaways:
──────────────
1.  Prompt template design significantly affects answer quality
2.  k parameter trades off coverage vs relevance
3.  MMR provides diverse results when needed
4.  Fallbacks handle low-confidence situations gracefully
5.  Different chain types suit different use cases:
    - stuff: Fast, good for small context
    - map_reduce: Parallelizable, handles large docs
    - refine: Highest quality, slow
6.  Streaming improves UX for long answers
7.  Memory enables multi-turn conversations
8.  LLM-as-judge can detect hallucinations

Next: Move on to Module 5 - Evaluation!
""")
