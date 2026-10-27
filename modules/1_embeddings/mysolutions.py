# =============================================================================
# IMPORTS
# =============================================================================
import json
import numpy as np # For working with embeddings as vectors
import os # To access environment variables
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt  # For visualizing similarity scores for later
from dotenv import load_dotenv # To load environment variables from .env file
import time # For comparing speed of batch vs. single embedding

# =============================================================================
# SETUP: Load Environment Variables
# =============================================================================

load_dotenv()

# =============================================================================
# INITIALIZE OPENAI CLIENT
# =============================================================================

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')

# =============================================================================
# LOAD DATA: Support Tickets
# =============================================================================
print("\nLoading support tickets...")
with open('../../data/synthetic_tickets.json', 'r') as f:
    # tickets is a list of dictionaries, each representing a support ticket
    tickets = json.load(f)

# texts is a list of strings, of length N = 20
texts = [f"{ticket['title']}. {ticket['description']}" for ticket in tickets]
response = client.embeddings.create(
    model=embedding_model,
    input=texts
)

# response.data is a list of objects, length N = 20, each has '.embedding' attribute.
# embeddings is a 2D NumPy array of shape (N, 1536), where each row is the embedding for a ticket
embeddings = np.array([data.embedding for data in response.data]) 

print(f"✓ Loaded {len(tickets)} tickets\n")
print(f"✓ Generated embeddings with shape: {embeddings.shape}")


# ============================================================================
# Exercise 1: Change the Search Query (Easy)
# ============================================================================
print("=" * 80)
print("EXERCISE 1: Change the Search Query")
print("=" * 80)

query = "Database is running very slowly"  # Changed from original

query_response = client.embeddings.create(
    model=embedding_model,
    input=[query]
)

# query_embedding is a 1D NumPy array of shape (1536,), wrapped as [query_embedding] for cosine similarity, shape (1, 1536)
query_embedding = np.array(query_response.data[0].embedding) 

# cosine_similarity is a 2D NumPy array of shape (1, N). [0] takes the only row -> similarity scores, 1D array, shape (N,)
similarity_scores = cosine_similarity([query_embedding], embeddings)[0] 

top_k = 5
top_k_indices = similarity_scores.argsort()[-top_k:][::-1] 

print(f"\nQuery: '{query}'")

print(f"\nTop {top_k} tickets most similar to the query:")
for rank, idx in enumerate(top_k_indices, start=1):
    ticket = tickets[idx]
    score = similarity_scores[idx]
    print(f"{rank}. Ticket ID: {ticket['ticket_id']}")
    print(f"    Similarity Score: {score:.4f}")
    print(f"    Title: {ticket['title']}")
    print(f"    Category: {ticket['category']}")
    print(f"    Description: {ticket['description'][:150]}...\n")

# ============================================================================
# Exercise 2: Adjust the Number of Results (Easy)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 2: Adjust the Number of Results")
print("=" * 80)

# Solution: Change top_k from 5 to 10
top_k = 10

query = "Users can't login after changing password"
query_response = client.embeddings.create(
    model=embedding_model,
    input=[query]
)
query_embedding = np.array(query_response.data[0].embedding)
similarity_scores = cosine_similarity([query_embedding], embeddings)[0] 

top_k_indices = similarity_scores.argsort()[-top_k:][::-1]

print(f"\nQuery: '{query}'")
print(f"Showing top {top_k} results:")
print("-" * 50)

below_threshold_rank = None
for rank, idx in enumerate(top_k_indices, start=1):
    ticket = tickets[idx]
    score = similarity_scores[idx]
    print(f"{rank}. Ticket ID: {ticket['ticket_id']}")
    print(f"    Similarity Score: {score:.4f}")
    print(f"    Title: {ticket['title']}")
    print(f"    Category: {ticket['category']}")
    print(f"    Description: {ticket['description'][:150]}...\n")
    
    # Check if the similarity score is below 0.5
    if score < 0.5 and below_threshold_rank is None:
        below_threshold_rank = rank

print(f"\n Score drops below 0.5 at rank: {below_threshold_rank}" if below_threshold_rank else "\n All scores are above 0.5")

# ============================================================================
# Exercise 3: Add a Similarity Threshold (Easy)
# ============================================================================

print("\n" + "=" * 80)
print("EXERCISE 3: Add a Similarity Threshold")
print("=" * 80)

def search_with_threshold(query_text, threshold=0.5, top_k=10):
    query_response = client.embeddings.create(
        model=embedding_model,
        input=[query_text]
    )
    query_embedding = np.array(query_response.data[0].embedding)
    similarity_scores = cosine_similarity([query_embedding], embeddings)[0] 

    top_k_indices = similarity_scores.argsort()[::-1][:top_k]

    print(f"\nQuery: '{query_text}' (threshold: {threshold})")
    print("-" * 50)

    count_above_threshold = 0
    for rank, idx in enumerate(top_k_indices, start=1):
        ticket = tickets[idx]
        score = similarity_scores[idx]
        if score < threshold:
            continue  # Skip tickets below the threshold
        count_above_threshold += 1
        print(f"\n {rank}. Ticket ID: {ticket['ticket_id']}")
        print(f"    Similarity Score: {score:.4f}")
        print(f"    Title: {ticket['title']}")
        print(f"    Category: {ticket['category']}")
        print(f"    Description: {ticket['description'][:150]}...")

    if count_above_threshold == 0:
        print(f"\n No tickets found with similarity score above {threshold}")
    else:
        print(f"\n Found {count_above_threshold} tickets with similarity score above {threshold}")

# Test with relevant query
search_with_threshold("Users can't login after changing password")
search_with_threshold("login Authentication problem")
search_with_threshold("How to make pizza")

# ============================================================================
# Exercise 4: Compare Two Queries (Easy)
# ============================================================================

print("\n" + "=" * 80)
print("EXERCISE 4: Compare Two Queries")
print("=" * 80)

# Compare two queries
query1 = "Login authentication failed"
query_2 = "Slow database performance"

for q in [query1, query_2]:
    response = client.embeddings.create(
        model=embedding_model,
        input=[q]
    )
    query_embedding = np.array(response.data[0].embedding)
    similarity_scores = cosine_similarity([query_embedding], embeddings)[0]
    top_index = similarity_scores.argmax()

    print(f"\nQuery: '{q}'")
    print(f"    Best Match: {tickets[top_index]['title']}")
    print(f"    Category: {tickets[top_index]['category']}")
    print(f"    Similarity Score: {similarity_scores[top_index]:.4f}")

# ============================================================================
# Exercise 5: Test Semantic Understanding (Medium)
# ============================================================================

print("\n" + "=" * 80)
print("EXERCISE 5: Test Semantic Understanding")
print("=" * 80)

# These mean the SAME thing but use DIFFERENT words
texts = [
    "User authentication failed",      # Original
    "Login credentials rejected",       # Same meaning, different words
    "Cannot sign in to account",        # Same meaning, different words
    "Database connection timeout",      # DIFFERENT topic
]

# Generate embeddings for these texts
response = client.embeddings.create(
    model=embedding_model,
    input=texts
)
embeddings = np.array([data.embedding for data in response.data])

# Calculate all pairwise cosine similarities
similarity_matrix = cosine_similarity(embeddings)

# Print results
print("\n Similarity Matrix:")
print('-'*50)

for i, text1 in enumerate(texts):
    for j, text2 in enumerate(texts):
        if i < j: # Print upper triangle only
            sim = similarity_matrix[i][j]
            print(f"{sim:.3f}  '{text1[:30]}...' vs '{text2[:30]}...'")

print("\n→ Notice: Auth/Login texts have HIGH similarity despite different words")
print("→ Notice: Database text has LOW similarity - different topic entirely")

# ============================================================================
# Exercise 6: Filter by Category (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 6: Filter by Category")
print("=" * 80)

def search_with_category(query, category_filter=None, top_k=5):
    """Search tickets, optionally filtering by category"""

    print(f"\nQuery: '{query}' (category filter: {category_filter})")

    # Get query embedding
    response = client.embeddings.create(
        model=embedding_model,
        input=[query]
    )
    query_emb = np.array([response.data[0].embedding])

    # Calculate similarities
    similarities = cosine_similarity(query_emb, embeddings)[0]

    # Get results with category filter
    results = []
    for idx in np.argsort(similarities)[::-1]:
        ticket = tickets[idx]

        # THE SOLUTION: skip if category doesn't match filter
        # If category_filter is set AND ticket category doesn't match, skip
        if category_filter and ticket['category'] != category_filter:
            continue

        results.append((ticket, similarities[idx]))

        if len(results) >= top_k:
            break

    return results

# Test it
print("All categories:")
for ticket, score in search_with_category("login problem"):
    print(f"  {score:.3f} [{ticket['category']}] {ticket['title']}")

print("\nOnly 'Authentication' category:")
for ticket, score in search_with_category("login problem", category_filter="Authentication"):
    print(f"  {score:.3f} [{ticket['category']}] {ticket['title']}")

# ============================================================================
# Exercise 7: Batch vs Single Embedding (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 7: Batch vs Single Embedding")
print("=" * 80)

batch_texts = [
    "Password reset not working",
    "Database connection timeout", 
    "App crashes on startup",
    "Payment declined error",
    "Email notifications delayed",
]

# Method 1: SLOW - One API call per text
print("Method 1: Single API calls...")
start = time.time()
for text in batch_texts:
    response = client.embeddings.create(model=embedding_model, input=[text])
time_slow = time.time() - start
print(f"  Time: {time_slow:.2f} seconds")

# Method 2: FAST - One API call for all texts
print("\nMethod 2: Batch API call...")
start = time.time()
response = client.embeddings.create(model=embedding_model, input=texts)
time_fast = time.time() - start
print(f"  Time: {time_fast:.2f} seconds")

# Compare
speedup = time_slow / time_fast if time_fast > 0 else float('inf')
print(f"\n✓ Batch is {speedup:.1f}x faster!")
print("  Always batch your embeddings in production!")

# ============================================================================
# Bonus: Similarity Matrix Heatmap (Challenge)
# ============================================================================
print("\n" + "=" * 80)
print("BONUS: Similarity Matrix Heatmap")
print("=" * 80)

# Use first 10 tickets
sample_tickets = tickets[:10]
sample_texts = [t['title'] for t in sample_tickets]

# Generate embeddings
response = client.embeddings.create(
    model=embedding_model,
    input=sample_texts
)
sample_embeddings = np.array([data.embedding for data in response.data])

# Compute similarity matrix
sim_matrix = cosine_similarity(sample_embeddings)

# Create heatmap
plt.figure(figsize=(12,10))
plt.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1)
plt.colorbar(label='Cosine Similarity')

# Add labels
labels = [f"{t['ticket_id']}\n{t['category'][:8]}" for t in sample_tickets]
plt.xticks(range(10), labels, rotation=45, ha='right', fontsize=8)
plt.yticks(range(10), labels, fontsize=8)

# Add values to cells
for i in range(10):
    for j in range(10):
        plt.text(
            j, i, f'{sim_matrix[i, j]:.2f}',
            ha='center', va='center', fontsize=8,
            color='white' if sim_matrix[i, j] < 0.5 else 'black'
        )
plt.title('Ticket Similarity Matrix\n(First 10 Tickets)')
plt.tight_layout()
plt.savefig('mysolutions_similarity_heatmap.png', dpi=150, bbox_inches='tight')
print("✓ Saved as solution_similarity_heatmap.png")
# plt.show() # Uncomment to display the heatmap in an interactive environment

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("ALL SOLUTIONS COMPLETE!")
print("=" * 80)
print("""
Key Takeaways:
──────────────
1. Embeddings capture semantic meaning, not just keywords
2. Cosine similarity measures how related two texts are
3. Threshold filtering removes irrelevant results
4. Category filtering narrows search to specific domains
5. Batch API calls are 5-10x faster than single calls
6. Similarity matrices reveal relationships between documents

Next: Move on to Module 2 - Chunking & Vector Stores!
""")
