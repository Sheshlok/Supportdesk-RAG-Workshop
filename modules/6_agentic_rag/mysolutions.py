# -*- coding: utf-8 -*-
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from mytools import SupportTicketToolsEnhanced

from dotenv import load_dotenv
load_dotenv()   

# Initialize enhanced tools
print("Creating enhanced tools...")
tool_manager = SupportTicketToolsEnhanced()
tools = tool_manager.get_tools() 

print(f"✓ Created {len(tools)} tools")

# Initialize LLM with tools
llm = ChatOpenAI(model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'), temperature=0)
llm_with_tools = llm.bind_tools(tools)
tool_map = {tool.name: tool for tool in tools}

# ============================================================================
# Exercise 6: Custom Agent Prompt
# ============================================================================
CUSTOM_SYSTEM_PROMPT = """You are an expert support desk assistant with access to a ticket database.

ALWAYS follow these rules:
1. State which tool you're about to use before calling it
2. If the query is ambiguous, ask a clarifying question FIRST
3. When providing solutions, rate your confidence (High/Medium/Low)
4. After answering, suggest one related topic the user might want to explore
5. Reference ticket IDs when providing information

Your tools:
- SearchSimilarTickets: For troubleshooting and "how to fix" questions
- GetTicketByID: For looking up specific tickets (TICK-001, etc.)
- SearchByCategory: For finding all tickets in a category
- SearchByPriority: For finding all tickets with a specific priority (e.g., Critical)
- GetTicketStatistics: For database overview and counts
"""
def run_agent(query: str, max_iterations: int = 5, custom_prompt: str = None):
    system_prompt = custom_prompt or CUSTOM_SYSTEM_PROMPT
    messages = [
        SystemMessage(content=system_prompt), 
        HumanMessage(content=query)
    ]

    tools_used = []

    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return {
                'response': response.content,
                'tools_used': tools_used,
                'iterations': i + 1
            }

        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            print(f"\n🔧 Tool: {tool_name} | Args: {str(tool_call['args'])[:50]}...")
            tools_used.append(tool_name)

            tool = tool_map.get(tool_name)

            if tool is None:
                messages.append(ToolMessage(
                    content=f"Error: Tool {tool_name} not found",
                    tool_call_id=tool_call["id"]
                ))
            else:
                # Tools take a single string; the model may send {} for no-input tools.
                args = tool_call['args']
                tool_input = next(iter(args.values()), "")
                messages.append(ToolMessage(
                    content=tool.func(tool_input),
                    tool_call_id=tool_call["id"]
                ))
    return {
        'response': "Maximum iterations reached without a final response",
        'tools_used': tools_used,
        'iterations': max_iterations
    }

# ============================================================================
# Exercise 1 & 2: Test Different Queries
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 1 & 2: Test Different Queries")
print("=" * 80)

test_queries = [
    ("How do I fix authentication problems?", "SearchSimilarTickets"),
    ("Show me ticket TICK-005", "GetTicketByID"),
    ("What payment issues have we seen?", "SearchByCategory"),
    ("Give me an overview of tickets", "GetTicketStatistics"),
    ("How many critical tickets are there?", "SearchByPriority"),  # Exercise 4 small patch
]

for i, (query, expected_tool) in enumerate(test_queries, 1):
    print(f"\n--- Query {i}: '{query}' ---")
    print(f"Expected tool: {expected_tool}")
        
    result = run_agent(query)
        
    print(f"Actual tools used: {result['tools_used']}")
    print(f"Iterations: {result['iterations']}")
    correct = expected_tool in result['tools_used']
    print(f"Correct: {'✓' if correct else '✗'}")
    print(f"Response: {result['response'][:150]}...")

# ============================================================================
# Exercise 5: Multi-Step Queries
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 5: Multi-Step Queries")
print("=" * 80)

multi_step_queries = [
    "How many payment tickets do we have and what was the most recent resolution?",
    "Find all high priority tickets and give me details on the first one",
    "Compare the resolution for TICK-001 and TICK-005",
    "What Authentication issues do we have? Give me details on each one.",
]

for query in multi_step_queries:
    print(f"\n--- Multi-step Query ---")
    print(f"Query: '{query}'")
    
    result = run_agent(query)
    
    print(f"Tools used (in order): {result['tools_used']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Response: {result['response'][:200]}...")

# ============================================================================
# Exercise 7: Interactive Conversation Loop
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 7: Interactive Chat [without chat history]")
print("=" * 80)

while True:
    user_input = input("\nYou: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    if not user_input:
        continue
    
    response = run_agent(user_input)
    print(f"\nAssistant: {response}")

# ============================================================================
# Exercise 8: Error Handling
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 8: Error Handling")
print("=" * 80)

error_test_cases = [
    "Get ticket XYZ123",           # Invalid format
    "Get ticket TICK-999",          # Not found
    "Show me category FooBar",      # Invalid category
    "Show me critical incidents",   # Priority-aware branch in existing tool
]

for query in error_test_cases:
    print(f"\n--- Error case: '{query}' ---")
    result = run_agent(query)
    print(f"Response: {result['response'][:150]}...")

# ============================================================================
# Bonus: Conversation with Memory
# ============================================================================
print("\n" + "=" * 80)
print("BONUS: Conversation with Memory")
print("=" * 80)

conversation_history = []

def chat_with_memory(user_message: str, max_iterations: int = 5) -> str:
    """
    Memory-enabled chat: carries prior Human/AI messages into every new turn.
    Only the final user message and final answer are stored, not the tool traffic.
    """
    messages = [
        SystemMessage(content="""You are a support assistant.
Remember our conversation and use context from previous messages.
When the user asks follow-up questions, refer to what we discussed.""")
    ] + conversation_history + [HumanMessage(content=user_message)]

    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Final answer: save this turn to memory, then return it.
        if not response.tool_calls:
            conversation_history.append(HumanMessage(content=user_message))
            conversation_history.append(AIMessage(content=response.content))
            return response.content

        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            print(f"  🔧 {tool_name}")

            tool = tool_map.get(tool_name)

            if tool is None:
                messages.append(ToolMessage(
                    content=f"Error: Tool {tool_name} not found",
                    tool_call_id=tool_call["id"]
                ))
            else:
                # Tools take a single string; the model may send {} for no-input tools.
                args = tool_call['args']
                tool_input = next(iter(args.values()), "")
                messages.append(ToolMessage(
                    content=tool.func(tool_input),
                    tool_call_id=tool_call["id"]
                ))

    return "Maximum iterations reached without a final response"


# Test memory
memory_conversation = [
    "What database issues have we had?",
    "What was the ticket ID for that?",  # Should remember context
    "How was it resolved?"               # Should still remember
]

print("\nTesting conversation memory:")
for query in memory_conversation:
    print(f"\nUser: {query}")
    response = chat_with_memory(query)
    print(f"Assistant: {response[:150]}...")


# ============================================================================
# Bonus: Agent Evaluation
# ============================================================================
print("\n" + "=" * 80)
print("BONUS: Agent Evaluation")
print("=" * 80)

evaluation_cases = [
    {"query": "How do I fix login issues?", "expected_tool": "SearchSimilarTickets"},
    {"query": "Show ticket TICK-001", "expected_tool": "GetTicketByID"},
    {"query": "How many tickets?", "expected_tool": "GetTicketStatistics"},
    {"query": "All payment tickets", "expected_tool": "SearchByCategory"},
    {"query": "How many critical tickets are there?", "expected_tool": "SearchByPriority"},
]

correct = 0
total = len(evaluation_cases)

print("\nRunning evaluation...\n")
for test in evaluation_cases:
    result = run_agent(test['query'])
    passed = test['expected_tool'] in result['tools_used']
    if passed:
        correct += 1
    print(f"  {'✓' if passed else '✗'} '{test['query'][:30]}...' → {result['tools_used']}")
    print("-" * 70)

accuracy = correct / total * 100
print(f"\nTool Selection Accuracy: {correct}/{total} ({accuracy:.0f}%)")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("ALL SOLUTIONS COMPLETE!")
print("=" * 80)
print("""
Key Takeaways:
──────────────
1. Tool Descriptions Matter:
   - Clear, specific descriptions improve tool selection
   - Include keywords that match likely user queries
   - State when NOT to use a tool

2. Error Handling:
   - Validate inputs before processing
   - Return helpful error messages
   - Suggest corrections when possible

3. Multi-Step Reasoning:
   - Agents can chain multiple tools
   - Order of tool calls matters
   - Synthesis happens after all tools run

4. Conversation Memory:
   - Pass message history to maintain context
   - Enables follow-up questions
   - Be mindful of token limits

5. Evaluation:
   - Track tool selection accuracy
   - Measure iterations and latency
   - Use test sets for regression testing

When to Use Agentic RAG:
────────────────────────
✓ Complex multi-step queries
✓ Multiple data sources/actions
✓ Interactive conversations
✓ Flexible query patterns

When to Use Direct RAG:
───────────────────────
✓ Simple single-step retrieval
✓ Low latency requirements
✓ Predictable behavior needed
✓ Cost-sensitive applications
""")
