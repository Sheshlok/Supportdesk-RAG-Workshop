# -*- coding: utf-8 -*-

import json
import os
from langchain_core.tools import Tool
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()

class SupportTicketToolsEnhanced:
    """Enhanced tools with better error-handling and small-patch extensions."""

    def __init__(self, tickets_path: str = '../../data/synthetic_tickets.json'):
        with open(tickets_path, 'r', encoding='utf-8') as f:
            self.tickets = json.load(f)

        self.embeddings = OpenAIEmbeddings(
            model=os.getenv('OPENAI_EMBEDDINGS_MODEL', 'text-embedding-3-small')
        )
        self._setup_vectorstore()

    def _setup_vectorstore(self):
        """Index all tickets into a semantic vector store for similarity retrieval."""
        documents = []
        for ticket in self.tickets:
            content = f"""Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
Category: {ticket['category']}
Priority: {ticket['priority']}"""

            doc = Document(
                page_content=content,
                metadata={
                    'ticket_id': ticket['ticket_id'],
                    'title': ticket['title'],
                    'category': ticket['category'],
                    'priority': ticket['priority']
                }
            )

            documents.append(doc)

        # if the vector store already exists, load from the persistent directory
        if os.path.exists('./agent_solutions_vectorstore'):
            self.vector_store = Chroma(
            embedding_function=self.embeddings,
            collection_name='agent_solutions',
            persist_directory='./agent_solutions_vectorstore'
            )
        else:
            # Dedicated collection name avoids collision with other module demos
            self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name='agent_solutions',
            persist_directory='./agent_solutions_vectorstore'
            )

    def search_similar_tickets(self, query: str) -> str:
        """Search for similar tickets using semantic similarity."""
        if not query or not query.strip():
            return "Error: Please provide a search query."

        # Small-patch Exercise 4 behaviour
        # If the query explicitly asks for critical issues, apply a priority filter
        # using the existing search tool instead of adding a brand-new tool.
        if 'critical' in query.lower():
            results = self.vector_store.similarity_search(
                query=query,
                k=3,
                filter={'priority': 'Critical'}
            )
        else:
            # Semantic retrieval is best for "how to fix" and fuzzy symptom queries.
            results = self.vector_store.similarity_search(
                query=query,
                k=3
            )

        if not results:
            return "No similar tickets found."

        output = "Found similar tickets:\n\n"

        for i, doc in enumerate(results, start=1):
            output += f"--- Ticket {i} ---\n{doc.page_content}\n\n"

        return output

    def get_ticket_by_id(self, ticket_id: str) -> str:
        """Retrieve a specific ticket by ID with error handling."""

        if not ticket_id or not ticket_id.strip():
            return "Error: Please provide a ticket ID (e.g., TICK-001)"

        ticket_id = ticket_id.upper().strip()

        # Validate expected identifier shape before scanning dataset.
        if not ticket_id.startswith("TICK-"):
            return f"Error: Invalid format. Ticket IDs look like TICK-001, TICK-002, etc. You provided: '{ticket_id}'"

        for ticket in self.tickets:
            if ticket['ticket_id'] == ticket_id:
                return f"""Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
Category: {ticket['category']}
Priority: {ticket['priority']}
Created: {ticket['created_date']}
Resolved: {ticket['resolved_date']}"""

        return f"Ticket '{ticket_id}' not found. Check the ID and try again."

    def search_by_category(self, category: str) -> str:
        """Find all tickets in a specific category."""
        if not category or not category.strip():
            available = list(set(t['category'] for t in self.tickets))
            return f"Error: Please provide a category. Available categories are: {', '.join(available)}"

        category = category.strip()

        matching_tickets = [t for t in self.tickets if t['category'].lower() == category.lower()]

        if not matching_tickets:
            available = list(set(t['category'] for t in self.tickets))
            return f"No tickets found in category '{category}'. Available categories are: {', '.join(available)}"

        output = f"Found {len(matching_tickets)} tickets in '{category}':\n\n"
        for ticket in matching_tickets:
            output += f"• [{ticket['ticket_id']}] {ticket['title']} (Priority: {ticket['priority']})\n"

        return output

    # Optional reference implementation
    def search_by_priority(self, priority: str) -> str:
        """Find all tickets with a specific priority."""
        if not priority or not priority.strip():
            available = list(set(t['priority'] for t in self.tickets))
            return f"Error: Please provide a priority. Available priorities are: {', '.join(available)}"

        priority = priority.strip()

        matching_tickets = [t for t in self.tickets if t['priority'].lower() == priority.lower()]

        if not matching_tickets:
            available = list(set(t['priority'] for t in self.tickets))
            return f"No tickets found with priority '{priority}'. Available priorities are: {', '.join(available)}"

        output = f"Found {len(matching_tickets)} tickets with priority '{priority}':\n\n"
        for ticket in matching_tickets:
            output += f"• [{ticket['ticket_id']}] {ticket['title']} (Category: {ticket['category']})\n"

        return output

    def get_ticket_statistics(self, input: str = "") -> str:
        """Get statistics about ticket database"""

        categories = {}
        priorities = {}

        for ticket in self.tickets:
            categories[ticket['category']] = categories.get(ticket['category'], 0) + 1
            priorities[ticket['priority']] = priorities.get(ticket['priority'], 0) + 1

        output = "Ticket Database Statistics:\n"
        output += f"Total tickets: {len(self.tickets)}\n\n"

        output += "By Category:\n"
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            output += f"  • {category}: {count}\n"

        output += "\nBy Priority:\n"
        for priority, count in sorted(priorities.items(), key=lambda x: x[1], reverse=True):
            output += f"  • {priority}: {count}\n"

        return output

    def get_tools(self):
        return [
            Tool(
                name="SearchSimilarTickets",
                func=self.search_similar_tickets,
                description="""Search for similar support tickets using semantic similarity.
Use this for troubleshooting questions like "how to fix X", "why is X happening", 
"users experiencing X", or finding similar past issues.
Input: A clear description of the problem or question.
This is the PRIMARY tool for "how to fix" questions."""
            ),
            Tool(
                name="GetTicketByID",
                func=self.get_ticket_by_id,
                description="""Retrieve details of a specific ticket by its exact ID.
Use when user mentions a specific ticket number like TICK-001, TICK-005, etc.
Input: A ticket ID (e.g., 'TICK-001').
Do NOT use this for searching - only for getting known ticket details."""
            ),
            Tool(
                name="SearchByCategory",
                func=self.search_by_category,
                description="""Find ALL tickets in a specific category.
Use when user asks "what X issues have we seen" or "show me all X tickets".
Categories include: Authentication, Database, Payment, Performance, Email, Mobile.
Input: Category name (e.g., 'Database', 'Authentication')."""
            ),
            Tool(
                name="SearchByPriority",
                func=self.search_by_priority,
                description="""Find all tickets with a specific priority level.
Input should be: Critical, High, Medium, or Low.
Use when user asks about urgent/important issues or priority levels."""
            ),
            Tool(
                name="GetTicketStatistics",
                func=self.get_ticket_statistics,
                description="""Get statistics and overview of the ticket database.
Use when user asks "how many tickets", "give me an overview", or "statistics".
No input needed - provides total count, category breakdown, and priority breakdown."""
            )
        ]
       

