from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.vectordb.search import SearchType
from agno.embedder.openai import OpenAIEmbedder
from textwrap import dedent
import random
import os
from typing import List, Dict, Any, Optional
import traceback
from tqdm import tqdm
import time
import concurrent.futures
import shutil


def create_text_files_for_kurals(kural_data: List[Dict[str, str]],
                                output_dir: str = "kural_texts") -> str:
    """
    Create text files for each kural, which Agno TextKnowledgeBase can load.

    Args:
        kural_data: List of dictionaries with 'tamil' and 'english' keys
        output_dir: Directory to store the text files

    Returns:
        Path to the directory with text files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Creating text files for {len(kural_data)} kurals in {output_dir}...")

    # Create a file for each kural
    for i, kural in enumerate(kural_data):
        file_path = os.path.join(output_dir, f"kural_{i:04d}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            # Combine Tamil and English, clearly labeling them
            f.write(f"Tamil:\n{kural['tamil']}\n\nEnglish:\n{kural['english']}")

    print(f"Finished creating {len(kural_data)} text files in {output_dir}")
    return output_dir


def batch_create_text_files(kural_data: List[Dict[str, str]], 
                           output_dir: str = "kural_texts", 
                           batch_size: int = 50) -> str:
    """
    Create text files for kurals in batches for more efficient processing.
    
    Args:
        kural_data: List of dictionaries with 'tamil' and 'english' keys
        output_dir: Directory to store the text files
        batch_size: Number of files to process in each batch
        
    Returns:
        Path to the directory with text files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    total_files = len(kural_data)
    print(f"Creating {total_files} text files in batches of {batch_size}...")
    
    # Process in batches with progress bar
    with tqdm(total=total_files, desc="Creating text files") as pbar:
        for i in range(0, total_files, batch_size):
            end_idx = min(i + batch_size, total_files)
            batch = kural_data[i:end_idx]
            
            # Process this batch
            for j, kural in enumerate(batch):
                file_idx = i + j
                file_path = os.path.join(output_dir, f"kural_{file_idx:04d}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    # Combine Tamil and English, clearly labeling them
                    f.write(f"Tamil:\n{kural['tamil']}\n\nEnglish:\n{kural['english']}")
            
            pbar.update(len(batch))
    
    print(f"Finished creating {total_files} text files in {output_dir}")
    return output_dir


def setup_agno_kural_knowledge(
    text_files_dir: str,
    db_url: str = "postgresql+psycopg://ai:ai@localhost:5532/ai",
    table_name: str = "agno_kurals",
    embedding_model: str = "text-embedding-3-small"
) -> TextKnowledgeBase:
    """
    Sets up Agno TextKnowledgeBase for Thirukurals using PgVector.

    Args:
        text_files_dir: Directory containing the kural .txt files.
        db_url: PostgreSQL connection string.
        table_name: Name for the PgVector table.
        embedding_model: OpenAI embedding model ID.

    Returns:
        Configured TextKnowledgeBase instance (not loaded yet).
    """
    print(f"Setting up Agno TextKnowledgeBase from path: {text_files_dir}")
    print(f"Using PgVector with connection: {db_url}, table: {table_name}")
    print(f"Embedding model: {embedding_model}")

    try:
        # Configure the vector database (PgVector) with an embedder
        vector_db = PgVector(
            table_name=table_name,
            db_url=db_url,
            # Using hybrid search might be beneficial, depends on query types
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(id=embedding_model)
        )
        print("PgVector database configured.")

        # Configure the TextKnowledgeBase
        knowledge_base = TextKnowledgeBase(
            path=text_files_dir,  # Directory with .txt files
            vector_db=vector_db
        )
        print("TextKnowledgeBase configured.")

        return knowledge_base

    except Exception as e:
        print(f"ERROR: Failed to set up Agno knowledge base: {str(e)}")
        traceback.print_exc()
        raise Exception(f"Failed to set up Agno knowledge base: {str(e)}")


def batch_process_knowledge_base(
    knowledge_base: TextKnowledgeBase,
    batch_size: int = 50,
    recreate: bool = True,
    max_workers: int = 4
) -> None:
    """
    Process and load the knowledge base in batches to handle large document collections.
    This is more efficient for loading ~500 documents into PgVector.
    
    Args:
        knowledge_base: Configured TextKnowledgeBase instance
        batch_size: Number of documents to process in each batch
        recreate: Whether to recreate the vector database
        max_workers: Maximum number of worker threads for concurrent processing
    """
    print(f"Starting batch processing of knowledge base with batch_size={batch_size}")
    
    # First, collect all file paths
    if isinstance(knowledge_base.path, str):
        base_path = knowledge_base.path
    else:
        base_path = str(knowledge_base.path)
    
    all_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.txt'):
                all_files.append(os.path.join(root, file))
    
    total_files = len(all_files)
    print(f"Found {total_files} documents to process")
    
    if total_files == 0:
        print("No documents found to process")
        return
    
    # Process in batches using direct API calls
    start_time = time.time()
    
    # For the first batch, use recreate=True if specified
    processed_count = 0
    
    with tqdm(total=total_files, desc="Processing documents") as pbar:
        # First load with recreate if needed
        if recreate:
            print(f"\nInitializing database table with recreate=True")
            try:
                knowledge_base.load(recreate=True)
                print("Database table initialized successfully")
            except Exception as e:
                print(f"Error initializing database table: {str(e)}")
                traceback.print_exc()
        
        # Process each batch using load with upsert
        batch_count = (total_files + batch_size - 1) // batch_size
        for i in range(0, total_files, batch_size):
            batch_start = time.time()
            end_idx = min(i + batch_size, total_files)
            current_batch = all_files[i:end_idx]
            batch_size_actual = len(current_batch)
            
            print(f"\nProcessing batch {i//batch_size + 1}/{batch_count}: {batch_size_actual} documents")
            
            # We can't pass files_to_load directly to load(), so we need a different approach
            # 1. For TextKnowledgeBase, we'll temporarily modify its path to point to a specific batch
            try:
                # Keep track of the original path
                original_path = knowledge_base.path
                
                # Create a temporary directory for this batch
                batch_dir = f"tmp_batch_{i//batch_size + 1}"
                os.makedirs(batch_dir, exist_ok=True)
                
                # Copy the current batch files to the temporary directory
                for file_path in current_batch:
                    file_name = os.path.basename(file_path)
                    shutil.copy(file_path, os.path.join(batch_dir, file_name))
                
                # Temporarily set the knowledge base path to the batch directory
                knowledge_base.path = batch_dir
                
                # Load just this batch with upsert=True
                print(f"Loading batch with upsert=True")
                knowledge_base.load(upsert=True)
                
                processed_count += batch_size_actual
                batch_duration = time.time() - batch_start
                print(f"Batch processed in {batch_duration:.2f} seconds ({batch_size_actual/batch_duration:.2f} docs/sec)")
                
                # Restore the original path
                knowledge_base.path = original_path
                
                # Clean up the temporary directory
                shutil.rmtree(batch_dir)
                
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                traceback.print_exc()
                
                # Make sure to restore the original path even if there's an error
                if 'original_path' in locals():
                    knowledge_base.path = original_path
                
                # Clean up the temporary directory if it exists
                if 'batch_dir' in locals() and os.path.exists(batch_dir):
                    shutil.rmtree(batch_dir)
            
            pbar.update(batch_size_actual)
    
    total_duration = time.time() - start_time
    print(f"\nKnowledge base batch processing complete!")
    print(f"Processed {processed_count}/{total_files} documents in {total_duration:.2f} seconds")
    print(f"Average processing speed: {processed_count/total_duration:.2f} documents/second")
    
    # Final database statistics
    try:
        # Try to get count using the knowledge base's methods
        # This will depend on the specific implementation
        print("Checking database statistics...")
        if hasattr(knowledge_base.vector_db, 'get_count'):
            count = knowledge_base.vector_db.get_count()
            print(f"Vector database contains {count} entries")
        else:
            # Try to use underlying SQLAlchemy connection
            try:
                import sqlalchemy
                from sqlalchemy import text
                if hasattr(knowledge_base.vector_db, 'engine'):
                    engine = knowledge_base.vector_db.engine
                    with engine.connect() as conn:
                        table_name = getattr(knowledge_base.vector_db, 'full_table_name', 
                                            getattr(knowledge_base.vector_db, 'table_name', 'unknown'))
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                        row = result.fetchone()
                        if row:
                            print(f"Vector database contains {row[0]} entries")
            except ImportError:
                print("SQLAlchemy not available for statistics")
            except Exception as e:
                print(f"Error getting database statistics via SQLAlchemy: {str(e)}")
    except Exception as e:
        print(f"Unable to retrieve vector database statistics: {str(e)}")
    
    print("Knowledge base is ready for use")


def get_kural_agent(
    model_id: str = "gpt-3.5-turbo",
    knowledge: Optional[TextKnowledgeBase] = None # Changed type hint
) -> Agent:
    """
    Create and return an Agno Agent specialized for exploring Thirukurals.

    Args:
        model_id: OpenAI model ID to use.
        knowledge: Optional Agno knowledge base instance.

    Returns:
        Agent instance.
    """
    print(f"Creating Kural Explorer agent with model: {model_id}")
    print(f"Knowledge base provided: {'Yes' if knowledge else 'No'}")

    # Simplified model compatibility check (Agno handles many models)
    if "gpt-4.1" in model_id:
        print(f"Note: Using model {model_id}. Consider 'gpt-4o' if compatibility issues arise.")
        # No automatic downgrade, let the user manage model choice

    agent = Agent(
        name="Kural Explorer",
        model=OpenAIChat(id=model_id),
        knowledge=knowledge, # Pass Agno knowledge base
        search_knowledge=knowledge is not None, # Enable search only if KB exists
        markdown=True,
        description=dedent("""\
You are a Tamil Thirukural expert. You provide original Tamil couplets, English translations, and modern explanations in a relatable style.
        """),
        instructions=dedent("""\
When asked to fetch N Kurals, produce them in sets of:
1. Tamil original
2. English translation
3. Relatable modern explanation in casual tone.

When questions are about Thirukurals, use your knowledge base to find relevant couplets and provide accurate information based on them. Use the search tool if the knowledge base seems insufficient or the question is broader.
When chatting about other topics, respond knowledgeably in a relatable style.
        """),
        show_tool_calls=True # Useful for debugging RAG
    )

    print(f"Agent initialized with model: {model_id}. Search enabled: {agent.search_knowledge}")
    return agent


def fetch_kurals(agent: Agent, *args) -> list[dict]:
    """
    Fetch Thirukurals explanations; supports two call signatures:
      - fetch_kurals(agent, kurals_list)
      - fetch_kurals(agent, n, kurals_list)

    Returns a list of {'tamil','english','explanation'} dicts.
    Assumes kurals_list contains dicts with 'tamil' and 'english' keys.
    """
    # Determine entries list
    entries = []
    if len(args) == 1 and isinstance(args[0], list):
        # Called as fetch_kurals(agent, kurals_list)
        entries = args[0]
        print(f"Fetching explanations for {len(entries)} provided kurals.")
    elif len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], list):
        # Called as fetch_kurals(agent, n, kurals_list)
        n, kurals_list = args
        num_to_sample = min(n, len(kurals_list))
        print(f"Randomly selecting {num_to_sample} kurals from {len(kurals_list)} to fetch explanations.")
        if num_to_sample > 0:
            entries = random.sample(kurals_list, num_to_sample)
        else:
            print("Warning: Cannot sample 0 or fewer kurals.")
            return []
    else:
        print("Error: Invalid arguments for fetch_kurals.")
        return []

    if not entries:
        print("No kural entries to process.")
        return []

    results = []
    print(f"Generating explanations for {len(entries)} kurals...")
    for i, entry in enumerate(entries):
        tamil_kural = entry.get('tamil', '[Tamil text not found]')
        english_translation = entry.get('english', '[English text not found]')

        # Check for missing text
        if tamil_kural == '[Tamil text not found]' or english_translation == '[English text not found]':
            print(f"Warning: Skipping kural {i+1} due to missing text.")
            explanation = "[Explanation skipped due to missing text]"
        else:
            prompt = dedent(f"""
Provide ONLY a relatable, modern-day explanation for the following Thirukural in a casual 'guy-to-guy' tone. Keep it concise and focus solely on the explanation.

Tamil: {tamil_kural}
English: {english_translation}

Explanation:""")
            try:
                print(f"  Processing kural {i+1}/{len(entries)}...")
                # Use standard string format for better compatibility
                response = agent.run(prompt)
                explanation = response.content.strip() if response and response.content else "[Error: Empty response from agent]"
            except Exception as e:
                print(f"Error getting explanation for kural {i+1}: {e}")
                explanation = f"[Error getting explanation: {e}]"

        results.append({
            'tamil': tamil_kural,
            'english': english_translation,
            'explanation': explanation,
        })
        # Optional: Add a small delay if hitting rate limits
        # time.sleep(0.1)

    print(f"Finished fetching explanations for {len(results)} kurals.")
    return results


def chat(agent: Agent, prompt: str):
    """Process user input and get a response from the agent (RAG handled internally)."""
    print(f"\nProcessing chat request: '{prompt[:100]}...'")
    print(f"Agent will use knowledge base for search: {agent.search_knowledge}")

    try:
        # Pass the prompt directly to the agent.
        # Agno's agent will automatically use its knowledge base if search_knowledge=True
        print("Sending prompt to agent...")
        response = agent.run(prompt)
        print("Received response from agent.")
        
        # Get the content safely based on response type
        # It could be a RunResponse object, dictionary, string, or other format
        if response is None:
            return "No response received"
            
        if isinstance(response, str):
            return response
            
        # For RunResponse or similar objects with content attribute
        if hasattr(response, 'content'):
            if response.content is None:
                return "Empty response content"
            return response.content
            
        # For dictionary-like responses
        if isinstance(response, dict) and 'content' in response:
            return response['content']
            
        # Last resort: convert to string
        return str(response)

    except Exception as e:
        print(f"ERROR in chat: {str(e)}")
        traceback.print_exc()
        return f"Error processing request: {str(e)}"

# Example usage with batch processing:
# 1. Load kural_data (e.g., from JSON)
# 2. Call batch_create_text_files(kural_data) -> text_dir
# 3. Call setup_agno_kural_knowledge(text_dir) -> knowledge_base
# 4. Call batch_process_knowledge_base(knowledge_base, batch_size=50)
# 5. Call get_kural_agent(knowledge=knowledge_base) -> agent
# 6. Call chat(agent, user_prompt) 