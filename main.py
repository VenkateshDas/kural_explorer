import streamlit as st
import os
import json
import random
from dotenv import load_dotenv
import traceback
import logging
import time
import sys

from agent import (
    get_kural_agent, 
    fetch_kurals, 
    chat, 
    setup_agno_kural_knowledge, 
    create_text_files_for_kurals, 
    batch_create_text_files, 
    batch_process_knowledge_base
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("kural_app.log")
    ]
)
logger = logging.getLogger("kural_explorer")

# Log startup information
logger.info("Starting Kural Explorer application")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")

# Page configuration and global styling
st.set_page_config(page_title="Kural Explorer", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    '''
    <style>
    /* Main container padding */
    .block-container {padding:2rem 3rem;}
    /* Sidebar background */
    .sidebar .sidebar-content {background-color:#1e1e1e; color:#fff;}
    /* Button customization */
    .stButton>button {background-color:#007acc; color: white; border-radius:8px; padding:0.5rem 1rem;}
    /* Kural card styling */
    .kural-card {background-color:#1f1f1f; padding:1rem; border-radius:10px; margin-bottom:2rem; box-shadow:0 4px 6px rgba(0,0,0,0.3);}
    .kural-card h3 {color:#fff; margin-bottom:0.5rem;}
    .kural-tamil {color:#ffcc00; font-family:'Latha', serif; white-space:pre-wrap;}
    .kural-english {color:#aad4ff; white-space:pre-wrap;}
    .kural-explanation {background-color:#2b2b2b; padding:0.75rem; border-left:4px solid #007acc; color:#ddd;}
    </style>
    ''', unsafe_allow_html=True
)

# Add CSS for a fixed-height scrollable chat window
st.markdown(
    '''
    <style>
    .chat-window { max-height: 400px; overflow-y: auto; }
    </style>
    ''', unsafe_allow_html=True
)

# Load environment variables from .env file
load_dotenv()
logger.info("Environment variables loaded")

# Check for API key early
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY is not set. Please add it to your .env file or environment.")
    st.error("OPENAI_API_KEY is not set. Please add it to your .env file or environment.")
    st.stop()
else:
    logger.info("OPENAI_API_KEY found in environment")

@st.cache_data
def load_kural_data(filepath="data/all_kural.json"):
    """Load and parse the Kural JSON file into a list of dicts with keys 'tamil' and 'english'."""
    logger.info(f"Loading kural data from {filepath}")
    start_time = time.time()
    parsed = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            logger.info(f"Reading JSON file: {filepath}")
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load Kural data: {e}")
        st.error(f"Failed to load Kural data: {e}")
        return []
    # Determine list of entries
    entries = []
    if isinstance(data, dict) and 'kurals' in data:
        entries = data['kurals']
        logger.info(f"Found {len(entries)} kurals in 'kurals' key")
    elif isinstance(data, list):
        entries = data
        logger.info(f"Found {len(entries)} kurals in list format")
    else:
        logger.error("Unexpected JSON structure in all_kural.json; expected list or dict with 'kurals'.")
        st.error("Unexpected JSON structure in all_kural.json; expected list or dict with 'kurals'.")
        return []
    # Parse each entry
    success_count = 0
    for entry in entries:
        text = entry if isinstance(entry, str) else str(entry)
        lines = text.splitlines()
        tamil = None
        english = None
        # Find couplet start and capture multiline Tamil
        start_idx = None
        for idx, raw in enumerate(lines):
            line = raw.strip()
            if line.startswith("குறள் -"):
                start_idx = idx
                break
        if start_idx is not None:
            tamil_lines = []
            for j in range(start_idx, len(lines)):
                ln = lines[j].strip()
                if not ln:
                    break
                if j == start_idx:
                    ln = ln.split("குறள் -", 1)[1].strip()
                tamil_lines.append(ln)
            tamil = "\n".join(tamil_lines)
        # Extract English translation (single line or until blank)
        for idx, raw in enumerate(lines):
            line = raw.strip()
            if line.lower().startswith("translation:"):
                # capture this line and any continuation lines until blank
                eng_lines = []
                for k in range(idx, len(lines)):
                    ln2 = lines[k].strip()
                    if not ln2:
                        break
                    if k == idx:
                        ln2 = ln2.split("translation:", 1)[1].strip().strip('"\'')
                    eng_lines.append(ln2)
                english = " ".join(eng_lines)
                break
        if tamil and english:
            parsed.append({'tamil': tamil, 'english': english})
            success_count += 1
    
    duration = time.time() - start_time
    logger.info(f"Successfully parsed {success_count}/{len(entries)} kurals in {duration:.2f} seconds")
    
    if not parsed:
        logger.error("No Kurals parsed; check your JSON format and parsing logic.")
        st.error("No Kurals parsed; check your JSON format and parsing logic.")
    return parsed

# Load data at the start
logger.info("Loading kural data from cache or disk")
kural_data_list = load_kural_data()
logger.info(f"Loaded {len(kural_data_list)} kurals")

# Updated: Set up Agno TextKnowledgeBase for the agent with PgVector
@st.cache_resource
def get_knowledge_base(
    kural_data, 
    text_dir="kural_texts", 
    db_url="postgresql+psycopg://ai:ai@localhost:5532/ai", 
    table_name="agno_kurals", 
    embedding_model="text-embedding-3-small",
    batch_processing=True,
    batch_size=50
):
    """Initialize and cache the Agno TextKnowledgeBase with PgVector"""
    logger.info(f"Setting up Agno knowledge base with {len(kural_data) if kural_data else 0} kurals")
    logger.info(f"Knowledge base parameters: text_dir={text_dir}, db_url={db_url}, table_name={table_name}")
    logger.info(f"Batch processing: {batch_processing}, batch_size: {batch_size}")
    
    if not kural_data:
        logger.warning("No kural data provided for knowledge base")
        return None
    
    # First, ensure we have text files for the kurals
    if not os.path.exists(text_dir) or not os.listdir(text_dir):
        logger.info(f"Text directory {text_dir} doesn't exist or is empty. Creating text files...")
        try:
            if batch_processing:
                logger.info(f"Using batch processing to create text files with batch_size={batch_size}")
                text_dir = batch_create_text_files(kural_data, output_dir=text_dir, batch_size=batch_size)
            else:
                text_dir = create_text_files_for_kurals(kural_data, output_dir=text_dir)
            logger.info(f"Created text files in {text_dir}")
        except Exception as e:
            logger.error(f"Error creating text files: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Error creating text files: {str(e)}")
            return None
    
    # Now set up the Agno knowledge base with PgVector
    start_time = time.time()
    try:
        kb = setup_agno_kural_knowledge(
            text_files_dir=text_dir,
            db_url=db_url,
            table_name=table_name,
            embedding_model=embedding_model
        )
        
        # Load the knowledge base, using batch processing for efficiency if enabled
        if batch_processing:
            logger.info(f"Using batch processing to load knowledge base with batch_size={batch_size}")
            # Use our custom batch processing function
            batch_process_knowledge_base(kb, batch_size=batch_size, recreate=True)
            logger.info("Knowledge base loaded successfully via batch processing")
        else:
            # Use standard loading method
            logger.info("Loading knowledge base into PgVector database (standard method)...")
            kb.load(recreate=True)  # Set to True to recreate the vector DB
            logger.info("Knowledge base loaded successfully via standard method")
        
        duration = time.time() - start_time
        logger.info(f"Knowledge base setup completed in {duration:.2f} seconds")
        return kb
    except Exception as e:
        logger.error(f"Error setting up knowledge base: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error setting up knowledge base: {str(e)}")
        return None

# Updated: Settings in sidebar for PgVector
st.sidebar.header("Knowledge Base Settings")
db_url = st.sidebar.text_input("PostgreSQL Connection String", "postgresql+psycopg://ai:ai@localhost:5532/ai")
text_dir = st.sidebar.text_input("Text Files Directory", "kural_texts") 
table_name = st.sidebar.text_input("Table Name", "agno_kurals")
embedding_model = st.sidebar.selectbox("Embedding Model", 
                                       ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
                                       index=0)

# Add batch processing settings
st.sidebar.subheader("Batch Processing")
use_batch_processing = st.sidebar.checkbox("Enable Batch Processing", value=True, help="Process documents in batches for better performance with large collections")
batch_size = st.sidebar.slider("Batch Size", min_value=10, max_value=100, value=50, help="Number of documents to process in each batch")

# Add information about batch processing
if use_batch_processing:
    st.sidebar.info("""
    🚀 **Batch Processing Enabled**
    
    Processing large document collections in batches:
    - Improves memory usage
    - Provides progress tracking
    - More reliable for 500+ documents
    - Better error handling
    """)

# Add information about PgVector
st.sidebar.info("""
⚠️ This app requires a PostgreSQL database with pgvector extension.
Make sure the database is running at the connection string above.

You can run it locally with Docker:
```
docker run -d -p 5532:5432 \\
  -e POSTGRES_DB=ai \\
  -e POSTGRES_USER=ai \\
  -e POSTGRES_PASSWORD=ai \\
  agnohq/pgvector:16
```
""")

# Simplified UI for text files generation - with batch option
if st.sidebar.button("Generate Text Files"):
    with st.spinner('Creating text files for Kurals...'):
        try:
            logger.info(f"Generating text files in directory: {text_dir}")
            start_time = time.time()
            
            if use_batch_processing:
                logger.info(f"Using batch processing with batch_size={batch_size}")
                text_dir = batch_create_text_files(kural_data_list, output_dir=text_dir, batch_size=batch_size)
            else:
                text_dir = create_text_files_for_kurals(kural_data_list, output_dir=text_dir)
                
            duration = time.time() - start_time
            logger.info(f"Text files created in {text_dir} in {duration:.2f} seconds")
            st.sidebar.success(f"Text files created in {text_dir}")
        except Exception as e:
            logger.error(f"Error creating text files: {str(e)}")
            logger.error(traceback.format_exc())
            st.sidebar.error(f"Error creating text files: {str(e)}")

# Check if PostgreSQL is available before attempting to initialize knowledge base
try:
    import sqlalchemy
    from sqlalchemy import text
    
    # Test database connection
    engine = sqlalchemy.create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        row = result.fetchone()
        if row and row[0] == 1:
            logger.info("Successfully connected to PostgreSQL database")
            st.sidebar.success("✅ Database connection successful")
            
            # Initialize the knowledge base with PgVector - with batch processing option
            with st.spinner("Loading knowledge base..."):
                logger.info("Initializing Agno knowledge base with PgVector")
                logger.info(f"Batch processing enabled: {use_batch_processing}, batch size: {batch_size}")
                
                kural_knowledge = get_knowledge_base(
                    kural_data_list,
                    text_dir=text_dir,
                    db_url=db_url,
                    table_name=table_name,
                    embedding_model=embedding_model,
                    batch_processing=use_batch_processing,
                    batch_size=batch_size
                )
                if kural_knowledge:
                    logger.info("Knowledge base successfully initialized")
                else:
                    logger.warning("Knowledge base initialization failed or returned None")
        else:
            logger.error("Database connection test failed")
            st.sidebar.error("❌ Database connection failed: Test query returned unexpected result")
            kural_knowledge = None
            
except Exception as e:
    logger.error(f"Error connecting to database: {str(e)}")
    logger.error(traceback.format_exc())
    st.sidebar.error(f"❌ Database connection failed: {str(e)}")
    kural_knowledge = None

## Model selection and agent initialization
model_options = ['gpt-4.1-2025-04-14', 'gpt-4.1-mini-2025-04-14', 'gpt-4o', 'gpt-3.5-turbo']
model_choice = st.sidebar.selectbox('Model Selection', model_options, index=0)

# Initialize the agent based on knowledge base availability
if 'model_choice' not in st.session_state or st.session_state.model_choice != model_choice:
    logger.info(f"Initializing agent with model: {model_choice}")
    st.session_state.model_choice = model_choice
    # Pass the knowledge to the agent
    st.session_state.agent = get_kural_agent(model_choice, knowledge=kural_knowledge)
    logger.info("Agent initialized successfully")

if 'kural_data' not in st.session_state:
    st.session_state.kural_data = kural_data_list
    logger.info(f"Stored {len(kural_data_list)} kurals in session state")

if 'kurals_to_display' not in st.session_state:
    st.session_state.kurals_to_display = []
    logger.info("Initialized empty kurals_to_display in session state")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    logger.info("Initialized empty chat_history in session state")

# Sidebar controls
st.sidebar.header('Controls')
num_kurals = st.sidebar.slider('Number of Kurals', 1, 10, 1)

# Note: With Agno setup, RAG is handled by the agent automatically
# So we don't need a RAG toggle anymore, it's enabled whenever knowledge is provided
has_knowledge = kural_knowledge is not None
if has_knowledge:
    st.sidebar.success("Agno Knowledge Base is active - RAG is enabled with PgVector")
else:
    st.sidebar.warning("Knowledge Base is not available - Only basic chat is enabled")

if st.sidebar.button('Fetch Kurals'):
    if st.session_state.kural_data:
        with st.spinner('Generating explanations…'):
            try:
                logger.info(f"Fetching {num_kurals} random kurals")
                start_time = time.time()
                
                sel = random.sample(
                    st.session_state.kural_data,
                    min(num_kurals, len(st.session_state.kural_data))
                )
                logger.info(f"Selected {len(sel)} random kurals")
                
                st.session_state.kurals_to_display = fetch_kurals(
                    st.session_state.agent,
                    sel
                )
                
                duration = time.time() - start_time
                logger.info(f"Generated explanations for {len(st.session_state.kurals_to_display)} kurals in {duration:.2f} seconds")
            except Exception as e:
                logger.error(f"Error fetching kurals: {str(e)}")
                logger.error(traceback.format_exc())
                st.error(f"Error fetching kurals: {str(e)}")
    else:
        logger.warning("Kural data not available; cannot fetch")
        st.sidebar.warning("Kural data not available; cannot fetch.")

# --- Layout: main explorer and chat columns ---
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by Venkatesh")
explorer_col, chat_col = st.columns([2, 1], gap="large")

# Column: Kural Explorer (Main)
with explorer_col:
    st.title('Kural Explorer')
    
    # Show stats about batch processing
    if has_knowledge and use_batch_processing:
        st.info(f"""
        📊 **Knowledge Base Stats**:
        - Documents processed in batches of {batch_size}
        - Total documents: {len(kural_data_list)} Kurals
        - Vector Database: PostgreSQL with pgvector
        - Table: {table_name}
        """)
    
    if not st.session_state.kurals_to_display:
        st.info('Select number of Kurals and click "Fetch Kurals" above.')
    else:
        logger.info(f"Displaying {len(st.session_state.kurals_to_display)} kurals in UI")
        for idx, kural_info in enumerate(st.session_state.kurals_to_display, 1):
            # Render each Kural in a styled card
            card_html = f"""
            <div class="kural-card">
              <h3>{idx}. Thirukural</h3>
              <div class="kural-tamil">{kural_info['tamil']}</div>
              <div class="kural-english"><strong>Translation:</strong> {kural_info['english']}</div>
              <div class="kural-explanation"><strong>Explanation:</strong><br>{kural_info['explanation']}</div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

# Column: Chat
with chat_col:
    st.header('Chat')
    
    # Show Agno RAG status
    if has_knowledge:
        st.success("Agno RAG is active with PgVector: Responses are enhanced with relevant Thirukurals")
    else:
        st.info("Knowledge base not available: Standard responses without kural context")
    
    # Wrap chat history in a fixed-height scrollable div
    st.markdown('<div class="chat-window">', unsafe_allow_html=True)
    # Display chat history
    for speaker, text in st.session_state.chat_history:
        with st.chat_message(speaker.lower()):
            st.write(text)
    # Close scrollable chat window
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Type your message...")
    if user_input:
        # Display user message
        with st.chat_message("you"):
            st.write(user_input)
        
        # Add to history
        st.session_state.chat_history.append(('You', user_input))
        logger.info(f"User input: '{user_input[:50]}...' (if longer)")
        
        try:
            logger.info("Generating response")
            start_time = time.time()
            
            # With Agno, the agent now handles RAG internally when knowledge is provided
            reply = chat(st.session_state.agent, user_input)
            
            duration = time.time() - start_time
            logger.info(f"Generated response in {duration:.2f} seconds")
            
            # The response is now a string, not an object with content attribute
            logger.info(f"Response: '{reply[:50]}...' (if longer)")
            
            with st.chat_message("bot"):
                st.write(reply)
            
            # Add to history
            st.session_state.chat_history.append(('Bot', reply))
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Error generating response: {str(e)}")
            st.session_state.chat_history.append(('Bot', f"I encountered an error: {str(e)}"))
        
        st.rerun() 