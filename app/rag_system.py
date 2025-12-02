import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False
    import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
from typing import List, Dict
import glob

# Custom embedding class using a simpler, more reliable approach
class SentenceTransformerEmbeddings(Embeddings):
    """Custom embedding class using SentenceTransformer directly with workaround for Windows cache issues"""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from pathlib import Path
        import os
        
        try:
            from sentence_transformers import SentenceTransformer
            
            print("=" * 60)
            print("Initializing Embeddings Model")
            print("=" * 60)
            
            # Check if we have a locally downloaded model
            local_model_dir = Path(".") / "embedding_model"
            full_name = f"sentence-transformers/{model_name}"
            
            # Try loading from local directory first
            if local_model_dir.exists() and (local_model_dir / "config.json").exists():
                print("Loading from local model directory...")
                try:
                    self.model = SentenceTransformer(str(local_model_dir))
                    print("✓ Model loaded from local directory!")
                    print("=" * 60)
                    return
                except Exception as e:
                    print(f"Local load failed: {e}, trying download...")
            
            # If local doesn't work, try downloading directly
            print("Downloading model (this may take 2-3 minutes)...")
            print("Please be patient...")
            
            # Use a simple approach - let SentenceTransformer handle it
            # But set environment to use a local cache
            project_cache = Path(".") / ".model_cache"
            project_cache.mkdir(exist_ok=True)
            
            # Set environment variables
            os.environ['HF_HOME'] = str(project_cache.absolute())
            os.environ['TRANSFORMERS_CACHE'] = str(project_cache.absolute())
            
            try:
                # Try loading with the full name
                self.model = SentenceTransformer(full_name)
                print("✓ Model loaded successfully!")
            except Exception as e1:
                print(f"First attempt failed, trying alternative...")
                # Try with just the model name
                try:
                    self.model = SentenceTransformer(model_name)
                    print("✓ Model loaded successfully!")
                except Exception as e2:
                    # Last resort: try downloading manually
                    print("Attempting manual download...")
                    try:
                        from huggingface_hub import snapshot_download
                        # Download to local directory
                        local_dir = Path(".") / "embedding_model"
                        local_dir.mkdir(exist_ok=True)
                        
                        print("Downloading model files...")
                        snapshot_download(
                            repo_id=full_name,
                            local_dir=str(local_dir),
                            local_dir_use_symlinks=False
                        )
                        # Now load from local
                        self.model = SentenceTransformer(str(local_dir))
                        print("✓ Model downloaded and loaded successfully!")
                    except Exception as e3:
                        raise RuntimeError(
                            f"Could not load embedding model after multiple attempts.\n\n"
                            f"Please run this command first:\n"
                            f"python download_model.py\n\n"
                            f"This will download the model properly.\n\n"
                            f"Error: {str(e3)[:300]}"
                        )
            
            print("=" * 60)
            print("Embeddings ready!")
            print("=" * 60)
                    
        except ImportError as ie:
            raise RuntimeError(
                f"Required packages not installed.\n\n"
                f"Please run:\n"
                f"pip install sentence-transformers huggingface-hub\n\n"
                f"Error: {ie}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Embedding initialization failed.\n\n"
                f"Error: {str(e)}\n\n"
                f"Solution: Run 'python download_model.py' first to download the model properly."
            )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        return self.model.encode(texts, show_progress_bar=False).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

class RAGSystem:
    def __init__(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key
        # Initialize embeddings with proper error handling and cache management
        self.embeddings = self._initialize_embeddings()
        self.vectorstore = None
        self.llm = None
        self.prompts = {}
        self.documents = []
    
    def _initialize_embeddings(self):
        """Initialize embeddings using custom SentenceTransformer wrapper"""
        # Use our custom embedding class that bypasses HuggingFace cache issues
        return SentenceTransformerEmbeddings()
        
    def load_documents(self, data_dir: str = "."):
        """Load all PDF and TXT files from the directory"""
        self.documents = []
        
        # Load PDF files - EXCLUDE VCET UG Regulation R21.pdf and cse.pdf
        pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
        
        # Filter out files we don't want:
        # 1. VCET UG Regulation R21.pdf - not needed for syllabus questions
        # 2. cse.pdf - this is a placement file, not syllabus (causes confusion with 2.CSE2021.pdf)
        pdf_files = [f for f in pdf_files 
                    if "VCET UG Regulation R21" not in f 
                    and "regulation" not in f.lower()
                    and "cse.pdf" not in f.lower() 
                    and os.path.basename(f).lower() != "cse.pdf"]
        
        # Sort to load syllabus file FIRST for better processing
        syllabus_file = None
        other_pdfs = []
        for pdf_file in pdf_files:
            if "2.CSE2021" in pdf_file:
                syllabus_file = pdf_file
            else:
                other_pdfs.append(pdf_file)
        
        # Load syllabus file first
        if syllabus_file:
            pdf_files = [syllabus_file] + other_pdfs
        else:
            pdf_files = other_pdfs
        
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(pdf_file)
                docs = loader.load()
                # Add metadata about the source file
                for i, doc in enumerate(docs):
                    doc.metadata['source'] = os.path.basename(pdf_file)
                    doc.metadata['file_type'] = 'pdf'
                    doc.metadata['page'] = i  # Add page number
                    # Mark syllabus file for easier retrieval - ONLY 2.CSE2021.pdf
                    if "2.CSE2021" in pdf_file:
                        doc.metadata['is_syllabus'] = True
                        doc.metadata['is_curriculum'] = True  # Mark as curriculum source
                        # Try to extract semester info from text
                        text_lower = doc.page_content.lower()
                        for sem in ['semester i', 'semester ii', 'semester iii', 'semester iv', 
                                   'semester v', 'semester vi', 'semester vii', 'semester viii',
                                   'semester 1', 'semester 2', 'semester 3', 'semester 4',
                                   'semester 5', 'semester 6', 'semester 7', 'semester 8']:
                            if sem in text_lower:
                                doc.metadata['semester'] = sem
                                break
                self.documents.extend(docs)
                print(f"Loaded {len(docs)} pages from {pdf_file}")
            except Exception as e:
                print(f"Error loading {pdf_file}: {str(e)}")
        
        # Load TXT files
        txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
        for txt_file in txt_files:
            try:
                loader = TextLoader(txt_file, encoding='utf-8')
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = os.path.basename(txt_file)
                    doc.metadata['file_type'] = 'txt'
                self.documents.extend(docs)
                print(f"Loaded {len(docs)} pages from {txt_file}")
            except Exception as e:
                print(f"Error loading {txt_file}: {str(e)}")
        
        # Count unique files loaded
        unique_files = set()
        for doc in self.documents:
            source = doc.metadata.get('source', 'Unknown')
            unique_files.add(source)
        
        print(f"Total document objects (pages/chunks): {len(self.documents)}")
        print(f"Total unique files loaded: {len(unique_files)}")
        return len(self.documents)
    
    def create_vectorstore(self):
        """Create vector store from loaded documents"""
        if not self.documents:
            raise ValueError("No documents loaded. Please load documents first.")
        
        # Split documents into chunks - optimized for syllabus PDFs with tables
        # Use larger chunks to keep semester tables intact
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Large chunks to keep semester tables together
            chunk_overlap=400,  # Large overlap to ensure semester info isn't lost
            length_function=len,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]  # Better separators for PDF content
        )
        
        splits = text_splitter.split_documents(self.documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        print(f"Created vector store with {len(splits)} chunks")
        return self.vectorstore
    
    def is_academic_question(self, question: str) -> bool:
        """Determine if a question is related to academics, syllabus, or CSE department"""
        question_lower = question.lower()
        
        # STRONG academic indicators - these make it definitely academic
        strong_academic_indicators = [
            # Direct syllabus/course queries
            "subjects in", "papers in", "courses in", "syllabus of", "curriculum of",
            "subjects included", "papers included", "what subjects", "what papers",
            "list of subjects", "list of papers", "subjects for", "papers for",
            # Semester-specific queries
            "semester", "sem ", "1st semester", "2nd semester", "3rd semester", 
            "4th semester", "5th semester", "6th semester", "7th semester", "8th semester",
            "first semester", "second semester", "third semester", "fourth semester",
            "fifth semester", "sixth semester", "seventh semester", "eighth semester",
            # Department/institution specific
            "vcet", "velammal", "cse department", "department of",
            # Academic records
            "placement", "faculty", "infrastructure", "credit", "marks", "grade",
            "regulation", "program", "degree", "b.e", "bachelor"
        ]
        
        # Check for strong academic indicators first
        if any(indicator in question_lower for indicator in strong_academic_indicators):
            return True
        
        # General knowledge question patterns - these indicate general questions
        general_question_patterns = [
            "what is", "what are", "explain", "define", "meaning of", 
            "how does", "why", "difference between", "compare", 
            "tell me about", "example of", "examples of", "real time example",
            "real-time example", "give example", "give an example",
            "how to", "how can", "what does", "what do"
        ]
        
        # If question starts with general patterns and doesn't have strong academic context, it's general
        has_general_pattern = any(question_lower.startswith(pattern) or 
                                 f" {pattern} " in question_lower or 
                                 question_lower.endswith(f" {pattern}") 
                                 for pattern in general_question_patterns)
        
        # Check for weak academic keywords (might appear in general questions too)
        weak_academic_keywords = [
            "course", "subject", "paper", "syllabus", "curriculum"
        ]
        
        has_weak_academic = any(keyword in question_lower for keyword in weak_academic_keywords)
        
        # Decision logic:
        # 1. If it has strong academic indicators → academic
        # 2. If it's a general question pattern (what is, explain, example) → general
        # 3. If it has weak academic keywords but is asking "what is X" or "example of X" → general
        # 4. Otherwise → treat as academic to be safe
        
        # Priority 1: Check for strong academic indicators (definitely academic)
        if any(indicator in question_lower for indicator in strong_academic_indicators):
            return True
        
        # Priority 2: If it's a general knowledge question pattern, it's likely general
        if has_general_pattern:
            # Check if it's asking ABOUT the syllabus/course vs asking FOR general knowledge
            asking_about_syllabus = any(phrase in question_lower for phrase in [
                "in the syllabus", "in syllabus", "in course", "in the course",
                "taught in", "covered in", "in semester", "in curriculum",
                "what is taught", "what is covered", "what subjects", "what papers",
                "subjects in", "papers in", "courses in"
            ])
            
            # If asking about syllabus/course → academic
            if asking_about_syllabus:
                return True
            
            # If it's a general question (what is, explain, example) and NOT asking about syllabus → general
            # Examples: "what is turing machine", "real time example of turing machine", "explain state diagram"
            return False
        
        # Priority 3: If it has weak academic keywords but no general pattern, check context
        if has_weak_academic and not has_general_pattern:
            # If it mentions "course" or "subject" but in a way that suggests asking about syllabus
            if any(phrase in question_lower for phrase in [
                "course", "subject", "paper", "syllabus", "curriculum"
            ]):
                # Check if it's asking about the course/syllabus content
                if any(phrase in question_lower for phrase in [
                    "in", "of", "for", "about the"
                ]):
                    return True  # Likely asking about syllabus
        
        # Default: if unclear and has general pattern, treat as general (be helpful)
        if has_general_pattern:
            return False
        
        # Final default: if unclear, treat as academic to be safe
        return True

    def get_prompt_template(self, user_type: str = "student", is_general_question: bool = False):
        """Get prompt template based on user type and question type"""
        
        # General question template (for non-academic questions)
        general_template = """You are a helpful AI assistant with extensive knowledge. The user is asking a GENERAL KNOWLEDGE question that is NOT related to VCET CSE Department academics, syllabus, or curriculum.

🚨 CRITICAL INSTRUCTIONS:
1. This is a GENERAL KNOWLEDGE question - you MUST use your training knowledge and understanding to answer
2. DO NOT mention documents, PDFs, syllabus, or any college-specific information
3. DO NOT say "based on provided documents" or "the document doesn't have" - this is NOT a document-based question
4. Think and reason using your general knowledge to provide a comprehensive, accurate answer
5. Be helpful, informative, and detailed
6. Provide real-world examples, explanations, and practical information
7. Only say "I don't know" if you truly cannot answer with your general knowledge (which should be rare)

USER'S QUESTION: {question}

Answer this question using your general knowledge. Do NOT reference any documents or college-specific information. Provide a clear, comprehensive answer:"""
        
        # Define prompts based on user type
        prompts = {
            "student": """You are an expert assistant for VCET (Velammal College of Engineering and Technology) CSE (Computer Science and Engineering) Department ONLY.

CRITICAL RULES:
1. ONLY answer questions about the CSE (Computer Science and Engineering) Department
2. If asked about other departments, politely say: "I can only provide information about the CSE (Computer Science and Engineering) Department. Please ask questions related to CSE department."
3. ALWAYS use the provided context - it contains ALL the answers you need
4. The context includes these files:
   - 2.CSE2021.pdf: Complete syllabus from first year to final year (PRIMARY AND ONLY SOURCE for all syllabus/curriculum questions)
   - place17-21.pdf: Placement batch 2017-2021
   - place18-22.pdf: Placement batch 2018-2022
   - place19-23.pdf: Placement batch 2019-2023
   - place20-24.pdf: Placement batch 2020-2024
   - FACULTY.txt: Complete faculty details of CSE department
   - INFRASTRUCTURE.txt: Complete infrastructure details

🚨 CRITICAL: For ALL syllabus, curriculum, semester, subject, paper, or course questions:
   - Use ONLY information from 2.CSE2021.pdf
   - This file contains COMPLETE information for ALL 8 semesters (1st to 8th)
   - DO NOT use information from other files for syllabus questions
   - The file 2.CSE2021.pdf has ALL the semester information you need

YOUR TASK:
- Carefully READ and SEARCH through the provided context below
- Extract ALL relevant information from the context
- Answer the question using ONLY information from the context documents
- CRITICAL: For academic/syllabus/CSE department questions, you MUST ONLY use information from the provided documents
- DO NOT use general knowledge for academic questions - ONLY use document information
- If the information exists in the context, you MUST provide it - do NOT say "I don't know"
- Be thorough, detailed, and accurate

CONTEXT FROM COLLEGE DOCUMENTS (READ CAREFULLY AND SEARCH THROUGH IT):
{context}

STUDENT'S QUESTION: {question}

CRITICAL INSTRUCTIONS - READ CAREFULLY:

⚠️ CRITICAL: The file 2.CSE2021.pdf contains COMPLETE syllabus for ALL semesters (1st to 8th). 
The information you need IS in the context below. You MUST find it and extract it.
For syllabus/curriculum questions, ONLY use information from 2.CSE2021.pdf - IGNORE ALL OTHER FILES.
If you see sources like "cse.pdf", "place*.pdf", "FACULTY.txt", etc. in the context, DO NOT use them.
ONLY extract information from sources that contain "2.CSE2021" or "CSE2021" in the filename.

🚨 IMPORTANT: This is an ACADEMIC/SYLLABUS/CSE DEPARTMENT question.
- You MUST ONLY use information from the provided documents/context
- DO NOT use your general knowledge or training data
- DO NOT provide general explanations unless they are in the documents
- If the information is in the documents, extract and provide it
- If the information is NOT in the documents, say so clearly

1. QUESTION TYPE IDENTIFICATION:
   - "subject", "paper", "course", "syllabus", "subjects", "papers" → Extract SUBJECT NAMES ONLY
   - "credit", "credits", "credit distribution" → Provide CREDIT INFORMATION
   - NEVER mix them

2. FOR SEMESTER + SUBJECT QUESTIONS (CRITICAL):
   - The information EXISTS in the context - you MUST find it
   - Search for these EXACT patterns in the context:
     * For 1st: "SEMESTER-I", "SEMESTER I", "1st Semester", "Semester 1", "Sem I", "I"
     * For 2nd: "SEMESTER-II", "SEMESTER II", "2nd Semester", "Semester 2", "Sem II", "II"
     * For 3rd: "SEMESTER-III", "SEMESTER III", "3rd Semester", "Semester 3", "Sem III", "III"
     * For 4th: "SEMESTER-IV", "SEMESTER IV", "4th Semester", "Semester 4", "Sem IV", "IV"
     * For 5th: "SEMESTER-V", "SEMESTER V", "5th Semester", "Semester 5", "Sem V", "V"
     * For 6th: "SEMESTER-VI", "SEMESTER VI", "6th Semester", "Semester 6", "Sem VI", "VI"
     * For 7th: "SEMESTER-VII", "SEMESTER VII", "7th Semester", "Semester 7", "Sem VII", "VII"
     * For 8th: "SEMESTER-VIII", "SEMESTER VIII", "8th Semester", "Semester 8", "Sem VIII", "VIII"
   - CRITICAL: Find the table that starts with the EXACT semester you're asked about
   - Look for tables with these headers: "COURSE TITLE", "COURSE NAME", "SUBJECT"
   - Extract ONLY course titles from the "COURSE TITLE" column in THAT SPECIFIC semester's table
   - DO NOT mix subjects from different semesters
   - DO NOT add subjects that are not explicitly listed in that semester's table
   - DO NOT include laboratory courses unless they are listed separately in the COURSE TITLE column
   - The context contains this information - DO NOT say "I don't know" or "not available"

3. SEARCH STRATEGY (DO THIS):
   - Read the ENTIRE context below word by word
   - Look for semester numbers in ALL formats (III, 3rd, 3, three, etc.)
   - Find tables that start with semester information
   - Extract course titles from those tables
   - If you see ANY course titles in the context, list them

4. ANSWER FORMAT FOR SUBJECT NAMES:
   - Simple bullet list: • Course Title 1\\n• Course Title 2\\n• Course Title 3
   - NO credits, NO codes, NO categories
   - Just the course/subject names

5. IF YOU CANNOT FIND IT:
   - Only say "I cannot find" if you have searched EVERY part of the context
   - But remember: The syllabus file (2.CSE2021.pdf) contains ALL semester information
   - Try searching for variations: "Semester", "SEM", "Course", "Subject"

REMEMBER: The information IS in the context. Search more carefully!

Provide a comprehensive, accurate answer based on the context:""",
            
            "parent": """You are a professional assistant for parents of VCET CSE (Computer Science and Engineering) Department students ONLY.

CRITICAL RULES:
1. ONLY answer questions about the CSE (Computer Science and Engineering) Department
2. If asked about other departments, politely say: "I can only provide information about the CSE (Computer Science and Engineering) Department. Please ask questions related to CSE department."
3. ALWAYS use the provided context - it contains ALL the answers you need
4. The context includes these files:
   - place17-21.pdf: Placement batch 2017-2021
   - place18-22.pdf: Placement batch 2018-2022
   - place19-23.pdf: Placement batch 2019-2023
   - place20-24.pdf: Placement batch 2020-2024
   - FACULTY.txt: Complete faculty details of CSE department
   - INFRASTRUCTURE.txt: Complete infrastructure details
5. Do NOT provide academic curriculum or syllabus details (parents don't need this)

YOUR TASK:
- Carefully READ and SEARCH through the provided context below
- Extract ALL relevant information from the context
- Answer the question using ONLY information from the context
- If the information exists in the context, you MUST provide it - do NOT say "I don't know"
- Be thorough and informative

CONTEXT FROM COLLEGE DOCUMENTS (READ CAREFULLY AND SEARCH THROUGH IT):
{context}

PARENT'S QUESTION: {question}

INSTRUCTIONS:
1. Search through EVERY part of the context above for information related to the question
2. If you find the information, extract it and present it clearly
3. Format your answer professionally
4. Be confident - if the information is in the context, you MUST provide it
5. Only say "I don't know" if you truly cannot find ANY related information in the context above

Provide a clear, professional answer based on the context:""",
            
            "faculty": """You are a specialized assistant for VCET CSE (Computer Science and Engineering) Department faculty members ONLY.

CRITICAL RULES:
1. ONLY answer questions about the CSE (Computer Science and Engineering) Department
2. If asked about other departments, politely say: "I can only provide information about the CSE (Computer Science and Engineering) Department. Please ask questions related to CSE department."
3. ALWAYS use the provided context - it contains ALL the answers you need
4. The context includes these files:
   - 2.CSE2021.pdf: Complete syllabus from first year to final year
   - FACULTY.txt: Complete faculty details of CSE department
   - INFRASTRUCTURE.txt: Complete infrastructure details
   - CSE_Publication.pdf: Publications

YOUR TASK:
- Carefully READ and SEARCH through the provided context below
- Extract ALL relevant information from the context
- Answer the question using ONLY information from the context
- If the information exists in the context, you MUST provide it - do NOT say "I don't know"
- Be precise, detailed, and professional

CONTEXT FROM COLLEGE DOCUMENTS (READ CAREFULLY AND SEARCH THROUGH IT):
{context}

FACULTY QUESTION: {question}

INSTRUCTIONS:
1. Search through EVERY part of the context above for information related to the question
2. Pay special attention to tables, course lists, and semester information
3. For semester questions, look for "SEMESTER-V", "SEMESTER V", "Semester 5", "5th Semester", etc.
4. Extract ALL subject/course names you find - don't miss any
5. If you find the information, extract it and present it clearly
6. Format your answer professionally with proper structure
7. Be confident - if the information is in the context, you MUST provide it
8. Only say "I don't know" if you truly cannot find ANY related information in the context above

IMPORTANT: When asked about subjects/courses for a semester, list ALL the course titles you find in the context, even if they appear in tables or different formats.

🚨 CRITICAL: This is an ACADEMIC/CSE DEPARTMENT question.
- You MUST ONLY use information from the provided documents/context
- DO NOT use your general knowledge or training data
- DO NOT provide general explanations unless they are in the documents
- If the information is in the documents, extract and provide it
- If the information is NOT in the documents, say so clearly

Provide a detailed, professional answer based on the context:"""
        }
        
        # For general questions, return the string template directly (no PromptTemplate needed)
        if is_general_question:
            return general_template
        
        prompt_template = prompts.get(user_type, prompts["student"])
        
        return PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
    
    def query(self, question: str, user_type: str = "student"):
        """Query the RAG system"""
        if self.vectorstore is None:
            raise ValueError("Vector store not created. Please create vector store first.")
        
        # Determine if this is a general question (not academic/CSE related)
        is_general_question = not self.is_academic_question(question)
        
        # For general questions, allow LLM to use its knowledge without strict document constraints
        if is_general_question:
            # Initialize LLM if not already done (needed for general questions)
            if self.llm is None:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=self.gemini_api_key)
                    model_names = ['gemini-2.5-pro', 'gemini-1.5-pro', 'gemini-2.5-flash', 'gemini-pro']
                    model_initialized = False
                    for model_name in model_names:
                        try:
                            self.gemini_model = genai.GenerativeModel(model_name)
                            self.gemini_api_key_stored = self.gemini_api_key
                            self.use_direct_gemini = True
                            model_initialized = True
                            break
                        except:
                            continue
                    if not model_initialized:
                        raise Exception("Could not initialize any Gemini model")
                except Exception as e2:
                    raise RuntimeError(f"Could not initialize Gemini model: {str(e2)}")
            
            # Use general knowledge template
            prompt_template = self.get_prompt_template(user_type, is_general_question=True)
            # For general questions, the template is a string, not a PromptTemplate object
            if isinstance(prompt_template, str):
                formatted_prompt = prompt_template.format(question=question)
            else:
                formatted_prompt = prompt_template.format(context="", question=question)
            
            # Query LLM directly without document constraints
            try:
                if hasattr(self, 'use_direct_gemini') and self.use_direct_gemini:
                    response = self.gemini_model.generate_content(formatted_prompt)
                    answer = response.text
                elif GEMINI_AVAILABLE:
                    response = self.llm.invoke(formatted_prompt)
                    answer = response.content if hasattr(response, 'content') else str(response)
                else:
                    import google.generativeai as genai
                    genai.configure(api_key=self.gemini_api_key)
                    model = genai.GenerativeModel('gemini-2.5-pro')
                    response = model.generate_content(formatted_prompt)
                    answer = response.text
                
                return {
                    "answer": answer,
                    "sources": []  # No sources for general knowledge answers
                }
            except Exception as e:
                return {
                    "answer": f"Error generating response: {str(e)}",
                    "sources": []
                }
        
        # Filter question based on user type (for academic questions)
        if user_type == "parent":
            # Check if question is about academics
            academic_keywords = ["syllabus", "curriculum", "course", "subject", "semester", "exam", "study", "learn", "teach"]
            if any(keyword in question.lower() for keyword in academic_keywords):
                return {
                    "answer": "I'm sorry, but as a parent assistant, I don't provide academic curriculum or syllabus information. I can help you with placement records, department information (infrastructure, faculty, publications), or general college information. How can I assist you?",
                    "source_documents": []
                }
        
        # Initialize LLM if not already done
        if self.llm is None:
            # Use Google Gemini directly for better compatibility
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_api_key)
                # Use the latest Gemini 2.5 models for best quality
                # Try gemini-2.5-pro first for best accuracy, then fallback
                model_names = ['gemini-2.5-pro', 'gemini-1.5-pro', 'gemini-2.5-flash', 'gemini-pro']
                model_initialized = False
                for model_name in model_names:
                    try:
                        self.gemini_model = genai.GenerativeModel(model_name)
                        self.gemini_api_key_stored = self.gemini_api_key
                        self.use_direct_gemini = True
                        print(f"✓ Gemini model '{model_name}' initialized successfully!")
                        model_initialized = True
                        break
                    except Exception as model_error:
                        print(f"Tried {model_name}, error: {str(model_error)[:50]}")
                        continue
                if not model_initialized:
                    raise Exception("Could not initialize any Gemini model")
            except Exception as e2:
                raise RuntimeError(
                    f"Could not initialize Gemini model. Please check your API key.\n"
                    f"Error: {str(e2)}\n"
                    f"Make sure you have: pip install google-generativeai"
                )
        
        # Check if this is a syllabus/curriculum/semester question
        question_lower = question.lower()
        is_syllabus_question = any(word in question_lower for word in [
            "semester", "sem", "syllabus", "subject", "paper", "course", 
            "curriculum", "subjects", "papers", "courses"
        ])
        
        # For syllabus questions, ONLY use 2.CSE2021.pdf - create separate retriever
        if is_syllabus_question:
            # Get ONLY syllabus documents
            syllabus_docs = [doc for doc in self.documents 
                          if doc.metadata.get('is_syllabus') or 
                          '2.CSE2021' in doc.metadata.get('source', '')]
            
            if not syllabus_docs:
                # Fallback: try to find syllabus file by name
                syllabus_docs = [doc for doc in self.documents 
                                if 'CSE2021' in doc.metadata.get('source', '')]
            
            if syllabus_docs:
                # Create a separate vector store for syllabus ONLY
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                syllabus_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=3000,  # Large chunks to keep semester tables intact
                    chunk_overlap=600,  # Large overlap
                    length_function=len,
                    separators=["\n\n\n", "\n\n", "\nSEMESTER", "\n", ". ", " ", ""]
                )
                syllabus_splits = syllabus_splitter.split_documents(syllabus_docs)
                
                # Create temporary vector store for syllabus ONLY
                import tempfile
                import shutil
                temp_dir = tempfile.mkdtemp()
                try:
                    from langchain_community.vectorstores import Chroma
                    syllabus_store = Chroma.from_documents(
                        documents=syllabus_splits,
                        embedding=self.embeddings,
                        persist_directory=temp_dir
                    )
                    
                    # Use more documents for comprehensive retrieval
                    syllabus_retriever = syllabus_store.as_retriever(search_kwargs={"k": 60})
                    docs = syllabus_retriever.invoke(question)
                finally:
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except:
                        pass
            else:
                # If no syllabus docs found, use regular retriever but filter results
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": 40})
                all_docs = retriever.invoke(question)
                # Filter to only syllabus documents - strictly exclude cse.pdf
                docs = [doc for doc in all_docs 
                       if (('2.CSE2021' in doc.metadata.get('source', '') or 
                           'CSE2021' in doc.metadata.get('source', ''))
                           and 'cse.pdf' not in doc.metadata.get('source', '').lower())]
        else:
            # For non-syllabus questions, use regular retriever
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 20})
            docs = retriever.invoke(question)
        
        # Additional semester-specific searches for syllabus questions (only if we have syllabus retriever)
        if is_syllabus_question and 'syllabus_retriever' in locals():
            # For questions asking for multiple semesters (1st to 8th), search for ALL semesters
            if "1st" in question_lower or "first" in question_lower or "till" in question_lower or "to" in question_lower or "all" in question_lower:
                # User wants all semesters - search for each one
                all_semester_terms = []
                for sem_num, sem_terms in [
                    ("1", ["semester i", "semester 1", "1st semester", "sem i", "SEMESTER-I", "SEMESTER I"]),
                    ("2", ["semester ii", "semester 2", "2nd semester", "sem ii", "SEMESTER-II", "SEMESTER II"]),
                    ("3", ["semester iii", "semester 3", "3rd semester", "sem iii", "SEMESTER-III", "SEMESTER III"]),
                    ("4", ["semester iv", "semester 4", "4th semester", "sem iv", "SEMESTER-IV", "SEMESTER IV"]),
                    ("5", ["semester v", "semester 5", "5th semester", "sem v", "SEMESTER-V", "SEMESTER V"]),
                    ("6", ["semester vi", "semester 6", "6th semester", "sem vi", "SEMESTER-VI", "SEMESTER VI"]),
                    ("7", ["semester vii", "semester 7", "7th semester", "sem vii", "SEMESTER-VII", "SEMESTER VII"]),
                    ("8", ["semester viii", "semester 8", "8th semester", "sem viii", "SEMESTER-VIII", "SEMESTER VIII"])
                ]:
                    all_semester_terms.extend(sem_terms[:3])  # Top 3 variations per semester
                
                # Search for all semester variations
                additional_docs = []
                for term in all_semester_terms[:20]:  # Try top 20 variations
                    try:
                        if 'syllabus_store' in locals():
                            term_docs = syllabus_retriever.invoke(term)
                        else:
                            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 30})
                            term_docs = retriever.invoke(term)
                        additional_docs.extend(term_docs)
                    except:
                        pass
                
                # Combine and deduplicate
                seen_content = set()
                all_docs = []
                for doc in docs + additional_docs:
                    content_preview = doc.page_content[:300]
                    if content_preview not in seen_content:
                        seen_content.add(content_preview)
                        all_docs.append(doc)
                
                docs = all_docs[:80]  # Use up to 80 documents for all-semester queries
            else:
                # Single semester query - use existing logic
                semester_num = None
                search_terms = []
                if "3rd" in question_lower or "third" in question_lower or "semester iii" in question_lower:
                    semester_num = "3"
                    search_terms = ["semester iii", "semester 3", "3rd semester", "sem iii", "SEMESTER-III", "SEMESTER III"]
                elif "7th" in question_lower or "seventh" in question_lower or "semester vii" in question_lower:
                    semester_num = "7"
                    search_terms = ["semester vii", "semester 7", "7th semester", "sem vii", "SEMESTER-VII", "SEMESTER VII"]
                elif "4th" in question_lower or "fourth" in question_lower or "semester iv" in question_lower:
                    semester_num = "4"
                    search_terms = ["semester iv", "semester 4", "4th semester", "sem iv", "SEMESTER-IV", "SEMESTER IV"]
                elif "5th" in question_lower or "fifth" in question_lower or "semester v" in question_lower:
                    semester_num = "5"
                    search_terms = ["semester v", "semester 5", "5th semester", "sem v", "SEMESTER-V", "SEMESTER V"]
                elif "6th" in question_lower or "sixth" in question_lower or "semester vi" in question_lower:
                    semester_num = "6"
                    search_terms = ["semester vi", "semester 6", "6th semester", "sem vi", "SEMESTER-VI", "SEMESTER VI"]
                elif "1st" in question_lower or "first" in question_lower or "semester i" in question_lower:
                    semester_num = "1"
                    search_terms = ["semester i", "semester 1", "1st semester", "sem i", "SEMESTER-I", "SEMESTER I"]
                elif "2nd" in question_lower or "second" in question_lower or "semester ii" in question_lower:
                    semester_num = "2"
                    search_terms = ["semester ii", "semester 2", "2nd semester", "sem ii", "SEMESTER-II", "SEMESTER II"]
                elif "8th" in question_lower or "eighth" in question_lower or "semester viii" in question_lower:
                    semester_num = "8"
                    search_terms = ["semester viii", "semester 8", "8th semester", "sem viii", "SEMESTER-VIII", "SEMESTER VIII"]
                
                if semester_num and search_terms:
                    additional_docs = []
                    # Use syllabus retriever if available
                    if 'syllabus_retriever' in locals():
                        ret = syllabus_retriever
                    else:
                        ret = self.vectorstore.as_retriever(search_kwargs={"k": 30})
                    
                    for term in search_terms[:6]:
                        try:
                            term_docs = ret.invoke(term)
                            additional_docs.extend(term_docs)
                        except:
                            pass
                    
                    # Combine and deduplicate
                    seen_content = set()
                    all_docs = []
                    for doc in docs + additional_docs:
                        content_preview = doc.page_content[:300]
                        if content_preview not in seen_content:
                            seen_content.add(content_preview)
                            all_docs.append(doc)
                    
                    docs = all_docs[:50]  # Use up to 50 documents for single semester
        
        # Special handling for semester-specific queries
        semester_keywords = ["semester", "sem", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", 
                            "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
        
        if any(keyword in question.lower() for keyword in semester_keywords):
            # For semester queries, try additional searches with semester variations
            question_lower = question.lower()
            semester_queries = []
            
            # Map semester numbers to various formats
            if "5th" in question_lower or "fifth" in question_lower or "semester v" in question_lower or "sem v" in question_lower:
                semester_queries = ["semester v", "semester 5", "sem v", "5th semester", "fifth semester", "sem-5", "SEMESTER-V"]
            elif "1st" in question_lower or "first" in question_lower or "semester i" in question_lower:
                semester_queries = ["semester i", "semester 1", "sem i", "1st semester", "first semester", "sem-1", "SEMESTER-I"]
            elif "2nd" in question_lower or "second" in question_lower or "semester ii" in question_lower:
                semester_queries = ["semester ii", "semester 2", "sem ii", "2nd semester", "second semester", "sem-2", "SEMESTER-II"]
            elif "3rd" in question_lower or "third" in question_lower or "semester iii" in question_lower:
                semester_queries = ["semester iii", "semester 3", "sem iii", "3rd semester", "third semester", "sem-3", "SEMESTER-III"]
            elif "4th" in question_lower or "fourth" in question_lower or "semester iv" in question_lower:
                semester_queries = ["semester iv", "semester 4", "sem iv", "4th semester", "fourth semester", "sem-4", "SEMESTER-IV"]
            elif "6th" in question_lower or "sixth" in question_lower or "semester vi" in question_lower:
                semester_queries = ["semester vi", "semester 6", "sem vi", "6th semester", "sixth semester", "sem-6", "SEMESTER-VI"]
            elif "7th" in question_lower or "seventh" in question_lower or "semester vii" in question_lower:
                semester_queries = ["semester vii", "semester 7", "sem vii", "7th semester", "seventh semester", "sem-7", "SEMESTER-VII"]
            elif "8th" in question_lower or "eighth" in question_lower or "semester viii" in question_lower:
                semester_queries = ["semester viii", "semester 8", "sem viii", "8th semester", "eighth semester", "sem-8", "SEMESTER-VIII"]
            
            # Try additional searches with semester variations
            if semester_queries:
                additional_docs = []
                for query in semester_queries[:5]:  # Try top 5 variations
                    try:
                        query_docs = retriever.invoke(query)
                        additional_docs.extend(query_docs)
                    except:
                        pass
                
                # Also search for course-related terms if asking for subjects
                if any(word in question.lower() for word in ["subject", "paper", "course", "name", "title"]):
                    try:
                        course_docs = retriever.invoke("course title")
                        additional_docs.extend(course_docs)
                    except:
                        pass
                    try:
                        subject_docs = retriever.invoke("subject name")
                        additional_docs.extend(subject_docs)
                    except:
                        pass
                
                # Combine and deduplicate based on content
                seen_content = set()
                all_docs = []
                for doc in docs + additional_docs:
                    # Use first 200 chars as unique identifier
                    content_preview = doc.page_content[:200]
                    if content_preview not in seen_content:
                        seen_content.add(content_preview)
                        all_docs.append(doc)
                
                docs = all_docs[:25]  # Limit to top 25 unique documents
        
        # Also try MMR (Maximum Marginal Relevance) for diversity if we have few docs
        if len(docs) < 10:
            try:
                mmr_retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": 15, "fetch_k": 20},
                    search_type="mmr"
                )
                mmr_docs = mmr_retriever.invoke(question)
                # Combine both results, removing duplicates
                all_docs = docs + [d for d in mmr_docs if d not in docs]
                docs = all_docs[:20]  # Limit to top 20
            except:
                pass  # If MMR fails, continue with similarity search results
        
        # Combine context from retrieved documents with source information
        # Filter to only include documents from syllabus file for syllabus questions
        question_lower = question.lower()
        is_syllabus_q = any(word in question_lower for word in [
            "semester", "sem", "syllabus", "subject", "paper", "course", 
            "curriculum", "subjects", "papers", "courses"
        ])
        
        context_parts = []
        seen_content = set()  # Deduplicate based on content
        
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            
            # For syllabus questions, STRICTLY filter to ONLY 2.CSE2021.pdf
            if is_syllabus_q:
                # STRICT filtering - only 2.CSE2021.pdf
                if '2.CSE2021' not in source and 'CSE2021' not in source:
                    continue  # Skip ALL non-syllabus documents
                # Double check - reject cse.pdf, placement files, etc.
                if 'cse.pdf' in source.lower() or 'place' in source.lower() or 'faculty' in source.lower() or 'infrastructure' in source.lower():
                    continue  # Reject placement and other files
            
            content = doc.page_content
            
            # Deduplicate similar content
            content_hash = content[:200]  # Use first 200 chars as hash
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            # Add page number if available
            page = doc.metadata.get('page', '')
            source_info = f"[Source {i}: {source}"
            if page:
                source_info += f", Page {page}"
            source_info += "]"
            context_parts.append(f"{source_info}\n{content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # If no context after filtering, use all docs (fallback)
        if not context_parts and docs:
            for i, doc in enumerate(docs[:20], 1):  # Limit to top 20
                source = doc.metadata.get('source', 'Unknown')
                content = doc.page_content
                page = doc.metadata.get('page', '')
                source_info = f"[Source {i}: {source}"
                if page:
                    source_info += f", Page {page}"
                source_info += "]"
                context_parts.append(f"{source_info}\n{content}")
            context = "\n\n---\n\n".join(context_parts)
        
        # Get prompt template (for academic questions, use document-based template)
        prompt = self.get_prompt_template(user_type, is_general_question=False)
        
        # Format prompt with context and question
        # Add explicit instruction based on question type
        question_lower = question.lower()
        
        # Check if asking for subject names (not credits)
        asking_for_subjects = any(word in question_lower for word in ["subject", "paper", "course name", "course title", "subjects", "papers", "syllabus"]) and \
                             not any(word in question_lower for word in ["credit", "credits", "credit distribution", "credit wise", "total credit"])
        
        formatted_prompt = prompt.format(context=context, question=question)
        
        if asking_for_subjects:
            # Add STRONG reminder for subject name queries
            formatted_prompt += "\n\n" + "="*60 + "\n"
            formatted_prompt += "⚠️ CRITICAL INSTRUCTION - READ CAREFULLY:\n"
            formatted_prompt += "="*60 + "\n"
            formatted_prompt += "The user is asking for SUBJECT NAMES/COURSE TITLES ONLY.\n\n"
            formatted_prompt += "YOU MUST:\n"
            formatted_prompt += "1. Find the EXACT semester table asked about (e.g., if asked for 5th semester, find 'SEMESTER-V' table)\n"
            formatted_prompt += "2. Look ONLY at that specific semester's table - DO NOT look at other semesters\n"
            formatted_prompt += "3. Find the 'COURSE TITLE' column in that table\n"
            formatted_prompt += "4. Extract ONLY the course titles listed in that column for that specific semester\n"
            formatted_prompt += "5. List them EXACTLY as they appear in the COURSE TITLE column\n"
            formatted_prompt += "6. Example format: '• Theory of Computation\\n• Computer Networks\\n• Professional Elective-I'\n\n"
            formatted_prompt += "YOU MUST NOT:\n"
            formatted_prompt += "- Mix subjects from different semesters\n"
            formatted_prompt += "- Add subjects that are NOT in the COURSE TITLE column of the asked semester\n"
            formatted_prompt += "- Include laboratory courses unless they appear as separate entries in COURSE TITLE\n"
            formatted_prompt += "- Include credits, credit distribution, or total credits\n"
            formatted_prompt += "- Include course codes (like 21CS301)\n"
            formatted_prompt += "- Include categories (PC, PE, etc.)\n"
            formatted_prompt += "- Hallucinate or invent course names\n"
            formatted_prompt += "- Say 'credit distribution is...' or mention credits at all\n"
            formatted_prompt += "- Provide any information other than the exact course titles from that semester's table\n\n"
            formatted_prompt += "\n\n" + "🚨"*35 + "\n"
            formatted_prompt += "CRITICAL REMINDERS FOR SUBJECT NAME EXTRACTION:\n"
            formatted_prompt += "🚨"*35 + "\n"
            formatted_prompt += "1. The file 2.CSE2021.pdf contains COMPLETE syllabus for ALL 8 semesters\n"
            formatted_prompt += "2. The context above comes from 2.CSE2021.pdf - it HAS the information\n"
            formatted_prompt += "3. DO NOT say 'I don't know', 'not available', or 'cannot find'\n"
            formatted_prompt += "4. Search through EVERY part of the context for semester tables\n"
            formatted_prompt += "5. Look for ALL patterns: 'SEMESTER-I', 'SEMESTER-II', 'SEMESTER-III', 'SEMESTER-IV', 'SEMESTER-V', 'SEMESTER-VI', 'SEMESTER-VII', 'SEMESTER-VIII'\n"
            formatted_prompt += "6. Also look for: 'SEMESTER I', 'SEMESTER II', '3rd Semester', '4th Semester', etc.\n"
            formatted_prompt += "7. Find the 'COURSE TITLE' column in each semester's table\n"
            formatted_prompt += "8. Extract ALL course titles from that column - list them ALL\n"
            formatted_prompt += "9. If asked for '1st to 8th' or 'all semesters', provide subjects for EACH semester separately\n"
            formatted_prompt += "10. Format: SEMESTER-I:\\n• Subject 1\\n• Subject 2\\n\\nSEMESTER-II:\\n• Subject 1\\n• Subject 2\\n...\n"
            formatted_prompt += "11. The information IS in the context - you MUST find it!\n"
            formatted_prompt += "🚨"*35
        
        # Get response from Gemini
        try:
            if hasattr(self, 'use_direct_gemini') and self.use_direct_gemini:
                # Use direct Gemini API for better responses
                response = self.gemini_model.generate_content(
                    formatted_prompt,
                    generation_config={
                        "temperature": 0.0,  # Zero temperature for most deterministic, accurate responses
                        "max_output_tokens": 4096,
                        "top_p": 0.7,  # Lower top_p for more focused sampling
                        "top_k": 20,  # Lower top_k to reduce randomness
                        "candidate_count": 1  # Only one response
                    },
                    safety_settings=[
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                    ]
                )
                answer = response.text
            else:
                # Fallback to langchain wrapper if available
                response = self.llm.invoke(formatted_prompt)
                if hasattr(response, 'content'):
                    answer = response.content
                elif isinstance(response, str):
                    answer = response
                else:
                    answer = str(response)
        except Exception as e:
            answer = f"Error generating response: {str(e)}. Please check your Gemini API key."
        
        return {
            "answer": answer,
            "source_documents": docs
        }
    
    def load_existing_vectorstore(self):
        """Load existing vector store if available"""
        if os.path.exists("./chroma_db"):
            self.vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=self.embeddings
            )
            print("Loaded existing vector store")
            return True
        return False
