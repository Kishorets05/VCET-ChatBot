"""
Flask Web Application - HTML/CSS Frontend
Uses existing RAG system and timetable generator without modifications
"""
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
from rag_system import RAGSystem
from timetable_generator import TimetableGenerator
import PyPDF2

app = Flask(__name__)
CORS(app)

# Google Gemini API Key
GEMINI_API_KEY = "AIzaSyDbKNKegL5apzPGXwCNi0uXTc3LfCy_kKw"

# Global instances
rag_system = None
vectorstore_created = False
timetable_generator = TimetableGenerator()

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF"""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return ""

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/initialize', methods=['POST'])
def initialize_rag():
    """Initialize RAG system"""
    global rag_system, vectorstore_created
    
    try:
        if rag_system is None:
            rag_system = RAGSystem(GEMINI_API_KEY)
            
            # Try to load existing vectorstore
            if not rag_system.load_existing_vectorstore():
                # Create new vectorstore
                num_docs = rag_system.load_documents()
                if num_docs > 0:
                    unique_files = set()
                    for doc in rag_system.documents:
                        source = doc.metadata.get('source', 'Unknown')
                        unique_files.add(source)
                    
                    rag_system.create_vectorstore()
                    vectorstore_created = True
                    
                    # Verify vectorstore is queryable
                    try:
                        test_result = rag_system.vectorstore.similarity_search("test", k=1)
                        return jsonify({
                            'success': True,
                            'message': f'Loaded {len(unique_files)} files ({num_docs} document pages/chunks) successfully!'
                        })
                    except Exception as e:
                        return jsonify({'success': False, 'message': f'Vectorstore created but not queryable: {str(e)}'})
                else:
                    return jsonify({'success': False, 'message': 'No documents loaded. Please ensure PDF/TXT files are in the directory.'})
            else:
                # Verify existing vectorstore is queryable
                try:
                    test_result = rag_system.vectorstore.similarity_search("test", k=1)
                    vectorstore_created = True
                    return jsonify({'success': True, 'message': 'RAG system initialized with existing vectorstore'})
                except Exception as e:
                    # Vectorstore exists but not queryable, recreate it
                    num_docs = rag_system.load_documents()
                    if num_docs > 0:
                        unique_files = set()
                        for doc in rag_system.documents:
                            source = doc.metadata.get('source', 'Unknown')
                            unique_files.add(source)
                        rag_system.create_vectorstore()
                        vectorstore_created = True
                        return jsonify({
                            'success': True,
                            'message': f'Recreated vectorstore. Loaded {len(unique_files)} files ({num_docs} document pages/chunks) successfully!'
                        })
                    else:
                        return jsonify({'success': False, 'message': f'Existing vectorstore not queryable and no documents found: {str(e)}'})
        else:
            # Already initialized, verify it's still working
            if vectorstore_created:
                try:
                    test_result = rag_system.vectorstore.similarity_search("test", k=1)
                    return jsonify({'success': True, 'message': 'RAG system already initialized and working'})
                except Exception as e:
                    # Reset and reinitialize
                    rag_system = None
                    vectorstore_created = False
                    return jsonify({'success': False, 'message': f'RAG system needs reinitialization: {str(e)}'})
            else:
                return jsonify({'success': False, 'message': 'RAG system exists but vectorstore not created'})
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({'success': False, 'message': f'Initialization error: {str(e)}'})

@app.route('/api/check-initialized', methods=['GET'])
def check_initialized():
    """Check if chatbot is initialized"""
    global rag_system, vectorstore_created
    
    try:
        if rag_system is None or not vectorstore_created:
            return jsonify({'initialized': False})
        
        # Verify vectorstore is accessible
        if not hasattr(rag_system, 'vectorstore') or rag_system.vectorstore is None:
            return jsonify({'initialized': False})
        
        # Quick test query
        test_result = rag_system.vectorstore.similarity_search("test", k=1)
        return jsonify({'initialized': True})
    except Exception as e:
        return jsonify({'initialized': False, 'error': str(e)})

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat queries"""
    global rag_system, vectorstore_created
    
    # Validate request
    if not request.json:
        return jsonify({'success': False, 'message': 'Invalid request: No JSON data provided'})
    
    # Check initialization
    if rag_system is None:
        return jsonify({'success': False, 'message': 'Please initialize the chatbot first! Click the "Initialize Chatbot" button.'})
    
    if not vectorstore_created:
        return jsonify({'success': False, 'message': 'Vectorstore not created. Please initialize the chatbot first!'})
    
    # Verify vectorstore is accessible
    try:
        if not hasattr(rag_system, 'vectorstore') or rag_system.vectorstore is None:
            return jsonify({'success': False, 'message': 'Vectorstore not available. Please reinitialize the chatbot.'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Vectorstore error: {str(e)}. Please reinitialize.'})
    
    data = request.json
    question = data.get('question', '').strip()
    user_type = data.get('user_type', 'student')
    
    if not question:
        return jsonify({'success': False, 'message': 'Please provide a question.'})
    
    try:
        result = rag_system.query(question, user_type)
        answer = result.get('answer', '')
        sources = result.get('sources', [])
        
        if not answer:
            return jsonify({'success': False, 'message': 'No response generated. Please try again or rephrase your question.'})
        
        return jsonify({
            'success': True,
            'answer': answer,
            'sources': sources
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({'success': False, 'message': f'Error processing query: {str(e)}'})

@app.route('/api/reset', methods=['POST'])
def reset_chatbot():
    """Reset chatbot state (for testing/debugging)"""
    global rag_system, vectorstore_created
    rag_system = None
    vectorstore_created = False
    return jsonify({'success': True, 'message': 'Chatbot state reset. Please reinitialize.'})

@app.route('/api/generate-timetable', methods=['POST'])
def generate_timetable():
    """Generate timetable PDF"""
    global rag_system, vectorstore_created, timetable_generator
    
    if rag_system is None or not vectorstore_created:
        return jsonify({'success': False, 'message': 'Please initialize the chatbot first!'})
    
    data = request.json
    faculty_name = data.get('faculty_name', '')
    semester1 = data.get('semester1', '')
    semester2 = data.get('semester2', '')
    
    if not faculty_name or not semester1 or not semester2:
        return jsonify({'success': False, 'message': 'Missing required fields'})
    
    try:
        # Extract semester numbers
        sem1_num = semester1.split()[0].replace("st", "").replace("nd", "").replace("rd", "").replace("th", "")
        sem2_num = semester2.split()[0].replace("st", "").replace("nd", "").replace("rd", "").replace("th", "")
        
        # Query RAG system for subjects
        query1 = f"List all subjects or papers included in {semester1.lower()} of CSE department. Provide only the subject names from the COURSE TITLE column, one per line."
        query2 = f"List all subjects or papers included in {semester2.lower()} of CSE department. Provide only the subject names from the COURSE TITLE column, one per line."
        
        result1 = rag_system.query(query1, "faculty")
        result2 = rag_system.query(query2, "faculty")
        
        # Extract subject names
        answer1 = result1.get("answer", "")
        answer2 = result2.get("answer", "")
        
        error_keywords = ['error', 'quota', '429', 'exceeded', 'rate limit', 'violations']
        api_error = any(keyword in answer1.lower() or keyword in answer2.lower() for keyword in error_keywords)
        
        if api_error:
            # PDF fallback
            pdf_text = extract_text_from_pdf("2.CSE2021.pdf")
            import re
            
            def extract_from_pdf(pdf_text, semester_patterns):
                subjects = []
                lines = pdf_text.split('\n')
                in_semester = False
                
                for i, line in enumerate(lines):
                    line_upper = line.upper().strip()
                    if any(pattern.upper() in line_upper for pattern in semester_patterns):
                        in_semester = True
                        continue
                    
                    if in_semester:
                        if 'COURSE TITLE' in line_upper or 'SUBJECT' in line_upper:
                            continue
                        if 'SEMESTER' in line_upper and not any(p.upper() in line_upper for p in semester_patterns):
                            break
                        if 10 < len(line.strip()) < 80:
                            cleaned = line.strip()
                            cleaned = re.sub(r'^[A-Z]{2,4}\s*\d{4}\s*', '', cleaned)
                            if cleaned and not any(x in cleaned.upper() for x in ['CREDIT', 'L T P', 'TOTAL', 'SEMESTER']):
                                if cleaned not in subjects:
                                    subjects.append(cleaned)
                                    if len(subjects) >= 10:
                                        break
                return subjects[:10]
            
            sem1_patterns = [f"SEMESTER-{sem1_num}", f"SEMESTER {sem1_num}", f"{sem1_num}rd", f"{sem1_num}th", f"{sem1_num}st", f"{sem1_num}nd"]
            sem2_patterns = [f"SEMESTER-{sem2_num}", f"SEMESTER {sem2_num}", f"{sem2_num}rd", f"{sem2_num}th", f"{sem2_num}st", f"{sem2_num}nd"]
            
            subjects_sem1 = extract_from_pdf(pdf_text, sem1_patterns)
            subjects_sem2 = extract_from_pdf(pdf_text, sem2_patterns)
            
            if not subjects_sem1 or not subjects_sem2:
                return jsonify({'success': False, 'message': 'Could not extract subjects from PDF'})
            
            import random
            selected_subject1 = random.choice(subjects_sem1)
            selected_subject2 = random.choice(subjects_sem2)
        else:
            subjects_sem1 = timetable_generator.extract_subject_names_from_response(answer1)
            subjects_sem2 = timetable_generator.extract_subject_names_from_response(answer2)
            
            subjects_sem1 = [s for s in subjects_sem1 if len(s) > 5 and len(s) < 80 and 
                            not s.upper().startswith('SEMESTER') and 
                            'semester' not in s.lower()[:10] and
                            not any(x in s.lower() for x in ['http', 'error', 'quota', 'api'])]
            subjects_sem2 = [s for s in subjects_sem2 if len(s) > 5 and len(s) < 80 and 
                            not s.upper().startswith('SEMESTER') and 
                            'semester' not in s.lower()[:10] and
                            not any(x in s.lower() for x in ['http', 'error', 'quota', 'api'])]
            
            if not subjects_sem1 or not subjects_sem2:
                return jsonify({'success': False, 'message': 'Could not extract valid subjects'})
            
            import random
            selected_subject1 = random.choice(subjects_sem1)
            selected_subject2 = random.choice(subjects_sem2)
        
        # Generate timetable
        output_path = f"timetable_{faculty_name.replace(' ', '_')}.pdf"
        generated_path = timetable_generator.generate_timetable(
            faculty_name, semester1, semester2,
            selected_subject1, selected_subject2,
            "", "",  # No syllabus content
            output_path
        )
        
        return jsonify({
            'success': True,
            'message': 'Timetable generated successfully!',
            'file_path': generated_path,
            'subject1': selected_subject1,
            'subject2': selected_subject2
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/download/<filename>')
def download_file(filename):
    """Download generated PDF"""
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'message': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    print("=" * 60)
    print("Starting VCET CSE Chatbot Web Server")
    print("=" * 60)
    print("Server will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    app.run(debug=True, port=5000, host='0.0.0.0')


