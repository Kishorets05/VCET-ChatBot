from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import os
import re
import random

class TimetableGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        
    def setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1a237e'),
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#283593'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        self.normal_style = self.styles['Normal']
        
        self.syllabus_style = ParagraphStyle(
            'SyllabusStyle',
            parent=self.styles['Normal'],
            fontSize=12,
            leading=18,
            spaceAfter=12,
            leftIndent=0,
            rightIndent=0,
            firstLineIndent=0
        )
        
        self.syllabus_heading_style = ParagraphStyle(
            'SyllabusHeadingStyle',
            parent=self.styles['Heading2'],
            fontSize=13,
            textColor=colors.HexColor('#1a237e'),
            spaceAfter=8,
            spaceBefore=12
        )
    
    def extract_subjects_from_syllabus(self, syllabus_text: str, year: str) -> list:
        """Extract subjects for a given year from syllabus text"""
        subjects = []
        
        # Common patterns for subject extraction
        year_patterns = {
            "first": ["first year", "i year", "year 1", "semester 1", "semester 2"],
            "second": ["second year", "ii year", "year 2", "semester 3", "semester 4"],
            "third": ["third year", "iii year", "year 3", "semester 5", "semester 6"],
            "final": ["fourth year", "final year", "iv year", "year 4", "semester 7", "semester 8"]
        }
        
        # Convert year to lowercase for matching
        year_lower = year.lower()
        patterns = year_patterns.get(year_lower, [])
        
        # Split text into lines and search for subjects
        lines = syllabus_text.split('\n')
        in_year_section = False
        subject_count = 0
        
        for line in lines:
            line_lower = line.lower()
            
            # Check if we're entering the year section
            if any(pattern in line_lower for pattern in patterns):
                in_year_section = True
                continue
            
            # If in year section, look for subject names
            if in_year_section and subject_count < 10:  # Limit to reasonable number
                # Look for lines that might be subject names
                # Common patterns: course codes, subject names
                if re.match(r'^[A-Z]{2,4}\s*\d{4}', line.strip()):  # Course code pattern
                    subject = line.strip()
                    if len(subject) > 5 and len(subject) < 100:
                        subjects.append(subject)
                        subject_count += 1
                elif len(line.strip()) > 10 and len(line.strip()) < 80:
                    # Might be a subject name
                    if not any(word in line_lower for word in ['semester', 'year', 'regulation', 'credit']):
                        subjects.append(line.strip())
                        subject_count += 1
            
            # Stop if we've found enough subjects or moved to next section
            if subject_count >= 10:
                break
        
        # If no subjects found, return default subjects based on year
        if not subjects:
            subjects = self.get_default_subjects(year)
        
        return subjects[:10]  # Return max 10 subjects
    
    def get_default_subjects(self, year: str) -> list:
        """Get default subjects for a year if extraction fails"""
        default_subjects = {
            "first": [
                "Mathematics I",
                "Engineering Physics",
                "Engineering Chemistry",
                "Programming in C",
                "Engineering Graphics",
                "Communication Skills",
                "Mathematics II",
                "Data Structures",
                "Digital Electronics",
                "Computer Organization"
            ],
            "second": [
                "Data Structures and Algorithms",
                "Object Oriented Programming",
                "Database Management Systems",
                "Computer Networks",
                "Operating Systems",
                "Discrete Mathematics",
                "Software Engineering",
                "Web Technologies",
                "Microprocessors",
                "Theory of Computation"
            ],
            "third": [
                "Machine Learning",
                "Cloud Computing",
                "Mobile Application Development",
                "Information Security",
                "Compiler Design",
                "Distributed Systems",
                "Artificial Intelligence",
                "Data Mining",
                "Computer Graphics",
                "Cryptography"
            ],
            "final": [
                "Project Management",
                "Big Data Analytics",
                "Internet of Things",
                "Blockchain Technology",
                "Deep Learning",
                "Cyber Security",
                "Natural Language Processing",
                "Project Work",
                "Industrial Training",
                "Seminar"
            ]
        }
        
        year_lower = year.lower()
        if "first" in year_lower or "1" in year_lower:
            return default_subjects["first"]
        elif "second" in year_lower or "2" in year_lower:
            return default_subjects["second"]
        elif "third" in year_lower or "3" in year_lower:
            return default_subjects["third"]
        elif "final" in year_lower or "fourth" in year_lower or "4" in year_lower:
            return default_subjects["final"]
        else:
            return default_subjects["first"]
    
    def extract_subject_names_from_response(self, response_text: str) -> list:
        """Extract subject names from RAG response - improved parsing with error filtering"""
        subjects = []
        
        # CRITICAL: Check if response contains error messages
        error_indicators = [
            'error generating response', 'quota exceeded', '429', 'rate limit',
            'api', 'http', 'https://', 'violations', 'retry', 'exceeded',
            'billing', 'plan', 'quota', 'limit', 'gemini', 'google'
        ]
        
        response_lower = response_text.lower()
        if any(indicator in response_lower for indicator in error_indicators):
            # This is an error response, not subject names
            return []
        
        lines = response_text.split('\n')
        
        # Keywords to exclude
        exclude_keywords = ['semester', 'year', 'credit', 'total', 'source', 'based on', 
                           'provided', 'context', 'document', 'table', 'column', 'header',
                           'regulation', 'curriculum', 'syllabus', 'course code', 'category',
                           'error', 'quota', 'api', 'http', 'https', 'violations', 'retry']
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip lines that look like error messages or URLs
            if any(exclude in line.lower() for exclude in ['http://', 'https://', 'url:', 'error', 'quota', 'violations']):
                continue
                
            # Remove bullet points, numbers, dashes
            cleaned = re.sub(r'^[•\-\*\d\.\)\s]+', '', line).strip()
            
            # Skip if too short or too long
            if len(cleaned) < 5 or len(cleaned) > 100:
                continue
            
            # Skip if it's a header or contains exclude keywords
            line_lower = cleaned.lower()
            if any(keyword in line_lower for keyword in exclude_keywords):
                continue
            
            # Skip if it starts with semester indicators
            if cleaned.upper().startswith('SEMESTER') or cleaned.startswith('SEM '):
                continue
            
            # Skip if it's a source citation
            if cleaned.startswith('Source') or cleaned.startswith('📚'):
                continue
            
            # Skip if it contains URLs or error patterns
            if re.search(r'https?://|www\.|\.com|\.org|\.dev', cleaned, re.IGNORECASE):
                continue
            
            # Remove common prefixes/suffixes
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', cleaned)  # Remove leading numbers
            cleaned = re.sub(r'\s*\(.*?\)\s*$', '', cleaned)  # Remove trailing parentheses
            cleaned = re.sub(r'\s*\[.*?\]\s*$', '', cleaned)  # Remove trailing brackets
            
            # Skip if it's just a number or code
            if re.match(r'^[A-Z]{2,4}\s*\d{4}$', cleaned):  # Just course code
                continue
            
            # If it contains a course code, extract the subject name part
            # Pattern: CODE1234 Subject Name
            match = re.match(r'^[A-Z]{2,4}\s*\d{4}\s+(.+)$', cleaned)
            if match:
                cleaned = match.group(1).strip()
            
            # Final validation - must look like a real subject name
            if len(cleaned) >= 5 and len(cleaned) <= 100:
                # Check if it looks like a subject name (has letters, not just numbers/symbols)
                # Must have at least 3 consecutive letters
                if re.search(r'[A-Za-z]{3,}', cleaned):
                    # Must not be mostly special characters
                    letter_count = len(re.findall(r'[A-Za-z]', cleaned))
                    if letter_count >= len(cleaned) * 0.5:  # At least 50% letters
                        subjects.append(cleaned)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_subjects = []
        for subj in subjects:
            subj_lower = subj.lower().strip()
            if subj_lower and subj_lower not in seen:
                seen.add(subj_lower)
                unique_subjects.append(subj.strip())
        
        return unique_subjects
    
    def get_subject_syllabus(self, rag_system, subject_name: str, semester: str = "") -> str:
        """Get syllabus content for a specific subject using RAG system with PDF fallback"""
        try:
            query = f"detailed syllabus and contents of {subject_name}"
            if semester:
                query += f" from {semester}"
            result = rag_system.query(query, "faculty")
            answer = result.get("answer", "")
            
            # Check if response contains errors
            error_keywords = ['error', 'quota', '429', 'exceeded', 'rate limit', 'violations']
            if any(keyword in answer.lower() for keyword in error_keywords):
                # Use PDF fallback
                return self.get_syllabus_from_pdf(subject_name, semester)
            
            return answer
        except Exception as e:
            # Use PDF fallback on any error
            return self.get_syllabus_from_pdf(subject_name, semester)
    
    def get_syllabus_from_pdf(self, subject_name: str, semester: str = "") -> str:
        """Extract syllabus content directly from PDF as fallback"""
        try:
            import PyPDF2
            pdf_path = "2.CSE2021.pdf"
            if not os.path.exists(pdf_path):
                return f"Detailed syllabus for {subject_name} from {semester}. (PDF file not found)"
            
            pdf_text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text() + "\n"
            
            # Search for subject in PDF
            lines = pdf_text.split('\n')
            syllabus_content = []
            found_subject = False
            
            # Look for subject name in PDF
            subject_keywords = [kw for kw in subject_name.split() if len(kw) > 3]
            for i, line in enumerate(lines):
                line_upper = line.upper().strip()
                # Check if this line contains the subject name
                if subject_keywords and all(keyword.upper() in line_upper for keyword in subject_keywords):
                    found_subject = True
                    # Collect next 50 lines as syllabus content
                    for j in range(i, min(i + 50, len(lines))):
                        syllabus_content.append(lines[j])
                    break
            
            if found_subject and syllabus_content:
                # Filter out curriculum document headers/footers
                filtered_content = []
                exclude_patterns = [
                    'B.E.CSE CURRICULUM',
                    'B.E-CSE CURRICULUM',
                    'R-2021',
                    'CHOICE BASED CREDIT SYSTEM',
                    'I TO VIII SEMESTERS',
                    'SEMESTERS)',
                    'BoS-CHAIRMAN',
                    'CHAIRMAN'
                ]
                
                for line in syllabus_content:
                    line_upper = line.upper().strip()
                    # Skip lines that match exclusion patterns
                    if any(pattern in line_upper for pattern in exclude_patterns):
                        continue
                    # Skip very short lines that are likely headers
                    if len(line.strip()) < 5:
                        continue
                    filtered_content.append(line)
                
                # Return filtered content (max 30 lines)
                result = "\n".join(filtered_content[:30])
                return result if result.strip() else f"Detailed syllabus for {subject_name} from {semester}."
            else:
                return f"Detailed syllabus for {subject_name} from {semester}. (Subject found in curriculum but detailed content extraction from PDF requires manual review)"
        except Exception as e:
            return f"Detailed syllabus for {subject_name} from {semester}. (PDF extraction failed: {str(e)})"
    
    def generate_timetable(self, faculty_name: str, semester1: str, semester2: str, 
                          subject1: str, subject2: str,
                          syllabus1: str, syllabus2: str,
                          output_path: str = "timetable.pdf"):
        """Generate timetable PDF for faculty with two subjects from two semesters"""
        
        # Create PDF document with better margins for readability
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=1*inch,
            leftMargin=1*inch,
            topMargin=0.8*inch,
            bottomMargin=0.8*inch
        )
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Title with better spacing
        title = Paragraph(f"<b>FACULTY TIMETABLE</b>", self.title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.5*inch))
        
        # Faculty Information with better formatting and spacing
        faculty_info = f"<b>Faculty Name:</b> {faculty_name}<br/><br/>"
        faculty_info += f"<b>Academic Year:</b> {datetime.now().strftime('%Y-%m-%d')}<br/><br/>"
        faculty_info += f"<b>Semesters Assigned:</b> {semester1} & {semester2}"
        elements.append(Paragraph(faculty_info, ParagraphStyle(
            'FacultyInfoStyle',
            parent=self.normal_style,
            fontSize=12,
            leading=18,
            spaceAfter=20
        )))
        elements.append(Spacer(1, 0.5*inch))
        
        # Days of the week
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        time_slots = ["9:00-10:00", "10:00-11:00", "11:00-12:00", "12:00-1:00", 
                     "1:00-2:00 (Lunch)", "2:00-3:00", "3:00-4:00", "4:00-5:00"]
        
        # Combine subjects with their semester labels
        all_subjects = [subject1, subject2]
        semester_labels = [semester1, semester2]
        
        # Create timetable data
        # Header row
        header = ["Time"] + days
        data = [header]
        
        # Create timetable with 2-3 periods per day for each subject
        # Each subject gets 2-3 periods on each of their assigned days
        subject_slots = {}
        
        # Available time slots (excluding lunch)
        available_slots = [slot for slot in time_slots if "Lunch" not in slot]
        
        # Distribute subjects: 2-3 periods per day on assigned days
        for i, (subject, sem_label) in enumerate(zip(all_subjects, semester_labels)):
            subject_slots[subject] = {
                'semester': sem_label,
                'slots': []
            }
            
            if i == 0:  # First subject - Monday, Wednesday, Friday (2-3 periods each day)
                # Subject 1: Monday (3 periods), Wednesday (3 periods), Friday (2 periods)
                # available_slots indices: 0=9-10, 1=10-11, 2=11-12, 3=12-1, 4=2-3, 5=3-4, 6=4-5
                subject_slots[subject]['slots'] = [
                    (days[0], available_slots[0]),  # Monday 9-10
                    (days[0], available_slots[1]),  # Monday 10-11
                    (days[0], available_slots[2]),  # Monday 11-12
                    (days[2], available_slots[0]),  # Wednesday 9-10
                    (days[2], available_slots[1]),  # Wednesday 10-11
                    (days[2], available_slots[2]),  # Wednesday 11-12
                    (days[4], available_slots[4]),  # Friday 2-3
                    (days[4], available_slots[5]),  # Friday 3-4
                ]
            else:  # Second subject - Tuesday, Thursday, Saturday (2-3 periods each day)
                # Subject 2: Tuesday (3 periods), Thursday (3 periods), Saturday (2 periods)
                # available_slots indices: 0=9-10, 1=10-11, 2=11-12, 3=12-1, 4=2-3, 5=3-4, 6=4-5
                subject_slots[subject]['slots'] = [
                    (days[1], available_slots[1]),  # Tuesday 10-11
                    (days[1], available_slots[2]),  # Tuesday 11-12
                    (days[1], available_slots[3]),  # Tuesday 12-1
                    (days[3], available_slots[4]),  # Thursday 2-3
                    (days[3], available_slots[5]),  # Thursday 3-4
                    (days[3], available_slots[6]),  # Thursday 4-5
                    (days[5], available_slots[0]),  # Saturday 9-10
                    (days[5], available_slots[1]),  # Saturday 10-11
                ]
        
        # Fill timetable grid
        for time_slot in time_slots:
            if "Lunch" in time_slot:
                row = [time_slot] + ["LUNCH BREAK"] * len(days)
            else:
                row = [time_slot]
                for day in days:
                    cell_subjects = []
                    for subject, info in subject_slots.items():
                        for slot_day, slot_time in info['slots']:
                            if slot_day == day and slot_time == time_slot:
                                # Only subject name, no semester info
                                cell_subjects.append(subject)
                    
                    # Create cell content - only subject names
                    if cell_subjects:
                        # Format subject names with line breaks if needed
                        subject_text = ""
                        for idx, subj in enumerate(cell_subjects):
                            if idx > 0:
                                subject_text += "<br/>"
                            
                            # Wrap long subject names intelligently
                            if len(subj) > 20:
                                words = subj.split()
                                if len(words) > 1:
                                    mid = len(words) // 2
                                    subject_text += " ".join(words[:mid]) + "<br/>" + " ".join(words[mid:])
                                else:
                                    if len(subj) > 30:
                                        mid = len(subj) // 2
                                        subject_text += subj[:mid] + "-<br/>" + subj[mid:]
                                    else:
                                        subject_text += subj
                            else:
                                subject_text += subj
                        
                        # Use Paragraph for proper rendering (no literal HTML tags)
                        # Clean subject text - remove any remaining error messages or URLs
                        clean_text = subject_text
                        # Remove any URLs
                        clean_text = re.sub(r'https?://[^\s]+', '', clean_text)
                        clean_text = re.sub(r'www\.[^\s]+', '', clean_text)
                        # Remove error-related text
                        clean_text = re.sub(r'error|quota|violations|api|http', '', clean_text, flags=re.IGNORECASE)
                        
                        if clean_text.strip():
                            cell_para = Paragraph(
                                clean_text.strip(),
                                ParagraphStyle(
                                    'CellTextStyle',
                                    parent=self.normal_style,
                                    fontSize=10,
                                    leading=12,
                                    alignment=1,  # Center
                                    textColor=colors.black
                                )
                            )
                            row.append(cell_para)
                        else:
                            row.append("-")
                    else:
                        row.append("-")
            data.append(row)
        
        # Create table with better column widths
        table = Table(data, colWidths=[1.3*inch] + [1.15*inch] * len(days))
        
        # Apply table style with better spacing
        table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 15),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            
            # Time column
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#e3f2fd')),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (0, -1), 9),
            
            # Data rows
            ('FONTNAME', (1, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (1, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (1, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 1), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 15),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.5*inch))
        
        # Subject Details Section with better formatting
        elements.append(Paragraph(f"<b>Subject Details</b>", self.heading_style))
        elements.append(Spacer(1, 0.2*inch))
        
        subject_details_data = [["Semester", "Subject Name"]]
        subject_details_data.append([semester1, subject1])
        subject_details_data.append([semester2, subject2])
        
        subject_table = Table(subject_details_data, colWidths=[2.2*inch, 4.8*inch])
        subject_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#283593')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 15),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 12),
        ]))
        
        elements.append(subject_table)
        
        # Syllabus content removed - only timetable is generated
        
        # Build PDF
        doc.build(elements)
        
        return output_path
    
    def format_syllabus_content(self, content: str) -> str:
        """Format syllabus content for PDF display - fix HTML tags and remove unwanted headers"""
        import re
        
        # Remove any existing para tags
        content = content.replace('<para>', '').replace('</para>', '')
        
        # Remove curriculum document headers/footers (those blue lines)
        lines = content.split('\n')
        filtered_lines = []
        exclude_patterns = [
            'B.E.CSE CURRICULUM',
            'B.E-CSE CURRICULUM',
            'R-2021',
            'CHOICE BASED CREDIT SYSTEM',
            'I TO VIII SEMESTERS',
            'SEMESTERS)',
            'BoS-CHAIRMAN',
            'CHAIRMAN'
        ]
        
        for line in lines:
            line_upper = line.upper().strip()
            # Skip lines that match exclusion patterns
            if any(pattern in line_upper for pattern in exclude_patterns):
                continue
            # Skip very short lines that are likely headers
            if len(line.strip()) < 5 and not line.strip().isdigit():
                continue
            filtered_lines.append(line)
        
        content = '\n'.join(filtered_lines)
        
        # Fix common HTML issues: <b>text<b> should be <b>text</b>
        # Pattern: <b> followed by text, then another <b> (should be </b>)
        content = re.sub(r'<b>([^<]+)<b>', r'<b>\1</b>', content, flags=re.IGNORECASE)
        
        # Fix patterns like <b>VERTICAL 1: DATA SCIENCE<b> -> <b>VERTICAL 1: DATA SCIENCE</b>
        # Look for <b> followed by text and then <b> or end of line
        content = re.sub(r'<b>([^<]+?)(?=<b>|$)', lambda m: f'<b>{m.group(1)}</b>' if '</b>' not in m.group(0) else m.group(0), content)
        
        # Replace markdown-style formatting
        content = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', content)
        content = re.sub(r'\*([^*]+)\*', r'<b>\1</b>', content)
        
        # Remove standalone asterisks
        content = re.sub(r'(?<!\*)\*(?!\*)', '', content)
        
        # Remove markdown headers
        content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
        
        # Replace newlines with HTML breaks
        content = content.replace('\n', '<br/>')
        
        # Final cleanup - ensure all <b> tags are properly closed
        # Count tags
        open_b = len(re.findall(r'<b>', content, re.IGNORECASE))
        close_b = len(re.findall(r'</b>', content, re.IGNORECASE))
        
        # If there are unclosed tags, fix them
        if open_b > close_b:
            # Find positions of unclosed <b> tags
            parts = re.split(r'(<b>|</b>)', content)
            tag_stack = []
            result_parts = []
            
            for part in parts:
                if part.lower() == '<b>':
                    tag_stack.append('<b>')
                    result_parts.append(part)
                elif part.lower() == '</b>':
                    if tag_stack:
                        tag_stack.pop()
                    result_parts.append(part)
                else:
                    result_parts.append(part)
            
            # Close any remaining open tags
            content = ''.join(result_parts) + '</b>' * len(tag_stack)
        
        # Remove any other problematic HTML tags (keep only b, br, i, u)
        allowed_tags = ['<b>', '</b>', '<br/>', '<br>', '<i>', '</i>', '<u>', '</u>']
        # Remove tags that aren't in our allowed list
        def clean_tag(match):
            tag = match.group(0).lower()
            if any(allowed in tag for allowed in allowed_tags):
                return match.group(0)
            return ''
        
        content = re.sub(r'<[^>]+>', clean_tag, content)
        
        # Limit length to avoid PDF issues
        if len(content) > 5000:
            content = content[:5000] + "<br/><br/>(Content truncated for display)"
        
        return content
    
    def structure_syllabus_content(self, content: str) -> list:
        """Structure syllabus content into headings, subheadings, and paragraphs"""
        sections = []
        lines = content.split('<br/>')
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_paragraph:
                    sections.append({
                        'type': 'paragraph',
                        'content': '<br/>'.join(current_paragraph)
                    })
                    current_paragraph = []
                continue
            
            # Detect headings (all caps, starts with UNIT, COURSE, etc.)
            if (line.isupper() and len(line) > 5) or \
               re.match(r'^(UNIT|COURSE|PRESCRIBED|REFERENCE|TEXT|BOOK)', line, re.IGNORECASE) or \
               re.match(r'^\d+(\.\d+)*\s+[A-Z]', line):
                if current_paragraph:
                    sections.append({
                        'type': 'paragraph',
                        'content': '<br/>'.join(current_paragraph)
                    })
                    current_paragraph = []
                
                # Check if it's a main heading or subheading
                if re.match(r'^UNIT[-:]', line, re.IGNORECASE) or \
                   re.match(r'^(PRESCRIBED|REFERENCE|TEXT|BOOK)', line, re.IGNORECASE):
                    sections.append({
                        'type': 'heading',
                        'content': line
                    })
                else:
                    sections.append({
                        'type': 'subheading',
                        'content': line
                    })
            else:
                current_paragraph.append(line)
        
        # Add remaining paragraph
        if current_paragraph:
            sections.append({
                'type': 'paragraph',
                'content': '<br/>'.join(current_paragraph)
            })
        
        return sections
    
    def split_long_content(self, content: str, max_length: int = 1500) -> list:
        """Split long content into smaller chunks to avoid PDF parsing issues"""
        if len(content) <= max_length:
            return [content]
        
        parts = []
        # Split by <br/> tags first
        chunks = content.split('<br/>')
        current_part = ""
        
        for chunk in chunks:
            if len(current_part) + len(chunk) + 5 < max_length:  # +5 for <br/>
                current_part += chunk + "<br/>"
            else:
                if current_part:
                    parts.append(current_part)
                current_part = chunk + "<br/>"
        
        if current_part:
            parts.append(current_part)
        
        return parts if parts else [content]










