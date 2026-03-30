#!/usr/bin/env python3
"""
Process medical document and convert to fine-tuning format
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any

def extract_from_docx(file_path: str) -> str:
    """Extract text from Word document"""
    try:
        from docx import Document
        doc = Document(file_path)
        text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text.append(paragraph.text.strip())
        return '\n'.join(text)
    except ImportError:
        print("python-docx not available, trying alternative method...")
        # Alternative: try to read as text (might not work well)
        with open(file_path, 'rb') as f:
            content = f.read()
            # This is a basic fallback - not ideal for .docx
            return str(content)

def create_medical_training_data(text: str) -> List[Dict[str, str]]:
    """Convert medical text into training examples"""
    
    # Split by case patterns - look for case numbers, patient scenarios, etc.
    case_patterns = [
        r'Case\s+\d+',
        r'Patient\s+\d+',
        r'Scenario\s+\d+',
        r'Example\s+\d+',
        r'\d+\.\s+',  # Numbered items
        r'^\d+\)',    # Numbered with parentheses
    ]
    
    # Split text into potential cases
    sections = []
    current_section = []
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line starts a new case/section
        is_new_case = any(re.search(pattern, line, re.IGNORECASE) for pattern in case_patterns)
        
        if is_new_case and current_section:
            # Save previous section
            sections.append('\n'.join(current_section))
            current_section = [line]
        else:
            current_section.append(line)
    
    # Don't forget the last section
    if current_section:
        sections.append('\n'.join(current_section))
    
    # Convert sections to training examples
    training_data = []
    
    for i, section in enumerate(sections):
        if len(section.strip()) < 50:  # Skip very short sections
            continue
            
        # Try to extract medical scenarios and convert to Q&A format
        lines = [l.strip() for l in section.split('\n') if l.strip()]
        
        if len(lines) < 2:
            continue
            
        # Look for question-like content and answer-like content
        potential_question = lines[0]
        potential_answer = '\n'.join(lines[1:])
        
        # Create medical conversation format
        if len(potential_question) > 20 and len(potential_answer) > 20:
            
            # Make it more conversational
            if not potential_question.endswith('?'):
                if 'patient' in potential_question.lower():
                    potential_question = f"What would you recommend for this case: {potential_question}?"
                else:
                    potential_question = f"Please provide medical guidance: {potential_question}?"
            
            training_example = {
                "instruction": potential_question,
                "input": "",
                "output": potential_answer
            }
            training_data.append(training_example)
            
        # Also create a general medical Q&A from the full section
        if len(section) > 100:
            general_question = f"Can you analyze this oncology case and provide clinical insights?"
            training_example = {
                "instruction": general_question,
                "input": section[:500] + "..." if len(section) > 500 else section,
                "output": f"Based on this oncology case, here are the key clinical considerations and recommendations:\n\n{potential_answer}"
            }
            training_data.append(training_example)
    
    return training_data

def main():
    """Main processing function"""
    input_file = "/Users/jefneyongeri/Next attempt/EHR-AI-SIDEBAR-/backend/medical_document.docx"
    output_file = "/Users/jefneyongeri/Next attempt/EHR-AI-SIDEBAR-/backend/oncology_training_data.jsonl"
    
    print("🏥 Processing medical document...")
    
    # Extract text from document
    try:
        text = extract_from_docx(input_file)
        print(f"📄 Extracted {len(text)} characters from document")
    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        return
    
    # Create training data
    training_data = create_medical_training_data(text)
    print(f"🤖 Created {len(training_data)} training examples")
    
    # Save as JSONL format for fine-tuning
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in training_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved training data to: {output_file}")
    
    # Show a sample
    if training_data:
        print("\n📋 Sample training example:")
        sample = training_data[0]
        print(f"Instruction: {sample['instruction'][:100]}...")
        print(f"Output: {sample['output'][:100]}...")
    
    return output_file

if __name__ == "__main__":
    output_path = main()