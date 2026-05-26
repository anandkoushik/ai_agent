import re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LogicalUnit:
    type: str  # "question", "section", "text"
    content: str
    page_num: Optional[int] = None
    question_num: Optional[str] = None
    section_title: Optional[str] = None
    subquestion: Optional[str] = None
    equation_present: bool = False
    code_present: bool = False

class DocumentParser:
    @staticmethod
    def _is_code_or_assembly(line: str) -> bool:
        """Detects if a line looks like code, assembly, or hex values."""
        line = line.strip()
        # Assembly instructions and registers (e.g. MOV AX, CLR, RET, ORG, JMP)
        if re.search(r'\b(MOV|ADD|SUB|DIV|MUL|JMP|CMP|RET|ORG|CLR|PUSH|POP|INC|DEC)\b', line, re.IGNORECASE):
            return True
        # Hex values (e.g. 0000H, 0x1A)
        if re.search(r'\b[0-9A-Fa-f]+H\b|\b0x[0-9A-Fa-f]+\b', line):
            return True
        # Code labels (e.g. LOOP:)
        if re.match(r'^[A-Za-z0-9_]+:$', line):
            return True
        # Function calls (e.g. main())
        if re.search(r'\b[A-Za-z_][A-Za-z0-9_]*\(.*?\)', line):
            return True
        return False

    @staticmethod
    def _is_equation(line: str) -> bool:
        line = line.strip()
        # LaTeX math boundaries
        if re.search(r'\$[^$]+\$|\\\[.*\\\]', line):
            return True
        # Variable assignment/formula (e.g. y = (X3+X+1)/2)
        if re.search(r'\b[A-Za-z]\s*=\s*[0-9A-Za-z(]', line):
            return True
        # Common math operators heavily used
        if sum(line.count(op) for op in ['+', '=', '/', '*']) >= 2 and any(c.isalpha() for c in line) and any(c.isdigit() for c in line):
            return True
        return False

    @staticmethod
    def _extract_logical_units(text: str, current_page: int = 1) -> List[LogicalUnit]:
        """
        Structure-First Parsing:
        Splits text strictly by question/section boundary.
        Technical content (code/equations) are appended to the parent unit.
        """
        lines = text.split('\n')
        units = []
        current_unit_lines = []
        current_type = "text"
        
        current_q_num = None
        current_section = None
        current_subq = None
        has_equation = False
        has_code = False
        
        def push_unit():
            nonlocal current_unit_lines, current_type, has_equation, has_code
            if current_unit_lines:
                content = '\n'.join(current_unit_lines).strip()
                # Minimum semantic threshold - reject tiny fragments if not part of a larger block
                if len(content) > 10 or current_type != "text":
                    units.append(LogicalUnit(
                        type=current_type,
                        content=content,
                        page_num=current_page,
                        question_num=current_q_num,
                        section_title=current_section,
                        subquestion=current_subq,
                        equation_present=has_equation,
                        code_present=has_code
                    ))
                current_unit_lines = []
                has_equation = False
                has_code = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Detect boundaries
            is_boundary = False
            new_type = current_type
            
            # Detect Question Boundary (e.g. "4. ", "Q5: ", "5. a)")
            q_match = re.match(r'^(?:Q\s*)?(\d+)[.)]\s+(.*)', stripped)
            if q_match:
                is_boundary = True
                new_type = "question"
                current_q_num = q_match.group(1)
                current_subq = None
                
            # Detect Subquestion / MCQ Boundary (e.g. "a)", "A.") ONLY IF not code
            subq_match = re.match(r'^(\([a-zA-Z]\)|[a-zA-Z]\)|\([0-9]\)|[a-zA-Z]\.)\s(.*)', stripped)
            if subq_match and not DocumentParser._is_code_or_assembly(stripped):
                is_boundary = True
                new_type = "subquestion"
                current_subq = subq_match.group(1).replace('(', '').replace(')', '').replace('.', '')
                
            # Detect Section Header (Markdown or All Caps strictly if long enough and not assembly)
            section_match = re.match(r'^#{1,6}\s+(.*)', stripped)
            # Remove naive isupper() check to prevent breaking on uppercase assembly
            if section_match:
                is_boundary = True
                new_type = "section"
                current_section = stripped
                current_q_num = None  # Reset question under new section

            if is_boundary:
                push_unit()
                current_type = new_type
            
            # Check properties for metadata
            if DocumentParser._is_equation(stripped):
                has_equation = True
            if DocumentParser._is_code_or_assembly(stripped):
                has_code = True
                
            current_unit_lines.append(line)
            
        push_unit()
        return units

    @staticmethod
    def parse_to_logical_units(filepath: str) -> List[LogicalUnit]:
        """Reads file and returns a list of strict LogicalUnits."""
        raw_text = ""
        if filepath.lower().endswith('.pdf'):
            try:
                import fitz
                doc = fitz.open(filepath)
                units = []
                for i, page in enumerate(doc):
                    text = page.get_text("text")
                    units.extend(DocumentParser._extract_logical_units(text, current_page=i+1))
                
                print(f"--- PARSED {len(units)} LOGICAL UNITS ---")
                for u in units:
                    print(f"[Type: {u.type}] [Q: {u.question_num}] [Eq: {u.equation_present}] [Code: {u.code_present}] -> {u.content[:50]}...")
                
                return units
            except Exception as e:
                return [LogicalUnit(type="error", content=f"PDF parsing error: {e}")]
        else:
            try:
                import chardet
                with open(filepath, 'rb') as f:
                    raw_data = f.read()
                    encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
                    raw_text = raw_data.decode(encoding, errors='replace')
                return DocumentParser._extract_logical_units(raw_text, current_page=1)
            except Exception as e:
                return [LogicalUnit(type="error", content=f"Text parsing error: {e}")]
