#!/usr/bin/env python3
"""Fix Python 3.9 compatibility issues by converting union syntax to Optional/Union."""

import re
from pathlib import Path
from typing import List, Optional, Tuple


def fix_union_syntax(content: str) -> str:
    """Convert Python 3.10+ union syntax to Python 3.9 compatible syntax."""
    # First ensure imports are correct
    has_typing_import = "from typing import" in content
    needs_optional = False
    needs_union = False
    
    # Pattern to match union syntax: Optional[type]
    optional_pattern = r'(\w+(?:\[[\w\[\], ]+\])?)\s*\|\s*None'
    # Pattern to match union of types: type1 | type2
    union_pattern = r'(\w+(?:\[[\w\[\], ]+\])?)\s*\|\s*(\w+(?:\[[\w\[\], ]+\])?)'
    
    # Check if we need Optional or Union
    if re.search(optional_pattern, content):
        needs_optional = True
    if re.search(union_pattern, content) and not re.search(optional_pattern, content):
        needs_union = True
    
    # Replace Optional[type] with Optional[type]
    def replace_optional(match):
        return f"Optional[{match.group(1)}]"
    
    content = re.sub(optional_pattern, replace_optional, content)
    
    # Replace type1 | type2 with Union[type1, type2] (but not when type2 is None)
    def replace_union(match):
        type1, type2 = match.group(1), match.group(2)
        if type2 == "None":
            return f"Optional[{type1}]"
        return f"Union[{type1}, {type2}]"
    
    # Only replace unions that aren't already handled by Optional
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if '|' in line and 'None' not in line and not line.strip().startswith('#'):
            # This might be a union that needs fixing
            line = re.sub(r'(\w+(?:\[[\w\[\], ]+\])?)\s*\|\s*(\w+(?:\[[\w\[\], ]+\])?)', 
                         replace_union, line)
        new_lines.append(line)
    content = '\n'.join(new_lines)
    
    # Update imports if needed
    if has_typing_import and (needs_optional or needs_union):
        import_line_match = re.search(r'from typing import List, Optional, Tuple
        if import_line_match:
            imports = import_line_match.group(1).split(', ')
            if needs_optional and 'Optional' not in imports:
                imports.append('Optional')
            if needs_union and 'Union' not in imports:
                imports.append('Union')
            # Sort imports for consistency
            imports = sorted(set(imports))
            new_import_line = f"from typing import List, Optional, Tuple
            content = re.sub(r'from typing import List, Optional, Tuple
    
    return content


def process_file(file_path: Path) -> Tuple[bool, str]:
    """Process a single file and return (changed, message)."""
    try:
        content = file_path.read_text()
        original_content = content
        
        # Skip if no union syntax found
        if '|' not in content:
            return False, f"Skipped {file_path}: No union syntax found"
        
        # Apply fixes
        content = fix_union_syntax(content)
        
        if content != original_content:
            file_path.write_text(content)
            return True, f"Fixed {file_path}"
        else:
            return False, f"No changes needed in {file_path}"
    
    except Exception as e:
        return False, f"Error processing {file_path}: {e}"


def main():
    """Fix Python 3.9 compatibility issues in all Python files."""
    src_dir = Path(__file__).parent.parent / "src"
    scripts_dir = Path(__file__).parent
    tests_dir = Path(__file__).parent.parent / "tests"
    
    # Find all Python files
    python_files: List[Path] = []
    for directory in [src_dir, scripts_dir, tests_dir]:
        if directory.exists():
            python_files.extend(directory.rglob("*.py"))
    
    print(f"Found {len(python_files)} Python files to check")
    
    changed_files = 0
    errors = 0
    
    for file_path in python_files:
        changed, message = process_file(file_path)
        print(message)
        if changed:
            changed_files += 1
        if message.startswith("Error"):
            errors += 1
    
    print(f"\nSummary:")
    print(f"  Files checked: {len(python_files)}")
    print(f"  Files changed: {changed_files}")
    print(f"  Errors: {errors}")
    
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    exit(main())