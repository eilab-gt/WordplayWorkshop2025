"""Tests for the ScreenUI module."""
import pytest
import pandas as pd
from pathlib import Path
import openpyxl
from openpyxl.styles import PatternFill
from unittest.mock import patch, Mock

from src.lit_review.processing import ScreenUI


class TestScreenUI:
    """Test cases for ScreenUI class."""
    
    def test_init(self, sample_config):
        """Test ScreenUI initialization."""
        screen_ui = ScreenUI(sample_config)
        assert screen_ui.config is not None
        assert hasattr(screen_ui, 'screening_columns')
        assert hasattr(screen_ui, 'styles')
    
    def test_generate_sheet_basic(self, sample_config, sample_screening_df, temp_dir):
        """Test basic screening sheet generation."""
        screen_ui = ScreenUI(sample_config)
        
        output_path = Path(temp_dir) / 'test_screening.xlsx'
        result_path = screen_ui.generate_sheet(sample_screening_df, output_path)
        
        assert result_path == output_path
        assert output_path.exists()
        
        # Verify Excel structure
        wb = openpyxl.load_workbook(output_path)
        assert 'Screening' in wb.sheetnames
        
        ws = wb['Screening']
        # Check headers
        headers = [cell.value for cell in ws[1]]
        assert 'screening_id' in headers
        assert 'title' in headers
        assert 'include_ta' in headers
        assert 'include_ft' in headers
    
    def test_excel_formatting(self, sample_config, sample_screening_df, temp_dir):
        """Test Excel formatting and styling."""
        screen_ui = ScreenUI(sample_config)
        
        output_path = Path(temp_dir) / 'test_formatting.xlsx'
        screen_ui.generate_sheet(sample_screening_df, output_path)
        
        wb = openpyxl.load_workbook(output_path)
        ws = wb['Screening']
        
        # Check header formatting
        header_cell = ws['A1']
        assert header_cell.font.bold
        assert header_cell.fill.fill_type == 'solid'
        
        # Check column widths
        assert ws.column_dimensions['B'].width > 30  # Title column should be wide
        assert ws.column_dimensions['F'].width > 40  # Abstract column should be wide
        
        # Check data validation for include columns
        # Note: Actual data validation testing would require more complex checks
    
    def test_instructions_sheet(self, sample_config, sample_screening_df, temp_dir):
        """Test instructions sheet generation."""
        screen_ui = ScreenUI(sample_config)
        
        output_path = Path(temp_dir) / 'test_instructions.xlsx'
        screen_ui.generate_sheet(sample_screening_df, output_path)
        
        wb = openpyxl.load_workbook(output_path)
        assert 'Instructions' in wb.sheetnames
        
        ws = wb['Instructions']
        # Check that instructions exist
        instructions_found = False
        for row in ws.iter_rows():
            for cell in row:
                if cell.value and 'screening' in str(cell.value).lower():
                    instructions_found = True
                    break
        assert instructions_found
    
    def test_stats_sheet(self, sample_config, sample_screening_df, temp_dir):
        """Test statistics sheet generation."""
        screen_ui = ScreenUI(sample_config)
        
        output_path = Path(temp_dir) / 'test_stats.xlsx'
        screen_ui.generate_sheet(sample_screening_df, output_path)
        
        wb = openpyxl.load_workbook(output_path)
        assert 'Stats' in wb.sheetnames
        
        ws = wb['Stats']
        # Check for basic statistics
        stats_found = False
        for row in ws.iter_rows():
            for cell in row:
                if cell.value and 'total' in str(cell.value).lower():
                    stats_found = True
                    break
        assert stats_found
    
    def test_conditional_formatting(self, sample_config, temp_dir):
        """Test conditional formatting for include/exclude cells."""
        screen_ui = ScreenUI(sample_config)
        
        # Create DataFrame with various include/exclude values
        df = pd.DataFrame({
            'screening_id': ['SCREEN_0001', 'SCREEN_0002', 'SCREEN_0003'],
            'title': ['Paper 1', 'Paper 2', 'Paper 3'],
            'include_ta': ['yes', 'no', ''],
            'include_ft': ['', 'yes', 'no']
        })
        
        output_path = Path(temp_dir) / 'test_conditional.xlsx'
        screen_ui.generate_sheet(df, output_path)
        
        wb = openpyxl.load_workbook(output_path)
        ws = wb['Screening']
        
        # The actual conditional formatting rules are applied, 
        # but testing them requires checking the worksheet's conditional_formatting attribute
        assert len(ws.conditional_formatting) > 0
    
    def test_load_progress(self, sample_config, temp_dir):
        """Test loading screening progress from Excel."""
        screen_ui = ScreenUI(sample_config)
        
        # First generate a sheet
        df = pd.DataFrame({
            'screening_id': ['SCREEN_0001', 'SCREEN_0002'],
            'title': ['Paper 1', 'Paper 2'],
            'include_ta': ['', ''],
            'include_ft': ['', '']
        })
        
        excel_path = Path(temp_dir) / 'test_load.xlsx'
        screen_ui.generate_sheet(df, excel_path)
        
        # Modify the Excel file to add screening decisions
        wb = openpyxl.load_workbook(excel_path)
        ws = wb['Screening']
        
        # Find column indices
        headers = [cell.value for cell in ws[1]]
        include_ta_col = headers.index('include_ta') + 1
        reason_ta_col = headers.index('reason_ta') + 1
        
        # Add screening decisions
        ws.cell(row=2, column=include_ta_col, value='yes')
        ws.cell(row=3, column=include_ta_col, value='no')
        ws.cell(row=3, column=reason_ta_col, value='E1')
        
        wb.save(excel_path)
        
        # Load progress
        progress_df = screen_ui.load_progress(excel_path)
        
        assert isinstance(progress_df, pd.DataFrame)
        assert len(progress_df) == 2
        assert progress_df.loc[0, 'include_ta'] == 'yes'
        assert progress_df.loc[1, 'include_ta'] == 'no'
        assert progress_df.loc[1, 'reason_ta'] == 'E1'
    
    def test_empty_dataframe(self, sample_config, temp_dir):
        """Test handling of empty DataFrame."""
        screen_ui = ScreenUI(sample_config)
        
        empty_df = pd.DataFrame()
        output_path = Path(temp_dir) / 'test_empty.xlsx'
        
        # Should handle empty DataFrame gracefully
        result_path = screen_ui.generate_sheet(empty_df, output_path)
        assert result_path == output_path
        assert output_path.exists()
    
    def test_large_abstract_handling(self, sample_config, temp_dir):
        """Test handling of very long abstracts."""
        screen_ui = ScreenUI(sample_config)
        
        # Create DataFrame with very long abstract
        long_abstract = "This is a very long abstract. " * 100
        df = pd.DataFrame({
            'screening_id': ['SCREEN_0001'],
            'title': ['Paper with Long Abstract'],
            'abstract': [long_abstract],
            'include_ta': ['']
        })
        
        output_path = Path(temp_dir) / 'test_long_abstract.xlsx'
        screen_ui.generate_sheet(df, output_path)
        
        wb = openpyxl.load_workbook(output_path)
        ws = wb['Screening']
        
        # Check that abstract is included (possibly truncated)
        abstract_col = [cell.value for cell in ws[1]].index('abstract') + 1
        abstract_value = ws.cell(row=2, column=abstract_col).value
        assert abstract_value is not None
        assert len(str(abstract_value)) > 0
    
    def test_missing_columns_handling(self, sample_config, temp_dir):
        """Test handling of DataFrames with missing expected columns."""
        screen_ui = ScreenUI(sample_config)
        
        # Create DataFrame missing some expected columns
        df = pd.DataFrame({
            'screening_id': ['SCREEN_0001'],
            'title': ['Test Paper']
            # Missing many expected columns
        })
        
        output_path = Path(temp_dir) / 'test_missing_cols.xlsx'
        
        # Should handle missing columns gracefully
        result_path = screen_ui.generate_sheet(df, output_path)
        assert result_path == output_path
        assert output_path.exists()
        
        # Verify that missing columns are added
        wb = openpyxl.load_workbook(output_path)
        ws = wb['Screening']
        headers = [cell.value for cell in ws[1]]
        assert 'include_ta' in headers
        assert 'reason_ta' in headers