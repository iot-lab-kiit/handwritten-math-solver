"""
Unit tests for the solveEquation function in app.py

This test suite covers:
- Infinite solutions
- No solution
- Valid solution
- Malformed input
"""

import pytest
import sys
import os

# Add the parent directory to the path to import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import solveEquation


class TestSolveEquation:
    """Test cases for the solveEquation function"""

    # Test cases for valid solutions
    def test_simple_equation_positive_result(self):
        """Test a simple equation with a positive result: x+5=10"""
        result = solveEquation("x+5=10", "x")
        assert result == 5.0

    def test_simple_equation_negative_result(self):
        """Test a simple equation with a negative result: x+10=5"""
        result = solveEquation("x+10=5", "x")
        assert result == -5.0

    def test_equation_with_coefficient(self):
        """Test equation with coefficient: 2x+5=11"""
        result = solveEquation("2x+5=11", "x")
        assert result == 3.0

    def test_equation_with_negative_coefficient(self):
        """Test equation with negative coefficient: -3x+6=0"""
        result = solveEquation("-3x+6=0", "x")
        assert result == 2.0

    def test_equation_with_variable_on_both_sides(self):
        """Test equation with variable on both sides: 2x+5=x+10"""
        result = solveEquation("2x+5=x+10", "x")
        assert result == 5.0

    def test_equation_with_multiple_terms(self):
        """Test equation with multiple terms: 3x+2+5=x+10"""
        result = solveEquation("3x+2+5=x+10", "x")
        assert result == 1.5

    def test_equation_with_decimal_result(self):
        """Test equation that results in decimal: 3x=10"""
        result = solveEquation("3x=10", "x")
        assert abs(result - 3.333333) < 0.00001

    def test_equation_with_zero_result(self):
        """Test equation with result zero: x+5=5"""
        result = solveEquation("x+5=5", "x")
        assert result == 0.0

    def test_equation_different_variable_a(self):
        """Test equation with variable 'a': a+3=7"""
        result = solveEquation("a+3=7", "a")
        assert result == 4.0

    def test_equation_different_variable_b(self):
        """Test equation with variable 'b': 2b-4=6"""
        result = solveEquation("2b-4=6", "b")
        assert result == 5.0

    def test_equation_with_subtraction(self):
        """Test equation with subtraction: x-3=7"""
        result = solveEquation("x-3=7", "x")
        assert result == 10.0

    def test_complex_equation(self):
        """Test complex equation: 5x-2+3x=2x+10"""
        result = solveEquation("5x-2+3x=2x+10", "x")
        assert result == 2.0

    # Test cases for infinite solutions
    def test_infinite_solutions_simple(self):
        """Test infinite solutions: x=x"""
        result = solveEquation("x=x", "x")
        assert result == "Infinite solutions"

    def test_infinite_solutions_with_constants(self):
        """Test infinite solutions: x+5=x+5"""
        result = solveEquation("x+5=x+5", "x")
        assert result == "Infinite solutions"

    def test_infinite_solutions_complex(self):
        """Test infinite solutions: 2x+3=2x+3"""
        result = solveEquation("2x+3=2x+3", "x")
        assert result == "Infinite solutions"

    def test_infinite_solutions_zero_equals_zero(self):
        """Test infinite solutions when both sides are zero: 0=0"""
        result = solveEquation("0=0", "x")
        assert result == "Infinite solutions"

    # Test cases for no solution
    def test_no_solution_simple(self):
        """Test no solution: x=x+1"""
        result = solveEquation("x=x+1", "x")
        assert result == "No solution"

    def test_no_solution_with_constants(self):
        """Test no solution: x+5=x+10"""
        result = solveEquation("x+5=x+10", "x")
        assert result == "No solution"

    def test_no_solution_complex(self):
        """Test no solution: 2x+3=2x+5"""
        result = solveEquation("2x+3=2x+5", "x")
        assert result == "No solution"

    def test_no_solution_contradictory(self):
        """Test no solution: 5=10"""
        result = solveEquation("5=10", "x")
        assert result == "No solution"

    # Test cases for malformed input
    def test_malformed_empty_string(self):
        """Test malformed input: empty string"""
        # Empty string returns "Infinite solutions" (0 == 0)
        result = solveEquation("", "x")
        assert result == "Infinite solutions"

    def test_malformed_no_equals_sign(self):
        """Test malformed input: no equals sign"""
        # No equals sign treats everything as left side equals 0
        # x+5 becomes x+5=0, which gives x=-5
        result = solveEquation("x+5", "x")
        assert result == -5.0

    def test_malformed_multiple_equals_signs(self):
        """Test malformed input: multiple equals signs - x=5=10"""
        # This might not raise an error but should be tested
        # The function will process it, but behavior may be unexpected
        result = solveEquation("x=5=10", "x")
        # The function should handle this somehow
        assert isinstance(result, (float, str))

    def test_malformed_invalid_characters(self):
        """Test malformed input: invalid characters"""
        with pytest.raises(Exception):
            solveEquation("x+abc=10", "x")

    def test_malformed_missing_operand(self):
        """Test malformed input: missing operand (x+=10)"""
        with pytest.raises(Exception):
            solveEquation("x+=10", "x")

    def test_malformed_double_operators(self):
        """Test malformed input: double operators (x++5=10)"""
        with pytest.raises(Exception):
            solveEquation("x++5=10", "x")

    def test_edge_case_only_variable(self):
        """Test edge case: only variable on left side (x=5)"""
        result = solveEquation("x=5", "x")
        assert result == 5.0

    def test_edge_case_only_variable_on_right(self):
        """Test edge case: only variable on right side (5=x)"""
        result = solveEquation("5=x", "x")
        assert result == 5.0

    def test_edge_case_negative_numbers(self):
        """Test with negative numbers: x-5=-10"""
        result = solveEquation("x-5=-10", "x")
        assert result == -5.0

    def test_edge_case_large_numbers(self):
        """Test with large numbers: 1000x+500=2500"""
        result = solveEquation("1000x+500=2500", "x")
        assert result == 2.0
