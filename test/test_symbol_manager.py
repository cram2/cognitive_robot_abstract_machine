import pytest
import numpy as np
from unittest.mock import Mock
from typing import Tuple

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.god_map import god_map
from semantic_digital_twin.spatial_types.symbol_manager import SymbolManager, symbol_manager


class TestSymbolManager:

    def test_singleton_behavior(self):
        """Test that SymbolManager follows singleton pattern"""
        manager1 = SymbolManager()
        manager2 = SymbolManager()
        assert manager1 is manager2
        assert manager1 is symbol_manager

    def test_init(self):
        """Test proper initialization of SymbolManager"""
        manager = SymbolManager()
        assert hasattr(manager, 'symbol_to_provider')
        assert isinstance(manager.symbol_to_provider, dict)
        # assert hasattr(manager, 'time')
        # assert isinstance(manager.time, cas.Symbol)

    def test_register_symbol_provider_with_callable(self):
        """Test registering symbols with callable providers"""
        manager = SymbolManager()

        def provider():
            return 42.0

        symbol = manager.register_symbol_provider('test_symbol', provider)

        assert isinstance(symbol, cas.Symbol)
        assert symbol in manager.symbol_to_provider
        assert manager.symbol_to_provider[symbol] == provider

    def test_register_symbol_provider_with_float(self):
        """Test registering symbols with float values"""
        manager = SymbolManager()

        symbol = manager.register_symbol_provider('test_float', 3.14)

        assert isinstance(symbol, cas.Symbol)
        assert symbol in manager.symbol_to_provider
        assert manager.symbol_to_provider[symbol] == 3.14

    def test_register_point3(self):
        """Test registering 3D points"""
        manager = SymbolManager()

        def point_provider() -> Tuple[float, float, float]:
            return (1.0, 2.0, 3.0)

        point = manager.register_point3('test_point', point_provider)

        assert isinstance(point, cas.Point3)
        # Check that x, y, z symbols were registered
        x_symbol = None
        y_symbol = None
        z_symbol = None

        for symbol in manager.symbol_to_provider.keys():
            if hasattr(symbol, 'name'):
                if symbol.name == 'test_point.x':
                    x_symbol = symbol
                elif symbol.name == 'test_point.y':
                    y_symbol = symbol
                elif symbol.name == 'test_point.z':
                    z_symbol = symbol

        assert x_symbol is not None
        assert y_symbol is not None
        assert z_symbol is not None

        # Test that providers return correct values
        assert manager.symbol_to_provider[x_symbol]() == 1.0
        assert manager.symbol_to_provider[y_symbol]() == 2.0
        assert manager.symbol_to_provider[z_symbol]() == 3.0

    def test_register_vector3(self):
        """Test registering 3D vectors"""
        manager = SymbolManager()

        def vector_provider() -> Tuple[float, float, float]:
            return (4.0, 5.0, 6.0)

        vector = manager.register_vector3('test_vector', vector_provider)

        assert isinstance(vector, cas.Vector3)
        # Similar checks as for point3
        symbols_found = 0
        for symbol in manager.symbol_to_provider.keys():
            if hasattr(symbol, 'name'):
                name = symbol.name
                if name in ['test_vector.x', 'test_vector.y', 'test_vector.z']:
                    symbols_found += 1

        assert symbols_found == 3

    def test_register_quaternion(self):
        """Test registering quaternions"""
        manager = SymbolManager()

        def quat_provider() -> Tuple[float, float, float, float]:
            return (0.0, 0.0, 0.0, 1.0)

        quaternion = manager.register_quaternion('test_quat', quat_provider)

        assert isinstance(quaternion, cas.Quaternion)
        # Check that x, y, z, w symbols were registered
        symbols_found = 0
        for symbol in manager.symbol_to_provider.keys():
            if hasattr(symbol, 'name'):
                name = symbol.name
                if name in ['test_quat.x', 'test_quat.y', 'test_quat.z', 'test_quat.w']:
                    symbols_found += 1

        assert symbols_found == 4

    def test_register_transformation_matrix(self):
        """Test registering transformation matrices"""
        manager = SymbolManager()

        def matrix_provider():
            return (
                (1.0, 0.0, 0.0, 0.1),
                (0.0, 1.0, 0.0, 0.2),
                (0.0, 0.0, 1.0, 0.3)
            )

        trans_matrix = manager.register_transformation_matrix('test_matrix', matrix_provider)

        assert isinstance(trans_matrix, cas.TransformationMatrix)
        # Check that 12 symbols were registered (3x4 matrix)
        matrix_symbols = 0
        for symbol in manager.symbol_to_provider.keys():
            if hasattr(symbol, 'name'):
                name = symbol.name
                if 'test_matrix[' in name:
                    matrix_symbols += 1

        assert matrix_symbols == 12

    def test_resolve_symbols_single_list(self):
        """Test resolving a single list of symbols"""
        manager = SymbolManager()

        symbol1 = manager.register_symbol_provider('test1', lambda: 1.0)
        symbol2 = manager.register_symbol_provider('test2', lambda: 2.0)

        result = manager.resolve_symbols([symbol1, symbol2])

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0]))

    def test_resolve_symbols_nested_list(self):
        """Test resolving nested lists of symbols"""
        manager = SymbolManager()

        symbol1 = manager.register_symbol_provider('test1', lambda: 1.0)
        symbol2 = manager.register_symbol_provider('test2', lambda: 2.0)
        symbol3 = manager.register_symbol_provider('test3', lambda: 3.0)

        result = manager.resolve_symbols([[symbol1, symbol2], [symbol3]])

        assert isinstance(result, list)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(result[1], np.array([3.0]))

    def test_resolve_symbols_empty_list(self):
        """Test resolving empty symbol lists"""
        manager = SymbolManager()

        result = manager.resolve_symbols([])

        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_resolve_symbols_missing_symbol(self):
        """Test error handling for unregistered symbols"""
        manager = SymbolManager()

        # Create a symbol that's not registered
        unregistered_symbol = cas.Symbol('unregistered')

        with pytest.raises(KeyError, match='Cannot resolve'):
            manager.resolve_symbols([unregistered_symbol])

    def test_evaluate_expr_with_number(self):
        """Test evaluating numeric expressions"""
        manager = SymbolManager()

        result_int = manager.evaluate_expr(42)
        result_float = manager.evaluate_expr(3.14)

        assert result_int == 42
        assert result_float == 3.14

    def test_evaluate_expr_with_symbols(self):
        """Test evaluating symbolic expressions"""
        manager = SymbolManager()

        # Create a mock expression
        mock_expr = Mock()
        mock_compile = Mock()
        mock_compile.symbol_parameters = []
        mock_compile.fast_call.return_value = [5.0]
        mock_expr.compile.return_value = mock_compile
        mock_expr.to_np.return_value = 5.0

        result = manager.evaluate_expr(mock_expr)

        assert result == 5.0

    def test_evaluate_expr_no_params(self):
        """Test evaluating expressions without parameters"""
        manager = SymbolManager()

        # Create a mock expression with no parameters
        mock_expr = Mock()
        mock_compile = Mock()
        mock_compile.symbol_parameters = []
        mock_expr.compile.return_value = mock_compile
        mock_expr.to_np.return_value = 10.0

        result = manager.evaluate_expr(mock_expr)

        assert result == 10.0
        mock_expr.to_np.assert_called_once()

    def test_evaluate_expr_multiple_results(self):
        """Test evaluating expressions that return multiple values"""
        manager = SymbolManager()

        # Create a mock expression that returns multiple values
        expr = cas.Expression([1.0, 2.0, 3.0])

        result = manager.evaluate_expr(expr)

        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_provider_exception_handling(self):
        """Test that provider exceptions are properly handled"""
        manager = SymbolManager()

        def failing_provider():
            raise ValueError("Provider failed")

        symbol = manager.register_symbol_provider('failing', failing_provider)

        with pytest.raises(KeyError, match='Cannot resolve'):
            manager.resolve_symbols([symbol])

    def test_symbol_name_consistency(self):
        """Test that symbol names are consistent with registration names"""
        manager = SymbolManager()

        symbol = manager.register_symbol_provider('test_name', lambda: 1.0)

        # Note: This test assumes the symbol has a name() method
        # The actual implementation may vary based on the cas.Symbol class
        if hasattr(symbol, 'name'):
            assert symbol.name == 'test_name'