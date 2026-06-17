"""
Tests for nanochat.execution module - sandboxed Python code execution.

Run: python -m pytest tests/test_execution.py -v
"""

import pytest
from nanochat.execution import (
    ExecutionResult,
    execute_code,
    WriteOnlyStringIO,
    chdir,
)


class TestExecutionResult:
    """Test the ExecutionResult dataclass."""

    def test_success_repr(self):
        r = ExecutionResult(success=True, stdout="hello\n", stderr="")
        assert "success=True" in repr(r)
        assert "hello" in repr(r)

    def test_failure_repr(self):
        r = ExecutionResult(success=False, stdout="", stderr="", error="NameError: x")
        assert "success=False" in repr(r)
        assert "NameError" in repr(r)

    def test_timeout_repr(self):
        r = ExecutionResult(success=False, stdout="", stderr="", timeout=True, error="Timed out")
        assert "timeout=True" in repr(r)

    def test_memory_repr(self):
        r = ExecutionResult(success=False, stdout="", stderr="", memory_exceeded=True, error="OOM")
        assert "memory_exceeded=True" in repr(r)


class TestExecuteCode:
    """Test the execute_code function."""

    def test_simple_print(self):
        result = execute_code("print('hello world')")
        assert result.success is True
        assert result.stdout == "hello world\n"
        assert result.stderr == ""
        assert result.error is None

    def test_arithmetic(self):
        result = execute_code("print(2 + 2)")
        assert result.success is True
        assert result.stdout.strip() == "4"

    def test_multiline_code(self):
        code = """
x = 10
y = 20
print(x + y)
"""
        result = execute_code(code)
        assert result.success is True
        assert result.stdout.strip() == "30"

    def test_syntax_error(self):
        result = execute_code("def foo(")
        assert result.success is False
        assert result.error is not None
        assert "SyntaxError" in result.error

    def test_runtime_error(self):
        result = execute_code("1/0")
        assert result.success is False
        assert result.error is not None
        assert "ZeroDivisionError" in result.error

    def test_name_error(self):
        result = execute_code("print(undefined_variable)")
        assert result.success is False
        assert "NameError" in result.error

    def test_timeout(self):
        code = "import time; time.sleep(100)"
        result = execute_code(code, timeout=1.0)
        assert result.success is False
        assert result.timeout is True

    def test_import_standard_library(self):
        code = "import math; print(math.pi)"
        result = execute_code(code)
        assert result.success is True
        assert "3.14159" in result.stdout

    def test_stderr_capture(self):
        code = "import sys; print('error msg', file=sys.stderr)"
        result = execute_code(code)
        assert result.success is True
        assert "error msg" in result.stderr

    def test_empty_code(self):
        result = execute_code("")
        assert result.success is True
        assert result.stdout == ""

    def test_exception_with_traceback(self):
        code = """
def foo():
    raise ValueError("test error")
foo()
"""
        result = execute_code(code)
        assert result.success is False
        assert "ValueError" in result.error

    def test_list_comprehension(self):
        code = "print([x**2 for x in range(5)])"
        result = execute_code(code)
        assert result.success is True
        assert "[0, 1, 4, 9, 16]" in result.stdout

    def test_class_definition(self):
        code = """
class Foo:
    def __init__(self, x):
        self.x = x
    def bar(self):
        return self.x * 2

f = Foo(21)
print(f.bar())
"""
        result = execute_code(code)
        assert result.success is True
        assert "42" in result.stdout


class TestWriteOnlyStringIO:
    """Test the WriteOnlyStringIO class."""

    def test_write_works(self):
        sio = WriteOnlyStringIO()
        sio.write("hello")
        # getvalue still works for internal use
        assert sio.getvalue() == "hello"

    def test_read_raises(self):
        sio = WriteOnlyStringIO()
        with pytest.raises(IOError):
            sio.read()

    def test_readline_raises(self):
        sio = WriteOnlyStringIO()
        with pytest.raises(IOError):
            sio.readline()

    def test_readlines_raises(self):
        sio = WriteOnlyStringIO()
        with pytest.raises(IOError):
            sio.readlines()

    def test_not_readable(self):
        sio = WriteOnlyStringIO()
        assert sio.readable() is False


class TestChdir:
    """Test the chdir context manager."""

    def test_chdir_dot_is_noop(self):
        import os
        cwd_before = os.getcwd()
        with chdir("."):
            assert os.getcwd() == cwd_before
        assert os.getcwd() == cwd_before

    def test_chdir_changes_and_restores(self):
        import os
        import tempfile
        cwd_before = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            with chdir(tmpdir):
                assert os.getcwd() == tmpdir
            assert os.getcwd() == cwd_before
