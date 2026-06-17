"""
Tests for nanochat.tokenizer module - BPE tokenizer implementations.

Run: python -m pytest tests/test_tokenizer.py -v

Note: These tests use tiktoken's built-in GPT-2 encoding as a reference
tokenizer, avoiding the need for trained tokenizer files on disk.
"""

import os
import tempfile
import pytest

from nanochat.tokenizer import (
    SPECIAL_TOKENS,
    SPLIT_PATTERN,
    RustBPETokenizer,
    HuggingFaceTokenizer,
)


class TestSpecialTokens:
    """Test SPECIAL_TOKENS configuration."""

    def test_bos_in_special_tokens(self):
        assert "<|bos|>" in SPECIAL_TOKENS

    def test_user_tokens_present(self):
        assert "<|user_start|>" in SPECIAL_TOKENS
        assert "<|user_end|>" in SPECIAL_TOKENS

    def test_assistant_tokens_present(self):
        assert "<|assistant_start|>" in SPECIAL_TOKENS
        assert "<|assistant_end|>" in SPECIAL_TOKENS

    def test_python_tokens_present(self):
        assert "<|python_start|>" in SPECIAL_TOKENS
        assert "<|python_end|>" in SPECIAL_TOKENS

    def test_output_tokens_present(self):
        assert "<|output_start|>" in SPECIAL_TOKENS
        assert "<|output_end|>" in SPECIAL_TOKENS

    def test_total_count(self):
        assert len(SPECIAL_TOKENS) == 9

    def test_split_pattern_is_string(self):
        assert isinstance(SPLIT_PATTERN, str)
        assert len(SPLIT_PATTERN) > 0


class TestRustBPETokenizerFromPretrained:
    """Test RustBPETokenizer using tiktoken's built-in GPT-2 encoding."""

    @pytest.fixture
    def tokenizer(self):
        return RustBPETokenizer.from_pretrained("gpt2")

    def test_vocab_size(self, tokenizer):
        # GPT-2 has 50257 tokens
        assert tokenizer.get_vocab_size() == 50257

    def test_encode_simple(self, tokenizer):
        ids = tokenizer.encode("hello")
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_encode_decode_roundtrip(self, tokenizer):
        text = "Hello, world!"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_encode_with_prepend(self, tokenizer):
        ids_no_prepend = tokenizer.encode("test")
        ids_with_prepend = tokenizer.encode("test", prepend=42)
        assert ids_with_prepend[0] == 42
        assert ids_with_prepend[1:] == ids_no_prepend

    def test_encode_with_append(self, tokenizer):
        ids_no_append = tokenizer.encode("test")
        ids_with_append = tokenizer.encode("test", append=99)
        assert ids_with_append[-1] == 99
        assert ids_with_append[:-1] == ids_no_append

    def test_encode_batch(self, tokenizer):
        texts = ["hello", "world", "test"]
        result = tokenizer.encode(texts)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(r, list) for r in result)

    def test_encode_batch_with_prepend(self, tokenizer):
        texts = ["hello", "world"]
        result = tokenizer.encode(texts, prepend=7)
        for ids in result:
            assert ids[0] == 7

    def test_get_bos_token_id(self, tokenizer):
        bos = tokenizer.get_bos_token_id()
        assert isinstance(bos, int)
        assert bos >= 0

    def test_encode_special(self, tokenizer):
        # GPT-2 uses <|endoftext|> as BOS
        token_id = tokenizer.encode_special("<|endoftext|>")
        assert isinstance(token_id, int)
        assert token_id == 50256  # known GPT-2 token

    def test_callable(self, tokenizer):
        # Test __call__ works same as encode
        ids1 = tokenizer.encode("test text")
        ids2 = tokenizer("test text")
        assert ids1 == ids2

    def test_get_special_tokens(self, tokenizer):
        special = tokenizer.get_special_tokens()
        assert isinstance(special, set)
        assert "<|endoftext|>" in special

    def test_id_to_token(self, tokenizer):
        # Token 0 in GPT-2 is "!"
        token_str = tokenizer.id_to_token(0)
        assert isinstance(token_str, str)
        assert len(token_str) > 0

    def test_encode_empty_string(self, tokenizer):
        ids = tokenizer.encode("")
        assert ids == []

    def test_encode_unicode(self, tokenizer):
        ids = tokenizer.encode("café résumé")
        assert len(ids) > 0
        decoded = tokenizer.decode(ids)
        assert decoded == "café résumé"

    def test_save_creates_pickle(self, tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer.save(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "tokenizer.pkl"))


class TestHuggingFaceTokenizerFromPretrained:
    """Test HuggingFaceTokenizer using HuggingFace's built-in GPT-2."""

    @pytest.fixture
    def tokenizer(self):
        return HuggingFaceTokenizer.from_pretrained("gpt2")

    def test_vocab_size(self, tokenizer):
        size = tokenizer.get_vocab_size()
        assert isinstance(size, int)
        assert size > 0

    def test_encode_simple(self, tokenizer):
        ids = tokenizer.encode("hello world")
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_encode_with_prepend_int(self, tokenizer):
        ids = tokenizer.encode("test", prepend=5)
        assert ids[0] == 5

    def test_encode_with_append_int(self, tokenizer):
        ids = tokenizer.encode("test", append=10)
        assert ids[-1] == 10

    def test_encode_batch(self, tokenizer):
        texts = ["hello", "world"]
        result = tokenizer.encode(texts)
        assert len(result) == 2
        assert all(isinstance(r, list) for r in result)

    def test_decode(self, tokenizer):
        text = "Hello world"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert "Hello world" in decoded

    def test_get_bos_token_id(self, tokenizer):
        bos = tokenizer.get_bos_token_id()
        assert isinstance(bos, int)

    def test_callable(self, tokenizer):
        ids1 = tokenizer.encode("test")
        ids2 = tokenizer("test")
        assert ids1 == ids2

    def test_save_and_load(self, tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer.save(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "tokenizer.json"))
            loaded = HuggingFaceTokenizer.from_directory(tmpdir)
            assert tokenizer.encode("test") == loaded.encode("test")
