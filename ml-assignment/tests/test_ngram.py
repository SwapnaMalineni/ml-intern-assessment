import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.ngram_model import TrigramModel

@pytest.fixture
def model():
    """Provides a TrigramModel instance for each test."""
    return TrigramModel()

def test_fit_creates_correct_trigram_counts(model):
    """
    Tests if the fit method correctly processes text and builds the trigram_counts dictionary.
    """
    text = "I am a test."
    model.fit(text)
    
    # Expected tokens: ['<s>', '<s>', 'i', 'am', 'a', 'test', '<e>']
    expected_counts = {
        ('<s>', '<s>'): {'i': 1},
        ('<s>', 'i'): {'am': 1},
        ('i', 'am'): {'a': 1},
        ('am', 'a'): {'test': 1},
        ('a', 'test'): {'<e>': 1}
    }
    assert model.trigram_counts == expected_counts

def test_text_cleaning(model):
    """
    Tests the text cleaning logic for lowercasing and punctuation removal.
    """
    text = "I'm a TEST sentence! Another..."
    model.fit(text)
    # Expected tokens: ['<s>', '<s>', "i'm", 'a', 'test', 'sentence', 'another', '<e>']
    assert ('test', 'sentence') in model.trigram_counts
    assert ('sentence', 'another') in model.trigram_counts
    assert "i'm" in model.trigram_counts[('<s>', '<s>')]

def test_generate_returns_string(model):
    text = "I am a test sentence. This is another test sentence."
    model.fit(text)
    generated_text = model.generate(max_length=10)
    assert isinstance(generated_text, str)
    # With a simple corpus, generation should not be empty
    assert len(generated_text) > 0

def test_generate_on_empty_text(model):
    model.fit("")
    generated_text = model.generate()
    assert generated_text == ""

def test_generate_on_untrained_model(model):
    """Generation on a model that has not been fit should return an empty string."""
    generated_text = model.generate()
    assert generated_text == ""
    model = TrigramModel()
    text = "I am a test sentence. This is another test sentence."
    model.fit(text)
    generated_text = model.generate()
    assert isinstance(generated_text, str)
    assert len(generated_text.split()) > 0

def test_empty_text():
    model = TrigramModel()
    text = ""
    model.fit(text)
    generated_text = model.generate()
    assert generated_text == ""

def test_short_text():
    model = TrigramModel()
    text = "I am."
    model.fit(text)
    generated_text = model.generate()
    assert isinstance(generated_text, str)
