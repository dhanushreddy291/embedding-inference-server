import torch

from server import (
    CHUNK_OVERLAP_WORDS,
    MAX_CHUNK_WORDS,
    aggregate_embeddings,
    chunk_text,
    embed_text,
)


class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        text = "hello world"
        chunks = chunk_text(text)
        assert chunks == ["hello world"]

    def test_empty_text_returns_single_empty_chunk(self):
        chunks = chunk_text("")
        assert chunks == [""]

    def test_exact_max_words_returns_single_chunk(self):
        text = " ".join([f"w{i}" for i in range(MAX_CHUNK_WORDS)])
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_produces_multiple_chunks(self):
        word_count = MAX_CHUNK_WORDS * 3
        text = " ".join([f"w{i}" for i in range(word_count)])
        chunks = chunk_text(text)
        assert len(chunks) > 1

    def test_chunk_count_formula(self):
        word_count = 2500
        text = " ".join([f"w{i}" for i in range(word_count)])
        chunks = chunk_text(text)
        step = MAX_CHUNK_WORDS - CHUNK_OVERLAP_WORDS
        expected = 1 + (word_count - MAX_CHUNK_WORDS + step - 1) // step
        assert len(chunks) == expected

    def test_first_chunk_has_correct_length(self):
        text = " ".join([f"w{i}" for i in range(3000)])
        chunks = chunk_text(text)
        first_word_count = len(chunks[0].split())
        assert first_word_count == MAX_CHUNK_WORDS

    def test_overlap_between_consecutive_chunks(self):
        text = " ".join([f"w{i}" for i in range(3000)])
        chunks = chunk_text(text)

        for i in range(len(chunks) - 1):
            words_current = chunks[i].split()
            words_next = chunks[i + 1].split()
            overlap = words_current[-CHUNK_OVERLAP_WORDS:]
            expected_start = words_next[:CHUNK_OVERLAP_WORDS]
            assert overlap == expected_start

    def test_all_original_words_preserved(self):
        word_count = 2500
        original_words = [f"w{i}" for i in range(word_count)]
        text = " ".join(original_words)
        chunks = chunk_text(text)

        all_chunk_words = []
        for i, chunk in enumerate(chunks):
            words = chunk.split()
            if i == 0:
                all_chunk_words.extend(words)
            else:
                all_chunk_words.extend(words[CHUNK_OVERLAP_WORDS:])

        assert all_chunk_words == original_words

    def test_custom_chunk_size(self):
        text = " ".join([f"w{i}" for i in range(100)])
        chunks = chunk_text(text, max_words=30, overlap_words=10)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.split()) <= 30


class TestAggregateEmbeddings:
    def test_single_embedding_returns_normalized(self):
        embeddings = torch.tensor([[3.0, 4.0]])
        result = aggregate_embeddings(embeddings)
        norm = sum(x**2 for x in result) ** 0.5
        assert abs(norm - 1.0) < 1e-5

    def test_multiple_embeddings_returns_mean_normalized(self):
        e1 = torch.tensor([[1.0, 0.0, 0.0]])
        e2 = torch.tensor([[0.0, 1.0, 0.0]])
        embeddings = torch.cat([e1, e2], dim=0)
        result = aggregate_embeddings(embeddings)

        norm = sum(x**2 for x in result) ** 0.5
        assert abs(norm - 1.0) < 1e-5

        expected_component = (1.0 / (2**0.5))
        assert abs(result[0] - expected_component) < 1e-5
        assert abs(result[1] - expected_component) < 1e-5
        assert abs(result[2]) < 1e-5

    def test_identical_embeddings_return_same_direction(self):
        e = torch.tensor([[1.0, 2.0, 3.0]])
        embeddings = e.repeat(5, 1)
        result = aggregate_embeddings(embeddings)

        norm = sum(x**2 for x in result) ** 0.5
        assert abs(norm - 1.0) < 1e-5

        original_norm = (1 + 4 + 9) ** 0.5
        assert abs(result[0] - 1.0 / original_norm) < 1e-5
        assert abs(result[1] - 2.0 / original_norm) < 1e-5
        assert abs(result[2] - 3.0 / original_norm) < 1e-5

    def test_zero_vector_handling(self):
        embeddings = torch.zeros(3, 10)
        result = aggregate_embeddings(embeddings)
        assert all(x == 0.0 for x in result)

    def test_output_length_matches_embedding_dim(self):
        dim = 128
        embeddings = torch.randn(4, dim)
        result = aggregate_embeddings(embeddings)
        assert len(result) == dim


class TestEmbedText:
    def test_short_text_no_metadata(self, monkeypatch):
        import server

        dummy_tensor = torch.randn(1, 10)

        class MockModel:
            def encode_document(self, texts):
                return dummy_tensor

            def encode_query(self, texts):
                return dummy_tensor

        monkeypatch.setattr(server, "model", MockModel())

        embedding, chunk_count, original_length = embed_text("hello world", "document")
        assert chunk_count is None
        assert original_length is None
        assert isinstance(embedding, list)
        assert len(embedding) == 10

    def test_long_text_has_metadata(self, monkeypatch):
        import server

        text = " ".join([f"word{i}" for i in range(3000)])
        expected_chunks = chunk_text(text)
        n_chunks = len(expected_chunks)

        dummy_tensor = torch.randn(n_chunks, 10)

        class MockModel:
            def encode_document(self, texts):
                assert len(texts) == n_chunks
                return dummy_tensor

            def encode_query(self, texts):
                return dummy_tensor

        monkeypatch.setattr(server, "model", MockModel())

        embedding, chunk_count, original_length = embed_text(text, "document")
        assert chunk_count == n_chunks
        assert original_length == 3000
        assert isinstance(embedding, list)
        assert len(embedding) == 10

    def test_embedding_is_normalized(self, monkeypatch):
        import server

        dummy_tensor = torch.ones(1, 10) * 100

        class MockModel:
            def encode_document(self, texts):
                return dummy_tensor

        monkeypatch.setattr(server, "model", MockModel())

        embedding, _, _ = embed_text("test", "document")
        norm = sum(x**2 for x in embedding) ** 0.5
        assert abs(norm - 1.0) < 1e-5

    def test_query_task_type_uses_encode_query(self, monkeypatch):
        import server

        dummy_tensor = torch.randn(1, 10)
        called_with = []

        class MockModel:
            def encode_query(self, texts):
                called_with.append("query")
                return dummy_tensor

            def encode_document(self, texts):
                called_with.append("document")
                return dummy_tensor

        monkeypatch.setattr(server, "model", MockModel())

        embed_text("hello", "query")
        assert called_with == ["query"]
