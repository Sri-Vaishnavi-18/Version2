import torch

from contract_lens.models import BiLSTMClassifier, Vocabulary


def test_vocabulary_reserves_padding_unknown_and_encodes_empty_text():
    vocabulary = Vocabulary.build(["privacy data data", "policy data"])

    assert vocabulary.token_to_id["<PAD>"] == 0
    assert vocabulary.token_to_id["<UNK>"] == 1
    assert vocabulary.encode("") == [1]
    assert vocabulary.encode("not-in-vocabulary")[0] == 1


def test_attention_and_plain_bilstm_return_batch_logits():
    token_ids = torch.tensor([[2, 3, 4], [5, 6, 0]], dtype=torch.long)
    lengths = torch.tensor([3, 2], dtype=torch.long)

    for use_attention in (False, True):
        model = BiLSTMClassifier(
            vocab_size=8,
            num_classes=7,
            embedding_dim=12,
            hidden_dim=6,
            use_attention=use_attention,
        )
        logits = model(token_ids, lengths)
        assert logits.shape == (2, 7)
        assert torch.isfinite(logits).all()
