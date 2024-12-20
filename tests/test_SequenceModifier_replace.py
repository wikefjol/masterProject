from preprocessing.augmentation import SequenceModifier

def test_modifier_replace_start() -> None:
    alphabet = ['X']
    modifier = SequenceModifier(alphabet)

    for _ in range(100):
        seq = list('0123456789')
        aug_seq = seq[:]
        modifier._replace(aug_seq, 0)

        seq = "".join(seq)
        aug_seq = "".join(aug_seq)
        print(f"Replace start:")
        print(f"Before: {seq}")
        print(f"After: {aug_seq}")
        assert aug_seq in ('X123456789')

def test_modifier_replace_mid() -> None:
    alphabet = ['X']
    modifier = SequenceModifier(alphabet)

    for _ in range(100):
        seq = list('0123456789')
        aug_seq = seq[:]
        modifier._replace(aug_seq, 4)
        seq = "".join(seq)
        aug_seq = "".join(aug_seq)
        print(f"Replace mid:")
        print(f"Before: {seq}")
        print(f"After: {aug_seq}")
        assert aug_seq in ('0123X56789')

def test_modifier_replace_end() -> None:
    alphabet = ['X']
    modifier = SequenceModifier(alphabet)
    
    for _ in range(100):
        seq = list('0123456789')
        aug_seq = seq[:]
        modifier._replace(aug_seq, 9)
        seq = "".join(seq)
        aug_seq = "".join(aug_seq)
        print(f"Replace end:")
        print(f"Before: {seq}")
        print(f"After: {aug_seq}")
        assert aug_seq in ('012345678X')