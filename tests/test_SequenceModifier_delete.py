from preprocessing.augmentation import SequenceModifier

def test_SequenceModifier_delete_start() -> None:
    alphabet = ['X']
    modifier = SequenceModifier(alphabet)

    for _ in range(100):
        seq = list('0123456789')
        aug_seq = seq[:]
        modifier._delete(aug_seq, 0)

        seq = "".join(seq)
        aug_seq = "".join(aug_seq)
        print(f"Delete start:")
        print(f"Before: {seq}")
        print(f"After: {aug_seq}")
        assert aug_seq in ('123456789')

def test_SequenceModifier_delete_mid() -> None:
    alphabet = ['X']
    modifier = SequenceModifier(alphabet)

    for _ in range(100):
        seq = list('0123456789')
        aug_seq = seq[:]
        modifier._delete(aug_seq, 4)
        seq = "".join(seq)
        aug_seq = "".join(aug_seq)
        print(f"Delete mid:")
        print(f"Before: {seq}")
        print(f"After: {aug_seq}")
        assert aug_seq in ('012356789')

def test_SequenceModifier_delete_end() -> None:
    alphabet = ['X']
    modifier = SequenceModifier(alphabet)
    
    for _ in range(100):
        seq = list('0123456789')
        aug_seq = seq[:]
        modifier._delete(aug_seq, 9)
        seq = "".join(seq)
        aug_seq = "".join(aug_seq)
        print(f"Delete end:")
        print(f"Before: {seq}")
        print(f"After: {aug_seq}")
        assert aug_seq in ('012345678')