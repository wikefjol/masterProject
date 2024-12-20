from preprocessing.augmentation import SequenceModifier

def test_modifier_insert_start() -> None:
    alphabet = ['X']
    modifier = SequenceModifier(alphabet)

    for _ in range(100):
        seq = list('0123456789')
        aug_seq = seq[:]
        modifier._insert(aug_seq, 0)

        seq = "".join(seq)
        aug_seq = "".join(aug_seq)
        print(f"Insert start:")
        print(f"Before: {seq}")
        print(f"After: {aug_seq}")
        assert aug_seq in ('X0123456789', '0X123456789')

def test_modifier_insert_mid() -> None:
    alphabet = ['X']
    modifier = SequenceModifier(alphabet)

    for _ in range(100):
        seq = list('0123456789')
        aug_seq = seq[:]
        modifier._insert(aug_seq, 4)
        seq = "".join(seq)
        aug_seq = "".join(aug_seq)
        print(f"Insert mid:")
        print(f"Before: {seq}")
        print(f"After: {aug_seq}")
        assert aug_seq in ('0123X456789', '01234X56789')

def test_modifier_insert_end() -> None:
    alphabet = ['X']
    modifier = SequenceModifier(alphabet)
    
    for _ in range(100):
        seq = list('0123456789')
        aug_seq = seq[:]
        modifier._insert(aug_seq, 9)
        seq = "".join(seq)
        aug_seq = "".join(aug_seq)
        print(f"Insert end:")
        print(f"Before: {seq}")
        print(f"After: {aug_seq}")
        assert aug_seq in ('012345678X9', '0123456789X')