from preprocessing.augmentation import SequenceModifier

def test_modifier_swap_start() -> None:
    alphabet = ['X']
    modifier = SequenceModifier(alphabet)

    for _ in range(100):
        seq = list('0123456789')
        aug_seq = seq[:]
        modifier._swap(aug_seq, 0)

        seq = "".join(seq)
        aug_seq = "".join(aug_seq)
        print(f"Swap start:")
        print(f"Before: {seq}")
        print(f"After: {aug_seq}")
        assert aug_seq in ('1023456789','0123456789')

def test_modifier_swap_mid() -> None:
    alphabet = ['X']
    modifier = SequenceModifier(alphabet)

    for _ in range(100):
        seq = list('0123456789')
        aug_seq = seq[:]
        modifier._swap(aug_seq, 4)
        seq = "".join(seq)
        aug_seq = "".join(aug_seq)
        print(f"Swap mid:")
        print(f"Before: {seq}")
        print(f"After: {aug_seq}")
        assert aug_seq in ('0124356789','0123546789')

def test_modifier_swap_end() -> None:
    alphabet = ['X']
    modifier = SequenceModifier(alphabet)
    
    for _ in range(100):
        seq = list('0123456789')
        aug_seq = seq[:]
        modifier._swap(aug_seq, 9)
        seq = "".join(seq)
        aug_seq = "".join(aug_seq)
        print(f"Swap end:")
        print(f"Before: {seq}")
        print(f"After: {aug_seq}")
        assert aug_seq in ('0123456798', '0123456789')