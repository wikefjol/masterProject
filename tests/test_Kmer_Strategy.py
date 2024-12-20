from preprocessing.tokenization import KmerStrategy

def test_kmer_strategy():
    # Case 1: k=3, sequence length requires padding of 2 characters
    k_strat = KmerStrategy(k=3, padding_alphabet=['N'])
    test_sequence = list('0123456789')
    expected_output = [['012'], ['345'], ['678'], ['9NN']]
    output = k_strat.execute(test_sequence)
    assert output == expected_output, f"Failed for k=3, expected {expected_output}, but got {output}"

    # Case 2: k=2, sequence length requires no padding
    k_strat = KmerStrategy(k=2, padding_alphabet=['X'])
    test_sequence = list('01234567')
    expected_output = [['01'], ['23'], ['45'], ['67']]
    output = k_strat.execute(test_sequence)
    assert output == expected_output, f"Failed for k=2, expected {expected_output}, but got {output}"

    # Case 3: k=4, sequence length requires padding of 1 character
    k_strat = KmerStrategy(k=4, padding_alphabet=['P'])
    test_sequence = list('0123456')
    expected_output = [['0123'], ['456P']]
    output = k_strat.execute(test_sequence)
    assert output == expected_output, f"Failed for k=4, expected {expected_output}, but got {output}"

    # Case 4: k=5, sequence length requires padding of 4 characters
    k_strat = KmerStrategy(k=5, padding_alphabet=['Z'])
    test_sequence = list('ABCDE')
    expected_output = [['ABCDE']]
    output = k_strat.execute(test_sequence)
    assert output == expected_output, f"Failed for k=5, expected {expected_output}, but got {output}"

    # Case 5: k=3, single character sequence with padding needed
    k_strat = KmerStrategy(k=3, padding_alphabet=['Y'])
    test_sequence = list('X')
    expected_output = [['XYY']]
    output = k_strat.execute(test_sequence)
    assert output == expected_output, f"Failed for k=3 with single character, expected {expected_output}, but got {output}"

def test_ensure_divisible_by_k()->None:
   # Case: Standard padding needed
    k_strat = KmerStrategy(k=3, padding_alphabet=['N'])
    test_sequence = list('0123456789')
    expected_output = list('0123456789NN')
    output_sequence = k_strat._make_divisible_by_k(test_sequence)
    assert output_sequence == expected_output

    # Case: k=1, no padding needed
    k_strat = KmerStrategy(k=1, padding_alphabet=['N'])
    test_sequence = list('0123456789')
    expected_output = list('0123456789')
    output_sequence = k_strat._make_divisible_by_k(test_sequence)
    assert output_sequence == expected_output

    # Case: k greater than sequence length
    k_strat = KmerStrategy(k=11, padding_alphabet=['N'])
    test_sequence = list('0123456789')
    expected_output = list('0123456789N')
    output_sequence = k_strat._make_divisible_by_k(test_sequence)
    assert output_sequence == expected_output

    # Edge Case: Empty sequence
    k_strat = KmerStrategy(k=3, padding_alphabet=['N'])
    test_sequence = []
    expected_output = []
    output_sequence = k_strat._make_divisible_by_k(test_sequence)
    assert output_sequence == expected_output

    # Edge Case: k greater than sequence length, small sequence
    k_strat = KmerStrategy(k=5, padding_alphabet=['N'])
    test_sequence = list('012')
    expected_output = list('012NN')
    output_sequence = k_strat._make_divisible_by_k(test_sequence)
    assert output_sequence == expected_output

    # Edge Case: Exact divisibility
    k_strat = KmerStrategy(k=3, padding_alphabet=['N'])
    test_sequence = list('012345')
    expected_output = list('012345')
    output_sequence = k_strat._make_divisible_by_k(test_sequence)
    assert output_sequence == expected_output

