from preprocessing.padding import EndStrategy, FrontStrategy, RandomStrategy


def test_Frontpad_4():
    optimal_length = 4
    input_seq = [['123'], ['456'], ['789']]
    expected_output = [['PAD'],['123'], ['456'], ['789']]
    
    padding_strategy = FrontStrategy(optimal_length=optimal_length)
    output = padding_strategy.execute(input_seq)
    
    assert output == expected_output

def test_Frontpad_3():
    optimal_length = 3
    input_seq = [['123'], ['456'], ['789']]
    expected_output = [['123'], ['456'], ['789']]
    
    padding_strategy = FrontStrategy(optimal_length=optimal_length)
    output = padding_strategy.execute(input_seq)
    
    assert output == expected_output

def test_Frontpad_7():
    optimal_length = 7
    input_seq = [['123'], ['456'], ['789']]
    expected_output = [['PAD'], ['PAD'], ['PAD'], ['PAD'], ['123'], ['456'], ['789']]
    
    padding_strategy = FrontStrategy(optimal_length=optimal_length)
    output = padding_strategy.execute(input_seq)
    
    assert output == expected_output

def test_Frontpad_2():
    optimal_length = 2
    input_seq = [['123'], ['456'], ['789']]
    expected_output = [['123'], ['456'], ['789']]
    
    padding_strategy = FrontStrategy(optimal_length=optimal_length)
    output = padding_strategy.execute(input_seq)
    
    assert output == expected_output

def test_Endpad_4():
    optimal_length = 4
    input_seq = [['123'], ['456'], ['789']]
    expected_output = [['123'], ['456'], ['789'], ['PAD']]
    
    padding_strategy = EndStrategy(optimal_length=optimal_length)
    output = padding_strategy.execute(input_seq)
    
    assert output == expected_output

def test_Endpad_3():
    optimal_length = 3
    input_seq = [['123'], ['456'], ['789']]
    expected_output = [['123'], ['456'], ['789']]
    
    padding_strategy = EndStrategy(optimal_length=optimal_length)
    output = padding_strategy.execute(input_seq)
    
    assert output == expected_output

def test_Endpad_7():
    optimal_length = 7
    input_seq = [['123'], ['456'], ['789']]
    expected_output = [['123'], ['456'], ['789'], ['PAD'], ['PAD'], ['PAD'], ['PAD']]
    
    padding_strategy = EndStrategy(optimal_length=optimal_length)
    output = padding_strategy.execute(input_seq)
    
    assert output == expected_output

def test_Endpad_2():
    optimal_length = 2
    input_seq = [['123'], ['456'], ['789']]
    expected_output = [['123'], ['456'], ['789']]
    
    padding_strategy = EndStrategy(optimal_length=optimal_length)
    output = padding_strategy.execute(input_seq)
    
    assert output == expected_output

def test_frontpad_strategy():
    for optimal_length in [0,2,3,4,5,6,7,10,100,1000]:
        input_seq = [['value1'], ['value2'], ['value3']]
        
        padding_needed = max(0, optimal_length - len(input_seq))
        
        padding_strategy = FrontStrategy(optimal_length=optimal_length)
        output_seq = padding_strategy.execute(input_seq)
        
        # 1) Check that output has not been truncated
        assert len(output_seq) >= optimal_length, f"Output length {len(output_seq)} does not match optimal length {optimal_length}"
        
        # 2) Count number of ['PAD'] tokens in output
        actual_pad_count = sum(1 for item in output_seq if item == ['PAD'])
        assert actual_pad_count == padding_needed, f"Expected {padding_needed} padding tokens, but found {actual_pad_count}"
    
        # 3) Check for misplaced `['PAD']` tokens (all `['PAD']` should be at the beginning or end)
        # Find the first non-`['PAD']` index and the last non-`['PAD']` index
        non_pad_indices = [i for i, item in enumerate(output_seq) if item != ['PAD']]
        
        if non_pad_indices:
            first_non_pad = non_pad_indices[0]
            last_non_pad = non_pad_indices[-1]
            
            # Assert that all elements before `first_non_pad` and after `last_non_pad` are `['PAD']`
            for i in range(first_non_pad):
                assert output_seq[i] == ['PAD'], f"Misplaced `['PAD']` token at index {i}"
            for i in range(last_non_pad + 1, len(output_seq)):
                assert output_seq[i] == ['PAD'], f"Misplaced `['PAD']` token at index {i}"

def test_frontpad_strategy():
    for optimal_length in [0,2,3,4,5,6,7,10,100,1000]:
        input_seq = [['value1'], ['value2'], ['value3']]
        
        padding_needed = max(0, optimal_length - len(input_seq))
        
        padding_strategy = EndStrategy(optimal_length=optimal_length)
        output_seq = padding_strategy.execute(input_seq)
        
        # 1) Check that output has not been truncated
        assert len(output_seq) >= optimal_length, f"Output length {len(output_seq)} does not match optimal length {optimal_length}"
        
        # 2) Count number of ['PAD'] tokens in output
        actual_pad_count = sum(1 for item in output_seq if item == ['PAD'])
        assert actual_pad_count == padding_needed, f"Expected {padding_needed} padding tokens, but found {actual_pad_count}"
    
        # 3) Check for misplaced `['PAD']` tokens (all `['PAD']` should be at the beginning or end)
        # Find the first non-`['PAD']` index and the last non-`['PAD']` index
        non_pad_indices = [i for i, item in enumerate(output_seq) if item != ['PAD']]
        
        if non_pad_indices:
            first_non_pad = non_pad_indices[0]
            last_non_pad = non_pad_indices[-1]
            
            # Assert that all elements before `first_non_pad` and after `last_non_pad` are `['PAD']`
            for i in range(first_non_pad):
                assert output_seq[i] == ['PAD'], f"Misplaced `['PAD']` token at index {i}"
            for i in range(last_non_pad + 1, len(output_seq)):
                assert output_seq[i] == ['PAD'], f"Misplaced `['PAD']` token at index {i}"

def test_ramdompad_strategy():
    for optimal_length in [0,2,3,4,5,6,7,10,100,1000]:
        input_seq = [['value1'], ['value2'], ['value3']]
        
        padding_needed = max(0, optimal_length - len(input_seq))
        
        padding_strategy = RandomStrategy(optimal_length=optimal_length)
        output_seq = padding_strategy.execute(input_seq)
        
        # 1) Check that output has not been truncated
        assert len(output_seq) >= optimal_length, f"Output length {len(output_seq)} does not match optimal length {optimal_length}"
        
        # 2) Count number of ['PAD'] tokens in output
        actual_pad_count = sum(1 for item in output_seq if item == ['PAD'])
        assert actual_pad_count == padding_needed, f"Expected {padding_needed} padding tokens, but found {actual_pad_count}"
    
        # 3) Check for misplaced `['PAD']` tokens (all `['PAD']` should be at the beginning or end)
        # Find the first non-`['PAD']` index and the last non-`['PAD']` index
        non_pad_indices = [i for i, item in enumerate(output_seq) if item != ['PAD']]
        
        if non_pad_indices:
            first_non_pad = non_pad_indices[0]
            last_non_pad = non_pad_indices[-1]
            
            # Assert that all elements before `first_non_pad` and after `last_non_pad` are `['PAD']`
            for i in range(first_non_pad):
                assert output_seq[i] == ['PAD'], f"Misplaced `['PAD']` token at index {i}"
            for i in range(last_non_pad + 1, len(output_seq)):
                assert output_seq[i] == ['PAD'], f"Misplaced `['PAD']` token at index {i}"