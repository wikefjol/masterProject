from preprocessing.truncation import FrontStrategy, EndStrategy, SlidingwindowStrategy
def test_end_truncation_3():
    optimal_length = 3
    input_seq = [['PAD'], ['123'], ['456'], ['789']]
    expected_output = [['PAD'], ['123'], ['456']]
    
    truncator = EndStrategy(optimal_length)
    output = truncator.execute(input_seq)
    assert output == expected_output

def test_end_truncation_1():
    optimal_length = 1
    input_seq = [['PAD'], ['123'], ['456'], ['789']]
    expected_output = [['PAD']]
    
    truncator = EndStrategy(optimal_length)
    output = truncator.execute(input_seq)
    assert output == expected_output

def test_end_truncation_equal_length():
    optimal_length = 4
    input_seq = [['PAD'], ['123'], ['456'], ['789']]
    expected_output = [['PAD'], ['123'], ['456'], ['789']]
    
    truncator = EndStrategy(optimal_length)
    output = truncator.execute(input_seq)
    assert output == expected_output

def test_front_truncation_3():
    optimal_length = 3
    input_seq = [['PAD'], ['123'], ['456'], ['789']]
    expected_output = [['123'], ['456'], ['789']]
    
    truncator = FrontStrategy(optimal_length)
    output = truncator.execute(input_seq)
    assert output == expected_output

def test_front_truncation_1():
    optimal_length = 1
    input_seq = [['PAD'], ['123'], ['456'], ['789']]
    expected_output = [['789']]
    
    truncator = FrontStrategy(optimal_length)
    output = truncator.execute(input_seq)
    assert output == expected_output

def test_front_truncation_equal_length():
    optimal_length = 4
    input_seq = [['PAD'], ['123'], ['456'], ['789']]
    expected_output = [['PAD'], ['123'], ['456'], ['789']]
    
    truncator = FrontStrategy(optimal_length)
    output = truncator.execute(input_seq)
    assert output == expected_output


def test_slidingwindow_truncation_lengths():
    input_seq = [['PAD'], ['123'], ['456'], ['789'], ['123'], ['456'], ['789'], ['123'], ['456'], ['789'], ['123'], ['456'], ['789'], ['123'], ['456'], ['789'], ['123'], ['456'], ['789'], ['123'], ['456'], ['789']]

    for optimal_length in range(100):
        truncation_strategy = SlidingwindowStrategy(optimal_length)
        expected_length = min(optimal_length, len(input_seq))
        padded_seq = truncation_strategy.execute(input_seq)
        assert len(padded_seq)==expected_length

