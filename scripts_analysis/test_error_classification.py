#!/usr/bin/env python3
"""
Test script to verify error classification logic
"""

def classify_error_type(stimulus_type, correct_answer, extracted_answer, identity_value, flanker_value, stimulus_row):
    """
    Classify the error type based on the response.
    """
    # Handle empty responses
    if extracted_answer == -1:
        return 'Empty'
    
    # Correct response - chose the correct identity position
    if correct_answer == extracted_answer:
        return 'Corr'
    
    # Get the value at the chosen position
    if 1 <= extracted_answer <= len(stimulus_row):
        chosen_value = stimulus_row[extracted_answer - 1]  # Convert to 0-based indexing
    else:
        return 'Otr'  # Invalid position
    
    # Determine error type based on the value that was chosen
    if chosen_value == identity_value:
        # Chose identity value but not correct answer - Simon error
        return 'SmErr'
    elif chosen_value == flanker_value:
        # Chose flanker value - Flanker error
        return 'FkErr'
    else:
        # Chose something else entirely
        return 'Otr'


def test_error_classification():
    """Test the error classification with example data"""
    
    # Example from the JSON data: "3 2 2"
    # correct_answer: 1, identity: 3, flanker: 2
    stimulus_row = [3, 2, 2]
    identity_value = 3
    flanker_value = 2
    correct_answer = 1
    
    print("Testing stimulus row: [3, 2, 2]")
    print(f"Identity value: {identity_value}, Flanker value: {flanker_value}")
    print(f"Correct answer position: {correct_answer}")
    print()
    
    test_cases = [
        (1, "Should be Corr - chose position 1 (value 3, identity)"),
        (2, "Should be FkErr - chose position 2 (value 2, flanker)"),
        (3, "Should be FkErr - chose position 3 (value 2, flanker)"),
    ]
    
    for extracted_answer, description in test_cases:
        error_type = classify_error_type(3, correct_answer, extracted_answer, 
                                       identity_value, flanker_value, stimulus_row)
        chosen_value = stimulus_row[extracted_answer - 1]
        print(f"Extracted answer: {extracted_answer} (value: {chosen_value}) -> {error_type}")
        print(f"  {description}")
        print()
    
    # Test case where identity appears in multiple positions
    print("Testing stimulus row with identity in wrong position: [2, 3, 2]")
    stimulus_row2 = [2, 3, 2]
    identity_value2 = 3
    flanker_value2 = 2
    correct_answer2 = 2  # Identity is at position 2
    
    test_cases2 = [
        (1, "Should be FkErr - chose position 1 (value 2, flanker)"),
        (2, "Should be Corr - chose position 2 (value 3, identity, correct)"),
        (3, "Should be FkErr - chose position 3 (value 2, flanker)"),
    ]
    
    for extracted_answer, description in test_cases2:
        error_type = classify_error_type(3, correct_answer2, extracted_answer, 
                                       identity_value2, flanker_value2, stimulus_row2)
        chosen_value = stimulus_row2[extracted_answer - 1]
        print(f"Extracted answer: {extracted_answer} (value: {chosen_value}) -> {error_type}")
        print(f"  {description}")
        print()
    
    # Test SmErr case - identity appears in wrong position
    print("Testing SmErr case: [3, 2, 3]")
    stimulus_row3 = [3, 2, 3]
    identity_value3 = 3
    flanker_value3 = 2
    correct_answer3 = 1  # Correct identity is at position 1
    
    test_cases3 = [
        (1, "Should be Corr - chose position 1 (value 3, identity, correct)"),
        (2, "Should be FkErr - chose position 2 (value 2, flanker)"),
        (3, "Should be SmErr - chose position 3 (value 3, identity but wrong position)"),
    ]
    
    for extracted_answer, description in test_cases3:
        error_type = classify_error_type(3, correct_answer3, extracted_answer, 
                                       identity_value3, flanker_value3, stimulus_row3)
        chosen_value = stimulus_row3[extracted_answer - 1]
        print(f"Extracted answer: {extracted_answer} (value: {chosen_value}) -> {error_type}")
        print(f"  {description}")
        print()


if __name__ == "__main__":
    test_error_classification()
