import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List

model_name = "iiiorg/piiranha-v1-detect-personal-information"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def _apply_redaction_segment(masked_text: List[str], start: int, end: int, pii_type: str, aggregate_redaction: bool):
    for j in range(start, end):
        if 0 <= j < len(masked_text):
            masked_text[j] = ''
    
    if 0 <= start < len(masked_text):
        if aggregate_redaction:
            masked_text[start] = '[redacted]'
        else:
            masked_text[start] = f'[{pii_type}]'
    elif start == len(masked_text):
        if aggregate_redaction:
            masked_text.append('[redacted]')
        else:
            masked_text.append(f'[{pii_type}]')

def mask_pii(text: str, aggregate_redaction: bool = True) -> str:
    if not text:
        return ""

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=-1)

    encoded_inputs = tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=True)
    offset_mapping = encoded_inputs['offset_mapping']

    masked_text_chars = list(text)

    is_redacting = False
    redaction_start = 0
    current_pii_type = ''

    for i, (char_start, char_end) in enumerate(offset_mapping):
        if char_start == char_end and i != 0:
            continue
        
        if i >= len(predictions[0]):
            break

        label_id = predictions[0][i].item()
        label_name = model.config.id2label[label_id]

        if label_name != 'O':
            pii_type = label_name
            
            if not is_redacting:
                is_redacting = True
                redaction_start = char_start
                current_pii_type = pii_type
            elif not aggregate_redaction and pii_type != current_pii_type:
                _apply_redaction_segment(masked_text_chars, redaction_start, char_start, current_pii_type, aggregate_redaction)
                redaction_start = char_start
                current_pii_type = pii_type
        else:
            if is_redacting:
                _apply_redaction_segment(masked_text_chars, redaction_start, char_end, current_pii_type, aggregate_redaction)
                is_redacting = False
                current_pii_type = ''

    if is_redacting:
        _apply_redaction_segment(masked_text_chars, redaction_start, len(masked_text_chars), current_pii_type, aggregate_redaction)
    
    final_text = ''.join(filter(None, masked_text_chars))
    
    final_text = final_text.replace(' ]', ']').replace('[ ', '[').replace('  ', ' ')
    
    return final_text

# Example string for testing
if __name__ == "__main__":
    test_string = "My name is John Doe, and my email is john.doe@example.com. My phone number is 123-456-7890. I live at 123 Main St, Anytown, USA."

    # Run the function with the example string (aggregated redaction)
    print("--- Aggregated Redaction Example ---")
    redacted_text_aggregated = mask_pii(test_string, aggregate_redaction=True)
    print(f"Original: {test_string}")
    print(f"Redacted: {redacted_text_aggregated}")

    print("\n--- Detailed Redaction Example ---")
    # Run the function with the example string (detailed redaction)
    redacted_text_detailed = mask_pii(test_string, aggregate_redaction=False)
    print(f"Original: {test_string}")
    print(f"Redacted: {redacted_text_detailed}")