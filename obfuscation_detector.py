import re
import base64
import binascii
import urllib.parse
import string
import unicodedata
import codecs
from typing import Dict, List, Optional

class ObfuscationFlagDetector:
    """
    Enhanced detector that flags suspicious obfuscation patterns based on:
    1. A dynamically scaled number of consecutive encoded words.
    2. A dynamically scaled number of words using the same encoding technique.
    3. Entire text obfuscation detection (e.g., full base64, hex, binary strings).
    Only considers words with 3 or more alphabetic characters for obfuscation detection.
    """

    def __init__(self):
        # Patterns for quick detection
        self.binary_pattern = re.compile(r'^[01\s]+$')
        self.hex_pattern = re.compile(r'^[0-9a-fA-F\s]+$')
        self.base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
        self.url_encoded_pattern = re.compile(r'%[0-9a-fA-F]{2}')
        self.unicode_escape_pattern = re.compile(r'\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8}')

        # Leetspeak mapping for detection
        self.leet_chars = set('0134578@!$+|()<>')

        # Character swap detection (Cyrillic, Greek look-alikes)
        self.suspicious_chars = set('аеорсхуАВЕКМНОРСТУХαβγδεζηθικλμνοπρστυφχψω')

        # Invisible characters
        self.invisible_chars = set([
            '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
            '\u00a0', '\u2000', '\u2001', '\u2002', '\u2003',
            '\u2004', '\u2005', '\u2006', '\u2007', '\u2008',
            '\u2009', '\u200a'
        ])

    def _is_printable_bytes(self, data: bytes) -> bool:
        """
        Helper function: Checks if a byte string contains mostly printable ASCII characters.
        This helps to distinguish between meaningfully decoded data and random bytes.
        """
        # Define a range of common printable ASCII characters (32 to 126)
        # and common whitespace characters (9-13: tab, newline, return)
        printable_chars = set(range(32, 127)) | set(range(9, 14))

        if not data:
            return False

        # Count printable characters
        printable_count = sum(1 for byte in data if byte in printable_chars)

        # Consider it printable if a high percentage of characters are printable
        # A threshold of 85% is a good starting point, adjust if needed
        return printable_count / len(data) > 0.85

    def detect_whole_text_obfuscation(self, text: str) -> Optional[str]:
        """
        Detects if the entire text is obfuscated using a single encoding method.
        Returns the encoding type if detected, None otherwise.
        """
        # Remove leading/trailing whitespace
        clean_text = text.strip()
        
        # Skip if text is too short to be meaningful obfuscation
        if len(clean_text) < 10:
            return None
            
        # Remove internal whitespace for some checks
        no_spaces = re.sub(r'\s+', '', clean_text)
        
        # Check for full Base64 encoding
        if self.base64_pattern.match(no_spaces) and len(no_spaces) >= 16:
            # Must be multiple of 4 for valid base64
            if len(no_spaces) % 4 == 0:
                try:
                    decoded_bytes = base64.b64decode(no_spaces, validate=True)
                    if self._is_printable_bytes(decoded_bytes):
                        return 'whole_text_base64'
                except (binascii.Error, ValueError):
                    pass
        
        # Check for full Hex encoding
        if self.hex_pattern.match(no_spaces) and len(no_spaces) >= 16:
            # Must be even length for valid hex
            if len(no_spaces) % 2 == 0:
                try:
                    decoded_bytes = bytes.fromhex(no_spaces)
                    if self._is_printable_bytes(decoded_bytes):
                        return 'whole_text_hex'
                except ValueError:
                    pass
        
        # Check for full Binary encoding
        if self.binary_pattern.match(clean_text) and len(no_spaces) >= 32:
            # Must be multiple of 8 for valid binary
            if len(no_spaces) % 8 == 0:
                try:
                    # Convert binary to bytes
                    binary_chunks = [no_spaces[i:i+8] for i in range(0, len(no_spaces), 8)]
                    decoded_bytes = bytes([int(chunk, 2) for chunk in binary_chunks])
                    if self._is_printable_bytes(decoded_bytes):
                        return 'whole_text_binary'
                except ValueError:
                    pass
        
        # Check for full URL encoding (high percentage of % encoded chars)
        url_encoded_count = len(self.url_encoded_pattern.findall(clean_text))
        total_chars = len(clean_text.replace(' ', ''))
        if url_encoded_count > 0 and (url_encoded_count * 3) / total_chars > 0.3:  # 30% or more is URL encoded
            try:
                decoded_url = urllib.parse.unquote(clean_text)
                if decoded_url != clean_text and len(decoded_url) > 5:
                    return 'whole_text_url_encoded'
            except Exception:
                pass
        
        # Check for full Unicode escape encoding
        unicode_escape_count = len(self.unicode_escape_pattern.findall(clean_text))
        if unicode_escape_count >= 3:  # At least 3 unicode escapes
            try:
                decoded_unicode = codecs.decode(clean_text, 'unicode_escape')
                if decoded_unicode != clean_text and len(decoded_unicode) > 5:
                    return 'whole_text_unicode_escapes'
            except Exception:
                pass
        
        # Check for high concentration of invisible characters
        invisible_count = sum(1 for char in clean_text if char in self.invisible_chars)
        if invisible_count > len(clean_text) * 0.1:  # More than 10% invisible chars
            return 'whole_text_invisible_chars'
        
        # Check for high concentration of suspicious characters (Cyrillic/Greek)
        suspicious_count = sum(1 for char in clean_text if char in self.suspicious_chars)
        if suspicious_count > len(clean_text) * 0.3:  # More than 30% suspicious chars
            return 'whole_text_char_swaps'
        
        # Check if entire text appears to be reversed (simple heuristic)
        if len(clean_text) > 20:  # Only for longer texts
            reversed_text = clean_text[::-1]
            # Look for patterns suggesting this might be reversed English
            # This is a simple heuristic - check for common reversed endings
            if any(reversed_text.lower().startswith(ending) for ending in ['gn', 'tn', 'dn', 'de', 'se', 'er']):
                # Additional check: see if reversed version has more common English patterns
                return 'whole_text_reversed'
        
        return None

    def count_alphabetic_chars(self, word: str) -> int:
        """Count alphabetic characters in a word."""
        return sum(1 for c in word if c.isalpha())

    def quick_detect_encoding_type(self, word: str) -> Optional[str]:
        """
        Quickly determine if a word uses a specific encoding technique.
        Returns the encoding type or None if no encoding detected.
        """
        # Clean word for analysis (remove punctuation from edges)
        clean_word = word.strip(string.punctuation)

        # Only consider words with 3 or more total characters for quick detection
        if not clean_word or len(clean_word) < 3:
            return None

        # Binary detection
        binary_clean = re.sub(r'\s', '', clean_word)
        if (self.binary_pattern.match(binary_clean) and
            len(binary_clean) % 8 == 0 and
            len(binary_clean) >= 8): # Ensure a minimum length for meaningful binary
            return 'binary'

        # Base64 detection
        if (self.base64_pattern.match(clean_word) and
            len(clean_word) >= 4 and
            len(clean_word) % 4 == 0): # Base64 encoded strings are usually multiples of 4
            try:
                # Quick validation - try to decode
                decoded_bytes = base64.b64decode(clean_word, validate=True)
                # Add check for printable characters in decoded Base64 to avoid false positives
                if self._is_printable_bytes(decoded_bytes):
                    return 'base64'
            except binascii.Error:
                pass
            except ValueError:
                pass

        # Hex detection
        hex_clean = re.sub(r'[\s:,-]', '', clean_word) # Clean common separators
        if (self.hex_pattern.match(hex_clean) and
            len(hex_clean) % 2 == 0 and # Hex strings are usually even length (two chars per byte)
            len(hex_clean) >= 8): # Ensure a minimum length for meaningful hex
            try:
                decoded_bytes = bytes.fromhex(hex_clean)
                if self._is_printable_bytes(decoded_bytes): # Add printable check for hex too
                    return 'hex'
            except ValueError:
                pass

        # URL encoding detection
        if self.url_encoded_pattern.search(clean_word):
            try:
                decoded_url = urllib.parse.unquote(clean_word)
                # If decoded content changes significantly and isn't just random % chars
                # and contains at least some alphanumeric characters after decoding.
                if decoded_url != clean_word and any(c.isalnum() for c in decoded_url):
                    return 'url_encoded'
            except Exception:
                pass

        # Unicode escape detection
        if self.unicode_escape_pattern.search(clean_word):
            try:
                decoded_unicode = codecs.decode(clean_word, 'unicode_escape')
                # Check if decoding actually changed something and the result is mostly printable
                if decoded_unicode != clean_word and self._is_printable_bytes(decoded_unicode.encode('utf-8', errors='ignore')):
                    return 'unicode_escapes'
            except Exception:
                pass

        # Leetspeak detection (if contains leet chars and original word has alphabetic content)
        if any(char in self.leet_chars for char in clean_word) and any(c.isalpha() for c in clean_word):
            return 'leetspeak'

        # Invisible characters detection
        if any(char in self.invisible_chars for char in clean_word):
            return 'invisible_chars'

        # Character swaps detection (Cyrillic/Greek look-alikes) - simple heuristic
        if any(char in self.suspicious_chars for char in clean_word):
            # This is a weak heuristic, but flags potential look-alike attacks.
            return 'char_swaps'

        # Reversed text detection (heuristic - check if it looks like reversed English)
        # This is a very simple and prone-to-false-positives heuristic.
        reversed_word = clean_word[::-1]
        if len(clean_word) > 5: # Only check longer words for reversal heuristic
            # Look for common English word endings as reversed beginnings, e.g., "gn" (from -ing), "tn" (from -nt)
            if reversed_word.lower().startswith(('gn', 'tn', 'dn', 'rg', 'rt', 'rd', 'yc', 'yt', 'elp')):
                return 'reversed'

        # Excessive spacing detection (e.g., "w o r d")
        if ' ' in clean_word and len(clean_word.replace(' ', '')) >= 3:
            return 'spaced_text'

        # Alternating case detection (e.g., "HeLlO")
        if (clean_word != clean_word.lower() and
            clean_word != clean_word.upper() and
            clean_word != clean_word.title() and # Exclude common title case
            len(clean_word) > 3): # Only check longer words for this pattern
            # Check for frequent case changes as an indicator
            case_changes = sum(1 for i in range(1, len(clean_word))
                               if clean_word[i].isalpha() and clean_word[i-1].isalpha()
                               and clean_word[i].islower() != clean_word[i-1].islower())
            if case_changes > len(clean_word) / 3:  # More than 1/3 of characters change case from previous alpha char
                return 'alternating_case'

        return None

    def _calculate_dynamic_consecutive_threshold(self, total_words: int) -> int:
        """
        Calculates a dynamic threshold for consecutive encoded words based on the total number of words.
        Minimum threshold is 4. Scales up by 10% of total words, capped at 10.
        """
        min_threshold = 4
        proportional_threshold = round(total_words * 0.1)
        dynamic_threshold = max(min_threshold, min(proportional_threshold, 10))
        return dynamic_threshold

    def _calculate_dynamic_same_encoding_threshold(self, total_words: int) -> int:
        """
        Calculates a dynamic threshold for words using the same encoding based on the total number of words.
        Minimum threshold is 4. Scales up by 15% of total words, capped at 20.
        A slightly higher percentage and cap are used as these are typically more scattered.
        """
        min_threshold = 4
        proportional_threshold = round(total_words * 0.15)
        dynamic_threshold = max(min_threshold, min(proportional_threshold, 20))
        return dynamic_threshold

    def detect_obfuscation_flags(self, text: str) -> Dict:
        """
        Main detection function that flags based on:
        1. A dynamically scaled number of consecutive encoded words.
        2. A dynamically scaled number of words using the same encoding.
        3. Entire text obfuscation detection.
        """
        # First check for whole text obfuscation
        whole_text_encoding = self.detect_whole_text_obfuscation(text)
        
        if whole_text_encoding:
            return {
                "should_flag": True,
                "whole_text_obfuscation": True,
                "whole_text_encoding_type": whole_text_encoding,
                "consecutive_encoded_count": 0,
                "consecutive_threshold_applied": 0,
                "encoding_counts": {whole_text_encoding: 1},
                "same_encoding_threshold_applied": 0,
                "flag_reasons": [f"Entire text appears to be {whole_text_encoding.replace('_', ' ')} encoded"],
                "total_words_in_message": len(text.split()),
                "total_words_checked_for_obfuscation": 0,
                "total_encoded_words": 1,
                "encoded_word_positions": []
            }
        
        # If not whole text obfuscation, proceed with word-by-word analysis
        words = text.split()
        
        # Calculate dynamic thresholds based on the total number of words in the input text
        consecutive_threshold = self._calculate_dynamic_consecutive_threshold(len(words))
        same_encoding_threshold = self._calculate_dynamic_same_encoding_threshold(len(words))

        consecutive_count = 0
        max_consecutive = 0
        encoding_counts = {}
        encoded_positions = []  # Track which words are encoded

        for i, word in enumerate(words):
            # Only consider words with 3 or more alphabetic characters for obfuscation detection
            if self.count_alphabetic_chars(word) >= 3:
                encoding_type = self.quick_detect_encoding_type(word)

                if encoding_type:
                    consecutive_count += 1
                    encoding_counts[encoding_type] = encoding_counts.get(encoding_type, 0) + 1
                    encoded_positions.append(i)
                else:
                    # End of consecutive sequence
                    max_consecutive = max(max_consecutive, consecutive_count)
                    consecutive_count = 0
            else:
                # Word too short (in terms of alphabetic characters) to check, breaks consecutive sequence
                max_consecutive = max(max_consecutive, consecutive_count)
                consecutive_count = 0

        # Check final consecutive count after the loop
        max_consecutive = max(max_consecutive, consecutive_count)

        # Determine flag conditions
        consecutive_flag = max_consecutive >= consecutive_threshold
        
        # Check for same encoding flag
        same_encoding_flag = False
        same_encoding_flag_details = {}
        for enc_type, count in encoding_counts.items():
            if count >= same_encoding_threshold:
                same_encoding_flag = True
                same_encoding_flag_details[enc_type] = count

        # Build flag reasons
        flag_reasons = []
        if consecutive_flag:
            flag_reasons.append(f"More than {consecutive_threshold-1} consecutive encoded words ({max_consecutive} found). (Threshold: {consecutive_threshold})")
        
        if same_encoding_flag:
            for enc_type, count in same_encoding_flag_details.items():
                flag_reasons.append(f"More than {same_encoding_threshold-1} {enc_type} encoded words ({count} found). (Threshold: {same_encoding_threshold})")

        return {
            "should_flag": consecutive_flag or same_encoding_flag,
            "whole_text_obfuscation": False,
            "whole_text_encoding_type": None,
            "consecutive_encoded_count": max_consecutive,
            "consecutive_threshold_applied": consecutive_threshold,
            "encoding_counts": encoding_counts,
            "same_encoding_threshold_applied": same_encoding_threshold,
            "flag_reasons": flag_reasons,
            "total_words_in_message": len(words),
            "total_words_checked_for_obfuscation": sum(1 for word in words if self.count_alphabetic_chars(word) >= 3),
            "total_encoded_words": len(encoded_positions),
            "encoded_word_positions": encoded_positions
        }

def quick_flag_check(text: str) -> bool:
    """Quick utility function to just return True/False flag."""
    detector = ObfuscationFlagDetector()
    result = detector.detect_obfuscation_flags(text)
    return result["should_flag"]

def detailed_analysis(text: str) -> Dict:
    """Utility function for detailed analysis."""
    detector = ObfuscationFlagDetector()
    return detector.detect_obfuscation_flags(text)

# Demonstration and testing
if __name__ == "__main__":
    detector = ObfuscationFlagDetector()

    # Test cases for whole text obfuscation
    whole_text_test_cases = [
        # Base64 without spaces
        ("SGVsbG8gV29ybGQgVGhpcyBpcyBhIHRlc3QgbWVzc2FnZQ==", "Whole text Base64 - should FLAG"),
        
        # Hex without spaces
        ("48656c6c6f20576f726c642054686973206973206120746573742065787472612063686172616374657273", "Whole text Hex - should FLAG"),
        
        # Binary without spaces
        ("0100100001100101011011000110110001101111001000000101011101101111011100100110110001100100", "Whole text Binary - should FLAG"),
        
        # URL encoded
        ("%48%65%6C%6C%6F%20%57%6F%72%6C%64%20%54%68%69%73%20%69%73%20%61%20%74%65%73%74", "Whole text URL encoded - should FLAG"),
        
        # Unicode escapes
        ("\\u0048\\u0065\\u006c\\u006c\\u006f\\u0020\\u0057\\u006f\\u0072\\u006c\\u0064", "Whole text Unicode escapes - should FLAG"),
        
        # Normal text - should NOT flag
        ("This is normal text that should not be flagged", "Normal text - should NOT flag"),
        
        # Short text - should NOT flag
        ("Hi", "Short text - should NOT flag"),
        
        # Mixed content - should NOT flag as whole text
        ("SGVsbG8= normal text here", "Mixed content - should NOT flag as whole text"),
    ]

    # Original test cases (keeping some key ones)
    original_test_cases = [
        ("Hello world this is normal text", "Normal text - should NOT flag"),
        ("1gn0r3 erongI bypassed disabled", "4 consecutive leetspeak/reversed words - should FLAG"),
        ("SWdub3Jl dGVzdA== QnlwYXNz aGVsbG8= normal", "4 consecutive base64 words - should FLAG"),
    ]

    all_test_cases = whole_text_test_cases + original_test_cases

    print("=== Enhanced Obfuscation Flag Detection Results ===\n")

    for i, (test_text, description) in enumerate(all_test_cases, 1):
        print(f"Test {i}: {description}")
        print(f"Input: '{test_text}'")

        result = detector.detect_obfuscation_flags(test_text)

        flag_status = "🚩 FLAGGED" if result["should_flag"] else "✅ CLEAR"
        print(f"Result: {flag_status}")

        if result["should_flag"]:
            print(f"Reasons: {', '.join(result['flag_reasons'])}")
            
        if result.get("whole_text_obfuscation"):
            print(f"🔍 Whole text obfuscation detected: {result['whole_text_encoding_type']}")
        else:
            print(f"Stats: Total words: {result['total_words_in_message']}, Words checked: {result['total_words_checked_for_obfuscation']}")
            print(f"Consecutive encoded count: {result['consecutive_encoded_count']}")
            print(f"Encoding counts: {result['encoding_counts']}")

        print("-" * 70)

    # Quick utility tests for whole text
    print("\n=== Quick Utility Function Tests (Including Whole Text) ===")
    quick_tests = [
        "Normal text here",
        "SGVsbG8gV29ybGQgVGhpcyBpcyBhIHRlc3QgbWVzc2FnZQ==",  # Base64 whole text
        "48656c6c6f20576f726c64205468697320697320612074657374",  # Hex whole text
        "1gn0r3 erongI bypassed disabled totally",  # Word-based obfuscation
        "0100100001100101011011000110110001101111",  # Binary whole text
        "%48%65%6C%6C%6F%20%57%6F%72%6C%64",  # URL encoded whole text
    ]

    for test in quick_tests:
        flag = quick_flag_check(test)
        status = "🚩" if flag else "✅"
        print(f"{status} '{test}' -> Flagged: {flag}")