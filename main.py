#!/usr/bin/env python3
"""
Main entry point for the AI Defense Pipeline System
This script runs the orchestrator with a simple test interface
"""

from orchestrator import process_text_through_defense

def main():
    """Main function to run the AI Defense Pipeline"""
    print("🛡️  AI Defense Pipeline System")
    print("=" * 50)
    print("This system processes text through multiple security layers:")
    print("1. Obfuscation Detection")
    print("2. Translation (if needed)")
    print("3. Jailbreak Detection")
    print("4. PII Detection & Redaction")
    print("5. Main LLM Processing")
    print("6. Final PII Detection & Redaction")
    print("7. Translation Back (if needed)")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Process text through defense pipeline")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1-2): ").strip()
        
        if choice == "1":
            print("\n" + "="*60)
            user_input = input("Enter text to process: ").strip()
            
            if user_input:
                try:
                    result = process_text_through_defense(user_input)
                    print("\n" + "="*60)
                    print("🎯 FINAL RESULT:")
                    print(f"Status: {result['status']}")
                    if 'final_output' in result:
                        print(f"Output: {result['final_output']}")
                    print("="*60)
                except Exception as e:
                    print(f"❌ Error processing text: {e}")
            else:
                print("❌ Please enter some text to process.")
                
        elif choice == "2":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()