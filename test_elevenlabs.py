"""
Test script for ElevenLabs API key validation.
Run this to check if your API key is working correctly.

Usage: python test_elevenlabs.py
"""

import sys
from backend import test_elevenlabs_api_key, generate_audio

def main():
    print("=" * 60)
    print("ElevenLabs API Key Test Script")
    print("=" * 60)
    print()
    
    # Get API key from user
    print("Please enter your ElevenLabs API key:")
    print("(You can get one at: https://elevenlabs.io/app/settings/api-keys)")
    print()
    api_key = input("API Key: ").strip()
    
    if not api_key:
        print("\n❌ No API key entered. Exiting.")
        return
    
    print("\n" + "=" * 60)
    print("Step 1: Testing API Key Validity")
    print("=" * 60)
    
    is_valid, message = test_elevenlabs_api_key(api_key)
    print(message)
    
    if not is_valid:
        print("\n❌ API key test failed!")
        print("\nTroubleshooting steps:")
        print("1. Generate a new API key at: https://elevenlabs.io/app/settings/api-keys")
        print("2. Make sure you have credits in your ElevenLabs account")
        print("3. Check if your account is active (not suspended)")
        print("4. Verify your free tier quota hasn't been exceeded")
        print("5. Try with a fresh account if needed")
        return
    
    print("\n✅ API key is valid!")
    
    # Optional: Test actual audio generation
    print("\n" + "=" * 60)
    print("Step 2: Testing Audio Generation (Optional)")
    print("=" * 60)
    
    response = input("\nDo you want to test audio generation? (y/n): ").strip().lower()
    
    if response == 'y':
        test_text = "Hello! This is a test of the ElevenLabs text to speech API."
        print(f"\nGenerating audio for: '{test_text}'")
        print("Please wait...")
        
        result = generate_audio(test_text, api_key)
        
        if result.startswith("Error:"):
            print(f"\n❌ {result}")
        else:
            print(f"\n✅ Audio generated successfully!")
            print(f"📁 File saved as: {result}")
            print("\nYou can play this file to verify the audio quality.")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    
    if is_valid:
        print("\n✅ Your API key is working correctly.")
        print("You can now use it in the Streamlit app.")
    else:
        print("\n❌ There were issues with your API key.")
        print("Please follow the troubleshooting steps above.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
