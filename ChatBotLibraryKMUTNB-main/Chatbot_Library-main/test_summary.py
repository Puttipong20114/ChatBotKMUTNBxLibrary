#!/usr/bin/env python3
"""
Test script for KMUTNB document summary
"""

from document_reader import get_kmutnb_summary

def test_summary():
    """Test the document summary function"""
    print("Testing document summary...")
    
    try:
        summary = get_kmutnb_summary()
        print(f"✅ Summary generated successfully!")
        print(f"📄 Summary length: {len(summary):,} characters")
        print(f"📝 First 500 characters:")
        print("-" * 50)
        print(summary[:500])
        print("-" * 50)
        
        # Estimate token count (roughly 4 characters per token)
        estimated_tokens = len(summary) // 4
        print(f"🔢 Estimated tokens: ~{estimated_tokens:,}")
        
        if estimated_tokens < 100000:  # Well under the 250k limit
            print("✅ Token count is well within API limits!")
        else:
            print("⚠️ Token count might still be high")
            
        return True
    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        return False

if __name__ == "__main__":
    test_summary() 