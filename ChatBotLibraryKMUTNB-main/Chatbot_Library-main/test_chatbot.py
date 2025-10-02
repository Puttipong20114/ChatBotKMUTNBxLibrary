#!/usr/bin/env python3
"""
Test script for KMUTNB chatbot
"""

from document_reader import read_kmutnb_dataset

def test_document_reading():
    """Test if the document can be read successfully"""
    print("Testing document reading...")
    
    try:
        content = read_kmutnb_dataset()
        print(f"✅ Document loaded successfully!")
        print(f"📄 Content length: {len(content):,} characters")
        print(f"📝 First 200 characters:")
        print("-" * 50)
        print(content[:200])
        print("-" * 50)
        return True
    except Exception as e:
        print(f"❌ Error reading document: {e}")
        return False

def test_content_analysis():
    """Analyze the content to see what information is available"""
    print("\nAnalyzing document content...")
    
    try:
        content = read_kmutnb_dataset()
        
        # Look for key sections
        keywords = [
            "คณะ", "สาขา", "หลักสูตร", "การรับสมัคร", 
            "เกณฑ์", "คุณสมบัติ", "เอกสาร", "วันสอบ",
            "มหาวิทยาลัย", "KMUTNB", "มจพ"
        ]
        
        print("🔍 Found keywords in document:")
        for keyword in keywords:
            count = content.count(keyword)
            if count > 0:
                print(f"   {keyword}: {count} occurrences")
        
        # Look for specific sections
        sections = [
            "สรุปเกณฑ์การรับสมัคร",
            "คณะวิศวกรรมศาสตร์",
            "คณะวิทยาศาสตร์ประยุกต์",
            "คณะเทคโนโลยีและการจัดการ"
        ]
        
        print("\n📋 Document sections found:")
        for section in sections:
            if section in content:
                print(f"   ✅ {section}")
            else:
                print(f"   ❌ {section}")
                
        return True
    except Exception as e:
        print(f"❌ Error analyzing content: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing KMUTNB Chatbot Setup")
    print("=" * 50)
    
    # Test document reading
    doc_test = test_document_reading()
    
    # Test content analysis
    content_test = test_content_analysis()
    
    print("\n" + "=" * 50)
    if doc_test and content_test:
        print("🎉 All tests passed! Chatbot is ready to use.")
        print("💡 You can now run: streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the setup.") 