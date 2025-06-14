#!/usr/bin/env python3
"""
Test Gemini API functionality
"""

import sys
import os
from dotenv import load_dotenv
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.gemini_analyzer import GeminiAnalyzer

load_dotenv()

def test_gemini_basic():
    """Test basic Gemini API functionality"""
    print("🤖 Testing Gemini API...")
    
    # Get API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable required!")
        print("Set environment variable: export GEMINI_API_KEY='your-key-here'")
        return False, {}
    
    try:
        analyzer = GeminiAnalyzer(api_key=api_key)
        print(f"✅ Gemini analyzer initialized with model: {analyzer.model}")
        
        # Test with sample RL content
        test_content = """
        Title: Deep Reinforcement Learning for Autonomous Driving
        Abstract: This paper presents a novel approach using Proximal Policy Optimization (PPO) 
        for autonomous vehicle control. Our experiments show that PPO achieves 95% success rate 
        in complex urban environments, significantly outperforming previous Deep Q-Network (DQN) 
        approaches which only achieved 78% success rate. The PPO-based system demonstrates 
        superior sample efficiency and stability compared to traditional reinforcement learning methods.
        Authors: John Smith, Jane Doe
        Categories: cs.LG, cs.RO
        """
        
        print("\n📝 Testing with sample content...")
        print("Content preview:", test_content[:100] + "...")
        
        analysis = analyzer._analyze_with_gemini(test_content, "paper")
        
        print(f"\n🔍 Gemini Analysis Results:")
        print(f"Raw response: {analysis}")
        
        if "techniques" in analysis:
            print(f"\n✅ Found {len(analysis['techniques'])} techniques:")
            for tech in analysis["techniques"]:
                print(f"   - {tech['name']}: {tech['effectiveness_score']:.3f} (conf: {tech['confidence']:.3f})")
                print(f"     Reasoning: {tech['reasoning']}")
        else:
            print("⚠️  No techniques found in analysis")
        
        return True, analysis
        
    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

if __name__ == "__main__":
    success, analysis = test_gemini_basic()
    if success:
        print(f"\n🎉 Gemini test passed!")
    else:
        print("💥 Gemini test failed") 