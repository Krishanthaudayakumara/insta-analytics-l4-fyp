#!/usr/bin/env python3
"""
Test script to validate the profile generation Streamlit fix
"""

def test_profile_generation_fix():
    """Test that the profile generation Streamlit fix is working"""
    print("=== TESTING PROFILE GENERATION STREAMLIT FIX ===")
    
    try:
        # Check syntax
        import ast
        with open('app.py', 'r') as f:
            code = f.read()
        
        ast.parse(code)
        print("✅ app.py syntax is valid")
        
        # Check method signature
        if 'def show_profile_results(self, profiles, guidelines, context=' in code:
            print("✅ show_profile_results method signature updated")
        else:
            print("❌ Method signature not found")
            return False
            
        # Check first call
        if 'show_profile_results(profiles, guidelines, "new")' in code:
            print("✅ First call uses 'new' context")
        else:
            print("❌ First call not updated")
            return False
            
        # Check second call
        if 'show_profile_results(existing_profiles, existing_guidelines, "previous")' in code:
            print("✅ Second call uses 'previous' context")
        else:
            print("❌ Second call not updated")
            return False
            
        # Check unique key generation
        if 'unique_key = f"content_preference_distribution_{context}"' in code:
            print("✅ Unique key generation implemented")
        else:
            print("❌ Unique key generation not found")
            return False
            
        # Check that plotly chart uses unique key
        if 'st.plotly_chart(fig, use_container_width=True, key=unique_key)' in code:
            print("✅ Plotly chart uses unique key")
        else:
            print("❌ Plotly chart key not updated")
            return False
            
        print("\n🎉 ALL PROFILE GENERATION FIXES SUCCESSFUL!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_profile_generation_fix()
    if success:
        print("\n✅ Profile generation duplicate key issue has been fixed!")
        print("🚀 The Streamlit app should now work without duplicate key errors in profile generation.")
    else:
        print("\n❌ Some issues remain in the profile generation fix.")
