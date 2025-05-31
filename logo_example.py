import streamlit as st
from PIL import Image
import base64
from pathlib import Path

# Example of how to display PNG logos in your Streamlit app

# Method 1: Using st.image() - Simplest approach
def display_logo_method1():
    """Display logo using st.image()"""
    # Assuming your logo is in the same directory as the app
    logo_path = "logo.png"  # Replace with your actual logo filename
    
    # Check if file exists
    if Path(logo_path).exists():
        logo = Image.open(logo_path)
        st.image(logo, width=150)  # Adjust width as needed
    else:
        st.error(f"Logo file '{logo_path}' not found!")

# Method 2: Using base64 encoding for inline HTML
def display_logo_method2():
    """Display logo using base64 encoding in HTML"""
    logo_path = "logo.png"  # Replace with your actual logo filename
    
    if Path(logo_path).exists():
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        
        logo_html = f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{logo_data}" width="150">
        </div>
        """
        st.markdown(logo_html, unsafe_allow_html=True)
    else:
        st.error(f"Logo file '{logo_path}' not found!")

# Method 3: For sidebar logo with custom styling
def display_sidebar_logo():
    """Display logo in sidebar with custom styling"""
    logo_path = "logo.png"  # Replace with your actual logo filename
    
    if Path(logo_path).exists():
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        
        st.sidebar.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #A44A3F 0%, #2A2A2A 100%);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        ">
            <div style="
                width: 120px;
                height: 120px;
                margin: 0 auto 1rem auto;
                display: flex;
                align-items: center;
                justify-content: center;
            ">
                <img src="data:image/png;base64,{logo_data}" style="max-width: 100%; max-height: 100%; border-radius: 50%;">
            </div>
            <h2 style="color: #F5F2EA; margin: 0; font-family: 'Playfair Display', serif; font-weight: 700;">
                ML Suite
            </h2>
            <p style="color: #D7CEB2; margin: 0.5rem 0 0 0; font-size: 0.9rem; font-family: 'Lato', sans-serif;">
                Production-Ready Analytics
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.error(f"Logo file '{logo_path}' not found!")

# Example usage
if __name__ == "__main__":
    st.title("Logo Display Examples")
    
    st.header("Method 1: Using st.image()")
    display_logo_method1()
    
    st.header("Method 2: Using base64 encoding")
    display_logo_method2()
    
    st.header("Method 3: Sidebar logo")
    display_sidebar_logo()
    
    st.info("""
    To use these methods in your app:
    1. Save your logo PNG file(s) in the project directory
    2. Update the 'logo_path' variable with your actual filename
    3. Copy the relevant function into your enhanced_streamlit_app.py
    4. Replace the existing LOGO placeholder code with the function call
    """)