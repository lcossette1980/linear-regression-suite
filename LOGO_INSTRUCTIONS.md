# Logo Setup Instructions

Your Streamlit app is now configured to display PNG logo files. Here's how to add your logos:

## Quick Setup

1. **Add your logo file**: Place your PNG logo file in the project root directory (`/Users/lorencossette/linregapp/`)
   
2. **Name your logo file**: The code is currently looking for a file named `logo.png`. You can either:
   - Name your logo file `logo.png`, OR
   - Update the filename in the code (see below)

3. **Run your app**: The logo will automatically display in both the sidebar and home page

## Customizing the Logo Filename

If your logo has a different filename, update these lines in `enhanced_streamlit_app.py`:

- **Line 600** (in the sidebar function): `logo_path = "logo.png"`
- **Line 740** (in the home page section): `logo_path = "logo.png"`

Change `"logo.png"` to your actual filename, for example: `"company_logo.png"`

## Using Different Logos for Sidebar and Home

If you want different logos for the sidebar and home page:

1. Add both logo files to your project directory
2. Update line 600 with the sidebar logo filename
3. Update line 740 with the home page logo filename

## Logo Recommendations

- **Format**: PNG with transparent background works best
- **Size**: 300x300 pixels or larger for good quality
- **Shape**: Square logos work best with the circular display style

## Testing

Run your Streamlit app with:
```bash
streamlit run enhanced_streamlit_app.py
```

The logos will appear:
- In the sidebar at the top
- On the home page as the main logo

## Fallback Behavior

If the logo file is not found, the app will display "LOGO" text as a placeholder, so your app will still work even without the logo files.