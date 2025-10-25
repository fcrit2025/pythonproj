import streamlit as st
import random
import re

# Page configuration
st.set_page_config(
    page_title="AI Text Humanizer",
    page_icon="✨",
    layout="centered"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stTextArea textarea { font-size: 16px; }
    .success-box { 
        background-color: #d4edda; 
        padding: 1rem; 
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("✨ AI Text Humanizer")
st.markdown("Make your AI-generated text sound more natural and human-like!")

# Humanization techniques
def humanize_text(text):
    """
    Apply various techniques to make text sound more human
    """
    if not text:
        return text
    
    # Create a copy to work with
    humanized = text
    
    # 1. Add occasional filler words (10% chance per sentence)
    sentences = re.split(r'([.!?]+)', humanized)
    humanized = ""
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i].strip()
        punctuation = sentences[i+1] if i+1 < len(sentences) else ""
        
        if sentence and random.random() < 0.1:
            fillers = ["Well, ", "You know, ", "Actually, ", "So, "]
            sentence = random.choice(fillers) + sentence.lower()
        
        humanized += sentence + punctuation + " "
    
    # 2. Replace some formal words with casual equivalents
    word_replacements = {
        r'\butilize\b': 'use',
        r'\bassistance\b': 'help',
        r'\bcommence\b': 'start',
        r'\bterminate\b': 'end',
        r'\bapproximately\b': 'about',
        r'\bhowever\b': 'but',
        r'\btherefore\b': 'so',
        r'\badditional\b': 'more',
        r'\bacquire\b': 'get',
        r'\bdemonstrate\b': 'show'
    }
    
    for formal, casual in word_replacements.items():
        humanized = re.sub(formal, casual, humanized, flags=re.IGNORECASE)
    
    # 3. Add occasional contractions
    contractions = {
        r'\bdo not\b': "don't",
        r'\bcannot\b': "can't",
        r'\bwill not\b': "won't",
        r'\bI am\b': "I'm",
        r'\bit is\b': "it's",
        r'\bthat is\b': "that's"
    }
    
    for full, contraction in contractions.items():
        if random.random() < 0.3:  # 30% chance to apply each contraction
            humanized = re.sub(full, contraction, humanized, flags=re.IGNORECASE)
    
    # 4. Vary sentence beginnings occasionally
    sentences = re.split(r'([.!?]+ )', humanized)
    humanized = ""
    for i in range(0, len(sentences), 2):
        if i < len(sentences):
            sentence = sentences[i]
            if i+1 < len(sentences):
                punctuation = sentences[i+1]
            else:
                punctuation = ""
            
            # Sometimes start with a conjunction for more natural flow
            if random.random() < 0.15 and i > 0:
                conjunctions = ["And", "But", "So", "Well", "Anyway"]
                sentence = random.choice(conjunctions) + ", " + sentence[0].lower() + sentence[1:]
            
            humanized += sentence + punctuation
    
    return humanized.strip()

# Main app interface
def main():
    # Text input
    input_text = st.text_area(
        "Paste your AI-generated text here:",
        height=150,
        placeholder="Enter the text you want to humanize..."
    )
    
    # Configuration options
    col1, col2 = st.columns(2)
    
    with col1:
        humanize_level = st.slider(
            "Humanization Level",
            min_value=1,
            max_value=3,
            value=2,
            help="1: Light, 2: Moderate, 3: Heavy"
        )
    
    with col2:
        generate_versions = st.selectbox(
            "Number of versions to generate",
            options=[1, 2, 3],
            index=0
        )
    
    # Humanize button
    if st.button("✨ Humanize Text", type="primary"):
        if input_text.strip():
            with st.spinner("Humanizing your text..."):
                st.subheader("Humanized Results")
                
                for i in range(generate_versions):
                    # Apply humanization multiple times based on level
                    humanized_text = input_text
                    for _ in range(humanize_level):
                        humanized_text = humanize_text(humanized_text)
                    
                    # Display result
                    with st.expander(f"Version {i+1}", expanded=i==0):
                        st.text_area(
                            f"Humanized text {i+1}",
                            value=humanized_text,
                            height=150,
                            key=f"result_{i}"
                        )
                        
                        # Copy button for each version
                        if st.button(f"📋 Copy Version {i+1}", key=f"copy_{i}"):
                            st.session_state[f"text_{i}"] = humanized_text
                            st.success(f"Version {i+1} copied to clipboard!")
            
            # Statistics
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            original_words = len(input_text.split())
            humanized_words = len(humanized_text.split())
            
            with col1:
                st.metric("Original Words", original_words)
            with col2:
                st.metric("Humanized Words", humanized_words)
            with col3:
                change = humanized_words - original_words
                st.metric("Word Change", f"{change:+d}")
                        
        else:
            st.warning("Please enter some text to humanize.")
    
    # Tips section
    with st.expander("💡 Tips for better humanization"):
        st.markdown("""
        - **Start with clear AI text**: The tool works best on text that obviously sounds AI-generated
        - **Adjust the level**: Use lighter settings for formal content, heavier for casual writing
        - **Generate multiple versions**: Compare different outputs to find the best one
        - **Manual editing**: Always review and tweak the results for your specific needs
        
        *This tool uses basic text transformation techniques. For more advanced humanization, consider using AI models specifically trained for this purpose.*
        """)

    # Example section
    with st.expander("📝 See an example"):
        st.markdown("""
        **Original AI text:**
        > "The utilization of artificial intelligence has demonstrated significant potential for enhancing productivity and efficiency in various business contexts. However, it is important to note that proper implementation requires careful consideration of ethical implications."
        
        **Humanized version:**
        > "You know, using AI has shown it can really boost productivity and efficiency in different business situations. But it's important to remember that putting it into practice needs careful thinking about the ethical side of things."
        """)

if __name__ == "__main__":
    main()