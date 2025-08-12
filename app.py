import streamlit as st
from src.generate_names import load_model, generate_names

# --- Page Configuration ---
st.set_page_config(
    page_title="Indic Name Forge",
    page_icon="🇮🇳",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Model Loading ---
@st.cache_resource
def cached_load_model():
    """Load and cache the model to avoid reloading on every interaction."""
    try:
        return load_model()
    except FileNotFoundError:
        return None

model_data = cached_load_model()
if model_data is None:
    st.error(
        "**Model not found!** Please make sure the `models/wavenet_indian_names.pt` file is available."
    )
    st.stop()

model, stoi, itos = model_data

# --- UI: Title and Description ---
st.title("✍️ Indic Name Forge")
st.markdown(
    "Generate unique, Indian-sounding names using a Wavenet-inspired model. "
    "Adjust the settings in the sidebar to control the creativity and length of the generated names."
)

# --- UI: Sidebar for Settings ---
with st.sidebar:
    st.header("⚙️ Generation Settings")
    
    start_char = st.text_input(
        "Starting Letter",
        "A",
        max_chars=1,
        help="The first letter of the generated names. Must be a single alphabet character.",
    ).strip()

    n_names = st.slider(
        "Number of Names",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help="How many names to generate in one go.",
    )

    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.1,
        max_value=1.5,
        value=1.0,
        step=0.05,
        help=(
            "Controls the randomness of the output. "
            "Lower values (~0.7) produce more predictable, conservative names. "
            "Higher values (>1.0) produce more surprising, creative names."
        ),
    )

    generate_button = st.button("Generate Names", type="primary")

# --- Liked Names Initialization ---
if "liked_names" not in st.session_state:
    st.session_state.liked_names = []

# --- Name Generation and Display ---
st.header("Generated Names")

if generate_button:
    if not start_char.isalpha() or len(start_char) != 1:
        st.warning("Please enter a single starting letter (A-Z).")
    else:
        with st.spinner(f"Generating {n_names} names starting with '{start_char.upper()}'..."):
            names = generate_names(
                model=model,
                stoi=stoi,
                itos=itos,
                n=n_names,
                temperature=temperature,
                start_char=start_char,
            )
            st.session_state.generated_names = [name.capitalize() for name in names]

if "generated_names" in st.session_state and st.session_state.generated_names:
    for idx, name in enumerate(st.session_state.generated_names):
        is_liked = name in st.session_state.liked_names
        if st.checkbox(name, value=is_liked, key=f"cb_{idx}_{name}"):
            if name not in st.session_state.liked_names:
                st.session_state.liked_names.append(name)
        elif name in st.session_state.liked_names:
            st.session_state.liked_names.remove(name)
else:
    st.info("Click the 'Generate Names' button in the sidebar to start.")

# --- UI: Sidebar Liked Names and Download ---
with st.sidebar:
    st.header("❤️ Liked Names")
    if st.session_state.liked_names:
        for liked_name in st.session_state.liked_names:
            st.markdown(f"- {liked_name}")
        
        # Prepare download data
        download_data = "\n".join(st.session_state.liked_names)
        st.download_button(
            label="Download Liked Names",
            data=download_data,
            file_name="liked_names.txt",
            mime="text/plain",
        )
    else:
        st.info("Like some names to see them here.")
