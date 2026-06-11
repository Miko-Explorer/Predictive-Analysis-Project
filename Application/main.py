import streamlit as st

st.set_page_config(page_title="Outbreak Predictor", page_icon="🦠", layout="wide")

if 'page' not in st.session_state:
    st.session_state.page = "main"

if st.session_state.page == "main":
    st.title("Project Home Page")

    st.divider()

    st.markdown("""
        <style>
        .menu-title {
            font-size: 30px;
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 5px;
        }
        
        .menu-desc {
            font-size: 15px;
            margin-bottom: 15px;
            color: #666;
        }
        </style>
    """, unsafe_allow_html=True)  # Allows html to be read

    st.markdown('<div class="menu-title">📊 View Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="menu-desc">Allows the user to see the dataset used for this project and add/delete rows</div>', unsafe_allow_html=True)

    st.markdown('<div class="menu-title">🔮 View Predictions</div>', unsafe_allow_html=True)
    st.markdown('<div class="menu-desc">View the predictions for the Logistic Regression and Random Forest</div>', unsafe_allow_html=True)

    st.markdown('<div class="menu-title">📈 View Plots</div>', unsafe_allow_html=True)
    st.markdown('<div class="menu-desc">View the plot</div>', unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("View Dataset", use_container_width=True):
            st.session_state.page = "dataset"
            st.rerun()
    with col2:
        if st.button("View Predictions", use_container_width=True):
            st.session_state.page = "prediction"
            st.rerun()

elif st.session_state.page == "dataset":
    import dataset
    dataset.show()

elif st.session_state.page == "prediction":
    import prediction
    prediction.show()
