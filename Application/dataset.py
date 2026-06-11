import streamlit as st
import pandas as pd


def show():
    st.title("📋 Dataset")

    try:
        data = pd.read_csv("C:/Users/Thom/Downloads/final_dataset.csv")  # Get the path for csv
        st.dataframe(data)
        st.success(f"CSV Rows: {data.shape[0]}, CSV Columns: {data.shape[1]}")

        st.divider()
        st.subheader("Add New Row to Dataset")

        new_row_data = {}  # The set of values that will be inserted
        for column in data.columns:
            if column in ["No. of Cases", "MW"]:  # Number box
                value = st.number_input(
                    f"{column} (integer)",
                    min_value=0,
                    value=0,
                    step=1,
                    key=f"input_{column}"
                )
                new_row_data[column] = int(value)
            else:
                new_row_data[column] = st.text_input(f"{column}:", key=f"new_{column}")

        col1, col2 = st.columns([2, 35])  # Adjusting width

        with col1:
            if st.button("Add Row"):
                int_errors = []
                for int_column in ["No. of Cases", "MW"]:  # Check for integer errors, if applicable
                    if int_column in new_row_data:
                        try:
                            int(new_row_data[int_column])
                        except:
                            int_errors.append(f"{int_column} must be an integer")

                missing_values = [column for column in data.columns if column not in ["No. of Cases", "MW"] and
                                  str(new_row_data[column]).strip() == ""]  # Check for missing values

                if not int_errors and not missing_values:  # If pass both tests
                    new_row_data = pd.DataFrame([new_row_data])
                    data = pd.concat([data, new_row_data], ignore_index=True)
                    data.to_csv("C:/Users/Thom/Downloads/final_dataset.csv", index=False)
                    st.success("Row added")
                    st.rerun()
                else:  # If didn't pass any test
                    if int_errors:
                        for error in int_errors:
                            st.error(error)
                    if missing_values:
                        st.error(f"Please fill: {', '.join(missing_values)}")

        with col2:
            if st.button("Delete Last Row"):  # Get last row then save the rest besides it
                if len(data) > 0:
                    data = data.iloc[:-1]
                    data.to_csv("C:/Users/Thom/Downloads/final_dataset.csv", index=False)
                    st.success("Last row deleted")
                    st.rerun()
                else:
                    st.warning("Dataset is empty")

    except FileNotFoundError:
        st.write("File not found")

    st.divider()

    if st.button("← Back"):
        st.session_state.page = "main"
        st.rerun()
