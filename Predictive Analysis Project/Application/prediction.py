import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def show():
    st.title("🔮 Predictions")

    st.markdown("---")

    # Load the dataset
    df = pd.read_csv("C:/Users/Thom/Downloads/final_dataset.csv")

    # SECTION 1: Prepare the data

    st.subheader("🔧 Data Preparation")

    # Sort by Region, Disease, and Morbidity Week (MW)
    df = df.sort_values(['Region', 'Disease', 'MW'])

    # Create target variable: Will NEXT WEEK have high cases?
    # Shift cases forward by 1 week for each Region-Disease combination
    df['next_week_cases'] = df.groupby(['Region', 'Disease'])['No. of Cases'].shift(-1)

    # Calculate average cases (use current week's average for fair comparison)
    average_cases = df['No. of Cases'].mean()

    # Create target: 1 if NEXT WEEK has above-average cases, 0 otherwise
    df['risk_prediction'] = np.where(df['next_week_cases'] > average_cases, 1, 0)

    # Remove the last week for each group (no future data to predict)
    df = df.dropna(subset=['risk_prediction', 'next_week_cases'])

    st.success(f"Dataset loaded: {df.shape[0]} records")
    st.success(f"Average cases per location: {average_cases:.1f}")
    st.success(f"High risk locations: {sum(df['risk_prediction'] == 1)}")
    st.success(f"Low risk locations: {sum(df['risk_prediction'] == 0)}")

    # SECTION 2: Prepare data for Random Forest (all columns)

    st.subheader("🤖 Model Training Setup")

    # Random Forest will use all available information

    # Convert text columns to numbers using one-hot encoding
    df_encoded = pd.get_dummies(df, columns=['Region', 'PHI', 'Barangay', 'Disease'], drop_first=True)

    # Prepare features and target for Random Forest
    rf_features = [col for col in df_encoded.columns if col != 'risk_prediction']
    X_rf = df_encoded[rf_features]
    y_rf = df_encoded['risk_prediction']

    st.info(f"Random Forest will use **{len(rf_features)}** features")

    tab1, tab2 = st.tabs(["📈 Logistic Regression", "🌳 Random Forest"])

    # SECTION 3: Logistic Regression Model

    # Uses only number of cases and MW as features
    with tab1:
        st.header("Logistic Regression Model")
        st.write("Using features: No. of Cases and MW")

        progress_bar = st.progress(0)

        # Select only two features for Logistic Regression
        log_features = ['No. of Cases', 'MW']
        X_log = df[log_features]
        y_log = df['risk_prediction']

        # Split data into training and testing sets
        X_log_train, X_log_test, y_log_train, y_log_test = train_test_split(
            X_log, y_log, test_size=0.3, random_state=42, stratify=y_log
        )

        progress_bar.progress(30)

        # Create and train Logistic Regression model
        log_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        log_model.fit(X_log_train, y_log_train)

        progress_bar.progress(60)

        # Make predictions on test data
        log_predictions = log_model.predict(X_log_test)

        # Calculate performance metrics
        log_accuracy = accuracy_score(y_log_test, log_predictions)
        log_precision = precision_score(y_log_test, log_predictions, zero_division=0)
        log_recall = recall_score(y_log_test, log_predictions, zero_division=0)
        log_f1 = f1_score(y_log_test, log_predictions, zero_division=0)

        progress_bar.progress(100)

        st.subheader("📈 Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy:", f"{log_accuracy:.2%}")
        with col2:
            st.metric("Precision:", f"{log_precision:.2%}")
        with col3:
            st.metric("Recall:", f"{log_recall:.2%}")
        with col4:
            st.metric("F1 Score:", f"{log_f1:.2%}")

        st.subheader("Sample Predictions (First 20)")

        test_indices = list(y_log_test.index)[:18000]
        display = []

        for i, idx in enumerate(test_indices):
            region = df.loc[idx, 'Region']
            cases = df.loc[idx, 'No. of Cases']
            mw = df.loc[idx, 'MW']
            actual = "HIGH" if y_log_test.iloc[i] == 1 else "LOW"
            predicted = "HIGH" if log_predictions[i] == 1 else "LOW"
            correct = "CORRECT" if actual == predicted else "WRONG"

            display.append({"Region": region, "Cases": cases, "MW": mw, "Actual Risk": actual,
                            "Predicted Risk": predicted, "Result": correct})

        display_df = pd.DataFrame(display)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # SECTION 4: Random Forest Model

    # Uses all available features including encoded categorical data
    with tab2:
        st.header("Random Forest Model")
        st.write(f"**Features used:** All {len(rf_features)} available features")

        progress_bar = st.progress(0)

        # Split data for Random Forest
        X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(
            X_rf, y_rf, test_size=0.3, random_state=42, stratify=y_rf
        )

        progress_bar.progress(30)

        # Create and train Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf_model.fit(X_rf_train, y_rf_train)

        progress_bar.progress(60)

        # Make predictions
        rf_predictions = rf_model.predict(X_rf_test)

        # Calculate performance metrics
        rf_accuracy = accuracy_score(y_rf_test, rf_predictions)
        rf_precision = precision_score(y_rf_test, rf_predictions, zero_division=0)
        rf_recall = recall_score(y_rf_test, rf_predictions, zero_division=0)
        rf_f1 = f1_score(y_rf_test, rf_predictions, zero_division=0)

        progress_bar.progress(100)

        # Get the test indices (these are indices from the original df)
        test_indices = X_rf_test.index

        st.subheader("📈 Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy:", f"{rf_accuracy:.2%}")
        with col2:
            st.metric("Precision:", f"{rf_precision:.2%}")
        with col3:
            st.metric("Recall:", f"{rf_recall:.2%}")
        with col4:
            st.metric("F1 Score:", f"{rf_f1:.2%}")

        st.subheader("Sample Predictions (First 20)")

        # Get indices from y_rf_test and match to original df
        test_indices = list(y_rf_test.index)[:18000]
        display = []

        for i, idx in enumerate(test_indices):
            # Use .loc with the original index (this worked for Logistic Regression)
            try:
                region = df.loc[idx, 'Region']
                cases = df.loc[idx, 'No. of Cases']
                mw = df.loc[idx, 'MW']
                actual = "HIGH" if y_rf_test.iloc[i] == 1 else "LOW"
                predicted = "HIGH" if rf_predictions[i] == 1 else "LOW"
                correct = "CORRECT" if actual == predicted else "WRONG"

                display.append({"Region": region, "Cases": cases, "MW": mw, "Actual Risk": actual,
                                "Predicted Risk": predicted, "Result": correct})
            except KeyError:
                # If index doesn't exist in df, skip detailed display
                actual = "HIGH" if y_rf_test.iloc[i] == 1 else "LOW"
                predicted = "HIGH" if rf_predictions[i] == 1 else "LOW"
                correct = "CORRECT" if actual == predicted else "WRONG"

                display.append({"Region": f"Sample {i+1}", "Cases": "N/A", "MW": "N/A", "Actual Risk": actual,
                                "Predicted Rist": predicted, "Result": correct})

        display_df = pd.DataFrame(display)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Set global figure size
    plt.rcParams['figure.figsize'] = [15, 10]
    plt.rcParams['figure.dpi'] = 100

    # Create simpler visualizations
    plt.figure(figsize=(15, 10))

    # VISUALIZATION 1: Simple comparison of predictions

    # Simple bar-like visualization using points and lines
    fig, ax = plt.subplots(figsize=(10, 7))

    # Performance metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    log_values = [log_accuracy, log_precision, log_recall, log_f1]
    rf_values = [rf_accuracy, rf_precision, rf_recall, rf_f1]

    # Create positions for metrics
    x_pos = np.arange(len(metrics))

    # Plot points for each metric
    plt.scatter(x_pos - 0.1, log_values, s=150, color='red', alpha=0.7, label='Logistic')
    plt.scatter(x_pos + 0.1, rf_values, s=150, color='blue', alpha=0.7, label='Random Forest')

    # Connect each pair of points with a line
    for i in range(len(metrics)):
        plt.plot([x_pos[i] - 0.1, x_pos[i] + 0.1], [log_values[i], rf_values[i]],
                 'gray', alpha=0.5, linewidth=1)

    plt.xlabel('Performance Metric')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x_pos, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)

    # Add value labels
    for i, (log_val, rf_val) in enumerate(zip(log_values, rf_values)):
        plt.text(i - 0.1, log_val + 0.02, f'{log_val:.1%}', ha='center', fontsize=9)
        plt.text(i + 0.1, rf_val + 0.02, f'{rf_val:.1%}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()
    st.pyplot(plt.gcf())
    plt.clf()

    # VISUALIZATION 2: Model accuracy on test samples

    # Combine all Logistic Regression metrics into one list
    log_metrics = [log_accuracy, log_precision, log_recall, log_f1]

    # Combine all Random Forest metrics into one list
    rf_metrics = [rf_accuracy, rf_precision, rf_recall, rf_f1]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    x = range(len(metrics))

    bars_log = ax.bar([i - bar_width / 2 for i in x], log_metrics, bar_width,
                      label='Logistic Regression', color='blue', alpha=0.7)
    bars_rf = ax.bar([i + bar_width / 2 for i in x], rf_metrics, bar_width,
                     label='Random Forest', color='red', alpha=0.7)

    # Add value labels on top
    for bars in [bars_log, bars_rf]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                    f'{height:.2%}', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
    st.pyplot(plt.gcf())
    plt.clf()

    st.info("Performance graphs of Random Forest vs Logistic Regression")
    st.markdown("---")

    # Calculate agreement
    agreement = (log_predictions == rf_predictions)
    agreement_pct = agreement.mean() * 100

    # Create figure
    plt.figure(figsize=(10, 7))

    # Create donut chart
    labels = ['Agree', 'Disagree']
    sizes = [agreement_pct, 100 - agreement_pct]
    colors = ['blue', 'red']

    # Outer pie
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, pctdistance=0.85)

    # Inner circle for donut effect
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Add percentage in center
    plt.text(0, 0, f'{agreement_pct:.1f}%\nAGREEMENT',
             ha='center', va='center', fontsize=24, fontweight='bold')

    plt.title('Final Model Agreement Percentage\nLogistic Regression vs Random Forest',
              fontsize=16, fontweight='bold', pad=20)

    # Add subtitle with counts
    plt.figtext(0.5, 0.01,
                f'Total: {len(agreement):,} predictions | Agree: {agreement.sum():,} | Disagree: {(~agreement).sum():,}',
                ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.show()
    st.pyplot(plt.gcf())
    plt.clf()

    st.info("Shows the agreement of the predictions of both models")
    st.markdown("---")

    # VISUALIZATION 4: Model agreement over samples

    # Calculate risk by region
    risk_by_region = df.groupby('Region')['risk_prediction'].agg(
        high_risk_pct=lambda x: (x == 1).mean() * 100,
        total_locations='count'
    ).reset_index()

    # Sort by risk (highest first)
    risk_by_region = risk_by_region.sort_values('high_risk_pct', ascending=False)

    # Create simple bar chart
    plt.figure(figsize=(12, 6))

    # Plot bars with color based on risk level
    bars = plt.bar(risk_by_region['Region'],
                   risk_by_region['high_risk_pct'],
                   color=['orange' if pct > 50 else 'blue' if pct > 25 else 'red'
                          for pct in risk_by_region['high_risk_pct']],
                   edgecolor='black',
                   linewidth=1)

    # Add value labels on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2,
                 height + 1,
                 f'{height:.1f}%',
                 ha='center',
                 va='bottom',
                 fontsize=10)

    # Calculate and show overall average
    overall_avg = df['risk_prediction'].mean() * 100
    plt.axhline(y=overall_avg, color='gray', linestyle='--', alpha=0.7)
    plt.text(len(risk_by_region) - 0.5, overall_avg + 1,
             f'Average: {overall_avg:.1f}%',
             ha='right',
             color='gray')

    # Simple labels and title
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('High Risk Locations (%)', fontsize=12)
    plt.title('Risk Distribution by Region', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, min(100, risk_by_region['high_risk_pct'].max() * 1.2))

    # Add light grid for readability
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
    st.pyplot(plt.gcf())
    plt.clf()

    st.info("Shows the region with the highest and lowest risk of an outbreak")
    st.markdown("---")

    # Prepare the data
    # Group by Region and Morbidity Week, count high/low risk cases
    risk_by_week = df.groupby(['Region', 'MW', 'risk_prediction']).size().unstack(fill_value=0)
    risk_by_week.columns = ['Low_Risk', 'High_Risk']
    risk_by_week = risk_by_week.reset_index()

    # Create the line graph
    plt.figure(figsize=(14, 8))

    # Get unique regions for coloring
    regions = risk_by_week['Region'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(regions)))  # Distinct colors for each region

    # Plot each region
    for region, color in zip(regions, colors):
        region_data = risk_by_week[risk_by_week['Region'] == region].sort_values('MW')

        # High Risk line (solid)
        plt.plot(region_data['MW'], region_data['High_Risk'],
                 color=color, linewidth=2, marker='o', markersize=5,
                 label=f'{region} - High Risk')

        # Low Risk line (dashed)
        plt.plot(region_data['MW'], region_data['Low_Risk'],
                 color=color, linewidth=2, marker='s', markersize=4, linestyle='--',
                 label=f'{region} - Low Risk')

    # Customize the graph
    plt.xlabel('Morbidity Week (MW)', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Cases', fontsize=12, fontweight='bold')
    plt.title('High vs Low Risk Case Distribution by Region\n(Logistic Regression Model)',
              fontsize=14, fontweight='bold', pad=20)

    # Add grid for readability
    plt.grid(True, alpha=0.3, linestyle='--')

    # Adjust legend to avoid clutter
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=9)

    # Set y-axis to start at 0
    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.show()
    st.pyplot(plt.gcf())
    plt.clf()

    # Optional: Create separate plots for High Risk and Low Risk
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot High Risk only
    for region, color in zip(regions, colors):
        region_data = risk_by_week[risk_by_week['Region'] == region].sort_values('MW')
        ax1.plot(region_data['MW'], region_data['High_Risk'],
                 color=color, linewidth=2, marker='o', label=region)

    ax1.set_title('High Risk Cases by Region', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Morbidity Week (MW)', fontsize=11)
    ax1.set_ylabel('Number of High Risk Cases', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot Low Risk only
    for region, color in zip(regions, colors):
        region_data = risk_by_week[risk_by_week['Region'] == region].sort_values('MW')
        ax2.plot(region_data['MW'], region_data['Low_Risk'],
                 color=color, linewidth=2, marker='s', label=region)

    ax2.set_title('Low Risk Cases by Region', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Morbidity Week (MW)', fontsize=11)
    ax2.set_ylabel('Number of Low Risk Cases', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()
    st.pyplot(plt.gcf())
    plt.clf()

    # Prepare the data (same as before, using risk_prediction from Random Forest)
    risk_by_week = df.groupby(['Region', 'MW', 'risk_prediction']).size().unstack(fill_value=0)
    risk_by_week.columns = ['Low_Risk', 'High_Risk']
    risk_by_week = risk_by_week.reset_index()

    # Create the line graph with SOLID lines
    plt.figure(figsize=(14, 8))

    regions = risk_by_week['Region'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(regions)))

    # Plot each region
    for region, color in zip(regions, colors):
        region_data = risk_by_week[risk_by_week['Region'] == region].sort_values('MW')

        # High Risk line - SOLID
        plt.plot(region_data['MW'], region_data['High_Risk'],
                 color=color, linewidth=2, marker='o', markersize=5,
                 label=f'{region} - High Risk')

        # Low Risk line - ALSO SOLID
        plt.plot(region_data['MW'], region_data['Low_Risk'],
                 color=color, linewidth=2, marker='s', markersize=4,
                 label=f'{region} - Low Risk')

    # Customize the graph for Random Forest
    plt.xlabel('Morbidity Week (MW)', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Cases', fontsize=12, fontweight='bold')
    plt.title('High vs Low Risk Case Distribution by Region\n(Random Forest Classifier)',
              fontsize=14, fontweight='bold', pad=20)

    plt.grid(True, alpha=0.3, linestyle='-')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=9)
    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.show()
    st.pyplot(plt.gcf())
    plt.clf()

    st.info("Shows the distribution of high and low risk cases")
    st.markdown("---")

    # Optional: Add Random Forest Feature Importance Bar Chart
    # (This is unique to Random Forest - shows which features mattered most)
    plt.figure(figsize=(12, 6))

    # Assuming you have trained rf_model
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_rf.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)

    # Create horizontal bar chart
    bars = plt.barh(feature_importance['Feature'], feature_importance['Importance'],
                    color='steelblue', edgecolor='black')

    plt.xlabel('Feature Importance Score', fontsize=12)
    plt.title('Random Forest - Top 15 Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                 f'{width:.3f}', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()
    st.pyplot(plt.gcf())
    plt.clf()

    st.info("Shows feature importance per variable")
    st.markdown("---")

    if st.button("← Back"):
        st.session_state.page = "main"
        st.rerun()
