import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Load and preprocess dataset
df = pd.read_csv("Student_performance_data .csv")
X = df.drop(columns=["GradeClass"])
y = df["GradeClass"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
features = X.columns.tolist()

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)
feature_importances = model.feature_importances_

# App init
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Student Grade Predictor"

# Layout
app.layout = dbc.Container([
    html.H1("🎓 Student Grade Predictor (Random Forest)", className="my-4"),

    dbc.Row([
        dbc.Col([
            html.H5("Enter student information:"),
            *[
                dbc.Input(id=f"input-{feature}", placeholder=f"Enter {feature}", type="text", className="mb-2")
                for feature in features
            ],
            dbc.Button("Predict GradeClass", id="predict-button", color="primary", className="mt-2"),
            html.Div(id="prediction-output", className="mt-3 text-info fw-bold")
        ], md=4),

        dbc.Col([
            html.H5("Feature Importances:"),
            dcc.Graph(id="feature-importance-chart"),
            html.Hr(),
            html.H5("Confusion Matrix:"),
            dcc.Graph(id="confusion-matrix"),
            html.Div([
                html.H6("Evaluation Metrics:"),
                html.Ul([
                    html.Li(f"Accuracy: {accuracy:.4f}"),
                    html.Li(f"Precision: {precision:.4f}"),
                    html.Li(f"Recall: {recall:.4f}"),
                    html.Li(f"F1 Score: {f1:.4f}")
                ])
            ], className="mt-4")
        ], md=8)
    ])
], fluid=True)

# Callbacks
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    [State(f"input-{feature}", "value") for feature in features]
)
def predict_grade(n_clicks, *vals):
    if not n_clicks:
        return ""
    try:
        input_df = pd.DataFrame([list(vals)], columns=features)
        prediction = model.predict(input_df)[0]
        return f"Predicted GradeClass: {prediction}"
    except Exception as e:
        return f"Error: {e}"

@app.callback(
    Output("feature-importance-chart", "figure"),
    Input("predict-button", "n_clicks")
)
def update_feature_chart(n):
    fig = px.bar(x=features, y=feature_importances, labels={"x": "Feature", "y": "Importance"},
                 title="Feature Importances")
    fig.update_layout(xaxis_tickangle=-45)
    return fig

@app.callback(
    Output("confusion-matrix", "figure"),
    Input("predict-button", "n_clicks")
)
def update_conf_matrix(n):
    fig = ff.create_annotated_heatmap(z=conf_matrix, colorscale="blues")
    fig.update_layout(title="Confusion Matrix")
    return fig

if __name__ == "__main__":
    app.run(debug=True)
