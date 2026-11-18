import gradio as gr
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)


# Prediction function
def predict(sepal_length, sepal_width,
            petal_length, petal_width):

    prediction = model.predict([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])[0]

    return iris.target_names[prediction]


# Build interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Sepal Length (cm)", value=5.1),
        gr.Number(label="Sepal Width (cm)", value=3.5),
        gr.Number(label="Petal Length (cm)", value=1.4),
        gr.Number(label="Petal Width (cm)", value=0.2),
    ],
    outputs=gr.Textbox(label="Predicted Flower"),
    title="AI Iris Flower Classifier",
    description="Predict flower species using machine learning."
)

demo.launch(
    server_name="0.0.0.0",
    server_port=int(__import__("os").environ.get("PORT", 7860))
)