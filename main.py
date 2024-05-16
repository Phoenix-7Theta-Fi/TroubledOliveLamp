from flask import Flask, render_template, request
from transformers import pipeline, AutoTokenizer

app = Flask(__name__)

# Your Hugging Face API token (keep this safe!)
HF_API_TOKEN = "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" 

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
generator = pipeline(
    "text-generation",
    model="google/flan-t5-small",
    tokenizer=tokenizer,
    use_auth_token=HF_API_TOKEN  # Place the token here!
)

@app.route("/", methods=["GET", "POST"])
def health_tracker():
    if request.method == "POST":
        temperature = request.form.get("temperature")
        heart_rate = request.form.get("heart_rate")
        question = request.form.get("question").lower()

        # Generate response using the Hugging Face model 
        response = generator(question, max_length=100, num_return_sequences=1)[0]['generated_text']

        return render_template("index.html", temperature=temperature, heart_rate=heart_rate, response=response)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True) 