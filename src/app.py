from flask import Flask, request, render_template
from recommend import recommend_recipes

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get the ingredients from the form
    ingredients = request.form['ingredients']

    # Call your model or recommendation function
    recommended_recipes = recommend_recipes(ingredients)

    # Return the recommendations to the user
    return render_template('index.html', recipes=recommended_recipes)

if __name__ == "__main__":
    app.run(debug=True)
