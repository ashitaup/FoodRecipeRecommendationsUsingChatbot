import streamlit as st
from recipe_scrapers import scrape_me
import requests
from bs4 import BeautifulSoup
import re
import base64
import openai
import textwrap
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part

openai.api_key = "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"

def search_allrecipes(recipe_name):
    base_url = "https://www.allrecipes.com/search"
    params = {"q": recipe_name}
    
    response = requests.get(base_url, params=params).text
    soup = BeautifulSoup(response, 'html.parser')
    results = soup.find_all("a", {"id": re.compile(r'mntl-card-list-items_\d+-\d+')})
    
    return results

def scrape_recipe(recipe_url):
    scraper = scrape_me(recipe_url)
    return scraper

def extract_ingredient_names(ingredients):
    measurement_words = ['cup', 'cups', 'tablespoons', 'tablespoon', 'teaspoons', 'teaspoon', 'medium', 'sliced', 'chopped', 'dried', 'freshly', 'to', 'taste', 'grated','cover','or','more', 'broken','pinch','thinly','into','inch','package','ounce']
    exclusion_words = ['water']
    ingredient_names = []
    for ingredient in ingredients:
        ingredient_without_numbers = re.sub(r'\d+(?:\.\d+)?', '', ingredient).strip()
        words = ingredient_without_numbers.split()
        cleaned_words = [word for word in words if word.lower() not in measurement_words and word.lower() not in exclusion_words and not word.isdigit()]
        cleaned_words = [word.title() for word in cleaned_words]
        cleaned_ingredient = ' '.join(cleaned_words).rstrip(',')
        if cleaned_ingredient: 
            ingredient_names.append(cleaned_ingredient)
    return ingredient_names

def main():
    st.title("GroceRevo")
    dynamic_content = st.empty()
    recipe_name = st.text_input("Enter Recipe Name:")
    recipe_book_file = st.file_uploader("Upload PDF Recipe Book", type=["pdf"])
    pantry_images = st.file_uploader("Upload Pantry Images", type=["jpg", "jpeg", "png"])
    
    if st.button("Search"):
        if recipe_name:
            results = search_allrecipes(recipe_name)
            
            if results:
                st.subheader("Select a Recipe:")
                recipe_names = [result.text for result in results]
                selected_recipe = dynamic_content.selectbox("Choose a recipe:", recipe_names)

                selected_result = results[recipe_names.index(selected_recipe)]
                href_value = selected_result.get('href')

                scraper = scrape_recipe(href_value)
                ingredients = scraper.ingredients()
                ingredient_names = extract_ingredient_names(ingredients)

                st.subheader("Ingredients:")
                for ingredient in ingredient_names:
                    st.write(ingredient)
                    search_url = f"https://www.ralphs.com/search?query={ingredient.replace(' ', '+')}&savings=On%20Sale&fulfillment=all"
                    st.write(search_url)

                st.subheader("Recipe Details:")
                st.write(f"Link: {href_value}")
                st.write(f"Rating: {scraper.ratings()}")
                st.write(f"Description: {scraper.description()}")
                st.write(f"Directions: {scraper.instructions()}")
                st.subheader(f"Here is the nutritional information for {recipe_name}")
                st.markdown(f"Nutrition Facts (per serving): {scraper.nutrients()}")
                recipe_image = scraper.image()
                if recipe_image:
                    st.subheader("Recipe Image:")
                    st.image(recipe_image, caption="Recipe Image", use_column_width=True)

                # Use vertexai for multiturn_generate_content
                config = {
                    "max_output_tokens": 2048,
                    "temperature": 0.9,
                    "top_p": 1
                }
                model = GenerativeModel("gemini-pro")
                chat = model.start_chat()
                st.subheader("Multiturn Generate Content:")
                st.write(chat.send_message(f"recipe for {recipe_name}", generation_config=config))
                st.write(chat.send_message("can you give me the links of ingredients required", generation_config=config))

if __name__ == "__main__":
    main()
