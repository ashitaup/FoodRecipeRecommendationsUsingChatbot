import PyPDF2
import re

pdf_path = "../Downloads/veg.pdf"

def extract_ingredients_and_directions_from_pdf(pdf_path, target):
    ingredients = []
    directions = []
    found = False
    target_count = 0
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        recipe_text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            recipe_text += page_text

            if target in page_text:
                target_count += 1
                if target_count == 2 or target_count == 3:
                    found = True
                    text = page_text
                elif found:
                    break

        recipes = recipe_text.split(target)
        for recipe in recipes:
            ingredients_match = re.search(r'(?i)Ingredients\b', recipe)
            if ingredients_match:
                found = True
                ingredients_text = recipe[ingredients_match.start():]
                directions_text = recipe[:ingredients_match.start()]
                ingredients.extend([line.strip() for line in ingredients_text.split('\n') if line.strip() != ''])
                directions.extend([line.strip() for line in directions_text.split('\n') if line.strip() != ''])
    return ingredients, directions

target_recipe_name = input("Enter the name of the recipe you want to extract ingredients and directions for: ")
ingredients, directions = extract_ingredients_and_directions_from_pdf(pdf_path, target_recipe_name)

# Print ingredients
print("Ingredients:")
for ingredient in ingredients:
    print(ingredient)

# Print directions/recipe
print("\nDirections/Recipe:")
for step in directions:
    print(step)
