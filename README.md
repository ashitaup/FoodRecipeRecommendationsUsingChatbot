# FoodRecipeRecommendationsUsingChatbot

The dataset used includes information about Instacart’s online grocery baskets, which can be used to assist users in providing recommendations or suggest recipes and likely product they can buy according to the items they have in their basket.
I've used Kaggle API to download the dataset. 
I've defined the `explore_csv_excel` function that iterates through the dataset filenames.
   - For each CSV file, it loads the data into a Pandas DataFrame and performs the following tasks:
     - Displays the first few rows of the dataset.
     - Counts the number of rows and columns.
     - Identifies missing data by calculating the sum of null values in each column.
     - Generates data visualization plots using Seaborn and Matplotlib for specific CSV files ('aisles.csv', 'departments.csv', and 'orders.csv').
   - The code also performs data merging and cleansing.
   - Finally, it displays the first few rows of the merged and cleansed DataFrame and repeats the earlier exploratory steps for this combined dataset.
   - The `explore_ascii_html` function uses Selenium and Chrome WebDriver to access a specific website (`https://instacartcommunityaffairs.com/`) and collect HTML data.
   - The `explore_pdf_word` function extracts text from PDF or Word documents using the `pdfplumber` library.

  Further, I'll be using Pytesseract to extract images from PDFs or similar documents to be able to recognize different items and their nutritional value texts out of the images.
  The next step will be to use recommendation algorithm starting from content-based filtering to recommend recipes that contain ingredients similar to those in the customer's basket.
  I've not yet decided on the framework that I'm going to use for the chatbot.
  
