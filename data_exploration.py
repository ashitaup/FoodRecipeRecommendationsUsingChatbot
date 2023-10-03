import pdfplumber
import pandas as pd
import kaggle
from bs4 import BeautifulSoup
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

dataset_name = 'yasserh/instacart-online-grocery-basket-analysis-dataset'
destination_folder = '../ashita2789754543/data/kaggledata'

kaggle.api.dataset_download_files(dataset_name, path=destination_folder, unzip=True)
time.sleep(5)
os.makedirs(destination_folder, exist_ok=True)
dataset_filenames = ['aisles.csv', 'departments.csv', 'order_products__prior.csv', 'order_products__train.csv', 'orders.csv', 'products.csv']
ascii_text_file = '../LAB_2/Instacart_Marketplace.html'
pdf_or_word_document = '../LAB_2/InstacartFAQs.pdf'
def explore_csv_excel(dataset_filenames):
    try:
        for filename in dataset_filenames:
            file_path = os.path.join(destination_folder, filename)
            df = pd.read_csv(file_path)
            if filename =='products.csv':
                df1=pd.read_csv(file_path)
            if filename =='aisles.csv':
                df2=pd.read_csv(file_path)
            if filename =='departments.csv':
                df3=pd.read_csv(file_path)
            if filename =='order_products__train.csv':
                df4=pd.read_csv(file_path)
            if filename =='orders.csv':
                df5=pd.read_csv(file_path)
            print(f"Dataset: {filename}")
            print(df.head())
            print(f"Number of rows: {len(df)}")
            print(f"Number of columns: {len(df.columns)}")
            missing_data = df.isnull().sum()
            print("Missing data:")
            print(missing_data)
            if filename =='aisles.csv':
                sns.countplot(df.aisle, order=df.aisle.value_counts().index[:20])
                plt.title('Online Shopping Aisle-Frequency')
                plt.xticks(rotation=90)
                plt.show()
                time.sleep(5)    
            if filename=='orders.csv':
                sns.countplot(df.order_hour_of_day)
                plt.title('Online Shopping Horly-Frequency')
                plt.xticks(rotation=90)
                plt.show()
                time.sleep(5)    
        df12 = pd.merge(df2, df1, how='inner', on='aisle_id')
        df123 = pd.merge(df12, df3, how='inner', on='department_id')
        df1234 = pd.merge(df4, df123, how='inner', on='product_id')
        df12345 = pd.merge(df5, df1234, how='inner', on='order_id')

        df6 = df12345.drop(['aisle_id','department_id','eval_set','order_number'],axis=1)
        print("Dataset:")
        print(df6.head())
        print(f"Number of rows: {len(df6)}")
        print(f"Number of columns: {len(df6.columns)}")
        missing_data = df6.isnull().sum()
        print("Missing data:")
        print(missing_data)
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except Exception as e:
        print(f"An error occurred while exploring {filename}: {str(e)}")

def explore_ascii_html(file_path):
    if file_path.endswith(('.txt', '.html')):
        try:
            driver=webdriver.Chrome(service=Service(ChromeDriverManager().install()))
            url="https://instacartcommunityaffairs.com/"
            wait = WebDriverWait(driver, 5)

            driver.get(url)

            get_url = driver.current_url
            wait.until(EC.url_to_be(url))

            response = driver.page_source

            data_folder="../ashita2789754543/data"
            raw_data_folder=os.path.join(data_folder,"html_data")
            os.makedirs(raw_data_folder,exist_ok=True)

            soup=BeautifulSoup(response,"html.parser")
            web_data_path=os.path.join(raw_data_folder,"web_data.html")
            with open(web_data_path,"wb") as file:
                file.write(soup.prettify().encode("utf-8"))
                print("Data Collected and saved.")

            with open(web_data_path,"r",encoding="utf-8") as file:
                first_ten="\n".join([next(file) for _ in range(10)])
                print(first_ten)
        except Exception as e:
            print("Error loading ASCII Text/HTML data:", str(e))

def explore_pdf_word(file_path):
    if file_path.endswith(('.pdf', '.doc', '.docx')):
        try:
            if file_path.endswith(('.doc', '.docx')):
                pdf_file_path = 'temp_pdf.pdf'
                os.system(f'libreoffice --headless --convert-to pdf --outdir ./ {file_path}')
                file_path = pdf_file_path
            with pdfplumber.open(file_path) as pdf:
                full_text = ""
                word_count = 0
                for page in pdf.pages:
                    page_text = page.extract_text()
                    full_text += page_text + '\n'
                    page_word_count = len(page_text.split())
                    word_count += page_word_count
                print("PDF/Word Data Sample:")
                print(full_text[:500])
                print("Total Word Count:", word_count)
        except Exception as e:
            print("Error loading PDF/Word data:", str(e))
explore_csv_excel(dataset_filenames)
explore_ascii_html(ascii_text_file)
explore_pdf_word(pdf_or_word_document)
