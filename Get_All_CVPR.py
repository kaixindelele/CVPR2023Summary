import os
import requests
from bs4 import BeautifulSoup
import csv


# Function to download PDFs
def download_pdf(url, output_path):
    response = requests.get(url)
    with open(output_path, 'wb') as file:
        file.write(response.content)

# URL of the page
url = 'https://openaccess.thecvf.com/CVPR2023?day=all'

# Send a GET request
response = requests.get(url)

# Parse the content with BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find all dt elements, each of which corresponds to a paper
papers = soup.find_all('dt')

# Open a CSV file to write the data
with open('cvpr_papers.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['Title', 'PDF URL', 'Arxiv URL'])

    # Loop over the papers
    index = 0
    for paper in papers:
        # Get the title
        title = paper.find('a').text

        # Get the PDF URL
        temp = str(paper.find_next_sibling('dd').find_next_sibling('dd'))
        soup = BeautifulSoup(str(temp), 'html.parser')
        # Find all the <a> tags
        a_tags = soup.find_all('a')
        # Initialize PDF and Arxiv URL variables
        pdf_url = ''
        arxiv_url = ''
        for a in a_tags:
            if a.text == 'pdf' and a['href'].endswith('_paper.pdf'):
                pdf_url = 'https://openaccess.thecvf.com' + a['href']
            elif a.has_attr('href') and a['href'].startswith('http://arxiv.org/abs/'):
                arxiv_url = a['href']
        
        # Write the data to the CSV file       
        pdf_name = pdf_url.split("/")[-1]
        download_pdf(pdf_url, pdf_name)
        writer.writerow([title, pdf_url, arxiv_url])
        print("title:", title, index)
        index += 1

print('Scraping is done. Data is saved to cvpr_papers.csv')

