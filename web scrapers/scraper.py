import requests
from bs4 import BeautifulSoup
import pandas as pd
import time


def scrape_names_from_page(url):
    """
    Scrape the given URL for names.
    It looks for <a> tags inside <td> elements as observed from the website's structure.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    names = []

    # Select all <a> tags that are inside <td> elements.
    for tag in soup.select('td > a'):
        name = tag.get_text().strip()
        if name:
            names.append(name)

    return names


def main():
    # The two gender paths as seen in the URL structure.
    genders = ['baby-boy-name', 'baby-girl-name']
    # Letters A to Z.
    letters = [chr(i) for i in range(65, 91)]  # A-Z

    all_names = set()

    # Loop over each gender and each letter.
    for gender in genders:
        for letter in letters:
            # Base URL for the current gender and letter.
            base_url = f"https://www.baby360.in/baby-names/{gender}-starting-with-{letter}/"
            page = 1
            while True:
                # For page 1 use the base URL; for later pages, add the pagination path.
                if page == 1:
                    url = base_url
                else:
                    url = f"{base_url}page/{page}/"
                print(f"Scraping: {url}")

                names = scrape_names_from_page(url)
                # If no names are found, exit the loop for this letter.
                if not names:
                    print(f"No names found on {url}. Moving to next category.")
                    break

                # Add the names to the set (ensuring uniqueness).
                for name in names:
                    all_names.add(name)

                page += 1
                # Politeness: sleep for 1 second between requests.
                time.sleep(1)

    print(f"Total unique names scraped: {len(all_names)}")

    # Save the names to a CSV file.
    df = pd.DataFrame(list(all_names), columns=['name'])
    df.to_csv("indian_first_names.csv", index=False)
    print("Dataset saved as 'indian_first_names.csv'.")


if __name__ == '__main__':
    main()
