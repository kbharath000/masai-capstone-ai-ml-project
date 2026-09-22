from os import name

import requests
from bs4 import BeautifulSoup


class ConvertHtmlToData:

    def _scrap_books_data(self, json_response):

        soup = BeautifulSoup(json_response.content, "html.parser")

        # FIX: The class name on books.toscrape.com is just "page-header"
        header_div = soup.find("div", class_="page-header")
        category = header_div.find("h1").text.strip() if header_div else "Unknown"

        products = soup.find_all("article", class_="product_pod")

        books = []
        for product in products:
            title = product.find("h3").find("a").get("title")

            # .split('£') splits '£51.77' into ['', '51.77'].
            # We grab index 1 to get just the clean numeric string.
            price_raw = product.find("p", class_="price_color").text.strip()
            price = price_raw.split("£")[1] if "£" in price_raw else price_raw

            # Explicitly match the star-rating class to avoid getting unrelated classes
            star_rating = product.find("p", class_="star-rating").get("class")
            rating = star_rating[1] if len(star_rating) > 1 else "Unknown"

            availability = product.find("p", class_="instock availability").text.strip()
            # print(f"Title: {title} | Book Price: £{price} | Rating: {rating} | Availability: {availability} | Category: {category}")
            books.append(
                {
                    "title": title,
                    "price": price,
                    "rating": rating,
                    "availability": availability,
                    "category": category,
                }
            )

        return books

    def get_scraped_books_data(self, base_url: str, urls: list):
        """Collect the data for the shared categories of urls to the base url"""
        books_data = []
        for url in urls:
            if not url.startswith("/"):
                url = "/" + url

            formaed_base_url = f"{base_url}{url}"

            # Get the html response from the site
            response = requests.get(formaed_base_url)

            # Format the data and collect the infor from each category
            books = self._scrap_books_data(response)

            if not books:
                return f"Empty Books list from the page {url}"
            books_data.append(books)

        return books_data

    def _convert_gbp_inr(self, transform_curr):
        # Converts GBP to INR
        return transform_curr * 105.50

    def _convert_rating_to_numeric(self, rating):
        # Maps String to Numerical
        return {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}[rating]

    def clean_scraped_data(self, scraped_books_data):
        """Clean scraped data from html to formatted data to dictionary
        with book value conversion value of price GBP to INR,
        In Stock changed to boolean
        """
        if not scraped_books_data:
            return []  # Return a consistent data type (list) instead of a string

        cleaned_data = []

        for books_cat_data in scraped_books_data:

            # Ensure the inner category data is iterable
            if not isinstance(books_cat_data, list):
                continue

            for book_cat_data in books_cat_data:
                if not isinstance(book_cat_data, dict):
                    continue

                # 1. Handle price conversion safely and round AFTER currency conversion
                raw_price = book_cat_data.get("price", 0)

                try:
                    gbp_price = float(raw_price)
                except (ValueError, TypeError):
                    gbp_price = 0.0
                inr_price = self._convert_gbp_inr(gbp_price)
                price = round(inr_price, 2)

                # 2. Extract rating safely
                raw_rating = book_cat_data.get("rating")
                rating = self._convert_rating_to_numeric(raw_rating)

                # 3. Check stock status safely
                raw_availability = book_cat_data.get("availability", "")
                availability = 1 if "In stock" in str(raw_availability) else 0

                # Mutate and append raw to single dictionary format

                book_cat_data["price"] = raw_price
                book_cat_data["price_gbp"] = gbp_price
                book_cat_data["price_inr"] = inr_price
                book_cat_data["rating"] = rating
                book_cat_data["availability"] = availability
                cleaned_data.append(book_cat_data)
        return cleaned_data
