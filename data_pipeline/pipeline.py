from pipeline_utils import ConvertHtmlToData
from database import DataBase

import json
import os

import pandas as pd
import sqlite3


class DataPipeline:

    def scraped_data(self):
        # To run it:
        scraper = ConvertHtmlToData()
        urls = [
            "/travel_2/index.html",
            "/mystery_3/index.html",
            "historical-fiction_4/index.html",
            "/sequential-art_5/index.html",
        ]
        formaed_base_url = "https://books.toscrape.com/catalogue/category/books"
        scraped_data = scraper.get_scraped_books_data(formaed_base_url, urls)
        cleaned_data = scraper.clean_scraped_data(scraped_books_data=scraped_data)
        return cleaned_data

    def create_database(self):
        db = DataBase()
        db.create_conn_tables_database()
        return db
    
    def run_queries(self, queries: dict, cursor):
        query_results = {}
        query_log = []
        for query_name, query in queries.items():
            cursor.execute(query)
            rows = cursor.fetchall()
            query_results[query_name] = rows
            query_log.append({
                "name": query_name,
                "query": query.strip(),
                "output": rows,
            })
            print(f"\n--- {query_name} ---")
            print(query.strip())
            for row in rows:
                print(row)
        return query_results, query_log
    
    def dump_to_json(self, resources_dir_path: str, output_log: dict):
        # Save each query string and its output alongside the database
        queries_log_path = os.path.join(resources_dir_path, 'query_results.json')
        with open(queries_log_path, 'w') as f:
            json.dump(output_log, f, indent=2)
        print(f"\nSaved query strings and outputs to: {queries_log_path}")

if __name__ == "__main__":
    
    dp = DataPipeline()
    
    books_data = dp.scraped_data()
    
    db = dp.create_database()
    
    df = pd.DataFrame(books_data)
    
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()

    # Insert only categories that aren't already there; keeps the existing table/schema intact
    categories = df['category'].unique().tolist()
    cursor.executemany(
        "INSERT OR IGNORE INTO categories (category_name) VALUES (?)",
        [(category,) for category in categories]
    )
    conn.commit()

    # Map each category name to its category_id so books can reference it
    category_map = dict(cursor.execute("SELECT category_name, category_id FROM categories").fetchall())
    df['category_id'] = df['category'].map(category_map)

    #Insert books data frame to books data base
    books_df = df[['category_id', 'title', 'price_gbp', 'price_inr', 'rating', 'availability']]
    books_df.to_sql(
        name='books',           # Name of the SQL table
        con=conn,               # The database connection object
        if_exists='replace',     # 'append' adds rows; 'replace' drops/recreates; 'fail' raises error
        index=False             # Do not write the DataFrame row index as a database column
    )

    # --- SQL query demonstrations: SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN, JOIN ---
    resources_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')
    queries_path = os.path.join(resources_dir, 'queries.json')
    with open(queries_path) as f:
        queries = json.load(f)

    query_results, query_log = dp.run_queries(queries=queries, cursor=cursor)
    dp.dump_to_json(resources_dir_path=resources_dir, output_log=query_log)

    # Read back at least two of the saved queries as DataFrames via pd.read_sql
    select_where_df = pd.read_sql(queries['select_where'], conn)
    order_by_limit_df = pd.read_sql(queries['order_by_limit'], conn)
    print("\n--- select_where (read back via pd.read_sql) ---")
    print(select_where_df)
    print("\n--- order_by_limit (read back via pd.read_sql) ---")
    print(order_by_limit_df)

    # Reproduce the JOIN query's result purely in pandas (no SQL), using the
    # in-memory books/categories DataFrames, and compare it to the SQL output
    
    categories_df = pd.DataFrame(
        list(category_map.items()), columns=['category_name', 'category_id']
    )
    
    merged_df = df.merge(categories_df, on='category_id', how='inner', validate='many_to_one')
    
    merged_df['rank'] = (
        merged_df
        .sort_values(['category_name', 'rating', 'price_inr'], ascending=[True, False, True])
        .groupby('category_name')
        .cumcount() + 1
    )
    
    pandas_join_df = (
        merged_df[merged_df['rank'] <= 10]
        .sort_values(['category_name', 'rank'])[['category_name', 'title', 'rating', 'price_inr']]
        .reset_index(drop=True)
        .astype({'rating': 'float64', 'price_inr': 'float64'})
    )

    sql_join_df = pd.DataFrame(
        query_results['join_top_rated_per_category'],
        columns=['category_name', 'title', 'rating', 'price_inr']
    ).astype({'rating': 'float64', 'price_inr': 'float64'})

    print("\n--- join_top_rated_per_category via pandas merge (no SQL) ---")
    print(pandas_join_df)

    results_match = pandas_join_df.equals(sql_join_df)
    print(f"\nSQL JOIN result matches pandas merge result: {results_match}")

    conn.close()
    print(books_df)


   
