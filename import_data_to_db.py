# import os
# import pandas as pd
# import spacy
# import sqlite3
# import streamlit as st
# from pymongo import MongoClient
# import mysql.connector
# import psycopg2
# import matplotlib.pyplot as plt
# # Load SpaCy model
# nlp = spacy.load("en_core_web_sm")
# # Function to fetch column names from the database schema
# def get_column_names(connection, table_name, db_type):
#     cursor = connection.cursor()
#     if db_type == 'sqlite':
#         cursor.execute(f"PRAGMA table_info({table_name})")
#         columns = [row[1] for row in cursor.fetchall()]
#     elif db_type == 'mysql':
#         cursor.execute(f"SHOW COLUMNS FROM {table_name}")
#         columns = [row[0] for row in cursor.fetchall()]
#     elif db_type == 'postgresql':
#         cursor.execute(f"""
#             SELECT column_name
#             FROM information_schema.columns
#             WHERE table_name='{table_name}'
#         """)
#         columns = [row[0] for row in cursor.fetchall()]
#     return columns
# # Enhanced function to extract entities
# def extract_entities(prompt, column_names):
#     doc = nlp(prompt)
#     columns = []
#     conditions = []
#     # Extract columns from the prompt
#     for token in doc:
#         if token.lemma_.lower() in column_names:
#             columns.append(token.lemma_.lower())
#     # If no columns are explicitly mentioned, select all columns
#     if not columns:
#         columns = [col for col in column_names if col in prompt.lower()]
#     # Extract movie title as a condition
#     movie_title = None
#     for ent in doc.ents:
#         if ent.label_ == "WORK_OF_ART":
#             movie_title = ent.text
#             conditions.append(('original_title', movie_title))
#     return {"columns": columns, "conditions": conditions}
# # Function to generate SQL query
# def generate_sql_query(prompt, column_names):
#     entities = extract_entities(prompt, column_names)
#     select_clause = f'SELECT {", ".join(entities["columns"])}' if entities["columns"] else 'SELECT *'
#     from_clause = 'FROM movies2'
#     where_clause = ''
#     if entities["conditions"]:
#         conditions = [f"{col} = '{val}'" for col, val in entities["conditions"]]
#         where_clause = f'WHERE {" AND ".join(conditions)}'
#     query = f'{select_clause} {from_clause} {where_clause};'
#     return query
# # Function to execute query and return results as DataFrame
# def execute_query(connection, query, db_type):
#     if db_type in ['sqlite', 'mysql', 'postgresql']:
#         df = pd.read_sql_query(query, connection)
#         return df
#     return None
# # Function to create a bar plot
# def create_bar_plot(df, title):
#     if not df.empty:
#         # Dynamically choose x and y columns
#         x_col = df.columns[0]  # Choosing the first column as x-axis
#         y_col = df.columns[1]  # Choosing the second column as y-axis
#         plt.figure(figsize=(10, 6))
#         plt.bar(df[x_col], df[y_col], color='skyblue')
#         plt.xlabel(x_col)
#         plt.ylabel(y_col)
#         plt.title(title)
#         plt.xticks(rotation=45)
#         st.pyplot(plt)
#     else:
#         st.write("No data available for plotting.")
# # Your original functions (create_table, insert_data, etc.) remain unchanged
# def create_table(connection, table_name, df, db_type):
#     cursor = connection.cursor()
#     columns = df.columns
#     column_types = []
#     for column in columns:
#         if pd.api.types.is_integer_dtype(df[column]):
#             column_type = "INT"
#         elif pd.api.types.is_float_dtype(df[column]):
#             column_type = "FLOAT"
#         else:
#             max_length = df[column].map(lambda x: len(str(x)) if pd.notnull(x) else 0).max()
#             column_type = f"VARCHAR({max_length if max_length > 0 else 255})"
#         column_types.append(f"{column} {column_type}")
#     columns_definition = ", ".join(column_types)
#     create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_definition});"
#     cursor.execute(create_table_query)
#     connection.commit()
 
 
 
# def insert_data(connection, table_name, df, db_type):
#     cursor = connection.cursor()
#     columns = ", ".join([f"{col}" for col in df.columns])
#     values_placeholders = ", ".join(["%s"] * len(df.columns))
#     if db_type == "postgresql":
#         values_placeholders = ", ".join(["%s"] * len(df.columns))
#     elif db_type == "sqlite":
#         values_placeholders = ", ".join(["?"] * len(df.columns))
#     for _, row in df.iterrows():
#         if db_type == "mongodb":
#             collection = connection[table_name]
#             row_dict = row.to_dict()
#             collection.insert_one(row_dict)
#         else:
#             sql = f"INSERT INTO {table_name} ({columns}) VALUES ({values_placeholders})"
#             values = [None if pd.isna(value) else value for value in row]
#             cursor.execute(sql, values)
#     if db_type != "mongodb":
#         connection.commit()
# # Streamlit user interface
# st.title('SQL - NLP QUERY SYSTEM')
# selectionbox = st.selectbox(
#     'Select the DB type',
#     ('Select the DB type', 'MySQL', 'Postgresql', 'SQLite', 'MongoDB'))
# # Initialize session state
# if 'connect_clicked' not in st.session_state:
#     st.session_state.connect_clicked = False
# if 'user_name' not in st.session_state:
#     st.session_state.user_name = ''
# if 'host_name' not in st.session_state:
#     st.session_state.host_name = ''
# if 'user_password' not in st.session_state:
#     st.session_state.user_password = ''
# if 'database_name' not in st.session_state:
#     st.session_state.database_name = ''
# # Function to handle connect button click
# def connect():
#     st.session_state.connect_clicked = True
# if not st.session_state.connect_clicked:
#     if selectionbox != 'Select the DB type':
#         st.write('You selected:', selectionbox)
#         st.session_state.user_name = st.text_input('Username', value=st.session_state.user_name)
#         st.session_state.host_name = st.text_input('Hostname', value=st.session_state.host_name)
#         st.session_state.user_password = st.text_input('Password', type='password', value=st.session_state.user_password)
#         st.session_state.database_name = st.text_input('Database Name', value=st.session_state.database_name)
#         uploaded_file = st.file_uploader("Choose a file", type=['csv','json','xlsx'])
#         st.button('Connect', on_click=connect)
#     else:
#         st.write('Please select an option.')
# # Show a message after the connect button is clicked
# if st.session_state.connect_clicked:
#     st.write('Connected. Input fields are now hidden.')
#     query_prompt = st.text_input('Enter your query requirements')
#     if st.button('Generate Query'):
#         if query_prompt:
#             connection = None
#             schema = None
#             try:
#                 if selectionbox == 'MySQL':
#                     connection = mysql.connector.connect(
#                         host=st.session_state.host_name,
#                         user=st.session_state.user_name,
#                         password=st.session_state.user_password,
#                         database=st.session_state.database_name
#                     )
#                 elif selectionbox == 'Postgresql':
#                     connection = psycopg2.connect(
#                         host=st.session_state.host_name,
#                         user=st.session_state.user_name,
#                         password=st.session_state.user_password,
#                         database=st.session_state.database_name
#                     )
#                 elif selectionbox == 'SQLite':
#                     connection = sqlite3.connect(st.session_state.database_name)
#                 elif selectionbox == 'MongoDB':
#                     connection = MongoClient(f"mongodb://{st.session_state.user_name}:{st.session_state.user_password}@{st.session_state.host_name}/{st.session_state.database_name}")
#                     db = connection[st.session_state.database_name]
#                 if connection:
#                     column_names = get_column_names(connection, 'movies2', selectionbox.lower())
#                     st.write(f"Generated column_names: {column_names}")
#                     generated_query = generate_sql_query(query_prompt, column_names)
#                     st.write(f"Generated SQL Query: {generated_query}")
#                     df = execute_query(connection, generated_query, selectionbox.lower())
#                     if df is not None:
#                         st.write("Generated df:", df)
#                         st.write(df)
#                         # Create bar plot
#                         create_bar_plot(df, 'Query Result')
#                     else:
#                         st.write("No data returned from the query.")
#             except (Exception) as e:
#                 st.error(f"The error '{e}' occurred while connecting to the database")
#             finally:
#                 if selectionbox == 'MySQL' and connection and connection.is_connected():
#                     connection.close()
#                     st.success("MySQL connection is closed")
#                 elif selectionbox == 'Postgresql' and connection:
#                     connection.close()
#                     st.success("PostgreSQL connection is closed")
#                 elif selectionbox == 'SQLite' and connection:
#                     connection.close()
#                     st.success("SQLite connection is closed")
#                 elif selectionbox == 'MongoDB' and connection:
#                     connection.close()
#                     st.success("MongoDB connection is closed")
# if st.button('Upload and Insert Data'):
#     if uploaded_file is not None and st.session_state.user_name and st.session_state.host_name and st.session_state.user_password and st.session_state.database_name:
#         file_extension = os.path.splitext(uploaded_file.name)[1]
#         if '.csv' in file_extension:
#             df = pd.read_csv(uploaded_file)
#         elif '.xlsx' in file_extension:
#             df = pd.read_excel(uploaded_file)
#         elif '.json' in file_extension:
#             df = pd.read_json(uploaded_file)
#         df = df.where(pd.notnull(df), None)
#         table_name = uploaded_file.name.split('.')[0]
#         st.write(df)
#         connection = None
#         try:
#             if selectionbox == 'MySQL':
#                 connection = mysql.connector.connect(
#                     host=st.session_state.host_name,
#                     user=st.session_state.user_name,
#                     password=st.session_state.user_password,
#                     database=st.session_state.database_name
#                 )
#             elif selectionbox == 'Postgresql':
#                 connection = psycopg2.connect(
#                     host=st.session_state.host_name,
#                     user=st.session_state.user_name,
#                     password=st.session_state.user_password,
#                     database=st.session_state.database_name
#                 )
#             elif selectionbox == 'SQLite':
#                 connection = sqlite3.connect(st.session_state.database_name)
#             elif selectionbox == 'MongoDB':
#                 connection = MongoClient(f"mongodb://{st.session_state.user_name}:{st.session_state.user_password}@{st.session_state.host_name}/{st.session_state.database_name}")
#                 db = connection[st.session_state.database_name]
#             if selectionbox != 'MongoDB' and connection:
#                 st.success("Successfully connected to the database")
#                 # Create the table based on the DataFrame
#                 create_table(connection, table_name, df, selectionbox.lower())
#                 # Insert data into the database
#                 insert_data(connection, table_name, df, selectionbox.lower())
#         except (Exception) as e:
#             st.error(f"The error '{e}' occurred while connecting to the database")
#         finally:
#             if selectionbox == 'MySQL' and connection and connection.is_connected():
#                 connection.close()
#                 st.success("MySQL connection is closed")
#             elif selectionbox == 'Postgresql' and connection:
#                 connection.close()
#                 st.success("PostgreSQL connection is closed")
#             elif selectionbox == 'SQLite' and connection:
#                 connection.close()
#                 st.success("SQLite connection is closed")
#             elif selectionbox == 'MongoDB' and connection:
#                 connection.close()
#                 st.success("MongoDB connection is closed")
#     else:
#         st.error("Please fill in all fields and upload a file")
 


import os
import pandas as pd
import spacy
import sqlite3
import streamlit as st
from pymongo import MongoClient
import mysql.connector
import psycopg2
import matplotlib.pyplot as plt
import subprocess

# Run the setup.sh script to download the spacy model
subprocess.run(['./setup.sh'], check=True)


# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Function to fetch column names from the database schema
def get_column_names(connection, table_name, db_type):
    cursor = connection.cursor()
    if db_type == 'sqlite':
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
    elif db_type == 'mysql':
        cursor.execute(f"SHOW COLUMNS FROM {table_name}")
        columns = [row[0] for row in cursor.fetchall()]
    elif db_type == 'postgresql':
        cursor.execute(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name='{table_name}'
        """)
        columns = [row[0] for row in cursor.fetchall()]
    return columns

# Enhanced function to extract entities
def extract_entities(prompt, column_names):
    doc = nlp(prompt)
    columns = []
    conditions = []
    
    # Extract columns from the prompt
    for token in doc:
        if token.lemma_.lower() in column_names:
            columns.append(token.lemma_.lower())
    
    # If no columns are explicitly mentioned, select all columns
    if not columns:
        columns = [col for col in column_names if col in prompt.lower()]
    
    # Extract movie title as a condition
    movie_title = None
    for ent in doc.ents:
        if (ent.label_ == "WORK_OF_ART" or ent.label_ == "PERSON") and ent.text.lower() in prompt.lower():
            movie_title = ent.text
            conditions.append(('original_title', movie_title))
    
    return {"columns": columns, "conditions": conditions}

# Function to generate SQL query
def generate_sql_query(prompt, column_names):
    entities = extract_entities(prompt, column_names)
    select_clause = f'SELECT {", ".join(entities["columns"])}' if entities["columns"] else 'SELECT *'
    from_clause = 'FROM movies2'
    where_clause = ''
    
    if entities["conditions"]:
        conditions = [f"{col} = '{val}'" for col, val in entities["conditions"]]
        where_clause = f'WHERE {" AND ".join(conditions)}'
    
    query = f'{select_clause} {from_clause} {where_clause};'
    print("query", query)
    return query

# Function to execute query and return results as DataFrame
def execute_query(connection, query, db_type):
    if db_type in ['sqlite', 'mysql', 'postgresql']:
        df = pd.read_sql_query(query, connection)
        return df
    return None

# Function to create a bar plot
def create_bar_plot(df, title):
    if not df.empty:
        # Dynamically choose x and y columns
        x_col = df.columns[0]  # Choosing the first column as x-axis
        y_col = df.columns[1]  # Choosing the second column as y-axis
        
        plt.figure(figsize=(10, 6))
        plt.bar(df[x_col], df[y_col], color='skyblue')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(title)
        plt.xticks(rotation=45)
        st.pyplot(plt)
    else:
        st.write("No data available for plotting.")

# Function to create table in the database
def create_table(connection, table_name, df, db_type):
    cursor = connection.cursor()
    columns = df.columns
    column_types = []
    for column in columns:
        if pd.api.types.is_integer_dtype(df[column]):
            column_type = "INT"
        elif pd.api.types.is_float_dtype(df[column]):
            column_type = "FLOAT"
        else:
            max_length = df[column].map(lambda x: len(str(x)) if pd.notnull(x) else 0).max()
            column_type = f"VARCHAR({max_length if max_length > 0 else 255})"
        column_types.append(f"{column} {column_type}")
    columns_definition = ", ".join(column_types)
    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_definition});"
    cursor.execute(create_table_query)
    connection.commit()

# Function to insert data into the database
def insert_data(connection, table_name, df, db_type):
    cursor = connection.cursor()
    columns = ", ".join([f"{col}" for col in df.columns])
    values_placeholders = ", ".join(["%s"] * len(df.columns))
    if db_type == "postgresql":
        values_placeholders = ", ".join(["%s"] * len(df.columns))
    elif db_type == "sqlite":
        values_placeholders = ", ".join(["?"] * len(df.columns))
    for _, row in df.iterrows():
        if db_type == "mongodb":
            collection = connection[table_name]
            row_dict = row.to_dict()
            collection.insert_one(row_dict)
        else:
            sql = f"INSERT INTO {table_name} ({columns}) VALUES ({values_placeholders})"
            values = [None if pd.isna(value) else value for value in row]
            cursor.execute(sql, values)
    if db_type != "mongodb":
        connection.commit()

# Streamlit user interface
st.title('SQL - NLP QUERY SYSTEM')

selectionbox = st.selectbox(
    'Select the DB type',
    ('Select the DB type', 'MySQL', 'Postgresql', 'SQLite', 'MongoDB')
)

# Initialize session state
if 'connect_clicked' not in st.session_state:
    st.session_state.connect_clicked = False

if 'user_name' not in st.session_state:
    st.session_state.user_name = ''

if 'host_name' not in st.session_state:
    st.session_state.host_name = ''

if 'user_password' not in st.session_state:
    st.session_state.user_password = ''

if 'database_name' not in st.session_state:
    st.session_state.database_name = ''

# Function to handle connect button click
def connect():
    st.session_state.connect_clicked = True

# Function to handle file upload and data insertion
def upload_and_insert_data():
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'json', 'xlsx'])
    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        if '.csv' in file_extension:
            df = pd.read_csv(uploaded_file)
        elif '.xlsx' in file_extension:
            df = pd.read_excel(uploaded_file)
        elif '.json' in file_extension:
            df = pd.read_json(uploaded_file)
        df = df.where(pd.notnull(df), None)
        table_name = uploaded_file.name.split('.')[0]
        st.write(df)
        try:
            connection = None
            if selectionbox == 'MySQL':
                connection = mysql.connector.connect(
                    host=st.session_state.host_name,
                    user=st.session_state.user_name,
                    password=st.session_state.user_password,
                    database=st.session_state.database_name
                )
            elif selectionbox == 'Postgresql':
                connection = psycopg2.connect(
                    host=st.session_state.host_name,
                    user=st.session_state.user_name,
                    password=st.session_state.user_password,
                    database=st.session_state.database_name
                )
            elif selectionbox == 'SQLite':
                connection = sqlite3.connect(st.session_state.database_name)
            elif selectionbox == 'MongoDB':
                connection = MongoClient(f"mongodb://{st.session_state.user_name}:{st.session_state.user_password}@{st.session_state.host_name}/{st.session_state.database_name}")
                db = connection[st.session_state.database_name]
            
            if connection:
                st.success("Successfully connected to the database")
                # Create the table based on the DataFrame
                create_table(connection, table_name, df, selectionbox.lower())
                # Insert data into the database
                insert_data(connection, table_name, df, selectionbox.lower())
        
        except Exception as e:
            st.error(f"The error '{e}' occurred while connecting to the database")
        
        finally:
            if selectionbox == 'MySQL' and connection and connection.is_connected():
                connection.close()
                st.success("MySQL connection is closed")
            elif selectionbox == 'Postgresql' and connection:
                connection.close()
                st.success("PostgreSQL connection is closed")
            elif selectionbox == 'SQLite' and connection:
                connection.close()
                st.success("SQLite connection is closed")
            elif selectionbox == 'MongoDB' and connection:
                connection.close()
                st.success("MongoDB connection is closed")

# UI logic
if not st.session_state.connect_clicked:
    if selectionbox != 'Select the DB type':
        st.write('You selected:', selectionbox)
        st.session_state.user_name = st.text_input('Username', value=st.session_state.user_name)
        st.session_state.host_name = st.text_input('Hostname', value=st.session_state.host_name)
        st.session_state.user_password = st.text_input('Password', type='password', value=st.session_state.user_password)
        st.session_state.database_name = st.text_input('Database Name', value=st.session_state.database_name)
        st.button('Connect', on_click=connect)
    else:
        st.write('Please select an option.')

# Show a message after the connect button is clicked
if st.session_state.connect_clicked:
    st.write('Connected. Input fields are now hidden.')
    query_prompt = st.text_input('Enter your query requirements')
    if st.button('Generate Query'):
        if query_prompt:
            connection = None
            try:
                if selectionbox == 'MySQL':
                    connection = mysql.connector.connect(
                        host=st.session_state.host_name,
                        user=st.session_state.user_name,
                        password=st.session_state.user_password,
                        database=st.session_state.database_name
                    )
                elif selectionbox == 'Postgresql':
                    connection = psycopg2.connect(
                        host=st.session_state.host_name,
                        user=st.session_state.user_name,
                        password=st.session_state.user_password,
                        database=st.session_state.database_name
                    )
                elif selectionbox == 'SQLite':
                    connection = sqlite3.connect(st.session_state.database_name)
                elif selectionbox == 'MongoDB':
                    connection = MongoClient(f"mongodb://{st.session_state.user_name}:{st.session_state.user_password}@{st.session_state.host_name}/{st.session_state.database_name}")
                    db = connection[st.session_state.database_name]
                
                if connection:
                    column_names = get_column_names(connection, 'movies2', selectionbox.lower())
                    st.write(f"Generated column_names: {column_names}")
                    generated_query = generate_sql_query(query_prompt, column_names)
                    st.write(f"Generated SQL Query: {generated_query}")
                    df = execute_query(connection, generated_query, selectionbox.lower())
                    
                    if df is not None:
                        st.write("Generated df:")
                        st.write(df)
                        # Create bar plot
                        create_bar_plot(df, 'Query Result')
                    else:
                        st.write("No data returned from the query.")
            
            except Exception as e:
                st.error(f"The error '{e}' occurred while connecting to the database")
            
            finally:
                if selectionbox == 'MySQL' and connection and connection.is_connected():
                    connection.close()
                    st.success("MySQL connection is closed")
                elif selectionbox == 'Postgresql' and connection:
                    connection.close()
                    st.success("PostgreSQL connection is closed")
                elif selectionbox == 'SQLite' and connection:
                    connection.close()
                    st.success("SQLite connection is closed")
                elif selectionbox == 'MongoDB' and connection:
                    connection.close()
                    st.success("MongoDB connection is closed")

# Button to upload and insert data
if st.button('Upload and Insert Data'):
    upload_and_insert_data()



