# sql_exam.ipynb also can be checked
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey, text, MetaData
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine.url import URL

# 1:
filepath = '/homes/zhe/Desktop/programming/p5/exam/SQL_example.csv'
df = pd.read_csv(filepath)
print(df)

# 2 and 3: 
# Just pick what columns i need up to create a table
def drop_table_if_exists(engine):
    with engine.connect() as connection:
        connection.execute(text("DROP TABLE IF EXISTS student_data"))
        connection.commit()
    print("Existing 'student_data' table dropped (if it existed)")

# Function to create the exam table with selected columns
def create_exam_table(engine):
    drop_table_if_exists(engine)
    
    create_table_query = text("""
    CREATE TABLE student_data (
        school VARCHAR(5),
        sex CHAR(1),
        age INTEGER,
        address CHAR(1),
        Dalc INTEGER,
        Walc INTEGER
    )
    """)
    with engine.connect() as connection:
        connection.execute(create_table_query)
        connection.commit()
    print("New 'student_data' table created")

def insert_student_data(connection, student_data):
    insert_query = text("""
    INSERT INTO student_data (school, sex, age, address, Dalc, Walc)
    VALUES (:school, :sex, :age, :address, :Dalc, :Walc)
    """)
    try:
        connection.execute(insert_query, student_data)
        connection.commit()
        print(f"Inserted data: {student_data}")
    except Exception as error:
        print(f"Error inserting data: {str(error)}")
        raise

try:
    # Load data from CSV
    filepath = '/homes/zhe/Desktop/programming/p5/exam/SQL_example.csv'
    df = pd.read_csv(filepath)

    # Select only relevant columns
    selected_columns = ['school', 'sex', 'age', 'address', 'Dalc', 'Walc']
    df = df[selected_columns]

    # Convert numeric columns to appropriate types
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['Dalc'] = pd.to_numeric(df['Dalc'], errors='coerce')
    df['Walc'] = pd.to_numeric(df['Walc'], errors='coerce')

    # Replace NaN values with None for SQL compatibility
    df = df.where(pd.notna(df), None)

    # Connect to the database and insert data
    engine = create_engine('sqlite:////homes/zhe/Desktop/programming/p5/exam/file.db') 
    create_exam_table(engine)
    with engine.connect() as connection:
        for _, row in df.iterrows():
            try:
                insert_student_data(connection, row.to_dict())
            except Exception as e:
                #print(f"Error inserting row: {row.to_dict()}")
                print(f"Error message: {str(e)}")
except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    traceback.print_exc()
finally:
    if 'engine' in locals():
        engine.dispose()
        
    
# 4:
# Define the query
query = text("""
    SELECT 
        MAX(Dalc) AS Max_workday_intake,
        MAX(Walc) AS Max_weekend_intake
    FROM 
        student_data
""")

# Execute the query and fetch results
with engine.connect() as connection:
    result = connection.execute(query)
    row = result.fetchone()
    max_workday_intake = row[0]  # Accessing by index
    max_weekend_intake = row[1]  # Accessing by index

    # Determine when the maximum intake is higher
    if max_workday_intake > max_weekend_intake:
        print("The maximum alcohol consumption occurs during weekdays.")
    elif max_workday_intake < max_weekend_intake:
        print("The maximum alcohol consumption occurs during weekends.")
    else:
        print("The maximum alcohol consumption is the same for weekdays and weekends.\n")

    print(f"Max Workday Intake: {max_workday_intake}, Max Weekend Intake: {max_weekend_intake}\n")
    
# 5
def drop_columns(engine, table_name, columns_to_drop):
    with engine.connect() as connection:
        for column in columns_to_drop:
            query = text(f"ALTER TABLE {table_name} DROP COLUMN {column}")
            connection.execute(query)
        connection.commit()

def show_table_structure(engine, table_name):
    with engine.connect() as connection:
        query = text(f"PRAGMA table_info({table_name})")
        result = connection.execute(query)
        columns = result.fetchall()
        
        print(f"\nStructure of table '{table_name}':")
        for column in columns:
            print(f"Column: {column[1]}, Type: {column[2]}")
            
table_name = 'student_data'
columns_to_drop = ['school', 'address']

drop_columns(engine, table_name, columns_to_drop)
print('After droped school and address columns')
show_table_structure(engine, table_name)
connection.close()

